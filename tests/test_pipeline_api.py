"""End-to-end FastAPI integration tests for generation, curation, and LoRA distillation."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, state


@pytest.fixture
def client():
    """Create a FastAPI test client with reset state."""
    state.raw_samples = []
    state.curated_samples = []
    state.last_generation_info = None
    state.last_curation_report = None
    state.last_distillation_metrics = None
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """GET /health must return operational status, device, and torch info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["simulation_mode"] is True
    assert data["device"] in ("cpu", "cuda")
    assert "torch_version" in data
    assert data["version"] == "0.1.0"


def test_metrics_endpoint_initial(client: TestClient):
    """GET /metrics should return empty initial summary."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["last_generation"] is None
    assert data["last_curation"] is None
    assert data["last_distillation"] is None
    assert data["pipeline_summary"]["cached_raw_samples_count"] == 0


def test_pipeline_generate_endpoint(client: TestClient):
    """POST /pipeline/generate generates synthetic ChatML samples."""
    payload = {
        "num_samples": 12,
        "domains": ["reasoning", "code", "math"],
        "complexity_levels": [1, 2, 3],
        "noise_rate": 0.1,
        "seed": 42,
    }
    response = client.post("/pipeline/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_generated"] == 12
    assert len(data["samples_preview"]) <= 5
    assert len(data["samples_preview"]) > 0

    first_sample = data["samples_preview"][0]
    assert "messages" in first_sample
    assert len(first_sample["messages"]) >= 2
    assert first_sample["domain"] in ["reasoning", "code", "math"]


def test_pipeline_curate_endpoint(client: TestClient):
    """POST /pipeline/curate runs deduplication and quality rejection sampling."""
    # First generate
    gen_payload = {"num_samples": 15, "noise_rate": 0.2, "seed": 42}
    gen_resp = client.post("/pipeline/generate", json=gen_payload)
    assert gen_resp.status_code == 200

    # Then curate
    curate_payload = {
        "dedup_jaccard_threshold": 0.80,
        "contamination_threshold": 0.70,
        "quality_score_threshold": 0.65,
    }
    curate_resp = client.post("/pipeline/curate", json=curate_payload)
    assert curate_resp.status_code == 200
    data = curate_resp.json()
    assert data["status"] == "success"

    report = data["report"]
    assert report["total_input"] == 15
    assert report["final_yield_count"] > 0
    assert 0.0 <= report["yield_rate"] <= 1.0
    assert "duplicates_removed" in report
    assert "contaminated_removed" in report
    assert "quality_rejected" in report


def test_pipeline_train_lora_endpoint(client: TestClient):
    """POST /pipeline/train-lora executes CPU LoRA distillation."""
    # 1. Generate & curate
    client.post("/pipeline/generate", json={"num_samples": 10, "seed": 42})
    client.post("/pipeline/curate", json={"quality_score_threshold": 0.50})

    # 2. Train LoRA
    train_payload = {
        "lora_config": {
            "r": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "learning_rate": 3e-3,
            "batch_size": 4,
            "epochs": 2,
            "max_seq_len": 48,
            "device": "cpu",
        }
    }
    train_resp = client.post("/pipeline/train-lora", json=train_payload)
    assert train_resp.status_code == 200
    data = train_resp.json()
    assert data["status"] == "success"

    metrics = data["metrics"]
    assert metrics["trainable_percentage"] < 5.0
    assert metrics["total_parameters"] > 0
    assert metrics["trainable_parameters"] > 0
    assert metrics["frozen_parameters"] > 0
    assert len(metrics["loss_history"]) > 0
    assert metrics["structured_output_validity"] == 1.0


def test_metrics_endpoint_after_pipeline(client: TestClient):
    """GET /metrics should reflect executed generation, curation, and distillation."""
    client.post("/pipeline/generate", json={"num_samples": 8, "seed": 42})
    client.post("/pipeline/curate", json={"quality_score_threshold": 0.50})
    client.post(
        "/pipeline/train-lora",
        json={"lora_config": {"r": 4, "epochs": 1, "batch_size": 4, "max_seq_len": 32}},
    )

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["last_generation"] is not None
    assert data["last_curation"] is not None
    assert data["last_distillation"] is not None
    assert data["pipeline_summary"]["has_trained_adapter"] is True


def test_curate_without_data_returns_400(client: TestClient):
    """POST /pipeline/curate with empty state and no input returns HTTP 400."""
    response = client.post("/pipeline/curate", json={})
    assert response.status_code == 400
    assert "No samples provided" in response.json()["detail"]


def test_train_without_data_returns_400(client: TestClient):
    """POST /pipeline/train-lora with empty state and no curated samples returns HTTP 400."""
    response = client.post("/pipeline/train-lora", json={})
    assert response.status_code == 400
    assert "No curated samples available" in response.json()["detail"]


def test_invalid_request_schema_returns_422(client: TestClient):
    """POST /pipeline/generate with invalid parameter type returns HTTP 422."""
    response = client.post("/pipeline/generate", json={"num_samples": "invalid_number"})
    assert response.status_code == 422
