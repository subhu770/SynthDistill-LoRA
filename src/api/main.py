"""FastAPI application orchestrating synthetic dataset generation, curation, and LoRA distillation."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CurateRequest,
    CurateResponse,
    CurationReport,
    DistillationMetrics,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LoRAConfig,
    MetricsResponse,
    SyntheticSample,
    TrainLoRARequest,
    TrainLoRAResponse,
)
from src.data.deduplicator import Deduplicator
from src.data.quality_judge import QualityJudge
from src.data.synthetic_generator import SyntheticDataGenerator
from src.training.lora_trainer import LoRATrainer


class PipelineState:
    """In-memory state manager for pipeline runs and historical metrics."""

    def __init__(self) -> None:
        self.raw_samples: List[SyntheticSample] = []
        self.curated_samples: List[SyntheticSample] = []
        self.last_generation_info: Optional[Dict[str, Any]] = None
        self.last_curation_report: Optional[CurationReport] = None
        self.last_distillation_metrics: Optional[DistillationMetrics] = None


state = PipelineState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and teardown context manager."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    yield


app = FastAPI(
    title="SynthDistill-LoRA Pipeline Orchestrator",
    description="Automated synthetic dataset generation, quality filtering, and LoRA model distillation engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health() -> HealthResponse:
    """Return operational health, simulation mode status, and compute device."""
    sim_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy",
        simulation_mode=sim_mode,
        device=device_str,
        torch_version=torch.__version__,
        version="0.1.0",
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics() -> MetricsResponse:
    """Return latest metrics from generation, curation, and LoRA distillation."""
    summary = {
        "cached_raw_samples_count": len(state.raw_samples),
        "cached_curated_samples_count": len(state.curated_samples),
        "has_trained_adapter": state.last_distillation_metrics is not None,
    }
    if state.last_distillation_metrics:
        summary["trainable_percentage"] = state.last_distillation_metrics.trainable_percentage
        summary["distilled_accuracy"] = state.last_distillation_metrics.distilled_accuracy
    if state.last_curation_report:
        summary["curation_yield_rate"] = state.last_curation_report.yield_rate

    return MetricsResponse(
        status="success",
        last_generation=state.last_generation_info,
        last_curation=state.last_curation_report,
        last_distillation=state.last_distillation_metrics,
        pipeline_summary=summary,
    )


@app.post("/pipeline/generate", response_model=GenerateResponse, tags=["Pipeline"])
async def generate_dataset(req: GenerateRequest) -> GenerateResponse:
    """Trigger Evol-Instruct synthetic dataset generation."""
    t0 = time.time()
    try:
        generator = SyntheticDataGenerator(seed=req.seed)
        samples = generator.generate(
            num_samples=req.num_samples,
            domains=req.domains,
            complexity_levels=req.complexity_levels,
            noise_rate=req.noise_rate,
        )

        state.raw_samples = samples
        gen_duration = round(time.time() - t0, 4)

        out_path = req.output_path or "data/synthetic_raw.jsonl"
        SyntheticDataGenerator.save_to_jsonl(samples, out_path)

        state.last_generation_info = {
            "num_samples": len(samples),
            "duration_seconds": gen_duration,
            "output_path": out_path,
            "noise_rate": req.noise_rate,
        }

        # Return a preview of up to 5 samples
        preview = samples[: min(5, len(samples))]

        return GenerateResponse(
            status="success",
            total_generated=len(samples),
            samples_preview=preview,
            output_path=out_path,
            generation_time_sec=gen_duration,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthetic dataset generation failed: {str(e)}",
        )


@app.post("/pipeline/curate", response_model=CurateResponse, tags=["Pipeline"])
async def curate_dataset(req: CurateRequest) -> CurateResponse:
    """Execute deduplication, benchmark contamination gating, and quality rubric rejection sampling."""
    t0 = time.time()
    try:
        # Determine source samples
        samples_to_curate: List[SyntheticSample] = []
        if req.samples is not None and len(req.samples) > 0:
            samples_to_curate = req.samples
        elif req.input_path:
            samples_to_curate = SyntheticDataGenerator.load_from_jsonl(req.input_path)
        elif len(state.raw_samples) > 0:
            samples_to_curate = state.raw_samples
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No samples provided in request, input_path, or pipeline cache. Run /pipeline/generate first.",
            )

        deduplicator = Deduplicator(
            dedup_threshold=req.dedup_jaccard_threshold,
            contamination_threshold=req.contamination_threshold,
            contamination_ngram_n=req.contamination_ngram_n,
        )
        judge = QualityJudge(quality_threshold=req.quality_score_threshold)

        duration = time.time() - t0
        curated_samples, report = judge.curate_pipeline(
            samples=samples_to_curate,
            deduplicator=deduplicator,
            duration_seconds=duration,
        )

        state.curated_samples = curated_samples
        state.last_curation_report = report

        out_path = req.output_path or "data/curated.jsonl"
        SyntheticDataGenerator.save_to_jsonl(curated_samples, out_path)

        preview = curated_samples[: min(5, len(curated_samples))]

        return CurateResponse(
            status="success",
            report=report,
            curated_samples_preview=preview,
            output_path=out_path,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset curation failed: {str(e)}",
        )


@app.post("/pipeline/train-lora", response_model=TrainLoRAResponse, tags=["Pipeline"])
async def train_lora_distillation(req: TrainLoRARequest) -> TrainLoRAResponse:
    """Train LoRA adapter on curated dataset and save checkpoint weights."""
    try:
        train_samples: List[SyntheticSample] = []
        if req.samples is not None and len(req.samples) > 0:
            train_samples = req.samples
        elif req.curated_data_path:
            train_samples = SyntheticDataGenerator.load_from_jsonl(req.curated_data_path)
        elif len(state.curated_samples) > 0:
            train_samples = state.curated_samples
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No curated samples available for LoRA training. Run /pipeline/curate first.",
            )

        adapter_dir = req.save_adapter_path or "checkpoints/lora_adapter"
        trainer = LoRATrainer(config=req.lora_config)

        metrics = trainer.train(
            train_samples=train_samples,
            save_adapter_path=adapter_dir,
        )

        state.last_distillation_metrics = metrics

        return TrainLoRAResponse(
            status="success",
            metrics=metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LoRA distillation training failed: {str(e)}",
        )
