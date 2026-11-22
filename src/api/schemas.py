"""Pydantic v2 strict schemas for dataset samples, curation reports, and LoRA distillation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    """ChatML formatted single message turn."""

    model_config = ConfigDict(strict=True)

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message author role in ChatML standard"
    )
    content: str = Field(..., min_length=1, description="Message text payload")


class SyntheticSample(BaseModel):
    """Full schema for a synthesized instruction-response sample with curation metadata."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique sample identifier (e.g. UUID or hash)")
    messages: List[ChatMessage] = Field(
        ..., min_length=2, description="ChatML conversation turns"
    )
    domain: str = Field(..., description="Domain category (reasoning, code, math, safety, extraction)")
    complexity_level: int = Field(
        default=1, ge=1, le=5, description="Evol-Instruct complexity tier (1=Seed to 5=Concretized Multi-step)"
    )
    mutation_type: str = Field(
        default="seed",
        description="Applied mutation (seed, in_depth_constraint, in_breadth_domain_transfer, concretizing, noise_injection)",
    )
    quality_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual rubric scores (schema_adherence, completeness, token_density, composite)",
    )
    is_contaminated: bool = Field(
        default=False, description="Flag indicating test-set benchmark contamination"
    )
    is_duplicate: bool = Field(
        default=False, description="Flag indicating near-duplicate clustering"
    )
    passed_filters: bool = Field(
        default=True, description="Whether the sample passed deduplication & quality gates"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary domain/provenance metadata"
    )

    @field_validator("messages")
    @classmethod
    def validate_chatml_structure(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        """Ensure conversation contains at least a user prompt and assistant response."""
        roles = [msg.role for msg in v]
        if "user" not in roles or "assistant" not in roles:
            raise ValueError("ChatML conversation must contain both 'user' and 'assistant' roles")
        return v

    def get_prompt(self) -> str:
        """Extract user prompt text from conversation."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def get_response(self) -> str:
        """Extract assistant target response text from conversation."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return ""

    def get_system_prompt(self) -> str:
        """Extract system prompt text if present."""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return ""

    def to_chatml_jsonl(self) -> str:
        """Serialize sample to strict ChatML JSONL line."""
        return json.dumps(self.model_dump(), ensure_ascii=False)


class GenerateRequest(BaseModel):
    """Request payload for synthetic dataset generation."""

    model_config = ConfigDict(strict=True)

    num_samples: int = Field(default=100, ge=1, le=50000, description="Total samples to synthesize")
    domains: Optional[List[str]] = Field(
        default=None, description="Target domains (reasoning, code, math, safety, extraction)"
    )
    complexity_levels: Optional[List[int]] = Field(
        default=None, description="Allowed complexity levels (1-5)"
    )
    noise_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Rate of controlled noise injection"
    )
    seed: Optional[int] = Field(default=42, description="Random seed for reproducible generation")
    output_path: Optional[str] = Field(
        default=None, description="Optional filesystem path to export generated JSONL"
    )


class GenerateResponse(BaseModel):
    """Response payload returned from synthetic data generator."""

    model_config = ConfigDict(strict=True)

    status: str = "success"
    total_generated: int
    samples_preview: List[SyntheticSample]
    output_path: Optional[str] = None
    generation_time_sec: float


class CurateRequest(BaseModel):
    """Request payload for deduplication and quality rubric filtering."""

    model_config = ConfigDict(strict=True)

    input_path: Optional[str] = Field(default=None, description="Path to raw JSONL dataset")
    samples: Optional[List[SyntheticSample]] = Field(
        default=None, description="Direct in-memory list of samples to curate"
    )
    dedup_jaccard_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="MinHash / Jaccard similarity threshold for near-duplicate rejection"
    )
    contamination_ngram_n: int = Field(
        default=13, ge=2, le=50, description="N-gram shingle length for benchmark contamination gating"
    )
    contamination_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Max allowed benchmark overlap before contamination flag"
    )
    quality_score_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum composite rubric score required to pass"
    )
    output_path: Optional[str] = Field(
        default=None, description="Optional path to write curated JSONL"
    )


class CurationReport(BaseModel):
    """Statistical summary of the curation and rejection sampling pipeline."""

    model_config = ConfigDict(strict=True)

    total_input: int
    duplicates_removed: int
    contaminated_removed: int
    quality_rejected: int
    final_yield_count: int
    yield_rate: float = Field(..., description="Ratio of final curated samples to raw input")
    duration_seconds: float
    filtering_breakdown: Dict[str, int] = Field(default_factory=dict)


class CurateResponse(BaseModel):
    """Response payload returned from dataset curation."""

    model_config = ConfigDict(strict=True)

    status: str = "success"
    report: CurationReport
    curated_samples_preview: List[SyntheticSample]
    output_path: Optional[str] = None


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation hyperparameter configuration."""

    model_config = ConfigDict(strict=True)

    r: int = Field(default=4, ge=1, le=64, description="LoRA rank dimension (low-rank bottleneck)")
    alpha: float = Field(default=8.0, gt=0, description="LoRA scaling factor alpha")
    dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout rate")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Names of linear submodules to wrap with LoRA adapters",
    )
    learning_rate: float = Field(default=1e-3, gt=0, description="Optimization learning rate")
    batch_size: int = Field(default=16, ge=1, description="Mini-batch training size")
    epochs: int = Field(default=3, ge=1, description="Training epochs")
    weight_decay: float = Field(default=0.01, ge=0.0, description="AdamW weight decay")
    max_seq_len: int = Field(default=64, ge=16, le=512, description="Max sequence length for student model")
    device: str = Field(default="cpu", description="Compute device ('cpu' or 'cuda')")


class DistillationMetrics(BaseModel):
    """Detailed convergence, parameter efficiency, and benchmark evaluation metrics."""

    model_config = ConfigDict(strict=True)

    initial_loss: float
    final_loss: float
    loss_history: List[float]
    training_duration_sec: float
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float = Field(
        ..., description="Percentage of model parameters trained with LoRA (<5% target)"
    )
    base_accuracy: float = Field(
        ..., description="Task accuracy of frozen Base Student model before LoRA distillation"
    )
    distilled_accuracy: float = Field(
        ..., description="Task accuracy of Student model after LoRA distillation"
    )
    structured_output_validity: float = Field(
        ..., description="Percentage of outputs adhering to required structured ChatML syntax"
    )
    adapter_path: Optional[str] = None


class TrainLoRARequest(BaseModel):
    """Request payload for initiating LoRA distillation on curated data."""

    model_config = ConfigDict(strict=True)

    curated_data_path: Optional[str] = Field(default=None, description="Path to curated JSONL dataset")
    samples: Optional[List[SyntheticSample]] = Field(
        default=None, description="Direct in-memory curated samples"
    )
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)
    save_adapter_path: Optional[str] = Field(
        default=None, description="Path to save adapter checkpoint weights"
    )


class TrainLoRAResponse(BaseModel):
    """Response payload returned from LoRA distillation training."""

    model_config = ConfigDict(strict=True)

    status: str = "success"
    metrics: DistillationMetrics


class HealthResponse(BaseModel):
    """API health and operational readiness response."""

    model_config = ConfigDict(strict=True)

    status: str = "healthy"
    simulation_mode: bool = True
    device: str = "cpu"
    torch_version: str
    version: str = "0.1.0"


class MetricsResponse(BaseModel):
    """Current state metrics and historical execution tracking."""

    model_config = ConfigDict(strict=True)

    status: str = "success"
    last_generation: Optional[Dict[str, Any]] = None
    last_curation: Optional[CurationReport] = None
    last_distillation: Optional[DistillationMetrics] = None
    pipeline_summary: Dict[str, Any] = Field(default_factory=dict)
