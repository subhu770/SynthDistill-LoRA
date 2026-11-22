"""End-to-end Distillation Benchmark: 1,000-sample generation -> curation -> LoRA training -> held-out evaluation."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.schemas import LoRAConfig
from src.data.deduplicator import Deduplicator
from src.data.quality_judge import QualityJudge
from src.data.synthetic_generator import SyntheticDataGenerator
from src.evaluation.eval_harness import EvaluationHarness
from src.training.lora_layer import count_parameters
from src.training.lora_trainer import LoRATrainer
from src.training.student_model import ChatMLTokenizer, MiniCausalLM


def print_banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def run_benchmark() -> None:
    t_start = time.time()
    print_banner("SynthDistill-LoRA: End-to-End Distillation Benchmark")
    print(f"Environment: CPU | SIMULATION_MODE={os.getenv('SIMULATION_MODE', 'true')} | Seed=42")

    # ---------------------------------------------------------
    # STAGE 1: Synthetic Dataset Generation (1,000 Samples)
    # ---------------------------------------------------------
    print_banner("Stage 1: Evol-Instruct Synthetic Data Generation")
    num_samples = 1000
    noise_rate = 0.12
    print(f"Generating {num_samples} samples across 5 domains with {noise_rate*100:.1f}% controlled noise...")

    t0 = time.time()
    generator = SyntheticDataGenerator(seed=42)
    raw_samples = generator.generate(
        num_samples=num_samples,
        domains=["reasoning", "code", "math", "safety", "extraction"],
        complexity_levels=[1, 2, 3, 4, 5],
        noise_rate=noise_rate,
    )
    gen_time = time.time() - t0
    print(f"-> Generated {len(raw_samples)} raw ChatML samples in {gen_time:.2f}s ({len(raw_samples)/gen_time:.1f} samples/sec)")

    raw_path = "data/benchmark_raw_1000.jsonl"
    SyntheticDataGenerator.save_to_jsonl(raw_samples, raw_path)
    print(f"-> Saved raw dataset to: {raw_path}")

    # ---------------------------------------------------------
    # STAGE 2: Curation, Deduplication & Quality Gating
    # ---------------------------------------------------------
    print_banner("Stage 2: Deduplication, Contamination Gating & Quality Filtering")
    t0 = time.time()
    deduplicator = Deduplicator(
        dedup_threshold=0.85,
        contamination_threshold=0.70,
        ngram_n=3,
        contamination_ngram_n=13,
    )
    judge = QualityJudge(quality_threshold=0.70)

    curated_samples, curation_report = judge.curate_pipeline(
        samples=raw_samples,
        deduplicator=deduplicator,
        duration_seconds=0.0,
    )
    curate_time = time.time() - t0
    curation_report.duration_seconds = round(curate_time, 4)

    curated_path = "data/benchmark_curated.jsonl"
    SyntheticDataGenerator.save_to_jsonl(curated_samples, curated_path)

    print(f"-> Total Input Samples:       {curation_report.total_input}")
    print(f"-> Near-Duplicates Removed:   {curation_report.duplicates_removed}")
    print(f"-> Contaminated Benchmarks:   {curation_report.contaminated_removed}")
    print(f"-> Low Quality Rejections:    {curation_report.quality_rejected}")
    print(f"-> Final Curated Yield:       {curation_report.final_yield_count} / {curation_report.total_input}")
    print(f"-> Curation Yield Rate:       {curation_report.yield_rate * 100:.2f}%")
    print(f"-> Curation Duration:         {curate_time:.2f}s ({len(raw_samples)/curate_time:.1f} samples/sec)")

    # ---------------------------------------------------------
    # STAGE 3: LoRA Student Distillation Training
    # ---------------------------------------------------------
    print_banner("Stage 3: LoRA Student Model Distillation (CPU)")
    lora_cfg = LoRAConfig(
        r=4,
        alpha=8.0,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        learning_rate=3e-3,
        batch_size=16,
        epochs=4,
        max_seq_len=64,
        device="cpu",
    )

    tokenizer = ChatMLTokenizer()
    base_student = MiniCausalLM(
        vocab_size=tokenizer.VOCAB_SIZE,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=lora_cfg.max_seq_len,
    )

    trainer = LoRATrainer(
        config=lora_cfg,
        base_model=base_student,
        tokenizer=tokenizer,
    )

    total_p, train_p, frozen_p, train_pct = count_parameters(trainer.model)
    reduction_pct = 100.0 - train_pct

    print(f"Student Architecture: MiniCausalLM (2 Layers, 2 Heads, d_model=64)")
    print(f"-> Total Model Parameters:      {total_p:,}")
    print(f"-> Trainable LoRA Parameters:   {train_p:,}")
    print(f"-> Frozen Base Parameters:      {frozen_p:,}")
    print(f"-> Trainable Ratio:             {train_pct:.2f}% (Target: < 5%)")
    print(f"-> Parameter Reduction:         {reduction_pct:.2f}%")

    training_data = curated_samples[: min(250, len(curated_samples))]
    checkpoint_dir = "checkpoints/benchmark_adapter"

    print(f"\nTraining on {len(training_data)} curated samples for {lora_cfg.epochs} epochs...")
    t0 = time.time()
    distill_metrics = trainer.train(
        train_samples=training_data,
        save_adapter_path=checkpoint_dir,
    )
    train_time = time.time() - t0

    print(f"-> Initial Cross-Entropy Loss: {distill_metrics.initial_loss:.4f}")
    print(f"-> Final Cross-Entropy Loss:   {distill_metrics.final_loss:.4f}")
    print(f"-> Loss Delta:                 {distill_metrics.initial_loss - distill_metrics.final_loss:+.4f}")
    print(f"-> Training Time:              {train_time:.2f}s")
    print(f"-> Adapter Checkpoint:         {checkpoint_dir}/lora_weights.pt")

    # ---------------------------------------------------------
    # STAGE 4: Held-Out Benchmark Evaluation
    # ---------------------------------------------------------
    print_banner("Stage 4: Held-Out Test Set Evaluation")
    eval_harness = EvaluationHarness(tokenizer=tokenizer, test_samples_count=40, seed=123)

    unadapted_baseline = MiniCausalLM(
        vocab_size=tokenizer.VOCAB_SIZE,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=lora_cfg.max_seq_len,
    )

    comparison = eval_harness.compare_base_vs_distilled(
        base_model=unadapted_baseline,
        distilled_model=trainer.model,
        max_seq_len=lora_cfg.max_seq_len,
        device="cpu",
    )

    base_acc = comparison["base_student"]["accuracy"]
    dist_acc = comparison["lora_distilled_student"]["accuracy"]
    base_loss = comparison["base_student"]["loss"]
    dist_loss = comparison["lora_distilled_student"]["loss"]

    # ---------------------------------------------------------
    # FINAL BENCHMARK SUMMARY DASHBOARD
    # ---------------------------------------------------------
    total_pipeline_time = time.time() - t_start
    print_banner("SynthDistill-LoRA: Distillation Benchmark Report")
    report_text = f"""
+-----------------------------------------------------------------------------+
|                           BENCHMARK EXECUTION SUMMARY                       |
+-----------------------------------------------------------------------------+
| Metric                               | Result                               |
+--------------------------------------+--------------------------------------+
| Raw Samples Generated                | {num_samples:,} samples                    |
| Generation Latency                   | {gen_time:.2f}s ({num_samples/gen_time:.1f} samp/s)        |
| Near-Duplicates Removed              | {curation_report.duplicates_removed} samples ({curation_report.duplicates_removed/num_samples*100:.1f}%)         |
| Benchmark Contamination Gated        | {curation_report.contaminated_removed} samples ({curation_report.contaminated_removed/num_samples*100:.1f}%)         |
| Quality Rubric Rejected              | {curation_report.quality_rejected} samples ({curation_report.quality_rejected/num_samples*100:.1f}%)         |
| Curated Dataset Yield                | {curation_report.final_yield_count} samples ({curation_report.yield_rate*100:.1f}%)          |
+--------------------------------------+--------------------------------------+
| Total Student Model Parameters       | {total_p:,} parameters                |
| Trainable LoRA Parameters (r={lora_cfg.r})      | {train_p:,} parameters ({train_pct:.2f}%)       |
| Parameter Reduction Percentage       | {reduction_pct:.2f}% reduction                |
| Adapter Rank & Alpha                 | r={lora_cfg.r}, alpha={lora_cfg.alpha}                  |
| Distillation Loss Convergence        | {distill_metrics.initial_loss:.4f} -> {distill_metrics.final_loss:.4f} (Delta: {distill_metrics.initial_loss - distill_metrics.final_loss:+.4f}) |
+--------------------------------------+--------------------------------------+
| Base Student Held-Out Accuracy       | {base_acc*100:.2f}%                                |
| Distilled LoRA Student Accuracy      | {dist_acc*100:.2f}%                                |
| Distillation Accuracy Improvement    | +{(dist_acc - base_acc)*100:.2f}%                               |
| ChatML Structured Output Validity    | {comparison['lora_distilled_student']['structured_output_validity']*100:.1f}%                              |
| Total Benchmark Execution Time       | {total_pipeline_time:.2f}s                               |
+-----------------------------------------------------------------------------+
"""
    print(report_text)
    print(">> Benchmark completed successfully with 100% CPU compatibility & 0 external API calls.")


if __name__ == "__main__":
    run_benchmark()
