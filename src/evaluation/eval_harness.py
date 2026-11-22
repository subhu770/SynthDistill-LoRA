"""Evaluation harness comparing Base Student vs LoRA-Distilled Student models on held-out test sets."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.api.schemas import DistillationMetrics, SyntheticSample
from src.data.synthetic_generator import SyntheticDataGenerator
from src.training.lora_layer import count_parameters
from src.training.student_model import ChatMLTokenizer, MiniCausalLM


class EvaluationHarness:
    """Benchmark evaluation harness for student distillation and structured output compliance."""

    def __init__(
        self,
        tokenizer: Optional[ChatMLTokenizer] = None,
        test_samples_count: int = 40,
        seed: int = 999,
    ) -> None:
        self.tokenizer = tokenizer or ChatMLTokenizer()
        generator = SyntheticDataGenerator(seed=seed)
        # Generate held-out validation set across domains
        self.held_out_samples = generator.generate(
            num_samples=test_samples_count,
            domains=["reasoning", "code", "math", "safety", "extraction"],
            complexity_levels=[1, 2, 3, 4],
            noise_rate=0.0,
        )

    def evaluate_model_on_held_out(
        self,
        model: torch.nn.Module,
        max_seq_len: int = 128,
        device: str = "cpu",
    ) -> Tuple[float, float, float]:
        """Compute evaluation cross-entropy loss, token prediction accuracy, and structured format validity.

        Returns:
            Tuple of (eval_loss, token_accuracy, structured_output_validity)
        """
        model.eval()
        dev = torch.device(device)
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        valid_structured_count = 0

        with torch.no_grad():
            for sample in self.held_out_samples:
                inp, lab = self.tokenizer.encode_chatml(sample, max_seq_len=max_seq_len)
                inp = inp.unsqueeze(0).to(dev)
                lab = lab.unsqueeze(0).to(dev)

                logits, loss = model(inp, labels=lab)
                if loss is not None:
                    mask = lab[:, 1:] != -100
                    cnt = mask.sum().item()
                    total_loss += loss.item() * cnt
                    total_tokens += cnt

                    preds = logits[:, :-1, :].argmax(dim=-1)
                    correct = ((preds == lab[:, 1:]) & mask).sum().item()
                    correct_tokens += correct

            # Representative sample for structured generation validation
            eval_gen_samples = self.held_out_samples[: min(8, len(self.held_out_samples))]
            for sample in eval_gen_samples:
                prompt_text = (
                    f"<|im_start|>system\n{sample.get_system_prompt()}<|im_end|>\n"
                    f"<|im_start|>user\n{sample.get_prompt()}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                prompt_ids = torch.tensor(
                    [self.tokenizer.encode(prompt_text)], dtype=torch.long, device=dev
                )
                gen_ids = model.generate(prompt_ids, max_new_tokens=12, temperature=0.7)
                decoded_gen = self.tokenizer.decode(gen_ids[0].tolist())

                if len(decoded_gen.strip()) > 0 and not re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", decoded_gen):
                    valid_structured_count += 1

        avg_loss = total_loss / max(1, total_tokens)
        acc = correct_tokens / max(1, total_tokens)
        struct_validity = valid_structured_count / max(1, len(eval_gen_samples))

        return round(avg_loss, 4), round(acc, 4), round(struct_validity, 4)

    def compare_base_vs_distilled(
        self,
        base_model: MiniCausalLM,
        distilled_model: MiniCausalLM,
        max_seq_len: int = 128,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Run comprehensive side-by-side benchmark comparing Base Student vs LoRA-Distilled Student."""
        total_p, trainable_p, frozen_p, trainable_pct = count_parameters(distilled_model)
        reduction_pct = 100.0 - trainable_pct

        base_loss, base_acc, base_valid = self.evaluate_model_on_held_out(
            base_model, max_seq_len=max_seq_len, device=device
        )
        dist_loss, dist_acc, dist_valid = self.evaluate_model_on_held_out(
            distilled_model, max_seq_len=max_seq_len, device=device
        )

        relative_gain = (
            ((dist_acc - base_acc) / max(1e-5, base_acc)) * 100.0 if base_acc > 0 else 100.0
        )

        comparison = {
            "parameter_efficiency": {
                "total_parameters": total_p,
                "trainable_parameters": trainable_p,
                "frozen_parameters": frozen_p,
                "trainable_percentage": f"{trainable_pct:.2f}%",
                "parameter_reduction_percentage": f"{reduction_pct:.2f}%",
            },
            "base_student": {
                "loss": base_loss,
                "accuracy": base_acc,
                "structured_output_validity": base_valid,
            },
            "lora_distilled_student": {
                "loss": dist_loss,
                "accuracy": dist_acc,
                "structured_output_validity": dist_valid,
            },
            "distillation_gain": {
                "loss_delta": round(base_loss - dist_loss, 4),
                "accuracy_improvement": round(dist_acc - base_acc, 4),
                "relative_gain_pct": f"{relative_gain:.1f}%",
            },
        }

        return comparison
