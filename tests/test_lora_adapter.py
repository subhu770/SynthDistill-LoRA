"""Unit tests for PyTorch LoRA adapter, frozen gradients, and parameter efficiency."""

import os
import tempfile
import torch
import torch.nn as nn
import pytest

from src.api.schemas import LoRAConfig, SyntheticSample, ChatMessage
from src.training.lora_layer import (
    LoRALinear,
    apply_lora_to_model,
    count_parameters,
    get_lora_state_dict,
    load_lora_state_dict,
)
from src.training.student_model import ChatMLTokenizer, MiniCausalLM
from src.training.lora_trainer import LoRATrainer


class TestLoRALinearLayer:
    """Verifies LoRA linear math, weight freezing, scaling, and merge/unmerge."""

    def test_lora_initialization_equivalence(self):
        """At initialization, B=0 implies Delta W = 0, so output equals base layer output."""
        in_dim, out_dim, r = 32, 64, 4
        base_linear = nn.Linear(in_dim, out_dim)
        lora_layer = LoRALinear(base_layer=base_linear, r=r, alpha=8.0, dropout=0.0)

        # Base weights must be frozen
        assert lora_layer.base_layer.weight.requires_grad is False
        if lora_layer.base_layer.bias is not None:
            assert lora_layer.base_layer.bias.requires_grad is False

        # Only lora_A and lora_B are trainable
        assert lora_layer.lora_A.requires_grad is True
        assert lora_layer.lora_B.requires_grad is True

        # Test identical output at init
        x = torch.randn(2, 5, in_dim)
        with torch.no_grad():
            base_out = base_linear(x)
            lora_out = lora_layer(x)

        assert torch.allclose(base_out, lora_out, atol=1e-5), "LoRA output at init must match base layer exactly"

    def test_gradient_flow_only_to_lora_matrices(self):
        """Backpropagation must compute gradients ONLY for lora_A and lora_B."""
        in_dim, out_dim, r = 16, 32, 4
        base_linear = nn.Linear(in_dim, out_dim)
        lora_layer = LoRALinear(base_layer=base_linear, r=r, alpha=8.0)

        x = torch.randn(2, in_dim, requires_grad=True)
        out = lora_layer(x)
        loss = out.sum()
        loss.backward()

        assert lora_layer.base_layer.weight.grad is None, "Frozen base weight must have None grad"
        assert lora_layer.lora_A.grad is not None, "lora_A must receive gradients"
        assert lora_layer.lora_B.grad is not None, "lora_B must receive gradients"

    def test_merge_and_unmerge(self):
        """Merging updates base weights with Delta W; unmerging restores original weights."""
        in_dim, out_dim, r = 16, 16, 2
        base_linear = nn.Linear(in_dim, out_dim)
        orig_weight = base_linear.weight.clone()

        lora_layer = LoRALinear(base_layer=base_linear, r=r, alpha=4.0)
        # Give lora_B non-zero values to simulate post-training state
        nn.init.normal_(lora_layer.lora_B, std=0.1)

        x = torch.randn(1, in_dim)
        out_unmerged = lora_layer(x)

        # Merge
        lora_layer.merge()
        assert lora_layer.merged is True
        out_merged = lora_layer(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5), "Merged output must match adapted forward output"
        assert not torch.allclose(base_linear.weight, orig_weight), "Base weight must be modified after merge"

        # Unmerge
        lora_layer.unmerge()
        assert lora_layer.merged is False
        assert torch.allclose(base_linear.weight, orig_weight, atol=1e-5), "Unmerge must restore exact original weights"


class TestLoRAStudentDistillation:
    """Verifies parameter efficiency (<5% trainable) and training loss decrease."""

    def test_trainable_parameter_percentage_below_five_percent(self):
        """Verify that LoRA adaptation strictly targets < 5% trainable parameters."""
        model = MiniCausalLM(vocab_size=ChatMLTokenizer.VOCAB_SIZE, d_model=64, n_heads=2, n_layers=2)
        total_before = sum(p.numel() for p in model.parameters())

        adapted_model, replaced = apply_lora_to_model(
            model=model,
            target_modules=["q_proj", "v_proj"],
            r=4,
            alpha=8.0,
            dropout=0.05,
        )

        assert len(replaced) > 0
        total_p, train_p, frozen_p, train_pct = count_parameters(adapted_model)

        assert train_pct < 5.0, f"LoRA trainable parameters must be < 5%, got {train_pct:.2f}%"
        assert train_p > 0
        assert frozen_p > 0
        assert total_p == train_p + frozen_p

    def test_training_reduces_loss(self):
        """Verify that LoRA training on CPU causes cross-entropy loss to converge downwards."""
        config = LoRAConfig(
            r=4,
            alpha=8.0,
            learning_rate=3e-3,
            batch_size=8,
            epochs=5,
            max_seq_len=64,
            device="cpu",
        )

        samples = [
            SyntheticSample(
                id=f"train_{i}",
                messages=[
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content=f"Compute the output value for operation id {i % 3}"),
                    ChatMessage(role="assistant", content=f"The resolved value is constant key_{i % 3}"),
                ],
                domain="math",
            )
            for i in range(24)
        ]

        trainer = LoRATrainer(config=config)
        metrics = trainer.train(train_samples=samples)

        assert metrics.final_loss < metrics.initial_loss, (
            f"Final loss {metrics.final_loss} should be lower than initial loss {metrics.initial_loss}"
        )
        assert len(metrics.loss_history) > 1
        assert metrics.trainable_percentage < 5.0

    def test_adapter_save_and_load(self):
        """Verify low-rank adapter serialization and deserialization."""
        config = LoRAConfig(r=4, alpha=8.0, device="cpu")
        trainer = LoRATrainer(config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            weight_path = trainer.save_adapter(tmp_dir)
            assert os.path.exists(weight_path)
            assert os.path.exists(os.path.join(tmp_dir, "adapter_config.json"))

            # Create another fresh trainer and load adapter
            new_trainer = LoRATrainer(config=config)
            new_trainer.load_adapter(weight_path)

            orig_state = get_lora_state_dict(trainer.model)
            loaded_state = get_lora_state_dict(new_trainer.model)

            for k in orig_state:
                assert torch.allclose(orig_state[k], loaded_state[k])
