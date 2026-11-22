"""Pure PyTorch Low-Rank Adaptation (LoRA) implementation for Linear layers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps or replaces an nn.Linear layer with low-rank adapter matrices A and B.

    Mathematical formulation:
        h = W_0 x + (alpha / r) * B * A * dropout(x)
    where:
        W_0 is frozen: requires_grad = False
        A in R^(r x d_in): initialized with Gaussian / Kaiming uniform
        B in R^(d_out x r): initialized to zero (so Delta W = 0 at start)
        alpha / r is the scaling constant
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)

        # Freeze base linear layer weights and bias
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=base_layer.weight.dtype))

        # Dropout layer
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Weight merging state flag
        self.merged = False
        self.merge_weights = merge_weights

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize adapter weights: A with Kaiming uniform, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self) -> None:
        """Merge Delta W = (alpha/r) * B @ A directly into the base layer weight."""
        if self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
        self.merged = True

    def unmerge(self) -> None:
        """Subtract Delta W from base layer weight to restore frozen base weights."""
        if not self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing base linear projection plus scaled low-rank adapter projection."""
        if self.merged:
            return self.base_layer(x)

        # Base frozen projection: W_0 * x
        result = self.base_layer(x)

        # Adapter path: (alpha / r) * (x @ A^T @ B^T)
        dropped_x = self.lora_dropout(x)
        adapter_out = (dropped_x @ self.lora_A.t()) @ self.lora_B.t()
        result = result + adapter_out * self.scaling
        return result

    def get_adapter_state(self) -> Dict[str, torch.Tensor]:
        """Return low-rank parameters only."""
        return {
            "lora_A": self.lora_A.data.clone(),
            "lora_B": self.lora_B.data.clone(),
        }

    def load_adapter_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load low-rank parameters into adapter."""
        if "lora_A" in state_dict:
            self.lora_A.data.copy_(state_dict["lora_A"])
        if "lora_B" in state_dict:
            self.lora_B.data.copy_(state_dict["lora_B"])


def count_parameters(model: nn.Module) -> Tuple[int, int, int, float]:
    """Calculate parameter efficiency metrics.

    Returns:
        Tuple of (total_params, trainable_params, frozen_params, trainable_pct)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return total_params, trainable_params, frozen_params, round(trainable_pct, 4)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05,
) -> Tuple[nn.Module, List[str]]:
    """Freeze base model parameters and inject LoRALinear into specified target modules.

    Args:
        model: Base PyTorch neural network.
        target_modules: List of module name substrings to replace (e.g. ['q_proj', 'v_proj']).
        r: Low-rank dimension.
        alpha: LoRA scaling hyperparameter.
        dropout: LoRA dropout probability.

    Returns:
        Tuple of (adapted_model, list_of_modified_module_paths)
    """
    # 1. Freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False

    replaced_modules: List[str] = []

    # 2. Recursively find and replace target linear layers
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                # Locate parent module
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                layer_name = parts[-1]

                # Create LoRA adapter wrapping the original linear layer
                lora_layer = LoRALinear(
                    base_layer=module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, layer_name, lora_layer)
                replaced_modules.append(name)
                break

    return model, replaced_modules


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only trainable LoRA parameters from an adapted model."""
    state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            state_dict[name] = param.data.clone()
    return state_dict


def load_lora_state_dict(model: nn.Module, adapter_state: Dict[str, torch.Tensor]) -> None:
    """Load extracted LoRA parameters into matching layers."""
    model_state = model.state_dict()
    for name, tensor in adapter_state.items():
        if name in model_state:
            model_state[name].copy_(tensor)
        else:
            raise KeyError(f"Target LoRA parameter {name} not found in model state")
