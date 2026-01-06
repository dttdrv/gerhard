"""
LoRA (Low-Rank Adaptation) for Test-Time Training.

Reference: Section 7 of ASNN-Goose blueprint.

This module implements:
1. LoRAAdapter: Low-rank decomposition for weight updates
2. LoRALinear: Linear layer with LoRA integration
3. Utilities for applying LoRA to existing models

During TTT, only the LoRA parameters are updated while the main
model weights remain frozen. This enables efficient adaptation
with bounded parameter changes.

Reference: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Set, Tuple
import math


class LoRAAdapter(nn.Module):
    """
    Low-rank adapter for efficient fine-tuning.

    The adapter computes: h = W_0 * x + (B @ A) * x * (alpha / r)

    Where:
    - W_0: Original frozen weights
    - A: Low-rank down-projection (r x in_features)
    - B: Low-rank up-projection (out_features x r)
    - alpha: Scaling factor
    - r: Rank

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Low-rank dimension
        alpha: Scaling factor (default: rank, so scaling = 1)
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        # A: down-projection, initialized with Kaiming
        # B: up-projection, initialized with zeros (no initial change)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize LoRA parameters."""
        # A: Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: Zero initialization (initially no modification)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta: (x @ A.T) @ B.T * scaling

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output (..., out_features)
        """
        x = self.dropout(x)
        # Compute: x @ A.T @ B.T * scaling
        # = (x @ A.T) @ B.T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def get_delta_weight(self) -> torch.Tensor:
        """
        Get the effective weight delta: B @ A * scaling

        Returns:
            Weight delta (out_features, in_features)
        """
        return (self.lora_B @ self.lora_A) * self.scaling

    def get_delta_norm(self) -> float:
        """Compute norm of the effective weight delta."""
        delta = self.get_delta_weight()
        return delta.norm().item()

    def merge_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA into base weight.

        Args:
            weight: Base weight (out_features, in_features)

        Returns:
            Merged weight
        """
        return weight + self.get_delta_weight()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


class LoRALinear(nn.Module):
    """
    Linear layer with integrated LoRA adapter.

    Computes: y = W * x + b + LoRA(x)

    The base weight W can be frozen while LoRA parameters are trained.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        freeze_base: Whether to freeze base weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.freeze_base = freeze_base

        # Base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapter
        self.lora = LoRAAdapter(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Optionally freeze base
        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self):
        """Freeze the base linear layer weights."""
        for param in self.linear.parameters():
            param.requires_grad = False
        self.freeze_base = True

    def unfreeze_base_weights(self):
        """Unfreeze the base linear layer weights."""
        for param in self.linear.parameters():
            param.requires_grad = True
        self.freeze_base = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        return self.linear(x) + self.lora(x)

    def merge_and_unload(self) -> nn.Linear:
        """
        Merge LoRA into base weights and return standard Linear.

        Useful for inference without LoRA overhead.

        Returns:
            Merged nn.Linear layer
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.linear.bias is not None,
        )

        with torch.no_grad():
            merged.weight.copy_(self.lora.merge_weights(self.linear.weight))
            if self.linear.bias is not None:
                merged.bias.copy_(self.linear.bias)

        return merged

    def reset_lora(self):
        """Reset LoRA parameters to initial state (no modification)."""
        self.lora._reset_parameters()

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get list of LoRA parameters."""
        return [self.lora.lora_A, self.lora.lora_B]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.lora.rank}, frozen_base={self.freeze_base}"
        )


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[Set[str]] = None,
    freeze_base: bool = True,
) -> Tuple[nn.Module, Dict[str, LoRAAdapter]]:
    """
    Apply LoRA adapters to specified linear layers in a model.

    Args:
        model: Model to modify
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        target_modules: Set of module names to target (e.g., {"key_proj", "value_proj"})
                       If None, targets all Linear layers
        freeze_base: Whether to freeze base weights

    Returns:
        model: Modified model
        lora_modules: Dictionary of LoRA adapters by name
    """
    target_modules = target_modules or set()
    lora_modules: Dict[str, LoRAAdapter] = {}

    def should_apply_lora(name: str) -> bool:
        if not target_modules:
            return True  # Apply to all if no targets specified
        return any(target in name for target in target_modules)

    def apply_lora_recursive(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and should_apply_lora(full_name):
                # Replace with LoRALinear
                lora_linear = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                )

                # Copy original weights
                with torch.no_grad():
                    lora_linear.linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        lora_linear.linear.bias.copy_(child.bias)

                setattr(module, name, lora_linear)
                lora_modules[full_name] = lora_linear.lora

            else:
                # Recurse
                apply_lora_recursive(child, full_name)

    apply_lora_recursive(model)
    return model, lora_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.get_lora_parameters())
        elif isinstance(module, LoRAAdapter):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.

    Args:
        model: Model with LoRA layers

    Returns:
        State dict with LoRA parameters
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora.lora_A"] = module.lora.lora_A.data.clone()
            lora_state[f"{name}.lora.lora_B"] = module.lora.lora_B.data.clone()
        elif isinstance(module, LoRAAdapter):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.clone()
    return lora_state


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
):
    """
    Load LoRA parameters from a state dict.

    Args:
        model: Model with LoRA layers
        state_dict: State dict with LoRA parameters
        strict: Whether to raise error for missing/unexpected keys
    """
    model_state = model.state_dict()
    loaded_keys = []

    for key, value in state_dict.items():
        if key in model_state:
            model_state[key].copy_(value)
            loaded_keys.append(key)
        elif strict:
            raise KeyError(f"Unexpected key in state dict: {key}")

    if strict:
        expected_keys = {
            k for k in model_state.keys()
            if "lora_A" in k or "lora_B" in k
        }
        missing = expected_keys - set(loaded_keys)
        if missing:
            raise KeyError(f"Missing keys in state dict: {missing}")


def compute_lora_statistics(model: nn.Module) -> Dict[str, float]:
    """
    Compute statistics about LoRA adapters in a model.

    Args:
        model: Model with LoRA layers

    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_lora_layers": 0,
        "total_lora_params": 0,
        "mean_delta_norm": 0.0,
        "max_delta_norm": 0.0,
        "layer_norms": {},
    }

    delta_norms = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            norm = module.lora.get_delta_norm()
            delta_norms.append(norm)
            stats["layer_norms"][name] = norm
            stats["num_lora_layers"] += 1
            stats["total_lora_params"] += (
                module.lora.lora_A.numel() + module.lora.lora_B.numel()
            )
        elif isinstance(module, LoRAAdapter):
            norm = module.get_delta_norm()
            delta_norms.append(norm)
            stats["layer_norms"][name] = norm
            stats["num_lora_layers"] += 1
            stats["total_lora_params"] += module.lora_A.numel() + module.lora_B.numel()

    if delta_norms:
        stats["mean_delta_norm"] = sum(delta_norms) / len(delta_norms)
        stats["max_delta_norm"] = max(delta_norms)

    return stats
