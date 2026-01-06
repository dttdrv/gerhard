"""
INT8 Weight Quantization for ASNN-Goose.

Reference: Section 5 of ASNN-Goose blueprint.

This module implements:
1. Symmetric INT8 quantization for weights
2. Quantization-aware training (QAT) support
3. Post-training quantization (PTQ) support
4. Utilities for quantizing existing models

INT8 quantization reduces model size by 4x and enables efficient
integer arithmetic on Tensor Cores.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class INT8Quantizer(torch.autograd.Function):
    """
    Symmetric INT8 weight quantization with STE.

    Quantization formula:
        q = clamp(round(w / scale), -127, 127)
        w_q = q * scale

    Scale is computed as: scale = max(|w|) / 127
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to INT8.

        Args:
            weight: FP32 weights to quantize
            scale: Optional pre-computed scale. If None, computed from weights.

        Returns:
            quantized_weight: Simulated quantized weights (still FP32 for gradients)
            scale: Quantization scale used
        """
        # Compute scale if not provided
        if scale is None:
            max_val = torch.max(torch.abs(weight))
            scale = max_val / 127.0
            scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero

        # Quantize
        q = torch.round(weight / scale)
        q = torch.clamp(q, -127, 127)

        # Dequantize (simulated quantization for training)
        w_q = q * scale

        # Save for backward
        ctx.save_for_backward(weight, scale)

        return w_q, scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_scale: torch.Tensor):
        """
        STE backward pass.
        Gradient passes through the quantization unchanged.
        """
        weight, scale = ctx.saved_tensors

        # STE: pass gradient through
        grad_weight = grad_output.clone()

        # Optional: clip gradient for saturated values
        # Values that would clip at ±127 have zero true gradient
        saturated = (weight / scale).abs() >= 127
        grad_weight[saturated] *= 0.5  # Reduce but don't zero

        return grad_weight, None


class QuantizedLinear(nn.Module):
    """
    Linear layer with INT8 weight quantization.

    Supports both:
    - Quantization-aware training (QAT): quantize during forward pass
    - Post-training quantization (PTQ): quantize once after training

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        quantize_during_training: If True, use QAT. If False, use PTQ mode.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_during_training: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_during_training = quantize_during_training

        # FP32 weights (master copy for training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantization state
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("is_quantized", torch.tensor(False))

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights and bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        if self.quantize_during_training and self.training:
            # QAT: quantize weights during training
            w_q, scale = INT8Quantizer.apply(self.weight, None)
            self.weight_scale = scale.detach()
        elif self.is_quantized.item():
            # Inference with pre-quantized weights
            w_q = self.weight
        else:
            # No quantization (standard FP32)
            w_q = self.weight

        return F.linear(x, w_q, self.bias)

    def quantize_weights(self):
        """
        Permanently quantize weights for inference.
        Called once after training for PTQ.
        """
        with torch.no_grad():
            max_val = torch.max(torch.abs(self.weight))
            scale = max_val / 127.0
            scale = torch.clamp(scale, min=1e-8)

            q = torch.round(self.weight / scale)
            q = torch.clamp(q, -127, 127)
            w_q = q * scale

            self.weight.copy_(w_q)
            self.weight_scale = scale
            self.is_quantized = torch.tensor(True)

    def dequantize_weights(self):
        """
        Mark weights as not quantized (for fine-tuning).
        Note: This doesn't restore original FP32 precision.
        """
        self.is_quantized = torch.tensor(False)

    def get_quantization_stats(self) -> Dict[str, float]:
        """Get quantization statistics."""
        with torch.no_grad():
            max_val = torch.max(torch.abs(self.weight)).item()
            scale = self.weight_scale.item()

            # Compute quantization error
            q = torch.round(self.weight / scale)
            q = torch.clamp(q, -127, 127)
            w_q = q * scale
            error = (self.weight - w_q).abs().mean().item()

            return {
                "max_weight": max_val,
                "scale": scale,
                "mean_quantization_error": error,
                "is_quantized": self.is_quantized.item(),
            }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, qat={self.quantize_during_training}"
        )


def quantize_model_weights(
    model: nn.Module,
    inplace: bool = True,
) -> nn.Module:
    """
    Apply post-training quantization to all QuantizedLinear layers.

    Args:
        model: Model containing QuantizedLinear layers
        inplace: If True, modify model in place

    Returns:
        Quantized model
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.quantize_weights()

    return model


def replace_linear_with_quantized(
    model: nn.Module,
    quantize_during_training: bool = True,
    skip_names: Optional[set] = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with QuantizedLinear.

    Args:
        model: Model to modify
        quantize_during_training: QAT mode flag
        skip_names: Set of module names to skip (e.g., {"head", "embedding"})

    Returns:
        Modified model
    """
    skip_names = skip_names or set()

    for name, module in model.named_children():
        if name in skip_names:
            continue

        if isinstance(module, nn.Linear):
            # Replace with quantized version
            quantized = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                quantize_during_training=quantize_during_training,
            )
            # Copy weights
            quantized.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                quantized.bias.data.copy_(module.bias.data)

            setattr(model, name, quantized)
        else:
            # Recurse
            replace_linear_with_quantized(
                module, quantize_during_training, skip_names
            )

    return model


def get_model_quantization_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get quantization statistics for all QuantizedLinear layers.

    Args:
        model: Model to analyze

    Returns:
        Dictionary of layer statistics
    """
    summary = {
        "num_quantized_layers": 0,
        "total_quantized_params": 0,
        "layer_stats": {},
    }

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            stats = module.get_quantization_stats()
            summary["layer_stats"][name] = stats
            summary["num_quantized_layers"] += 1
            summary["total_quantized_params"] += (
                module.weight.numel() +
                (module.bias.numel() if module.bias is not None else 0)
            )

    # Compute average error
    if summary["layer_stats"]:
        errors = [s["mean_quantization_error"] for s in summary["layer_stats"].values()]
        summary["mean_error"] = sum(errors) / len(errors)
    else:
        summary["mean_error"] = 0.0

    return summary


class QuantizedEmbedding(nn.Module):
    """
    Embedding layer with INT8 quantization.

    Embeddings are quantized per-row to preserve vocabulary-specific scales.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        quantize_during_training: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.quantize_during_training = quantize_during_training

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.register_buffer("scales", torch.ones(num_embeddings))
        self.register_buffer("is_quantized", torch.tensor(False))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.

        Args:
            input: Token indices

        Returns:
            Embeddings
        """
        if self.quantize_during_training and self.training:
            # Per-row quantization
            with torch.no_grad():
                max_vals = torch.max(torch.abs(self.weight), dim=1)[0]
                scales = max_vals / 127.0
                scales = torch.clamp(scales, min=1e-8)
                self.scales = scales

            # Quantize and dequantize
            q = torch.round(self.weight / scales.unsqueeze(1))
            q = torch.clamp(q, -127, 127)
            w_q = q * scales.unsqueeze(1)

            return F.embedding(
                input, w_q,
                padding_idx=self.padding_idx,
            )
        else:
            return F.embedding(
                input, self.weight,
                padding_idx=self.padding_idx,
            )

    def quantize_weights(self):
        """Permanently quantize for inference."""
        with torch.no_grad():
            max_vals = torch.max(torch.abs(self.weight), dim=1)[0]
            scales = max_vals / 127.0
            scales = torch.clamp(scales, min=1e-8)

            q = torch.round(self.weight / scales.unsqueeze(1))
            q = torch.clamp(q, -127, 127)
            w_q = q * scales.unsqueeze(1)

            self.weight.copy_(w_q)
            self.scales = scales
            self.is_quantized = torch.tensor(True)
