"""
Ternary Spiking Activations with Straight-Through Estimator.

Reference: Sections 4 and 6.3 of ASNN-Goose blueprint.
Updated: BitNet b1.58 methodology (Microsoft Research, 2024)

This module implements:
1. TernaryQuantizer: Converts continuous activations to {-1, 0, +1}
2. AdaptiveTernarySpike: Uses data-dependent thresholds (Eq. 4)
3. FixedThresholdTernarySpike: Uses fixed threshold (baseline)
4. SpikeActivityTracker: Instrumentation for spike statistics
5. BitLinear: BitNet b1.58 linear layer with ternary weights

BitNet b1.58 Key Insights:
- Ternary weights {-1, 0, +1} with 1.58 bits per parameter
- Shadow weights (FP16/32) during training with on-the-fly quantization
- Absmean quantization for weights: scale = 1 / mean(|W|)
- Absmax quantization for activations: scale = 127 / max(|X|)
- STE via detach() pattern: w_q = w + lambda * (quant(w) - w).detach()
- Lambda warmup for gradual introduction of quantization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


# =============================================================================
# BitNet b1.58 Quantization Functions
# =============================================================================

def weight_quant_absmean(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """BitNet b1.58 weight quantization using absmean scaling.

    Maps weights to {-1, 0, +1} using:
    γ = 1 / mean(|W|)
    W_q = round(clip(W * γ, -1, 1))

    The scale factor γ is returned separately for dequantization during inference.
    During training with STE, we need W_q * (1/γ) to maintain proper magnitudes.

    Args:
        w: Weight tensor of any shape

    Returns:
        Tuple of (quantized_weight, scale) where quantized values are in {-1, 0, +1}
    """
    scale = w.abs().mean().clamp_(min=1e-5)
    gamma = 1.0 / scale
    w_quant = (w * gamma).round().clamp_(-1, 1)
    return w_quant, scale


def weight_quant_absmean_ste(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 weight quantization for STE training.

    Returns dequantized weights (W_q / γ) so magnitudes are preserved.
    The quantization is: W -> W_q -> W_q * scale

    This maintains the same output magnitude as the original weights
    while enforcing ternary structure.
    """
    scale = w.abs().mean().clamp_(min=1e-5)
    gamma = 1.0 / scale
    w_quant = (w * gamma).round().clamp_(-1, 1)
    # Dequantize to maintain magnitude for STE
    return w_quant * scale


def activation_quant_absmax(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """BitNet b1.58 activation quantization using absmax scaling.

    Maps activations to int8 range [-127, 127] using:
    scale = 127 / max(|X|)
    X_q = round(clamp(X * scale, -128, 127)) / scale

    Args:
        x: Activation tensor
        bits: Quantization bit width (default: 8)

    Returns:
        Quantized activation tensor (dequantized to original magnitude)
    """
    Qn = -(2 ** (bits - 1))
    Qp = 2 ** (bits - 1) - 1
    abs_max = x.abs().max(dim=-1, keepdim=True).values
    # Handle zero tensors: use scale of 1.0 (result will be zeros anyway)
    scale = torch.where(abs_max > 1e-5, Qp / abs_max, torch.ones_like(abs_max))
    x_quant = (x * scale).round().clamp_(Qn, Qp) / scale
    return x_quant


class BitLinear(nn.Linear):
    """BitNet b1.58 Linear layer with ternary weights and 8-bit activations.

    Uses shadow weights (full precision) with on-the-fly quantization.
    STE implemented via detach() pattern for proper gradient flow.

    Key Features:
    - Ternary weight quantization using absmean scaling
    - 8-bit activation quantization using absmax scaling
    - SubLN (LayerNorm before quantization) for training stability
    - Lambda parameter for warmup schedule

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: False for BitNet)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)
        self.ln = nn.LayerNorm(in_features)  # SubLN for stability
        self.register_buffer('lambda_', torch.tensor(0.0))  # Warmup factor

    def set_lambda(self, value: float):
        """Set quantization strength (0=full precision, 1=full quantization)."""
        self.lambda_.fill_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization based on lambda.

        Uses STE pattern: q = x + lambda * (quant(x) - x).detach()
        When lambda=0: full precision
        When lambda=1: full quantization
        """
        w = self.weight
        x_norm = self.ln(x)

        # Quantize activations with STE
        x_quant = x_norm + self.lambda_ * (activation_quant_absmax(x_norm) - x_norm).detach()

        # Quantize weights with STE (using the STE-compatible function)
        w_quant = w + self.lambda_ * (weight_quant_absmean_ste(w) - w).detach()

        return F.linear(x_quant, w_quant, self.bias)

    def get_quantized_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get fully quantized weight for export/inference.

        Returns:
            Tuple of (ternary_weights, scale) where ternary_weights is {-1, 0, +1}
        """
        return weight_quant_absmean(self.weight)

    def get_ternary_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ternary weight values {-1, 0, +1} with scale factor."""
        return weight_quant_absmean(self.weight)

    def extra_repr(self) -> str:
        return f'{self.in_features}, {self.out_features}, bias={self.bias is not None}, lambda={self.lambda_.item():.2f}'


class TernaryQuantizer(torch.autograd.Function):
    """
    Quantize continuous activations to {-1, 0, +1}.

    Forward: Apply ternary quantization based on threshold
    Backward: Straight-through estimator (STE) passes gradients unchanged

    This is the core operation that enables spiking behavior.
    The threshold determines the "firing threshold" for spikes.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        threshold: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Apply ternary quantization.

        Args:
            x: Continuous activations (any shape)
            threshold: Threshold for spike formation (scalar or broadcastable)
            training: Whether in training mode (affects gradient behavior)

        Returns:
            Ternary tensor in {-1, 0, +1}
        """
        # Save for backward
        ctx.save_for_backward(x, threshold)
        ctx.training = training

        # Ternary quantization
        output = torch.zeros_like(x)
        output[x > threshold] = 1.0
        output[x < -threshold] = -1.0
        # Values in [-threshold, threshold] remain 0 (no spike / no transmission)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Straight-through estimator (STE).

        The gradient passes through UNCHANGED as if quantization was identity.
        This allows training despite the non-differentiable quantization.

        IMPORTANT: Do NOT add gradient dampening here! The BitLinear class
        uses the detach() pattern which implements proper STE. Any modification
        here would conflict with that approach.

        Reference: Bengio et al. "Estimating or Propagating Gradients Through
        Stochastic Neurons for Conditional Computation"
        """
        # STE: gradient passes through unchanged - NO modifications!
        grad_x = grad_output.clone()

        # No gradient for threshold (computed from input statistics)
        return grad_x, None, None


class AdaptiveTernarySpike(nn.Module):
    """
    Adaptive-threshold ternary spiking (Eq. 4 from SpikingBrain integration).

    The threshold adapts to the input statistics:
        θ(x̃) = α * mean(|x̃|)
        x = sign(x̃) * I(|x̃| ≥ θ(x̃))

    This prevents the network from learning to scale activations
    to bypass the threshold, maintaining true sparsity.

    Args:
        alpha_init: Initial scale for adaptive threshold
        learnable_alpha: Whether alpha is a learnable parameter
        min_threshold: Minimum threshold to prevent collapse
        max_threshold: Maximum threshold to prevent dead neurons
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        learnable_alpha: bool = True,
        min_threshold: float = 0.01,
        max_threshold: float = 10.0,
    ):
        super().__init__()

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init))

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Running statistics for monitoring
        self.register_buffer("running_spike_density", torch.tensor(0.0))
        self.register_buffer("running_threshold", torch.tensor(alpha_init))
        self.register_buffer("num_updates", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive ternary quantization.

        Args:
            x: Pre-activation tensor (continuous)

        Returns:
            Ternary spikes in {-1, 0, +1}
        """
        # Compute adaptive threshold based on input statistics
        mean_abs = torch.mean(torch.abs(x))
        threshold = self.alpha * mean_abs

        # Clamp threshold to valid range
        threshold = torch.clamp(threshold, self.min_threshold, self.max_threshold)

        # Apply ternary quantization with STE
        spikes = TernaryQuantizer.apply(x, threshold, self.training)

        # Update running statistics
        if self.training:
            with torch.no_grad():
                density = (spikes != 0).float().mean()
                momentum = 0.99
                self.running_spike_density = (
                    momentum * self.running_spike_density + (1 - momentum) * density
                )
                self.running_threshold = (
                    momentum * self.running_threshold + (1 - momentum) * threshold
                )
                self.num_updates += 1

        return spikes

    def get_spike_density(self) -> float:
        """Return current running spike density."""
        return self.running_spike_density.item()

    def get_threshold(self) -> float:
        """Return current running threshold."""
        return self.running_threshold.item()

    def get_alpha(self) -> float:
        """Return current alpha value."""
        return self.alpha.item()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha.item():.3f}, learnable={isinstance(self.alpha, nn.Parameter)}"


class FixedThresholdTernarySpike(nn.Module):
    """
    Fixed-threshold ternary spiking (simpler baseline).

    Uses a constant threshold that doesn't adapt to input statistics.
    Useful as a comparison baseline and for understanding the
    importance of adaptive thresholds.

    Args:
        threshold: Fixed threshold value
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(threshold))
        self.register_buffer("running_spike_density", torch.tensor(0.0))
        self.register_buffer("num_updates", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fixed-threshold ternary quantization.

        Args:
            x: Pre-activation tensor (continuous)

        Returns:
            Ternary spikes in {-1, 0, +1}
        """
        spikes = TernaryQuantizer.apply(x, self.threshold, self.training)

        # Update running statistics
        if self.training:
            with torch.no_grad():
                density = (spikes != 0).float().mean()
                self.running_spike_density = (
                    0.99 * self.running_spike_density + 0.01 * density
                )
                self.num_updates += 1

        return spikes

    def get_spike_density(self) -> float:
        """Return current running spike density."""
        return self.running_spike_density.item()

    def extra_repr(self) -> str:
        return f"threshold={self.threshold.item():.3f}"


class SpikeActivityTracker(nn.Module):
    """
    Instrumentation layer for tracking spike statistics.

    Tracks:
    - Spike density: fraction of non-zero activations
    - Flicker rate: temporal variability (changes between timesteps)
    - Firing patterns: per-neuron activity for visualization

    Reference: Section 10.1 evaluation metrics.

    This module passes spikes through unchanged but records statistics
    for analysis and TTT triggering.
    """

    def __init__(self, name: str = "", max_history: int = 1000):
        super().__init__()
        self.name = name
        self.max_history = max_history
        self.reset_stats()

    def reset_stats(self):
        """Reset all tracked statistics."""
        self.densities: List[float] = []
        self.flicker_rates: List[float] = []
        self.firing_patterns: List[torch.Tensor] = []
        self.prev_spikes: Optional[torch.Tensor] = None
        self.step_count = 0

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Track statistics and pass through unchanged.

        Args:
            spikes: Ternary spike tensor

        Returns:
            Same tensor (pass-through)
        """
        with torch.no_grad():
            # Spike density: fraction of non-zero activations
            density = (spikes != 0).float().mean().item()
            self.densities.append(density)

            # Flicker rate: temporal variability
            if self.prev_spikes is not None and self.prev_spikes.shape == spikes.shape:
                flicker = (spikes != self.prev_spikes).float().mean().item()
                self.flicker_rates.append(flicker)

            # Store firing pattern for visualization (limit storage)
            if len(self.firing_patterns) < self.max_history:
                # Average across batch, keep per-neuron pattern
                if spikes.dim() >= 2:
                    pattern = (spikes != 0).float().mean(dim=0).cpu()
                else:
                    pattern = (spikes != 0).float().cpu()
                self.firing_patterns.append(pattern)

            self.prev_spikes = spikes.clone()
            self.step_count += 1

            # Limit history size
            if len(self.densities) > self.max_history:
                self.densities = self.densities[-self.max_history:]
            if len(self.flicker_rates) > self.max_history:
                self.flicker_rates = self.flicker_rates[-self.max_history:]

        return spikes

    def get_summary(self) -> Dict[str, float]:
        """Return summary statistics."""
        import numpy as np

        densities = np.array(self.densities) if self.densities else np.array([0.0])
        flickers = np.array(self.flicker_rates) if self.flicker_rates else np.array([0.0])

        return {
            "name": self.name,
            "mean_density": float(np.mean(densities)),
            "std_density": float(np.std(densities)),
            "min_density": float(np.min(densities)),
            "max_density": float(np.max(densities)),
            "mean_flicker": float(np.mean(flickers)),
            "std_flicker": float(np.std(flickers)),
            "step_count": self.step_count,
        }

    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """Get statistics from recent window."""
        import numpy as np

        recent_densities = self.densities[-window:] if self.densities else [0.0]
        recent_flickers = self.flicker_rates[-window:] if self.flicker_rates else [0.0]

        return {
            "mean_density": float(np.mean(recent_densities)),
            "mean_flicker": float(np.mean(recent_flickers)),
        }

    def get_firing_map(self) -> Optional[torch.Tensor]:
        """
        Get firing map for visualization.

        Returns:
            Tensor of shape (timesteps, neurons) with firing probabilities
        """
        if not self.firing_patterns:
            return None

        # Stack patterns into a single tensor
        try:
            return torch.stack(self.firing_patterns, dim=0)
        except RuntimeError:
            # Shapes might not match across timesteps
            return None

    def extra_repr(self) -> str:
        return f"name='{self.name}', history={len(self.densities)}"


class TernarySpikeLayer(nn.Module):
    """
    Complete ternary spike layer with optional tracking.

    Combines:
    - Adaptive or fixed threshold quantization
    - Optional activity tracking
    - Optional layer normalization before spiking

    This is a convenience wrapper for common use cases.
    """

    def __init__(
        self,
        d_model: int,
        adaptive: bool = True,
        alpha_init: float = 1.0,
        learnable_alpha: bool = True,
        threshold: float = 0.5,
        track_activity: bool = True,
        layer_name: str = "",
        use_layer_norm: bool = False,
    ):
        super().__init__()

        # Optional layer norm before spiking
        self.ln = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()

        # Spike layer
        if adaptive:
            self.spike = AdaptiveTernarySpike(
                alpha_init=alpha_init,
                learnable_alpha=learnable_alpha,
            )
        else:
            self.spike = FixedThresholdTernarySpike(threshold=threshold)

        # Activity tracking
        self.tracker = SpikeActivityTracker(name=layer_name) if track_activity else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization, spiking, and tracking.

        Args:
            x: Input tensor

        Returns:
            Ternary spikes
        """
        x = self.ln(x)
        spikes = self.spike(x)

        if self.tracker is not None:
            spikes = self.tracker(spikes)

        return spikes

    def get_spike_density(self) -> float:
        """Get current spike density."""
        return self.spike.get_spike_density()

    def get_stats(self) -> Dict[str, float]:
        """Get tracking statistics."""
        if self.tracker is not None:
            return self.tracker.get_summary()
        return {"density": self.get_spike_density()}

    def reset_stats(self):
        """Reset tracking statistics."""
        if self.tracker is not None:
            self.tracker.reset_stats()


def compute_spike_statistics(
    spikes: torch.Tensor,
    prev_spikes: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute spike statistics for a single tensor.

    Args:
        spikes: Ternary spike tensor
        prev_spikes: Previous timestep spikes (for flicker rate)

    Returns:
        Dictionary of statistics
    """
    with torch.no_grad():
        stats = {
            "density": (spikes != 0).float().mean().item(),
            "positive_fraction": (spikes == 1).float().mean().item(),
            "negative_fraction": (spikes == -1).float().mean().item(),
            "zero_fraction": (spikes == 0).float().mean().item(),
        }

        if prev_spikes is not None and prev_spikes.shape == spikes.shape:
            stats["flicker_rate"] = (spikes != prev_spikes).float().mean().item()

        return stats
