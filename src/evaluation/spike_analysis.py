"""
Comprehensive spike activity analysis for ASNN-Goose.

Reference: Sections 7.2 and 10.1 of blueprint.

This module provides:
1. SpikeStatistics: Container for spike metrics
2. SpikeAnalyzer: Compute density, flicker, warp alignment
3. TTT trigger detection based on spike behavior
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class SpikeStatistics:
    """Container for spike statistics."""
    mean_density: float
    std_density: float
    mean_flicker: float
    std_flicker: float
    density_percentiles: Dict[str, float]
    sparsity_structure_score: float
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    zero_ratio: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_density": self.mean_density,
            "std_density": self.std_density,
            "mean_flicker": self.mean_flicker,
            "std_flicker": self.std_flicker,
            "density_p25": self.density_percentiles.get("p25", 0.0),
            "density_p50": self.density_percentiles.get("p50", 0.0),
            "density_p75": self.density_percentiles.get("p75", 0.0),
            "density_p95": self.density_percentiles.get("p95", 0.0),
            "sparsity_structure_score": self.sparsity_structure_score,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "zero_ratio": self.zero_ratio,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SpikeAnalyzer:
    """
    Analyze spike patterns for evaluation and TTT triggering.

    Computes:
    - Spike density: fraction of non-zero activations
    - Flicker rate: temporal variability
    - Warp alignment: how well zeros align with GPU warp boundaries
    - Block sparsity: pattern analysis for structured sparsity kernels

    Reference: Sections 7.2 and 10.1 of blueprint.
    """

    def __init__(self, warp_size: int = 32):
        """
        Args:
            warp_size: GPU warp size for structure analysis (32 for NVIDIA)
        """
        self.warp_size = warp_size

    def compute_density(self, spikes: torch.Tensor) -> float:
        """
        Compute spike density: fraction of non-zero activations.

        Args:
            spikes: Ternary spike tensor

        Returns:
            Density in [0, 1]
        """
        return (spikes != 0).float().mean().item()

    def compute_ternary_ratios(
        self, spikes: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute ratios of positive, negative, and zero spikes.

        Args:
            spikes: Ternary spike tensor

        Returns:
            (positive_ratio, negative_ratio, zero_ratio)
        """
        total = spikes.numel()
        positive = (spikes == 1).sum().item() / total
        negative = (spikes == -1).sum().item() / total
        zero = (spikes == 0).sum().item() / total
        return positive, negative, zero

    def compute_flicker_rate(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> float:
        """
        Compute temporal variability: fraction of changed spike states.

        Args:
            current: Current timestep spikes
            previous: Previous timestep spikes

        Returns:
            Flicker rate in [0, 1]
        """
        if current.shape != previous.shape:
            return 0.0
        return (current != previous).float().mean().item()

    def compute_warp_alignment_score(self, spikes: torch.Tensor) -> float:
        """
        Measure how well spike zeros align with warp boundaries.

        Higher score = better potential for GPU acceleration.
        Perfect alignment means all lanes in a warp have the same
        zero/nonzero state, enabling warp-level skip.

        Reference: Section 5.4 kernel realism.

        Args:
            spikes: Spike tensor (at least 1D)

        Returns:
            Alignment score in [0, 1]
        """
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)

        batch_size, dim = spikes.shape[:2]
        spikes_flat = spikes.view(batch_size, -1)
        dim = spikes_flat.shape[1]

        # Pad to multiple of warp_size
        if dim % self.warp_size != 0:
            pad_size = self.warp_size - (dim % self.warp_size)
            spikes_flat = torch.nn.functional.pad(spikes_flat, (0, pad_size))

        # Reshape to (batch, num_warps, warp_size)
        num_warps = spikes_flat.shape[1] // self.warp_size
        warps = spikes_flat.view(batch_size, num_warps, self.warp_size)

        # For each warp, check if all lanes are zero or all are nonzero
        warp_is_zero = (warps == 0).all(dim=-1)
        warp_is_nonzero = (warps != 0).all(dim=-1)

        # Aligned warps are those where all lanes agree
        aligned_warps = (warp_is_zero | warp_is_nonzero).float().mean()

        return aligned_warps.item()

    def compute_block_sparsity(
        self,
        spikes: torch.Tensor,
        block_size: int = 16,
    ) -> Dict[str, float]:
        """
        Analyze block-level sparsity patterns.

        Returns fractions of blocks that are fully zero, fully nonzero, or mixed.
        Only fully-zero blocks can be skipped in kernel implementations.

        Args:
            spikes: Spike tensor
            block_size: Block size for analysis

        Returns:
            Dictionary with block sparsity statistics
        """
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)

        spikes_flat = spikes.view(spikes.shape[0], -1)
        batch_size, dim = spikes_flat.shape

        # Pad to multiple of block_size
        if dim % block_size != 0:
            pad_size = block_size - (dim % block_size)
            spikes_flat = torch.nn.functional.pad(spikes_flat, (0, pad_size))

        num_blocks = spikes_flat.shape[1] // block_size
        blocks = spikes_flat.view(batch_size, num_blocks, block_size)

        # Classify blocks
        block_sums = (blocks != 0).sum(dim=-1)

        fully_zero = (block_sums == 0).float().mean().item()
        fully_dense = (block_sums == block_size).float().mean().item()
        mixed = 1.0 - fully_zero - fully_dense

        return {
            "fully_zero": fully_zero,
            "fully_dense": fully_dense,
            "mixed": mixed,
            "effective_sparsity": fully_zero,  # Only fully-zero blocks skip work
        }

    def analyze_layer_spikes(
        self,
        spikes_over_time: List[torch.Tensor],
    ) -> SpikeStatistics:
        """
        Comprehensive analysis of spike patterns over time.

        Args:
            spikes_over_time: List of spike tensors across timesteps

        Returns:
            SpikeStatistics with all computed metrics
        """
        densities = []
        flickers = []
        alignment_scores = []
        all_positive = []
        all_negative = []
        all_zero = []

        prev_spikes = None
        for spikes in spikes_over_time:
            densities.append(self.compute_density(spikes))
            alignment_scores.append(self.compute_warp_alignment_score(spikes))

            pos, neg, zero = self.compute_ternary_ratios(spikes)
            all_positive.append(pos)
            all_negative.append(neg)
            all_zero.append(zero)

            if prev_spikes is not None:
                flickers.append(self.compute_flicker_rate(spikes, prev_spikes))
            prev_spikes = spikes

        densities = np.array(densities)
        flickers = np.array(flickers) if flickers else np.array([0.0])

        return SpikeStatistics(
            mean_density=float(np.mean(densities)),
            std_density=float(np.std(densities)),
            mean_flicker=float(np.mean(flickers)),
            std_flicker=float(np.std(flickers)),
            density_percentiles={
                "p25": float(np.percentile(densities, 25)),
                "p50": float(np.percentile(densities, 50)),
                "p75": float(np.percentile(densities, 75)),
                "p95": float(np.percentile(densities, 95)),
            },
            sparsity_structure_score=float(np.mean(alignment_scores)),
            positive_ratio=float(np.mean(all_positive)),
            negative_ratio=float(np.mean(all_negative)),
            zero_ratio=float(np.mean(all_zero)),
        )

    def check_ttt_trigger(
        self,
        recent_densities: List[float],
        recent_flickers: List[float],
        density_trigger: float = 0.8,
        flicker_trigger: float = 0.3,
    ) -> Tuple[bool, str]:
        """
        Check if TTT should be triggered based on spike behavior.

        Args:
            recent_densities: Recent spike density values
            recent_flickers: Recent flicker rate values
            density_trigger: Threshold for density trigger
            flicker_trigger: Threshold for flicker trigger

        Returns:
            (should_trigger, reason)
        """
        if not recent_densities:
            return False, "no_data"

        mean_density = np.mean(recent_densities)
        mean_flicker = np.mean(recent_flickers) if recent_flickers else 0.0

        # Check density trigger
        if mean_density > density_trigger:
            return True, f"high_density:{mean_density:.3f}"

        # Check flicker trigger
        if mean_flicker > flicker_trigger:
            return True, f"high_flicker:{mean_flicker:.3f}"

        return False, "normal"

    # =========================================================================
    # V15: SpikingBrain Health Metrics
    # =========================================================================

    def compute_per_channel_firing_rates(
        self,
        spike_tensors: List[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute per-channel firing rates across all samples.

        Args:
            spike_tensors: List of spike tensors, each (batch, seq, d_model)

        Returns:
            Array of shape (d_model,) with firing rates per channel in [0, 1]
        """
        if not spike_tensors:
            return np.array([])

        # Stack all spikes: flatten batch and seq dims
        all_spikes = torch.cat(
            [s.view(-1, s.shape[-1]) for s in spike_tensors], dim=0
        )
        # Firing rate = fraction of non-zero per channel
        rates = (all_spikes != 0).float().mean(dim=0).cpu().numpy()
        return rates

    def detect_dead_neurons(
        self,
        firing_rates: np.ndarray,
        threshold: float = 0.001,
    ) -> Tuple[float, np.ndarray]:
        """
        Detect neurons that never fire (always emit 0).

        Dead neurons indicate information loss in the spike encoding.
        Alert if dead neuron percentage exceeds 5%.

        Args:
            firing_rates: Per-channel firing rates from compute_per_channel_firing_rates
            threshold: Firing rate below which neuron is considered dead

        Returns:
            (dead_neuron_pct, dead_neuron_indices)
        """
        if len(firing_rates) == 0:
            return 0.0, np.array([])

        dead_mask = firing_rates < threshold
        dead_pct = float(dead_mask.mean())
        dead_indices = np.where(dead_mask)[0]
        return dead_pct, dead_indices

    def detect_saturated_neurons(
        self,
        spike_tensors: List[torch.Tensor],
    ) -> Tuple[float, np.ndarray]:
        """
        Detect neurons that always fire (never emit 0).

        Saturated neurons indicate threshold miscalibration.
        Alert if saturated neuron percentage exceeds 10%.

        Args:
            spike_tensors: List of spike tensors, each (batch, seq, d_model)

        Returns:
            (saturated_neuron_pct, saturated_neuron_indices)
        """
        if not spike_tensors:
            return 0.0, np.array([])

        # Stack all spikes
        all_spikes = torch.cat(
            [s.view(-1, s.shape[-1]) for s in spike_tensors], dim=0
        )
        # Saturated = never zero across all samples
        always_active = (all_spikes != 0).all(dim=0).cpu().numpy()
        saturated_pct = float(always_active.mean())
        saturated_indices = np.where(always_active)[0]
        return saturated_pct, saturated_indices

    @torch.no_grad()
    def analyze_model_spikes(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, SpikeStatistics]:
        """
        Analyze spike patterns from a model forward pass.

        Args:
            model: ASNNGoose model
            input_ids: Input token IDs
            device: Device to run on

        Returns:
            Dictionary mapping layer names to SpikeStatistics
        """
        model.eval()
        input_ids = input_ids.to(device)

        # Forward pass collecting spike info
        _, _, aux = model(input_ids, return_spike_info=True)

        results = {}
        spike_info = aux.get("spike_info", {})

        for layer_idx, layer_spikes in spike_info.items():
            # layer_spikes is a list of dicts with k_spikes and v_spikes
            k_spikes = [s["k_spikes"] for s in layer_spikes]
            v_spikes = [s["v_spikes"] for s in layer_spikes]

            results[f"layer_{layer_idx}_k"] = self.analyze_layer_spikes(k_spikes)
            results[f"layer_{layer_idx}_v"] = self.analyze_layer_spikes(v_spikes)

        return results


def compute_firing_map(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    Compute firing map for visualization.

    Returns a 2D tensor showing spike activity across neurons and time.

    Args:
        model: ASNNGoose model
        input_ids: Input token IDs
        device: Device to run on
        layer_idx: Which layer to analyze

    Returns:
        Tensor of shape (seq_len, d_model) with firing probabilities
    """
    model.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        _, _, aux = model(input_ids, return_spike_info=True)

    spike_info = aux.get("spike_info", {})
    if layer_idx not in spike_info:
        return torch.zeros(input_ids.shape[1], model.d_model)

    layer_spikes = spike_info[layer_idx]
    k_spikes = [s["k_spikes"] for s in layer_spikes]

    # Stack and compute mean activity
    firing_map = torch.stack(k_spikes, dim=1)  # (batch, seq, d_model)
    firing_map = (firing_map != 0).float().mean(dim=0)  # (seq, d_model)

    return firing_map
