"""
TTT Stability Analysis for ASNN-Goose.

Reference: Section 10.3 of blueprint.

This module provides:
1. TTTStabilityAnalyzer: Analyze drift and stability during TTT
2. Failure mode detection
3. Long-term adaptation analysis
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StabilityMetrics:
    """Container for stability analysis metrics."""
    drift_magnitude: float
    validation_variance: float
    reversion_rate: float
    improvement_rate: float
    mean_lora_norm: float
    max_lora_norm: float
    is_stable: bool
    failure_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_magnitude": self.drift_magnitude,
            "validation_variance": self.validation_variance,
            "reversion_rate": self.reversion_rate,
            "improvement_rate": self.improvement_rate,
            "mean_lora_norm": self.mean_lora_norm,
            "max_lora_norm": self.max_lora_norm,
            "is_stable": self.is_stable,
            "failure_mode": self.failure_mode,
        }


class TTTStabilityAnalyzer:
    """
    Analyze TTT stability and detect failure modes.

    Tracks:
    - Validation score trajectories
    - LoRA norm evolution
    - Update/reversion patterns
    - Drift detection

    Reference: Section 10.3 failure modes.
    """

    def __init__(
        self,
        reversion_threshold: float = 0.5,
        drift_threshold: float = 0.1,
        variance_threshold: float = 0.05,
    ):
        """
        Args:
            reversion_threshold: If reversion rate > this, flag as unstable
            drift_threshold: If drift magnitude > this, flag as drifting
            variance_threshold: If validation variance > this, flag as oscillating
        """
        self.reversion_threshold = reversion_threshold
        self.drift_threshold = drift_threshold
        self.variance_threshold = variance_threshold

        # Tracking data
        self.validation_scores: List[float] = []
        self.lora_norms: List[Dict[str, float]] = []
        self.update_points: List[int] = []
        self.reversion_points: List[int] = []
        self.improvements: List[float] = []

    def record_update(
        self,
        validation_score: float,
        lora_norms: Dict[str, float],
        improvement: float,
        reverted: bool,
        step: int,
    ):
        """
        Record a TTT update event.

        Args:
            validation_score: Validation score after update
            lora_norms: Current LoRA delta norms
            improvement: Score improvement from this update
            reverted: Whether the update was reverted
            step: Current step number
        """
        self.validation_scores.append(validation_score)
        self.lora_norms.append(lora_norms.copy())
        self.improvements.append(improvement)

        if reverted:
            self.reversion_points.append(step)
        else:
            self.update_points.append(step)

    def compute_drift_magnitude(self) -> float:
        """
        Compute overall drift magnitude.

        Drift is measured as the change in mean validation score
        from beginning to end of adaptation.
        """
        if len(self.validation_scores) < 10:
            return 0.0

        # Compare first and last 10% of scores
        n = len(self.validation_scores)
        window = max(1, n // 10)

        start_mean = np.mean(self.validation_scores[:window])
        end_mean = np.mean(self.validation_scores[-window:])

        return abs(end_mean - start_mean)

    def compute_validation_variance(self) -> float:
        """
        Compute validation score variance.

        High variance indicates oscillating/unstable adaptation.
        """
        if len(self.validation_scores) < 2:
            return 0.0
        return float(np.std(self.validation_scores))

    def compute_reversion_rate(self) -> float:
        """
        Compute fraction of updates that were reverted.
        """
        total = len(self.update_points) + len(self.reversion_points)
        if total == 0:
            return 0.0
        return len(self.reversion_points) / total

    def compute_improvement_rate(self) -> float:
        """
        Compute fraction of updates that improved performance.
        """
        if not self.improvements:
            return 0.0
        positive = sum(1 for i in self.improvements if i > 0)
        return positive / len(self.improvements)

    def get_lora_norm_stats(self) -> Tuple[float, float]:
        """
        Get mean and max LoRA norms across all updates.
        """
        if not self.lora_norms:
            return 0.0, 0.0

        all_norms = []
        for norms_dict in self.lora_norms:
            all_norms.extend(norms_dict.values())

        if not all_norms:
            return 0.0, 0.0

        return float(np.mean(all_norms)), float(np.max(all_norms))

    def detect_failure_mode(self) -> Optional[str]:
        """
        Detect if a failure mode is occurring.

        Failure modes (from Section 10.3):
        - "spike_densification": Too many spikes (caught by TTT trigger)
        - "reasoning_degradation": Validation score dropping
        - "ttt_drift": Updates not helping, high reversion
        - "oscillation": Validation variance too high
        """
        # Check for TTT drift (high reversion rate)
        if self.compute_reversion_rate() > self.reversion_threshold:
            return "ttt_drift"

        # Check for oscillation
        if self.compute_validation_variance() > self.variance_threshold:
            return "oscillation"

        # Check for drift (score degradation)
        if len(self.validation_scores) >= 10:
            n = len(self.validation_scores)
            window = max(1, n // 10)
            start_mean = np.mean(self.validation_scores[:window])
            end_mean = np.mean(self.validation_scores[-window:])

            if end_mean < start_mean - self.drift_threshold:
                return "reasoning_degradation"

        return None

    def analyze(self) -> StabilityMetrics:
        """
        Perform full stability analysis.

        Returns:
            StabilityMetrics with all computed values
        """
        drift = self.compute_drift_magnitude()
        variance = self.compute_validation_variance()
        reversion_rate = self.compute_reversion_rate()
        improvement_rate = self.compute_improvement_rate()
        mean_norm, max_norm = self.get_lora_norm_stats()
        failure_mode = self.detect_failure_mode()

        is_stable = (
            failure_mode is None
            and reversion_rate < self.reversion_threshold
            and drift < self.drift_threshold
        )

        return StabilityMetrics(
            drift_magnitude=drift,
            validation_variance=variance,
            reversion_rate=reversion_rate,
            improvement_rate=improvement_rate,
            mean_lora_norm=mean_norm,
            max_lora_norm=max_norm,
            is_stable=is_stable,
            failure_mode=failure_mode,
        )

    def get_trajectory_data(self) -> Dict[str, List[Any]]:
        """
        Get trajectory data for visualization.
        """
        return {
            "validation_scores": self.validation_scores.copy(),
            "update_points": self.update_points.copy(),
            "reversion_points": self.reversion_points.copy(),
            "improvements": self.improvements.copy(),
        }

    def reset(self):
        """Reset analyzer state."""
        self.validation_scores.clear()
        self.lora_norms.clear()
        self.update_points.clear()
        self.reversion_points.clear()
        self.improvements.clear()


def run_stability_test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ttt_controller: Any,
    device: torch.device,
    num_steps: int = 1000,
) -> Tuple[StabilityMetrics, Dict[str, List[Any]]]:
    """
    Run a stability test with TTT enabled.

    Args:
        model: ASNNGoose model with LoRA
        dataloader: Data loader for adaptation
        ttt_controller: TTTController instance
        device: Device to use
        num_steps: Number of steps to run

    Returns:
        Tuple of (StabilityMetrics, trajectory_data)
    """
    analyzer = TTTStabilityAnalyzer()
    model.eval()

    step = 0
    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)

        # Get spike stats
        with torch.no_grad():
            _, _, aux = model(input_ids, return_spike_info=True)

        # Compute mean density from spike info
        spike_info = aux.get("spike_info", {})
        densities = []
        for layer_spikes in spike_info.values():
            for s in layer_spikes:
                k_density = (s["k_spikes"] != 0).float().mean().item()
                densities.append(k_density)
        mean_density = np.mean(densities) if densities else 0.5

        # TTT step
        result = ttt_controller.step(input_ids, density=mean_density)

        # Record if update occurred
        if result.get("updated", False) or result.get("reverted", False):
            lora_norms = ttt_controller.get_lora_norms()
            analyzer.record_update(
                validation_score=result.get("validation_delta", 0.0),
                lora_norms=lora_norms,
                improvement=result.get("validation_delta", 0.0),
                reverted=result.get("reverted", False),
                step=step,
            )

        step += 1

    metrics = analyzer.analyze()
    trajectory = analyzer.get_trajectory_data()

    return metrics, trajectory
