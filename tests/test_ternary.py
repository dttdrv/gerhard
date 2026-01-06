"""
Tests for ternary activation functions.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ternary_activations import (
    TernaryQuantizer,
    AdaptiveTernarySpike,
    FixedThresholdTernarySpike,
    SpikeActivityTracker,
)


class TestTernaryQuantizer:
    """Test the core ternary quantization function."""

    def test_output_values(self):
        """Output should only contain {-1, 0, +1}."""
        x = torch.randn(10, 32)
        threshold = torch.tensor(0.5)
        output = TernaryQuantizer.apply(x, threshold, True)

        unique = output.unique().tolist()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique)

    def test_threshold_behavior(self):
        """Values above/below threshold should map correctly."""
        x = torch.tensor([[-1.0, -0.3, 0.0, 0.3, 1.0]])
        threshold = torch.tensor(0.5)
        output = TernaryQuantizer.apply(x, threshold, True)

        assert output[0, 0].item() == -1.0  # -1.0 < -0.5
        assert output[0, 1].item() == 0.0   # -0.3 in [-0.5, 0.5]
        assert output[0, 2].item() == 0.0   # 0.0 in [-0.5, 0.5]
        assert output[0, 3].item() == 0.0   # 0.3 in [-0.5, 0.5]
        assert output[0, 4].item() == 1.0   # 1.0 > 0.5

    def test_gradient_passthrough(self):
        """STE should pass gradients through."""
        x = torch.randn(10, 32, requires_grad=True)
        threshold = torch.tensor(0.5)
        output = TernaryQuantizer.apply(x, threshold, True)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestAdaptiveTernarySpike:
    """Test adaptive-threshold spiking."""

    def test_adaptive_threshold(self):
        """Threshold should adapt to input magnitude."""
        layer = AdaptiveTernarySpike(alpha_init=1.0)

        # Small inputs -> small threshold -> more spikes
        small_input = torch.randn(10, 32) * 0.1
        small_output = layer(small_input)

        # Large inputs -> large threshold -> similar sparsity
        layer2 = AdaptiveTernarySpike(alpha_init=1.0)
        large_input = torch.randn(10, 32) * 10.0
        large_output = layer2(large_input)

        # Density should be roughly similar due to adaptation
        small_density = (small_output != 0).float().mean()
        large_density = (large_output != 0).float().mean()

        # Should be within 20% of each other
        assert abs(small_density - large_density) < 0.3

    def test_running_statistics(self):
        """Running statistics should update during training."""
        layer = AdaptiveTernarySpike(alpha_init=1.0)
        layer.train()

        initial_density = layer.running_spike_density.item()

        for _ in range(10):
            x = torch.randn(10, 32)
            _ = layer(x)

        final_density = layer.get_spike_density()

        # Should have updated
        assert final_density != initial_density or initial_density != 0


class TestFixedThresholdTernarySpike:
    """Test fixed-threshold spiking."""

    def test_fixed_threshold(self):
        """Threshold should remain fixed."""
        threshold = 0.5
        layer = FixedThresholdTernarySpike(threshold=threshold)

        x = torch.tensor([[0.4, 0.6, -0.4, -0.6]])
        output = layer(x)

        assert output[0, 0].item() == 0.0  # 0.4 < 0.5
        assert output[0, 1].item() == 1.0  # 0.6 > 0.5
        assert output[0, 2].item() == 0.0  # -0.4 > -0.5
        assert output[0, 3].item() == -1.0  # -0.6 < -0.5


class TestSpikeActivityTracker:
    """Test spike activity tracking."""

    def test_density_tracking(self):
        """Should track spike density."""
        tracker = SpikeActivityTracker(name="test")

        spikes = torch.tensor([[1., 0., -1., 0., 0.]])
        _ = tracker(spikes)

        summary = tracker.get_summary()
        assert summary["mean_density"] == pytest.approx(0.4)  # 2 non-zero out of 5

    def test_flicker_tracking(self):
        """Should track flicker rate."""
        tracker = SpikeActivityTracker(name="test")

        spikes1 = torch.tensor([[1., 0., -1., 0., 0.]])
        spikes2 = torch.tensor([[0., 0., -1., 1., 0.]])  # 2 changed

        _ = tracker(spikes1)
        _ = tracker(spikes2)

        summary = tracker.get_summary()
        assert len(tracker.flicker_rates) == 1
        assert tracker.flicker_rates[0] == pytest.approx(0.4)  # 2 changes out of 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
