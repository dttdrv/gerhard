"""
Tests for LoRA adapters.
"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lora_adapter import (
    LoRAAdapter,
    LoRALinear,
    apply_lora_to_model,
    get_lora_parameters,
)


class TestLoRAAdapter:
    """Test LoRA adapter module."""

    def test_output_shape(self):
        """Output should match expected shape."""
        adapter = LoRAAdapter(
            in_features=64,
            out_features=64,
            rank=8,
            alpha=16.0,
        )

        x = torch.randn(4, 16, 64)
        output = adapter(x)

        assert output.shape == x.shape

    def test_initial_output_zero(self):
        """Initial output should be near zero (B initialized to zero)."""
        adapter = LoRAAdapter(
            in_features=64,
            out_features=64,
            rank=8,
        )

        x = torch.randn(4, 64)
        output = adapter(x)

        # Should be all zeros initially
        assert torch.allclose(output, torch.zeros_like(output))

    def test_delta_weight(self):
        """Should compute effective weight delta."""
        adapter = LoRAAdapter(
            in_features=32,
            out_features=64,
            rank=4,
        )

        delta = adapter.get_delta_weight()
        assert delta.shape == (64, 32)

    def test_delta_norm(self):
        """Should compute delta norm."""
        adapter = LoRAAdapter(
            in_features=32,
            out_features=64,
            rank=4,
        )

        norm = adapter.get_delta_norm()
        assert norm >= 0  # Initially should be 0


class TestLoRALinear:
    """Test LoRA-enabled linear layer."""

    def test_forward(self):
        """Forward pass should work correctly."""
        layer = LoRALinear(
            in_features=64,
            out_features=32,
            rank=4,
        )

        x = torch.randn(4, 64)
        output = layer(x)

        assert output.shape == (4, 32)

    def test_freeze_base(self):
        """Base weights should be frozen."""
        layer = LoRALinear(
            in_features=64,
            out_features=32,
            freeze_base=True,
        )

        assert not layer.linear.weight.requires_grad
        assert layer.lora.lora_A.requires_grad
        assert layer.lora.lora_B.requires_grad

    def test_merge_and_unload(self):
        """Should merge LoRA into base weights."""
        layer = LoRALinear(
            in_features=64,
            out_features=32,
            rank=4,
        )

        # Train LoRA a bit
        layer.lora.lora_B.data.fill_(0.1)

        merged = layer.merge_and_unload()

        assert isinstance(merged, nn.Linear)
        assert merged.weight.shape == (32, 64)

    def test_get_lora_parameters(self):
        """Should return LoRA parameters."""
        layer = LoRALinear(
            in_features=64,
            out_features=32,
            rank=4,
        )

        params = layer.get_lora_parameters()
        assert len(params) == 2  # lora_A and lora_B


class TestApplyLoRA:
    """Test applying LoRA to models."""

    def test_apply_to_model(self):
        """Should apply LoRA to specified modules."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        model, lora_modules = apply_lora_to_model(
            model,
            rank=4,
            target_modules=None,  # Apply to all
        )

        assert len(lora_modules) == 2  # Two linear layers

    def test_target_modules(self):
        """Should only apply to targeted modules."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.key_proj = nn.Linear(64, 64)
                self.value_proj = nn.Linear(64, 64)
                self.other_proj = nn.Linear(64, 64)

        model = SimpleModel()
        model, lora_modules = apply_lora_to_model(
            model,
            rank=4,
            target_modules={"key_proj", "value_proj"},
        )

        # Should only have LoRA on key and value
        assert "key_proj" in lora_modules
        assert "value_proj" in lora_modules
        assert "other_proj" not in lora_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
