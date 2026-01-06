"""
Tests for Goose recurrent layers.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.goose_backbone import (
    DeltaRuleState,
    GooseRecurrentLayer,
    GooseBackbone,
)


class TestDeltaRuleState:
    """Test recurrent state container."""

    def test_init_zeros(self):
        """Should initialize with zeros."""
        state = DeltaRuleState.init_zeros(
            batch_size=4,
            d_model=64,
            device=torch.device("cpu"),
        )

        assert state.S.shape == (4, 64)
        assert (state.S == 0).all()

    def test_detach(self):
        """Should detach from computation graph."""
        state = DeltaRuleState(S=torch.randn(4, 64, requires_grad=True))
        detached = state.detach()

        assert not detached.S.requires_grad


class TestGooseRecurrentLayer:
    """Test single recurrent layer."""

    def test_forward_shape(self):
        """Output and state should have correct shapes."""
        layer = GooseRecurrentLayer(d_model=64, layer_idx=0, n_layers=4)
        batch_size = 4

        x = torch.randn(batch_size, 64)
        state = DeltaRuleState.init_zeros(batch_size, 64, x.device)

        output, new_state = layer(x, state)

        assert output.shape == (batch_size, 64)
        assert new_state.S.shape == (batch_size, 64)

    def test_state_updates(self):
        """State should change after forward pass."""
        layer = GooseRecurrentLayer(d_model=64)
        batch_size = 4

        x = torch.randn(batch_size, 64)
        state = DeltaRuleState.init_zeros(batch_size, 64, x.device)
        initial_S = state.S.clone()

        _, new_state = layer(x, state)

        # State should have changed
        assert not torch.allclose(initial_S, new_state.S)

    def test_sequence_processing(self):
        """Should process full sequence correctly."""
        layer = GooseRecurrentLayer(d_model=64)
        batch_size = 4
        seq_len = 16

        x = torch.randn(batch_size, seq_len, 64)
        outputs, final_state, hidden = layer.forward_sequence(x)

        assert outputs.shape == (batch_size, seq_len, 64)
        assert final_state.S.shape == (batch_size, 64)
        assert len(hidden) == seq_len


class TestGooseBackbone:
    """Test full backbone model."""

    def test_forward_shape(self):
        """Output logits should have correct shape."""
        model = GooseBackbone(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            max_seq_len=32,
        )

        input_ids = torch.randint(0, 100, (4, 16))
        logits, states, _ = model(input_ids)

        assert logits.shape == (4, 16, 100)
        assert len(states) == 2

    def test_tied_weights(self):
        """Embedding and head weights should be tied."""
        model = GooseBackbone(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            tie_weights=True,
        )

        assert model.head.weight is model.embedding.weight

    def test_generation(self):
        """Should generate tokens."""
        model = GooseBackbone(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            max_seq_len=32,
        )

        prompt = torch.randint(0, 100, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)

        assert generated.shape == (1, 15)  # 5 prompt + 10 new

    def test_parameter_count(self):
        """Should count parameters correctly."""
        model = GooseBackbone(
            d_model=64,
            n_layers=2,
            vocab_size=100,
        )

        count = model.count_parameters()
        assert count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
