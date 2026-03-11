"""
Tests for SpikingBrain validator compatibility and metric aggregation.
"""
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.spiking_brain import SpikingBrainValidator
from src.models.teacher_model import TeacherModel


class FakeSpikeStudent(nn.Module):
    """Minimal student stub that emits deterministic spike traces."""

    def __init__(self, spike_info, vocab_size: int = 32):
        super().__init__()
        self._spike_info = spike_info
        self._vocab_size = vocab_size

    def forward(self, input_ids, return_spike_info: bool = False):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self._vocab_size)
        aux = {"spike_info": self._spike_info if return_spike_info else {}}
        return logits, [], aux


class HFStyleTeacher(nn.Module):
    """Teacher stub with Hugging Face hidden_states output."""

    def __init__(self, hidden_size: int, n_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, input_ids, output_hidden_states: bool = False):
        assert output_hidden_states is True
        batch_size, seq_len = input_ids.shape
        hidden_states = [torch.zeros(batch_size, seq_len, self.hidden_size)]
        for layer_idx in range(self.n_layers):
            hidden_states.append(
                torch.full(
                    (batch_size, seq_len, self.hidden_size),
                    float(layer_idx + 1),
                )
            )
        return SimpleNamespace(hidden_states=tuple(hidden_states))


def _make_seq_spike_list(batch_size: int, seq_len: int, d_model: int, base_value: float):
    return [
        {
            "k_spikes": torch.full((batch_size, d_model), base_value),
            "v_spikes": torch.full((batch_size, d_model), base_value + 1.0),
        }
        for _ in range(seq_len)
    ]


def _make_single_spike_dict(batch_size: int, d_model: int, base_value: float):
    return {
        "k_spikes": torch.full((batch_size, d_model), base_value),
        "v_spikes": torch.full((batch_size, d_model), base_value + 1.0),
    }


def test_collect_representations_supports_repo_teacher_model():
    """Validator should collect mapped layer hiddens from repo TeacherModel outputs."""
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 4
    d_model = 8
    layer_map = {0: 0, 1: 1, 2: 2}

    student = FakeSpikeStudent(
        {
            0: _make_seq_spike_list(batch_size, seq_len, d_model, 1.0),
            1: _make_seq_spike_list(batch_size, seq_len, d_model, 2.0),
            2: _make_seq_spike_list(batch_size, seq_len, d_model, 3.0),
        }
    )
    teacher = TeacherModel(
        d_model=d_model,
        n_layers=3,
        vocab_size=32,
        max_seq_len=8,
    )
    validator = SpikingBrainValidator(device=device, layer_map=layer_map)
    dataloader = [{"input_ids": torch.randint(0, 32, (batch_size, seq_len))}]

    spike_info, teacher_hiddens = validator.collect_representations(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        max_batches=1,
    )

    assert sorted(spike_info.keys()) == [0, 1, 2]
    assert sorted(teacher_hiddens.keys()) == [0, 1, 2]
    for layer_idx, hidden in teacher_hiddens.items():
        assert hidden.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(hidden).all(), f"non-finite hidden state at layer {layer_idx}"


def test_collect_representations_accepts_single_dict_spike_payload():
    """Collector should normalize single-dict spike payloads to the list collector format."""
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 3
    d_model = 5
    layer_map = {0: 0}

    student = FakeSpikeStudent({0: _make_single_spike_dict(batch_size, d_model, 7.0)})
    teacher = HFStyleTeacher(hidden_size=d_model, n_layers=1)
    validator = SpikingBrainValidator(device=device, layer_map=layer_map)
    dataloader = [{"input_ids": torch.randint(0, 32, (batch_size, seq_len))}]

    spike_info, teacher_hiddens = validator.collect_representations(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        max_batches=1,
    )

    assert len(spike_info[0]) == 1
    assert torch.allclose(spike_info[0][0]["k_spikes"], torch.full((batch_size, d_model), 7.0))
    assert teacher_hiddens[0].shape == (batch_size, seq_len, d_model)


def test_collect_representations_keeps_hf_hidden_state_path():
    """Validator should still support Hugging Face style hidden_states outputs."""
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 3
    d_model = 6
    layer_map = {0: 0, 1: 1}

    student = FakeSpikeStudent(
        {
            0: _make_seq_spike_list(batch_size, seq_len, d_model, 1.0),
            1: _make_seq_spike_list(batch_size, seq_len, d_model, 2.0),
        }
    )
    teacher = HFStyleTeacher(hidden_size=d_model, n_layers=2)
    validator = SpikingBrainValidator(device=device, layer_map=layer_map)
    dataloader = [(torch.randint(0, 32, (batch_size, seq_len)),)]

    _, teacher_hiddens = validator.collect_representations(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        max_batches=1,
    )

    assert torch.allclose(teacher_hiddens[0], torch.ones(batch_size, seq_len, d_model))
    assert torch.allclose(teacher_hiddens[1], torch.full((batch_size, seq_len, d_model), 2.0))


def test_validate_aggregates_metrics_across_layers_and_kv(monkeypatch):
    """Notebook-parity validation must aggregate K/V metrics across every mapped layer."""
    device = torch.device("cpu")
    batch_size = 1
    seq_len = 2
    d_model = 4
    layer_map = {0: 0, 2: 1}

    student = FakeSpikeStudent(
        {
            0: _make_seq_spike_list(batch_size, seq_len, d_model, 1.0),
            2: _make_seq_spike_list(batch_size, seq_len, d_model, 3.0),
        }
    )
    teacher = HFStyleTeacher(hidden_size=d_model, n_layers=2)
    validator = SpikingBrainValidator(
        device=device,
        layer_map=layer_map,
        dead_threshold=1.1,
        saturated_threshold=1.1,
        firing_rate_range=(0.0, 1.1),
        mi_threshold=0.0,
        cka_threshold=0.0,
    )
    dataloader = [{"input_ids": torch.randint(0, 32, (batch_size, seq_len))}]

    def fake_estimate_mi(spikes, _teacher_hidden):
        base = float(spikes.reshape(-1)[0].item())
        return {
            "mutual_information": base / 10.0,
            "method": "binning_sign_discretization",
        }

    def fake_linear_cka(spikes, _teacher_hidden, eps=1e-8):
        del eps
        return float(spikes.reshape(-1)[0].item() / 20.0)

    monkeypatch.setattr(validator.mi_estimator, "estimate_mi", fake_estimate_mi)
    monkeypatch.setattr(validator.rep_analyzer, "linear_cka", fake_linear_cka)

    results = validator.validate(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        max_batches=1,
    )

    assert results.mutual_information["method"] == "binning_sign_discretization"
    assert results.mutual_information["layer_0_to_0_k"] == pytest.approx(0.1)
    assert results.mutual_information["layer_0_to_0_v"] == pytest.approx(0.2)
    assert results.mutual_information["layer_0_to_0"] == pytest.approx(0.15)
    assert results.mutual_information["layer_2_to_1_k"] == pytest.approx(0.3)
    assert results.mutual_information["layer_2_to_1_v"] == pytest.approx(0.4)
    assert results.mutual_information["layer_2_to_1"] == pytest.approx(0.35)
    assert results.mutual_information["mutual_information"] == pytest.approx(0.25)

    assert results.cka["method"] == "linear_cka"
    assert results.cka["layer_0_to_0_k"] == pytest.approx(0.05)
    assert results.cka["layer_0_to_0_v"] == pytest.approx(0.10)
    assert results.cka["layer_0_to_0"] == pytest.approx(0.075)
    assert results.cka["layer_2_to_1_k"] == pytest.approx(0.15)
    assert results.cka["layer_2_to_1_v"] == pytest.approx(0.20)
    assert results.cka["layer_2_to_1"] == pytest.approx(0.175)
    assert results.cka["cka_mean"] == pytest.approx(0.125)
