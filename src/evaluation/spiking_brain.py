"""
V15: SpikingBrain - Information Encoding Validation for ASNN-Goose.

Validates that spike patterns {-1, 0, +1} encode meaningful semantic
information, NOT arbitrary quantization artifacts.

This is a prerequisite for v16 (sparse ops) - no point optimizing
garbage spikes.

Components:
1. SpikeHealthMetrics: Dead/saturated neurons, firing rates
2. MutualInformationEstimator: Binning-based MI between spikes and teacher
3. RepresentationAnalyzer: CKA similarity between spike and teacher representations
4. SpikingBrainValidator: Main orchestrator for full validation suite
"""
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .spike_analysis import SpikeAnalyzer


@dataclass
class SpikeHealthMetrics:
    """Container for spike health validation results."""

    dead_neuron_pct: float
    saturated_neuron_pct: float
    firing_rate_mean: float
    firing_rate_std: float
    per_channel_rates: Dict[str, np.ndarray] = field(default_factory=dict)
    health_pass: bool = True
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dead_neuron_pct": self.dead_neuron_pct,
            "saturated_neuron_pct": self.saturated_neuron_pct,
            "firing_rate_mean": self.firing_rate_mean,
            "firing_rate_std": self.firing_rate_std,
            "health_pass": self.health_pass,
            "alerts": self.alerts,
        }


@dataclass
class SpikingBrainValidation:
    """Container for all SpikingBrain validation results."""

    health: SpikeHealthMetrics
    mutual_information: Dict[str, float]
    cka: Dict[str, float]
    overall_pass: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health": self.health.to_dict(),
            "mutual_information": self.mutual_information,
            "cka": self.cka,
            "overall_pass": self.overall_pass,
            "summary": self.summary,
        }


class MutualInformationEstimator:
    """
    Estimate mutual information between spikes and teacher hidden states.

    Uses binning-based estimation for speed (~2s per validation).
    MI measures how much information about teacher representations
    is preserved in the ternary spike patterns.

    Expected ranges:
    - MI < 0.1: Poor encoding (spikes are noise)
    - MI 0.1-0.5: Moderate encoding (some information preserved)
    - MI > 0.5: Good encoding (strong correlation)
    """

    def __init__(self, n_bins: int = 32, n_dims: int = 8):
        """
        Args:
            n_bins: Number of bins for discretizing teacher activations
            n_dims: Number of dimensions to analyze (for speed)
        """
        self.n_bins = n_bins
        self.n_dims = n_dims

    def binning_mi(
        self,
        spikes: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> float:
        """
        Fast binning-based MI estimation.

        Discretizes both distributions and computes:
        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Args:
            spikes: Ternary spike tensor (batch, seq, d_spike)
            teacher_hidden: Teacher hidden states (batch, seq, d_teacher)

        Returns:
            Estimated mutual information in bits.
        """
        with torch.no_grad():
            s_flat = spikes.reshape(-1, spikes.shape[-1]).float().cpu().numpy()
            t_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1]).float().cpu().numpy()

            n = min(s_flat.shape[0], t_flat.shape[0], 10000)
            if n == 0:
                return 0.0

            s_flat = s_flat[:n]
            t_flat = t_flat[:n]
            n_dims = min(self.n_dims, s_flat.shape[1], t_flat.shape[1])
            if n_dims == 0:
                return 0.0

            mi_values = []
            for dim in range(n_dims):
                s_col = s_flat[:, dim]
                t_col = t_flat[:, dim]

                t_min = float(t_col.min())
                t_max = float(t_col.max())
                if abs(t_max - t_min) < 1e-12:
                    continue

                t_bins = np.digitize(
                    t_col,
                    np.linspace(t_min, t_max, self.n_bins + 1)[1:-1],
                )

                # Notebook parity: collapse ternary values to sign buckets.
                s_disc = np.ones_like(s_col, dtype=np.int32)
                s_disc[s_col > 1e-6] = 2
                s_disc[s_col < -1e-6] = 0

                joint = np.zeros((3, self.n_bins), dtype=np.float64)
                for idx in range(n):
                    t_bin = max(0, min(int(t_bins[idx]), self.n_bins - 1))
                    joint[s_disc[idx], t_bin] += 1.0

                joint = joint / (joint.sum() + 1e-12)
                p_s = joint.sum(axis=1, keepdims=True) + 1e-12
                p_t = joint.sum(axis=0, keepdims=True) + 1e-12

                mi = float(np.sum(joint * np.log2((joint + 1e-12) / (p_s * p_t))))
                mi_values.append(max(0.0, mi))

            if not mi_values:
                return 0.0
            return float(np.mean(mi_values))

    def estimate_mi(
        self,
        spikes: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Estimate mutual information.

        Args:
            spikes: Ternary spike tensor
            teacher_hidden: Teacher hidden states

        Returns:
            Dictionary with MI value and method used
        """
        mi = self.binning_mi(spikes, teacher_hidden)
        return {
            "mutual_information": float(mi),
            "method": "binning_sign_discretization",
            "n_bins": self.n_bins,
            "n_dims_analyzed": self.n_dims,
        }


class RepresentationAnalyzer:
    """
    Analyze quality of spike representations using CKA.

    CKA (Centered Kernel Alignment) measures similarity between
    two representation spaces, independent of their dimensionality.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def linear_cka(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        eps: float = 1e-12,
        max_samples: int = 5000,
    ) -> float:
        """
        Compute linear CKA between two representation matrices.

        Adapted from v14 implementation with float32 stability fix.

        Args:
            X: First representation (N, d1)
            Y: Second representation (N, d2)
            eps: Small constant for numerical stability

        Returns:
            CKA similarity in [0, 1]
        """
        x = X.reshape(-1, X.shape[-1]).float().cpu().numpy()
        y = Y.reshape(-1, Y.shape[-1]).float().cpu().numpy()

        n = min(x.shape[0], y.shape[0], max_samples)
        if n == 0:
            return 0.0

        x = x[:n]
        y = y[:n]
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)

        hsic_xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
        hsic_xx = np.linalg.norm(x.T @ x, ord="fro") ** 2
        hsic_yy = np.linalg.norm(y.T @ y, ord="fro") ** 2

        denom = math.sqrt(float(hsic_xx * hsic_yy)) + eps
        return float(hsic_xy / denom)

    def compute_spike_cka(
        self,
        spike_info: Dict[int, List[Dict[str, torch.Tensor]]],
        teacher_hiddens: Dict[int, torch.Tensor],
        layer_map: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Compute CKA between spike representations and teacher hiddens.

        Args:
            spike_info: Spike tensors from student model
            teacher_hiddens: Teacher layer activations
            layer_map: Mapping from student layer to teacher layer

        Returns:
            Dictionary with CKA values per layer pair and mean
        """
        results = {}

        for s_layer, t_layer in layer_map.items():
            if s_layer not in spike_info:
                continue
            if t_layer not in teacher_hiddens:
                continue

            # Collect spike channels from this layer
            layer_spikes = spike_info[s_layer]
            if not layer_spikes:
                continue

            teacher_h = teacher_hiddens[t_layer].view(
                -1, teacher_hiddens[t_layer].shape[-1]
            ).to(self.device)

            local_cka = []
            for spike_key, suffix in (("k_spikes", "k"), ("v_spikes", "v")):
                tensors = [s[spike_key] for s in layer_spikes if spike_key in s]
                if not tensors:
                    continue

                spikes = torch.cat(
                    [tensor.view(-1, tensor.shape[-1]) for tensor in tensors],
                    dim=0,
                ).to(self.device)
                cka = self.linear_cka(spikes, teacher_h)
                results[f"layer_{s_layer}_to_{t_layer}_{suffix}"] = cka
                local_cka.append(cka)

            if local_cka:
                results[f"layer_{s_layer}_to_{t_layer}"] = float(np.mean(local_cka))

        if results:
            results["cka_mean"] = float(np.mean(list(results.values())))
            results["method"] = "linear_cka"

        return results


class SpikingBrainValidator:
    """
    Comprehensive validator for spike encoding quality.

    Validates that spike patterns encode meaningful semantic information,
    NOT arbitrary quantization artifacts.

    Usage:
        validator = SpikingBrainValidator(device)
        results = validator.validate(
            student=model,
            teacher=teacher,
            dataloader=val_loader,
        )
        print(results.summary)
    """

    def __init__(
        self,
        device: torch.device,
        layer_map: Optional[Dict[int, int]] = None,
        dead_threshold: float = 0.05,
        saturated_threshold: float = 0.10,
        mi_threshold: float = 0.1,
        cka_threshold: float = 0.3,
        firing_rate_range: Tuple[float, float] = (0.2, 0.6),
    ):
        """
        Args:
            device: Torch device
            layer_map: Student to teacher layer mapping (default: {0:2, 2:7, 4:11})
            dead_threshold: Alert if dead neurons exceed this fraction
            saturated_threshold: Alert if saturated neurons exceed this fraction
            mi_threshold: Minimum acceptable MI for pass
            cka_threshold: Minimum acceptable CKA for pass
            firing_rate_range: Acceptable range for mean firing rate
        """
        self.device = device
        self.layer_map = layer_map or {0: 2, 2: 7, 4: 11}

        # Thresholds
        self.dead_threshold = dead_threshold
        self.saturated_threshold = saturated_threshold
        self.mi_threshold = mi_threshold
        self.cka_threshold = cka_threshold
        self.firing_rate_range = firing_rate_range

        # Components
        self.spike_analyzer = SpikeAnalyzer()
        self.mi_estimator = MutualInformationEstimator()
        self.rep_analyzer = RepresentationAnalyzer(device)

    def _accumulate_layer_spikes(
        self,
        all_spike_info: Dict[int, List[Dict[str, torch.Tensor]]],
        layer_idx: int,
        spikes: Any,
    ) -> None:
        """Normalize spike traces to a list-of-dicts collector shape."""
        if layer_idx not in all_spike_info:
            all_spike_info[layer_idx] = []

        if isinstance(spikes, dict):
            all_spike_info[layer_idx].append(spikes)
            return

        if isinstance(spikes, list):
            all_spike_info[layer_idx].extend(spikes)
            return

        raise RuntimeError(
            f"Unsupported spike_info payload for layer {layer_idx}: {type(spikes)!r}"
        )

    def _collect_teacher_layer_hiddens(
        self,
        teacher: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Support both Hugging Face-style and repo-native teacher interfaces."""
        mapped_layers = tuple(dict.fromkeys(self.layer_map.values()))
        teacher_out = None

        try:
            teacher_out = teacher(input_ids, output_hidden_states=True)
        except TypeError:
            teacher_out = None

        hidden_states = getattr(teacher_out, "hidden_states", None)
        if hidden_states is not None:
            teacher_hiddens: Dict[int, torch.Tensor] = {}
            for t_layer in mapped_layers:
                hidden_index = t_layer + 1
                if hidden_index >= len(hidden_states):
                    raise RuntimeError(
                        f"Teacher hidden_states missing mapped layer {t_layer}."
                    )
                teacher_hiddens[t_layer] = hidden_states[hidden_index].detach().cpu()
            return teacher_hiddens

        if teacher_out is None:
            teacher_out = teacher(input_ids, return_features=True)

        if not (isinstance(teacher_out, tuple) and len(teacher_out) == 3):
            raise RuntimeError(
                "Teacher must return Hugging Face hidden_states or a repo-native "
                "(logits, states, aux) tuple."
            )

        _, _, teacher_aux = teacher_out
        layer_activations = (
            teacher_aux.get("layer_activations")
            if isinstance(teacher_aux, dict)
            else None
        )
        if not isinstance(layer_activations, dict):
            raise RuntimeError(
                "Teacher repo-native forward missing aux['layer_activations']."
            )

        teacher_hiddens = {}
        for t_layer in mapped_layers:
            if t_layer not in layer_activations:
                raise RuntimeError(
                    f"Teacher repo-native forward missing layer activation for "
                    f"mapped layer {t_layer}."
                )
            teacher_hiddens[t_layer] = layer_activations[t_layer].detach().cpu()

        return teacher_hiddens

    @torch.no_grad()
    def collect_representations(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader,
        max_batches: int = 20,
    ) -> Tuple[Dict[int, List[Dict[str, torch.Tensor]]], Dict[int, torch.Tensor]]:
        """
        Collect spike and teacher representations for analysis.

        Args:
            student: ASNN-Goose student model
            teacher: GPT-2 teacher model
            dataloader: Validation data loader
            max_batches: Maximum batches to process

        Returns:
            (spike_info_dict, teacher_hiddens_dict)
        """
        student.eval()
        teacher.eval()

        all_spike_info: Dict[int, List[Dict[str, torch.Tensor]]] = {}
        all_teacher_hiddens: Dict[int, List[torch.Tensor]] = {}

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
            else:
                input_ids = batch[0].to(self.device)

            # Student forward with spike info
            _, _, student_aux = student(input_ids, return_spike_info=True)

            # Teacher forward with hidden states
            teacher_hiddens = self._collect_teacher_layer_hiddens(teacher, input_ids)

            # Accumulate spike info
            spike_info = student_aux.get("spike_info", {})
            for layer_idx, spikes in spike_info.items():
                self._accumulate_layer_spikes(all_spike_info, layer_idx, spikes)

            # Accumulate teacher hiddens (only mapped layers)
            for t_layer, teacher_hidden in teacher_hiddens.items():
                if t_layer not in all_teacher_hiddens:
                    all_teacher_hiddens[t_layer] = []
                all_teacher_hiddens[t_layer].append(teacher_hidden)

        # Stack teacher hiddens
        teacher_hiddens_stacked = {}
        for layer_idx, hiddens in all_teacher_hiddens.items():
            teacher_hiddens_stacked[layer_idx] = torch.cat(hiddens, dim=0)

        return all_spike_info, teacher_hiddens_stacked

    def compute_health_metrics(
        self,
        spike_info: Dict[int, List[Dict[str, torch.Tensor]]],
    ) -> SpikeHealthMetrics:
        """
        Compute spike health metrics.

        Args:
            spike_info: Spike tensors from student model

        Returns:
            SpikeHealthMetrics with dead/saturated neurons and firing rates
        """
        alerts = []
        per_channel_rates = {}
        all_firing_rates = []

        for layer_idx, layer_spikes in spike_info.items():
            # Collect K and V spike tensors
            k_tensors = [s["k_spikes"] for s in layer_spikes]
            v_tensors = [s["v_spikes"] for s in layer_spikes]

            # Per-channel firing rates
            k_rates = self.spike_analyzer.compute_per_channel_firing_rates(k_tensors)
            v_rates = self.spike_analyzer.compute_per_channel_firing_rates(v_tensors)

            per_channel_rates[f"layer_{layer_idx}_k"] = k_rates
            per_channel_rates[f"layer_{layer_idx}_v"] = v_rates

            if len(k_rates) > 0:
                all_firing_rates.append(k_rates)
            if len(v_rates) > 0:
                all_firing_rates.append(v_rates)

        if not all_firing_rates:
            return SpikeHealthMetrics(
                dead_neuron_pct=0.0,
                saturated_neuron_pct=0.0,
                firing_rate_mean=0.0,
                firing_rate_std=0.0,
                per_channel_rates=per_channel_rates,
                health_pass=False,
                alerts=["No spike data collected"],
            )

        # Aggregate all firing rates
        combined_rates = np.concatenate(all_firing_rates)

        # Dead neurons
        dead_pct, _ = self.spike_analyzer.detect_dead_neurons(combined_rates)
        if dead_pct > self.dead_threshold:
            alerts.append(
                f"ALERT: {dead_pct*100:.1f}% dead neurons (>{self.dead_threshold*100}%)"
            )

        # Saturated neurons - need original tensors
        all_k_tensors = []
        for layer_spikes in spike_info.values():
            all_k_tensors.extend([s["k_spikes"] for s in layer_spikes])

        saturated_pct, _ = self.spike_analyzer.detect_saturated_neurons(all_k_tensors)
        if saturated_pct > self.saturated_threshold:
            alerts.append(
                f"ALERT: {saturated_pct*100:.1f}% saturated neurons (>{self.saturated_threshold*100}%)"
            )

        # Firing rate range check
        fr_mean = float(combined_rates.mean())
        fr_std = float(combined_rates.std())

        if not (self.firing_rate_range[0] <= fr_mean <= self.firing_rate_range[1]):
            alerts.append(
                f"ALERT: Firing rate {fr_mean:.3f} outside [{self.firing_rate_range[0]}, {self.firing_rate_range[1]}]"
            )

        health_pass = len(alerts) == 0

        return SpikeHealthMetrics(
            dead_neuron_pct=dead_pct,
            saturated_neuron_pct=saturated_pct,
            firing_rate_mean=fr_mean,
            firing_rate_std=fr_std,
            per_channel_rates=per_channel_rates,
            health_pass=health_pass,
            alerts=alerts,
        )

    def _compute_information_and_cka(
        self,
        spike_info: Dict[int, List[Dict[str, torch.Tensor]]],
        teacher_hiddens: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Mirror the reset notebook's per-layer K/V aggregation logic."""
        mi_results: Dict[str, float] = {}
        cka_results: Dict[str, float] = {}
        mi_metadata: Dict[str, float] = {}

        for s_layer, t_layer in self.layer_map.items():
            if s_layer not in spike_info or t_layer not in teacher_hiddens:
                continue

            layer_spikes = spike_info[s_layer]
            teacher_h = teacher_hiddens[t_layer].view(
                -1, teacher_hiddens[t_layer].shape[-1]
            )

            local_mi: List[float] = []
            local_cka: List[float] = []

            for spike_key, suffix in (("k_spikes", "k"), ("v_spikes", "v")):
                spike_tensors = [s[spike_key] for s in layer_spikes if spike_key in s]
                if not spike_tensors:
                    continue

                spikes_flat = torch.cat(
                    [spikes.view(-1, spikes.shape[-1]) for spikes in spike_tensors],
                    dim=0,
                )

                mi_payload = self.mi_estimator.estimate_mi(spikes_flat, teacher_h)
                mi_value = float(mi_payload.get("mutual_information", 0.0))
                mi_results[f"layer_{s_layer}_to_{t_layer}_{suffix}"] = mi_value
                local_mi.append(mi_value)

                if not mi_metadata:
                    mi_metadata = {
                        key: value
                        for key, value in mi_payload.items()
                        if key != "mutual_information"
                    }

                cka_value = self.rep_analyzer.linear_cka(spikes_flat, teacher_h)
                cka_results[f"layer_{s_layer}_to_{t_layer}_{suffix}"] = cka_value
                local_cka.append(cka_value)

            if local_mi:
                mi_results[f"layer_{s_layer}_to_{t_layer}"] = float(np.mean(local_mi))
            if local_cka:
                cka_results[f"layer_{s_layer}_to_{t_layer}"] = float(np.mean(local_cka))

        mi_mean = float(np.mean(list(mi_results.values()))) if mi_results else 0.0
        cka_mean = float(np.mean(list(cka_results.values()))) if cka_results else 0.0

        mi_summary: Dict[str, float] = {
            **mi_results,
            "mutual_information": mi_mean,
            "method": mi_metadata.get("method", "binning"),
        }
        for key in ("n_bins", "n_dims_analyzed"):
            if key in mi_metadata:
                mi_summary[key] = mi_metadata[key]

        cka_summary: Dict[str, float] = {
            **cka_results,
            "cka_mean": cka_mean,
            "method": "linear_cka",
        }

        return mi_summary, cka_summary

    def validate(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader,
        max_batches: int = 20,
    ) -> SpikingBrainValidation:
        """
        Run full SpikingBrain validation suite.

        Args:
            student: ASNN-Goose student model
            teacher: GPT-2 teacher model
            dataloader: Validation data loader
            max_batches: Maximum batches to analyze

        Returns:
            SpikingBrainValidation with all results
        """
        print("Collecting representations...")
        spike_info, teacher_hiddens = self.collect_representations(
            student, teacher, dataloader, max_batches
        )

        # 1. Health Check
        print("Computing health metrics...")
        health = self.compute_health_metrics(spike_info)

        # 2-3. Mutual Information + CKA
        print("Estimating mutual information...")
        print("Computing CKA similarity...")
        mi_results, cka_results = self._compute_information_and_cka(
            spike_info, teacher_hiddens
        )

        # Overall assessment
        mi_pass = mi_results.get("mutual_information", 0) > self.mi_threshold
        cka_pass = cka_results.get("cka_mean", 0) > self.cka_threshold

        overall_pass = health.health_pass and mi_pass and cka_pass

        summary = self._generate_summary(health, mi_results, cka_results, overall_pass)

        return SpikingBrainValidation(
            health=health,
            mutual_information=mi_results,
            cka=cka_results,
            overall_pass=overall_pass,
            summary=summary,
        )

    def _generate_summary(
        self,
        health: SpikeHealthMetrics,
        mi: Dict[str, float],
        cka: Dict[str, float],
        overall_pass: bool,
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "SPIKINGBRAIN VALIDATION SUMMARY",
            "=" * 60,
            "",
            "[HEALTH]",
            f"  Dead neurons: {health.dead_neuron_pct*100:.1f}% {'PASS' if health.dead_neuron_pct < self.dead_threshold else 'FAIL'}",
            f"  Saturated neurons: {health.saturated_neuron_pct*100:.1f}% {'PASS' if health.saturated_neuron_pct < self.saturated_threshold else 'FAIL'}",
            f"  Firing rate: {health.firing_rate_mean:.3f} +/- {health.firing_rate_std:.3f}",
            "",
            "[INFORMATION]",
            f"  Mutual Information: {mi.get('mutual_information', 0):.4f} ({mi.get('method', 'binning')}) {'PASS' if mi.get('mutual_information', 0) > self.mi_threshold else 'FAIL'}",
            "",
            "[REPRESENTATION]",
            f"  CKA (mean): {cka.get('cka_mean', 0):.4f} {'PASS' if cka.get('cka_mean', 0) > self.cka_threshold else 'FAIL'}",
        ]

        # Add per-layer CKA
        for key, value in cka.items():
            if key.startswith("cka_layer") or key.startswith("layer_"):
                lines.append(f"    {key}: {value:.4f}")

        # Alerts
        if health.alerts:
            lines.extend(["", "[ALERTS]"])
            for alert in health.alerts:
                lines.append(f"  {alert}")

        lines.extend([
            "",
            "=" * 60,
            f"OVERALL: {'PASS - Ready for v16 (sparse ops)' if overall_pass else 'NEEDS ATTENTION'}",
            "=" * 60,
        ])

        return "\n".join(lines)
