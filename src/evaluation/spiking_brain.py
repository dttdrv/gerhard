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
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            Estimated mutual information in nats (can convert to bits by /log(2))
        """
        with torch.no_grad():
            # Flatten to (N, d)
            s_flat = spikes.view(-1, spikes.shape[-1]).float().cpu().numpy()
            t_flat = teacher_hidden.view(-1, teacher_hidden.shape[-1]).float().cpu().numpy()

            # Use first n_dims dimensions for speed
            n_dims = min(self.n_dims, s_flat.shape[1], t_flat.shape[1])
            s_reduced = s_flat[:, :n_dims]
            t_reduced = t_flat[:, :n_dims]

            # Compute MI per dimension and average
            mi_sum = 0.0
            for d in range(n_dims):
                # Spike values are already discrete: {-1, 0, 1} -> {0, 1, 2}
                s_bins = (s_reduced[:, d] + 1).astype(int)
                s_bins = np.clip(s_bins, 0, 2)

                # Bin teacher values into n_bins
                t_col = t_reduced[:, d]
                t_min, t_max = t_col.min(), t_col.max()
                if t_max - t_min < 1e-8:
                    continue  # Constant column, no information

                t_bins = np.digitize(
                    t_col,
                    np.linspace(t_min, t_max, self.n_bins + 1)[1:-1]
                )

                # Joint and marginal histograms
                joint_hist, _, _ = np.histogram2d(
                    s_bins, t_bins,
                    bins=[3, self.n_bins],
                    range=[[0, 3], [0, self.n_bins]]
                )
                joint_prob = joint_hist / (joint_hist.sum() + 1e-10)

                s_marginal = joint_prob.sum(axis=1) + 1e-10
                t_marginal = joint_prob.sum(axis=0) + 1e-10

                # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
                for i in range(3):
                    for j in range(self.n_bins):
                        p_xy = joint_prob[i, j]
                        if p_xy > 1e-10:
                            mi_sum += p_xy * np.log(
                                p_xy / (s_marginal[i] * t_marginal[j])
                            )

            return mi_sum / max(n_dims, 1)

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
            "method": "binning",
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
        eps: float = 1e-8,
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
        with torch.cuda.amp.autocast(enabled=False):
            X = X.float().view(-1, X.shape[-1])
            Y = Y.float().view(-1, Y.shape[-1])

            # Ensure same number of samples
            n = min(X.shape[0], Y.shape[0])
            X = X[:n]
            Y = Y[:n]

            # Center
            X_centered = X - X.mean(dim=0, keepdim=True)
            Y_centered = Y - Y.mean(dim=0, keepdim=True)

            # Row normalize for stability
            X_norm = X_centered / (X_centered.norm(dim=1, keepdim=True) + eps)
            Y_norm = Y_centered / (Y_centered.norm(dim=1, keepdim=True) + eps)

            # Gram matrices
            K_X = X_norm @ X_norm.T
            K_Y = Y_norm @ Y_norm.T

            # HSIC (Hilbert-Schmidt Independence Criterion)
            hsic_xy = (K_X * K_Y).sum()
            hsic_xx = (K_X * K_X).sum()
            hsic_yy = (K_Y * K_Y).sum()

            # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
            cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)

            return float(cka.item())

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

            # Collect K spikes from this layer
            layer_spikes = spike_info[s_layer]
            if not layer_spikes:
                continue

            k_spikes = torch.cat(
                [s["k_spikes"].view(-1, s["k_spikes"].shape[-1])
                 for s in layer_spikes],
                dim=0
            ).to(self.device)

            teacher_h = teacher_hiddens[t_layer].view(
                -1, teacher_hiddens[t_layer].shape[-1]
            ).to(self.device)

            cka = self.linear_cka(k_spikes, teacher_h)
            results[f"cka_layer_{s_layer}_to_{t_layer}"] = cka

        if results:
            cka_values = [v for k, v in results.items() if k.startswith("cka_layer")]
            results["cka_mean"] = float(np.mean(cka_values))

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
            teacher_out = teacher(input_ids, output_hidden_states=True)
            teacher_hiddens = teacher_out.hidden_states

            # Accumulate spike info
            spike_info = student_aux.get("spike_info", {})
            for layer_idx, spikes in spike_info.items():
                if layer_idx not in all_spike_info:
                    all_spike_info[layer_idx] = []
                # spikes is a list of dicts per timestep
                all_spike_info[layer_idx].extend(spikes)

            # Accumulate teacher hiddens (only mapped layers)
            for t_layer in self.layer_map.values():
                if t_layer not in all_teacher_hiddens:
                    all_teacher_hiddens[t_layer] = []
                # +1 because hidden_states[0] is embedding
                all_teacher_hiddens[t_layer].append(
                    teacher_hiddens[t_layer + 1].detach().cpu()
                )

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

        # 2. Mutual Information
        print("Estimating mutual information...")
        # Use first mapped layer pair
        s_layer = list(self.layer_map.keys())[0]
        t_layer = self.layer_map[s_layer]

        if s_layer in spike_info and spike_info[s_layer]:
            k_spikes = torch.cat(
                [s["k_spikes"].view(-1, s["k_spikes"].shape[-1])
                 for s in spike_info[s_layer]],
                dim=0
            )
            teacher_h = teacher_hiddens[t_layer].view(
                -1, teacher_hiddens[t_layer].shape[-1]
            )
            mi_results = self.mi_estimator.estimate_mi(k_spikes, teacher_h)
        else:
            mi_results = {"mutual_information": 0.0, "method": "binning"}

        # 3. CKA
        print("Computing CKA similarity...")
        cka_results = self.rep_analyzer.compute_spike_cka(
            spike_info, teacher_hiddens, self.layer_map
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
            if key.startswith("cka_layer"):
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
