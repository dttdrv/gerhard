# =============================================================================
# cell 27: V15 SpikingBrain - Information Encoding Validation
# =============================================================================
# Validate that spike patterns encode meaningful semantic information
# Prerequisite for v16 (sparse ops)

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# INLINE: SpikingBrain Validation Classes
# =============================================================================

@dataclass
class SpikeHealthMetrics:
    """Container for spike health metrics."""
    dead_neuron_pct: float
    dead_neuron_indices: Dict[str, np.ndarray]
    saturated_neuron_pct: float
    saturated_neuron_indices: Dict[str, np.ndarray]
    firing_rate_mean: float
    firing_rate_std: float
    per_channel_rates: Dict[str, np.ndarray]
    health_pass: bool
    alerts: List[str]

@dataclass
class SpikingBrainValidation:
    """Complete validation results."""
    health: SpikeHealthMetrics
    mutual_information: Dict[str, float]
    cka: Dict[str, float]
    overall_pass: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'health': {
                'dead_neuron_pct': self.health.dead_neuron_pct,
                'saturated_neuron_pct': self.health.saturated_neuron_pct,
                'firing_rate_mean': self.health.firing_rate_mean,
                'firing_rate_std': self.health.firing_rate_std,
                'health_pass': self.health.health_pass,
                'alerts': self.health.alerts,
            },
            'mutual_information': self.mutual_information,
            'cka': self.cka,
            'overall_pass': self.overall_pass,
        }


class MutualInformationEstimator:
    """Estimate MI between spikes and teacher hiddens using binning."""

    def __init__(self, n_dims: int = 8, n_bins: int = 32):
        self.n_dims = n_dims
        self.n_bins = n_bins

    @torch.no_grad()
    def estimate(
        self,
        spikes: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> float:
        """Binning-based MI estimation."""
        # Flatten to 2D
        spikes_flat = spikes.reshape(-1, spikes.shape[-1])[:, :self.n_dims]
        teacher_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])[:, :self.n_dims]

        n_samples = min(spikes_flat.shape[0], 10000)
        spikes_flat = spikes_flat[:n_samples].cpu().numpy()
        teacher_flat = teacher_flat[:n_samples].cpu().numpy()

        # Bin teacher values
        teacher_binned = np.zeros_like(teacher_flat, dtype=np.int32)
        for d in range(self.n_dims):
            col = teacher_flat[:, d]
            bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, self.n_bins + 1)
            teacher_binned[:, d] = np.digitize(col, bins) - 1

        # Robust ternary discretization by sign (independent of learned amplitude).
        spikes_discrete = np.ones_like(spikes_flat, dtype=np.int32)
        spikes_discrete[spikes_flat > 1e-6] = 2
        spikes_discrete[spikes_flat < -1e-6] = 0

        # Compute MI per dimension and average
        mi_per_dim = []
        for d in range(self.n_dims):
            # Joint histogram
            joint = np.zeros((3, self.n_bins))
            for i in range(n_samples):
                s_idx = spikes_discrete[i, d]
                t_idx = max(0, min(teacher_binned[i, d], self.n_bins - 1))
                joint[s_idx, t_idx] += 1
            joint = joint / n_samples + 1e-10

            # Marginals
            p_s = joint.sum(axis=1, keepdims=True)
            p_t = joint.sum(axis=0, keepdims=True)

            # MI
            mi = np.sum(joint * np.log2(joint / (p_s * p_t + 1e-10)))
            mi_per_dim.append(max(0, mi))

        return float(np.mean(mi_per_dim))


class RepresentationAnalyzer:
    """Compute CKA between spike patterns and teacher representations."""

    @staticmethod
    def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA similarity."""
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
        hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
        hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2

        return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))

    @torch.no_grad()
    def compute_cka(
        self,
        spikes: torch.Tensor,
        teacher_hidden: torch.Tensor,
        max_samples: int = 5000,
    ) -> float:
        """Compute CKA between spikes and teacher hidden states."""
        spikes_flat = spikes.reshape(-1, spikes.shape[-1])
        teacher_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])

        n_samples = min(spikes_flat.shape[0], max_samples)
        X = spikes_flat[:n_samples].float().cpu().numpy()
        Y = teacher_flat[:n_samples].float().cpu().numpy()

        return self.linear_cka(X, Y)


class SpikingBrainValidator:
    """Main validator for V15 SpikingBrain validation."""

    def __init__(
        self,
        device: torch.device,
        layer_map: Dict[int, int],
        dead_threshold: float = 0.05,
        saturated_threshold: float = 0.10,
        mi_threshold: float = 0.1,
        cka_threshold: float = 0.3,
        firing_rate_range: Tuple[float, float] = (0.2, 0.6),
    ):
        self.device = device
        self.layer_map = layer_map
        self.dead_threshold = dead_threshold
        self.saturated_threshold = saturated_threshold
        self.mi_threshold = mi_threshold
        self.cka_threshold = cka_threshold
        self.firing_rate_range = firing_rate_range

        self.mi_estimator = MutualInformationEstimator()
        self.cka_analyzer = RepresentationAnalyzer()

    @torch.no_grad()
    def validate(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        dataloader,
        max_batches: int = 20,
    ) -> SpikingBrainValidation:
        """Run complete SpikingBrain validation."""
        student.eval()
        teacher.eval()

        # Collect spikes and teacher hiddens
        all_spikes = {}  # layer_idx -> {'k': [], 'v': []}
        all_teacher_hiddens = {}  # teacher_layer -> list of tensors

        print("Collecting spikes and teacher representations...")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
            else:
                raise TypeError(f"Unsupported batch type for validation: {type(batch)}")

            # Student forward with spikes
            student_out = student(input_ids, return_spike_info=True)
            aux = {}
            if isinstance(student_out, tuple):
                if len(student_out) == 2:
                    _, aux = student_out
                elif len(student_out) == 3:
                    _, _, aux = student_out
            spike_info = aux.get('spike_info', {}) if isinstance(aux, dict) else {}

            for layer_idx, layer_spikes in spike_info.items():
                if layer_idx not in all_spikes:
                    all_spikes[layer_idx] = {'k': [], 'v': []}
                if isinstance(layer_spikes, dict):
                    k_spikes = layer_spikes.get('k_spikes')
                    v_spikes = layer_spikes.get('v_spikes')
                    if k_spikes is not None:
                        all_spikes[layer_idx]['k'].append(k_spikes.cpu())
                    if v_spikes is not None:
                        all_spikes[layer_idx]['v'].append(v_spikes.cpu())
                elif isinstance(layer_spikes, list):
                    for s in layer_spikes:
                        if isinstance(s, dict):
                            if 'k_spikes' in s:
                                all_spikes[layer_idx]['k'].append(s['k_spikes'].cpu())
                            if 'v_spikes' in s:
                                all_spikes[layer_idx]['v'].append(s['v_spikes'].cpu())

            # Teacher forward (get hidden states from mapped layers)
            with torch.no_grad():
                teacher_out = teacher(input_ids, output_hidden_states=True)
                for student_layer, teacher_layer in self.layer_map.items():
                    if teacher_layer not in all_teacher_hiddens:
                        all_teacher_hiddens[teacher_layer] = []
                    h = teacher_out.hidden_states[teacher_layer + 1].cpu()
                    all_teacher_hiddens[teacher_layer].append(h)

        # 1. Compute health metrics
        print("Computing health metrics...")
        health = self._compute_health(all_spikes)

        # 2. Compute MI
        print("Estimating mutual information...")
        mi_results = self._compute_mi(all_spikes, all_teacher_hiddens)

        # 3. Compute CKA
        print("Computing CKA similarity...")
        cka_results = self._compute_cka(all_spikes, all_teacher_hiddens)

        # 4. Overall pass check
        overall_pass = (
            health.health_pass and
            mi_results.get('mutual_information', 0) >= self.mi_threshold and
            cka_results.get('cka_mean', 0) >= self.cka_threshold
        )

        # 5. Generate summary
        summary = self._generate_summary(health, mi_results, cka_results, overall_pass)

        return SpikingBrainValidation(
            health=health,
            mutual_information=mi_results,
            cka=cka_results,
            overall_pass=overall_pass,
            summary=summary,
        )

    def _compute_health(self, all_spikes: Dict[int, Dict[str, List[torch.Tensor]]]) -> SpikeHealthMetrics:
        """Compute spike health metrics."""
        dead_indices = {}
        saturated_indices = {}
        per_channel_rates = {}
        all_rates = []

        total_dead = 0
        total_saturated = 0
        total_channels = 0

        for layer_idx, layer_spikes in all_spikes.items():
            k_list = layer_spikes.get('k', [])
            v_list = layer_spikes.get('v', [])
            if not k_list and not v_list:
                continue

            if k_list and v_list:
                k_stacked = torch.cat([s.view(-1, s.shape[-1]) for s in k_list], dim=0)
                v_stacked = torch.cat([s.view(-1, s.shape[-1]) for s in v_list], dim=0)
                active = ((k_stacked != 0) | (v_stacked != 0)).float()
            else:
                base_list = k_list if k_list else v_list
                base = torch.cat([s.view(-1, s.shape[-1]) for s in base_list], dim=0)
                active = (base != 0).float()

            d_model = active.shape[-1]

            # Per-channel firing rates
            rates = active.mean(dim=0).numpy()
            per_channel_rates[f'layer_{layer_idx}'] = rates
            all_rates.append(rates)

            # Dead neurons (firing rate < 0.001)
            dead_mask = rates < 0.001
            dead_indices[f'layer_{layer_idx}'] = np.where(dead_mask)[0]
            total_dead += dead_mask.sum()

            # Saturated neurons (always fire)
            always_active = (active > 0.999).all(dim=0).numpy()
            saturated_indices[f'layer_{layer_idx}'] = np.where(always_active)[0]
            total_saturated += always_active.sum()

            total_channels += d_model

        if not all_rates:
            return SpikeHealthMetrics(
                dead_neuron_pct=1.0,
                dead_neuron_indices={},
                saturated_neuron_pct=0.0,
                saturated_neuron_indices={},
                firing_rate_mean=0.0,
                firing_rate_std=0.0,
                per_channel_rates={},
                health_pass=False,
                alerts=['No spike tensors captured during validation.'],
            )

        all_rates_flat = np.concatenate(all_rates)
        dead_pct = total_dead / total_channels if total_channels > 0 else 0
        saturated_pct = total_saturated / total_channels if total_channels > 0 else 0

        # Check health
        alerts = []
        if dead_pct > self.dead_threshold:
            alerts.append(f"Dead neurons: {dead_pct*100:.1f}% > {self.dead_threshold*100:.0f}%")
        if saturated_pct > self.saturated_threshold:
            alerts.append(f"Saturated neurons: {saturated_pct*100:.1f}% > {self.saturated_threshold*100:.0f}%")

        fr_mean = float(np.mean(all_rates_flat))
        if not (self.firing_rate_range[0] <= fr_mean <= self.firing_rate_range[1]):
            alerts.append(f"Firing rate {fr_mean:.3f} outside range {self.firing_rate_range}")

        health_pass = len(alerts) == 0

        return SpikeHealthMetrics(
            dead_neuron_pct=float(dead_pct),
            dead_neuron_indices=dead_indices,
            saturated_neuron_pct=float(saturated_pct),
            saturated_neuron_indices=saturated_indices,
            firing_rate_mean=fr_mean,
            firing_rate_std=float(np.std(all_rates_flat)),
            per_channel_rates=per_channel_rates,
            health_pass=health_pass,
            alerts=alerts,
        )

    def _compute_mi(
        self,
        all_spikes: Dict[int, Dict[str, List[torch.Tensor]]],
        all_teacher_hiddens: Dict[int, List[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute mutual information."""
        mi_per_layer = {}

        for student_layer, teacher_layer in self.layer_map.items():
            if student_layer not in all_spikes or teacher_layer not in all_teacher_hiddens:
                continue

            k_list = all_spikes[student_layer].get('k', [])
            v_list = all_spikes[student_layer].get('v', [])
            if not k_list and not v_list:
                continue
            hiddens = torch.cat(all_teacher_hiddens[teacher_layer], dim=0)

            layer_vals = []
            if k_list:
                mi_k = self.mi_estimator.estimate(torch.cat(k_list, dim=0), hiddens)
                mi_per_layer[f'layer_{student_layer}_to_{teacher_layer}_k'] = mi_k
                layer_vals.append(mi_k)
            if v_list:
                mi_v = self.mi_estimator.estimate(torch.cat(v_list, dim=0), hiddens)
                mi_per_layer[f'layer_{student_layer}_to_{teacher_layer}_v'] = mi_v
                layer_vals.append(mi_v)
            if layer_vals:
                mi_per_layer[f'layer_{student_layer}_to_{teacher_layer}'] = float(np.mean(layer_vals))

        mi_mean = np.mean(list(mi_per_layer.values())) if mi_per_layer else 0.0

        return {
            **mi_per_layer,
            'mutual_information': float(mi_mean),
        }

    def _compute_cka(
        self,
        all_spikes: Dict[int, Dict[str, List[torch.Tensor]]],
        all_teacher_hiddens: Dict[int, List[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute CKA similarity."""
        cka_per_layer = {}

        for student_layer, teacher_layer in self.layer_map.items():
            if student_layer not in all_spikes or teacher_layer not in all_teacher_hiddens:
                continue

            k_list = all_spikes[student_layer].get('k', [])
            v_list = all_spikes[student_layer].get('v', [])
            if not k_list and not v_list:
                continue
            hiddens = torch.cat(all_teacher_hiddens[teacher_layer], dim=0)

            layer_vals = []
            if k_list:
                cka_k = self.cka_analyzer.compute_cka(torch.cat(k_list, dim=0), hiddens)
                cka_per_layer[f'layer_{student_layer}_to_{teacher_layer}_k'] = cka_k
                layer_vals.append(cka_k)
            if v_list:
                cka_v = self.cka_analyzer.compute_cka(torch.cat(v_list, dim=0), hiddens)
                cka_per_layer[f'layer_{student_layer}_to_{teacher_layer}_v'] = cka_v
                layer_vals.append(cka_v)
            if layer_vals:
                cka_per_layer[f'layer_{student_layer}_to_{teacher_layer}'] = float(np.mean(layer_vals))

        cka_mean = np.mean(list(cka_per_layer.values())) if cka_per_layer else 0.0

        return {
            **cka_per_layer,
            'cka_mean': float(cka_mean),
        }

    def _generate_summary(
        self,
        health: SpikeHealthMetrics,
        mi_results: Dict[str, float],
        cka_results: Dict[str, float],
        overall_pass: bool,
    ) -> str:
        """Generate validation summary."""
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
            f"  Mutual Information: {mi_results.get('mutual_information', 0):.4f} {'PASS' if mi_results.get('mutual_information', 0) >= self.mi_threshold else 'FAIL'}",
            "",
            "[REPRESENTATION]",
            f"  CKA (mean): {cka_results.get('cka_mean', 0):.4f} {'PASS' if cka_results.get('cka_mean', 0) >= self.cka_threshold else 'FAIL'}",
            "",
            "=" * 60,
            f"OVERALL: {'PASS - Ready for v16 (sparse ops)' if overall_pass else 'NEEDS ATTENTION'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# RUN VALIDATION
# =============================================================================

print('='*60)
print('V15: SPIKINGBRAIN INFORMATION ENCODING VALIDATION')
print('='*60)

# Initialize validator with v14 layer mapping
validator = SpikingBrainValidator(
    device=DEVICE,
    layer_map={0: 2, 2: 7, 4: 11},  # Student -> Teacher layer mapping
    dead_threshold=0.05,      # Alert if >5% dead neurons
    saturated_threshold=0.10,  # Alert if >10% saturated neurons
    mi_threshold=0.1,         # Minimum acceptable MI
    cka_threshold=0.3,        # Minimum acceptable CKA
    firing_rate_range=(0.2, 0.6),  # Healthy firing rate range
)

# Run validation
v15_results = validator.validate(
    student=student,
    teacher=teacher,
    dataloader=val_loader,
    max_batches=20,
)

# Print summary
print(v15_results.summary)
