# =============================================================================
# cell 7: hardware and spike stats collectors (same as v9)
# =============================================================================
class HardwareStatsCollector:
    """collect gpu memory, timing, and throughput metrics."""

    def __init__(self):
        self.gpu_memory_history = []
        self.step_times = []
        self.tokens_processed = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_step(self, batch_size: int, seq_len: int):
        if torch.cuda.is_available():
            self.gpu_memory_history.append(torch.cuda.memory_allocated() / 1e9)
        self.tokens_processed += batch_size * seq_len
        self.step_times.append(time.time())

    def get_throughput(self) -> float:
        if len(self.step_times) < 2:
            return 0.0
        elapsed = self.step_times[-1] - self.step_times[0]
        return self.tokens_processed / elapsed if elapsed > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'peak_gpu_memory_gb': max(self.gpu_memory_history) if self.gpu_memory_history else 0,
            'avg_gpu_memory_gb': float(np.mean(self.gpu_memory_history)) if self.gpu_memory_history else 0,
            'total_training_time_s': elapsed,
            'total_training_time_min': elapsed / 60,
            'tokens_processed': self.tokens_processed,
            'throughput_tokens_per_sec': self.get_throughput(),
        }


class SpikeStatsCollector:
    """collect per-layer spike density and amplitude evolution."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.density_history = {i: {'k': [], 'v': []} for i in range(n_layers)}
        self.amplitude_history = {i: {'k': [], 'v': []} for i in range(n_layers)}
        self.step_densities = []

    def record(self, student, step: int):
        stats = student.get_spike_stats()
        all_densities = []
        for i in range(self.n_layers):
            layer_key = f'layer_{i}'
            if layer_key in stats:
                k_density = stats[layer_key].get('k', 0)
                v_density = stats[layer_key].get('v', 0)
                k_amp = stats[layer_key].get('k_amp', 1.0)
                v_amp = stats[layer_key].get('v_amp', 1.0)

                self.density_history[i]['k'].append(k_density)
                self.density_history[i]['v'].append(v_density)
                self.amplitude_history[i]['k'].append(k_amp)
                self.amplitude_history[i]['v'].append(v_amp)
                all_densities.extend([k_density, v_density])

        if all_densities:
            self.step_densities.append({'step': step, 'density': float(np.mean(all_densities))})

    def get_summary(self) -> Dict[str, Any]:
        per_layer = {}
        all_k, all_v = [], []
        all_k_amp, all_v_amp = [], []

        for i in range(self.n_layers):
            k_vals = self.density_history[i]['k']
            v_vals = self.density_history[i]['v']
            k_amps = self.amplitude_history[i]['k']
            v_amps = self.amplitude_history[i]['v']

            per_layer[f'layer_{i}'] = {
                'k_mean': float(np.mean(k_vals)) if k_vals else 0,
                'k_std': float(np.std(k_vals)) if k_vals else 0,
                'k_final': float(k_vals[-1]) if k_vals else 0,
                'v_mean': float(np.mean(v_vals)) if v_vals else 0,
                'v_std': float(np.std(v_vals)) if v_vals else 0,
                'v_final': float(v_vals[-1]) if v_vals else 0,
                'k_amp_final': float(k_amps[-1]) if k_amps else 1.0,
                'v_amp_final': float(v_amps[-1]) if v_amps else 1.0,
            }
            all_k.extend(k_vals)
            all_v.extend(v_vals)
            if k_amps: all_k_amp.append(k_amps[-1])
            if v_amps: all_v_amp.append(v_amps[-1])

        return {
            'per_layer': per_layer,
            'overall_k_density': float(np.mean(all_k)) if all_k else 0,
            'overall_v_density': float(np.mean(all_v)) if all_v else 0,
            'overall_density': float(np.mean(all_k + all_v)) if (all_k or all_v) else 0,
            'amplitudes': {'k': all_k_amp, 'v': all_v_amp},
            'density_history': self.step_densities,
        }

print("collectors defined")
