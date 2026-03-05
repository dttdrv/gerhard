# =============================================================================
# cell 8: spiking goose model (v14 - channel-wise spikes + gradient checkpointing)
# =============================================================================
class SpikingGooseRecurrentLayer(nn.Module):
    """
    RWKV-style recurrence with trainable ternary spiking.
    
    Supports channel-wise ternary spikes (when use_channel_wise=True)
    """

    def __init__(self, d_model, layer_idx=0, n_layers=4, spike_alpha=1.0,
                 use_channel_wise: bool = False, threshold_mix: float = 0.35,
                 surrogate_temp: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_channel_wise = use_channel_wise
        self.ln = nn.LayerNorm(d_model)

        ratio = layer_idx / max(n_layers - 1, 1)
        self.time_mix_k = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.decay_weight = nn.Parameter(torch.zeros(d_model) - 0.5)

        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.receptance_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        # v14: Use channel-wise spikes if enabled
        if use_channel_wise:
            self.k_spike = ChannelWiseTernarySpike(d_model, alpha_init=spike_alpha)
            self.v_spike = ChannelWiseTernarySpike(d_model, alpha_init=spike_alpha)
        else:
            self.k_spike = TrainableTernarySpike(
                alpha=spike_alpha,
                threshold_mix=threshold_mix,
                surrogate_temp=surrogate_temp,
            )
            self.v_spike = TrainableTernarySpike(
                alpha=spike_alpha,
                threshold_mix=threshold_mix,
                surrogate_temp=surrogate_temp,
            )

        self.register_buffer('running_k_density', torch.tensor(0.0))
        self.register_buffer('running_v_density', torch.tensor(0.0))
        self._init_weights()

    def _init_weights(self):
        std = 0.1 / math.sqrt(self.d_model)
        for m in [self.key_proj, self.value_proj, self.receptance_proj, self.output_proj]:
            nn.init.normal_(m.weight, std=std)

    def forward(self, x, return_spikes: bool = False, detach_spikes: bool = True):
        B, T, D = x.shape
        x_norm = self.ln(x)
        prev_x = F.pad(x_norm[:, :-1, :], (0, 0, 1, 0))

        xk = x_norm * self.time_mix_k + prev_x * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + prev_x * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + prev_x * (1 - self.time_mix_r)

        k_pre = self.key_proj(xk)
        v_pre = self.value_proj(xv)

        k_aux = {}
        v_aux = {}
        if return_spikes:
            k, k_aux = self.k_spike(k_pre, return_aux=True)
            v, v_aux = self.v_spike(v_pre, return_aux=True)
        else:
            k = self.k_spike(k_pre)
            v = self.v_spike(v_pre)
        r = torch.sigmoid(self.receptance_proj(xr))

        kv = k * v
        decay = torch.sigmoid(self.decay_weight)
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype)
        decay_powers = decay.unsqueeze(0) ** t_idx.unsqueeze(1)

        kv_weighted = kv / (decay_powers.unsqueeze(0) + 1e-8)
        S = torch.cumsum(kv_weighted, dim=1) * decay_powers.unsqueeze(0)

        if self.training:
            with torch.no_grad():
                self.running_k_density = 0.99 * self.running_k_density + 0.01 * (k != 0).float().mean()
                self.running_v_density = 0.99 * self.running_v_density + 0.01 * (v != 0).float().mean()

        out = x + r * self.output_proj(S)
        if return_spikes:
            k_out = k.detach() if detach_spikes else k
            v_out = v.detach() if detach_spikes else v
            return out, {
                'k_spikes': k_out,
                'v_spikes': v_out,
                'k_soft_activity': k_aux.get('soft_activity'),
                'v_soft_activity': v_aux.get('soft_activity'),
            }
        return out

    def get_spike_density(self):
        return {
            'k': self.running_k_density.item(),
            'v': self.running_v_density.item(),
            'k_amp': self.k_spike.get_amplitude(),
            'v_amp': self.v_spike.get_amplitude(),
        }
    
    def get_channel_wise_stats(self) -> dict:
        """Get channel-wise spike statistics (only available if use_channel_wise=True)."""
        if self.use_channel_wise:
            return {
                'k': self.k_spike.get_stats(),
                'v': self.v_spike.get_stats(),
            }
        return None


class GooseFFN(nn.Module):
    def __init__(self, d_model, expand=4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_model * expand, bias=False)
        self.w2 = nn.Linear(d_model * expand, d_model, bias=False)

    def forward(self, x):
        return x + self.w2(F.silu(self.w1(self.ln(x))))


class StudentSpikingGoose(nn.Module):
    """
    Spiking student model with trainable ternary activations.
    
    Supports channel-wise ternary spikes + gradient checkpointing.
    """

    def __init__(self, cfg, use_checkpointing=True):
        super().__init__()
        self.cfg = cfg
        self.use_checkpointing = use_checkpointing and USE_GRADIENT_CHECKPOINTING
        
        # v14: Check for channel-wise spikes flag
        use_channel_wise = getattr(cfg, 'use_channel_wise_spikes', False)
        
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'rec': SpikingGooseRecurrentLayer(
                    cfg.d_model, i, cfg.n_layers, cfg.spike_alpha,
                    use_channel_wise=use_channel_wise,
                    threshold_mix=cfg.spike_threshold_mix,
                    surrogate_temp=cfg.spike_surrogate_temp,
                ),
                'ffn': GooseFFN(cfg.d_model),
            })
            for i in range(cfg.n_layers)
        ])

        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def _layer_forward(self, layer, x):
        """helper for gradient checkpointing - processes one layer."""
        x = layer['rec'](x)
        x = layer['ffn'](x)
        return x

    def forward(self, input_ids, return_hiddens=False, return_spike_info=False, detach_spikes: bool = True):
        """forward pass with optional hidden state return for alignment."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        hiddens = [x] if return_hiddens else None
        spike_info = {} if return_spike_info else None

        for layer_idx, layer in enumerate(self.layers):
            # Checkpoint path is tensor-only; skip it when spike tensors are requested.
            if self.use_checkpointing and self.training and not return_spike_info:
                x = checkpoint(self._layer_forward, layer, x, use_reentrant=False)
            else:
                if return_spike_info:
                    x, layer_spikes = layer['rec'](
                        x,
                        return_spikes=True,
                        detach_spikes=detach_spikes,
                    )
                    x = layer['ffn'](x)
                    spike_info[layer_idx] = layer_spikes
                else:
                    x = self._layer_forward(layer, x)

            if return_hiddens:
                hiddens.append(x)

        logits = self.head(self.ln_out(x))

        if return_hiddens and return_spike_info:
            return logits, hiddens, {'spike_info': spike_info}
        if return_hiddens:
            return logits, hiddens
        if return_spike_info:
            return logits, {'spike_info': spike_info}
        return logits

    def get_spike_stats(self):
        return {f'layer_{i}': layer['rec'].get_spike_density() for i, layer in enumerate(self.layers)}

    def get_avg_spike_density(self):
        densities = []
        for layer in self.layers:
            d = layer['rec'].get_spike_density()
            densities.extend([d['k'], d['v']])
        return float(np.mean(densities)) if densities else 0.0

    def get_amplitudes(self):
        return {f'layer_{i}': {'k': layer['rec'].k_spike.get_amplitude(), 'v': layer['rec'].v_spike.get_amplitude()}
                for i, layer in enumerate(self.layers)}
    
    def get_channel_amplitude_variance(self) -> float:
        """Get total variance of channel-wise amplitudes (for regularization)."""
        total_var = 0.0
        for layer in self.layers:
            rec = layer['rec']
            if hasattr(rec.k_spike, 'amplitude') and rec.k_spike.amplitude.numel() > 1:
                total_var += rec.k_spike.amplitude.var().item()
                total_var += rec.v_spike.amplitude.var().item()
        return total_var

print("student model defined (v14: channel-wise spikes + gradient checkpointing)")
print(f"  gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
print(f"  channel-wise spikes: {config.use_channel_wise_spikes}")
