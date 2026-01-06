"""
ASNN-Goose: Full Student Model with Ternary Spiking Activations.

Reference: Sections 4-7 of ASNN-Goose blueprint.

This module integrates:
1. Goose backbone (RWKV-style recurrence)
2. Ternary spiking activations ({-1, 0, +1})
3. INT8 weight quantization
4. LoRA adapters for test-time training

The student model is trained via distillation from a dense teacher
and can adapt at test-time using LoRA updates triggered by spike
behavior anomalies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import math

from .goose_backbone import DeltaRuleState, GooseFFN
from .ternary_activations import (
    AdaptiveTernarySpike,
    FixedThresholdTernarySpike,
    SpikeActivityTracker,
    TernarySpikeLayer,
)
from .quantized_weights import QuantizedLinear
from .lora_adapter import LoRAAdapter, apply_lora_to_model


class SpikingGooseRecurrentLayer(nn.Module):
    """
    Spiking version of GooseRecurrentLayer.

    Replaces continuous activations with ternary spikes after
    key processing points. This creates a sparse communication
    pattern where zero spikes mean no computation needed.

    Reference: Section 5 of blueprint.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        n_layers: int = 4,
        adaptive_threshold: bool = True,
        threshold_alpha: float = 1.0,
        learnable_alpha: bool = True,
        quantize_weights: bool = True,
        track_spikes: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx

        # Layer normalization
        self.ln = nn.LayerNorm(d_model)

        # Time-mixing parameters
        ratio = layer_idx / max(n_layers - 1, 1)
        self.time_mix_k = nn.Parameter(torch.ones(d_model) * (1.0 - ratio))
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * (1.0 - ratio))
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * (1.0 - ratio))

        # Decay parameters
        self.decay_weight = nn.Parameter(torch.zeros(d_model) - 0.5)

        # Projections (optionally quantized)
        Linear = QuantizedLinear if quantize_weights else nn.Linear
        self.key_proj = Linear(d_model, d_model, bias=False)
        self.value_proj = Linear(d_model, d_model, bias=False)
        self.receptance_proj = Linear(d_model, d_model, bias=False)
        self.output_proj = Linear(d_model, d_model, bias=False)

        # Spiking activations
        if adaptive_threshold:
            self.spike_k = AdaptiveTernarySpike(
                alpha_init=threshold_alpha,
                learnable_alpha=learnable_alpha,
            )
            self.spike_v = AdaptiveTernarySpike(
                alpha_init=threshold_alpha,
                learnable_alpha=learnable_alpha,
            )
        else:
            self.spike_k = FixedThresholdTernarySpike(threshold=0.5)
            self.spike_v = FixedThresholdTernarySpike(threshold=0.5)

        # Spike tracking
        if track_spikes:
            self.tracker_k = SpikeActivityTracker(name=f"layer{layer_idx}_k")
            self.tracker_v = SpikeActivityTracker(name=f"layer{layer_idx}_v")
        else:
            self.tracker_k = None
            self.tracker_v = None

        # Initialize
        self._init_weights()

    def _init_weights(self):
        scale = 0.1 / math.sqrt(self.d_model)
        for proj in [self.key_proj, self.value_proj,
                     self.receptance_proj, self.output_proj]:
            if hasattr(proj, 'weight'):
                nn.init.normal_(proj.weight, std=scale)

    def forward(
        self,
        x: torch.Tensor,
        state: DeltaRuleState,
        prev_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState, Dict[str, torch.Tensor]]:
        """
        Process single timestep with spiking activations.

        Args:
            x: Current input (batch, d_model)
            state: Previous recurrent state
            prev_x: Previous input for time-mixing

        Returns:
            output: (batch, d_model)
            new_state: Updated DeltaRuleState
            spike_info: Dictionary with spike tensors for analysis
        """
        x_norm = self.ln(x)

        if prev_x is None:
            prev_x = torch.zeros_like(x_norm)
        else:
            prev_x = self.ln(prev_x)

        # Time-mixing
        xk = x_norm * self.time_mix_k + prev_x * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + prev_x * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + prev_x * (1 - self.time_mix_r)

        # Compute K, V, R with SPIKING activations
        k_pre = self.key_proj(xk)
        v_pre = self.value_proj(xv)

        # Apply ternary spiking
        k = self.spike_k(k_pre)
        v = self.spike_v(v_pre)

        # Track spikes
        if self.tracker_k is not None:
            k = self.tracker_k(k)
        if self.tracker_v is not None:
            v = self.tracker_v(v)

        # Receptance (continuous gate, not spiked)
        r = torch.sigmoid(self.receptance_proj(xr))

        # State update with spiked K and V
        decay = torch.sigmoid(self.decay_weight)
        new_S = decay * state.S + k * v

        # Output
        output = r * self.output_proj(new_S)
        output = x + output

        new_state = DeltaRuleState(S=new_S)
        spike_info = {"k_spikes": k, "v_spikes": v}

        return output, new_state, spike_info

    def forward_sequence(
        self,
        x: torch.Tensor,
        initial_state: Optional[DeltaRuleState] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState, List[Dict[str, torch.Tensor]]]:
        """Process full sequence."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if initial_state is None:
            state = DeltaRuleState.init_zeros(batch_size, self.d_model, device, dtype)
        else:
            state = initial_state

        outputs = []
        all_spike_info = []
        prev_x = None

        for t in range(seq_len):
            h = x[:, t, :]
            output, state, spike_info = self.forward(h, state, prev_x)
            outputs.append(output)
            all_spike_info.append(spike_info)
            prev_x = h

        outputs = torch.stack(outputs, dim=1)
        return outputs, state, all_spike_info

    def get_spike_stats(self) -> Dict[str, float]:
        """Get current spike statistics."""
        stats = {}
        if self.tracker_k is not None:
            stats.update({f"k_{k}": v for k, v in self.tracker_k.get_summary().items()})
        if self.tracker_v is not None:
            stats.update({f"v_{k}": v for k, v in self.tracker_v.get_summary().items()})
        stats["k_density"] = self.spike_k.get_spike_density()
        stats["v_density"] = self.spike_v.get_spike_density()
        return stats

    def reset_spike_stats(self):
        """Reset spike tracking statistics."""
        if self.tracker_k is not None:
            self.tracker_k.reset_stats()
        if self.tracker_v is not None:
            self.tracker_v.reset_stats()


class SpikingGooseBlock(nn.Module):
    """Spiking Goose block: Spiking recurrence + FFN."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        n_layers: int = 4,
        adaptive_threshold: bool = True,
        threshold_alpha: float = 1.0,
        learnable_alpha: bool = True,
        quantize_weights: bool = True,
        track_spikes: bool = True,
    ):
        super().__init__()
        self.recurrent = SpikingGooseRecurrentLayer(
            d_model=d_model,
            layer_idx=layer_idx,
            n_layers=n_layers,
            adaptive_threshold=adaptive_threshold,
            threshold_alpha=threshold_alpha,
            learnable_alpha=learnable_alpha,
            quantize_weights=quantize_weights,
            track_spikes=track_spikes,
        )
        self.ffn = GooseFFN(d_model=d_model, expand_factor=4)

    def forward(
        self,
        x: torch.Tensor,
        state: DeltaRuleState,
        prev_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState, Dict[str, torch.Tensor]]:
        x, state, spike_info = self.recurrent(x, state, prev_x)
        x = self.ffn(x)
        return x, state, spike_info

    def forward_sequence(
        self,
        x: torch.Tensor,
        initial_state: Optional[DeltaRuleState] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState, List[Dict[str, torch.Tensor]]]:
        x, state, spike_info = self.recurrent.forward_sequence(x, initial_state)
        x = self.ffn(x)
        return x, state, spike_info


class ASNNGoose(nn.Module):
    """
    ASNN-Goose: Adaptive Spiking Neural Network with Goose Backbone.

    This is the full student model combining:
    - RWKV-style recurrence
    - Ternary spiking activations
    - INT8 weight quantization
    - LoRA adapters for TTT

    Reference: Full blueprint.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        vocab_size: int = 32000,
        max_seq_len: int = 1024,
        adaptive_threshold: bool = True,
        threshold_alpha: float = 1.0,
        learnable_alpha: bool = True,
        quantize_weights: bool = True,
        track_spikes: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_target_modules: Optional[List[str]] = None,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Spiking layers
        self.layers = nn.ModuleList([
            SpikingGooseBlock(
                d_model=d_model,
                layer_idx=i,
                n_layers=n_layers,
                adaptive_threshold=adaptive_threshold,
                threshold_alpha=threshold_alpha,
                learnable_alpha=learnable_alpha,
                quantize_weights=quantize_weights,
                track_spikes=track_spikes,
            )
            for i in range(n_layers)
        ])

        # Output
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        if tie_weights:
            self.head.weight = self.embedding.weight

        # Initialize
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # LoRA modules (populated after calling apply_lora())
        self.lora_modules: Dict[str, LoRAAdapter] = {}
        self.lora_applied = False

        # Store config for LoRA
        self._lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "target_modules": lora_target_modules or ["key_proj", "value_proj"],
        }

    def apply_lora(
        self,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        freeze_base: bool = True,
    ):
        """
        Apply LoRA adapters to the model.

        Args:
            rank: LoRA rank (default from config)
            alpha: LoRA alpha (default from config)
            target_modules: Modules to target (default from config)
            freeze_base: Whether to freeze base weights
        """
        rank = rank or self._lora_config["rank"]
        alpha = alpha or self._lora_config["alpha"]
        target_modules = target_modules or self._lora_config["target_modules"]

        _, self.lora_modules = apply_lora_to_model(
            self,
            rank=rank,
            alpha=alpha,
            target_modules=set(target_modules),
            freeze_base=freeze_base,
        )
        self.lora_applied = True

    def init_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> List[DeltaRuleState]:
        """Initialize recurrent states for all layers."""
        return [
            DeltaRuleState.init_zeros(batch_size, self.d_model, device, dtype)
            for _ in range(self.n_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[DeltaRuleState]] = None,
        return_spike_info: bool = False,
    ) -> Tuple[torch.Tensor, List[DeltaRuleState], Dict[str, Any]]:
        """
        Forward pass with spiking activations.

        Args:
            input_ids: Token IDs (batch, seq_len)
            states: Optional initial states
            return_spike_info: Whether to return detailed spike info

        Returns:
            logits: (batch, seq_len, vocab_size)
            final_states: List of final states per layer
            aux_outputs: Dict with spike info and layer activations
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.embedding.weight.dtype

        if states is None:
            states = self.init_states(batch_size, device, dtype)

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Process through spiking layers
        aux_outputs: Dict[str, Any] = {
            "layer_activations": {},
            "spike_info": {} if return_spike_info else None,
        }

        for layer_idx, layer in enumerate(self.layers):
            x, states[layer_idx], spike_info = layer.forward_sequence(x, states[layer_idx])

            if return_spike_info:
                aux_outputs["spike_info"][layer_idx] = spike_info
                aux_outputs["layer_activations"][layer_idx] = x.detach()

        # Output
        x = self.ln_out(x)
        logits = self.head(x)

        return logits, states, aux_outputs

    def get_spike_stats(self) -> Dict[str, Dict[str, float]]:
        """Get spike statistics from all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f"layer_{i}"] = layer.recurrent.get_spike_stats()
        return stats

    def reset_spike_stats(self):
        """Reset spike statistics in all layers."""
        for layer in self.layers:
            layer.recurrent.reset_spike_stats()

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for TTT optimization."""
        params = []
        for lora in self.lora_modules.values():
            params.append(lora.lora_A)
            params.append(lora.lora_B)
        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by type."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in self.get_lora_parameters())

        return {
            "total": total,
            "trainable": trainable,
            "lora": lora,
            "frozen": total - trainable,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "lora_applied": self.lora_applied,
            "parameters": self.count_parameters(),
        }

    @classmethod
    def from_teacher(
        cls,
        teacher: nn.Module,
        adaptive_threshold: bool = True,
        threshold_alpha: float = 1.0,
        quantize_weights: bool = True,
        **kwargs,
    ) -> "ASNNGoose":
        """
        Create student model from teacher configuration.

        Args:
            teacher: Teacher model (GooseBackbone)
            adaptive_threshold: Use adaptive spiking thresholds
            threshold_alpha: Initial alpha for adaptive threshold
            quantize_weights: Use INT8 weight quantization
            **kwargs: Additional arguments

        Returns:
            Initialized ASNNGoose student model
        """
        # Extract config from teacher
        config = teacher.get_config()

        student = cls(
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            adaptive_threshold=adaptive_threshold,
            threshold_alpha=threshold_alpha,
            quantize_weights=quantize_weights,
            **kwargs,
        )

        # Copy embedding weights
        with torch.no_grad():
            student.embedding.weight.copy_(teacher.embedding.weight)
            student.pos_embedding.weight.copy_(teacher.pos_embedding.weight)

        return student
