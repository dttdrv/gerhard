"""
Goose Backbone: RWKV-7 style recurrence with delta-rule updates.

Reference: Section 5 of ASNN-Goose blueprint.
This implements the dense teacher model backbone that will be distilled
into the spiking student model.

State update form (Eq. 1):
    S_t = S_{t-1} - Decay(S_{t-1}, θ) + NewInfo(x_t, θ)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class DeltaRuleState:
    """
    Container for recurrent state in delta-rule formulation.

    The state captures accumulated information across timesteps,
    updated via the delta-rule: S_t = decay * S_{t-1} + new_info
    """
    S: torch.Tensor  # (batch, d_model) or (batch, n_heads, d_head)

    @classmethod
    def init_zeros(
        cls,
        batch_size: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> "DeltaRuleState":
        """Initialize state with zeros."""
        return cls(S=torch.zeros(batch_size, d_model, device=device, dtype=dtype))

    def detach(self) -> "DeltaRuleState":
        """Detach state from computation graph (for TBPTT)."""
        return DeltaRuleState(S=self.S.detach())

    def clone(self) -> "DeltaRuleState":
        """Create a copy of the state."""
        return DeltaRuleState(S=self.S.clone())


class GooseRecurrentLayer(nn.Module):
    """
    Single layer of RWKV-style recurrence with delta-rule state updates.

    This layer implements:
    1. Time-mixing: interpolation between current and previous timestep
    2. K/V/R projections: compute key, value, and receptance
    3. State update: delta-rule with learned decay
    4. Output gating: receptance-gated output

    State update form (Eq. 1):
        S_t = decay * S_{t-1} + k * v  (simplified outer product)
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: int = 2,
        layer_idx: int = 0,
        n_layers: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx

        # Layer normalization (pre-norm architecture)
        self.ln = nn.LayerNorm(d_model)

        # Time-mixing parameters (learnable interpolation weights)
        # Initialize based on layer position (deeper layers mix more current)
        ratio_0 = layer_idx / max(n_layers - 1, 1)
        ratio_1 = 1.0 - ratio_0

        self.time_mix_k = nn.Parameter(torch.ones(d_model) * ratio_1)
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * ratio_1)
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * ratio_1)

        # Decay parameters (learnable, initialized for stable recurrence)
        # Using sigmoid to ensure decay in [0, 1]
        self.decay_weight = nn.Parameter(torch.zeros(d_model) - 0.5)

        # Projections
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.receptance_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        # Initialize projections
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful scaling."""
        # Small initialization for stable training
        scale = 0.1 / math.sqrt(self.d_model)
        nn.init.normal_(self.key_proj.weight, std=scale)
        nn.init.normal_(self.value_proj.weight, std=scale)
        nn.init.normal_(self.receptance_proj.weight, std=scale)
        nn.init.normal_(self.output_proj.weight, std=scale)

    def forward(
        self,
        x: torch.Tensor,           # (batch, d_model) - single timestep
        state: DeltaRuleState,
        prev_x: Optional[torch.Tensor] = None,  # Previous timestep for mixing
    ) -> Tuple[torch.Tensor, DeltaRuleState]:
        """
        Process single timestep with state update.

        Args:
            x: Current input (batch, d_model)
            state: Previous recurrent state
            prev_x: Previous input for time-mixing (optional)

        Returns:
            output: (batch, d_model)
            new_state: Updated DeltaRuleState
        """
        batch_size = x.shape[0]

        # Layer norm
        x_norm = self.ln(x)

        # Handle first timestep
        if prev_x is None:
            prev_x = torch.zeros_like(x_norm)
        else:
            prev_x = self.ln(prev_x)

        # Time-mixing: interpolate with previous timestep
        xk = x_norm * self.time_mix_k + prev_x * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + prev_x * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + prev_x * (1 - self.time_mix_r)

        # Compute K, V, R projections
        k = self.key_proj(xk)      # Key: what to attend to
        v = self.value_proj(xv)    # Value: what to retrieve
        r = torch.sigmoid(self.receptance_proj(xr))  # Receptance: output gate

        # Compute decay (element-wise, in [0, 1])
        decay = torch.sigmoid(self.decay_weight)  # (d_model,)

        # Delta-rule state update (Eq. 1)
        # S_t = decay * S_{t-1} + k * v
        # This is a simplified version; full RWKV uses outer product attention
        new_S = decay * state.S + k * v  # Element-wise product for efficiency

        # Output with receptance gating
        output = r * self.output_proj(new_S)

        # Residual connection
        output = x + output

        # Create new state
        new_state = DeltaRuleState(S=new_S)

        return output, new_state

    def forward_sequence(
        self,
        x: torch.Tensor,           # (batch, seq_len, d_model)
        initial_state: Optional[DeltaRuleState] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState, List[torch.Tensor]]:
        """
        Process full sequence (for training efficiency).

        Returns:
            outputs: (batch, seq_len, d_model)
            final_state: Final recurrent state
            all_hidden: List of hidden states for analysis
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if initial_state is None:
            state = DeltaRuleState.init_zeros(batch_size, self.d_model, device, dtype)
        else:
            state = initial_state

        outputs = []
        all_hidden = []
        prev_x = None

        for t in range(seq_len):
            h = x[:, t, :]  # (batch, d_model)
            output, state = self.forward(h, state, prev_x)
            outputs.append(output)
            all_hidden.append(state.S.detach())
            prev_x = h

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return outputs, state, all_hidden


class GooseFFN(nn.Module):
    """
    Feed-forward network for Goose backbone.
    Uses SiLU (Swish) activation as in modern architectures.
    """

    def __init__(self, d_model: int, expand_factor: int = 4):
        super().__init__()
        d_ffn = d_model * expand_factor

        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

        # Initialize
        scale = 0.1 / math.sqrt(d_model)
        nn.init.normal_(self.w1.weight, std=scale)
        nn.init.normal_(self.w2.weight, std=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model)
        Returns:
            Output with same shape as input
        """
        residual = x
        x = self.ln(x)
        x = self.w2(F.silu(self.w1(x)))
        return residual + x


class GooseBlock(nn.Module):
    """
    Single Goose block: Recurrence + FFN.
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: int = 2,
        layer_idx: int = 0,
        n_layers: int = 4,
    ):
        super().__init__()
        self.recurrent = GooseRecurrentLayer(
            d_model=d_model,
            expand_factor=expand_factor,
            layer_idx=layer_idx,
            n_layers=n_layers,
        )
        self.ffn = GooseFFN(d_model=d_model, expand_factor=4)

    def forward(
        self,
        x: torch.Tensor,
        state: DeltaRuleState,
        prev_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState]:
        """Process single timestep."""
        x, state = self.recurrent(x, state, prev_x)
        x = self.ffn(x)
        return x, state

    def forward_sequence(
        self,
        x: torch.Tensor,
        initial_state: Optional[DeltaRuleState] = None,
    ) -> Tuple[torch.Tensor, DeltaRuleState, List[torch.Tensor]]:
        """Process full sequence."""
        x, state, hidden = self.recurrent.forward_sequence(x, initial_state)
        x = self.ffn(x)
        return x, state, hidden


class GooseBackbone(nn.Module):
    """
    Full Goose backbone with multiple recurrent layers.

    This is the DENSE teacher model with continuous activations.
    The student model (ASNNGoose) will replace activations with ternary spikes.

    Architecture:
    1. Token embedding
    2. N x GooseBlock (recurrence + FFN)
    3. Layer norm + LM head

    Reference: Section 5 of ASNN-Goose blueprint.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        vocab_size: int = 32000,
        max_seq_len: int = 1024,
        expand_factor: int = 2,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Recurrent layers
        self.layers = nn.ModuleList([
            GooseBlock(
                d_model=d_model,
                expand_factor=expand_factor,
                layer_idx=i,
                n_layers=n_layers,
            )
            for i in range(n_layers)
        ])

        # Output
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        if tie_weights:
            self.head.weight = self.embedding.weight

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

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
        input_ids: torch.Tensor,  # (batch, seq_len)
        states: Optional[List[DeltaRuleState]] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List[DeltaRuleState], Dict[str, Any]]:
        """
        Forward pass with sequential processing.

        Args:
            input_ids: Token IDs (batch, seq_len)
            states: Optional initial states for each layer
            return_hidden: Whether to return intermediate activations

        Returns:
            logits: (batch, seq_len, vocab_size)
            final_states: List of final states per layer
            aux_outputs: Dict with intermediate activations for analysis
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.embedding.weight.dtype

        # Initialize states if not provided
        if states is None:
            states = self.init_states(batch_size, device, dtype)

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Process through layers
        aux_outputs: Dict[str, Any] = {"layer_activations": {}}

        for layer_idx, layer in enumerate(self.layers):
            x, states[layer_idx], hidden = layer.forward_sequence(x, states[layer_idx])
            if return_hidden:
                aux_outputs["layer_activations"][layer_idx] = torch.stack(hidden, dim=1)

        # Output
        x = self.ln_out(x)
        logits = self.head(x)

        return logits, states, aux_outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Prompt tokens (batch, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)

        Returns:
            Generated tokens (batch, prompt_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Process prompt
        states = self.init_states(batch_size, device)
        logits, states, _ = self.forward(input_ids, states)

        # Start generation
        generated = [input_ids]
        next_token_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)

            # Forward step
            logits, states, _ = self.forward(next_token, states)
            next_token_logits = logits[:, -1, :]

        return torch.cat(generated, dim=1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "num_parameters": self.count_parameters(),
        }
