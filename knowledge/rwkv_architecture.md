# RWKV Architecture

## 1. Core Concept

**RWKV = "Receptance Weighted Key Value"**

- RNN-like inference: O(1) per token (constant memory)
- Transformer-like training: Parallelizable
- No attention matrix: Avoids O(n²) complexity

---

## 2. Delta-Rule State Update

The core mechanism:

```python
S_t = decay × S_{t-1} + k_t × v_t
output = r_t × S_t
```

- **S_t**: Running state (like RNN hidden state)
- **decay**: Learnable per-channel, controls memory retention
- **k_t, v_t**: Key and Value projections (ternary in ASNN-Goose)
- **r_t**: Receptance (sigmoid gate, stays continuous)

### Why Delta-Rule?
- Adds new information (k × v) to existing state
- Decay forgets old information gradually
- Receptance gates what to output

---

## 3. Time-Mixing (Attention Equivalent)

```python
# Interpolation between current and previous token
x_mix = lerp(x_prev, x_curr, ratio)

# Project mixed representation
k = W_k @ x_mix  # Key
v = W_v @ x_mix  # Value
r = sigmoid(W_r @ x_mix)  # Receptance (gate)

# Update state
state = decay * state + k * v

# Output
output = r * state
```

### In ASNN-Goose:
- K and V become **ternary spikes** {-1, 0, +1}
- R remains continuous (sigmoid)
- State update becomes add/subtract/do-nothing

---

## 4. Channel-Mixing (FFN Equivalent)

```python
x_mix = lerp(x_prev, x_curr, ratio)
k = W_k @ x_mix
v = W_v @ (silu(k) * k)  # Gated FFN
```

- Similar to transformer FFN with gating
- SiLU (Swish) activation for non-linearity
- 4x or 2x expansion factor

---

## 5. ARWKV: Transformer-to-RWKV Distillation (2025)

**Highly relevant paper for ASNN-Goose!**

### Three-Stage Training:
1. **Attention alignment**: Transform attention patterns → RNN attention
2. **Knowledge distillation**: Transfer capabilities via KL divergence
3. **SFT + DPO**: Fine-tuning for specific tasks

### Results:
- Distilled from Qwen 2.5 (32B) to RWKV 7B
- MMLU: 62.41 after stage 2
- WinoGrande: 68.67

### Key Findings:
- Word-level KL-Divergence > sequence-level
- Attention alignment critical for architecture mismatch
- KD without gating is suboptimal

**Citation**: https://arxiv.org/abs/2501.15570

---

## 6. RWKV Versions

| Version | Key Feature |
|---------|-------------|
| RWKV-4 | Original design |
| RWKV-5 | Improved time-mixing |
| RWKV-6 | Better long-range modeling |
| RWKV-7 | Native attention option |

### ASNN-Goose Backbone:
Based on simplified RWKV-style recurrence with:
- Delta-rule state updates
- 4 layers (configurable)
- 256 hidden dim (v8), 320 (v10)
- No position encoding needed (implicit in RNN)

---

## 7. Comparison: Transformer vs RWKV

| Aspect | Transformer | RWKV |
|--------|-------------|------|
| Memory per token | O(n) | O(1) |
| Compute per token | O(n) | O(1) |
| Training | Parallel | Parallel |
| Inference | Sequential | Sequential |
| Long context | Limited | Theoretically unlimited |
| Attention pattern | Explicit | Implicit in state |

### Why RWKV for ASNN-Goose?
- Constant memory enables edge deployment
- No attention matrix to quantize
- State update naturally maps to add/subtract operations

---

## 8. Implementation in ASNN-Goose

### GooseBackbone (Dense Teacher):
```python
class GooseRecurrentLayer(nn.Module):
    def __init__(self, d_model):
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.receptance_proj = nn.Linear(d_model, d_model)
        self.decay = nn.Parameter(torch.ones(d_model) * -0.5)

    def forward(self, x, state):
        k = self.key_proj(x)
        v = self.value_proj(x)
        r = torch.sigmoid(self.receptance_proj(x))

        decay = torch.sigmoid(self.decay)
        state = decay * state + k * v
        output = r * state

        return output, state
```

### SpikingGooseRecurrentLayer (Student):
```python
class SpikingGooseRecurrentLayer(nn.Module):
    def __init__(self, d_model):
        # Same projections
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.receptance_proj = nn.Linear(d_model, d_model)
        self.decay = nn.Parameter(torch.ones(d_model) * -0.5)

        # Ternary spike modules
        self.k_spike = TrainableTernarySpike()
        self.v_spike = TrainableTernarySpike()

    def forward(self, x, state):
        k_pre = self.key_proj(x)
        v_pre = self.value_proj(x)
        r = torch.sigmoid(self.receptance_proj(x))

        # Ternary quantization
        k = self.k_spike(k_pre)  # {-1, 0, +1}
        v = self.v_spike(v_pre)  # {-1, 0, +1}

        decay = torch.sigmoid(self.decay)
        state = decay * state + k * v  # Now: add, subtract, or do-nothing
        output = r * state

        return output, state
```

---

## 9. State Management

### Training:
- Process sequence in parallel
- State computed for each position
- Memory scales with sequence length

### Inference:
- Process token by token
- Maintain running state
- Memory is constant

### For Test-Time Training (TTT):
- LoRA adapters on K, V projections
- Only LoRA params updated
- State provides context without recomputation
