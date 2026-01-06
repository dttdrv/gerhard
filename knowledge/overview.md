# ASNN-Goose: Core Ideas

## What is ASNN?

**ASNN = Adapted Spiking Neural Network**

ASNN adapts spiking neural network principles for **current GPUs**, NOT neuromorphic hardware. The key insight: ternary activations {-1, 0, +1} enable computational shortcuts that standard floating-point networks cannot exploit.

## The Efficiency Thesis

### Why Ternary Spikes?

Standard neural networks use floating-point multiplications:
```
output = weight × activation  # Expensive FP32 multiply
```

Ternary activations {-1, 0, +1} convert multiplications to simpler operations:
```
activation = +1  →  output = +weight     # Addition
activation = -1  →  output = -weight     # Subtraction
activation =  0  →  output = 0           # Skip entirely
```

### Theoretical Benefits

| Operation | Energy (45nm CMOS) | Speedup |
|-----------|-------------------|---------|
| FP32 MAC  | ~4.6 pJ           | 1x      |
| FP32 ADD  | ~0.9 pJ           | ~5x     |
| Skip (zero)| 0 pJ             | ∞       |

### The Sparsity Requirement

GPU sparse operations only benefit when sparsity > 75%. Current ASNN:
- Spike density: ~38% non-zero (62% zeros)
- Not sparse enough for GPU benefit YET
- Dense implementation used during quality development
- Sparse ops planned for v16 after quality goals met

## Architecture

### Teacher-Student Setup

```
Teacher: GPT-2 (124M params, 12 layers, d=768, Transformer)
         └── PPL: 44.6 (baseline quality)

Student: ASNN-Goose (74M params, 5 layers, d=768, RWKV-style)
         └── PPL: 306.89 (v14.3 best)
         └── Gap: 6.9x worse than teacher
```

### Student Architecture (RWKV + Ternary Spikes)

```python
# RWKV-style recurrence with ternary spikes
state = decay * state + k * v      # State update
output = r * state                  # Gated output

# K and V are TERNARY:
k = TernarySpike(key_projection(x))    # {-1, 0, +1}
v = TernarySpike(value_projection(x))  # {-1, 0, +1}
r = sigmoid(receptance(x))              # Continuous [0, 1]
```

Why RWKV instead of Transformer?
- Linear complexity O(n) vs quadratic O(n²)
- Recurrent state amenable to sparse updates
- No attention matrix = less memory

### Ternary Spike Generation

```python
class TernarySpike(nn.Module):
    def __init__(self, alpha_init=1.0):
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.amplitude = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        threshold = self.alpha * x.abs().mean()

        # Ternary quantization
        spikes = torch.zeros_like(x)
        spikes[x > threshold] = 1.0
        spikes[x < -threshold] = -1.0

        # Straight-through estimator for gradients
        return self.amplitude * (x - x.detach() + spikes.detach())
```

## Key Innovations

### 1. CTKD: Curriculum Temperature Knowledge Distillation

**Problem**: Fixed temperature in KL divergence limits distillation quality.

**Solution**: Learnable temperature with adversarial optimization:
- Student minimizes KL loss (normal gradients)
- Temperature maximizes KL loss via Gradient Reversal Layer
- Creates min-max game: `min_θS max_τ L_KD(τ)`

**Result**: Temperature self-tunes from 2.0 → ~1.5, finding optimal difficulty.

### 2. FDD: Feature Dynamics Distillation

**Problem**: Architecture mismatch (Transformer → RWKV) means hidden states don't align.

**Solution**: Align HOW features change, not WHAT they are:
```python
# Feature dynamics (velocity through layers)
delta_student = h[l+1] - h[l]   # Student layer change
delta_teacher = h[l+1] - h[l]   # Teacher layer change

# Align dynamics using CKA (projector-free)
loss = cka_loss(delta_student, delta_teacher)
```

**Result**: Teaches student to transform features like teacher, despite different architectures.

### 3. SpikingBrain: Information Encoding Validation

**Problem**: Are ternary spikes actually encoding information, or just arbitrary quantization?

**Solution**: Validate spike patterns carry semantic meaning:
- Mutual Information: `I(spikes; teacher_hidden)` > 0.1
- CKA similarity between spike and teacher representations > 0.3
- Dead neuron detection (< 5% always zero)
- Saturated neuron detection (< 10% always ±1)

**Purpose**: Prerequisite for sparse ops - no point optimizing garbage spikes.

## Information Capacity

### Why Ternary > Binary

| Type    | States      | Entropy (bits) | Combinations (d=768) |
|---------|-------------|----------------|----------------------|
| Binary  | {0, 1}      | 1.0            | 2^768                |
| Ternary | {-1, 0, +1} | log₂(3) ≈ 1.58 | 3^768                |

Ternary provides 58% more information per neuron.

### The "Ternary Tax"

Ternary quantization loses information compared to continuous activations. This manifests as a PPL gap:
- Dense 74M model: PPL ~60-80 (estimated)
- Ternary 74M model: PPL 306.89 (v14.3)
- Gap: ~4-5x - the cost of discrete spikes

Reducing this gap is the core research challenge.

## Version History Summary

| Version | PPL | Key Innovation |
|---------|-----|----------------|
| v6 | 627 | First working distillation |
| v10 | 514 | Increased capacity (320d, 5L) |
| v12.1 | 446 | CTKD with GRL |
| v13.1 | 434 | Extended training |
| v14 | 425 | FDD with CKA |
| v14.1 | 321 | d_model=512 breakthrough |
| **v14.3** | **307** | **d_model=768 (current best)** |

## Current Status

**v14.3 Achieved**:
- PPL: 306.89 (best ever)
- Params: 74M (d=768, 5 layers)
- Gap to teacher: 6.9x (was 14x at v6)
- Spike density: 38.2%

**v15 In Progress**:
- SpikingBrain validation
- Validating spike information encoding
- Prerequisite for sparse ops

## Research Goals

### Short-term (v15-v17)
1. Validate spikes encode meaningful information
2. Implement sparse operations
3. Measure actual efficiency gains

### Long-term (v18-v19)
1. Controlled ablation studies
2. Publication-ready results
3. Reproducibility package

## Key Constraints

1. **Ternary activations are non-negotiable** - core thesis
2. **Quality before efficiency** - can't claim speedup on garbage output
3. **GPU-focused** - not targeting neuromorphic hardware
4. **Kaggle T4 compatible** - 16GB VRAM limit
