# Spiking Neural Networks & Ternary Quantization

## 1. LIF Neuron Fundamentals

### Leaky Integrate-and-Fire Dynamics
```
τ_m × du(t)/dt = -(u(t) - u_rest) + R × I(t)
```

### Discrete-Time Implementation
```python
# Membrane potential update
u[t] = β × u[t-1] × (1 - s[t-1]) + Σ w_ij × s_j[t]

# Spike generation
s[t] = Θ(u[t] - V_th)  # Heaviside step function
```

**Key Parameters:**
| Parameter | Description | Effect |
|-----------|-------------|--------|
| β (decay) | e^(-Δt/τ_m) | Close to 1 = long memory, close to 0 = fast forgetting |
| V_th | Firing threshold | Lower = more spikes, higher = sparser |

### Reset Mechanisms
| Type | Formula | Use Case |
|------|---------|----------|
| **Hard Reset** | u → 0 on spike | Simple, erases surplus |
| **Soft Reset** | u → u - V_th | Preserves surplus, **crucial for NLP** |

**For NLP**: Use soft reset to minimize information loss from quantization.

---

## 2. Ternary vs Binary Spikes

### Information Capacity
| Type | States | Entropy (bits) | Combinatorial (D=768) |
|------|--------|----------------|----------------------|
| Binary | {0, 1} | 1.0 | 2^768 |
| Ternary | {-1, 0, +1} | log₂(3) ≈ 1.58 | 3^768 |

**Ternary advantage**: ~58% more information per spike, **massive** combinatorial expansion.

### Trainable Amplitudes
Output: {-α, 0, +α} where α is learnable per layer

```python
class TernarySpike(nn.Module):
    def __init__(self, alpha_init=1.0):
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.amplitude = nn.Parameter(torch.ones(1))
```

**Inference optimization**: Absorb α into next layer weights:
```
W_next · (α·s) = (W_next·α) · s
```

---

## 3. Surrogate Gradient Functions

### The Problem
Spike generation uses Heaviside step function Θ(x), which has:
- ∂s/∂u = δ(u - V_th) → infinite at threshold, zero elsewhere
- **Dead gradient problem**: blocks error propagation

### Surrogate Functions

| Function | Formula | Characteristics |
|----------|---------|-----------------|
| **STE** | 1 if \|x-V_th\| < w/2, else 0 | Constant in window |
| **Triangular** | γ × max(0, 1-\|u-V_th\|/w) | Higher gradient near threshold |
| **SuperSpike** | 1/(1+k\|u-V_th\|)² | Infinite support, solves dead neurons |
| **ATan** | α/(2(1+(π/2·α(u-V_th))²)) | Heavy-tailed, good for Transformers |

### Implementation
```python
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, V_th):
        ctx.save_for_backward(u, V_th)
        return (u > V_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        u, V_th = ctx.saved_tensors
        # SuperSpike surrogate
        grad = 1 / (1 + 10 * (u - V_th).abs()) ** 2
        return grad_output * grad, None
```

### Learnable Surrogate Gradients (LSG)
- Steepness (γ or k) and width (w) as trainable parameters
- "Annealing": starts wide (exploration) → narrows (precision)

---

## 4. Trained Ternary Quantization (TTQ)

### Asymmetric Scaling
TTQ uses {-W_n, 0, +W_p} where W_n ≠ W_p:
```
w_t = +W_p  if w̃ > Δ
w_t = 0     if |w̃| ≤ Δ
w_t = -W_n  if w̃ < -Δ
```

### Gradient Computation
```python
# Gradient for scales (via chain rule)
∂L/∂W_p = Σ_{i∈I_p} ∂L/∂w_t,i  # Sum over positive weights
∂L/∂W_n = Σ_{i∈I_n} ∂L/∂w_t,i × (-1)  # Sum over negative weights
```

### Latent Weights (Shadow Copy)
- Maintain full-precision W̃ for gradient accumulation
- Forward: quantize W̃ → W_t
- Backward: gradients applied to W̃ (STE)

**Performance**: TTQ recovers ~5% accuracy over fixed-threshold TWN.

**Citation**: https://openreview.net/pdf?id=S1_pAu9xl

---

## 5. Channel-wise Ternary (TerViT)

### The Problem with Layer-wise
- Different channels have different feature scales
- Outlier channels force global α/Δ to be large
- "Quiet" channels get zeroed out entirely → **dead weights**

### Channel-wise Solution
```python
class ChannelWiseTernarySpike(nn.Module):
    def __init__(self, d_model, alpha_init=1.0):
        self.alpha = nn.Parameter(torch.ones(d_model) * alpha_init)  # Per-channel
        self.amplitude = nn.Parameter(torch.ones(d_model))  # Per-channel

    def forward(self, x):
        # x: [batch, seq, d_model]
        x_abs_mean = x.abs().mean(dim=(0, 1), keepdim=True)
        threshold = (self.alpha * x_abs_mean).clamp(min=0.01, max=10.0)

        with torch.no_grad():
            spike_signs = (x > threshold).float() - (x < -threshold).float()

        return self.amplitude * spike_signs + (x - x.detach())
```

### Parameter Calculation
- **α_c**: Channel-wise Absolute Mean = mean(|W_c|)
- **Δ_c**: 0.7 × ||W_c||₁ / N

### Storage Overhead
For C_out = 1024 channels:
- Layer-wise: 1 scale (4 bytes)
- Channel-wise: 1024 scales (4 KB)
- Overhead: <2% of total (negligible for accuracy gain)

### Results
| Model | Method | Granularity | Accuracy |
|-------|--------|-------------|----------|
| DeiT-S | TWN | Layer-wise | 72.9% |
| DeiT-S | TerViT | Channel-wise | 74.2% |
| DeiT-S | TerViT + KD | Channel-wise | 76.8% |
| Swin-S | TerViT | Channel-wise | 79.3% |

---

## 6. Threshold Management

### Statistics-based (Fixed)
```python
Δ = 0.7 × E(|W|)  # Optimal for Gaussian distribution
```
**Limitation**: Assumes symmetric distribution, fails for ReLU networks.

### Learned (LSQ)
```python
Δ = nn.Parameter(torch.tensor(0.05))  # Trainable
# Updated via gradient descent
```

### Soft Thresholding (STTN)
Probabilistic transition instead of hard step → gradients exist near threshold.

### Deadzone Management (TEQUILA)
**Problem**: Weights in (-Δ, Δ) receive zero gradients → "trapped dead weights"

**Solution**: Reactivation mechanism for dead weights:
- Secondary gradient pathway
- Allow dead weights to grow out of deadzone if relevant

**Result**: +3.8% on LLaMA-1B over standard ternary

---

## 7. Regularization Strategies

### Amplitude Regularization
Creates multi-modal landscape favoring ternary states:
```python
R_amp = Σ (|w| - α)² × |w|²
```
Minimal when |w| ≈ α (ternary states ±1) or w ≈ 0.

### Sparsity Regularization
```python
R_sparsity = β × Σ |w_ternary|
```

### Channel Amplitude Variance
Prevent channel amplitudes from diverging:
```python
channel_reg = 0.01 * amplitude.var()
```

### Entropy Regularization
Prevent layer collapse (all weights → 0):
```python
R_entropy = -Σ p(w) log p(w)
```

---

## 8. SNN Architectures for NLP

### SpikeBERT (Lv et al.)
- Based on Spikformer architecture
- Spiking Self-Attention (SSA): Q, K, V as spike trains
- Softmax replaced with spike-based normalization
- Two-stage distillation (general + task-specific)

**Performance**: 91.8% SST-2 (vs 92.1% BERT) at ~28% energy

### SpikingBERT (Bal & Sengupta)
- Implicit Differentiation at Equilibrium
- Views SNN as dynamical system converging to steady state
- **Memory**: O(L×T) → O(L)
- Enables deep Spiking Transformers

### SpikeGPT (Zhu et al.)
- **Uses RWKV backbone** (Linear Transformer)
- Up to 260M params
- Binary embedding via Token Shift + thresholding
- **22x less energy** than standard DL

**Citation**: https://arxiv.org/abs/2302.13939

---

## 9. Energy Efficiency

### MAC vs AC Operations
| Operation | Energy (45nm CMOS) |
|-----------|-------------------|
| 32-bit FP MAC | ~4.6 pJ |
| 32-bit FP AC | ~0.9 pJ |

**Theoretical**: ~5x reduction in compute energy.

### Complete Energy Model
```
E_SNN = T × [s_in × E_read + (1-s_out) × E_compute + ...]
```
- T: time steps (SNN penalty)
- s_in, s_out: sparsity (SNN advantage)

**Break-even**: For T=4, need >75% sparsity to beat ANN.

### Hardware Reality
- **GPU**: Cannot skip memory reads for zeros → often E_SNN > E_ANN
- **Neuromorphic** (Loihi, SpiNNaker): True event-driven → full savings

**SpikeBERT claim**: ~3.5x energy reduction with ~0.3% accuracy drop

---

## 10. SNN Training Strategies

### Progressive Quantization (Annealing)
1. **Phase 1**: Train FP32/INT8 (warm-up)
2. **Phase 2**: Soft ternary with low temperature
3. **Phase 3**: Hard ternary (τ → 0)

### Knowledge Distillation is Non-Negotiable
Without KD: >10% accuracy drop
With KD: Within 1-2% of FP32 performance

### Mixed Precision
Keep first/last layers in FP16/INT8:
- First layer: processes continuous input
- Last layer: produces class probabilities
- Middle layers: ternarize (>95% of parameters)

---

## 11. SNN Quality Metrics

### Mean Firing Rate (MFR)
```python
mfr = spikes.float().mean()  # Per layer
# Target: log-normal distribution
```

### Dead Neuron Detection
```python
dead_pct = (mfr < 0.001).float().mean()
# Alert if > 5%
```

### Surrogate Gradient Magnitude
If gradient → 0 while firing rate > 0: surrogate scale too small.

### Time-to-First-Spike (TTFS)
For latency-critical applications: measure decision speed.

---

## Application to ASNN-Goose

### Current Status (v11.1):
- Ternary activations {-1, 0, +1} on K, V projections
- Trainable amplitude per layer (not channel-wise)
- Adaptive threshold with learnable α
- Spike density: ~0.38 (healthy)

### v8 Amplitude Results:
```
layer_0: 0.74 (damped)  ← Early layer
layer_1: 1.08
layer_2: 1.05
layer_3: 1.07
```

### Recommended Roadmap
| Version | Feature | Impact |
|---------|---------|--------|
| v11.3 | Channel-wise α and amplitude | +1-2% accuracy |
| v12 | Soft thresholding | Better gradient flow |
| v13 | Deadzone reactivation | Prevent weight death |

### Key Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| α clamp | [0.01, 10.0] | Prevent degenerate thresholds |
| Amplitude target | [0.3, 3.0] | Healthy range |
| Channel reg weight | 0.01 | Prevent divergence |
| Dead neuron alert | >5% | Investigate if exceeded |
