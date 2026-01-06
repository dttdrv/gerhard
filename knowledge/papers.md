# Research Papers Summary

## Critical Papers for PPL Reduction

| Paper | Year | Key Contribution | Impact | Link |
|-------|------|------------------|--------|------|
| **ARWKV** | 2025 | Transformer→RWKV distillation | Direct methodology | [arxiv](https://arxiv.org/abs/2501.15570) |
| **POCL** | 2025 | Progressive curriculum distillation | Training stability | [arxiv](https://arxiv.org/abs/2506.05695) |
| **FDD** | 2025 | Feature dynamics alignment | Intermediate matching | ACL 2025.acl-long.1125 |
| **CTKD** | 2022 | Learnable temperature via GRL | Adaptive difficulty | [arxiv](https://arxiv.org/abs/2211.16231) |
| **TinyBERT** | 2020 | Two-stage multi-loss distillation | Hidden state alignment | [arxiv](https://arxiv.org/abs/1909.10351) |
| **MiniLM** | 2020 | Value-relation distillation | Projector-free | arxiv |
| **SpikeLLM** | 2024 | Spiking LLM methodology | SNN-LLM bridge | [arxiv](https://arxiv.org/abs/2407.04752) |
| **SpikeGPT** | 2023 | RWKV-inspired spiking | Architecture precedent | [arxiv](https://arxiv.org/abs/2302.13939) |
| **TTQ** | 2017 | Trained ternary quantization | Asymmetric scaling | [openreview](https://openreview.net/pdf?id=S1_pAu9xl) |
| **TerViT** | 2023 | Channel-wise ternarization | Per-channel scales | arxiv |
| **TEQUILA** | 2024 | Deadzone reactivation | LLM ternary fix | arxiv |
| **AdaKD** | 2024 | Token-adaptive temperature | Fine-grained control | arxiv |
| **Patient KD** | 2022 | Extended training benefits | Training duration | CVPR 2022 |

---

## 1. ARWKV (2025) - MOST RELEVANT

**Title**: ARWKV: Attention-free RWKV with Alignment

**Key Contribution**: First successful distillation of large transformer LLMs into RWKV-style models.

**Method**:
1. Attention alignment stage
2. Knowledge distillation stage
3. SFT + DPO fine-tuning

**Results**:
- Qwen 2.5 (32B) → RWKV 7B
- MMLU: 62.41
- WinoGrande: 68.67

**Key Insights for ASNN-Goose**:
- Word-level KL-Divergence > sequence-level
- Attention alignment critical for architecture mismatch
- Gating mechanisms important during distillation

**Link**: https://arxiv.org/abs/2501.15570

---

## 2. SpikeGPT (2023)

**Title**: SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

**Key Contribution**: Largest backprop-trained SNN for language modeling (45M-216M params).

**Architecture**:
- Based on RWKV design
- Binary spiking activations
- Event-driven computation

**Results**:
- 22x less energy than standard DL
- 20x fewer operations on neuromorphic hardware

**Relevance**: ASNN-Goose uses similar approach with ternary instead of binary spikes.

**Link**: https://arxiv.org/abs/2302.13939

---

## 3. TinyBERT (2020)

**Title**: TinyBERT: Distilling BERT for Natural Language Understanding

**Method**:
- Two-stage distillation (pre-training + task-specific)
- Multi-layer loss (embedding, hidden, attention, prediction)

**Results**:
- 96.8% of BERT performance
- 7.5x smaller
- 9.4x faster

**Key Insight**: Multi-component loss outperforms KL-only distillation.

**Link**: https://arxiv.org/abs/1909.10351

---

## 4. CTKD (2022) - CRITICAL FOR ASNN-GOOSE ✅ IMPLEMENTED

**Title**: Curriculum Temperature for Knowledge Distillation

**Key Contribution**: Transform temperature from fixed hyperparameter to adversarially-learned variable.

### 🎉 ASNN-Goose v12.1 Implementation SUCCESS

| Metric | Before (v10) | After (v12.1) | Improvement |
|--------|--------------|---------------|-------------|
| PPL | 514.5 | **445.61** | -68.89 (13.4%) |
| Temperature | Fixed 2.0 | Learned 1.58 | Optimal found |
| Tests | N/A | 10/10 | All passed |

**What We Learned**:
- GRL is REQUIRED - simple regularization causes temperature runaway (v12 failed)
- Sigmoid bounding [1.0, 5.0] better than clamp [1.0, 10.0] for LLM distillation
- Temperature naturally decreased from 2.0 → 1.58 (opposite to paper's CV finding)
- Lambda warmup (20%) critical for stable training

### The Adversarial Min-Max Game
```
min_{θ_S} max_{θ_A} L_KD(P_T(τ), P_S(τ))
```
- **Student (S)**: Minimizes distillation loss
- **Temperature Module (A)**: Maximizes it (finds hardest temperature)

### Gradient Reversal Layer (GRL)
- **Forward Pass**: Identity mapping `R(x) = x`
- **Backward Pass**: Reverses gradient `dR/dx = -λI`
- Allows simultaneous min-max optimization in single backprop

### Our Implementation
```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()  # Identity

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None  # Negate!

class CTKDTemperature(nn.Module):
    def __init__(self, tau_min=1.0, tau_max=5.0, init=2.0):
        # Sigmoid bounding: tau = tau_min + tau_range * sigmoid(raw)
        self.grl = GradientReversalLayer()

    def forward(self, lambda_):
        raw_reversed = self.grl(self.raw_temp)  # GRL applied here!
        return self.tau_min + self.tau_range * torch.sigmoid(raw_reversed)
```

### Temperature Bounding
```python
τ = τ_min + τ_range × sigmoid(raw_temp)  # τ ∈ [1.0, 5.0] for LLM
```
**Note**: Paper uses [1, 21] for image classification; we found [1, 5] works better for LLM distillation.

### Temperature Evolution (Observed in v12.1)
```
Step    0 (  0%): T=2.0000  (initial)
Step  600 ( 20%): T=2.0014  (warmup phase, λ≈0)
Step 1500 ( 50%): T=1.6422  (GRL active, T decreasing)
Step 2999 (100%): T=1.5751  (stabilized at optimal)
```

**Our Finding**: For RWKV-style LLM distillation, temperature DECREASED over training (2.0 → 1.58), opposite to CTKD paper's vision finding. This suggests sharper targets work better for LLM KD.

### λ Scheduling (Adversarial Strength)
Cosine schedule: weak adversary early, strong late
- Early: λ ≈ 0 (temperature learns freely)
- Late: λ → 1 (full adversarial pressure)

```python
def get_lambda(step, total_steps, lambda_max=1.0, warmup_ratio=0.2):
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return 0.0  # No adversarial during warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lambda_max * (1 - math.cos(math.pi * progress)) / 2
```

### Results
| Dataset | Improvement over Vanilla KD |
|---------|----------------------------|
| CIFAR-100 | +0.53 to +1.68% |
| ImageNet | 71.32% vs 70.66% |
| **ASNN-Goose** | **PPL 514.5 → 445.61 (13.4%)** |

### CRITICAL LESSONS LEARNED

**V12 Failure (No GRL)**:
Without GRL, optimizer pushes T to max clamp:
- Higher T = softer softmax = superficially lower KL loss
- But T=10 produces meaningless gradients for actual learning
- **ALWAYS regularize**: `temp_reg_loss = 0.1 × (T - anchor)²`

**Link**: https://arxiv.org/abs/2211.16231

---

## 5. POCL (2025) - PROGRESSIVE TRAINING

**Title**: Progressive Overload Curriculum Learning for Knowledge Distillation

### Progressive Overload Principle
Borrowed from physiology (strength training):
- **Volume** (dataset size) must increase over time
- **Intensity** (difficulty/temperature) must increase over time
- Counter to traditional "annealing" approaches

### Difficulty Measurer (RRF)
Reciprocal Rank Fusion combining two signals:
```
Score(x) = 1/(k + r_Loss) + 1/(k + r_InvROUGE)
```
- **r_Loss**: High loss = hard sample
- **r_InvROUGE**: Low ROUGE = hard sample

### Baby Step Scheduler
| Phase | Data Volume | Temperature | α (KD weight) |
|-------|-------------|-------------|---------------|
| 1 (Easy) | 25-33% | 1.0 | 0.9 (mostly SFT) |
| 2 (Medium) | 50-70% | 1.5 | 0.7 |
| 3 (Full) | 100% | 2.0 | 0.5 (balanced) |

### Rising Temperature Schedule
```
τ(t) = τ_min + (τ_max - τ_min) × (t / T_total)
```
- **Principle**: "Weakness requires Sharpness"
- Early: Low T (sharp targets for syntax)
- Late: High T (soft targets for nuance)

### Stage Transitions
- **Time-based**: Fixed percentages (40%, 70%, 100%)
- **Loss-based**: Move when dL/dt < ε (loss plateaus)
- **Validation-based**: When val PPL stops improving

### Efficiency
- **39% training runtime reduction** (early epochs on small subsets)
- Superior ROUGE-L scores vs full-batch baselines

**Link**: https://arxiv.org/abs/2506.05695

---

## 6. SpikeLLM (2024)

**Title**: SpikeLLM: Spiking Large Language Models

**Key Contribution**: First spiking LLM at 7B-70B scale.

**Method**:
- Bio-plausible spiking mechanisms
- Saliency-based neuron activation

**Results**:
- 11.01% WikiText2 PPL reduction (in OmniQuant pipeline)
- 2.55% reasoning improvement

**Link**: https://arxiv.org/abs/2407.04752

---

## 7. TTQ (2017) - TERNARY QUANTIZATION

**Title**: Trained Ternary Quantization

**Key Contribution**: Asymmetric scaling with learned positive/negative factors.

### Asymmetric Scaling
Standard: `{-α, 0, +α}` (symmetric)
TTQ: `{-W_n, 0, +W_p}` where `W_n ≠ W_p`
```
w_t = +W_p  if w̃ > Δ
w_t = 0     if |w̃| ≤ Δ
w_t = -W_n  if w̃ < -Δ
```

### Latent Weights (Shadow Copy)
- Maintain full-precision W̃ for gradient accumulation
- Forward: quantize W̃ → W_t
- Backward: gradients applied to W̃ (STE)

### Gradient Computation
```python
# Gradient for scales (via chain rule)
∂L/∂W_p = Σ_{i∈I_p} ∂L/∂w_t,i  # Sum over positive weights
∂L/∂W_n = Σ_{i∈I_n} ∂L/∂w_t,i × (-1)  # Sum over negative
```

### Inference Optimization
Scale factors absorbed into next layer:
```
W_next · (α·s) = (W_next·α) · s
```
Runtime remains accumulator-only (no MAC).

### Results
| Model | Method | Top-1 Accuracy |
|-------|--------|----------------|
| ResNet-18 | FP32 | 69.7% |
| ResNet-18 | TWN (fixed) | 61.8% |
| ResNet-18 | TTQ | 66.6% |

**Performance**: TTQ recovers ~5% accuracy over fixed-threshold TWN.

**Link**: https://openreview.net/pdf?id=S1_pAu9xl

---

## 8. Feature Dynamics Distillation (2025)

**Title**: Beyond Logits: Aligning Feature Dynamics for Knowledge Distillation

**Key Contribution**: View transformers as ODEs, align feature trajectories.

**Method**:
- Match feature evolution path
- Match first-order derivatives
- Layer selection less important than thought

**Key Insight**: Trajectory alignment > point-wise alignment

**Link**: ACL 2025.acl-long.1125

---

## 9. Multi-Teacher KD (2023)

**Title**: Multi-teacher Knowledge Distillation as Ensemble Compression

**Method**:
- Instance-level teacher weighting
- Adaptive teacher selection
- Multi-level hints

**Key Insight**: Different teachers provide complementary knowledge.

**Link**: https://arxiv.org/abs/2302.07215

---

## 10. Patient KD (2022)

**Title**: Knowledge Distillation: A Good Teacher Is Patient and Consistent

**Key Finding**: Extended training with KD beats short training with complex techniques.

**Results**:
- ResNet-50: 82.8% ImageNet (4.4% improvement)
- Consistency in augmentation critical

**Key Insight**: Train longer, not harder.

**Link**: CVPR 2022

---

## 11. TerViT (2023) - CHANNEL-WISE QUANTIZATION

**Title**: Ternary Vision Transformers

**Problem**: Layer-wise quantization causes "dead weights" in ViTs
- Outlier channels force global α/Δ to be large
- "Quiet" channels zeroed out entirely

### Channel-wise Solution
```python
# Per-channel parameters instead of per-layer
α_c = mean(|W_c|)  # Channel-wise Absolute Mean
Δ_c = 0.7 × ||W_c||₁ / N
```

### Storage Overhead
- Layer-wise: 1 scale (4 bytes)
- Channel-wise: C_out scales (4 KB for 1024 channels)
- Overhead: <2% (negligible for accuracy gain)

### Results
| Model | Method | Granularity | Accuracy |
|-------|--------|-------------|----------|
| DeiT-S | TWN | Layer-wise | 72.9% |
| DeiT-S | TerViT | Channel-wise | 74.2% |
| DeiT-S | TerViT + KD | Channel-wise | 76.8% |
| Swin-S | TerViT | Channel-wise | 79.3% |

---

## 12. TEQUILA (2024) - DEADZONE FIX

**Title**: Trapping-free TErnary QUantization for LLMs

**Problem**: Weights in deadzone (-Δ, Δ) receive zero gradients → "trapped dead weights"

### The Deadzone Trapping Problem
- In LLMs, massive deadzone exists
- Weights initialized in zone never escape
- Huge portion of parameters become useless

### Solution: Reactivation Mechanism
- Secondary gradient pathway for dead weights
- Allows dead weights to grow out of deadzone if relevant
- Dynamic bias repurposing

### Results
| Model | Method | Avg Accuracy |
|-------|--------|--------------|
| LLaMA-1B | Standard Ternary | 38.6% |
| LLaMA-1B | TEQUILA | 42.4% |

**Improvement**: +3.8% on LLaMA-1B over standard ternary

---

## 13. AdaKD (2024) - TOKEN-ADAPTIVE DISTILLATION

**Title**: Adaptive Knowledge Distillation with Token-level Temperature

### Inverse Difficulty Temperature Scaling (IDTS)
```python
τ_t = F(difficulty_score_t)  # Per-token temperature
```

**Logic**:
- **Hard Tokens** (high error): Low T → sharp, corrective signal
- **Easy Tokens** (low error): High T → focus on dark knowledge

### Key Insight
Both POCL and AdaKD agree: **"Weakness requires Sharpness"**
- Weak on sample/token → needs Low T (guidance)
- Strong on sample/token → can handle High T (nuance)

---

## 14. MiniLM (2020) - PROJECTOR-FREE DISTILLATION

**Title**: MiniLM: Deep Self-Attention Distillation

**Key Innovation**: Distill only last layer's relational structures.

### Value-Relation Loss
```
R_V = Softmax(V V^T / √d_k)
L_VR = KL(R_V_teacher || R_V_student)
```
- Measures how token content relates to each other
- Independent of hidden dimension → no projector needed!

### Advantages
- No layer mapping problem
- Arbitrary student depth/width
- Task-agnostic compression

---

## 15. SpikingBERT Variants

### SpikeBERT (Lv et al.)
- Based on **Spikformer** architecture
- Spiking Self-Attention (SSA)
- Two-stage distillation (general + task-specific)
- Performance: 91.8% SST-2 at ~28% energy

### SpikingBERT (Bal & Sengupta)
- Uses **Implicit Differentiation at Equilibrium**
- Views SNN as dynamical system at steady state
- Memory: O(L×T) → O(L)
- Enables deep Spiking Transformers

---

## 16. Hidden State Alignment Theory

### Layer Mapping Strategies
| Strategy | Formula | Best For |
|----------|---------|----------|
| Uniform | φ(j) = j × (N/M) | General purpose |
| PKD-Skip | Every k-th layer | Large depth ratio |
| PKD-Last | Last M layers only | Semantic-focused |
| LAD (Learned) | Gated combination | Adaptive |

### Dimension Mismatch Solutions
| Method | Description | Overhead |
|--------|-------------|----------|
| Linear Projection | W_proj ∈ ℝ^(d_T × d_S) | Low |
| CKA (Projector-free) | Gram matrix similarity | None |

### CKA Formula
```
CKA(K_S, K_T) = tr(K_S K_T^T) / (||K_S||_F ||K_T||_F)
```
where K = X X^T (Gram matrix)

### Failure Modes
| Mode | Symptom | Mitigation |
|------|---------|------------|
| Gradient Conflict | Task vs Align opposite | DOT, Dual-Head KD |
| Projector Overfitting | Align→0, Task stalls | Linear projectors only |
| Negative Transfer | Student worse with align | Teacher Assistant (TA) |

---

## Additional Resources

### Spiking Neural Networks:
- SpikeLM (2024): https://arxiv.org/abs/2406.03287
- Spiking Vision Transformers: https://arxiv.org/abs/2210.06686
- MOHAWK (Mamba-2 distillation): Transformer→SSM

### Quantization:
- TerViT: Channel-wise ternary ViT
- TerDiT: Ternary diffusion transformers (RMS Norm critical)
- LSQ/LSQ+: Learned Step Size Quantization

### RWKV:
- Original RWKV: https://arxiv.org/abs/2305.13048
- RWKV-6/7 updates on GitHub
- Generalized Delta Rule (Goose)

### Distillation Surveys:
- KD for LLMs Survey: https://arxiv.org/abs/2402.13116
- Comprehensive KD Survey: https://arxiv.org/abs/2503.12067

### Surrogate Gradients:
- SuperSpike: 1/(1+k|u-V_th|)²
- ATan: Heavy-tailed, good for Transformers
- Learnable Surrogate Gradients (LSG)
