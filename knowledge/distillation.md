# Knowledge Distillation Techniques

## 1. Core Distillation Loss (Hinton et al.)

```
L_kl = T² × KL(softmax(teacher_logits/T) || softmax(student_logits/T))
```

### Mathematical Foundation
- **T (temperature)**: Higher T → softer targets → reveals inter-class relationships ("dark knowledge")
- **T² scaling**: Compensates for gradient magnitude reduction (gradients scale as 1/T²)
- **As T→∞**: KL divergence → MSE logit matching
- **T effectively interpolates**: probability matching (low T) ↔ logit regression (high T)

### KL Direction Matters
- **Forward KL** `D_KL(P_teacher || Q_student)`: Mean-seeking, forces student to "cover" teacher distribution
- **Reverse KL** `D_KL(Q_student || P_teacher)`: Mode-seeking, student locks onto dominant mode
- **For SNNs**: Reverse KL can be more stable (prevents spurious firing in low-probability regions)

### Recommended T by Task
| Task | Optimal T Range |
|------|-----------------|
| Image Classification | 3.0 - 20.0 |
| Language Modeling | 2.0 - 4.0 |
| SNNs | 2.0 - 6.0 |

---

## 2. Curriculum Temperature (CTKD - ArXiv 2211.16231)

### Core Innovation
CTKD makes temperature **learnable** via an adversarial min-max game:
- **Student**: Minimizes distillation loss
- **Temperature module**: Maximizes it (finds hardest temperature)
- Uses **Gradient Reversal Layer (GRL)** for adversarial optimization

### Implementation
```python
class LearnableTemperature(nn.Module):
    def __init__(self, init=2.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init)))

    def forward(self):
        return torch.exp(self.log_temp).clamp(1.0, 10.0)
```

### Temperature Bounding
```
τ = τ_init + τ_range × sigmoid(T_pred)
```
- Typical range: τ ∈ [1, 21] with τ_init=1, τ_range=20

### "Reverse Annealing" Phenomenon
**Critical insight**: CTKD tends to INCREASE temperature over time (contrary to intuition)
- **Early training** (low T): Sharp targets → clear directional signal
- **Late training** (high T): Flat targets → forces learning nuance
- Principle: **"Specificity precedes nuance"**

### λ Scheduling (Adversarial Strength)
- Cosine schedule for curriculum: weak adversary early, strong late
- Early: λ ≈ 0 (temperature learns freely)
- Late: λ → 1 (full adversarial pressure)

### Performance
| Dataset | Improvement over Vanilla KD |
|---------|----------------------------|
| CIFAR-100 | +0.53 to +1.68% |
| ImageNet | 71.32% vs 70.66% |

### CRITICAL: Temperature Regularization
**Lesson from v11.1 failure**: Without regularization, optimizer pushes T to max clamp (higher T = superficially lower KL loss, but meaningless gradients)

```python
# ALWAYS regularize temperature towards an anchor
if temp_module is not None:
    anchor_temp = cfg.temperature  # e.g., 2.0
    temp_reg_loss = 0.1 * (T - anchor_temp) ** 2
```

**Citation**: https://arxiv.org/abs/2211.16231

---

## 3. Progressive Distillation (POCL - ArXiv 2506.05695)

### Progressive Overload Principle
Borrowed from physiology (strength training):
- **Volume** (dataset size) AND **Intensity** (difficulty/temperature) must increase over time
- Counter to traditional "annealing" approaches

### Difficulty Measurer (RRF)
Uses Reciprocal Rank Fusion combining:
```
Score(x) = 1/(k + r_Loss) + 1/(k + r_InvROUGE)
```
- **r_Loss**: Cross-entropy loss rank (high loss = hard)
- **r_InvROUGE**: Inverse ROUGE-L rank (low ROUGE = hard)

### Baby Step Scheduler
Cumulative partitioning:
| Phase | Data Volume | Temperature | α (KD weight) |
|-------|-------------|-------------|---------------|
| 1 (Easy) | 25-33% | 1.0 | 0.9 (mostly SFT) |
| 2 (Medium) | 50-70% | 1.5 | 0.7 |
| 3 (Full) | 100% | 2.0 | 0.5 (balanced) |

### Rising Temperature Schedule
```
τ(t) = τ_min + (τ_max - τ_min) × (t / T_total)
```
- Early: Low T (sharp targets for syntax acquisition)
- Late: High T (soft targets for semantic nuance)
- **"Weakness requires Sharpness"**

### Stage Transition Mechanisms
- **Time-based**: Fixed percentages (40%, 70%, 100%)
- **Loss-based**: Move when dL/dt < ε (loss plateaus)
- **Validation-based**: When val PPL stops improving

### Efficiency
- **39% training runtime reduction** (early epochs on small subsets)

**Citation**: https://arxiv.org/abs/2506.05695

---

## 4. Hidden State Alignment (TinyBERT - ArXiv 1909.10351)

### Theoretical Motivation
- **Vanishing guidance problem**: With logit-only distillation, error signal attenuates through deep networks
- Hidden states provide **deep supervision** - short-circuiting backprop
- Teacher's hidden states at layer l provide "scaffolding" for layer l+1

### Layer Mapping Strategies

| Strategy | Formula | Best For |
|----------|---------|----------|
| **Uniform** | φ(j) = j × (N/M) | General purpose |
| **PKD-Skip** | Every k-th layer | Large depth ratio |
| **PKD-Last** | Last M layers only | Semantic-focused |
| **LAD (Learned)** | Gated combination | Adaptive |
| **MiniLM** | Last layer only | Task-agnostic |

For 4-layer student → 12-layer teacher:
- Uniform: {1→3, 2→6, 3→9, 4→12}

### Dimension Mismatch Solutions

| Method | Description | Overhead |
|--------|-------------|----------|
| **Linear Projection** | W_proj ∈ ℝ^(d_T × d_S) | Low |
| **MLP Projection** | Non-linear mapping | Risk of overfitting |
| **CKA (Projector-free)** | Gram matrix similarity | None |

**CKA Formula**:
```
CKA(K_S, K_T) = tr(K_S K_T^T) / (||K_S||_F ||K_T||_F)
```
where K_S = X_S X_S^T (N×N Gram matrix)

### Loss Functions

| Loss | Formula | Best For |
|------|---------|----------|
| **MSE** | \|\|proj(h_s) - h_t\|\|² | Hidden states |
| **Cosine** | 1 - cos(h_s, h_t) | Embeddings |
| **KL** | KL(A_t \|\| A_s) | Attention matrices |
| **Value-Relation** | KL(Softmax(VV^T/√d)) | MiniLM-style |

### Loss Weighting Guidelines
- **TinyBERT**: λ_hidden = 1.0 (aggressive)
- **Production heuristic**: λ_hidden ∈ [0.01, 0.1] as regularizer
- **Start low**: λ = 0.01, increase if stable

### Failure Modes

| Mode | Symptom | Mitigation |
|------|---------|------------|
| **Gradient Conflict** | Task vs Align pulling opposite | DOT, Dual-Head KD |
| **Projector Overfitting** | Align loss → 0, Task stalls | Linear projectors only |
| **Negative Transfer** | Student worse with alignment | Teacher Assistant (TA) |

**Citation**: https://arxiv.org/abs/1909.10351

---

## 5. Two-Stage Distillation (SpikeBERT)

### Stage 1: General Distillation (Pre-training Transfer)
- **Teacher**: Pre-trained BERT
- **Data**: Large unlabeled corpora
- **Method**: Layer-wise feature distillation
- **Alignment**: SNN features averaged over time: h̄ = (1/T)Σh[t]
- **Goal**: Imprint syntax and semantic structure

### Stage 2: Task-Specific Distillation
- **Teacher**: Fine-tuned BERT on specific task
- **Data**: Task-specific labeled data
- **Method**: Logit-based distillation
- **Goal**: Refine firing thresholds for classification

### Critical Finding
Removing Stage 1 → **>10% accuracy drop**

**Citation**: Lv et al. (SpikeBERT)

---

## 6. DistilBERT (Sanh et al., 2019)

- 40% smaller than BERT
- 60% faster inference
- Retains 97% of BERT capabilities
- Uses triple loss: language modeling + distillation + cosine embedding

---

## 7. Multi-Teacher Distillation

### AMTML-KD
- Learns instance-level teacher importance weights
- Adaptively selects relevant teachers per example
- Gathers hints from multiple intermediate layers

**Citation**: https://arxiv.org/abs/2103.04062

### DIVERSEDISTILL
- Addresses teacher heterogeneity
- Dynamic weighting based on student understanding

---

## 8. Patient Knowledge Distillation (CVPR 2022)

- Training longer outperforms aggressive schedules
- Consistency in data augmentation crucial
- ResNet-50 achieved 82.8% ImageNet (4.4% improvement)

**Key insight**: Extended training with KD beats short training with complex techniques.

---

## 9. Self-Distillation (2019)

- Model distills knowledge within itself
- Attaches classifiers at different depths
- Distills from deepest to shallower layers
- 3.49% accuracy boost on CIFAR-100

**Citation**: https://arxiv.org/abs/1905.08094

---

## 10. Feature Dynamics Distillation (FDD - ACL 2025)

- Views transformers as ODEs discretized over layers
- Aligns entire feature trajectory, not just final states
- Matches first-order derivatives of feature evolution
- Layer selection strategy matters less than thought

**Citation**: ACL 2025.acl-long.1125

---

## Application to ASNN-Goose

### Current Implementation (v11.1):
- KL divergence loss with learnable T (CTKD)
- Temperature regularization towards anchor (T=2.0)
- Hidden alignment disabled (weight=0.0)
- 3000 steps distillation

### Lessons Learned
1. **Temperature runaway**: Without regularization, T→max → PPL stuck at ~1300
2. **torch.no_grad vs torch.inference_mode**: Use no_grad when downstream ops need gradients (like learnable T)
3. **Regularization is mandatory**: temp_reg_loss = 0.1 × (T - anchor)²

### Recommended Roadmap
| Version | Technique | Status |
|---------|-----------|--------|
| v11.1 | Learnable Temperature | In Progress |
| v11.2 | + Progressive Stages | Planned |
| v11.3 | + Channel-wise Spikes | Planned |
| v12 | + Hidden Alignment | Planned |
| v13 | Extended Training (5000+ steps) | Planned |

### Key Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature LR | 0.01 | Separate from main LR |
| Temp regularization | 0.1 | Strength of anchor constraint |
| Hidden align weight | 0.01 | Start low |
| Stage transitions | 40%, 70% | For POCL |
