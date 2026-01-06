# ASNN-Goose: Comprehensive Context for External LLM Consultation

**Date**: 2026-01-04
**Version**: Post-v14 Results Analysis
**Purpose**: Seeking guidance on breaking the PPL ~425 plateau and achieving <400

---

## 1. PROJECT OVERVIEW

### 1.1 What is ASNN?

**ASNN = Adapted Spiking Neural Network**

**CRITICAL CLARIFICATION**: ASNN is NOT designed for neuromorphic hardware. It is adapted for **CURRENT GPUs**.

The core innovation is using ternary activations {-1, 0, +1} to enable:
- **Additions instead of multiplications**: `weight * (+1) = add`, `weight * (-1) = subtract`
- **Sparsity**: `weight * 0 = skip computation entirely`
- **Information encoding in spike patterns**: The pattern of {-1, 0, +1} carries semantic meaning

### 1.2 Architecture

```
Teacher: GPT-2 (124M params, 12 layers, d=768, Transformer)
         └── PPL: 44.6

Student: ASNN-Goose (22M params, 5 layers, d=320, RWKV-style)
         └── PPL: 424.81 (current best, v14)
         └── Compression: 5.6x
         └── Gap: 9.5x worse than teacher
```

**Student Architecture (RWKV-style with Ternary Spikes)**:
```python
# Core recurrence (delta-rule state update)
state = decay * state + k * v
output = r * state

# Where k and v are TERNARY SPIKES:
k = TernarySpike(key_projection(x))   # {-1, 0, +1}
v = TernarySpike(value_projection(x)) # {-1, 0, +1}
r = sigmoid(receptance(x))            # Continuous [0, 1]
```

### 1.3 Current Implementation State

**IMPORTANT**: Current implementation uses **DENSE ternary operations**, not sparse.

```python
# Current (dense):
k = self.spike_k(k_pre)      # Returns dense tensor with values {-1, 0, +1}
v = self.spike_v(v_pre)      # Returns dense tensor with values {-1, 0, +1}
new_S = decay * state + k * v  # Standard dense element-wise multiply
```

This is **intentional for now** because:
1. Focus is on PPL quality first (can't claim efficiency if model outputs garbage)
2. GPU sparse operations need >75% sparsity to be beneficial
3. Current spike density: ~37% non-zero (63% zeros) - not sparse enough for GPU benefit
4. Sparse implementation planned for v16 after quality goals are met

---

## 2. THE REAL PROBLEM

### 2.1 The Gap

| Model | PPL | Quality |
|-------|-----|---------|
| Teacher (GPT-2) | 44.6 | Usable |
| Student (v14) | 424.81 | **9.5x worse** |
| Target | <400 | Still 9x worse |
| Ideal | ~100-150 | 2-3x worse (acceptable?) |

**A PPL of 424 is catastrophically bad for practical use.** The model produces low-quality text.

### 2.2 The Bottleneck

From previous external LLM consultation (v13.1 analysis):
- PPL ~434 represents a "soft ceiling" with current approach
- **Architecture mismatch** (Transformer → RWKV) is key bottleneck
- Current distillation only aligns final logits (KL divergence)
- Student and teacher process information fundamentally differently
- Need to align internal representations, not just outputs

---

## 3. COMPLETE VERSION HISTORY

### 3.1 Summary Table

| Version | PPL | Delta | Status | Key Change |
|---------|-----|-------|--------|------------|
| v6 | 627.3 | - | Baseline | First working distillation |
| v7 | 1655 | +1028 | FAILED | hidden_align_weight=1.0 (too strong) |
| v8 | 559 | -68 | Fixed | Reset to safe defaults |
| v9 | 541.7 | -17 | OK | Capacity increase (320d, 5L) |
| v10 | 514.5 | -27 | Baseline | 320d/5L stable baseline |
| v11 | 512.67 | -2 | BUG | Channel-wise with regularization (suppressed learning) |
| v11.1 | 512.04 | -0.6 | FAILED | Channel-wise without reg (structural symmetry) |
| v12 | ~1000 | +500 | FAILED | Learnable temp without GRL (runaway) |
| v12.1 | 445.61 | -69 | SUCCESS | CTKD with GRL (breakthrough) |
| v13 | 1125.94 | +680 | FAILED | POCL curriculum (catastrophic) |
| v13.1 | 434.44 | -11 | SUCCESS | Extended training (5000 steps) |
| **v14** | **424.81** | **-9.6** | **PARTIAL** | FDD with CKA (improved but missed <400) |

### 3.2 Detailed Failure Analysis

#### v7 Failure: Hidden Alignment Too Strong
```python
hidden_align_weight = 1.0  # WAY too strong
# Gradients from alignment dominated KL loss
# Model learned to match hidden states but forgot language modeling
```
**Lesson**: Hidden alignment must be very light (0.001 or less)

#### v11.1 Failure: Structural Symmetry
```python
# K and V amplitudes stayed IDENTICAL (to 16 decimal places!)
# Both initialized to torch.ones(d_model)
# kv = k * v creates symmetric gradient: dL/dk ≈ dL/dv
# Adam state also identical → no symmetry breaking
```
**Lesson**: Channel-wise requires asymmetric initialization or pretrained weights

#### v12 Failure: Temperature Runaway
```python
# Without GRL, optimizer pushed T to maximum (10.0)
# Higher T = softer softmax = superficially lower KL loss
# But T=10 produces meaningless gradients
```
**Lesson**: CTKD requires adversarial min-max via Gradient Reversal Layer

#### v13 Failure: POCL Curriculum
```python
# Pre-training on full data, then restricted to "easy" 33%
# Caused catastrophic forgetting
# T=1.0 too sharp for SNN distillation
# Difficulty ranking from continuous models doesn't transfer to SNNs
```
**Lesson**: Curriculum learning doesn't work for SNN-LLM distillation

---

## 4. V14 IMPLEMENTATION DETAILS

### 4.1 Innovation: Feature Dynamics Distillation (FDD)

**Theory**: Instead of matching hidden states directly (failed in v7), match the DYNAMICS:
```
delta_h = h_{l+1} - h_l  # How features CHANGE through layers
```

This teaches the student HOW to transform features, not just WHAT features to have.

**Implementation**:
```python
def compute_fdd_loss(student_hiddens, teacher_hiddens, layer_map):
    for s_layer, t_layer in layer_map.items():
        # Compute dynamics (velocity)
        delta_s = student_hiddens[s_layer + 1] - student_hiddens[s_layer]
        delta_t = teacher_hiddens[t_layer + 1] - teacher_hiddens[t_layer]

        # Align using CKA (projector-free, dimension-agnostic)
        loss += cka_loss(delta_s, delta_t)

    return loss / n_pairs
```

### 4.2 CKA (Centered Kernel Alignment)

**Why CKA**:
- Projector-free (no extra parameters)
- Dimension-agnostic (student=320, teacher=768, no problem)
- Invariant to orthogonal transformations

**Critical Bug Found and Fixed**:
```python
# BEFORE (broken): Float16 overflow in mixed precision
hsic_xx = (K_X * K_X).sum()  # 4M elements, sum can exceed 65504

# AFTER (fixed): Force float32
with torch.cuda.amp.autocast(enabled=False):
    X = X.float()
    Y = Y.float()
    # Also added row normalization for additional stability
    X_norm = X_centered / (X_centered.norm(dim=1, keepdim=True) + eps)
```

### 4.3 Layer Mapping
```python
# Student (5 layers) → Teacher (12 layers)
layer_map = {0: 2, 2: 7, 4: 11}  # Early, middle, late
```

### 4.4 Safety Measures
1. **Very low weight**: `fdd_weight = 0.001` (v7 used 1.0 and failed)
2. **Warmup**: FDD disabled for first 500 steps
3. **Kill switch**: Disable FDD if PPL regresses >10%

---

## 5. V14 RESULTS

### 5.1 Final Metrics

| Metric | Value | vs v13.1 |
|--------|-------|----------|
| **PPL** | 424.81 | -9.63 (2.2% better) |
| Temperature | 1.49 | Similar (1.50) |
| Lambda | 1.0 | Same |
| Spike Density | 37.1% | Similar |
| Training | 4500 steps (early stopped) | 5000 |

### 5.2 Training Trajectory

```
Step  300: PPL 936.0  (starting)
Step  600: PPL 706.4  (FDD kicks in at 500)
Step  900: PPL 602.3
Step 1200: PPL 559.5
Step 1500: PPL 517.3
Step 1800: PPL 469.1
Step 2100: PPL 455.4
Step 2400: PPL 446.6
Step 2700: PPL 434.0  (matched v13.1!)
Step 3000: PPL 431.5
Step 3300: PPL 433.5  (plateau begins)
Step 3600: PPL 427.5
Step 3900: PPL 425.2  (BEST)
Step 4200: PPL 425.7
Step 4500: PPL 426.3  (early stopped)
```

### 5.3 What Worked

1. **CKA float32 fix**: No NaN/Inf, stable training
2. **FDD active**: 4000 steps of alignment signal
3. **CTKD still working**: Temperature evolved 2.0 → 1.49
4. **Improvement over v13.1**: 424.81 vs 434.44 (9.63 PPL gain)

### 5.4 What Didn't Work

1. **Missed <400 target**: Best was 425.19
2. **FDD loss barely decreased**: 0.791 → 0.777 (small change)
3. **Plateau around 425**: Model couldn't break through
4. **Early stopped**: Suggests learning stalled

### 5.5 Validation Tests

```
✗ PPL < 400: best_ppl=425.19, target=400
✓ Improved over v13.1: improvement=9.25 PPL
✓ Spike density [0.1, 0.9]: density=0.371
✓ Amplitudes [0.3, 3.0]: range=[0.71, 1.18]
✓ Training completed: Early stopped at 4500 steps
✓ No NaN/Inf loss: All losses finite
✓ VRAM < 8GB: peak=1.85GB
✓ FDD was active: Active for 4000 steps
✓ Temperature evolved: start=2.00, end=1.49
✓ Early stopping working: stopped at 4500
✓ FDD loss decreased: start=0.7910, end=0.7770
✓ Extended training (5000+): distill_steps=5000

Passed: 11/12, Failed: 1
```

---

## 6. SPIKINGBRAIN AND INFORMATION ENCODING

### 6.1 Critical Clarification (from discussion with researcher)

**SpikingBrain's role is NOT about sparse operations.** It validates the **Information Encoding Mechanism**.

The question SpikingBrain answers:
> "Are the ternary spike patterns {-1, 0, +1} actually encoding meaningful information, or are they just arbitrary quantizations?"

### 6.2 What Needs Validation

1. **Mutual Information**: `I(spikes; teacher_hidden)` - do spikes carry information about teacher's representations?
2. **Encoding Efficiency**: Is information encoded redundantly or efficiently?
3. **Spike Pattern Analysis**: Firing rates, temporal patterns, population coding
4. **Health Metrics**: Dead neurons (always 0), saturated neurons (always ±1)

### 6.3 Current Status

- SpikingBrain validation: **NOT YET IMPLEMENTED**
- Planned for: v15 (prerequisite for sparse ops)
- Current spike tracking: Only density (37.1% non-zero)

---

## 7. PUBLICATION ROADMAP

### 7.1 Correct Ordering (with dependencies)

```
v14: FDD + CKA (DONE)
 ↓ need quality first
v15: SpikingBrain - Information Encoding Validation
 ↓ must prove spikes are meaningful before optimizing
v16: Sparse Ops - torch.sparse implementation
 ↓ this IS the efficiency mechanism
v17: Efficiency Metrics - FLOPs, memory, latency benchmarks
 ↓ NOW we measure (on sparse implementation, not dense!)
v18: Ablations - Controlled experiments
 ↓
v19: Publication - LaTeX, figures, reproducibility
```

### 7.2 Why This Order Matters

**User correction during discussion:**
> "Why are efficiency metrics BEFORE sparse ops? Is that not redundant at best?"

The efficiency we claim comes from sparse ternary operations. Measuring the current dense implementation proves nothing about the ASNN thesis. Therefore:
- v15: Validate spikes encode information (prerequisite for sparse)
- v16: Implement sparse ops (the actual efficiency mechanism)
- v17: THEN measure efficiency (on the sparse implementation)

---

## 8. KEY Q&A EXCHANGES (Critical Understanding)

### Q1: "Are we using SpikingBrain?"

**User's correction**: "SpikingBrain serves one specific, critical purpose: It validates your Information Encoding Mechanism."

**My initial error**: I thought SpikingBrain was about sparse operations or neuromorphic hardware.

**Correct understanding**: SpikingBrain validates that spike patterns carry meaningful information - a prerequisite for everything else.

### Q2: "What is ASNN really for?"

**User's correction**: "Neuromorphic hardware is probably too hard to get into mainstream. This is the purpose of ASNN (ADAPTED Spiking Neural Network), where we adapt the network to current GPUs."

**My initial error**: I stated the goal was neuromorphic hardware deployment.

**Correct understanding**: ASNN adapts SNNs to work efficiently on CURRENT GPUs using ternary activations.

### Q3: "Why dense ternary operations?"

**Reality**: Current implementation is dense because:
1. PPL quality is the blocker (can't optimize garbage)
2. GPU sparse break-even needs >75% sparsity, we have 63%
3. Sparse ops planned for v16 after quality is achieved

### Q4: "PPL 400 is extremely bad"

**User's correction**: "Well, promising is overblown. A PPL of 400 is extremely bad."

**Context**:
- Teacher PPL: 44.6
- PPL 400 = 9x worse than teacher
- Even achieving <400 is still catastrophically bad for practical use
- Need to close the gap significantly more

---

## 9. QUESTIONS FOR EXTERNAL LLM

### Primary Question
**How do we break below PPL 400 and continue closing the 9.5x gap to the teacher?**

### Specific Technical Questions

1. **FDD Tuning**:
   - FDD loss decreased only 0.791 → 0.777 (1.8%). Is this signal too weak?
   - Should we increase fdd_weight from 0.001? What's the risk?
   - Should we align more layer pairs? Currently {0:2, 2:7, 4:11}

2. **Architecture Mismatch**:
   - The fundamental mismatch is Transformer → RWKV
   - Is there a better alignment strategy than CKA?
   - Should we consider intermediate architectures?

3. **Plateau Analysis**:
   - Model plateaued at 425 despite 4000 steps of FDD
   - Is this a capacity limit (22M params)?
   - Is there information loss in ternary quantization that we can't recover?

4. **Temperature Dynamics**:
   - CTKD found optimal T ≈ 1.49 (consistent across v12.1, v13.1, v14)
   - Is this temperature optimal, or should we explore different ranges?

5. **Spike Density**:
   - Current: 37% non-zero (63% zeros)
   - Is this optimal for information encoding?
   - Would encouraging more/less sparsity help?

6. **Next Techniques to Try**:
   - What techniques specifically target the Transformer → RWKV gap?
   - Are there other alignment losses beyond CKA?
   - Should we consider attention transfer (despite RWKV having no attention)?

### Context for Recommendations

Any recommendations should consider:
- PPL quality is priority #1 (efficiency claims meaningless without it)
- We have 22M param budget (5.6x compression from 124M)
- Training on Kaggle T4 (16GB VRAM, but using only 1.85GB currently)
- Ternary activations are non-negotiable (core of ASNN thesis)
- Publication is end goal (need significant improvement for novel contribution)

---

## 10. ATTACHED MATERIALS

For the external LLM's reference:
1. **Results JSON**: Full v14 training results with all metrics
2. **Notebook**: `asnn_goose_colab_v14.ipynb` with complete implementation
3. **Knowledge Base**: `roadmap.md` with full version history and roadmap

---

## 11. SUMMARY

**Where we are**:
- v14 achieved PPL 424.81 (9.63 improvement over v13.1)
- FDD with CKA helped but didn't break <400
- 11/12 validation tests passed

**The core problem**:
- Still 9.5x worse than teacher (424 vs 44.6)
- Need techniques that specifically address Transformer → RWKV mismatch
- Current approach (CTKD + FDD) seems near its limit

**What we need**:
- Novel techniques or insights to break the 425 plateau
- Path toward PPL <300 (ideally <200) for practical usefulness
- Understanding of whether this gap is fundamental or addressable
