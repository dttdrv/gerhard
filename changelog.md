# ASNN-Goose Changelog

## [v15] 2026-01-05 - SpikingBrain: Information Encoding Validation

**Status**: IMPLEMENTED - Prerequisite for v16 (sparse ops)

**Purpose**: Validate that spike patterns {-1, 0, +1} encode meaningful semantic information, NOT arbitrary quantization artifacts.

**Components**:
1. **SpikeHealthChecker**: Dead/saturated neuron detection, firing rate analysis
2. **MutualInformationEstimator**: Binning-based MI between spikes and teacher hiddens
3. **RepresentationAnalyzer**: CKA similarity between spike and teacher representations
4. **SpikingBrainValidator**: Main orchestrator for full validation suite

**Success Criteria**:
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Dead neurons | < 5% | Information loss |
| Saturated neurons | < 10% | Threshold miscalibration |
| MI (binning) | > 0.1 | Spikes carry signal |
| CKA mean | > 0.3 | Representations aligned |
| Firing rate | [0.2, 0.6] | Healthy sparsity |

**Files Added/Modified**:
- `src/evaluation/spiking_brain.py` (NEW) - Main validation module
- `src/evaluation/spike_analysis.py` - Added health metrics methods
- `src/utils/visualization.py` - Added firing rate histogram and CKA plots
- `notebooks/asnn_goose_colab_v14.ipynb` - Added cells 27-29 for validation

**Usage**:
```python
from src.evaluation.spiking_brain import SpikingBrainValidator

validator = SpikingBrainValidator(device, layer_map={0:2, 2:7, 4:11})
results = validator.validate(student, teacher, val_dataloader, max_batches=20)
print(results.summary)
```

**Next**: Run validation on trained v14.3 model, then proceed to v16 (sparse ops) if all tests pass.

---

## [v14.3] 2026-01-04 - SUCCESS ✅ (d_model=768, PPL 306.89!)

**Status**: SUCCESS - Broke 310 target! Best PPL to date.

**Results**:
- **PPL: 306.89** ✅ (Best at step 3300)
- **Final PPL**: 307.70 (early stopped at 4200)
- **Improvement**: 127.55 PPL from v13.1 (434.44 → 306.89 = **29.4%**)
- **Params**: 74M (d_model=768, 5 layers)
- **Temperature**: 2.0 → 1.61 (CTKD found optimal)
- **Lambda**: 0 → 1.0 (full adversarial strength)
- **VRAM**: 5.57GB peak
- **Validation**: 12/12 tests passed ✅

**Training Trajectory**:
```
Step   300: PPL 512.71  (starting)
Step   600: PPL 407.16  (FDD active)
Step   900: PPL 370.34
Step  1200: PPL 342.10
Step  1800: PPL 323.59
Step  2100: PPL 318.07
Step  2700: PPL 314.10
Step  3300: PPL 306.89  ← BEST
Step  4200: PPL 308.05  (early stopped)
```

**Why v14.3 Succeeded Where v14.2 Failed**:
| Factor | v14.2 (FAILED) | v14.3 (SUCCESS) |
|--------|----------------|-----------------|
| d_model | 1024 (too big) | 768 (sweet spot) |
| distill_lr | 3e-4 (too high) | 2e-4 |
| lambda_warmup | 0.2 (too fast) | 0.25 |
| patience | 500 | 800 |

**Key Metrics**:
- **FDD loss**: 0.7677 → 0.6206 (**19% decrease** - alignment working!)
- **Spike density**: 38.2% (healthy range)
- **Gap to teacher**: 6.9x (was 9.5x in v14 = **27% improvement**)

---

## [v14.2] 2026-01-04 - FAILED ❌ (d_model=1024 regressed)

**Status**: FAILED - Scaling too aggressive, caused regression

**Results**:
- **PPL: 330.89** (best at step 1800) - **WORSE than v14.1's 321.48**
- **Final PPL: 393.27** (regressed badly, early stopped at 2400)
- **Params**: ~165M (d_model=1024)
- **VRAM**: 6GB (well under limit)

**Training Trajectory**:
```
Step  600: PPL 385.9   (FDD active)
Step  900: PPL 355.6
Step 1200: PPL 340.6   (lambda=0.01)
Step 1500: PPL 333.6   (lambda=0.04)
Step 1800: PPL 330.9   ← BEST (lambda=0.10)
Step 2100: PPL 334.5   (regression starts, lambda=0.17)
Step 2400: PPL 393.3   ← CATASTROPHIC (lambda=0.27, early stopped)
```

**Failure Analysis**:
1. **Under-training**: 4x more params but SAME training budget
2. **LR not scaled**: Same 3e-4 for 165M (should reduce for larger models)
3. **CTKD too aggressive**: Lambda ramp-up destabilized training
4. **Model was improving**: Regression happened as lambda increased

**Lesson Learned**: Scaling requires proportional increase in training steps and LR adjustment.

---

## [v14.1] 2026-01-04 - SUCCESS ✅ BREAKTHROUGH!

**Status**: SUCCESS - Broke PPL 400 barrier! Target exceeded by 78 PPL.

**Results**:
- **PPL: 321.48** ✅ (Best: 320.34 at step 2100)
- **Improvement**: 103.33 PPL reduction from v14 (424.81 → 321.48 = **24.3%**)
- **Params**: 41.6M (d_model=512, 5 layers)
- **Temperature**: 2.0 → 1.56
- **VRAM**: 6GB peak (10GB headroom!)
- **Training**: Early stopped at 2700 steps (plateau)
- **Validation**: 12/12 tests passed ✅

**Training Trajectory**:
```
Step  300: PPL 510.75  (starting)
Step  600: PPL 418.33  ← Beat v14's 424!
Step  900: PPL 369.01  ← Broke 400!
Step 1200: PPL 357.41
Step 1500: PPL 340.47
Step 1800: PPL 328.84
Step 2100: PPL 320.34  ← BEST
Step 2700: PPL 321.63  (early stopped)
```

**Why This Worked** (External Analysis):
1. **Capacity was the bottleneck**: Ternary neurons encode only 3 values vs infinite for float32
2. **d_model 320→512**: Exponentially increased state space ($3^{512}$ vs $3^{320}$)
3. **FDD/CTKD were "suffocated"** at smaller capacity - now they work
4. **The "Ternary Tax"**: 42M dense → PPL ~60-80; 42M ternary → PPL 321

**Key Insight**: d_model=512 is the **minimum floor** for ternary to function on this task.

**Gap Analysis**:
- Teacher (GPT-2): PPL 44.6
- Student (v14.1): PPL 321.48 = **7.2x worse** (was 9.5x)
- Closed 24% of the gap in one version!

---

## [v14] 2026-01-04 - SUCCESS ✅ (FDD: Feature Dynamics Distillation)

**Status**: SUCCESS - Improved 9.63 PPL over v13.1

**Results**:
- **PPL: 424.81** ✅ (Best at step 3900)
- **Improvement**: 9.63 PPL reduction from v13.1 (434.44 → 424.81 = 2.2%)
- **Temperature**: 2.0 → 1.49 (similar to v13.1)
- **Lambda**: 0 → 1.0 (full adversarial strength)
- **Training**: Early stopped at 4500 steps (plateau detected)
- **Validation**: 11/12 tests passed (missed <400 target)

**Training Trajectory**:
```
Step 2700: PPL 434.0 (matched v13.1!)
Step 3900: PPL 425.2 (BEST)
Step 4500: PPL 426.3 (early stopped)
```

**Innovation**: Feature Dynamics Distillation (FDD) with CKA Loss
- Aligns layer DYNAMICS (Δh = h_{l+1} - h_l), not just hidden states
- Uses CKA (Centered Kernel Alignment) - projector-free, dimension-agnostic
- Addresses architecture mismatch between transformer and RWKV

**Key Config**:
```python
use_fdd: bool = True
fdd_weight: float = 0.001         # Very conservative (v7 used 1.0 and failed)
fdd_warmup_steps: int = 500       # Don't enable until step 500
fdd_loss_type: str = "cka"        # Projector-free alignment
fdd_n_align_layers: int = 3       # Align 3 layer pairs
fdd_kill_threshold: float = 0.10  # Disable if PPL increases >10%
```

**Why Not <400?** (External LLM Analysis):
- FDD weight too low (0.001 → signal negligible)
- Capacity too small (320d may not be enough for ternary)
- Missing hard distillation (only soft KL targets)

---

## [v13.1] 2025-12-31 - SUCCESS ✅

**Status**: SUCCESS - Extended training improved PPL further

**Results**:
- **PPL: 434.44** ✅ (Best at step 4500)
- **Improvement**: 11.17 PPL reduction from v12.1 (445.61 → 434.44 = 2.5%)
- **Temperature**: 2.0 → 1.50 (similar to v12.1's 1.575)
- **Lambda**: 0 → 1.0 (full adversarial strength)
- **Training**: Completed all 5000 steps (no early stopping triggered)
- **Compression**: 5.6x (124M → 22M params)

**Key insight**: Extended training works! Model kept improving past 3000:
```
Step 3000: PPL 440.8
Step 3300: PPL 437.7
Step 3900: PPL 435.5
Step 4500: PPL 434.44 (BEST)
```

**Changes from failed v13**:
- POCL: **DISABLED** (caused catastrophic regression)
- CTKD: **RE-ENABLED** (proven to work)
- Extended training: 5000 steps (kept)
- Early stopping: patience=500 (not triggered)

---

## [v13] 2025-12-31 - FAILED (POCL caused regression)

**Status**: FAILED - PPL 1125.94 (catastrophic regression from 445.61)

**Results**:
```
Step 300: PPL=980 (already terrible after pre-training!)
Step 600: PPL=978.5
Step 900: PPL=1090 (getting WORSE)
Step 1200: PPL=1134 (early stopped)
Final: PPL=1125.94 (2.5x worse than v12.1!)
```

**Root Cause Analysis**:
1. Pre-training corruption: 100 steps on FULL data with T=2.0
2. Then restricted to EASY 33% data → catastrophic forgetting
3. T=1.0 too sharp for SNN distillation
4. Never reached stage 2/3 (early stopped at step 1200)

**Lesson Learned**: POCL curriculum learning does NOT work for SNN-LLM distillation
- The "difficulty" ranking from continuous models doesn't transfer to SNNs
- Pre-training before curriculum corrupts the training dynamics
- Temperature 1.0 is too aggressive (v12.1 found 1.58 optimal)

**Target**: PPL <420 (from v12.1's 445.61)

**Innovations (all FAILED)**:
1. **POCL (Progressive Overload Curriculum Learning)**
   - 3-stage curriculum: Easy 33% → Medium 66% → Hard 100%
   - Data partitioned by difficulty (KL + CE loss ranking)
   - Pre-training for 100 steps before difficulty scoring

2. **Fixed Temperature Schedule**
   - Rising: 1.0 → 1.5 → 2.0 (per POCL paper)
   - CTKD disabled (using fixed schedule instead)

3. **Extended Training**
   - 5000 steps (was 3000 in v12.1)
   - Early stopping: patience=500, min_delta=1.0 PPL
   - Lower final LR: 1e-6

**Config**:
```python
use_pocl: bool = True
pocl_stages: int = 3
pocl_temp_schedule: tuple = (1.0, 1.5, 2.0)
pocl_pretrain_steps: int = 100
distill_steps: int = 5000
use_early_stopping: bool = True
early_stopping_patience: int = 500
min_ppl_delta: float = 1.0
use_ctkd: bool = False  # Disabled for v13
```

**Key Question**: Temperature direction conflict
- POCL paper: Rising T (1.0 → 2.0) - "soft targets late for nuance"
- v12.1 CTKD: Falling T (2.0 → 1.58) - "sharp targets better for SNN"
- Decision: Start with POCL's rising schedule, test falling as ablation if needed

**References**:
- [POCL Paper](https://arxiv.org/abs/2506.05695) - Progressive Overload Curriculum Learning (2025)
- [v13 Plan](../knowledge/witty-imagining-marshmallow.md) - Detailed implementation plan

---

## [v12.1] 2025-12-30 - SUCCESS ✅

**Status**: SUCCESS - Proper CTKD with Gradient Reversal Layer

**Results**:
- **PPL: 445.61** ✅ (Target <500 MET!)
- **Improvement**: 68.89 PPL reduction from v10 baseline (514.5 → 445.61 = 13.4%)
- **Temperature**: 2.0 → 1.575 (evolved naturally, not stuck at bounds)
- **Tests**: 10/10 passed
- **Training**: Kaggle T4, VRAM 4.14GB peak
- **Compression**: 5.6x (124M → 22M params)

**Innovation**: Adversarial min-max temperature learning via GRL (Gradient Reversal Layer).

**Why v12 Failed**:
- v12 used simple regularization: `0.1 * (T - 2.0)²`
- Without GRL, optimizer pushed T to max (10.0) to minimize KL loss superficially
- Result: PPL regression to 1000+ at step 600 (temperature runaway)

**CTKD Proper Implementation**:
The key insight from CTKD paper (ArXiv 2211.16231):
- **Student** minimizes KL loss (normal gradients)
- **Temperature** maximizes KL loss (finds hardest difficulty)
- This creates adversarial min-max game: `min_θS max_θT L_KD(τ)`

**Gradient Reversal Layer (GRL)**:
- Forward pass: Identity `GRL(x) = x`
- Backward pass: Negation `dGRL/dx = -λ`
- Effect: Temperature gradients reversed → T seeks to maximize KL

**Temperature Evolution** (observed):
```
Step    0 (  0%): T=2.0000  (initial)
Step  600 ( 20%): T=2.0014  (warmup, minimal change)
Step 1500 ( 50%): T=1.6422  (GRL active, T decreasing)
Step 2999 (100%): T=1.5751  (stabilized)
```
- CTKD found optimal T ≈ 1.58 (lower than initial 2.0)
- Matches "reverse annealing" phenomenon from CTKD paper

**Config** (v12.1):
- d_model: 320
- n_layers: 5
- params: ~22M
- use_ctkd: **True** ← v12.1 KEY
- tau_min: 1.0, tau_max: 5.0, tau_init: 2.0
- lambda_max: 1.0 (full adversarial strength)
- lambda_warmup_ratio: 0.2 (20% warmup with λ=0)
- Sigmoid bounding (smooth gradients, no harsh clamp)
- NO manual regularization (GRL handles it)

**Key Implementation**:
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

**Sources**:
- [CTKD Paper](https://arxiv.org/abs/2211.16231) - Curriculum Temperature for KD
- [GRL Origin](https://arxiv.org/abs/1409.7495) - Ganin & Lempitsky Domain Adaptation
- [Implementation Plan](../knowledge/ctkd_implementation_plan.md) - Detailed analysis

---

## [v12] 2025-12-28 - FAILED (temperature runaway)

**Status**: Failed - Simplified CTKD without GRL caused temperature runaway

**Results**: PPL ~1000+ at step 600 (severe regression from v10's 514.5)

**Root Cause**: Without Gradient Reversal Layer, both student and temperature minimize the same loss
- Optimizer pushed T to max clamp (10.0) to get superficially lower KL loss
- High T = soft softmax = easy match but meaningless gradients
- Student couldn't learn - gradients too diffuse

**Lesson Learned**: CTKD requires adversarial min-max optimization via GRL, not simple regularization

---

## [v11.1] 2025-12-28 - FAILED (structural symmetry)

**Status**: Failed - Channel-wise spikes don't work with RWKV architecture

**Results**: PPL 512.04 (only 0.63 improvement from v11's 512.67)

**Root Cause**: STRUCTURAL SYMMETRY (not regularization)
- K and V amplitudes stayed IDENTICAL even without regularization
- Both start at `torch.ones(d_model)` - identical initialization
- `kv = k * v` creates symmetric gradient path: `∂L/∂k ≈ ∂L/∂v`
- Optimizer state (Adam momentum/variance) initialized identically
- TerViT works on ViT because pretrained weights break symmetry; RWKV has none

**Lesson Learned**: Channel-wise techniques require asymmetric initialization or pretrained weights

---

## [v11] 2025-12-28 - BUG (amplitude regularization suppressed learning)

**Status**: Completed - minor improvement but amplitude bug discovered

**Results**:
- PPL: 512.67 (marginal improvement from 514.5)
- Training: 15.6 min on T4

**Bug Found**:
- K and V amplitudes were EXACTLY identical per layer (to 16 decimal places!)
- Amplitudes barely moved from 1.0 (±4%)
- Root cause: `amplitude.var() * 0.01` penalty forced all channels to identical values

**Innovation**: Per-channel learnable alpha and amplitude for ternary spikes (TerViT paper).

---

## [v10] 2025-12-28 - BASELINE

**Status**: Previous stable version

**Config**:
- d_model: 320
- n_layers: 5
- params: ~22M
- All advanced features: DISABLED

**Results**:
- PPL: 514.5
- Training: ~22 min on T4

---

## [v11.x] 2025-12-28 - FAILED (ARCHIVED)

Old v11.x experiments with multiple techniques failed. Archived:
- v11.2: PPL 641.5 (POCL + channel-wise + progressive alpha)
- v11.1: PPL 754.1 (temperature runaway fixed but still regressed)
- v11: PPL 749 (all four techniques at once)

---

## Version Summary

See `knowledge/roadmap.md` for full roadmap and details.

| Version | Date | PPL | Status | Notes |
|---------|------|-----|--------|-------|
| v15 | 2026-01-05 | - | IMPLEMENTED | SpikingBrain: Information Encoding Validation |
| v14.3 | 2026-01-04 | **306.89** | SUCCESS ✅ | d=768 (74M), BEST PPL! 27% gap closed |
| v14.2 | 2026-01-04 | 330.89 | FAILED ❌ | d=1024 regressed (under-trained) |
| v14.1 | 2026-01-04 | **321.48** | SUCCESS ✅ | d=512 (42M), BREAKTHROUGH 24.3%! |
| v14 | 2026-01-04 | **424.81** | SUCCESS ✅ | FDD + CKA (2.2% improvement) |
| v13.1 | 2025-12-31 | **434.44** | SUCCESS ✅ | CTKD + Extended (2.5% improvement) |
| v13 | 2025-12-31 | 1125.94 | FAILED | POCL caused regression |
| v12.1 | 2025-12-30 | 445.61 | SUCCESS ✅ | CTKD+GRL (13.4% improvement!) |
| v12 | 2025-12-28 | ~1000 | FAILED | Temp runaway (no GRL) |
| v11.1 | 2025-12-28 | 512.04 | FAILED | Channel-wise NO reg (symmetry issue) |
| v11 | 2025-12-28 | 512.67 | BUG | Channel-wise WITH reg (suppressed) |
| v10 | 2025-12-28 | 514.5 | SUCCESS | 320d/5L baseline |
| v9 | 2025-12-27 | 541.7 | SUCCESS | 320d/5L baseline validated |
| v8 | 2025-12-27 | 559 | SUCCESS | Fixed v7 regression |
| v7 | 2025-12-26 | 1655 | FAILED | Hidden alignment weight=1.0 |
| v6 | 2025-12-26 | 627 | SUCCESS | First working distillation |
