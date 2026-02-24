# ASNN-Goose: Roadmap & Version History

**As Of**: 2026-02-23
**Last Updated**: 2026-02-23
**Current Best**: v14.3 (PPL 306.89)
**Current Work**: v15 (SpikingBrain Validation remediation + full rerun)

---

## Operations Snapshot (A-K)

This table is the quick status board for the integrated roadmap:
v15→v16→v17→v18→v19 remains the immutable core chain. H/I/J/K are add-on tracks.

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| A | Engineering Guardrails | IN PROGRESS | Eval/data/verifier/retrieval scaffolds added; full reproducibility guardrails still pending. |
| B | v15 SpikingBrain Validation | BLOCKED | Latest run failed scientific gate; remediation patch applied, waiting full rerun outputs. |
| C | Temporal Coding Proof Suite | NOT STARTED | Parallel scientific validation track not yet built. |
| D | v16 Sparse Ops | NOT STARTED | Starts only after v15 pass gate. |
| E | v17 Efficiency Metrics | NOT STARTED | Includes model-only and retrieval end-to-end benchmarks. |
| F | v18 Ablations | NOT STARTED | Expanded matrix pending (KD, retrieval, post-training comparisons). |
| G | v19 Publication/Repro | NOT STARTED | Repro scripts and publication artifacts pending. |
| H | Generalist Scorecard + Mixture | IN PROGRESS | `eval/suite.yaml` + `data/mixture.yaml` scaffolds exist; baselines/gates pending. |
| I | World-Fit Retrieval | IN PROGRESS | Retrieval interface scaffold exists; runtime/index/eval gate pending. |
| J | Post-Training (Distill + RLVR + Pref) | NOT STARTED | Planned only after gating infrastructure is stable. |
| K | Scaling Ladder to 1B | NOT STARTED | Blocked until prior gates pass. |

---

## Operations Links

- Ops model: `docs/ops/AUTONOMY_OPERATING_MODEL.md`
- Live status board: `docs/ops/STATUS_BOARD.md`
- Gate policy: `docs/ops/GATE_POLICY.md`
- Reporting contract: `docs/ops/REPORTING_CONTRACT.md`
- Program state: `state/program_status.yaml`
- Gate state: `state/gate_results.yaml`
- Autopilot queue: `state/autopilot_queue.yaml`
- Report index: `reports/index.md`

---

## Current Status

### v14.3 - CURRENT BEST

| Metric | Value |
|--------|-------|
| **PPL** | **306.89** |
| Params | 74M (d=768, 5 layers) |
| Gap to Teacher | 6.9x (teacher PPL 44.6) |
| Spike Density | 38.2% |
| VRAM | 5.57GB peak |
| Training | 4200 steps (early stopped) |

**Key Achievement**: Broke 310 target. 29.4% improvement over v13.1.

### v15 - BLOCKED (rerun required)

SpikingBrain: Information Encoding Validation

**Purpose**: Validate that spike patterns {-1, 0, +1} encode meaningful semantic information, NOT arbitrary quantization.

**Success Criteria**:
| Metric | Threshold |
|--------|-----------|
| Dead neurons | < 5% |
| Saturated neurons | < 10% |
| Mutual Information | > 0.1 |
| CKA mean | > 0.3 |
| Firing rate | [0.2, 0.6] |

**Status**: Latest run `v15_2026-02-23_155547` failed scientific thresholds (`dead neurons`, `MI`, `CKA`) and Phase B is paused.
Remediation patch is now applied in notebook; waiting for full rerun outputs from RunPod.

**Runtime + scientific remediation update (2026-02-23)**:
1. Added dependency bootstrap and optional plotting fallback.
2. Fixed v15 validator batch/forward mismatches (`return_spike_info`, tuple loaders, `val_loader` reference).
3. Added hard guards for empty loaders and zero-token evaluation.
4. Added mixed token/channel ternary thresholding to reduce dead channels.
5. Added spike semantic alignment loss (teacher-to-spike ternary target alignment).
6. Updated validator to use both `k` and `v` spikes and robust MI discretization.
7. Preserved autonomous artifact bundle and report-ingestion flow in final save cell.

---

## Complete Roadmap

| Version | Focus | Target | Status |
|---------|-------|--------|--------|
| v6-v10 | Baseline | Working distillation | DONE |
| v11.x | Channel-wise spikes | - | FAILED (symmetry) |
| v12.1 | CTKD+GRL | PPL <500 | DONE (445.61) |
| v13.1 | Extended training | PPL <420 | DONE (434.44) |
| v14 | FDD+CKA | PPL <400 | DONE (424.81) |
| v14.1 | d_model=512 | PPL <400 | DONE (321.48) |
| v14.3 | d_model=768 | PPL <310 | **DONE (306.89)** |
| **v15** | **SpikingBrain** | **Validate encoding** | **BLOCKED (rerun pending)** |
| v16 | Sparse Ops | torch.sparse | PLANNED |
| v17 | Efficiency Metrics | FLOPs/latency | PLANNED |
| v18 | Ablations | Experiments | PLANNED |
| v19 | Publication | LaTeX/figures | PLANNED |

---

## Version Details

### v14.3 (2026-01-04) - SUCCESS

**Result**: PPL 306.89

**Why It Worked**:
- d_model=768 provides sufficient capacity for ternary encoding
- Lower LR (2e-4) for larger model
- Longer lambda warmup (0.25)
- Patience 800 for early stopping

**Training Trajectory**:
```
Step   300: PPL 512.71
Step   900: PPL 370.34
Step  1800: PPL 323.59
Step  3300: PPL 306.89  ← BEST
Step  4200: PPL 308.05  (early stopped)
```

---

### v14.1 (2026-01-04) - BREAKTHROUGH

**Result**: PPL 321.48 (broke 400 barrier!)

**Key Insight**: Capacity was the bottleneck. Ternary neurons encode only 3 values, so need exponentially more neurons to match continuous networks.

```
d_model 320 → 512: State space 3^320 → 3^512 (exponential increase)
```

---

### v14 (2026-01-04) - SUCCESS

**Result**: PPL 424.81

**Innovation**: Feature Dynamics Distillation (FDD)
- Aligns layer DYNAMICS (Δh = h[l+1] - h[l]), not hidden states
- Uses CKA loss (projector-free, dimension-agnostic)
- Addresses Transformer → RWKV architecture mismatch

---

### v12.1 (2025-12-30) - SUCCESS

**Result**: PPL 445.61

**Innovation**: CTKD with Gradient Reversal Layer
- Adversarial min-max temperature learning
- Temperature self-tunes 2.0 → 1.58
- GRL reverses gradients so T maximizes KL loss

---

### Failed Experiments

| Version | PPL | Failure Mode |
|---------|-----|--------------|
| v7 | 1655 | Hidden alignment weight=1.0 (too strong) |
| v11.1 | 512 | Channel-wise spikes (structural symmetry) |
| v12 | ~1000 | CTKD without GRL (temperature runaway to 10.0) |
| v13 | 1126 | POCL curriculum (catastrophic forgetting) |
| v14.2 | 331 | d_model=1024 (under-trained, LR too high) |

**Key Lessons**:
1. Hidden alignment must be very light (0.001 not 1.0)
2. CTKD requires adversarial optimization via GRL
3. Curriculum learning doesn't work for SNN distillation
4. Larger models need proportionally more training and lower LR

---

## Future Work

### v16: Sparse Operations

**Objective**: Implement actual sparse computation

**Decision Criteria**:
- Current spike density: 38% non-zero (62% zeros)
- GPU sparse break-even: ~75% sparsity needed
- May need structured sparsity patterns

**Implementation Options**:
1. torch.sparse COO/CSR format
2. Custom ternary sparse kernels
3. Block-sparse patterns
4. Hybrid dense/sparse per layer

---

### v17: Efficiency Metrics

**Objective**: Quantify computational savings (on sparse implementation)

**Metrics**:
1. FLOPs reduction ratio
2. Memory footprint (activation + weights)
3. Inference latency (ms/token)
4. Throughput (tokens/second)
5. Energy estimation (MAC vs ADD)

**Hardware Targets**: T4, A100, CPU

---

### v18: Ablation Studies

**Required Ablations**:

1. **Distillation**: No KD → Logit-only → +CTKD → +FDD → Full
2. **Spike Type**: Dense → Binary → Ternary → Quaternary
3. **Capacity**: 16M → 22M → 42M → 74M
4. **Components**: ±GRL, ±FDD warmup, ±kill switch

---

### v19: Publication

**Deliverables**:
1. Main results table (PPL, params, FLOPs, latency, energy)
2. Training curves and visualizations
3. Complete reproducibility package
4. LaTeX manuscript

---

## Gap Analysis

| Milestone | PPL | Gap to Teacher | Status |
|-----------|-----|----------------|--------|
| v6 (start) | 627 | 14.1x | - |
| v12.1 | 446 | 10.0x | CTKD works |
| v14.3 | 307 | 6.9x | Capacity scaled |
| Target | <200 | <4.5x | Research goal |
| Teacher | 44.6 | 1.0x | Baseline |

**Progress**: Closed 51% of the gap (14.1x → 6.9x)

---

## Key Hyperparameters (v14.3)

```python
# Architecture
d_model = 768
n_layers = 5
params = 74M

# Training
distill_steps = 5000
distill_lr = 2e-4
batch_size = 8
warmup_ratio = 0.1

# CTKD
tau_min = 1.0
tau_max = 5.0
tau_init = 2.0
lambda_warmup_ratio = 0.25

# FDD
fdd_weight = 0.001
fdd_warmup_steps = 500
layer_map = {0: 2, 2: 7, 4: 11}

# Early stopping
patience = 800
min_delta = 0.5
```

---

## Success Criteria

### Quality (v14.3 - ACHIEVED)
- [x] PPL < 310
- [x] All validation tests pass
- [x] Spike density in [0.2, 0.6]
- [x] No training instability

### Validation (v15 - BLOCKED)
- [ ] Dead neurons < 5%
- [ ] Saturated neurons < 10%
- [ ] MI > 0.1
- [ ] CKA > 0.3

### Efficiency (v16-v17 - PLANNED)
- [ ] Sparse implementation complete
- [ ] FLOPs reduction measured
- [ ] Latency benchmarked

### Publication (v18-v19 - PLANNED)
- [ ] Ablations complete
- [ ] Figures generated
- [ ] Manuscript drafted
