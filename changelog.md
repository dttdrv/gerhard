# ASNN-Goose Changelog

## 2026-03-11 - Phase 0 Cycle 1: Truth + Gate Drift Alignment

### Action 1 - Reporting gate hardening
**Facts**
- `scripts/register_notebook_run.py` now inspects reproducibility metadata and emits a deterministic `reproducibility_metadata` gate instead of silently writing `commit: unknown` as if the evidence were clean.
- `tests/test_register_notebook_run.py` now uses the nested `validation.health`, `validation.mutual_information`, and `validation.cka` Phase B artifact shape and covers the missing-reproducibility path.
- Fresh verification on 2026-03-11:
  - `python -m pytest .\tests\test_register_notebook_run.py -q` -> `9 passed`
  - `python -m pytest -q` -> `43 passed`

**Hypotheses**
- Tightening the registration gate now will prevent future false-green reports from being treated as authoritative evidence.

**Inferences**
- Archived February 2026 reports remain historically valuable but must be described as pre-tightening evidence for reproducibility claims.

### Action 2 - Governance bridge
**Facts**
- Added `PROJECT_BRIEF.md` as a pointer-only document for mission, constraints, and source-order discovery.
- Added `docs/ops/TIME_CAPSULE.md` as the living handoff memory with required sections seeded from authoritative current sources.
- Updated `reports/README.md`, `docs/ops/REPORTING_CONTRACT.md`, and `docs/ops/STATUS_BOARD.md` to keep the existing canonical source order primary and to mark February 2026 archived runs as pre-tightening evidence.

**Hypotheses**
- Bridging the current truth stack is lower-risk than introducing root `STATE.yaml` / `LOG.md` during this cycle.

**Inferences**
- The next bounded intervention should target the repo-native Phase B checkpoint preflight adapter, not model math or a fresh expensive rerun.

### Action 3 - Verification archive
**Facts**
- A synthetic registration repo under `%TEMP%` produced `CONTINUE` and coherent `reports/index.md`, `state/program_status.yaml`, `state/gate_results.yaml`, and `docs/ops/STATUS_BOARD.md`.
- The authoritative dossier for `v15_2026-02-23_200258` was reopened over localhost and still matches the blocked MI/CKA story in the archived evidence.
- `trivy` filesystem scanning still failed through the MCP server on 2026-03-11 with an internal tool error; no clean vulnerability scan artifact was produced in this cycle.

**Hypotheses**
- The Trivy failure is environmental/tooling drift rather than repo content drift, but it remains unproven until a clean scan succeeds.

**Inferences**
- Trivy failure must stay visible in `docs/ops/TIME_CAPSULE.md` and cannot be silently omitted from verification reporting.

### Action 4 - Repo-native Phase B preflight adapter
**Facts**
- `src/evaluation/spiking_brain.py` now normalizes teacher activations from either Hugging Face-style `.hidden_states` outputs or repo-native `TeacherModel` `aux["layer_activations"]` during representation collection.
- The validator now aggregates MI/CKA across mapped layers and both `k` / `v` spike channels in the same core shape as the reset notebook instead of using only the first mapped `k` path.
- The same collector path now accepts spike traces as either a single dict or a list of per-timestep dicts and raises an explicit error when a mapped repo teacher layer activation is missing.
- Added `tests/test_spiking_brain.py` to cover repo-native teacher compatibility, preserved Hugging Face-style compatibility, single-dict spike normalization, and notebook-parity MI/CKA aggregation.
- Fresh verification on 2026-03-11:
  - `python -m pytest .\tests\test_spiking_brain.py -q` -> `4 passed`
  - `python -m pytest -q` -> `47 passed`
  - A direct CPU smoke call to `SpikingBrainValidator.validate()` with small repo-native `TeacherModel` / `ASNNGoose` instances completed without interface errors.
- `PROJECT_BRIEF.md`, `docs/ops/TIME_CAPSULE.md`, and `docs/ops/STATUS_BOARD.md` were updated so the next bounded action is checkpoint-only `SMOKE` / `DIAGNOSE`, not another adapter build pass.

**Hypotheses**
- The immediate blocker for repo-native checkpoint preflight was evaluator drift in the collector seam and core MI/CKA aggregation path, not model math.
- If the next checkpoint-only validation still disagrees with the reset notebook, the remaining drift is more likely in control-suite logic than in teacher activation access or core metric aggregation.

**Inferences**
- The cheapest honest next step is to run checkpoint-only `SMOKE` and `DIAGNOSE` against an existing checkpoint before considering any notebook rerun.
- `trivy` remains intentionally out of scope for this thread by explicit user instruction and is not part of the current verification loop.

### Action 5 - Phase B finish-path hardening
**Facts**
- `scripts/register_dossier_run.py` now accepts both consolidated dossier payloads and the reset notebook's embedded raw-payload dossier shape.
- Dossier ingestion now rejects unsafe `run_id` values, rebuilds a clean staging directory on repeated ingest, and emits clean operator-facing errors for malformed or incomplete dossier inputs.
- `scripts/register_notebook_run.py` now validates `run_id` values before writing under `outputs/<run_id>/` and records `commit_source` explicitly in run reports.
- New registration coverage was added in `tests/test_register_dossier_run.py` for reset-notebook dossiers, consolidated dossiers, unsafe `run_id` rejection, and repeated-ingest staging cleanup.
- `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`, `docs/ops/STATUS_BOARD.md`, `docs/ops/REPORTING_CONTRACT.md`, `reports/README.md`, `PROJECT_BRIEF.md`, `docs/ops/TIME_CAPSULE.md`, `tmp/build_v15_reset_notebook.py`, and `notebooks/asnn_goose_v15_reset_master.ipynb` were aligned to the reset-notebook `SMOKE -> DIAGNOSE -> FULL` flow with laptop-side dossier registration preferred and notebook-side registration disabled by default.
- Fresh verification on 2026-03-11:
  - `python -m pytest .\tests\test_register_dossier_run.py -q` -> `4 passed`
  - `python -m pytest .\tests\test_register_notebook_run.py -q` -> `10 passed`
  - `python -m pytest -q` -> `52 passed`
  - synthetic dossier CLI registration in a temp repo -> `CONTINUE`
  - archived dossier CLI registration for `v15_2026-02-23_200258` in a temp repo -> `PAUSE_NEEDS_INPUT`

**Hypotheses**
- The remaining blocking risk before the user runs the reset notebook is now operational execution quality on RunPod, not repo-side reporting or dossier-ingestion drift.

**Inferences**
- The next honest step is no longer another local patch cycle; it is the staged RunPod execution path from the reset notebook with an existing checkpoint.
- The preferred local registration path is now safe enough to recommend, provided the notebook artifacts carry commit + fingerprint metadata.

### Action 6 - Thin RunPod operator notebook
**Facts**
- Added `notebooks/asnn_goose_v15_runpod_operator.ipynb` as a thin control notebook for RunPod execution.
- The new notebook does not duplicate science logic; it validates Phase B parameters, keeps notebook-side registration off, and executes `notebooks/asnn_goose_v15_reset_master.ipynb` through `python -m jupyter nbconvert --execute`.
- Added `tests/test_runpod_operator_notebook.py` to verify that the operator notebook still targets the canonical reset notebook and keeps `GERHARD_ENABLE_REGISTER_RUN` disabled.
- Fresh verification on 2026-03-11:
  - `python -m pytest .\tests\test_runpod_operator_notebook.py -q` -> `1 passed`
  - `python -c "import json, pathlib; ..."` notebook parse check -> `cells 7`, `valid_json True`
  - `python -m pytest -q` -> `53 passed`

**Hypotheses**
- A thin operator notebook is the smallest useful intervention that improves RunPod usability without creating a second authoritative research notebook.

**Inferences**
- The user can now run Phase B from one notebook surface while preserving the reset notebook as the only source of research logic.

### Action 7 - Concrete Colab T4 single-cell launcher
**Facts**
- Added `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` as a one-cell Colab launcher notebook with `https://github.com/dttdrv/gerhard.git` prefilled.
- The launcher mounts Google Drive when appropriate, auto-discovers likely v15/v14.3 checkpoints under Colab/Drive paths, keeps notebook-side registration off, and executes `notebooks/asnn_goose_v15_reset_master.ipynb` through `python -m jupyter nbconvert --execute`.
- Added `tests/test_colab_t4_single_cell_notebook.py` to verify that the notebook stays single-cell, targets the canonical reset notebook, and keeps `GERHARD_ENABLE_REGISTER_RUN` disabled.
- Fresh verification on 2026-03-11:
  - `python -m pytest .\tests\test_colab_t4_single_cell_notebook.py -q` -> `1 passed`
  - `python -m pytest -q` -> `54 passed`

**Hypotheses**
- A concrete one-cell Colab launcher reduces the last user-side setup burden enough to make Colab T4 a viable execution surface without reintroducing notebook-logic drift.

**Inferences**
- The user can now execute the staged Phase B flow on Colab with a single notebook file and no repo URL editing.

### Action 8 - Fresh-rerun launcher for the no-checkpoint case
**Facts**
- Verified on 2026-03-11 that the repo contains no committed `*.pt`, `*.pth`, or `*.ckpt` checkpoint compatible with the reset-notebook preflight path.
- `notebooks/asnn_goose_colab_v15.ipynb` is the only current notebook that still contains the full distillation/training path, so it is now the active training target for a fresh rerun.
- `notebooks/asnn_goose_colab_v15.ipynb` now honors env overrides for `SEED`, `distill_steps`, `batch_size`, `accumulation_steps`, `eval_interval`, `run_id`, notebook-side registration, and dossier auto-download.
- `notebooks/asnn_goose_colab_v15.ipynb` now copies `outputs/checkpoints/v15_best.pt` into the per-run artifact bundle.
- Added `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` as the concrete one-cell Colab launcher for the fresh-rerun path.
- Added `tests/test_colab_fresh_rerun_single_cell_notebook.py` to verify the launcher target, disabled notebook-side registration, env-controlled training knobs, and checkpoint bundling contract.

**Hypotheses**
- The smallest honest way to get new Phase B evidence without an existing checkpoint is to keep the old training notebook’s science path but wrap it in a launcher that enforces the newer evidence-handling rules.

**Inferences**
- The default next action is no longer checkpoint-only `SMOKE -> DIAGNOSE -> FULL`; it is a fresh Colab rerun that produces a new dossier bundle and `v15_best.pt`, followed by laptop-side registration.

### Action 9 - Detailed evidence-bundle expansion for fresh reruns
**Facts**
- `notebooks/asnn_goose_colab_v15.ipynb` now writes detailed machine-readable artifacts into each run directory: `environment.json`, `training_curves.json`, `hardware_stats.json`, `spike_analysis.json`, `validation_tests.json`, `control_suite.json`, `checkpoint_metadata.json`, `figures_index.json`, `detailed_results.json`, and `artifact_manifest.json`.
- The single-file dossier now embeds the raw payloads for those detailed artifacts in addition to the consolidated payload and `results.json` snapshot.
- `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` now copies `executed_training_notebook.ipynb` and `operator_env.json` into the run bundle and writes `launcher_bundle_manifest.json`.
- The recursive artifact manifest now walks the whole run directory tree, so copied figures under `figures/` are part of the evidence record instead of being silently omitted.
- `tests/test_colab_fresh_rerun_single_cell_notebook.py` now verifies that the launcher and training notebook still expose the expanded detailed-artifact surface.

**Hypotheses**
- A richer evidence bundle is the smallest credible way to satisfy “much more detailed results of absolutely everything” without changing model behavior or the registration contract.

**Inferences**
- After the next rerun, post-run diagnosis should be possible from the bundle itself instead of requiring another notebook reconstruction pass.

### Action 10 - Supervisor report package
**Facts**
- Added `reports/2026/03/phase_b_supervisor_report_2026-03-12.md` as the supervisor-facing status report generated from the canonical truth stack and the latest authoritative Phase B report.
- Added `reports/2026/03/phase_b_supervisor_report_2026-03-12.html` as a sendable HTML companion of the same package for forwarding or print export.
- The report package records the current authoritative run (`v15_2026-02-23_200258`), the exact red scientific metrics (`mutual_information=0.0435`, `cka_mean=0.0196`), the March 11 hardening work, the no-checkpoint constraint, and the bounded next action of one fresh Colab rerun.
- Fresh verification on 2026-03-12:
  - `git rev-parse HEAD` -> `954588c3bffa01e2e9559b0e54f7ff4d86e4d557`
  - `git status --short` -> clean worktree before report packaging
  - `python -m pytest -q` -> `56 passed`
- External provenance used in the report package:
  - Jupyter nbclient documentation for executed-notebook persistence: `https://github.com/jupyter/nbclient/blob/main/docs/client.rst`

**Hypotheses**
- A supervisor-facing package that cleanly separates current facts from hypotheses and inferences will reduce the risk of the March 11 engineering hardening being mistaken for a scientific pass.

**Inferences**
- The repo now contains a sendable supervisor artifact that can be shared without restitching the project state by hand.

---

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
