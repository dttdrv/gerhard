# Gerhard Status Board

**As Of**: 2026-03-12  
**Autopilot Mode**: Conditional (continue on green, pause on red)

## Executive Snapshot
- Core chain lock: `v15 -> v16 -> v17 -> v18 -> v19` (unchanged)
- Execution mode: notebook-first; repo automation ingests notebook outputs post-run
- Current active phase: **B_v15_spikingbrain_validation**
- Latest known best: **v14.3 PPL 306.89**
- Latest authoritative run remains `v15_2026-02-23_200258`, but it is pre-tightening evidence for reproducibility metadata.
- No committed `v14.3` / `v15` checkpoint exists in the repo, so the reset-notebook checkpoint-preflight path is not the default next step anymore.
- Add-on scaffolds:
  - `eval/` scaffold: present
  - `data/mixture.yaml`: present
  - `verifiers/` stubs: present
  - `retrieval/` stub: present
- Key blocker: Phase B scientific thresholds failed on latest run (MI, CKA).

<!-- AUTOGEN_LATEST_RUN_START -->
## Latest Run Update
- Run ID: `v15_2026-02-23_200258`
- Timestamp UTC: `2026-02-23T20:21:53.512775Z`
- Phase: `B_v15_spikingbrain_validation`
- Decision: `PAUSE_NEEDS_INPUT`
- Red gates: `1`
- Yellow gates: `0`
- Next action: Address red gates and provide the requested outputs in the needs_input report.
<!-- AUTOGEN_LATEST_RUN_END -->

## Truth Alignment Update (2026-03-11)
- `PROJECT_BRIEF.md` now points agents and humans to the existing canonical truth stack instead of introducing a second state system.
- `docs/ops/TIME_CAPSULE.md` is now the living handoff memory and must be updated after meaningful actions.
- `scripts/register_notebook_run.py` now treats missing artifact-provided commit/fingerprint metadata as a deterministic red reproducibility gate for new registrations.
- `tests/test_register_notebook_run.py` now reflects the nested Phase B artifact schema and the reproducibility gate.
- Archived February 2026 run bundles and reports were not rewritten; they remain historical evidence and must be described as pre-tightening evidence.

## Dossier Ingestion Hardening (2026-03-11)
- `scripts/register_dossier_run.py` now accepts both consolidated dossier payloads and the reset notebook's embedded raw-payload format.
- Dossier ingestion now rejects unsafe `run_id` values, rebuilds a clean staging directory on repeated ingest, and emits clean operator-facing errors for malformed dossier inputs.
- `tests/test_register_dossier_run.py` now covers reset-notebook dossier ingestion, consolidated dossier ingestion, unsafe `run_id` rejection, and repeated-ingest staging cleanup.
- Preferred local registration remains dossier-first, but only with artifact-provided commit + fingerprint metadata.

## RunPod Operator Notebook (2026-03-11)
- Added `notebooks/asnn_goose_v15_runpod_operator.ipynb` as a thin RunPod control notebook.
- It sets Phase B parameters and executes `notebooks/asnn_goose_v15_reset_master.ipynb` through `jupyter nbconvert`, so the reset notebook remains the only source of research logic.
- `tests/test_runpod_operator_notebook.py` now checks that the operator notebook still targets the canonical reset notebook and keeps notebook-side registration off.

## Colab T4 Single-Cell Notebook (2026-03-11)
- Added `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` as the concrete Colab launcher notebook.
- It clones `https://github.com/dttdrv/gerhard.git` when needed, mounts Google Drive when appropriate, auto-discovers a likely v15/v14.3 checkpoint, keeps notebook-side registration off, and executes the canonical reset notebook in one cell.
- `tests/test_colab_t4_single_cell_notebook.py` now verifies that the notebook stays single-cell, targets the canonical reset notebook, and keeps dossier-first local registration assumptions intact.

## Fresh-Rerun Colab Launcher (2026-03-11)
- Added `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` as the default fresh-rerun launcher when no checkpoint exists.
- The launcher executes `notebooks/asnn_goose_colab_v15.ipynb` in a clean subprocess, injects `GIT_COMMIT` plus env-controlled training knobs, keeps notebook-side registration off, and zips the run artifact directory for laptop-side ingestion.
- `notebooks/asnn_goose_colab_v15.ipynb` now honors env overrides for `SEED`, `distill_steps`, `batch_size`, `accumulation_steps`, `eval_interval`, `run_id`, notebook-side registration, and dossier auto-download.
- The fresh training notebook now copies `outputs/checkpoints/v15_best.pt` into the per-run artifact bundle so the rerun itself produces the next checkpoint evidence.
- The fresh training path now emits detailed evidence artifacts: `environment.json`, `training_curves.json`, `hardware_stats.json`, `spike_analysis.json`, `validation_tests.json`, `control_suite.json`, `checkpoint_metadata.json`, `figures_index.json`, `detailed_results.json`, `artifact_manifest.json`, plus launcher-copied `executed_training_notebook.ipynb`, `operator_env.json`, and `launcher_bundle_manifest.json`.
- `tests/test_colab_fresh_rerun_single_cell_notebook.py` now verifies the launcher target and the training notebook’s env-controlled registration/knob surface.

## Supervisor Report Package (2026-03-12)
- Added `reports/2026/03/phase_b_supervisor_report_2026-03-12.md` as the supervisor-facing narrative built directly from the canonical truth stack plus the latest authoritative run report.
- Added `reports/2026/03/phase_b_supervisor_report_2026-03-12.html` as the sendable HTML companion for forwarding or print export.
- The package states the current blocked scientific status, the March 11 hardening work, the no-checkpoint constraint, and the bounded next step of one fresh Colab rerun with the deep evidence bundle.

## Preflight Adapter Update (2026-03-11)
- `src/evaluation/spiking_brain.py` now accepts both Hugging Face-style `hidden_states` outputs and repo-native `TeacherModel` `layer_activations` during representation collection.
- `src/evaluation/spiking_brain.py` now aggregates MI/CKA across mapped layers and both `k` / `v` spike channels in the same core shape as the reset notebook.
- `tests/test_spiking_brain.py` now covers repo-native teacher compatibility, preserved Hugging Face-style compatibility, single-dict spike normalization, and the notebook-parity MI/CKA aggregation path.
- The next bounded step is now checkpoint-only `SMOKE` / `DIAGNOSE` validation against an existing checkpoint, not another adapter build pass.

## Live Engineering Update (2026-03-11)
- Canonical checkpoint-validation notebook: `notebooks/asnn_goose_v15_reset_master.ipynb`
- Current fresh-training notebook logic: `notebooks/asnn_goose_colab_v15.ipynb`
- Default fresh-rerun execution surface: `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb`
- Runtime hardening applied:
  - dependency bootstrap with optional plotting fallback,
  - `return_spike_info` support added in student/layer forward path,
  - v15 validator fixed for tuple/dict batch formats,
  - `val_dataloader` reference fixed to `val_loader`,
  - guards added for empty loaders and zero-token evaluation.
- Full-rerun remediation patch applied:
  - disabled `torch.compile` for stability with rich spike instrumentation,
  - added mixed token/channel thresholding in ternary spikes to reduce dead channels,
  - added spike semantic alignment loss (teacher-to-spike ternary target alignment),
  - expanded training logs with semantic-loss curves,
  - fixed validator bias by using both `k` and `v` spikes and robust MI discretization.
- Fresh-rerun launcher hardening applied:
  - repo clone + commit capture on Colab,
  - env-controlled training overrides for memory-constrained GPUs,
  - notebook-side registration disabled by default,
  - per-run checkpoint copy into the artifact bundle.
- Reset-notebook fingerprint logic exists in `notebooks/asnn_goose_v15_reset_master.ipynb`, but the checked February 2026 archived artifacts do not yet satisfy the tightened reproducibility contract.
- Latest execution remains the RunPod-ingested `v15_2026-02-23_200258`.
- Canonical archived bundle for that run is limited to the checked files under `outputs/v15_2026-02-23_200258/`; do not assume missing fingerprint fields or `figures_detailed/*` exist unless a newer run proves it.
- Single-file dossier export remains the preferred evidence bundle for future runs: `outputs/<run_id>/run_dossier_<run_id>.html`, with laptop-side registration preferred over notebook-side registration for this phase finish flow.

## Phase Status (A-K)
| Phase | Name | Status | Owner State |
|------|------|--------|-------------|
| A | Engineering Guardrails | IN PROGRESS | Scaffolds in place; full reproducibility guardrails pending. |
| B | v15 SpikingBrain Validation | BLOCKED | Latest full rerun still below MI/CKA thresholds; no repo checkpoint exists, so the next step is a fresh Colab rerun that produces a new dossier bundle and `v15_best.pt`. |
| C | Temporal Coding Proof | NOT STARTED | Waiting for B gate maturity. |
| D | v16 Sparse Ops | NOT STARTED | Blocked by B pass. |
| E | v17 Efficiency | NOT STARTED | Blocked by D readiness. |
| F | v18 Ablations | NOT STARTED | Blocked by D/E readiness. |
| G | v19 Publication/Repro | NOT STARTED | Blocked by F readiness. |
| H | Generalist Scorecard + Mixture | IN PROGRESS | Config scaffolds exist; baselines and gates pending. |
| I | Retrieval World-Fit | IN PROGRESS | Interface stub exists; runtime + eval gate pending. |
| J | Post-Training | NOT STARTED | Depends on gate infrastructure and data readiness. |
| K | Scaling Ladder | NOT STARTED | Depends on B/H/I/J gate maturity. |

## Current Red/Yellow Signals
1. RED: `phase_b_scientific_thresholds` failed on latest run (`mutual_information`, `cka_mean`).
2. YELLOW: archived February 2026 authoritative evidence predates the March 11 reproducibility-gate tightening; the next authoritative run must carry commit + fingerprint metadata cleanly.
3. YELLOW: scorecard baselines are not yet established for H regression gates.
4. YELLOW: no repo-local checkpoint exists, so the checkpoint-preflight notebook path is blocked pending a fresh rerun or recovered external checkpoint.
5. YELLOW: the next fresh rerun must satisfy the deep evidence-bundle contract, not just the minimal registration files.

## Next Autonomous Actions
1. Run `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` on Colab and let it execute `notebooks/asnn_goose_colab_v15.ipynb` with notebook-side registration off.
2. Bring the dossier, the full deep evidence bundle, and bundled `v15_best.pt` back to the laptop and register locally with `scripts/register_dossier_run.py`.
3. Read `reports/index.md`, `state/program_status.yaml`, `state/gate_results.yaml`, and `docs/ops/STATUS_BOARD.md`, then stop.
4. If the fresh rerun yields a usable checkpoint but Phase B is still structurally ambiguous, use `notebooks/asnn_goose_v15_runpod_operator.ipynb` plus `notebooks/asnn_goose_v15_reset_master.ipynb` for checkpoint-only follow-on diagnosis.

## Key Links
- Operating model: `docs/ops/AUTONOMY_OPERATING_MODEL.md`
- Gate policy: `docs/ops/GATE_POLICY.md`
- Reporting contract: `docs/ops/REPORTING_CONTRACT.md`
- RunPod notebook handoff: `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`
- Program state: `state/program_status.yaml`
- Gate state: `state/gate_results.yaml`
- Reports index: `reports/index.md`
- Time capsule: `docs/ops/TIME_CAPSULE.md`
