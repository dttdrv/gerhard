# Gerhard Status Board

**As Of**: 2026-03-11  
**Autopilot Mode**: Conditional (continue on green, pause on red)

## Executive Snapshot
- Core chain lock: `v15 -> v16 -> v17 -> v18 -> v19` (unchanged)
- Execution mode: notebook-first; repo automation ingests notebook outputs post-run
- Current active phase: **B_v15_spikingbrain_validation**
- Latest known best: **v14.3 PPL 306.89**
- Latest authoritative run remains `v15_2026-02-23_200258`, but it is pre-tightening evidence for reproducibility metadata.
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

## Preflight Adapter Update (2026-03-11)
- `src/evaluation/spiking_brain.py` now accepts both Hugging Face-style `hidden_states` outputs and repo-native `TeacherModel` `layer_activations` during representation collection.
- `src/evaluation/spiking_brain.py` now aggregates MI/CKA across mapped layers and both `k` / `v` spike channels in the same core shape as the reset notebook.
- `tests/test_spiking_brain.py` now covers repo-native teacher compatibility, preserved Hugging Face-style compatibility, single-dict spike normalization, and the notebook-parity MI/CKA aggregation path.
- The next bounded step is now checkpoint-only `SMOKE` / `DIAGNOSE` validation against an existing checkpoint, not another adapter build pass.

## Live Engineering Update (2026-03-11)
- Canonical execution notebook: `notebooks/asnn_goose_v15_reset_master.ipynb`
- Historical notebook retained for evidence only: `notebooks/asnn_goose_colab_v15.ipynb`
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
- Reset-notebook fingerprint logic exists in `notebooks/asnn_goose_v15_reset_master.ipynb`, but the checked February 2026 archived artifacts do not yet satisfy the tightened reproducibility contract.
- Latest execution remains the RunPod-ingested `v15_2026-02-23_200258`.
- Canonical archived bundle for that run is limited to the checked files under `outputs/v15_2026-02-23_200258/`; do not assume missing fingerprint fields or `figures_detailed/*` exist unless a newer run proves it.
- Single-file dossier export remains the preferred evidence bundle for future runs: `outputs/<run_id>/run_dossier_<run_id>.html`, with laptop-side registration preferred over notebook-side registration for this phase finish flow.

## Phase Status (A-K)
| Phase | Name | Status | Owner State |
|------|------|--------|-------------|
| A | Engineering Guardrails | IN PROGRESS | Scaffolds in place; full reproducibility guardrails pending. |
| B | v15 SpikingBrain Validation | BLOCKED | Latest full rerun still below MI/CKA thresholds; repo-native preflight adapter landed and checkpoint-only smoke/diagnose is now pending. |
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

## Next Autonomous Actions
1. Run checkpoint-only `SMOKE` on RunPod from `notebooks/asnn_goose_v15_runpod_operator.ipynb`, with notebook-side registration off.
2. If `SMOKE` is structurally clean, switch the operator notebook to `DIAGNOSE`; if that is structurally clean, continue to `FULL`.
3. Bring the dossier and artifact bundle back to the laptop and register locally with `scripts/register_dossier_run.py`.
4. Read `reports/index.md`, `state/program_status.yaml`, `state/gate_results.yaml`, and `docs/ops/STATUS_BOARD.md`, then stop.

Colab alternative:
- Run the same staged flow from `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` on a T4 instance if Colab is the preferred execution surface.

## Key Links
- Operating model: `docs/ops/AUTONOMY_OPERATING_MODEL.md`
- Gate policy: `docs/ops/GATE_POLICY.md`
- Reporting contract: `docs/ops/REPORTING_CONTRACT.md`
- RunPod notebook handoff: `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`
- Program state: `state/program_status.yaml`
- Gate state: `state/gate_results.yaml`
- Reports index: `reports/index.md`
- Time capsule: `docs/ops/TIME_CAPSULE.md`
