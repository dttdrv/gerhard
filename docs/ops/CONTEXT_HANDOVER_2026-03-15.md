# Gerhard Context Handover

Prepared: 2026-03-15  
Purpose: provide a dated resume-from-here snapshot for the next operator, agent, or reviewer. This file is a point-in-time handover note. `docs/ops/TIME_CAPSULE.md` remains the living memory record.

## Read This First
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`
5. `docs/ops/TIME_CAPSULE.md`
6. `PROJECT_BRIEF.md`

## Current Authoritative State
- Fact: the active phase is `B_v15_spikingbrain_validation`.
- Fact: the latest authoritative run is `v15_2026-02-23_200258`.
- Fact: the current decision is `PAUSE_NEEDS_INPUT`.
- Fact: Phase B is blocked on scientific thresholds, not on missing artifacts.
- Fact: the red gate is `phase_b_scientific_thresholds`.
- Fact: the failing observed values are `mutual_information=0.0435` and `cka_mean=0.0196`.
- Fact: required artifacts, phase-specific artifacts, and inspected numeric sanity were green on the authoritative run.

## What Changed Since The February Authoritative Run
- Fact: the repo-side evidence pipeline was tightened on 2026-03-11.
- Fact: `scripts/register_notebook_run.py` now treats missing commit/fingerprint evidence as a deterministic red reproducibility gate.
- Fact: `scripts/register_dossier_run.py` now accepts both consolidated and raw-payload dossier shapes, rejects unsafe `run_id` values, and rebuilds clean staging on repeated ingest.
- Fact: `src/evaluation/spiking_brain.py` now accepts repo-native `TeacherModel` activations and aggregates MI/CKA across mapped layers and both `k` and `v` spike channels in the notebook-parity core path.
- Fact: the repo now has three explicit notebook execution surfaces:
  - `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` for the current no-checkpoint fresh-rerun case
  - `notebooks/asnn_goose_v15_runpod_operator.ipynb` for checkpoint-only follow-on diagnosis
  - `notebooks/asnn_goose_v15_reset_master.ipynb` as the canonical checkpoint-gated validation notebook
- Fact: the repo now also has a supervisor-ready status package:
  - `reports/2026/03/phase_b_supervisor_report_2026-03-12.md`
  - `reports/2026/03/phase_b_supervisor_report_2026-03-12.html`

## What Is Blocked Right Now
- Fact: there is no committed `v14.3` or `v15` checkpoint in the repo.
- Fact: that means the checkpoint-preflight flow is not the default next step.
- Fact: no new rerun has been executed yet under the March 11 fresh-rerun launcher plus laptop-side dossier-registration flow.
- Fact: the February 2026 run remains the best scientific evidence available, but it predates the tightened reproducibility contract and should be treated as pre-tightening evidence for provenance.

## Trusted Components
- `reports/index.md`, `state/program_status.yaml`, `state/gate_results.yaml`
- `docs/ops/STATUS_BOARD.md`
- `docs/ops/TIME_CAPSULE.md`
- `PROJECT_BRIEF.md`
- `scripts/register_notebook_run.py`
- `scripts/register_dossier_run.py`
- `src/evaluation/spiking_brain.py`
- `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb`
- `notebooks/asnn_goose_colab_v15.ipynb`

## Drifted Or Conditional Components
- `notebooks/asnn_goose_v15_reset_master.ipynb`
  - Fact: still checkpoint-gated.
  - Inference: do not treat it as the default next notebook while no checkpoint exists.
- February 2026 archived bundles and reports
  - Fact: still authoritative historical evidence.
  - Fact: not reproducibility-clean under the tightened March 11 contract.
- `src/models/asnn_goose.py::from_teacher()`
  - Fact: still listed in living handoff memory as needing a later compatibility pass.

## Default Next Action
1. Run `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` on Colab.
2. Let it execute `notebooks/asnn_goose_colab_v15.ipynb` with notebook-side registration disabled.
3. Require the full deep evidence bundle plus `v15_best.pt`.
4. Bring the bundle back to the laptop.
5. Register locally with `scripts/register_dossier_run.py`.
6. Re-read the canonical truth files and stop.

## Expected Deep Evidence Bundle
- `run_dossier_<run_id>.html`
- `eval_suite.json`
- `metrics.json`
- `config.yaml`
- `seed.txt`
- `v15_spikingbrain.json`
- `results.json`
- `v15_best.pt`
- `environment.json`
- `training_curves.json`
- `hardware_stats.json`
- `spike_analysis.json`
- `validation_tests.json`
- `control_suite.json`
- `checkpoint_metadata.json`
- `figures_index.json`
- `detailed_results.json`
- `artifact_manifest.json`
- `executed_training_notebook.ipynb`
- `operator_env.json`
- `launcher_bundle_manifest.json`

## What Not To Do
- Do not claim Phase B is improved from theory alone.
- Do not treat the March 11 engineering hardening as a scientific pass.
- Do not run the reset-notebook checkpoint path as the default next step unless a real checkpoint exists.
- Do not mutate archived February 2026 evidence to make it look provenance-clean after the fact.
- Do not use `trivy` in this thread; it is explicitly disabled by user instruction.

## Facts / Hypotheses / Inferences

### Facts
- The program is blocked in Phase B.
- The current best historical model metric remains `v14.3 PPL 306.89`.
- The latest authoritative Phase B run is red on MI and CKA.
- The repo-side rerun infrastructure is materially stronger than it was on February 23.

### Hypotheses
- The next fresh rerun will be easier to diagnose because of the deeper evidence bundle.
- The next rerun may still fail the scientific threshold even if all tooling works cleanly.

### Inferences
- The next honest unit of work is one bounded fresh rerun, not another local interpretation pass.
- Any status upgrade must come from a new registered run, not from argument.

## Verification Baseline At Handover
- Fact: `reports/index.md`, `state/program_status.yaml`, `state/gate_results.yaml`, `docs/ops/STATUS_BOARD.md`, `docs/ops/TIME_CAPSULE.md`, and `PROJECT_BRIEF.md` were re-read while preparing this handover.
- Fact: repo head at handover start was `3626d420dc4c6c6e9b2ed4413df85d5255803d75`.
- Fact: the worktree was clean at handover start.
- Fact: the living time source for this handover timestamp is `2026-03-15T21:33:05+02:00` (`Europe/Sofia`).

## If You Need A Human-Facing Summary Instead
Use the supervisor package first:
- `reports/2026/03/phase_b_supervisor_report_2026-03-12.md`
- `reports/2026/03/phase_b_supervisor_report_2026-03-12.html`
