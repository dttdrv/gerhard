# Gerhard Phase B Supervisor Report

Prepared: 2026-03-12  
Prepared from canonical repo state at commit `954588c3bffa01e2e9559b0e54f7ff4d86e4d557`

## 1. Executive Summary

Gerhard is currently blocked in **Phase B: v15 SpikingBrain validation**. The latest authoritative run remains `v15_2026-02-23_200258`, and it failed the scientific gate on two key metrics:

- `mutual_information = 0.0435` against a threshold of `> 0.10`
- `cka_mean = 0.0196` against a threshold of `> 0.30`

This is a real scientific block, not an artifact-presence failure. The run produced the expected files, and inspected numeric outputs were finite. The block is specifically that the learned spike representations did not yet meet the information-carrying and representation-alignment thresholds required for Phase B.

Since that run, the repository has been tightened significantly on evidence integrity, notebook execution control, dossier ingestion, and rerun packaging. The repo is now in a materially better state to generate a fresh authoritative rerun with a much deeper evidence bundle. However, **no new scientific run has been executed since the February 23 result**, so the program state remains blocked until a fresh rerun is completed and registered locally.

## 2. Current Authoritative Status

### Canonical source order
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

### Current status from canonical sources
- Current phase: `B_v15_spikingbrain_validation`
- Latest authoritative run: `v15_2026-02-23_200258`
- Current decision: `PAUSE_NEEDS_INPUT`
- Program status: `Phase B blocked`
- Blocking reason: `Phase B failed scientific thresholds. Provide a new run after model/config changes.`

### Gate outcome on the authoritative run

| Gate | Status | Evidence |
|------|--------|----------|
| `required_artifacts_presence` | green | all required artifacts present |
| `phase_b_artifacts_presence` | green | all phase-specific artifacts present |
| `metrics_numerical_sanity` | green | numeric values finite in inspected JSON artifacts |
| `phase_b_scientific_thresholds` | red | `mutual_information=0.0435 <= 0.10; cka_mean=0.0196 <= 0.30` |

## 3. What The February Run Proved

The February 23 authoritative dossier already established the following:

- The Phase B evaluation path was capable of producing a complete dossier and the expected Phase B artifact set.
- The failure was not caused by missing files.
- The inspected JSON outputs were numerically sane.
- The scientific thresholds still failed, specifically on mutual information and CKA.

What the February run did **not** prove:

- It did not satisfy the March 11 tightened reproducibility contract, because the archived report JSON still records `commit: unknown`.
- It did not test the March 11 evaluator seam repairs.
- It did not produce the deeper post-run evidence bundle now required for detailed diagnosis.

## 4. Engineering Work Completed Since The Latest Authoritative Run

The repository was materially hardened on 2026-03-11. These changes improved evidence integrity and rerun readiness, but they did not themselves change the scientific state.

### 4.1 Evidence and reporting hardening
- `scripts/register_notebook_run.py` now emits a deterministic red reproducibility gate when commit or fingerprint metadata is missing.
- `scripts/register_dossier_run.py` now accepts both consolidated dossiers and reset-notebook raw-payload dossiers, rejects unsafe `run_id` values, and rebuilds clean staging on repeated ingestion.
- `PROJECT_BRIEF.md` and `docs/ops/TIME_CAPSULE.md` were added to keep human and agent handoffs aligned with the existing truth stack rather than replacing it.

### 4.2 Evaluator seam repair
- `src/evaluation/spiking_brain.py` now accepts both Hugging Face-style teacher outputs and the repo-native `TeacherModel` activation format.
- The MI/CKA collection path now aggregates across mapped layers and both `k` and `v` spike channels in the same core shape as the reset notebook.
- Focused regression coverage was added in `tests/test_spiking_brain.py`.

### 4.3 Notebook execution control
- `notebooks/asnn_goose_v15_runpod_operator.ipynb` was added as a thin checkpoint-validation launcher that executes the canonical reset notebook without duplicating research logic.
- `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` was added for the checkpoint-present Colab case.
- `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` was added as the default launcher for the current no-checkpoint case.

### 4.4 Fresh-rerun evidence expansion
The fresh-rerun notebook path now requires and packages a much deeper artifact bundle:

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

This is the main change needed to satisfy the requirement for “much more detailed results of absolutely everything.”

## 5. Current Verification State

Verification immediately before this report package:

- `git rev-parse HEAD` -> `954588c3bffa01e2e9559b0e54f7ff4d86e4d557`
- `git status --short` -> clean worktree before creating this report package
- `python -m pytest -q` -> `56 passed`

These checks support the claim that the repo is engineering-ready for the next experiment. They do **not** change the scientific status, which remains blocked until a fresh rerun is executed.

## 6. Why A Fresh Rerun Is Now Required

The current no-checkpoint reality matters.

- There is no committed `v14.3` or `v15` checkpoint in the repo.
- The reset-notebook preflight path is therefore not the default next step.
- The honest next experiment is a fresh rerun that generates both a new dossier and a new checkpoint.

The default execution surface is now:

- launcher: `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb`
- training notebook: `notebooks/asnn_goose_colab_v15.ipynb`
- registration path: laptop-side `scripts/register_dossier_run.py`

## 7. Recommended Next Action

Run one fresh Colab rerun and stop after local registration.

### Execution sequence
1. Execute `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` on Colab GPU.
2. Let it clone the repo, inject the controlled env overrides, and execute `notebooks/asnn_goose_colab_v15.ipynb`.
3. Require the full deep evidence bundle and bundled `v15_best.pt`.
4. Bring the bundle back to the laptop.
5. Register locally with `python scripts/register_dossier_run.py --dossier <path_to_run_dossier_<run_id>.html> --phase B`.
6. Re-read `reports/index.md`, `state/program_status.yaml`, `state/gate_results.yaml`, and `docs/ops/STATUS_BOARD.md`.
7. Stop and interpret the new authoritative result before starting any further patch cycle.

### Success condition for the next experiment
- The rerun completes structurally.
- The deep evidence bundle is complete.
- The rerun produces `v15_best.pt`.
- The dossier registers locally without manual repair.
- The canonical state files reflect the new run coherently.

## 8. Supervisor Decision Points

The program does not currently need a new architectural proposal. It needs approval and/or prioritization for one clean rerun under the tightened evidence contract.

Recommended supervisor-level decisions:

1. Approve a single fresh Colab rerun as the next authoritative experiment.
2. Treat the current repository state as engineering-ready but scientifically blocked.
3. Require the next run to be evaluated from the registered dossier and state files, not from notebook impressions alone.
4. Treat the February evidence as historically useful but not reproducibility-clean under the tightened March 11 standard.

## 9. Facts, Hypotheses, Inferences

### Facts
- The latest authoritative run is `v15_2026-02-23_200258`.
- That run is blocked on `mutual_information=0.0435` and `cka_mean=0.0196`.
- Artifact presence and inspected numeric sanity were green on that run.
- The repo currently contains no committed checkpoint suitable for the checkpoint-only preflight path.
- The repo now has a fresh-rerun Colab launcher and a materially richer evidence-bundle contract.
- The repository test suite passed at `56 passed` immediately before this report package.

### Hypotheses
- The evaluator seam repairs and evidence hardening reduce the chance that the next failure is caused by tooling drift rather than actual model behavior.
- The deeper evidence bundle will reduce ambiguity during post-run diagnosis.
- The next rerun may still fail the scientific thresholds; if that happens, the blocker is scientific rather than primarily measurement-related.

### Inferences
- No honest status upgrade is possible without a fresh rerun.
- The repo is ready for that rerun from an engineering and evidence-integrity perspective.
- The correct next unit of work is one bounded rerun, not another local patch cycle.

## 10. External Provenance Note

The current launcher approach executes notebooks programmatically and preserves the executed notebook artifact. This is consistent with official Jupyter notebook-execution guidance: executed notebooks can be saved with outputs after programmatic execution through the Jupyter execution stack. Source used for this external note:

- Jupyter nbclient documentation: https://github.com/jupyter/nbclient/blob/main/docs/client.rst

This external note is supportive only. The authoritative project state remains the repo-local truth stack listed above.

## 11. Report Attachments / Source Pointers

Primary sources:
- `reports/index.md`
- `state/program_status.yaml`
- `state/gate_results.yaml`
- `docs/ops/STATUS_BOARD.md`
- `docs/ops/TIME_CAPSULE.md`
- `PROJECT_BRIEF.md`
- `reports/2026/02/v15_2026-02-23_200258.md`
- `reports/2026/02/v15_2026-02-23_200258.json`
- `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`

Relevant current notebook surfaces:
- `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb`
- `notebooks/asnn_goose_colab_v15.ipynb`
- `notebooks/asnn_goose_v15_reset_master.ipynb`

## 12. Bottom Line

The program is not stalled because the repo is disorganized. It is blocked because the last authoritative scientific result was red, and no new authoritative rerun has yet been executed under the improved evidence contract.

The engineering side is now sufficiently tightened to justify one fresh rerun. That is the next honest step.
