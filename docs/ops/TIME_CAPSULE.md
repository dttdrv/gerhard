# Gerhard Time Capsule

## current mission
- Fact: complete Phase 0 Cycle 1 truth + gate drift alignment without changing model math.
- Fact: keep the existing canonical truth stack primary and add handoff memory around it, not instead of it.
- Fact: `src/evaluation/spiking_brain.py` now accepts both Hugging Face-style `hidden_states` outputs and repo-native `TeacherModel` `layer_activations` for representation collection.
- Fact: `src/evaluation/spiking_brain.py` now aggregates MI/CKA across mapped layers and both `k` / `v` spike channels in the same core shape as the reset notebook.
- Fact: the Phase B finish path is now aligned around the reset notebook on RunPod plus laptop-side dossier registration after export.
- Inference: the next capability increase must come from checkpoint-only `SMOKE` / `DIAGNOSE` runs, not a new expensive rerun.

## current blocker
- Fact: the latest authoritative run `v15_2026-02-23_200258` is blocked on Phase B scientific thresholds: `mutual_information=0.0435` and `cka_mean=0.0196`.
- Fact: archived February 2026 reports were created before the March 11 reproducibility-gate tightening and therefore do not carry the required commit/fingerprint evidence in their checked artifacts.
- Fact: no new RunPod `SMOKE` / `DIAGNOSE` / `FULL` execution has been performed yet under the reset-notebook-plus-dossier flow.
- Hypothesis: now that the teacher-interface seam and core MI/CKA aggregation are repaired, the next blocker to falsify is whether the scientific failure survives checkpoint-only preflight or whether remaining drift is confined to notebook control-suite logic.

## latest authoritative run
- Run ID: `v15_2026-02-23_200258`
- Timestamp UTC: `2026-02-23T20:21:53.512775Z`
- Phase: `B_v15_spikingbrain_validation`
- Evidence:
  - `reports/2026/02/v15_2026-02-23_200258.md`
  - `reports/2026/02/v15_2026-02-23_200258.json`
  - `outputs/v15_2026-02-23_200258/run_dossier_v15_2026-02-23_200258.html`
- Note: this run remains the latest authoritative scientific evidence, but it is pre-tightening evidence for reproducibility metadata.

## latest decision
- Fact: the latest authoritative run decision is `PAUSE_NEEDS_INPUT`.
- Fact: the March 11 engineering decision is to bridge the existing truth stack, add `TIME_CAPSULE.md` and `PROJECT_BRIEF.md`, tighten the registration gate on reproducibility metadata, and defer model changes.
- Fact: the March 11 preflight decision is to land only the narrow collector-seam adapter in `src/evaluation/spiking_brain.py`, defer MI/CKA math changes, and use cheap checkpoint-only validation to determine whether further parity work is necessary.
- Fact: the March 11 finish-path decision is to use `notebooks/asnn_goose_v15_reset_master.ipynb` as the canonical execution notebook, keep notebook-side registration off, and prefer laptop-side dossier ingestion after export.
- Fact: dossier ingestion is now hardened to accept both consolidated and reset-notebook dossier shapes, reject unsafe `run_id` values, and rebuild a clean staging directory on repeated ingest.
- Fact: `notebooks/asnn_goose_v15_runpod_operator.ipynb` now exists as a thin RunPod operator notebook that executes the canonical reset notebook via `jupyter nbconvert` instead of forking the research logic.
- Fact: `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` now exists as the concrete Colab T4 one-cell launcher with the repo remote prefilled and checkpoint auto-discovery.
- Fact: `trivy` is intentionally disabled for the remainder of this thread by explicit user instruction; do not schedule further `trivy` actions here.

## canonical source order
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

Strategy context lives in:
- `docs/GERHARD_MASTER_PLAN_WITH_ADDON_2026-02-23.md`
- `knowledge/roadmap.md`
- `PROJECT_BRIEF.md`

## trusted components
- `reports/index.md`, `state/program_status.yaml`, and `state/gate_results.yaml` for current run/state truth.
- `scripts/register_notebook_run.py`, `scripts/register_dossier_run.py`, `tests/test_register_notebook_run.py`, and `tests/test_register_dossier_run.py` after the March 11 truth-alignment and dossier-hardening cycle.
- `notebooks/asnn_goose_v15_reset_master.ipynb` as the canonical rerun notebook.
- `notebooks/asnn_goose_v15_runpod_operator.ipynb` as the thin execution helper for RunPod staging.
- `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` as the concrete Colab T4 launcher.
- `src/evaluation/spiking_brain.py` collector compatibility path plus `tests/test_spiking_brain.py` for teacher-interface normalization and core MI/CKA aggregation parity.
- `src/models/goose_backbone.py`, `src/models/ternary_activations.py`, and `src/models/lora_adapter.py` as the best-evidenced model-side modules from current test coverage.

## broken/drifted/scaffold components
- Broken or drifted:
  - `src/evaluation/spiking_brain.py` now supports repo `TeacherModel` and core `k`/`v` aggregation parity, but the reset notebook's broader control-suite parity is still not ported.
  - `src/models/asnn_goose.py::from_teacher()` still needs a compatibility pass in the next cycle.
  - Archived February 2026 reports and bundles are truthful historical evidence but not reproducibility-clean under the tightened contract.
  - `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md` and status records are now aligned to the reset notebook, but no new authoritative RunPod checkpoint preflight has been executed yet under that tightened flow.
- Scaffold:
  - `eval/`
  - `retrieval/`
  - `verifiers/`
  - `data/mixture.yaml`

## active hypotheses
- Hypothesis: the collector-seam incompatibility and core MI/CKA aggregation drift were real blockers, and repairing them enables meaningful checkpoint-only preflight on repo-native teacher outputs.
- Hypothesis: the reporting path was a real source of evidence drift, and tightening it now will prevent false-green progress signals later.
- Hypothesis: even after the adapter path works, MI/CKA may still remain red; if so, the blocker is scientific rather than measurement-only.

## next actions
1. Run checkpoint-only `SMOKE` from either `notebooks/asnn_goose_v15_runpod_operator.ipynb` on RunPod or `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` on Colab T4, with `GERHARD_ENABLE_REGISTER_RUN=0`.
2. If `SMOKE` is structurally clean, switch the operator notebook to `DIAGNOSE`; if that is structurally clean, continue to `FULL`.
3. Bring the dossier and artifact bundle back to the laptop and register locally with `scripts/register_dossier_run.py`.
4. If cheap validation still shows structural evaluator drift, patch notebook control-suite parity before any further rerun.

## open questions
- Should the notebook emit package-version metadata directly, or is commit + fingerprint sufficient for Phase 0?
- Do the current MI/CKA thresholds need scientific recalibration after the repo-native adapter and control-suite parity are in place?
- Should the next authoritative run confirm that the reset notebook is now the only execution notebook and `asnn_goose_colab_v15.ipynb` is historical evidence only?

## last updated
- 2026-03-11 15:02:00 Europe/Sofia
