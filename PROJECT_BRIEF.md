# Gerhard Project Brief

## Mission
- Gerhard is a notebook-first research program around ASNN-Goose, not a generic application repo.
- The immutable core chain remains `v15 -> v16 -> v17 -> v18 -> v19`.
- Phase 0 Cycle 1 is focused on truth + gate drift alignment, not on new model behavior.

## Constraints
- Preserve evidence integrity over apparent progress.
- Prefer cheap verification before expensive notebook or GPU runs.
- Keep the existing canonical state stack primary during this cycle.

## Canonical Source Order
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

`docs/GERHARD_MASTER_PLAN_WITH_ADDON_2026-02-23.md` and `knowledge/roadmap.md` remain the strategy sources for sequencing and scope.

## Operating Notes
- `docs/ops/TIME_CAPSULE.md` is the living handoff memory. It must be updated after meaningful actions, but it does not override run artifacts or state files.
- `changelog.md` now carries dated operational entries in addition to the historical version log.
- Archived February 2026 runs remain historical evidence, but they predate the March 11 reproducibility-gate tightening and should not be treated as reproducibility-clean.

## Deferred Root State Files
- Root `STATE.yaml` is intentionally deferred in Phase 0 Cycle 1.
- Root `LOG.md` is intentionally deferred in Phase 0 Cycle 1.

## Current Bounded Next Step
- Fact: there is no committed `v14.3` / `v15` checkpoint in this repo, so the checkpoint-only preflight path cannot run as the default next step.
- Run a fresh rerun from `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` on Colab. That launcher executes `notebooks/asnn_goose_colab_v15.ipynb` with notebook-side registration disabled and exports a dossier bundle for laptop-side registration.
- The fresh rerun now emits a detailed evidence bundle, not just the minimum registration files: environment snapshot, raw training curves, validation tests, control-suite payload, figures index, detailed results, artifact manifest, executed notebook copy, and launcher env snapshot.
- Register the resulting dossier locally on the laptop with `scripts/register_dossier_run.py`, then read the canonical truth files and stop.
- After the fresh rerun produces `v15_best.pt`, the checkpoint-only reset notebook path becomes usable again for follow-on diagnosis if needed.
