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
- Run checkpoint-only `SMOKE` on RunPod from `notebooks/asnn_goose_v15_runpod_operator.ipynb`, which executes `notebooks/asnn_goose_v15_reset_master.ipynb` with an existing checkpoint and `GERHARD_ENABLE_REGISTER_RUN=0`.
- Or run the same staged flow on Colab T4 from `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb`, which clones the repo if needed and auto-discovers a likely checkpoint.
- If `SMOKE` is structurally clean, rerun from a fresh kernel in `DIAGNOSE`, then continue to `FULL`.
- Register the resulting dossier locally on the laptop with `scripts/register_dossier_run.py`, then read the canonical truth files and stop.
