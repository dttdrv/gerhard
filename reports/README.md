# Gerhard Reports

## Purpose
This folder is the operational report system for autonomous execution tracking.

## Contents
1. `index.md` - newest-first list of runs, decisions, and links.
2. `templates/` - canonical report templates used by automation/manual reporting.
3. `<YYYY>/<MM>/` - per-run report outputs (`.md` and `.json`).

## Contract
Each run must produce:
1. One Markdown report.
2. One JSON report.
3. One self-contained dossier under `outputs/<run_id>/run_dossier_<run_id>.html`.
4. Deterministic decision line in Markdown:
   - `AUTOPILOT_DECISION: CONTINUE`
   - `AUTOPILOT_DECISION: PAUSE_NEEDS_INPUT`

Preferred laptop-side ingestion command:

`python scripts/register_dossier_run.py --dossier <path_to_run_dossier_<run_id>.html> --phase B`

Fallback artifact-folder ingestion:

`python scripts/register_notebook_run.py --run-id <run_id> --phase <A..K> --source-dir <notebook_output_dir>`

For the current Phase B finish flow, keep notebook-side registration disabled and register on the laptop after the dossier is exported.

## Source Of Truth Priority
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

## Handoff Memory
- `docs/ops/TIME_CAPSULE.md` is the living handoff memory and remains subordinate to the four canonical current-state sources above.

## Pointer Document
- `PROJECT_BRIEF.md` is the onboarding pointer for mission, constraints, and source-order discovery.
- `PROJECT_BRIEF.md` is not a co-equal state file and must not override run/state artifacts.

## Historical Evidence Note
- Archived February 2026 run reports remain valid historical records.
- They predate the March 11 reproducibility-gate tightening and should not be treated as reproducibility-clean without that caveat.
