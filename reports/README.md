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

Notebook-first ingestion command:

`python3 scripts/register_notebook_run.py --run-id <run_id> --phase <A..K> --source-dir <notebook_output_dir>`

## Source Of Truth Priority
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`
