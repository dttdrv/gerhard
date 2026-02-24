# Gerhard Reporting Contract

## Purpose
Standardize run inputs and report outputs so user interaction is only needed on red gates.

## Notebook-First Execution Policy
All primary execution is notebook-driven (RunPod/Colab/Kaggle notebook passes).
The repo automation layer does not replace notebook execution; it ingests notebook outputs after the run.

Ingestion command (from repo root):

`python3 scripts/register_notebook_run.py --run-id <run_id> --phase <A..K> --source-dir <notebook_output_dir>`

Notebook API option:
`from scripts.register_notebook_run import register_run`

RunPod handoff reference:
`docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`

## Required Artifact Layout
Per run, artifacts must live under:

`outputs/<run_id>/`

Minimum required files:
1. `eval_suite.json` (or current phase equivalent evaluation output)
2. `metrics.json`
3. `config.yaml`
4. `seed.txt` (or seed embedded in config)
5. phase-specific report file (for example `v15_spikingbrain.json`)
6. single-file detailed dossier (recommended): `run_dossier_<run_id>.html`

## Per-Run Report Outputs
1. Markdown report:
   - `reports/<YYYY>/<MM>/<run_id>.md`
2. JSON report:
   - `reports/<YYYY>/<MM>/<run_id>.json`
3. Notebook dossier (self-contained evidence + figures):
   - `outputs/<run_id>/run_dossier_<run_id>.html`

## Report Structure (Two-Tier)
### Tier 1: Executive
1. What changed.
2. Pass/fail summary.
3. Risks.
4. Next autonomous action.
5. Whether user input is required.

### Tier 2: Technical Evidence
1. Metric deltas vs baseline.
2. Gate scorecard with observed vs thresholds.
3. Artifact paths and checks.
4. Command/config fingerprints and reproducibility metadata.

Mandatory final line:
`AUTOPILOT_DECISION: CONTINUE` or `AUTOPILOT_DECISION: PAUSE_NEEDS_INPUT`

## Machine-Readable JSON Fields
Required top-level fields:
1. `run_id`
2. `timestamp_utc`
3. `commit`
4. `phase`
5. `executive_summary`
6. `metric_deltas`
7. `gate_scorecard`
8. `artifacts`
9. `decision`
10. `next_steps`
11. `needs_user_input`

## Required State Updates Per Run
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

The ingestion script updates (1)-(4) automatically and writes canonical run reports.

## Pause Contract
When decision is `PAUSE_NEEDS_INPUT`, create a concrete request file:
- `reports/<YYYY>/<MM>/<run_id>_needs_input.md`

This file must contain:
1. Exact missing outputs/decisions needed.
2. Why they are needed (gate context).
3. Expected location/format for provided outputs.
