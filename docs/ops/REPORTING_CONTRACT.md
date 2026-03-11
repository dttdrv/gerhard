# Gerhard Reporting Contract

## Purpose
Standardize run inputs and report outputs so user interaction is only needed on red gates.

## Notebook-First Execution Policy
All primary execution is notebook-driven (RunPod/Colab/Kaggle notebook passes).
The repo automation layer does not replace notebook execution; it ingests notebook outputs after the run.

Preferred ingestion command (from repo root):

`python scripts/register_dossier_run.py --dossier <path_to_run_dossier_<run_id>.html> --phase B`

Fallback artifact-folder ingestion:

`python scripts/register_notebook_run.py --run-id <run_id> --phase <A..K> --source-dir <notebook_output_dir>`

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

## Reproducibility Metadata Gate
For new registrations, the report/gate path must surface reproducibility metadata explicitly:
1. a real git commit carried by the notebook-produced artifacts, and
2. fingerprint fields `config_sha256` and `recipe_sha256`.

If either item is missing, the registration must not silently serialize that gap as healthy evidence. It must emit a deterministic gate/report issue.
The gate name for this condition is `reproducibility_metadata`.
For the current Phase B finish flow, notebook-side registration should remain disabled and local laptop-side registration should be used after dossier export.

## Required State Updates Per Run
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`
5. `docs/ops/TIME_CAPSULE.md` (manual/agent update; handoff memory, not script-owned)
6. `changelog.md` (manual/agent update; dated operational record)

The ingestion script updates (1)-(4) automatically and writes canonical run reports.
The acting agent or operator must update (5) and (6) after meaningful actions.

## Source Of Truth Order
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

`PROJECT_BRIEF.md` is a pointer for humans and agents; it is not a co-equal runtime state file.
`docs/ops/TIME_CAPSULE.md` is the handoff memory and must stay subordinate to the four sources above.

## Pause Contract
When decision is `PAUSE_NEEDS_INPUT`, create a concrete request file:
- `reports/<YYYY>/<MM>/<run_id>_needs_input.md`

This file must contain:
1. Exact missing outputs/decisions needed.
2. Why they are needed (gate context).
3. Expected location/format for provided outputs.

## Historical Evidence Note
- Archived February 2026 runs remain historical evidence and are not to be rewritten retroactively.
- They should be marked as pre-tightening evidence whenever reproducibility cleanliness is discussed.
