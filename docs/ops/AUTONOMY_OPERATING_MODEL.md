# Gerhard Autonomous Operating Model

## Purpose
Define how Gerhard runs autonomously with minimal user interaction while preserving the core scientific dependency chain:

`v15 -> v16 -> v17 -> v18 -> v19`

Add-on tracks (H/I/J/K) are executed as parallel scaffolding and gating support.

## Operating Principles
1. Chain lock is non-negotiable: no reordering of v15->v19.
2. Proof-first: every claim must reference deterministic artifacts.
3. Cost hygiene: CPU-first checks; short GPU smoke; bounded runs (`--max_steps` or equivalent).
4. Capability gate discipline: baseline, ablation, evaluation gate, rollback flag.

## Autonomy Loop (Per Run)
1. Run notebook pass on target environment (RunPod/Colab/Kaggle).
2. Ingest run artifacts from notebook output folder into `outputs/<run_id>/`.
2. Evaluate gates for current phase.
3. Write run report (Markdown + JSON).
4. Update state files:
   - `state/program_status.yaml`
   - `state/gate_results.yaml`
   - `state/autopilot_queue.yaml`
5. Update `docs/ops/STATUS_BOARD.md` and `reports/index.md`.
7. Decide:
   - all required gates green -> continue autonomously
   - any required gate red -> pause and emit `needs_input.md`

Ingestion is performed via:
`python3 scripts/register_notebook_run.py --run-id <run_id> --phase <A..K> --source-dir <notebook_output_dir>`

RunPod notebook handoff details:
`docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`

## Required Sources Of Truth
1. Strategy and scientific intent:
   - `docs/GERHARD_MASTER_PLAN_WITH_ADDON_2026-02-23.md`
   - `knowledge/roadmap.md`
2. Day-to-day autonomous control:
   - `docs/ops/GATE_POLICY.md`
   - `docs/ops/REPORTING_CONTRACT.md`
   - `state/*.yaml`
   - `reports/index.md`

## User Interaction Contract
User only needs to:
1. Read latest report entry in `reports/index.md`.
2. Provide explicitly requested outputs/decisions when `AUTOPILOT_DECISION: PAUSE_NEEDS_INPUT`.

No approval is needed for green-gate continuation.

## Current Bootstrap State (2026-02-23)
1. Current phase: v15 validation in progress.
2. Add-on scaffolds (`eval/`, `data/`, `verifiers/`, `retrieval/`) exist as stubs.
3. Autonomous confidence blocker: execution environment verification capability must be present for full trust.
