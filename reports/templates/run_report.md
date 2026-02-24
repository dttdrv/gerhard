# Gerhard Run Report

## Tier 1: Executive Summary
- Run ID: `<run_id>`
- Timestamp UTC: `<timestamp_utc>`
- Phase: `<phase>`
- Commit: `<commit>`
- What changed:
  - `<change_1>`
  - `<change_2>`
- Gate outcome: `<overall_gate_outcome>`
- Risks:
  - `<risk_1>`
- Next autonomous action: `<next_action>`
- User input required: `<yes_or_no>`

## Tier 2: Technical Evidence

### Metric Deltas vs Baseline
| Metric | Baseline | Observed | Delta | Status |
|-------|----------|----------|-------|--------|
| `<metric>` | `<baseline>` | `<observed>` | `<delta>` | `<green/yellow/red>` |

### Gate Scorecard
| Gate | Observed | Threshold | Status | Notes |
|------|----------|-----------|--------|-------|
| `<gate_name>` | `<observed>` | `<threshold>` | `<status>` | `<notes>` |

### Artifacts
- `outputs/<run_id>/eval_suite.json`
- `outputs/<run_id>/metrics.json`
- `outputs/<run_id>/config.yaml`
- `outputs/<run_id>/seed.txt`
- `outputs/<run_id>/<phase_specific_report_file>`

### Reproducibility Fingerprint
- Command/profile: `<command_or_job_profile>`
- Seed source: `<seed_source>`
- Environment: `<hardware_and_runtime>`
- Package snapshot: `<package_manifest_path_or_hash>`

## Decision
- Decision: `<CONTINUE_or_PAUSE_NEEDS_INPUT>`
- Rationale:
  - `<decision_reason_1>`
  - `<decision_reason_2>`

AUTOPILOT_DECISION: <CONTINUE_or_PAUSE_NEEDS_INPUT>
