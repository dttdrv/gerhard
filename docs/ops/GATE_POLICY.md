# Gerhard Gate Policy

## Purpose
Define deterministic gate rules that decide whether autopilot continues or pauses.

## Status Levels
1. `green`: gate passed.
2. `yellow`: minor regression/noise within tolerance; continue allowed with mitigation.
3. `red`: hard failure or regression outside tolerance; autopilot must pause.

## Hard Red Conditions (Immediate Pause)
1. Required tests fail for current phase.
2. Required artifacts are missing from `outputs/<run_id>/`.
3. Any required metric contains NaN or Inf.
4. Reproducibility metadata missing (config, seed, commit/package fingerprint).
5. Explicit phase pass criteria fail (for example v15 thresholds not met).

## Metric Regression Rules (Balanced Mode)
Compare against latest accepted baseline.

### Red thresholds
1. PPL regression > +1.5%.
2. Instruction/code/reasoning pass-rate drop > 2.0 absolute points.
3. Citation correctness drop > 2.0 points.
4. Hallucination increase > 2.0 points.
5. Latency overhead above declared budget cap.

### Yellow thresholds
1. Regression exists but stays within red threshold.
2. Mitigation note is mandatory in next run plan.

### Green thresholds
1. All required metrics meet or exceed gate targets, or
2. No regression beyond accepted noise.

## Phase Gate Dependencies
1. B must pass before D can start.
2. D must complete correctness gate before E benchmarking is authoritative.
3. E benchmark integrity required before F ablation conclusions.
4. F complete before G publication freeze.
5. H scorecard gate must pass before K scaling decisions.
6. I retrieval gate must pass before retrieval is enabled in default path.
7. J RLVR/post-training changes must pass H regression gate each run.

## Decision Output Requirement
Every run report must end with exactly one of:
1. `AUTOPILOT_DECISION: CONTINUE`
2. `AUTOPILOT_DECISION: PAUSE_NEEDS_INPUT`
