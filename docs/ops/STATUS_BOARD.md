# Gerhard Status Board

**As Of**: 2026-02-23  
**Autopilot Mode**: Conditional (continue on green, pause on red)

## Executive Snapshot
- Core chain lock: `v15 -> v16 -> v17 -> v18 -> v19` (unchanged)
- Execution mode: notebook-first; repo automation ingests notebook outputs post-run
- Current active phase: **B_v15_spikingbrain_validation**
- Latest known best: **v14.3 PPL 306.89**
- Add-on scaffolds:
  - `eval/` scaffold: present
  - `data/mixture.yaml`: present
  - `verifiers/` stubs: present
  - `retrieval/` stub: present
- Key blocker: Phase B scientific thresholds failed on latest run (MI, CKA).

<!-- AUTOGEN_LATEST_RUN_START -->
## Latest Run Update
- Run ID: `v15_2026-02-23_200258`
- Timestamp UTC: `2026-02-23T20:21:53.512775Z`
- Phase: `B_v15_spikingbrain_validation`
- Decision: `PAUSE_NEEDS_INPUT`
- Red gates: `1`
- Yellow gates: `0`
- Next action: Address red gates and provide the requested outputs in the needs_input report.
<!-- AUTOGEN_LATEST_RUN_END -->

## Live Engineering Update (2026-02-23)
- Notebook patched: `notebooks/asnn_goose_colab_v15.ipynb`
- Runtime hardening applied:
  - dependency bootstrap with optional plotting fallback,
  - `return_spike_info` support added in student/layer forward path,
  - v15 validator fixed for tuple/dict batch formats,
  - `val_dataloader` reference fixed to `val_loader`,
  - guards added for empty loaders and zero-token evaluation.
- Full-rerun remediation patch applied:
  - disabled `torch.compile` for stability with rich spike instrumentation,
  - added mixed token/channel thresholding in ternary spikes to reduce dead channels,
  - added spike semantic alignment loss (teacher-to-spike ternary target alignment),
  - expanded training logs with semantic-loss curves,
  - fixed validator bias by using both `k` and `v` spikes and robust MI discretization.
- Latest execution: patched v15 run ingested from RunPod (RTX 6000 Ada) as `v15_2026-02-23_200258`.
- Canonical bundle archived: `outputs/v15_2026-02-23_200258/{eval_suite.json,metrics.json,config.yaml,seed.txt,v15_spikingbrain.json,results.json,run_dossier_v15_2026-02-23_200258.html,figures_detailed/*}`.
- Single-file dossier export is enabled in notebook final cell: `outputs/<run_id>/run_dossier_<run_id>.html` (auto-download attempted by notebook runtime).

## Phase Status (A-K)
| Phase | Name | Status | Owner State |
|------|------|--------|-------------|
| A | Engineering Guardrails | IN PROGRESS | Scaffolds in place; full reproducibility guardrails pending. |
| B | v15 SpikingBrain Validation | BLOCKED | Latest full rerun still below MI/CKA thresholds; another notebook patch + rerun required. |
| C | Temporal Coding Proof | NOT STARTED | Waiting for B gate maturity. |
| D | v16 Sparse Ops | NOT STARTED | Blocked by B pass. |
| E | v17 Efficiency | NOT STARTED | Blocked by D readiness. |
| F | v18 Ablations | NOT STARTED | Blocked by D/E readiness. |
| G | v19 Publication/Repro | NOT STARTED | Blocked by F readiness. |
| H | Generalist Scorecard + Mixture | IN PROGRESS | Config scaffolds exist; baselines and gates pending. |
| I | Retrieval World-Fit | IN PROGRESS | Interface stub exists; runtime + eval gate pending. |
| J | Post-Training | NOT STARTED | Depends on gate infrastructure and data readiness. |
| K | Scaling Ladder | NOT STARTED | Depends on B/H/I/J gate maturity. |

## Current Red/Yellow Signals
1. RED: `phase_b_scientific_thresholds` failed on latest run (`mutual_information`, `cka_mean`).
2. YELLOW: scorecard baselines are not yet established for H regression gates.

## Next Autonomous Actions
1. Run the full patched `notebooks/asnn_goose_colab_v15.ipynb` on RunPod.
2. Ingest the resulting single-file dossier through `scripts/register_dossier_run.py` (preferred), or ingest the full bundle through `scripts/register_notebook_run.py`.
3. Re-evaluate `phase_b_scientific_thresholds`; keep B blocked unless gate turns green.
4. Only then advance to H baseline capture and downstream phases.

## Key Links
- Operating model: `docs/ops/AUTONOMY_OPERATING_MODEL.md`
- Gate policy: `docs/ops/GATE_POLICY.md`
- Reporting contract: `docs/ops/REPORTING_CONTRACT.md`
- RunPod notebook handoff: `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md`
- Program state: `state/program_status.yaml`
- Gate state: `state/gate_results.yaml`
- Reports index: `reports/index.md`
