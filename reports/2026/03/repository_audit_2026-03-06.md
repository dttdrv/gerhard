# Gerhard Repository Audit Report

Generated: 2026-03-06  
Workspace audited: `C:\Users\deyan\Projects\gerhard`  
Audit scope: 225 workspace files, excluding `.git/` internals

## Executive Summary

This repository is a hybrid of three things:

1. A real research codebase for compressing GPT-2 into an RWKV-style ternary spiking model (`ASNN-Goose`).
2. A notebook-first experiment archive that records the evolution from early Kaggle prototypes through the `v14.3` breakthrough.
3. An operations/reporting layer added later to turn notebook outputs into reproducible run bundles, reports, and gate decisions.

The historical model work clearly achieved meaningful progress through `v14.3`, which is consistently recorded as the best known result at `PPL 306.89` for a `74M` parameter, `5`-layer student. The newer `v15` phase was not a new performance win; it was a scientific validation gate intended to prove that spike patterns carry semantic information rather than acting as arbitrary quantization artifacts.

That `v15` gate is the current blocker. Two real archived runs from 2026-02-23 both failed the Phase B scientific thresholds. The second run materially improved spike health, but it still failed the mutual information and CKA thresholds, and it slightly regressed language-model perplexity compared with the better run.

At the repository level, the reporting pipeline worked mechanically: runs were ingested, reports were generated, state files were updated, and the status board was rewritten. However, several governance promises are not actually enforced in code or artifacts yet. Examples include missing commit fingerprints, missing documented hash fields, placeholder evaluation scaffolds, stubbed retrieval/verifier modules, and a current test drift inside the run-registration flow.

The short management conclusion is:

- The research program worked up to `v14.3`.
- The operational reporting layer mostly worked.
- The current `v15` scientific validation objective did not work yet.
- The repo contains many solid components, but only a subset are well evidenced by tests.
- The repo is blocked by science/integration quality, not by missing folders or missing reports.

## Audit Method

This report is based on:

- Full workspace inventory and file classification.
- Git history and current git status.
- Repository documentation, changelog, state files, reports, run outputs, prompts, notebooks, and scripts.
- Fresh verification runs performed during the audit:
  - `python -m pytest -q` -> `38 passed, 2 failed`
  - `python -m pytest tests/test_lora.py tests/test_recurrence.py tests/test_ternary.py -q` -> `27 passed`
- A coverage run performed during the audit by a subagent, reporting low overall code coverage (`23%`) with strongest coverage in `src/models/goose_backbone.py`, `src/models/ternary_activations.py`, and `src/models/lora_adapter.py`.
- Subagent reviews covering:
  - source + tests
  - scripts/eval/retrieval/verifiers
  - docs/state/reports/outputs
  - notebooks/knowledge/prompts/tmp
- Context7 documentation checks for `pytest` cache semantics and `coverage.py` data files to classify `.pytest_cache/` and `.coverage`.

## Top-Level Findings

### What Worked

- The research iteration from `v10` through `v14.3` produced genuine quality gains and has a coherent record in `changelog.md`, `README.md`, `knowledge/roadmap.md`, and the notebook archive.
- The core dense recurrence implementation appears to be the most mature subsystem. The recurrence tests passed cleanly.
- The ternary activation primitives and basic LoRA components also appear materially functional. Their direct tests passed cleanly.
- The notebook-ingestion/reporting flow worked in practice for two Phase B runs:
  - artifacts were copied into `outputs/<run_id>/`
  - reports were written to `reports/2026/02/`
  - state files were updated under `state/`
  - `docs/ops/STATUS_BOARD.md` was updated
- The patched `v15` rerun improved spike-health metrics substantially:
  - dead neurons improved from `11.33%` to `0.65%`
  - firing rate stayed within the target range
  - `health_pass` flipped from false to true
- The repository contains a complete historical trail rather than only final code. That is useful for management, reproducibility, and forensic review.

### What Did Not Work

- `v15` scientific validation did not pass in any archived run.
- The second `v15` run improved spike-health metrics but still failed:
  - `mutual_information = 0.0435` vs required `> 0.10`
  - `cka_mean = 0.0196` vs required `> 0.30`
- The better scientific-health run also worsened student perplexity relative to the earlier archived run:
  - run `v15_2026-02-23_155547`: `student_ppl = 306.87`
  - run `v15_2026-02-23_200258`: `student_ppl = 314.06`
- The generic `eval/` harness is still scaffold-only. It writes placeholder results rather than real evaluation outcomes.
- Retrieval and verifier subsystems are explicitly stubbed.
- Two tests in `tests/test_register_notebook_run.py` now fail because the code requires richer `v15_spikingbrain.json` structure than the tests provide.
- Coverage is low overall. Much of the repo is implemented but lightly validated or unvalidated.
- Some evaluation/integration paths are concretely broken according to subagent verification:
  - copy-task benchmark shape mismatch in `src/evaluation/benchmarks.py`
  - `SpikingBrain` evaluator assumes a Hugging Face-style teacher interface that the repo’s own `TeacherModel` does not provide
  - `ASNNGoose.from_teacher()` expects a raw backbone-like object more than the `TeacherModel` wrapper it claims to accept

### Current Status

- Branch: `main`
- Latest commit: `52da670` (`Create v15 rerun notebook`, dated 2026-03-05)
- Current git status during audit: only untracked `.coverage`
- Program state: `Phase B blocked`
- Latest authoritative run: `v15_2026-02-23_200258`
- Latest authoritative decision: `PAUSE_NEEDS_INPUT`
- Queue state: still waiting for another meaningful `v15` rerun after model/config changes

## Timeline Reconstruction

### Research / Model Timeline

- `2025-12-28`: baseline and early experiments (`v10`, `v11`, `v11.1`, `v12`) were explored.
- `2025-12-30`: `v12.1` fixed CTKD by using GRL/adversarial temperature learning instead of a naive learnable temperature.
- `2025-12-31`: `v13` POCL/curriculum work failed badly; `v13.1` recovered by disabling POCL and extending training.
- `2026-01-04`: `v14`, `v14.1`, `v14.2`, `v14.3` produced the strongest improvement sequence, with `v14.3` reaching `PPL 306.89`.
- `2026-01-05`: `v15` was defined as a scientific validation gate rather than a direct performance release.

### Repo / Ops Timeline

- `2026-01-06`: large initial import of code, notebooks, docs, knowledge base, tests, and changelog.
- `2026-01-23`: `asnn_goose_colab_v14.ipynb` was renamed forward into `asnn_goose_colab_v15.ipynb`; this suggests `v15` began as a continuation notebook rather than a fresh clean-room notebook.
- `2026-02-20`: `README.md` was rewritten into a more concise public-facing summary.
- `2026-02-23` / commit `3ac7a71`: the repo gained the autonomy/reporting layer, state files, run templates, dossier tooling, retrieval/verifier scaffolds, and add-on plan docs.
- `2026-03-05` / commits `169a1a2` and `52da670`: the monolithic `v15` notebook was patched for deterministic fingerprints and a new reset/rerun notebook was created, along with extraction/build helpers under `tmp/`.

## Evidence From Fresh Verification

### Test Results

- `python -m pytest -q` returned `38 passed, 2 failed`.
- The two failing tests were:
  - `tests/test_register_notebook_run.py::test_evaluate_gates_continue_when_artifacts_present_and_finite`
  - `tests/test_register_notebook_run.py::test_register_run_callable_creates_reports_and_state`
- The failure reason is straightforward: the tests still assume a minimal `{"overall_pass": true}` `v15_spikingbrain.json`, while the code now insists on nested scientific metrics under `validation.health`, `validation.mutual_information`, and `validation.cka`.
- `.pytest_cache/v/cache/lastfailed` records exactly those same two failures.

### Targeted Model-Core Test Results

- `python -m pytest tests/test_lora.py tests/test_recurrence.py tests/test_ternary.py -q` returned `27 passed`.
- This supports the conclusion that the most credible working code today is:
  - dense recurrence
  - ternary activation primitives
  - basic LoRA support

### Coverage

The audit subagent’s coverage run reported low overall coverage (`23%`). Confidence is concentrated in:

- `src/models/goose_backbone.py`
- `src/models/ternary_activations.py`
- `src/models/lora_adapter.py`
- parts of `scripts/register_notebook_run.py`

This does not mean the rest of the repo is non-functional. It means the rest is not strongly evidenced by automated tests.

### Security / Dependency Scan

- `trivy` is installed (`v0.68.2`), but filesystem scans failed twice with internal tool errors during the audit.
- No clean Trivy result is available from this session.
- That means there is no current vulnerability/misconfiguration scan evidence for management to rely on.

## Governance and Consistency Gaps

These are not the main scientific blocker, but they matter for project management:

- `GATE_POLICY.md` says missing reproducibility metadata is a hard red gate, yet the archived JSON reports still store `"commit": "unknown"`.
- `STATUS_BOARD.md` claims fingerprint hashes were added to artifacts, but the archived `200258` run artifacts do not expose `config_sha256`, `recipe_sha256`, or `notebook_sha256` in the checked files inspected during this audit.
- `STATUS_BOARD.md` also says the canonical `200258` bundle includes `figures_detailed/*`, but the actual canonical folder does not.
- The dossier ecosystem is not fully standardized:
  - one dossier/generator path uses `phase_artifact`
  - another dossier-ingestion path expects `v15_validation`
- `config.py` still represents a smaller prototype configuration and no longer matches the actual later-stage `v14.3`/`v15` runs.
- The repo does not use the root-level `PROJECT_BRIEF.md`, `STATE.yaml`, and `LOG.md` pattern requested by the planning skill; it instead uses `state/*.yaml`, `changelog.md`, `reports/`, and docs under `docs/ops/`.

## File-by-File Inventory

Legend used below:

- `canonical`: primary source of truth for current repo behavior
- `implemented`: working or partly working code with some evidence
- `scaffold`: planned interface/structure, not yet fully wired
- `archived`: historical run or experiment artifact
- `transient`: generated cache, compiled output, or helper artifact
- `drifted`: still relevant, but not aligned with current repo reality
- `superseded`: historically useful, but replaced by newer material

## Root Files

- `.coverage` — `transient`. Coverage database created during local verification on 2026-03-05/06; not tracked by git and not part of historical project evidence.
- `.gitignore` — `canonical`. Ignores Python caches, checkpoints, notebook junk, and `outputs/`; current committed outputs look like intentional exceptions rather than default policy.
- `README.md` — `canonical`. Concise public summary of project purpose, current numbers, and roadmap.
- `changelog.md` — `canonical`. Best single historical narrative of version-to-version successes and failures from `v6` through `v15`.
- `config.py` — `drifted`. Central config container for an earlier/smaller prototype; no longer matches the later real `v14.3`/`v15` run shapes.
- `requirements.txt` — `canonical`. Minimal Python dependency list centered on PyTorch, Hugging Face tooling, plotting, Jupyter, and pytest.
- `skip4000.bat` — `transient`. Tiny helper batch file for inspecting notebook content after skipping the first 4000 lines.
- `temp_nb_inspect.py` — `transient`. Small notebook-inspection utility; indicates ad hoc notebook surgery/debugging work.

## Pytest Cache

All files in this section are `transient` local pytest cache artifacts, not source-of-truth project assets.

- `.pytest_cache/.gitignore`
- `.pytest_cache/CACHEDIR.TAG`
- `.pytest_cache/README.md`
- `.pytest_cache/v/cache/lastfailed`
- `.pytest_cache/v/cache/nodeids`

Specific notes:

- `.pytest_cache/README.md` explicitly says the directory is created by pytest’s cache plugin and should not be committed.
- `.pytest_cache/v/cache/lastfailed` is useful evidence because it records the same two failing registration tests seen in the fresh audit run.
- `.pytest_cache/v/cache/nodeids` shows the repo currently exposes 40 collected tests, mostly around recurrence, ternary activations, LoRA, scaffolds, and run registration.

## Data

- `data/mixture.yaml` — `scaffold`. Future-oriented mixture/generalist track placeholder, not a live data pipeline.

## Documentation and Ops Docs

- `docs/GERHARD_MASTER_PLAN_WITH_ADDON_2026-02-23.md` — `canonical`. The most complete management plan integrating the core `v15 -> v19` chain with add-on tracks `H/I/J/K`.
- `docs/ops/AUTONOMY_OPERATING_MODEL.md` — `canonical`. Defines the notebook-first autonomous operating model and phase lock sequencing.
- `docs/ops/GATE_POLICY.md` — `canonical`. Defines green/yellow/red gate semantics and phase dependencies.
- `docs/ops/REPORTING_CONTRACT.md` — `canonical`. Defines required artifacts, required reports, required state updates, and the pause contract.
- `docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md` — `canonical`. Documents the intended notebook-to-repo handoff, including dossier-first ingestion.
- `docs/ops/STATUS_BOARD.md` — `canonical but drifted`. Main human-readable status board; largely correct on blocked Phase B status, but some claims about fingerprints and archived figure content do not match checked artifacts.

## Eval Package

- `eval/__init__.py` — `scaffold`. Package export for the Phase A unified evaluation harness.
- `eval/run_suite.py` — `scaffold`. Real code that writes a single JSON artifact, but default evaluators are explicit placeholders and not connected to the real notebook runs.
- `eval/suite.yaml` — `scaffold`. Declarative evaluation task list and gates; thresholds are unset and retrieval is disabled.

The files below are `transient` compiled bytecode generated from the corresponding source files:

- `eval/__pycache__/run_suite.cpython-314.pyc`
- `eval/__pycache__/__init__.cpython-314.pyc`

## Knowledge Base

- `knowledge/ctkd_implementation_plan.md` — `canonical research context`. Detailed plan explaining the `v12 -> v12.1` CTKD/GRL correction.
- `knowledge/distillation.md` — `canonical research context`. Internal summary of distillation techniques used as design background.
- `knowledge/external_llm_context_v14_results.md` — `superseded but useful`. External-analysis packet generated around the earlier `v14` result; historically useful, no longer the current best state.
- `knowledge/overview.md` — `canonical research context`. High-level project thesis and architectural framing.
- `knowledge/papers.md` — `canonical research context`. Long internal literature summary supporting project decisions.
- `knowledge/roadmap.md` — `canonical`. Best concise source for actual dependency chain and current blocked state.
- `knowledge/rwkv_architecture.md` — `canonical research context`. Background on the recurrence/backbone side.
- `knowledge/spiking_networks.md` — `canonical research context`. Background on spike-based modeling choices.
- `knowledge/training_strategies.md` — `canonical research context`. Training tactics reference.

These seven files are `canonical research inputs` generated from prompt-driven research passes and feed the higher-level knowledge docs:

- `knowledge/research/prompt1.txt`
- `knowledge/research/prompt2.txt`
- `knowledge/research/prompt3.txt`
- `knowledge/research/prompt4.txt`
- `knowledge/research/prompt5.txt`
- `knowledge/research/prompt6.txt`
- `knowledge/research/prompt7.txt`

## Notebooks

- `notebooks/asnn_goose_kaggle_v3.ipynb` — `archived`. Earliest visible Kaggle-era prototype.
- `notebooks/asnn_goose_kaggle_v4.ipynb` — `archived`. Follow-on Kaggle architecture iteration.
- `notebooks/asnn_goose_kaggle_v5.ipynb` — `archived`. Kaggle version that looks like the first full training pipeline.
- `notebooks/asnn_goose_colab_v6.ipynb` — `archived`. First working distillation notebook in the changelog lineage.
- `notebooks/asnn_goose_colab_v8.ipynb` — `archived`. Recovery notebook after the `v7` regression.
- `notebooks/asnn_goose_colab_v9.ipynb` — `archived`. Capacity-increase notebook.
- `notebooks/asnn_goose_colab_v10.ipynb` — `archived`. Baseline preceding the more experimental `v11+` phase.
- `notebooks/asnn_goose_colab_v11_allfour_backup.ipynb` — `archived`. “All four advanced techniques” backup notebook; strong evidence of an aggressive but unstable experimentation phase.
- `notebooks/asnn_goose_colab_v12.1.ipynb` — `archived but important`. Notebook associated with the successful CTKD/GRL correction.
- `notebooks/asnn_goose_colab_v13.ipynb` — `archived`. POCL/curriculum-era notebook; important mainly because the branch failed.
- `notebooks/asnn_goose_colab_v15.ipynb` — `drifted canonical notebook`. Large inherited workbench that mixes old and new logic; still the main historical execution notebook, but it looks difficult to trust as a clean rerun vehicle.
- `notebooks/asnn_goose_v15_reset_master.ipynb` — `canonical current notebook`. Newer, smaller, deterministic reset/rerun notebook created to replace the accumulated `v15` monolith.

The file below is `transient` compiled bytecode from notebook extraction/import work:

- `notebooks/__pycache__/asnn_goose_colab_v15_extracted.cpython-314.pyc`

## Outputs Root

- `outputs/eval_suite.json` — `scaffold artifact`. Placeholder unified-eval output with `not_run` / `skipped` statuses; not a real benchmark report.

## Incoming Run Bundles

These folders are `archived source ingest bundles`, meaning they are pre-canonical copies of notebook or dossier material before registration logic stored them under canonical `outputs/<run_id>/`.

### `outputs/incoming/v15_2026-02-23_155547_source/`

- `outputs/incoming/v15_2026-02-23_155547_source/config.yaml`
- `outputs/incoming/v15_2026-02-23_155547_source/eval_suite.json`
- `outputs/incoming/v15_2026-02-23_155547_source/metrics.json`
- `outputs/incoming/v15_2026-02-23_155547_source/results.json`
- `outputs/incoming/v15_2026-02-23_155547_source/seed.txt`
- `outputs/incoming/v15_2026-02-23_155547_source/v15_spikingbrain.json`

### `outputs/incoming/v15_2026-02-23_200258_from_dossier/`

- `outputs/incoming/v15_2026-02-23_200258_from_dossier/config.yaml`
- `outputs/incoming/v15_2026-02-23_200258_from_dossier/eval_suite.json`
- `outputs/incoming/v15_2026-02-23_200258_from_dossier/metrics.json`
- `outputs/incoming/v15_2026-02-23_200258_from_dossier/results.json`
- `outputs/incoming/v15_2026-02-23_200258_from_dossier/run_dossier_v15_2026-02-23_200258.html`
- `outputs/incoming/v15_2026-02-23_200258_from_dossier/seed.txt`
- `outputs/incoming/v15_2026-02-23_200258_from_dossier/v15_spikingbrain.json`

## Canonical Archived Run `v15_2026-02-23_155547`

This run is `archived` and scientifically failed. It is still valuable because it preserves the pre-patch baseline for the Phase B validation attempt.

- `outputs/v15_2026-02-23_155547/config.yaml` — run config and metadata for the first archived Phase B run.
- `outputs/v15_2026-02-23_155547/eval_suite.json` — compact gate-facing summary for the run; marks recommendation as red.
- `outputs/v15_2026-02-23_155547/metrics.json` — executive metrics; best student PPL here is `306.87`.
- `outputs/v15_2026-02-23_155547/results.json` — largest structured run artifact; includes curves, hardware stats, spike analysis, comparisons, and autonomy artifacts.
- `outputs/v15_2026-02-23_155547/run_dossier_v15_2026-02-23_155547.html` — self-contained HTML dossier generated from the run bundle.
- `outputs/v15_2026-02-23_155547/seed.txt` — seed record (`42`).
- `outputs/v15_2026-02-23_155547/v15_spikingbrain.json` — decisive scientific validation artifact; failed dead-neuron, MI, and CKA thresholds.

These twelve files are `archived generated figures` embedded in or associated with the first dossier:

- `outputs/v15_2026-02-23_155547/figures_detailed/01_loss_components.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/02_loss_components_log.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/03_total_loss_smoothed.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/04_ppl_curve.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/05_learning_rate.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/06_temperature.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/08_spike_density_per_layer.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/09_spike_amplitude_per_layer.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/10_overall_spike_density_timeline.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/11_ttt_loss.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/12_validation_tests.png`
- `outputs/v15_2026-02-23_155547/figures_detailed/13_hardware_summary.png`

## Canonical Archived Run `v15_2026-02-23_200258`

This run is `archived` and is the current source-of-truth latest run. It improved spike health but still failed MI and CKA, so it remains blocked.

- `outputs/v15_2026-02-23_200258/config.yaml` — reconstructed config from dossier ingestion; less detailed than the first run’s config and marked as dossier-sourced.
- `outputs/v15_2026-02-23_200258/eval_suite.json` — compact evaluation summary; red recommendation because scientific thresholds still failed.
- `outputs/v15_2026-02-23_200258/metrics.json` — executive metrics; worse PPL than the earlier archived run but better spike health.
- `outputs/v15_2026-02-23_200258/results.json` — large reconstructed result bundle containing training curves, hardware stats, spike analysis, and comparison metadata.
- `outputs/v15_2026-02-23_200258/run_dossier_v15_2026-02-23_200258.html` — self-contained HTML dossier used for single-file ingestion.
- `outputs/v15_2026-02-23_200258/seed.txt` — seed record (`42`).
- `outputs/v15_2026-02-23_200258/v15_spikingbrain.json` — decisive scientific validation artifact; health passed, MI and CKA still failed.

## Prompt Assets

- `prompts/codex_code_review_v11.md` — `superseded prompt asset`. Review prompt aimed at a notebook name no longer present; historically useful but no longer aligned with current files.
- `prompts/deep_research_prompts.md` — `canonical research prompt asset`. Prompt deck that appears to have driven the research knowledge base.

## Reports

- `reports/README.md` — `canonical`. States source-of-truth order and report contract.
- `reports/index.md` — `canonical`. Latest-first index of recorded runs; currently points to a blocked `v15_2026-02-23_200258`.

These six files are `archived operational reports` created by the run-ingestion system:

- `reports/2026/02/v15_2026-02-23_155547.json`
- `reports/2026/02/v15_2026-02-23_155547.md`
- `reports/2026/02/v15_2026-02-23_155547_needs_input.md`
- `reports/2026/02/v15_2026-02-23_200258.json`
- `reports/2026/02/v15_2026-02-23_200258.md`
- `reports/2026/02/v15_2026-02-23_200258_needs_input.md`

Specific notes:

- The two main `.md` and `.json` files correctly record blocked Phase B outcomes.
- The two `_needs_input.md` files are much thinner than the fuller template expects; they work operationally, but they are not rich management documents.

These three files are `canonical templates` for future reporting:

- `reports/templates/needs_input.md`
- `reports/templates/run_report.json`
- `reports/templates/run_report.md`

## Retrieval Package

- `retrieval/base.py` — `scaffold`. Defines retrieval mode enum and `Passage` dataclass; good interface seed.
- `retrieval/runtime.py` — `scaffold`. `retrieve()` returns `[]`; diagnostics explicitly report `implemented: false`.
- `retrieval/__init__.py` — `scaffold`. Package export layer.

The files below are `transient` compiled bytecode:

- `retrieval/__pycache__/base.cpython-314.pyc`
- `retrieval/__pycache__/runtime.cpython-314.pyc`
- `retrieval/__pycache__/__init__.cpython-314.pyc`

## Scripts

- `scripts/generate_run_dossier.py` — `implemented`. Generates self-contained HTML dossiers with embedded figures and raw payloads.
- `scripts/register_dossier_run.py` — `implemented but drifted`. Ingests a single dossier into canonical artifacts; schema expectations are not perfectly aligned with all dossier variants.
- `scripts/register_notebook_run.py` — `implemented`. Core operational script that copies artifacts, evaluates gates, writes reports, updates state, and updates the status board.
- `scripts/__init__.py` — `canonical small utility export`.

The files below are `transient` compiled bytecode:

- `scripts/__pycache__/generate_run_dossier.cpython-314.pyc`
- `scripts/__pycache__/register_dossier_run.cpython-314.pyc`
- `scripts/__pycache__/register_notebook_run.cpython-314.pyc`
- `scripts/__pycache__/__init__.cpython-314.pyc`

## Source Package Root

- `src/__init__.py` — `canonical package root`.

The file below is `transient` compiled bytecode:

- `src/__pycache__/__init__.cpython-314.pyc`

## Source: Evaluation

- `src/evaluation/benchmarks.py` — `implemented but broken in places`. Contains perplexity/copy/retrieval benchmark logic, but the copy-task path is currently shape-buggy according to subagent verification.
- `src/evaluation/spike_analysis.py` — `implemented`. Large analysis utility for spike metrics; little automated evidence.
- `src/evaluation/spiking_brain.py` — `implemented but integration-broken`. Main `v15` validation module; conceptually central, but current teacher-interface assumptions do not align cleanly with the repo’s own teacher wrapper.
- `src/evaluation/stability_tests.py` — `implemented but lightly validated`. Stability/TTT analysis logic with at least one suspicious metric-recording issue.
- `src/evaluation/__init__.py` — `canonical package export`.

The files below are `transient` compiled bytecode:

- `src/evaluation/__pycache__/benchmarks.cpython-314.pyc`
- `src/evaluation/__pycache__/spike_analysis.cpython-314.pyc`
- `src/evaluation/__pycache__/spiking_brain.cpython-314.pyc`
- `src/evaluation/__pycache__/stability_tests.cpython-314.pyc`
- `src/evaluation/__pycache__/__init__.cpython-314.pyc`

## Source: Kernels

- `src/kernels/sparsity_analysis.py` — `implemented but lightly validated`. Research utility for sparsity/kernel analysis.
- `src/kernels/__init__.py` — `canonical package export`.

The files below are `transient` compiled bytecode:

- `src/kernels/__pycache__/sparsity_analysis.cpython-314.pyc`
- `src/kernels/__pycache__/__init__.cpython-314.pyc`

## Source: Models

- `src/models/asnn_goose.py` — `implemented but lightly validated`. Full spiking student model with quantized projections and optional LoRA integration.
- `src/models/goose_backbone.py` — `implemented and best-evidenced`. Most mature core module; recurrence/state/generation behavior is directly exercised by passing tests.
- `src/models/lora_adapter.py` — `implemented and partly evidenced`. Basic LoRA adapter system with passing tests for core behavior.
- `src/models/quantized_weights.py` — `implemented but unproven`. Full INT8 quantization surface with little direct test evidence.
- `src/models/teacher_model.py` — `implemented but lightly validated`. Distillation teacher wrapper; does not integrate cleanly with all evaluation code.
- `src/models/ternary_activations.py` — `implemented and partly evidenced`. Ternary spike primitives with passing direct tests.
- `src/models/__init__.py` — `canonical package export`.

The files below are `transient` compiled bytecode:

- `src/models/__pycache__/asnn_goose.cpython-314.pyc`
- `src/models/__pycache__/goose_backbone.cpython-314.pyc`
- `src/models/__pycache__/lora_adapter.cpython-314.pyc`
- `src/models/__pycache__/quantized_weights.cpython-314.pyc`
- `src/models/__pycache__/teacher_model.cpython-314.pyc`
- `src/models/__pycache__/ternary_activations.cpython-314.pyc`
- `src/models/__pycache__/__init__.cpython-314.pyc`

## Source: Training

- `src/training/distillation.py` — `implemented but lightly validated`. Distillation/training logic with little automated evidence.
- `src/training/trainers.py` — `implemented but lightly validated`. Training-loop utilities and orchestration.
- `src/training/ttt_controller.py` — `implemented but lightly validated`. Test-time-training controller.
- `src/training/__init__.py` — `canonical package export`.

The files below are `transient` compiled bytecode:

- `src/training/__pycache__/distillation.cpython-314.pyc`
- `src/training/__pycache__/trainers.cpython-314.pyc`
- `src/training/__pycache__/ttt_controller.cpython-314.pyc`
- `src/training/__pycache__/__init__.cpython-314.pyc`

## Source: Utils

- `src/utils/checkpoint.py` — `implemented but lightly validated`. Checkpoint/save-load utilities.
- `src/utils/logging_utils.py` — `implemented but lightly validated`. Logging helpers.
- `src/utils/visualization.py` — `implemented but lightly validated`. Plot/report support utilities used by notebooks/dossiers.
- `src/utils/__init__.py` — `canonical package export`.

The files below are `transient` compiled bytecode:

- `src/utils/__pycache__/checkpoint.cpython-314.pyc`
- `src/utils/__pycache__/logging_utils.cpython-314.pyc`
- `src/utils/__pycache__/visualization.cpython-314.pyc`
- `src/utils/__pycache__/__init__.cpython-314.pyc`

## State Files

- `state/autopilot_queue.yaml` — `canonical`. Ordered autonomous task queue; still waiting on a meaningful `v15` rerun before later phases.
- `state/gate_results.yaml` — `canonical`. Most compact machine-readable statement of why the latest run is blocked.
- `state/program_status.yaml` — `canonical`. Main machine-readable program state; explicitly records Phase B as blocked and points to the latest report.

## Tests

- `tests/test_eval_harness_addon.py` — `implemented scaffold test`. Verifies the unified eval harness writes one JSON artifact and handles retrieval gating.
- `tests/test_lora.py` — `implemented core test`. Strongest direct evidence for LoRA basics.
- `tests/test_recurrence.py` — `implemented core test`. Strongest direct evidence for recurrence/backbone behavior.
- `tests/test_register_notebook_run.py` — `drifted test`. Mostly useful, but two expectations are stale versus tightened Phase B gate behavior.
- `tests/test_retrieval_addon.py` — `implemented scaffold test`. Confirms retrieval stub contract only.
- `tests/test_ternary.py` — `implemented core test`. Strongest direct evidence for ternary activation primitives.
- `tests/test_verifiers_addon.py` — `implemented scaffold test`. Confirms verifier stub contracts only.
- `tests/__init__.py` — `canonical test package marker`.

The files below are `transient` compiled bytecode from pytest/import runs:

- `tests/__pycache__/test_eval_harness_addon.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/test_lora.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/test_recurrence.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/test_register_notebook_run.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/test_retrieval_addon.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/test_ternary.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/test_verifiers_addon.cpython-314-pytest-9.0.2.pyc`
- `tests/__pycache__/__init__.cpython-314.pyc`

## Temp / Notebook Surgery Workspace

- `tmp/build_v15_reset_notebook.py` — `canonical helper`. Script that programmatically builds the new deterministic `v15` reset notebook.

The files below are `transient working extracts` from the large legacy `v15` notebook, used to inspect or rebuild notebook logic cell-by-cell:

- `tmp/notebook_extract/cell_00.py`
- `tmp/notebook_extract/cell_01.py`
- `tmp/notebook_extract/cell_02.py`
- `tmp/notebook_extract/cell_03.py`
- `tmp/notebook_extract/cell_04.py`
- `tmp/notebook_extract/cell_05.py`
- `tmp/notebook_extract/cell_06.py`
- `tmp/notebook_extract/cell_07.py`
- `tmp/notebook_extract/cell_08.py`
- `tmp/notebook_extract/cell_09.py`
- `tmp/notebook_extract/cell_10.py`
- `tmp/notebook_extract/cell_11.py`
- `tmp/notebook_extract/cell_12.py`
- `tmp/notebook_extract/cell_13.py`
- `tmp/notebook_extract/cell_14.py`
- `tmp/notebook_extract/cell_15.py`
- `tmp/notebook_extract/cell_16.py`
- `tmp/notebook_extract/cell_17.py`
- `tmp/notebook_extract/cell_18.py`
- `tmp/notebook_extract/cell_19.py`
- `tmp/notebook_extract/cell_20.py`
- `tmp/notebook_extract/cell_21.py`
- `tmp/notebook_extract/cell_22.py`
- `tmp/notebook_extract/cell_23.py`
- `tmp/notebook_extract/cell_24.py`
- `tmp/notebook_extract/cell_25.py`
- `tmp/notebook_extract/cell_26.py`

## Verifiers

- `verifiers/base.py` — `scaffold`. Verifier protocol plus deterministic stub-result helper.
- `verifiers/code_sandbox_verifier.py` — `scaffold`. Explicit Phase A stub.
- `verifiers/python_math_verifier.py` — `scaffold`. Explicit Phase A stub.
- `verifiers/retrieval_citation_verifier.py` — `scaffold`. Explicit Phase A stub.
- `verifiers/__init__.py` — `scaffold package export`.

The files below are `transient` compiled bytecode:

- `verifiers/__pycache__/base.cpython-314.pyc`
- `verifiers/__pycache__/code_sandbox_verifier.cpython-314.pyc`
- `verifiers/__pycache__/python_math_verifier.cpython-314.pyc`
- `verifiers/__pycache__/retrieval_citation_verifier.cpython-314.pyc`
- `verifiers/__pycache__/__init__.cpython-314.pyc`

## Root Bytecode Cache

- `__pycache__/config.cpython-314.pyc` — `transient`. Compiled bytecode for `config.py`.

## Bottom-Line Assessment For Project Management

If this repository is judged as a research program, it has a strong and well-documented arc through `v14.3` and a credible operational history.

If it is judged as a production-grade Python package, it is not there yet. The codebase still contains:

- a real core prototype,
- a large amount of notebook-centered research logic,
- multiple scaffold subsystems,
- partial governance enforcement,
- low test coverage outside a few core files,
- and a current scientific blocker at `v15`.

The most accurate management statement is:

`Gerhard` is a serious research repo with working core components and working reporting mechanics, but the current program objective is blocked because the semantic-spike validation gate has not been passed, and several newer supporting systems are still scaffolds or have drifted from their intended contracts.
