# RunPod Notebook Handoff

## Purpose
Define the exact execution flow for Phase B when heavy validation or training runs happen on RunPod.

This flow is notebook-first and keeps repo mutation on the laptop side.

## Notebook Roles
- Fresh rerun on Colab: use `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb`.
- Checkpoint-only validation on RunPod: use `notebooks/asnn_goose_v15_runpod_operator.ipynb`.
- Current fresh-training research logic lives in `notebooks/asnn_goose_colab_v15.ipynb`, but direct human execution should go through the fresh-rerun launcher.
- `notebooks/asnn_goose_v15_reset_master.ipynb` remains the checkpoint-gated validation notebook, not the default fresh-training target.

## Execution Order
1. Run `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` on Colab.
2. Let the launcher clone `https://github.com/dttdrv/gerhard.git` if needed, set env overrides, and execute `notebooks/asnn_goose_colab_v15.ipynb` in a clean subprocess.
3. Bring the final dossier and artifact bundle back to the laptop repo.
4. Register locally on the laptop from the dossier, then read the repo truth files and stop.

Why this changed:
- there is no repo-local checkpoint available for the reset notebook path right now
- `notebooks/asnn_goose_v15_reset_master.ipynb` is explicitly checkpoint-gated
- the fresh rerun must therefore start from the older training-capable notebook, but with notebook-side registration disabled and dossier-first local ingestion preserved

## Fresh-Rerun Launcher Knobs
Edit only the top constants in `notebooks/asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb` when needed:

```python
RUN_LABEL = "fresh_rerun"
SEED = 42
DISTILL_STEPS_OVERRIDE = ""
BATCH_SIZE_OVERRIDE = ""
ACCUMULATION_STEPS_OVERRIDE = ""
EVAL_INTERVAL_OVERRIDE = ""
```

The launcher passes these into the training notebook:
- `GERHARD_RUN_ID`
- `GERHARD_SEED`
- `GERHARD_BATCH_SIZE`
- `GERHARD_ACCUMULATION_STEPS`
- `GERHARD_EVAL_INTERVAL`
- `GERHARD_DISTILL_STEPS` only when overridden
- `GERHARD_ENABLE_REGISTER_RUN=0`
- `GERHARD_ENABLE_AUTODOWNLOAD_DOSSIER=0`
- `GERHARD_NOTEBOOK_PATH`
- `GIT_COMMIT`

Default GPU policy:
- T4: batch size `4`, accumulation `4`
- L4: batch size `8`, accumulation `2`
- unknown GPU: batch size `4`, accumulation `2`

Do not change notebook logic, thresholds, or model math during the rerun.

## Structural Stop Conditions
Stop if any of these fail:
- the training notebook exits non-zero
- any required output file is missing
- `v15_best.pt` is missing from the per-run artifact bundle
- the notebook emits a traceback before dossier generation

Red MI/CKA thresholds remain scientific evidence, not tooling failure.

## Files To Bring Back To The Laptop
Keep these files together in one folder:
1. `run_dossier_<run_id>.html`
2. `eval_suite.json`
3. `metrics.json`
4. `config.yaml`
5. `seed.txt`
6. `v15_spikingbrain.json`
7. `v15_best.pt`

The fresh-rerun launcher zips the run artifact directory automatically after a structurally complete run.

Detailed evidence files now produced by the fresh rerun:
- `environment.json`
- `training_curves.json`
- `hardware_stats.json`
- `spike_analysis.json`
- `validation_tests.json`
- `control_suite.json`
- `checkpoint_metadata.json`
- `figures_index.json`
- `detailed_results.json`
- `artifact_manifest.json`
- `executed_training_notebook.ipynb`
- `operator_env.json`
- `launcher_bundle_manifest.json`

## Laptop-Side Registration
Preferred:

`python scripts/register_dossier_run.py --dossier <path_to_run_dossier_<run_id>.html> --phase B`

Fallback if dossier ingestion is not possible:

`python scripts/register_notebook_run.py --run-id <run_id> --phase B --source-dir <artifact_dir>`

The preferred dossier ingestion path now:
- accepts both consolidated dossiers and reset-notebook raw-payload dossiers,
- rejects unsafe `run_id` values,
- reconstructs into a clean staging directory,
- requires artifact-provided commit plus fingerprint fields for a green reproducibility gate.

## Checkpoint-Only Path After Fresh Rerun
Once the fresh rerun produces a new `v15_best.pt`, the checkpoint-only path becomes available again:
1. use `notebooks/asnn_goose_v15_runpod_operator.ipynb`
2. point it at the exported checkpoint
3. run `SMOKE -> DIAGNOSE -> FULL` against `notebooks/asnn_goose_v15_reset_master.ipynb` only if additional diagnosis is still needed

## Read The Truth Files And Stop
After local registration, inspect:
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

That is the stopping point for this phase execution cycle.
