# RunPod Notebook Handoff

## Purpose
Define the exact execution flow for Phase B when heavy validation or training runs happen on RunPod.

This flow is notebook-first and keeps repo mutation on the laptop side.

## Notebook Roles
- Open `notebooks/asnn_goose_v15_runpod_operator.ipynb` on RunPod as the operator notebook.
- For Colab T4, use `notebooks/asnn_goose_v15_colab_t4_single_cell.ipynb` as the one-cell launcher notebook.
- The canonical research logic notebook remains `notebooks/asnn_goose_v15_reset_master.ipynb`.
- Treat `notebooks/asnn_goose_colab_v15.ipynb` as historical evidence, not the current execution target.

## Execution Order
1. Use the operator notebook to run checkpoint-only `SMOKE`.
2. If `SMOKE` is structurally clean, switch the operator notebook to `DIAGNOSE`.
3. If `DIAGNOSE` is structurally clean, switch the operator notebook to `FULL`.
4. Export the final dossier and artifact bundle back to the laptop repo.
5. Register locally on the laptop from the dossier, then read the repo truth files and stop.

Do not start with a full rerun. Do not register the run from inside the notebook for the current finish flow.
The operator notebook executes the canonical reset notebook in a fresh subprocess so each mode starts cleanly without creating a second logic fork.
The Colab T4 single-cell notebook does the same thing, but also clones `https://github.com/dttdrv/gerhard.git` automatically when the repo is not already present and attempts checkpoint auto-discovery under Google Drive / Colab paths.

## Required Environment Variables
If you use the operator notebook, edit its config cell instead of manually exporting env vars in the reset notebook.
The effective env vars passed into the canonical reset notebook are:

```python
import os

os.environ["GERHARD_RUN_MODE"] = "SMOKE"  # later: DIAGNOSE, then FULL
os.environ["GERHARD_RUN_ID"] = "v15_preflight_smoke_<timestamp>"
os.environ["GERHARD_CHECKPOINT_PATH"] = "/absolute/path/to/your/checkpoint.pt"

os.environ["GERHARD_ENABLE_DOSSIER_EXPORT"] = "1"
os.environ["GERHARD_ENABLE_REGISTER_RUN"] = "0"
os.environ["GERHARD_ENABLE_AUTODOWNLOAD_DOSSIER"] = "1"

os.environ["GERHARD_BATCH_SIZE"] = "8"
os.environ["GERHARD_SMOKE_BATCHES"] = "2"
os.environ["GERHARD_FULL_BATCHES_SMOKE"] = "4"
```

Mode-specific overrides:

- `SMOKE`
  - `GERHARD_RUN_MODE=SMOKE`
  - `GERHARD_RUN_ID=v15_preflight_smoke_<timestamp>`
- `DIAGNOSE`
  - `GERHARD_RUN_MODE=DIAGNOSE`
  - `GERHARD_RUN_ID=v15_preflight_diagnose_<timestamp>`
  - `GERHARD_FULL_BATCHES_DIAGNOSE=40`
- `FULL`
  - `GERHARD_RUN_MODE=FULL`
  - `GERHARD_RUN_ID=v15_full_<timestamp>`
  - `GERHARD_FULL_BATCHES=20`

If GPU memory is tighter than expected, reduce only `GERHARD_BATCH_SIZE` in this order:
1. `4`
2. `2`

Do not change notebook logic, thresholds, or model math during these runs.

## Structural Stop Conditions
Stop after `SMOKE` or `DIAGNOSE` if any of these fail:
- checkpoint does not load
- student forward fails
- spike info is missing
- logits contain NaN or Inf
- teacher hidden states / activations are missing
- zero-batch or empty-loader guards trigger
- expected output files are missing

Red MI/CKA thresholds alone are not a structural stop condition. They are scientific evidence, not tooling failure.

## Files To Bring Back To The Laptop
Keep these files together in one folder:
1. `run_dossier_<run_id>.html`
2. `eval_suite.json`
3. `metrics.json`
4. `config.yaml`
5. `seed.txt`
6. `v15_spikingbrain.json`

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

## Read The Truth Files And Stop
After local registration, inspect:
1. `reports/index.md`
2. `state/program_status.yaml`
3. `state/gate_results.yaml`
4. `docs/ops/STATUS_BOARD.md`

That is the stopping point for this phase execution cycle.
