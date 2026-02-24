# RunPod Notebook Handoff

## Purpose
Define the exact flow when execution happens inside notebooks.

This flow is offline-safe: no external connectivity is required.

## Step 1: Run Notebook
Execute the relevant notebook pass on RunPod (for example `notebooks/asnn_goose_colab_v15.ipynb`).
For current Phase B recovery, run the **full notebook end-to-end** (not a smoke pass).

For `notebooks/asnn_goose_colab_v15.ipynb`, the final save cell now:
1. writes the canonical bundle under `outputs/<run_id>/`,
2. emits required artifacts (`eval_suite.json`, `metrics.json`, `config.yaml`, `seed.txt`, `v15_spikingbrain.json`),
3. generates a single-file dossier with embedded figures and raw data (`run_dossier_<run_id>.html`),
4. attempts auto-download of that dossier in notebook environments,
5. attempts `register_run(...)` automatically when `scripts/register_notebook_run.py` is available.

If notebook code was patched after a failed run, restart kernel and rerun from the first cell.
The v15 runtime fixes modify model/validator definitions and require a clean execution order.

## Step 2: Export Notebook Outputs
At minimum, collect these files from notebook output cells into one folder:
1. `eval_suite.json`
2. `metrics.json`
3. `config.yaml`
4. `seed.txt`
5. phase-specific artifact (for Phase B: `v15_spikingbrain.json`)
6. `run_dossier_<run_id>.html` (single-file detailed report with embedded graphs)

## Step 3: Register The Run In Repo
From repo root:

Preferred (single-file ingestion):

`python3 scripts/register_dossier_run.py --dossier <path_to_run_dossier_<run_id>.html> --phase B`

Alternative (full artifact folder ingestion):

`python3 scripts/register_notebook_run.py --run-id <run_id> --phase B --source-dir <notebook_output_dir>`

Or directly inside a notebook Python cell (preferred integration):

```python
from pathlib import Path
from scripts.register_notebook_run import register_run

result = register_run(
    run_id="20260223T190000Z_v15",
    phase="B",
    source_dir=Path("/workspace/notebook_outputs/v15_run"),
    repo_root=Path("/workspace/gerhard"),
    summary="v15 notebook pass on RunPod",
    next_action="Proceed to next queued autonomous task if gates are green.",
)
print(result)
```

Either command will:
1. Archive outputs into `outputs/<run_id>/`.
2. Generate reports under `reports/<YYYY>/<MM>/`.
3. Update `reports/index.md`.
4. Update `state/program_status.yaml`.
5. Update `state/gate_results.yaml`.
6. Emit needs-input file automatically if red gates are detected.

If the notebook run predates dossier support, generate one in-repo:

`python3 scripts/generate_run_dossier.py --run-dir outputs/<run_id> --phase B`

## Step 4: Read Status
1. Open `reports/index.md` for latest decision.
2. Open `docs/ops/STATUS_BOARD.md` for program-level view.
3. Open `outputs/<run_id>/run_dossier_<run_id>.html` for a fully self-contained, detailed run report.
