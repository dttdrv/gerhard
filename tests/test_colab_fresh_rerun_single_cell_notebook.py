"""
Structural checks for the fresh-rerun Colab single-cell launcher and its training target.
"""
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "notebooks" / "asnn_goose_v15_colab_fresh_rerun_single_cell.ipynb"
TRAINING_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "asnn_goose_colab_v15.ipynb"


def test_fresh_rerun_launcher_is_single_cell_and_targets_training_notebook():
    notebook = json.loads(LAUNCHER_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    assert len(cells) == 1

    joined_source = "".join(cells[0].get("source", []))
    assert "https://github.com/dttdrv/gerhard.git" in joined_source
    assert "asnn_goose_colab_v15.ipynb" in joined_source
    assert '"GERHARD_ENABLE_REGISTER_RUN": "0"' in joined_source
    assert '"GERHARD_ENABLE_AUTODOWNLOAD_DOSSIER": "0"' in joined_source
    assert '"GIT_COMMIT": git_commit' in joined_source
    assert "v15_best.pt" in joined_source


def test_training_notebook_has_env_controlled_registration_and_fresh_rerun_knobs():
    notebook = json.loads(TRAINING_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    assert "GERHARD_SEED" in joined_source
    assert "GERHARD_DISTILL_STEPS" in joined_source
    assert "GERHARD_BATCH_SIZE" in joined_source
    assert "GERHARD_ACCUMULATION_STEPS" in joined_source
    assert "GERHARD_EVAL_INTERVAL" in joined_source
    assert "GERHARD_RUN_ID" in joined_source
    assert "GERHARD_ENABLE_REGISTER_RUN" in joined_source
    assert "GERHARD_ENABLE_AUTODOWNLOAD_DOSSIER" in joined_source
    assert "Notebook-side registration disabled" in joined_source
    assert "Auto-download disabled via GERHARD_ENABLE_AUTODOWNLOAD_DOSSIER=0" in joined_source
    assert "v15_best.pt" in joined_source
