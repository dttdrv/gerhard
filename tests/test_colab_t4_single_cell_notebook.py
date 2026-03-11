"""
Structural checks for the Colab T4 single-cell launcher notebook.
"""
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "asnn_goose_v15_colab_t4_single_cell.ipynb"


def test_colab_t4_single_cell_notebook_is_single_cell_and_clones_repo():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    assert len(cells) == 1

    joined_source = "".join(cells[0].get("source", []))
    assert "https://github.com/dttdrv/gerhard.git" in joined_source
    assert "asnn_goose_v15_reset_master.ipynb" in joined_source
    assert "GERHARD_ENABLE_REGISTER_RUN" in joined_source
    assert '"GERHARD_ENABLE_REGISTER_RUN": "0"' in joined_source
    assert 'RUN_MODE = "SMOKE"' in joined_source
