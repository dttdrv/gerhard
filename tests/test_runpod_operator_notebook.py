"""
Structural checks for the RunPod operator notebook.
"""
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "asnn_goose_v15_runpod_operator.ipynb"


def test_runpod_operator_notebook_targets_reset_notebook_and_disables_registration():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    joined_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict)
    )

    assert "notebooks/asnn_goose_v15_reset_master.ipynb" in joined_source
    assert "asnn_goose_colab_v15.ipynb" not in joined_source
    assert "GERHARD_ENABLE_REGISTER_RUN = False" in joined_source
    assert "nbconvert" in joined_source
    assert "operator_env.json" in joined_source
