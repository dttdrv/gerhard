"""
Tests for single-file dossier ingestion.
"""
import html
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.register_dossier_run import reconstruct_from_dossier
from scripts.register_notebook_run import register_run


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_phase_b_validation_payload() -> dict:
    return {
        "validation": {
            "health": {
                "dead_neuron_pct": 0.01,
                "saturated_neuron_pct": 0.02,
                "firing_rate_mean": 0.35,
            },
            "mutual_information": {
                "mutual_information": 0.21,
            },
            "cka": {
                "cka_mean": 0.41,
            },
        },
        "overall_pass": True,
    }


def _make_notebook_style_dossier(run_id: str) -> str:
    fingerprint = {
        "config_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
    }
    metrics = {
        "run_id": run_id,
        "seed": 42,
        "fingerprint": fingerprint,
        "v15_mutual_information": 0.21,
        "v15_cka_mean": 0.41,
    }
    eval_suite = {
        "summary": {"task_count": 5},
        "git_hash": "0123456789abcdef0123456789abcdef01234567",
        "fingerprint": fingerprint,
    }
    config = {
        "run_id": run_id,
        "phase": "B",
        "seed": 42,
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "fingerprint": fingerprint,
        "architecture": {
            "d_model": 768,
            "n_layers": 5,
        },
    }
    v15_payload = _make_phase_b_validation_payload()

    return f"""<!DOCTYPE html>
<html>
<body>
  <h2>Config</h2>
  <pre>{html.escape(json.dumps(config, indent=2))}</pre>
  <h2>Embedded Raw Payloads</h2>
  <details><summary>metrics.json</summary><pre>{html.escape(json.dumps(metrics, indent=2))}</pre></details>
  <details><summary>eval_suite.json</summary><pre>{html.escape(json.dumps(eval_suite, indent=2))}</pre></details>
  <details><summary>v15_spikingbrain.json</summary><pre>{html.escape(json.dumps(v15_payload, indent=2))}</pre></details>
</body>
</html>
"""


def _make_consolidated_dossier(run_id: str) -> str:
    commit = "0123456789abcdef0123456789abcdef01234567"
    fingerprint = {
        "config_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
    }
    consolidated = {
        "run_id": run_id,
        "summary_metrics": {
            "run_id": run_id,
            "seed": 42,
            "fingerprint": fingerprint,
            "v15_mutual_information": 0.21,
            "v15_cka_mean": 0.41,
        },
        "eval_suite": {
            "summary": {"task_count": 5},
            "git_hash": commit,
            "fingerprint": fingerprint,
        },
        "v15_validation": _make_phase_b_validation_payload(),
    }
    results_snapshot = {
        "training_config": {"batch_size": 8},
        "architecture": {"d_model": 768, "n_layers": 5},
        "description": "consolidated dossier",
    }

    return f"""<!DOCTYPE html>
<html>
<body>
  <details><summary>Consolidated payload</summary><pre>{html.escape(json.dumps(consolidated, indent=2))}</pre></details>
  <details><summary>results.json snapshot</summary><pre>{html.escape(json.dumps(results_snapshot, indent=2))}</pre></details>
</body>
</html>
"""


def test_register_run_from_reset_notebook_style_dossier(tmp_path):
    repo_root = tmp_path / "repo"
    dossier_path = tmp_path / "run_dossier_v15_reset.html"

    _write_text(dossier_path, _make_notebook_style_dossier("v15_reset_example"))

    (repo_root / "docs" / "ops").mkdir(parents=True, exist_ok=True)
    (repo_root / "state").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "ops" / "STATUS_BOARD.md").write_text(
        "# Board\n\n"
        "<!-- AUTOGEN_LATEST_RUN_START -->\n"
        "old block\n"
        "<!-- AUTOGEN_LATEST_RUN_END -->\n",
        encoding="utf-8",
    )

    run_id, source_dir = reconstruct_from_dossier(
        dossier_path=dossier_path,
        phase="B",
        repo_root=repo_root,
    )

    result = register_run(
        run_id=run_id,
        phase="B",
        source_dir=source_dir,
        repo_root=repo_root,
        summary="reset notebook dossier ingested",
        next_action="continue",
    )

    report_json = json.loads((repo_root / result["report_json"]).read_text(encoding="utf-8"))
    gate_status = {g["gate_name"]: g["status"] for g in report_json["gate_scorecard"]}

    assert result["decision"] == "CONTINUE"
    assert report_json["commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert gate_status["reproducibility_metadata"] == "green"
    assert (source_dir / "v15_spikingbrain.json").exists()
    assert (source_dir / dossier_path.name).exists()


def test_register_run_from_consolidated_dossier(tmp_path):
    repo_root = tmp_path / "repo"
    dossier_path = tmp_path / "run_dossier_v15_consolidated.html"

    _write_text(dossier_path, _make_consolidated_dossier("v15_consolidated_example"))

    (repo_root / "docs" / "ops").mkdir(parents=True, exist_ok=True)
    (repo_root / "state").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "ops" / "STATUS_BOARD.md").write_text(
        "# Board\n\n"
        "<!-- AUTOGEN_LATEST_RUN_START -->\n"
        "old block\n"
        "<!-- AUTOGEN_LATEST_RUN_END -->\n",
        encoding="utf-8",
    )

    run_id, source_dir = reconstruct_from_dossier(
        dossier_path=dossier_path,
        phase="B",
        repo_root=repo_root,
    )

    result = register_run(
        run_id=run_id,
        phase="B",
        source_dir=source_dir,
        repo_root=repo_root,
        summary="consolidated dossier ingested",
        next_action="continue",
    )

    report_json = json.loads((repo_root / result["report_json"]).read_text(encoding="utf-8"))
    gate_status = {g["gate_name"]: g["status"] for g in report_json["gate_scorecard"]}

    assert result["decision"] == "CONTINUE"
    assert report_json["commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert gate_status["reproducibility_metadata"] == "green"
    assert (source_dir / "results.json").exists()


def test_reconstruct_from_dossier_rejects_unsafe_run_id(tmp_path):
    repo_root = tmp_path / "repo"
    dossier_path = tmp_path / "run_dossier_escape.html"

    _write_text(dossier_path, _make_notebook_style_dossier("../../escape"))

    with pytest.raises(ValueError, match="Unsafe run_id"):
        reconstruct_from_dossier(
            dossier_path=dossier_path,
            phase="B",
            repo_root=repo_root,
        )


def test_reconstruct_from_dossier_replaces_existing_staging_dir(tmp_path):
    repo_root = tmp_path / "repo"
    dossier_path = tmp_path / "run_dossier_v15_reset.html"

    _write_text(dossier_path, _make_notebook_style_dossier("v15_repeatable"))

    _, source_dir = reconstruct_from_dossier(
        dossier_path=dossier_path,
        phase="B",
        repo_root=repo_root,
    )
    stale_path = source_dir / "stale.txt"
    stale_path.write_text("stale\n", encoding="utf-8")

    _, rebuilt_source_dir = reconstruct_from_dossier(
        dossier_path=dossier_path,
        phase="B",
        repo_root=repo_root,
    )

    assert rebuilt_source_dir == source_dir
    assert not stale_path.exists()
