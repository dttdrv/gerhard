"""
Tests for notebook run ingestion helpers.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.register_notebook_run import (
    decide_autopilot,
    evaluate_gates,
    register_run,
    update_status_board,
)


def _write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _repro_metadata():
    return (
        "0123456789abcdef0123456789abcdef01234567",
        {
            "config_sha256": "a" * 64,
            "recipe_sha256": "b" * 64,
        },
    )


def _valid_phase_b_artifact():
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


def _write_phase_b_bundle(
    base_dir: Path,
    *,
    include_reproducibility: bool = True,
    v15_payload=None,
):
    commit, fingerprint = _repro_metadata()

    eval_suite = {"summary": {"task_count": 5}}
    metrics = {"ppl": 123.45, "ok": True}
    config = {"seed": 42}

    if include_reproducibility:
        eval_suite["git_hash"] = commit
        eval_suite["fingerprint"] = fingerprint
        metrics["fingerprint"] = fingerprint
        config["git_commit"] = commit
        config["fingerprint"] = fingerprint

    _write_json(base_dir / "eval_suite.json", eval_suite)
    _write_json(base_dir / "metrics.json", metrics)
    (base_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (base_dir / "seed.txt").write_text("42\n", encoding="utf-8")
    _write_json(base_dir / "v15_spikingbrain.json", v15_payload or _valid_phase_b_artifact())


def test_evaluate_gates_red_when_required_artifacts_missing(tmp_path):
    gates, needs_input = evaluate_gates(tmp_path, phase="B")

    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["required_artifacts_presence"] == "red"
    assert statuses["phase_b_artifacts_presence"] == "red"
    assert decide_autopilot(gates, force_mode="auto") == "PAUSE_NEEDS_INPUT"
    assert len(needs_input) >= 1


def test_evaluate_gates_continue_when_artifacts_present_and_finite(tmp_path):
    _write_phase_b_bundle(tmp_path)

    gates, needs_input = evaluate_gates(tmp_path, phase="B")
    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["required_artifacts_presence"] == "green"
    assert statuses["phase_b_artifacts_presence"] == "green"
    assert statuses["metrics_numerical_sanity"] == "green"
    assert statuses["reproducibility_metadata"] == "green"
    assert statuses["phase_b_scientific_thresholds"] == "green"
    assert needs_input == []
    assert decide_autopilot(gates, force_mode="auto") == "CONTINUE"


def test_evaluate_gates_red_when_phase_b_nested_metrics_missing(tmp_path):
    _write_phase_b_bundle(
        tmp_path,
        v15_payload={"validation": {"health": {"dead_neuron_pct": 0.01}}},
    )

    gates, needs_input = evaluate_gates(tmp_path, phase="B")
    status_by_name = {g["gate_name"]: g for g in gates}
    gate = status_by_name["phase_b_scientific_thresholds"]
    assert gate["status"] == "red"
    assert "missing metrics" in gate["observed"]
    assert "Provide complete v15_spikingbrain.json with health, mutual_information, cka metrics." in needs_input


def test_evaluate_gates_red_on_non_finite_metrics(tmp_path):
    _write_phase_b_bundle(tmp_path)
    _write_json(tmp_path / "metrics.json", {"ppl": float("inf")})

    gates, _ = evaluate_gates(tmp_path, phase="B")
    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["metrics_numerical_sanity"] == "red"


def test_evaluate_gates_red_on_missing_reproducibility_metadata(tmp_path):
    _write_phase_b_bundle(tmp_path, include_reproducibility=False)

    gates, needs_input = evaluate_gates(tmp_path, phase="B")
    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["reproducibility_metadata"] == "red"
    assert (
        "Provide reproducibility metadata: git commit and fingerprint fields "
        "(config_sha256, recipe_sha256)."
    ) in needs_input
    assert decide_autopilot(gates, force_mode="auto") == "PAUSE_NEEDS_INPUT"


def test_decision_force_modes_override_auto():
    gates = [{"gate_name": "x", "status": "red"}]
    assert decide_autopilot(gates, force_mode="continue") == "CONTINUE"
    assert decide_autopilot(gates, force_mode="pause") == "PAUSE_NEEDS_INPUT"


def test_update_status_board_rewrites_autogen_block(tmp_path):
    board = tmp_path / "STATUS_BOARD.md"
    board.write_text(
        "# Board\n\n"
        "**As Of**: 2026-02-01  \n"
        "- Current active phase: **old phase**\n\n"
        "<!-- AUTOGEN_LATEST_RUN_START -->\n"
        "old block\n"
        "<!-- AUTOGEN_LATEST_RUN_END -->\n",
        encoding="utf-8",
    )

    gates = [
        {"gate_name": "a", "status": "red"},
        {"gate_name": "b", "status": "yellow"},
    ]
    update_status_board(
        status_board_path=board,
        run_id="r1",
        timestamp_utc="2026-02-23T12:00:00Z",
        phase_name="B_v15_spikingbrain_validation",
        decision="PAUSE_NEEDS_INPUT",
        gates=gates,
        next_action="next step",
    )

    text = board.read_text(encoding="utf-8")
    assert "Run ID: `r1`" in text
    assert "Phase: `B_v15_spikingbrain_validation`" in text
    assert "Red gates: `1`" in text
    assert "Yellow gates: `1`" in text


def test_register_run_callable_creates_reports_and_state(tmp_path):
    repo_root = tmp_path / "repo"
    source = tmp_path / "notebook_out"
    source.mkdir(parents=True, exist_ok=True)

    _write_phase_b_bundle(source)

    (repo_root / "docs" / "ops").mkdir(parents=True, exist_ok=True)
    (repo_root / "state").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "ops" / "STATUS_BOARD.md").write_text(
        "# Board\n\n"
        "**As Of**: 2026-02-01  \n"
        "- Current active phase: **old phase**\n\n"
        "<!-- AUTOGEN_LATEST_RUN_START -->\n"
        "old block\n"
        "<!-- AUTOGEN_LATEST_RUN_END -->\n",
        encoding="utf-8",
    )

    result = register_run(
        run_id="r2",
        phase="B",
        source_dir=source,
        repo_root=repo_root,
        decision_mode="auto",
        summary="notebook pass",
        next_action="next action",
    )

    assert result["decision"] == "CONTINUE"
    assert (repo_root / result["report_md"]).exists()
    assert (repo_root / result["report_json"]).exists()
    assert (repo_root / "reports" / "index.md").exists()
    program_status_path = repo_root / "state" / "program_status.yaml"
    assert program_status_path.exists()
    assert (repo_root / "state" / "gate_results.yaml").exists()
    assert (repo_root / "outputs" / "r2" / "v15_spikingbrain.json").exists()

    program_status_text = program_status_path.read_text(encoding="utf-8")
    assert "phase_status:" in program_status_text
    assert "current_best_metrics:" in program_status_text

    report_json = json.loads((repo_root / result["report_json"]).read_text(encoding="utf-8"))
    gate_status = {g["gate_name"]: g["status"] for g in report_json["gate_scorecard"]}
    assert report_json["commit"] == _repro_metadata()[0]
    assert report_json["commit_source"] == "artifact"
    assert gate_status["reproducibility_metadata"] == "green"


def test_register_run_pauses_when_reproducibility_metadata_missing(tmp_path):
    repo_root = tmp_path / "repo"
    source = tmp_path / "notebook_out"
    source.mkdir(parents=True, exist_ok=True)

    _write_phase_b_bundle(source, include_reproducibility=False)

    (repo_root / "docs" / "ops").mkdir(parents=True, exist_ok=True)
    (repo_root / "state").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "ops" / "STATUS_BOARD.md").write_text(
        "# Board\n\n"
        "**As Of**: 2026-02-01  \n"
        "- Current active phase: **old phase**\n\n"
        "<!-- AUTOGEN_LATEST_RUN_START -->\n"
        "old block\n"
        "<!-- AUTOGEN_LATEST_RUN_END -->\n",
        encoding="utf-8",
    )

    result = register_run(
        run_id="r3",
        phase="B",
        source_dir=source,
        repo_root=repo_root,
        decision_mode="auto",
        summary="notebook pass",
        next_action="next action",
    )

    assert result["decision"] == "PAUSE_NEEDS_INPUT"
    report_json = json.loads((repo_root / result["report_json"]).read_text(encoding="utf-8"))
    gate_by_name = {g["gate_name"]: g for g in report_json["gate_scorecard"]}
    assert report_json["commit"] is None
    assert gate_by_name["reproducibility_metadata"]["status"] == "red"
    assert "commit" in gate_by_name["reproducibility_metadata"]["observed"]
    needs_input_path = repo_root / "reports" / "2026" / "03" / "r3_needs_input.md"
    assert needs_input_path.exists()
    assert "reproducibility" in needs_input_path.read_text(encoding="utf-8").lower()


def test_register_run_rejects_unsafe_run_id(tmp_path):
    repo_root = tmp_path / "repo"
    source = tmp_path / "notebook_out"
    source.mkdir(parents=True, exist_ok=True)
    _write_phase_b_bundle(source)

    with pytest.raises(ValueError, match="Unsafe run_id"):
        register_run(
            run_id="../escape",
            phase="B",
            source_dir=source,
            repo_root=repo_root,
            decision_mode="auto",
            summary="notebook pass",
            next_action="next action",
        )
