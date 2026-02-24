"""
Tests for notebook run ingestion helpers.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.register_notebook_run import (
    evaluate_gates,
    decide_autopilot,
    register_run,
    update_status_board,
)


def _write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_evaluate_gates_red_when_required_artifacts_missing(tmp_path):
    gates, needs_input = evaluate_gates(tmp_path, phase="B")

    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["required_artifacts_presence"] == "red"
    assert statuses["phase_b_artifacts_presence"] == "red"
    assert decide_autopilot(gates, force_mode="auto") == "PAUSE_NEEDS_INPUT"
    assert len(needs_input) >= 1


def test_evaluate_gates_continue_when_artifacts_present_and_finite(tmp_path):
    _write_json(tmp_path / "eval_suite.json", {"summary": {"task_count": 5}})
    _write_json(tmp_path / "metrics.json", {"ppl": 123.45, "ok": True})
    (tmp_path / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (tmp_path / "seed.txt").write_text("42\n", encoding="utf-8")
    _write_json(tmp_path / "v15_spikingbrain.json", {"overall_pass": True})

    gates, needs_input = evaluate_gates(tmp_path, phase="B")
    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["required_artifacts_presence"] == "green"
    assert statuses["phase_b_artifacts_presence"] == "green"
    assert statuses["metrics_numerical_sanity"] == "green"
    assert needs_input == []
    assert decide_autopilot(gates, force_mode="auto") == "CONTINUE"


def test_evaluate_gates_red_on_non_finite_metrics(tmp_path):
    _write_json(tmp_path / "eval_suite.json", {"summary": {"task_count": 5}})
    _write_json(tmp_path / "metrics.json", {"ppl": float("inf")})
    (tmp_path / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (tmp_path / "seed.txt").write_text("42\n", encoding="utf-8")
    _write_json(tmp_path / "v15_spikingbrain.json", {"overall_pass": True})

    gates, _ = evaluate_gates(tmp_path, phase="B")
    statuses = {g["gate_name"]: g["status"] for g in gates}
    assert statuses["metrics_numerical_sanity"] == "red"


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

    _write_json(source / "eval_suite.json", {"summary": {"task_count": 5}})
    _write_json(source / "metrics.json", {"ppl": 123.45, "ok": True})
    (source / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (source / "seed.txt").write_text("42\n", encoding="utf-8")
    _write_json(source / "v15_spikingbrain.json", {"overall_pass": True})

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
