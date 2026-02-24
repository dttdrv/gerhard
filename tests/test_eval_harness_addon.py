"""
Tests for Phase A add-on evaluation harness scaffolding.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_suite import run_eval_suite


def _stub_ok(metric_name: str, value: float):
    def _evaluator(_context):
        return {
            "status": "ok",
            "metric": metric_name,
            "value": value,
            "notes": "stubbed evaluator",
        }
    return _evaluator


def test_run_eval_suite_writes_single_json(tmp_path):
    """Harness should write one versioned JSON with metadata and task results."""
    output_path = tmp_path / "eval_suite.json"
    suite = run_eval_suite(
        output_path=output_path,
        seed=7,
        include_retrieval=False,
        git_hash="deadbeef",
        evaluators={
            "ppl": _stub_ok("ppl", 42.0),
            "instruction_following": _stub_ok("pass_rate", 0.8),
            "code": _stub_ok("pass_rate", 0.75),
            "reasoning": _stub_ok("pass_rate", 0.66),
        },
    )

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == suite
    assert loaded["metadata"]["seed"] == 7
    assert loaded["metadata"]["git_hash"] == "deadbeef"
    assert loaded["metadata"]["schema_version"].startswith("v")
    assert loaded["tasks"]["retrieval_grounded"]["status"] == "skipped"


def test_retrieval_eval_only_runs_when_enabled(tmp_path):
    """Retrieval-grounded evaluation should only run when explicitly enabled."""
    output_path = tmp_path / "eval_suite_retrieval.json"
    suite = run_eval_suite(
        output_path=output_path,
        seed=11,
        include_retrieval=True,
        git_hash="cafebabe",
        evaluators={
            "ppl": _stub_ok("ppl", 10.5),
            "instruction_following": _stub_ok("pass_rate", 0.5),
            "code": _stub_ok("pass_rate", 0.4),
            "reasoning": _stub_ok("pass_rate", 0.6),
            "retrieval_grounded": _stub_ok("citation_correctness", 0.7),
        },
    )

    assert output_path.exists()
    assert suite["tasks"]["retrieval_grounded"]["status"] == "ok"
    assert suite["tasks"]["retrieval_grounded"]["metric"] == "citation_correctness"
