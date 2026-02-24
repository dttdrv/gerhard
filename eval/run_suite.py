"""
Phase A unified evaluation harness (skeleton).

This module provides one entrypoint that writes a single JSON artifact at:
`outputs/eval_suite.json` (or a caller-provided path).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import yaml


DEFAULT_SUITE_CONFIG_PATH = Path("eval/suite.yaml")
DEFAULT_OUTPUT_PATH = Path("outputs/eval_suite.json")
SCHEMA_VERSION_FALLBACK = "v0.1"

TaskEvaluator = Callable[[Dict[str, Any]], Dict[str, Any]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        return value or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _placeholder_result(metric_name: str, note: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "metric": metric_name,
        "value": None,
        "notes": note,
    }


def _skipped_result(metric_name: str, note: str) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "metric": metric_name,
        "value": None,
        "notes": note,
    }


def load_suite_config(
    config_path: Union[Path, str] = DEFAULT_SUITE_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Load evaluation suite YAML configuration.

    If the file does not exist, returns a minimal in-memory default.
    """
    path = Path(config_path)
    if not path.exists():
        return {
            "version": SCHEMA_VERSION_FALLBACK,
            "tasks": [
                {"name": "ppl"},
                {"name": "instruction_following"},
                {"name": "code"},
                {"name": "reasoning"},
                {"name": "retrieval_grounded"},
            ],
        }

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid suite config at {path}: expected mapping")
    return loaded


def _default_evaluators() -> Dict[str, TaskEvaluator]:
    return {
        "ppl": lambda _ctx: _placeholder_result(
            "ppl",
            "PPL evaluator not wired yet (Phase A scaffold).",
        ),
        "instruction_following": lambda _ctx: _placeholder_result(
            "pass_rate",
            "Instruction-following evaluator not wired yet (Phase A scaffold).",
        ),
        "code": lambda _ctx: _placeholder_result(
            "pass_rate",
            "Code evaluator not wired yet (Phase A scaffold).",
        ),
        "reasoning": lambda _ctx: _placeholder_result(
            "pass_rate",
            "Reasoning evaluator not wired yet (Phase A scaffold).",
        ),
        "retrieval_grounded": lambda _ctx: _placeholder_result(
            "citation_correctness",
            "Retrieval-grounded evaluator not wired yet (Phase A scaffold).",
        ),
    }


def _run_task(
    task_name: str,
    evaluator: Optional[TaskEvaluator],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    if evaluator is None:
        return _placeholder_result(
            "unknown",
            f"No evaluator registered for task '{task_name}'.",
        )

    result = evaluator(context)
    if not isinstance(result, dict):
        raise TypeError(
            f"Evaluator for task '{task_name}' returned non-dict: {type(result)!r}"
        )
    result.setdefault("status", "ok")
    result.setdefault("notes", "")
    return result


def run_eval_suite(
    output_path: Union[Path, str] = DEFAULT_OUTPUT_PATH,
    suite_config_path: Union[Path, str] = DEFAULT_SUITE_CONFIG_PATH,
    seed: int = 42,
    include_retrieval: bool = False,
    evaluators: Optional[Mapping[str, TaskEvaluator]] = None,
    git_hash: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the unified evaluation harness and write one JSON artifact.

    Returns:
        The same dictionary written to disk.
    """
    suite_config = load_suite_config(suite_config_path)
    configured_evaluators = dict(_default_evaluators())
    if evaluators:
        configured_evaluators.update(evaluators)

    context = {
        "seed": seed,
        "include_retrieval": include_retrieval,
        "suite_config": suite_config,
    }

    tasks: Dict[str, Dict[str, Any]] = {}
    ordered_tasks = [
        "ppl",
        "instruction_following",
        "code",
        "reasoning",
        "retrieval_grounded",
    ]
    for task_name in ordered_tasks:
        if task_name == "retrieval_grounded" and not include_retrieval:
            tasks[task_name] = _skipped_result(
                "citation_correctness",
                "Retrieval disabled for this run.",
            )
            continue
        tasks[task_name] = _run_task(
            task_name=task_name,
            evaluator=configured_evaluators.get(task_name),
            context=context,
        )

    metadata = {
        "schema_version": suite_config.get("version", SCHEMA_VERSION_FALLBACK),
        "created_at_utc": _utc_now_iso(),
        "seed": seed,
        "git_hash": git_hash or _get_git_hash(),
        "suite_config_path": str(Path(suite_config_path)),
        "retrieval_enabled": include_retrieval,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    status_counts: Dict[str, int] = {}
    for task_result in tasks.values():
        status = str(task_result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    suite_result = {
        "metadata": metadata,
        "tasks": tasks,
        "summary": {
            "task_count": len(tasks),
            "status_counts": status_counts,
        },
    }

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(suite_result, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return suite_result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASNN-Goose unified eval suite.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path (default: outputs/eval_suite.json).",
    )
    parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE_CONFIG_PATH),
        help="Suite YAML path (default: eval/suite.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed metadata for this run.",
    )
    parser.add_argument(
        "--enable_retrieval",
        action="store_true",
        help="Enable retrieval-grounded task execution.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    run_eval_suite(
        output_path=Path(args.output),
        suite_config_path=Path(args.suite),
        seed=args.seed,
        include_retrieval=args.enable_retrieval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
