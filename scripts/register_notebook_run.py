"""
Notebook-first run ingestion for Gerhard.

This script archives notebook-produced outputs and updates autonomous state/report files:
- outputs/<run_id>/*
- reports/<YYYY>/<MM>/<run_id>.md
- reports/<YYYY>/<MM>/<run_id>.json
- reports/index.md
- state/program_status.yaml
- state/gate_results.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


PHASE_TO_NAME = {
    "A": "A_engineering_guardrails",
    "B": "B_v15_spikingbrain_validation",
    "C": "C_temporal_coding_proof",
    "D": "D_v16_sparse_ops",
    "E": "E_v17_efficiency_metrics",
    "F": "F_v18_ablations",
    "G": "G_v19_publication_repro",
    "H": "H_generalist_scorecard",
    "I": "I_retrieval_world_fit",
    "J": "J_posttraining",
    "K": "K_scaling_ladder",
}

PHASE_KEYS = tuple(PHASE_TO_NAME.keys())

REQUIRED_ARTIFACTS = [
    "eval_suite.json",
    "metrics.json",
    "config.yaml",
    "seed.txt",
]

PHASE_REQUIRED_ARTIFACTS = {
    "B": ["v15_spikingbrain.json"],
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _copy_tree_merge(src: Path, dst: Path) -> None:
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, int):
        return False
    if isinstance(value, dict):
        return any(_contains_non_finite(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(v) for v in value)
    return False


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _collect_artifact_paths(output_dir: Path, repo_root: Path | None = None) -> List[str]:
    files: List[str] = []
    for p in output_dir.rglob("*"):
        if p.is_file():
            if repo_root is not None:
                try:
                    files.append(p.relative_to(repo_root).as_posix())
                    continue
                except ValueError:
                    pass
            files.append(str(p.as_posix()))
    files.sort()
    return files


def evaluate_gates(output_dir: Path, phase: str) -> Tuple[List[Dict[str, str]], List[str]]:
    missing_required = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]
    missing_phase_required = [
        name for name in PHASE_REQUIRED_ARTIFACTS.get(phase, [])
        if not (output_dir / name).exists()
    ]

    metrics_non_finite = False
    for candidate in ("metrics.json", "eval_suite.json"):
        p = output_dir / candidate
        if not p.exists():
            continue
        try:
            metrics_non_finite = metrics_non_finite or _contains_non_finite(_load_json(p))
        except json.JSONDecodeError:
            metrics_non_finite = True

    gates: List[Dict[str, str]] = []
    needs_input: List[str] = []

    if missing_required:
        gates.append({
            "gate_name": "required_artifacts_presence",
            "status": "red",
            "observed": f"missing: {', '.join(missing_required)}",
            "threshold": "all required artifacts must exist",
            "delta_vs_baseline": "n/a",
            "notes": "Upload notebook outputs including all required files.",
        })
        needs_input.append(f"Provide missing required artifacts: {', '.join(missing_required)}")
    else:
        gates.append({
            "gate_name": "required_artifacts_presence",
            "status": "green",
            "observed": "all required artifacts present",
            "threshold": "all required artifacts must exist",
            "delta_vs_baseline": "n/a",
            "notes": "pass",
        })

    if missing_phase_required:
        gates.append({
            "gate_name": f"phase_{phase.lower()}_artifacts_presence",
            "status": "red",
            "observed": f"missing: {', '.join(missing_phase_required)}",
            "threshold": "all phase-specific artifacts must exist",
            "delta_vs_baseline": "n/a",
            "notes": "Phase-specific required artifact missing.",
        })
        needs_input.append(
            f"Provide phase-{phase} artifact(s): {', '.join(missing_phase_required)}"
        )
    else:
        if PHASE_REQUIRED_ARTIFACTS.get(phase):
            gates.append({
                "gate_name": f"phase_{phase.lower()}_artifacts_presence",
                "status": "green",
                "observed": "all phase-specific artifacts present",
                "threshold": "all phase-specific artifacts must exist",
                "delta_vs_baseline": "n/a",
                "notes": "pass",
            })

    if metrics_non_finite:
        gates.append({
            "gate_name": "metrics_numerical_sanity",
            "status": "red",
            "observed": "NaN/Inf or invalid JSON found in metrics artifacts",
            "threshold": "all numeric values must be finite",
            "delta_vs_baseline": "n/a",
            "notes": "Fix run outputs before continuation.",
        })
        needs_input.append("Provide metrics artifacts with finite numeric values only.")
    else:
        gates.append({
            "gate_name": "metrics_numerical_sanity",
            "status": "green",
            "observed": "numeric values finite in inspected JSON artifacts",
            "threshold": "all numeric values must be finite",
            "delta_vs_baseline": "n/a",
            "notes": "pass",
        })

    if phase == "B" and not missing_phase_required:
        v15_path = output_dir / "v15_spikingbrain.json"
        v15_data: Dict[str, Any] = {}
        try:
            v15_data = _load_json(v15_path)
        except json.JSONDecodeError:
            v15_data = {}

        validation = v15_data.get("validation", {}) if isinstance(v15_data, dict) else {}
        health = validation.get("health", {}) if isinstance(validation, dict) else {}
        mi_data = validation.get("mutual_information", {}) if isinstance(validation, dict) else {}
        cka_data = validation.get("cka", {}) if isinstance(validation, dict) else {}

        def _to_float(value: Any) -> float | None:
            try:
                out = float(value)
                if not math.isfinite(out):
                    return None
                return out
            except (TypeError, ValueError):
                return None

        dead_neuron_pct = _to_float(health.get("dead_neuron_pct"))
        saturated_neuron_pct = _to_float(health.get("saturated_neuron_pct"))
        firing_rate_mean = _to_float(health.get("firing_rate_mean"))
        mutual_information = _to_float(mi_data.get("mutual_information"))
        cka_mean = _to_float(cka_data.get("cka_mean"))

        threshold_failures: List[str] = []
        missing_metrics: List[str] = []

        if dead_neuron_pct is None:
            missing_metrics.append("dead_neuron_pct")
        elif dead_neuron_pct > 0.05:
            threshold_failures.append(f"dead_neuron_pct={dead_neuron_pct:.4f} > 0.05")

        if saturated_neuron_pct is None:
            missing_metrics.append("saturated_neuron_pct")
        elif saturated_neuron_pct >= 0.10:
            threshold_failures.append(f"saturated_neuron_pct={saturated_neuron_pct:.4f} >= 0.10")

        if mutual_information is None:
            missing_metrics.append("mutual_information")
        elif mutual_information <= 0.10:
            threshold_failures.append(f"mutual_information={mutual_information:.4f} <= 0.10")

        if cka_mean is None:
            missing_metrics.append("cka_mean")
        elif cka_mean <= 0.30:
            threshold_failures.append(f"cka_mean={cka_mean:.4f} <= 0.30")

        if firing_rate_mean is None:
            missing_metrics.append("firing_rate_mean")
        elif firing_rate_mean < 0.20 or firing_rate_mean > 0.60:
            threshold_failures.append(f"firing_rate_mean={firing_rate_mean:.4f} outside [0.20, 0.60]")

        if missing_metrics:
            gates.append({
                "gate_name": "phase_b_scientific_thresholds",
                "status": "red",
                "observed": f"missing metrics: {', '.join(missing_metrics)}",
                "threshold": "all v15 scientific metrics required",
                "delta_vs_baseline": "n/a",
                "notes": "cannot evaluate scientific pass criteria",
            })
            needs_input.append(
                "Provide complete v15_spikingbrain.json with health, mutual_information, cka metrics."
            )
        elif threshold_failures:
            gates.append({
                "gate_name": "phase_b_scientific_thresholds",
                "status": "red",
                "observed": "; ".join(threshold_failures),
                "threshold": (
                    "dead<0.05, sat<0.10, mi>0.10, cka>0.30, firing_rate in [0.20,0.60]"
                ),
                "delta_vs_baseline": "n/a",
                "notes": "phase B scientific criteria not met",
            })
            needs_input.append(
                "Phase B failed scientific thresholds. Provide a new run after model/config changes."
            )
        else:
            gates.append({
                "gate_name": "phase_b_scientific_thresholds",
                "status": "green",
                "observed": (
                    f"dead={dead_neuron_pct:.4f}, sat={saturated_neuron_pct:.4f}, "
                    f"mi={mutual_information:.4f}, cka={cka_mean:.4f}, firing={firing_rate_mean:.4f}"
                ),
                "threshold": (
                    "dead<0.05, sat<0.10, mi>0.10, cka>0.30, firing_rate in [0.20,0.60]"
                ),
                "delta_vs_baseline": "n/a",
                "notes": "pass",
            })

    return gates, needs_input


def decide_autopilot(gates: Iterable[Dict[str, str]], force_mode: str) -> str:
    if force_mode == "continue":
        return "CONTINUE"
    if force_mode == "pause":
        return "PAUSE_NEEDS_INPUT"
    for gate in gates:
        if gate.get("status") == "red":
            return "PAUSE_NEEDS_INPUT"
    return "CONTINUE"


def _markdown_report(
    run_id: str,
    timestamp_utc: str,
    phase: str,
    phase_name: str,
    summary: str,
    decision: str,
    gates: List[Dict[str, str]],
    artifact_paths: List[str],
    needs_input: List[str],
    next_action: str,
) -> str:
    gate_lines = []
    for g in gates:
        gate_lines.append(
            f"| `{g['gate_name']}` | {g['observed']} | {g['threshold']} | {g['status']} | {g['notes']} |"
        )
    if not gate_lines:
        gate_lines.append("| `none` | n/a | n/a | n/a | n/a |")

    input_lines = "\n".join([f"- {x}" for x in needs_input]) if needs_input else "- none"
    artifact_lines = "\n".join([f"- `{p}`" for p in artifact_paths]) if artifact_paths else "- none"

    return f"""# Gerhard Run Report

## Tier 1: Executive Summary
- Run ID: `{run_id}`
- Timestamp UTC: `{timestamp_utc}`
- Phase: `{phase_name}` ({phase})
- What changed: {summary}
- Gate outcome: `{decision}`
- Next autonomous action: {next_action}
- User input required: {"yes" if needs_input else "no"}

## Tier 2: Technical Evidence

### Gate Scorecard
| Gate | Observed | Threshold | Status | Notes |
|------|----------|-----------|--------|-------|
{chr(10).join(gate_lines)}

### Artifacts Archived
{artifact_lines}

### Needs Input (if any)
{input_lines}

AUTOPILOT_DECISION: {decision}
"""


def _json_report(
    run_id: str,
    timestamp_utc: str,
    phase: str,
    phase_name: str,
    summary: str,
    decision: str,
    gates: List[Dict[str, str]],
    artifact_paths: List[str],
    needs_input: List[str],
    next_action: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "commit": "unknown",
        "phase": phase,
        "phase_name": phase_name,
        "executive_summary": {
            "what_changed": [summary],
            "overall_gate_outcome": decision,
            "risks": [],
            "next_autonomous_action": next_action,
            "user_input_required": "yes" if needs_input else "no",
        },
        "metric_deltas": [],
        "gate_scorecard": gates,
        "artifacts": artifact_paths,
        "decision": decision,
        "next_steps": [next_action],
        "needs_user_input": needs_input,
    }


def update_reports_index(
    index_path: Path,
    run_id: str,
    timestamp_utc: str,
    phase_name: str,
    decision: str,
    report_md_rel: str,
    reason: str,
) -> None:
    existing_rows: List[str] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("| ") and not line.startswith("| Run ID"):
                existing_rows.append(line)

    new_row = f"| {run_id} | {timestamp_utc} | {phase_name} | {decision} | {report_md_rel} |"
    rows = [new_row] + [r for r in existing_rows if not r.startswith(f"| {run_id} |")]

    content = f"""# Gerhard Run Reports Index

## Latest
- Run ID: `{run_id}`
- Timestamp: `{timestamp_utc}`
- Phase: `{phase_name}`
- Decision: `{decision}`
- Reason: {reason}
- Markdown report: `{report_md_rel}`

## History
| Run ID | Timestamp UTC | Phase | Decision | Report |
|-------|----------------|-------|----------|--------|
{chr(10).join(rows)}
"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(content, encoding="utf-8")


def update_program_status(
    program_status_path: Path,
    run_id: str,
    phase: str,
    decision: str,
    report_md_rel: str,
    next_action: str,
    needs_input: List[str],
) -> None:
    data: Dict[str, Any] = {}
    if program_status_path.exists():
        loaded = yaml.safe_load(program_status_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    phase_status = data.get("phase_status")
    if not isinstance(phase_status, dict):
        phase_status = {}
    for key in PHASE_KEYS:
        phase_status.setdefault(key, "not_started")
    phase_status[phase] = "blocked" if decision == "PAUSE_NEEDS_INPUT" else "in_progress"

    current_best_metrics = data.get("current_best_metrics")
    if not isinstance(current_best_metrics, dict):
        current_best_metrics = {}

    data["as_of_date"] = _today_iso()
    data["current_phase"] = PHASE_TO_NAME.get(phase, phase)
    data["phase_status"] = phase_status
    data["current_best_metrics"] = current_best_metrics
    data["last_run_id"] = run_id
    data["last_report_path"] = report_md_rel
    data["next_autonomous_action"] = next_action
    if decision == "PAUSE_NEEDS_INPUT":
        data["blocking_reason"] = "; ".join(needs_input) if needs_input else "Red gate triggered."
    else:
        data["blocking_reason"] = ""

    program_status_path.parent.mkdir(parents=True, exist_ok=True)
    program_status_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def write_gate_results(gate_results_path: Path, run_id: str, gates: List[Dict[str, str]]) -> None:
    gate_results_path.parent.mkdir(parents=True, exist_ok=True)
    gate_results_path.write_text(
        yaml.safe_dump({"run_id": run_id, "gates": gates}, sort_keys=False),
        encoding="utf-8",
    )


def update_status_board(
    status_board_path: Path,
    run_id: str,
    timestamp_utc: str,
    phase_name: str,
    decision: str,
    gates: List[Dict[str, str]],
    next_action: str,
) -> None:
    if not status_board_path.exists():
        return

    text = status_board_path.read_text(encoding="utf-8")
    text = re.sub(r"\*\*As Of\*\*: .*", f"**As Of**: {_today_iso()}  ", text)
    text = re.sub(
        r"- Current active phase: \*\*.*\*\*",
        f"- Current active phase: **{phase_name}**",
        text,
    )

    red_count = sum(1 for g in gates if g.get("status") == "red")
    yellow_count = sum(1 for g in gates if g.get("status") == "yellow")

    latest_block = (
        "<!-- AUTOGEN_LATEST_RUN_START -->\n"
        "## Latest Run Update\n"
        f"- Run ID: `{run_id}`\n"
        f"- Timestamp UTC: `{timestamp_utc}`\n"
        f"- Phase: `{phase_name}`\n"
        f"- Decision: `{decision}`\n"
        f"- Red gates: `{red_count}`\n"
        f"- Yellow gates: `{yellow_count}`\n"
        f"- Next action: {next_action}\n"
        "<!-- AUTOGEN_LATEST_RUN_END -->"
    )

    start = "<!-- AUTOGEN_LATEST_RUN_START -->"
    end = "<!-- AUTOGEN_LATEST_RUN_END -->"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = f"{prefix}{latest_block}{suffix}"
    else:
        text = text.rstrip() + "\n\n" + latest_block + "\n"

    status_board_path.write_text(text, encoding="utf-8")


def register_run(
    run_id: str,
    phase: str,
    source_dir: Path,
    repo_root: Path,
    decision_mode: str = "auto",
    summary: str = "Notebook run outputs archived and evaluated.",
    next_action: str = "Proceed to next queued autonomous task if gates are green.",
) -> Dict[str, str]:
    """
    Register notebook outputs and update reports/state files.

    Returns:
        Dict containing report paths and final decision.
    """
    normalized_phase = phase.strip().upper()
    if normalized_phase not in PHASE_TO_NAME:
        raise ValueError(
            f"Unsupported phase '{phase}'. Expected one of: {', '.join(PHASE_TO_NAME)}"
        )
    if decision_mode not in {"auto", "continue", "pause"}:
        raise ValueError("decision_mode must be one of: auto, continue, pause")

    resolved_repo = repo_root.resolve()
    resolved_source = source_dir.resolve()
    if not resolved_source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {resolved_source}")

    output_dir = resolved_repo / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if resolved_source != output_dir.resolve():
        _copy_tree_merge(resolved_source, output_dir)

    gates, needs_input = evaluate_gates(output_dir, phase=normalized_phase)
    decision = decide_autopilot(gates, force_mode=decision_mode)
    effective_next_action = next_action
    if decision == "PAUSE_NEEDS_INPUT":
        effective_next_action = "Address red gates and provide the requested outputs in the needs_input report."
    timestamp_utc = _utc_now_iso()
    phase_name = PHASE_TO_NAME[normalized_phase]
    artifact_paths = _collect_artifact_paths(output_dir, repo_root=resolved_repo)

    date = dt.datetime.now(dt.timezone.utc)
    report_dir = resolved_repo / "reports" / f"{date.year:04d}" / f"{date.month:02d}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = report_dir / f"{run_id}.md"
    report_json_path = report_dir / f"{run_id}.json"
    report_md_rel = report_md_path.relative_to(resolved_repo).as_posix()
    report_json_rel = report_json_path.relative_to(resolved_repo).as_posix()

    report_md_text = _markdown_report(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        phase=normalized_phase,
        phase_name=phase_name,
        summary=summary,
        decision=decision,
        gates=gates,
        artifact_paths=artifact_paths,
        needs_input=needs_input,
        next_action=effective_next_action,
    )
    report_md_path.write_text(report_md_text, encoding="utf-8")

    report_json_obj = _json_report(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        phase=normalized_phase,
        phase_name=phase_name,
        summary=summary,
        decision=decision,
        gates=gates,
        artifact_paths=artifact_paths,
        needs_input=needs_input,
        next_action=effective_next_action,
    )
    report_json_path.write_text(json.dumps(report_json_obj, indent=2), encoding="utf-8")

    index_path = resolved_repo / "reports" / "index.md"
    reason = (
        "one or more red gates"
        if decision == "PAUSE_NEEDS_INPUT"
        else "all required gates green"
    )
    update_reports_index(
        index_path=index_path,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        phase_name=phase_name,
        decision=decision,
        report_md_rel=report_md_rel,
        reason=reason,
    )
    update_program_status(
        program_status_path=resolved_repo / "state" / "program_status.yaml",
        run_id=run_id,
        phase=normalized_phase,
        decision=decision,
        report_md_rel=report_md_rel,
        next_action=effective_next_action,
        needs_input=needs_input,
    )
    write_gate_results(
        gate_results_path=resolved_repo / "state" / "gate_results.yaml",
        run_id=run_id,
        gates=gates,
    )
    update_status_board(
        status_board_path=resolved_repo / "docs" / "ops" / "STATUS_BOARD.md",
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        phase_name=phase_name,
        decision=decision,
        gates=gates,
        next_action=effective_next_action,
    )

    if decision == "PAUSE_NEEDS_INPUT":
        needs_input_path = report_dir / f"{run_id}_needs_input.md"
        needs_input_body = "\n".join(f"- {item}" for item in needs_input) or "- no explicit items captured"
        needs_input_path.write_text(
            f"# Needs Input\n\nRun `{run_id}` paused.\n\n## Required inputs\n{needs_input_body}\n",
            encoding="utf-8",
        )

    return {
        "report_md": report_md_rel,
        "report_json": report_json_rel,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Register notebook outputs as an autonomous run report.")
    parser.add_argument("--run-id", required=True, help="Unique run id, e.g. 20260223T190000Z_v15_a")
    parser.add_argument("--phase", required=True, help="Phase letter A..K")
    parser.add_argument("--source-dir", required=True, help="Notebook output folder to archive.")
    parser.add_argument(
        "--decision-mode",
        default="auto",
        choices=["auto", "continue", "pause"],
        help="Autopilot decision mode.",
    )
    parser.add_argument(
        "--summary",
        default="Notebook run outputs archived and evaluated.",
        help="Short executive summary line.",
    )
    parser.add_argument(
        "--next-action",
        default="Proceed to next queued autonomous task if gates are green.",
        help="Next autonomous action text.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root path.",
    )
    args = parser.parse_args()

    try:
        result = register_run(
            run_id=args.run_id,
            phase=args.phase,
            source_dir=Path(args.source_dir),
            repo_root=Path(args.repo_root),
            decision_mode=args.decision_mode,
            summary=args.summary,
            next_action=args.next_action,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Report (md): {result['report_md']}")
    print(f"Report (json): {result['report_json']}")
    print(f"Decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
