"""
Register a run directly from a single-file HTML dossier.

This reconstructs required run artifacts from the embedded JSON payloads in:
  outputs/<run_id>/run_dossier_<run_id>.html
and then delegates to scripts/register_notebook_run.py.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.register_notebook_run import register_run, validate_run_id


def _decode_embedded_json(payload_escaped: str, label: str) -> Dict[str, Any]:
    payload_json = html.unescape(payload_escaped)
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid embedded JSON for {label}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Embedded JSON for {label} must decode to an object.")
    return data


def _extract_json_block(dossier_html: str, summary_label: str) -> Dict[str, Any]:
    pattern = rf"<summary>\s*{re.escape(summary_label)}\s*</summary>\s*<pre>(.*?)</pre>"
    match = re.search(pattern, dossier_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON block for summary: {summary_label}")
    return _decode_embedded_json(match.group(1).strip(), f"summary: {summary_label}")


def _extract_json_block_if_present(dossier_html: str, summary_label: str) -> Dict[str, Any] | None:
    pattern = rf"<summary>\s*{re.escape(summary_label)}\s*</summary>\s*<pre>(.*?)</pre>"
    match = re.search(pattern, dossier_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _decode_embedded_json(match.group(1).strip(), f"summary: {summary_label}")


def _extract_json_h2_block_if_present(dossier_html: str, heading_label: str) -> Dict[str, Any] | None:
    pattern = rf"<h2>\s*{re.escape(heading_label)}\s*</h2>\s*<pre>(.*?)</pre>"
    match = re.search(pattern, dossier_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _decode_embedded_json(match.group(1).strip(), f"heading: {heading_label}")


def _build_config_payload(
    run_id: str,
    phase: str,
    consolidated: Dict[str, Any],
    results_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    summary_metrics = consolidated.get("summary_metrics", {})
    version = summary_metrics.get("version", "unknown")
    seed = summary_metrics.get("seed", -1)
    training_cfg = results_snapshot.get("training_config", {})
    arch = results_snapshot.get("architecture", {})
    description = results_snapshot.get("description", "dossier-reconstructed run")
    return {
        "run_id": run_id,
        "phase": phase,
        "version": version,
        "version_desc": description,
        "seed": seed,
        "source": "single_file_dossier_reconstruction",
        "config": {
            "training_config": training_cfg,
            "architecture": arch,
        },
    }


def reconstruct_from_dossier(
    dossier_path: Path,
    phase: str,
    repo_root: Path,
) -> Tuple[str, Path]:
    dossier_abs = dossier_path.resolve()
    if not dossier_abs.exists():
        raise FileNotFoundError(f"Dossier file not found: {dossier_abs}")

    html_text = dossier_abs.read_text(encoding="utf-8", errors="ignore")

    consolidated = _extract_json_block_if_present(html_text, "Consolidated payload")
    results_snapshot = _extract_json_block_if_present(html_text, "results.json snapshot") or {}
    notebook_config = _extract_json_h2_block_if_present(html_text, "Config") or {}

    metrics: Dict[str, Any] = {}
    eval_suite: Dict[str, Any] = {}
    phase_artifact: Dict[str, Any] = {}

    if isinstance(consolidated, dict) and consolidated:
        metrics = consolidated.get("summary_metrics", {})
        eval_suite = consolidated.get("eval_suite", {})
        phase_artifact = (
            consolidated.get("v15_validation")
            or consolidated.get("phase_artifact")
            or {}
        )
    else:
        metrics = _extract_json_block_if_present(html_text, "metrics.json") or {}
        eval_suite = _extract_json_block_if_present(html_text, "eval_suite.json") or {}
        if phase.upper() == "B":
            phase_artifact = _extract_json_block_if_present(html_text, "v15_spikingbrain.json") or {}

    run_id = (
        (consolidated or {}).get("run_id")
        or metrics.get("run_id")
        or eval_suite.get("run_id")
        or notebook_config.get("run_id")
        or dossier_abs.stem.replace("run_dossier_", "")
    )
    if not run_id:
        raise ValueError("Unable to determine run_id from dossier.")
    run_id = validate_run_id(run_id)

    if not isinstance(eval_suite, dict) or not eval_suite:
        raise ValueError("Consolidated payload missing eval_suite.")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Consolidated payload missing summary_metrics.")
    if phase.upper() == "B" and (not isinstance(phase_artifact, dict) or not phase_artifact):
        raise ValueError(
            "Phase B dossier missing phase artifact. Accepted sources: "
            "`v15_validation`, `phase_artifact`, or `v15_spikingbrain.json`."
        )

    incoming_root = (repo_root / "outputs" / "incoming").resolve()
    incoming_root.mkdir(parents=True, exist_ok=True)
    source_dir = (incoming_root / f"{run_id}_from_dossier").resolve()
    try:
        source_dir.relative_to(incoming_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe staging path for run_id {run_id!r}.") from exc
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    (source_dir / "eval_suite.json").write_text(json.dumps(eval_suite, indent=2), encoding="utf-8")
    (source_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (source_dir / "results.json").write_text(json.dumps(results_snapshot, indent=2), encoding="utf-8")

    if phase.upper() == "B":
        (source_dir / "v15_spikingbrain.json").write_text(
            json.dumps(phase_artifact, indent=2),
            encoding="utf-8",
        )

    seed = metrics.get("seed")
    seed_value = str(seed if seed is not None else "-1")
    (source_dir / "seed.txt").write_text(seed_value + "\n", encoding="utf-8")

    if notebook_config:
        cfg_payload = notebook_config
    else:
        cfg_payload = _build_config_payload(
            run_id,
            phase.upper(),
            consolidated or {"summary_metrics": metrics},
            results_snapshot,
        )
    (source_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg_payload, sort_keys=False),
        encoding="utf-8",
    )

    shutil.copy2(dossier_abs, source_dir / dossier_abs.name)

    return run_id, source_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a run from a single-file run dossier.")
    parser.add_argument("--dossier", required=True, help="Path to run_dossier_<run_id>.html")
    parser.add_argument("--phase", default="B", help="Phase letter A..K")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    try:
        run_id, source_dir = reconstruct_from_dossier(
            dossier_path=Path(args.dossier),
            phase=args.phase,
            repo_root=repo_root,
        )

        result = register_run(
            run_id=run_id,
            phase=args.phase,
            source_dir=source_dir,
            repo_root=repo_root,
            summary="Single-file dossier ingested and reconstructed into canonical artifacts.",
            next_action="Proceed according to gate decision (continue on green, pause on red).",
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"run_id: {run_id}")
    print(f"source_dir: {source_dir.as_posix()}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")
    print(f"decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
