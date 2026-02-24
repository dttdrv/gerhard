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

from scripts.register_notebook_run import register_run


def _extract_json_block(dossier_html: str, summary_label: str) -> Dict[str, Any]:
    pattern = rf"<summary>\s*{re.escape(summary_label)}\s*</summary>\s*<pre>(.*?)</pre>"
    match = re.search(pattern, dossier_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON block for summary: {summary_label}")
    payload_escaped = match.group(1).strip()
    payload_json = html.unescape(payload_escaped)
    return json.loads(payload_json)


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

    consolidated = _extract_json_block(html_text, "Consolidated payload")
    results_snapshot = _extract_json_block(html_text, "results.json snapshot")

    run_id = (
        consolidated.get("run_id")
        or consolidated.get("summary_metrics", {}).get("run_id")
        or dossier_abs.stem.replace("run_dossier_", "")
    )
    if not run_id:
        raise ValueError("Unable to determine run_id from dossier.")

    eval_suite = consolidated.get("eval_suite", {})
    metrics = consolidated.get("summary_metrics", {})
    v15_validation = consolidated.get("v15_validation", {})

    if not isinstance(eval_suite, dict) or not eval_suite:
        raise ValueError("Consolidated payload missing eval_suite.")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Consolidated payload missing summary_metrics.")
    if phase.upper() == "B" and (not isinstance(v15_validation, dict) or not v15_validation):
        raise ValueError("Consolidated payload missing v15_validation for phase B.")

    source_dir = repo_root / "outputs" / "incoming" / f"{run_id}_from_dossier"
    source_dir.mkdir(parents=True, exist_ok=True)

    (source_dir / "eval_suite.json").write_text(json.dumps(eval_suite, indent=2), encoding="utf-8")
    (source_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (source_dir / "results.json").write_text(json.dumps(results_snapshot, indent=2), encoding="utf-8")

    if phase.upper() == "B":
        (source_dir / "v15_spikingbrain.json").write_text(
            json.dumps(v15_validation, indent=2),
            encoding="utf-8",
        )

    seed = metrics.get("seed")
    seed_value = str(seed if seed is not None else "-1")
    (source_dir / "seed.txt").write_text(seed_value + "\n", encoding="utf-8")

    cfg_payload = _build_config_payload(run_id, phase.upper(), consolidated, results_snapshot)
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
    print(f"run_id: {run_id}")
    print(f"source_dir: {source_dir.as_posix()}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")
    print(f"decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
