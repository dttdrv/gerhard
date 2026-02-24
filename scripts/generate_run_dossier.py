"""
Generate a single-file HTML dossier for a run artifact folder.

Usage:
  python scripts/generate_run_dossier.py --run-dir outputs/<run_id>
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
from datetime import datetime, timezone
from html import escape as html_escape
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _series(history: Any, value_key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    if not isinstance(history, list):
        return xs, ys
    for row in history:
        if not isinstance(row, dict):
            continue
        if "step" not in row or value_key not in row:
            continue
        try:
            xv = int(row["step"])
            yv = float(row[value_key])
        except Exception:
            continue
        if math.isfinite(yv):
            xs.append(xv)
            ys.append(yv)
    return xs, ys


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "last": float(values[-1]),
    }


def _moving_avg(values: List[float], window: int = 100) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _flatten(prefix: str, value: Any, rows: List[Tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, rows)
        return
    if isinstance(value, list):
        rows.append((prefix, f"list[{len(value)}]"))
        return
    rows.append((prefix, str(value)))


def generate_dossier(run_dir: Path, run_id: str | None = None, phase: str = "B") -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir does not exist: {run_dir}")

    metrics = _load_json(run_dir / "metrics.json", {})
    eval_suite = _load_json(run_dir / "eval_suite.json", {})
    v15_data = _load_json(run_dir / "v15_spikingbrain.json", {})
    results = _load_json(run_dir / "results.json", {})

    resolved_run_id = run_id or metrics.get("run_id") or run_dir.name
    single_file_path = run_dir / f"run_dossier_{resolved_run_id}.html"
    figures_dir = run_dir / "figures_detailed"
    figures_dir.mkdir(parents=True, exist_ok=True)

    detailed_figures: Dict[str, Dict[str, str]] = {}
    detailed_metrics: Dict[str, Dict[str, float]] = {}

    matplotlib_available = False
    plt = None
    np = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np

        plt = _plt
        np = _np
        matplotlib_available = True
    except Exception:
        matplotlib_available = False

    if matplotlib_available and plt is not None and np is not None:

        def _save_fig(fig: Any, name: str) -> Tuple[str, str]:
            file_path = figures_dir / f"{name}.png"
            fig.savefig(file_path, dpi=180, bbox_inches="tight")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return file_path.as_posix(), b64

        def _plot_lines(
            name: str,
            title: str,
            xlabel: str,
            ylabel: str,
            lines: List[Dict[str, Any]],
            logy: bool = False,
        ) -> None:
            fig, ax = plt.subplots(figsize=(12, 5))
            any_line = False
            for line in lines:
                label = line.get("label")
                xs = line.get("x", [])
                ys = line.get("y", [])
                color = line.get("color")
                if xs and ys:
                    ax.plot(xs, ys, label=label, linewidth=1.5, color=color)
                    any_line = True
            if not any_line:
                plt.close(fig)
                return
            if logy:
                ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            path, b64 = _save_fig(fig, name)
            detailed_figures[name] = {"title": title, "path": path, "base64": b64}
            plt.close(fig)

        tc = results.get("training_curves", {}) if isinstance(results, dict) else {}

        loss_x, loss_y = _series(tc.get("loss_history", []), "loss")
        kl_x, kl_y = _series(tc.get("kl_loss_history", []), "loss")
        align_x, align_y = _series(tc.get("align_loss_history", []), "loss")
        ce_x, ce_y = _series(tc.get("ce_loss_history", []), "loss")
        fdd_x, fdd_y = _series(tc.get("fdd_loss_history", []), "loss")
        ppl_x, ppl_y = _series(tc.get("ppl_history", []), "ppl")
        lr_x, lr_y = _series(tc.get("lr_history", []), "lr")
        temp_x, temp_y = _series(tc.get("temp_history", []), "temperature")
        lam_x, lam_y = _series(tc.get("temp_history", []), "lambda")
        if not lam_y:
            lam_x, lam_y = _series(tc.get("lambda_history", []), "lambda")

        detailed_metrics["loss"] = _stats(loss_y)
        detailed_metrics["kl_loss"] = _stats(kl_y)
        detailed_metrics["align_loss"] = _stats(align_y)
        detailed_metrics["ce_loss"] = _stats(ce_y)
        detailed_metrics["fdd_loss"] = _stats(fdd_y)
        detailed_metrics["ppl"] = _stats(ppl_y)
        detailed_metrics["lr"] = _stats(lr_y)
        detailed_metrics["temperature"] = _stats(temp_y)
        detailed_metrics["lambda"] = _stats(lam_y)

        _plot_lines(
            name="01_loss_components",
            title="Loss Components Over Steps",
            xlabel="step",
            ylabel="loss",
            lines=[
                {"label": "total_loss", "x": loss_x, "y": loss_y, "color": "black"},
                {"label": "kl_loss", "x": kl_x, "y": kl_y, "color": "tab:blue"},
                {"label": "ce_loss", "x": ce_x, "y": ce_y, "color": "tab:orange"},
                {"label": "fdd_loss", "x": fdd_x, "y": fdd_y, "color": "tab:green"},
                {"label": "align_loss", "x": align_x, "y": align_y, "color": "tab:red"},
            ],
        )
        _plot_lines(
            name="02_loss_components_log",
            title="Loss Components Over Steps (log scale)",
            xlabel="step",
            ylabel="loss",
            lines=[
                {"label": "total_loss", "x": loss_x, "y": loss_y, "color": "black"},
                {"label": "kl_loss", "x": kl_x, "y": kl_y, "color": "tab:blue"},
                {"label": "ce_loss", "x": ce_x, "y": ce_y, "color": "tab:orange"},
                {"label": "fdd_loss", "x": fdd_x, "y": fdd_y, "color": "tab:green"},
                {"label": "align_loss", "x": align_x, "y": align_y, "color": "tab:red"},
            ],
            logy=True,
        )
        _plot_lines(
            name="03_total_loss_smoothed",
            title="Total Loss (raw + moving average)",
            xlabel="step",
            ylabel="loss",
            lines=[
                {"label": "total_loss_raw", "x": loss_x, "y": loss_y, "color": "lightgray"},
                {
                    "label": "total_loss_ma100",
                    "x": loss_x,
                    "y": _moving_avg(loss_y, window=100),
                    "color": "black",
                },
            ],
        )
        _plot_lines(
            name="04_ppl_curve",
            title="Validation PPL Over Eval Steps",
            xlabel="step",
            ylabel="ppl",
            lines=[{"label": "val_ppl", "x": ppl_x, "y": ppl_y, "color": "tab:purple"}],
        )
        _plot_lines(
            name="05_learning_rate",
            title="Learning Rate Schedule",
            xlabel="step",
            ylabel="lr",
            lines=[{"label": "lr", "x": lr_x, "y": lr_y, "color": "tab:green"}],
        )
        _plot_lines(
            name="06_temperature",
            title="CTKD Temperature",
            xlabel="step",
            ylabel="temperature",
            lines=[{"label": "temperature", "x": temp_x, "y": temp_y, "color": "tab:orange"}],
        )
        _plot_lines(
            name="07_lambda",
            title="CTKD Lambda / GRL Strength",
            xlabel="step",
            ylabel="lambda",
            lines=[{"label": "lambda", "x": lam_x, "y": lam_y, "color": "tab:red"}],
        )

        spike_summary = results.get("spike_analysis", {}) if isinstance(results, dict) else {}
        per_layer = spike_summary.get("per_layer", {}) if isinstance(spike_summary, dict) else {}
        if isinstance(per_layer, dict) and per_layer:
            layer_names = sorted(per_layer.keys(), key=lambda x: int(str(x).split("_")[-1]))
            k_density = [float(per_layer[n].get("k_final", 0.0)) for n in layer_names]
            v_density = [float(per_layer[n].get("v_final", 0.0)) for n in layer_names]
            k_amp = [float(per_layer[n].get("k_amp_final", 0.0)) for n in layer_names]
            v_amp = [float(per_layer[n].get("v_amp_final", 0.0)) for n in layer_names]

            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(layer_names))
            ax.bar(x - 0.2, k_density, 0.4, label="k_density")
            ax.bar(x + 0.2, v_density, 0.4, label="v_density")
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names, rotation=30)
            ax.set_title("Per-layer Spike Density")
            ax.set_ylabel("density")
            ax.grid(True, alpha=0.3)
            ax.legend()
            path, b64 = _save_fig(fig, "08_spike_density_per_layer")
            detailed_figures["08_spike_density_per_layer"] = {
                "title": "Per-layer Spike Density",
                "path": path,
                "base64": b64,
            }
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(layer_names))
            ax.bar(x - 0.2, k_amp, 0.4, label="k_amplitude")
            ax.bar(x + 0.2, v_amp, 0.4, label="v_amplitude")
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names, rotation=30)
            ax.set_title("Per-layer Spike Amplitude")
            ax.set_ylabel("amplitude")
            ax.grid(True, alpha=0.3)
            ax.legend()
            path, b64 = _save_fig(fig, "09_spike_amplitude_per_layer")
            detailed_figures["09_spike_amplitude_per_layer"] = {
                "title": "Per-layer Spike Amplitude",
                "path": path,
                "base64": b64,
            }
            plt.close(fig)

        density_history = spike_summary.get("density_history", []) if isinstance(spike_summary, dict) else []
        if isinstance(density_history, list) and density_history:
            dx: List[int] = []
            dy: List[float] = []
            for row in density_history:
                if not isinstance(row, dict):
                    continue
                if "step" not in row or "density" not in row:
                    continue
                try:
                    xv = int(row["step"])
                    yv = float(row["density"])
                except Exception:
                    continue
                if math.isfinite(yv):
                    dx.append(xv)
                    dy.append(yv)
            _plot_lines(
                name="10_overall_spike_density_timeline",
                title="Overall Spike Density Timeline",
                xlabel="step",
                ylabel="density",
                lines=[{"label": "overall_density", "x": dx, "y": dy, "color": "tab:blue"}],
            )

        ttt = results.get("ttt", {}) if isinstance(results, dict) else {}
        ttt_x, ttt_y = _series(ttt.get("loss_history", []), "loss")
        _plot_lines(
            name="11_ttt_loss",
            title="TTT LoRA Loss",
            xlabel="step",
            ylabel="loss",
            lines=[{"label": "ttt_loss", "x": ttt_x, "y": ttt_y, "color": "tab:brown"}],
        )

        validation_tests = results.get("validation_tests", {}) if isinstance(results, dict) else {}
        test_rows = validation_tests.get("tests", []) if isinstance(validation_tests, dict) else []
        if isinstance(test_rows, list) and test_rows:
            names = [str(t[0]) for t in test_rows if isinstance(t, list) and len(t) >= 2]
            vals = [1 if bool(t[1]) else 0 for t in test_rows if isinstance(t, list) and len(t) >= 2]
            if names and vals and len(names) == len(vals):
                fig, ax = plt.subplots(figsize=(14, 6))
                x = np.arange(len(names))
                colors = ["tab:green" if v == 1 else "tab:red" for v in vals]
                ax.bar(x, vals, color=colors)
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=45, ha="right")
                ax.set_ylim(0, 1.2)
                ax.set_title("Validation Test Outcomes")
                ax.set_ylabel("pass (1) / fail (0)")
                ax.grid(True, alpha=0.3, axis="y")
                path, b64 = _save_fig(fig, "12_validation_tests")
                detailed_figures["12_validation_tests"] = {
                    "title": "Validation Test Outcomes",
                    "path": path,
                    "base64": b64,
                }
                plt.close(fig)

        hw = results.get("hardware_stats", {}) if isinstance(results, dict) else {}
        hw_labels: List[str] = []
        hw_vals: List[float] = []
        for k in [
            "peak_gpu_memory_gb",
            "avg_gpu_memory_gb",
            "throughput_tokens_per_sec",
            "total_training_time_min",
        ]:
            if k in hw:
                try:
                    hw_labels.append(k)
                    hw_vals.append(float(hw[k]))
                except Exception:
                    continue
        if hw_labels and hw_vals:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(hw_labels))
            ax.bar(x, hw_vals, color="tab:cyan")
            ax.set_xticks(x)
            ax.set_xticklabels(hw_labels, rotation=20, ha="right")
            ax.set_title("Hardware / Runtime Summary")
            ax.grid(True, alpha=0.3, axis="y")
            path, b64 = _save_fig(fig, "13_hardware_summary")
            detailed_figures["13_hardware_summary"] = {
                "title": "Hardware / Runtime Summary",
                "path": path,
                "base64": b64,
            }
            plt.close(fig)

    legacy_plot = {}
    if isinstance(results, dict):
        figs = results.get("figures", {})
        if isinstance(figs, dict):
            legacy_plot = figs.get("training_plot", {})
    if isinstance(legacy_plot, dict) and isinstance(legacy_plot.get("base64"), str) and legacy_plot["base64"]:
        detailed_figures["00_legacy_training_plot"] = {
            "title": "Legacy Training Plot",
            "path": str(legacy_plot.get("filename", "legacy_training_plot.png")),
            "base64": legacy_plot["base64"],
        }

    consolidated_payload = {
        "schema_version": "1.0",
        "run_id": resolved_run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "phase": phase,
        "summary_metrics": metrics,
        "eval_suite": eval_suite,
        "phase_artifact": v15_data,
        "curve_stats": detailed_metrics,
        "detailed_figures_index": {
            k: {"title": v.get("title"), "path": v.get("path")} for k, v in detailed_figures.items()
        },
        "matplotlib_available": matplotlib_available,
    }

    metrics_rows: List[Tuple[str, str]] = []
    _flatten("", metrics if isinstance(metrics, dict) else {}, metrics_rows)
    metrics_table_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{html_escape(v)}</td></tr>" for k, v in metrics_rows
    )

    eval_rows: List[Tuple[str, str]] = []
    _flatten("", eval_suite if isinstance(eval_suite, dict) else {}, eval_rows)
    eval_table_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{html_escape(v)}</td></tr>" for k, v in eval_rows
    )

    curve_rows = ""
    for name, stats in detailed_metrics.items():
        if not stats:
            continue
        curve_rows += (
            f"<tr><td>{html_escape(name)}</td>"
            f"<td>{stats.get('count')}</td>"
            f"<td>{stats.get('min')}</td>"
            f"<td>{stats.get('max')}</td>"
            f"<td>{stats.get('mean')}</td>"
            f"<td>{stats.get('std')}</td>"
            f"<td>{stats.get('last')}</td></tr>"
        )

    fig_blocks = ""
    for name, meta in sorted(detailed_figures.items()):
        b64 = meta.get("base64", "")
        if not b64:
            continue
        title = meta.get("title", name)
        fig_blocks += (
            f"<h3>{html_escape(title)}</h3>"
            f"<p><code>{html_escape(name)}</code></p>"
            f"<img src='data:image/png;base64,{b64}' "
            f"style='max-width:100%;border:1px solid #ddd;padding:6px;background:#fff;'/>"
        )
    if not fig_blocks:
        fig_blocks = "<p>No figures were embedded (matplotlib unavailable or figure data missing).</p>"

    html_report = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Gerhard Run Dossier - {html_escape(resolved_run_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #111; }}
    h1, h2, h3 {{ margin-top: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f5f5; padding: 2px 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    details {{ margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Gerhard Single-File Dossier</h1>
  <p><b>Run ID:</b> {html_escape(resolved_run_id)}<br/>
     <b>Generated UTC:</b> {html_escape(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))}<br/>
     <b>Run Directory:</b> <code>{html_escape(run_dir.as_posix())}</code><br/>
     <b>Phase:</b> {html_escape(phase)}</p>

  <h2>Summary Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {metrics_table_rows}
  </table>

  <h2>Eval Suite Snapshot</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    {eval_table_rows}
  </table>

  <h2>Curve Statistics</h2>
  <table>
    <tr>
      <th>Curve</th><th>Count</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th><th>Last</th>
    </tr>
    {curve_rows}
  </table>

  <h2>Detailed Figures</h2>
  {fig_blocks}

  <h2>Raw Data (Embedded)</h2>
  <details>
    <summary>Consolidated payload</summary>
    <pre>{html_escape(json.dumps(consolidated_payload, indent=2, default=str))}</pre>
  </details>
  <details>
    <summary>results.json snapshot</summary>
    <pre>{html_escape(json.dumps(results, indent=2, default=str))}</pre>
  </details>
</body>
</html>
"""

    single_file_path.write_text(html_report, encoding="utf-8")
    return single_file_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one-file HTML dossier for a run output folder.")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/<run_id> directory.")
    parser.add_argument("--run-id", default=None, help="Override run id used in report filename.")
    parser.add_argument("--phase", default="B", help="Phase label to annotate in dossier.")
    args = parser.parse_args()

    out = generate_dossier(run_dir=Path(args.run_dir), run_id=args.run_id, phase=args.phase)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"dossier: {out.as_posix()}")
    print(f"size_mb: {size_mb:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
