# =============================================================================
# cell 26: FINAL save + autonomous v15 artifact bundle + single-file dossier
# =============================================================================
print("="*60)
print("FINAL SAVE + AUTONOMY ARTIFACTS (v15)")
print("="*60)

# Add validation_tests to results
results['validation_tests'] = validation_results

# Save final legacy results json (kept for backward compatibility)
results_path = f'{OUTPUT_DIR}/results/results_{RUN_TIMESTAMP}.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"saved legacy results: {results_path}")
print(f"size: {os.path.getsize(results_path) / 1024:.1f} KB")

# -----------------------------------------------------------------------------
# Build canonical per-run artifact pack for autonomous ingestion
# -----------------------------------------------------------------------------
run_id = f"{config.VERSION}_{RUN_TIMESTAMP}".replace(" ", "_").replace(":", "-")
run_artifact_dir = f"{OUTPUT_DIR}/{run_id}"
os.makedirs(run_artifact_dir, exist_ok=True)

import hashlib

def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _build_run_fingerprint() -> Dict[str, Any]:
    cfg_dict = asdict(config)
    cfg_json = json.dumps(cfg_dict, sort_keys=True, default=str).encode('utf-8')
    config_sha256 = hashlib.sha256(cfg_json).hexdigest()

    notebook_candidates = [
        os.environ.get('GERHARD_NOTEBOOK_PATH', ''),
        f"{OUTPUT_DIR}/asnn_goose_colab_v15.ipynb",
        'asnn_goose_colab_v15.ipynb',
        'notebooks/asnn_goose_colab_v15.ipynb',
        '/workspace/asnn_goose_colab_v15.ipynb',
        '/workspace/gerhard/notebooks/asnn_goose_colab_v15.ipynb',
    ]
    notebook_path = None
    notebook_sha256 = None
    for c in notebook_candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists() and p.is_file():
            notebook_path = str(p.resolve())
            notebook_sha256 = _sha256_file(p)
            if notebook_sha256:
                break

    recipe_basis = {
        'version': config.VERSION,
        'version_desc': config.VERSION_DESC,
        'seed': int(SEED),
        'distill_steps': int(config.distill_steps),
        'batch_size': int(config.batch_size),
        'accumulation_steps': int(config.accumulation_steps),
        'distill_lr': float(config.distill_lr),
        'fdd_weight': float(config.fdd_weight),
        'fdd_warmup_steps': int(config.fdd_warmup_steps),
        'use_spike_semantic_loss': bool(config.use_spike_semantic_loss),
        'spike_semantic_weight': float(config.spike_semantic_weight),
        'spike_target_threshold_scale': float(config.spike_target_threshold_scale),
    }
    recipe_sha256 = hashlib.sha256(
        json.dumps(recipe_basis, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()

    return {
        'config_sha256': config_sha256,
        'recipe_sha256': recipe_sha256,
        'notebook_path': notebook_path,
        'notebook_sha256': notebook_sha256,
    }


run_fingerprint = _build_run_fingerprint()

best_ppl = None
best_step = None
if distill_logs.get('ppl_history'):
    best_entry = min(distill_logs['ppl_history'], key=lambda x: x['ppl'])
    best_ppl = float(best_entry['ppl'])
    best_step = int(best_entry['step'])

metrics_payload = {
    "run_id": run_id,
    "phase": "B",
    "version": config.VERSION,
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "fingerprint": run_fingerprint,
    "seed": int(SEED),
    "teacher_ppl": float(teacher_ppl),
    "student_ppl": float(student_ppl),
    "best_student_ppl": best_ppl,
    "best_step": best_step,
    "ppl_gap": float(student_ppl - teacher_ppl),
    "spike_density": float(student.get_avg_spike_density()),
    "fingerprint": run_fingerprint,
    "v15_overall_pass": bool(v15_results.overall_pass),
    "v15_firing_rate_mean": float(v15_results.health.firing_rate_mean),
    "v15_mutual_information": float(v15_results.mutual_information.get('mutual_information', 0.0)),
    "v15_cka_mean": float(v15_results.cka.get('cka_mean', 0.0)),
}

eval_suite_payload = {
    "schema_version": "1.0",
    "run_id": run_id,
    "phase": "B",
    "version": config.VERSION,
    "git_hash": os.environ.get("GIT_COMMIT", "unknown"),
    "seed": int(SEED),
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "tasks": {
        "lm_ppl": {
            "metric": "ppl",
            "value": float(student_ppl),
            "baseline_reference": 306.89,
        },
        "v15_spike_health": {
            "dead_neuron_pct": float(v15_results.health.dead_neuron_pct),
            "saturated_neuron_pct": float(v15_results.health.saturated_neuron_pct),
            "firing_rate_mean": float(v15_results.health.firing_rate_mean),
            "pass": bool(v15_results.health.health_pass),
        },
        "v15_information": {
            "mutual_information": float(v15_results.mutual_information.get('mutual_information', 0.0)),
            "cka_mean": float(v15_results.cka.get('cka_mean', 0.0)),
            "overall_pass": bool(v15_results.overall_pass),
        },
    },
    "gate_recommendation": "green" if v15_results.overall_pass else "red",
}

v15_payload = {
    "run_id": run_id,
    "phase": "B",
    "version": config.VERSION,
    "validation": v15_results.to_dict(),
    "success_criteria_tests": {name: {"passed": passed, "value": value} for name, passed, value in tests},
    "overall_pass": bool(v15_results.overall_pass),
}

config_payload = {
    "run_id": run_id,
    "phase": "B",
    "version": config.VERSION,
    "version_desc": config.VERSION_DESC,
    "seed": int(SEED),
    "platform": PLATFORM,
    "device": str(DEVICE),
    "output_dir": OUTPUT_DIR,
    "run_timestamp": RUN_TIMESTAMP,
    "fingerprint": run_fingerprint,
    "config": asdict(config),
}

with open(f"{run_artifact_dir}/metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=2, default=str)

with open(f"{run_artifact_dir}/eval_suite.json", "w") as f:
    json.dump(eval_suite_payload, f, indent=2, default=str)

with open(f"{run_artifact_dir}/v15_spikingbrain.json", "w") as f:
    json.dump(v15_payload, f, indent=2, default=str)

with open(f"{run_artifact_dir}/seed.txt", "w") as f:
    f.write(str(SEED) + "\n")

config_yaml_path = f"{run_artifact_dir}/config.yaml"
try:
    import yaml
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)
except Exception as e:
    # Fallback: keep required artifact name, store JSON-formatted content.
    with open(config_yaml_path, "w") as f:
        f.write(json.dumps(config_payload, indent=2, default=str))
    print(f"warning: yaml export fallback used ({e})")

# Copy key legacy outputs into artifact bundle
try:
    import shutil
    if os.path.exists(results_path):
        shutil.copy2(results_path, f"{run_artifact_dir}/results.json")
except Exception as e:
    print(f"warning: could not copy legacy outputs: {e}")

print("")
print("Canonical artifacts written:")
print(f"  {run_artifact_dir}/eval_suite.json")
print(f"  {run_artifact_dir}/metrics.json")
print(f"  {run_artifact_dir}/config.yaml")
print(f"  {run_artifact_dir}/seed.txt")
print(f"  {run_artifact_dir}/v15_spikingbrain.json")

# -----------------------------------------------------------------------------
# Build a single-file detailed dossier (HTML with embedded figures and raw data)
# -----------------------------------------------------------------------------
import base64
import math
import statistics
from io import BytesIO
from html import escape as html_escape

single_file_path = f"{run_artifact_dir}/run_dossier_{run_id}.html"
single_file_primary_output = single_file_path

def _series(history, value_key):
    xs, ys = [], []
    for row in history or []:
        if not isinstance(row, dict):
            continue
        if 'step' not in row or value_key not in row:
            continue
        try:
            y = float(row[value_key])
            x = int(row['step'])
        except Exception:
            continue
        if math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys

def _stats(values):
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

def _moving_avg(values, window=100):
    if not values:
        return []
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i+1]
        out.append(sum(chunk) / len(chunk))
    return out

detailed_figures = {}
detailed_metrics = {}

if MATPLOTLIB_AVAILABLE:
    def _save_fig(fig, name):
        # Single-file mode: keep figures embedded only; do not emit sidecar PNGs.
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return "embedded", b64

    def _plot_lines(name, title, xlabel, ylabel, lines, logy=False):
        fig, ax = plt.subplots(figsize=(12, 5))
        any_line = False
        for line in lines:
            label = line.get('label')
            xs = line.get('x', [])
            ys = line.get('y', [])
            color = line.get('color')
            if xs and ys:
                ax.plot(xs, ys, label=label, linewidth=1.5, color=color)
                any_line = True
        if not any_line:
            plt.close(fig)
            return
        if logy:
            ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        path, b64 = _save_fig(fig, name)
        detailed_figures[name] = {"title": title, "path": path, "base64": b64}
        plt.close(fig)

    tc = results.get('training_curves', {})

    loss_x, loss_y = _series(tc.get('loss_history', []), 'loss')
    kl_x, kl_y = _series(tc.get('kl_loss_history', []), 'loss')
    align_x, align_y = _series(tc.get('align_loss_history', []), 'loss')
    ce_x, ce_y = _series(tc.get('ce_loss_history', []), 'loss')
    fdd_x, fdd_y = _series(tc.get('fdd_loss_history', []), 'loss')
    ppl_x, ppl_y = _series(tc.get('ppl_history', []), 'ppl')
    lr_x, lr_y = _series(tc.get('lr_history', []), 'lr')
    temp_x, temp_y = _series(tc.get('temp_history', []), 'temperature')
    lam_x, lam_y = _series(tc.get('temp_history', []), 'lambda')
    if not lam_y:
        lam_x, lam_y = _series(tc.get('lambda_history', []), 'lambda')

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

    loss_ma = _moving_avg(loss_y, window=100)
    _plot_lines(
        name="03_total_loss_smoothed",
        title="Total Loss (raw + moving average)",
        xlabel="step",
        ylabel="loss",
        lines=[
            {"label": "total_loss_raw", "x": loss_x, "y": loss_y, "color": "lightgray"},
            {"label": "total_loss_ma100", "x": loss_x, "y": loss_ma, "color": "black"},
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

    # Spike summary figures
    spike_summary = results.get('spike_analysis', {})
    per_layer = spike_summary.get('per_layer', {})
    if per_layer:
        layer_names = sorted(per_layer.keys(), key=lambda x: int(x.split('_')[-1]))
        k_density = [float(per_layer[n].get('k_final', 0.0)) for n in layer_names]
        v_density = [float(per_layer[n].get('v_final', 0.0)) for n in layer_names]
        k_amp = [float(per_layer[n].get('k_amp_final', 0.0)) for n in layer_names]
        v_amp = [float(per_layer[n].get('v_amp_final', 0.0)) for n in layer_names]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(layer_names))
        ax.bar(x - 0.2, k_density, 0.4, label='k_density')
        ax.bar(x + 0.2, v_density, 0.4, label='v_density')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=30)
        ax.set_title("Per-layer Spike Density")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path, b64 = _save_fig(fig, "08_spike_density_per_layer")
        detailed_figures["08_spike_density_per_layer"] = {"title": "Per-layer Spike Density", "path": path, "base64": b64}
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(layer_names))
        ax.bar(x - 0.2, k_amp, 0.4, label='k_amplitude')
        ax.bar(x + 0.2, v_amp, 0.4, label='v_amplitude')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=30)
        ax.set_title("Per-layer Spike Amplitude")
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path, b64 = _save_fig(fig, "09_spike_amplitude_per_layer")
        detailed_figures["09_spike_amplitude_per_layer"] = {"title": "Per-layer Spike Amplitude", "path": path, "base64": b64}
        plt.close(fig)

    # Density history timeline
    density_history = spike_summary.get('density_history', [])
    if density_history:
        dx = []
        dy = []
        for row in density_history:
            if isinstance(row, dict) and 'step' in row and 'density' in row:
                try:
                    xv = int(row['step'])
                    yv = float(row['density'])
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

    # TTT loss
    ttt = results.get('ttt', {})
    ttt_x, ttt_y = _series(ttt.get('loss_history', []), 'loss')
    _plot_lines(
        name="11_ttt_loss",
        title="TTT LoRA Loss",
        xlabel="step",
        ylabel="loss",
        lines=[{"label": "ttt_loss", "x": ttt_x, "y": ttt_y, "color": "tab:brown"}],
    )

    # Validation test pass/fail chart
    test_rows = validation_results.get('tests', []) if isinstance(validation_results, dict) else []
    if test_rows:
        names = [str(t[0]) for t in test_rows]
        vals = [1 if bool(t[1]) else 0 for t in test_rows]
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(names))
        colors = ['tab:green' if v == 1 else 'tab:red' for v in vals]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 1.2)
        ax.set_title("Validation Test Outcomes")
        ax.set_ylabel("pass (1) / fail (0)")
        ax.grid(True, alpha=0.3, axis='y')
        path, b64 = _save_fig(fig, "12_validation_tests")
        detailed_figures["12_validation_tests"] = {"title": "Validation Test Outcomes", "path": path, "base64": b64}
        plt.close(fig)

    # Hardware summary chart
    hw = results.get('hardware_stats', {})
    hw_labels = []
    hw_vals = []
    for k in ["peak_gpu_memory_gb", "avg_gpu_memory_gb", "throughput_tokens_per_sec", "total_training_time_min"]:
        if k in hw:
            try:
                hw_labels.append(k)
                hw_vals.append(float(hw[k]))
            except Exception:
                pass
    if hw_labels:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(hw_labels))
        ax.bar(x, hw_vals, color='tab:cyan')
        ax.set_xticks(x)
        ax.set_xticklabels(hw_labels, rotation=20, ha='right')
        ax.set_title("Hardware / Runtime Summary")
        ax.grid(True, alpha=0.3, axis='y')
        path, b64 = _save_fig(fig, "13_hardware_summary")
        detailed_figures["13_hardware_summary"] = {"title": "Hardware / Runtime Summary", "path": path, "base64": b64}
        plt.close(fig)
else:
    print("matplotlib unavailable: detailed figure generation skipped in single-file dossier.")

# Include legacy training plot if present
legacy_plot = results.get('figures', {}).get('training_plot', {})
if isinstance(legacy_plot, dict) and isinstance(legacy_plot.get('base64'), str) and legacy_plot['base64']:
    detailed_figures["00_legacy_training_plot"] = {
        "title": "Legacy Training Plot",
        "path": legacy_plot.get("filename", "legacy_training_plot.png"),
        "base64": legacy_plot["base64"],
    }

consolidated_payload = {
    "schema_version": "1.0",
    "run_id": run_id,
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "phase": "B",
    "fingerprint": run_fingerprint,
    "summary_metrics": metrics_payload,
    "eval_suite": eval_suite_payload,
    "v15_validation": v15_payload,
    "curve_stats": detailed_metrics,
    "detailed_figures_index": {
        k: {"title": v.get("title"), "path": v.get("path")}
        for k, v in detailed_figures.items()
    },
}

def _json_block(obj):
    return html_escape(json.dumps(obj, indent=2, default=str))

summary_rows = "".join(
    f"<tr><td>{html_escape(str(k))}</td><td>{html_escape(str(v))}</td></tr>"
    for k, v in metrics_payload.items()
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
    title = meta.get("title", name)
    fig_blocks += (
        f"<h3>{html_escape(title)}</h3>"
        f"<p><code>{html_escape(name)}</code></p>"
        f"<img src='data:image/png;base64,{b64}' style='max-width:100%;border:1px solid #ddd;padding:6px;background:#fff;'/>"
    )

html_report = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Gerhard V15 Dossier - {html_escape(run_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #111; }}
    h1, h2, h3 {{ margin-top: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f5f5; padding: 2px 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    details {{ margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Gerhard V15 Single-File Dossier</h1>
  <p><b>Run ID:</b> {html_escape(run_id)}<br/>
     <b>Generated:</b> {html_escape(datetime.utcnow().isoformat() + "Z")}<br/>
     <b>Phase:</b> B (v15 SpikingBrain validation)</p>

  <h2>Executive Summary Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {summary_rows}
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
    <pre>{_json_block(consolidated_payload)}</pre>
  </details>
  <details>
    <summary>results.json snapshot</summary>
    <pre>{_json_block(results)}</pre>
  </details>
</body>
</html>
"""

with open(single_file_path, "w", encoding="utf-8") as f:
    f.write(html_report)

single_file_size_mb = os.path.getsize(single_file_path) / (1024 * 1024)
print("")
print(f"Single-file dossier saved: {single_file_path} ({single_file_size_mb:.2f} MB)")
print(f"Primary one-file output: {single_file_primary_output}")

# -----------------------------------------------------------------------------
# Optional automatic registration into repo autonomous state/report system
# -----------------------------------------------------------------------------
registration_result = None
registration_error = None
repo_root = None

candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path('/workspace/gerhard'),
    Path('/kaggle/working/gerhard'),
]

for candidate in candidate_roots:
    if (candidate / 'scripts' / 'register_notebook_run.py').exists():
        repo_root = candidate
        break

if repo_root is not None:
    try:
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        from scripts.register_notebook_run import register_run

        registration_result = register_run(
            run_id=run_id,
            phase='B',
            source_dir=Path(run_artifact_dir),
            repo_root=repo_root,
            summary='v15 notebook pass with canonical artifact bundle and single-file dossier',
            next_action='Proceed according to gate decision (continue on green, pause on red).',
        )
        print("")
        print("Autonomous registration complete:")
        print(registration_result)
    except Exception as e:
        registration_error = str(e)
        print("")
        print(f"Autonomous registration skipped due to error: {registration_error}")
else:
    registration_error = (
        "register_notebook_run.py not found under candidate repo roots. "
        "Run registration manually later with this source dir."
    )
    print("")
    print("Autonomous registration helper not found in this environment.")
    print(f"Manual source dir for later registration: {run_artifact_dir}")

results['autonomy_artifacts'] = {
    'run_id': run_id,
    'artifact_dir': run_artifact_dir,
    'single_file_dossier': single_file_path,
    'single_file_primary_output': single_file_primary_output,
    'required_files': [
        'eval_suite.json',
        'metrics.json',
        'config.yaml',
        'seed.txt',
        'v15_spikingbrain.json',
        f'run_dossier_{run_id}.html',
    ],
    'registration_result': registration_result,
    'registration_error': registration_error,
}

# update legacy results snapshot with autonomy metadata
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

def _attempt_auto_download(path: str):
    if IS_COLAB:
        try:
            from google.colab import files
            files.download(path)
            return True
        except Exception as e:
            print(f"colab download failed: {e}")
            return False
    # Try Jupyter front-end auto-open in non-colab notebook environments
    try:
        from IPython.display import Javascript, display
        abs_path = os.path.abspath(path).replace("\\", "/")
        display(Javascript(f"window.open('/files/{abs_path}', '_blank');"))
        print(f"triggered browser download/open for: /files/{abs_path}")
        return True
    except Exception as e:
        print(f"non-colab auto-download not available: {e}")
        return False

print("")
print("Auto-download single-file dossier")
_attempt_auto_download(single_file_primary_output)
print(f"single-file dossier path: {single_file_primary_output}")
