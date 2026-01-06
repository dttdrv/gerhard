"""
Publication-quality visualization utilities.

All figures saved as PNG (300 DPI) + PDF for paper use.
Designed for Kaggle notebook compatibility.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import os

# Set non-interactive backend for Kaggle
matplotlib.use('Agg')


def setup_plotting_style():
    """
    Configure matplotlib for publication-quality figures.
    Call this at the start of notebooks.
    """
    plt.style.use('seaborn-v0_8-paper')

    # Update rcParams for better defaults
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (8, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
    })


class FigureSaver:
    """
    Context manager for saving figures as PNG + PDF.

    Usage:
        with FigureSaver("outputs/figures/my_plot") as (fig, ax):
            ax.plot(x, y)
            ax.set_xlabel("X")
    """

    def __init__(
        self,
        filepath: str,
        dpi: int = 300,
        figsize: Tuple[float, float] = (8, 6),
    ):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

    def __enter__(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        return self.fig, self.ax

    def __exit__(self, *args):
        plt.tight_layout()

        # Save PNG
        plt.savefig(
            self.filepath.with_suffix('.png'),
            dpi=self.dpi,
            bbox_inches='tight',
            facecolor='white',
        )

        # Save PDF
        plt.savefig(
            self.filepath.with_suffix('.pdf'),
            bbox_inches='tight',
            facecolor='white',
        )

        plt.close()
        print(f"Saved: {self.filepath.with_suffix('.png')} and .pdf")


def save_figure(fig, filepath: str, dpi: int = 300):
    """
    Save a figure as PNG + PDF.

    Args:
        fig: Matplotlib figure
        filepath: Output path (without extension)
        dpi: DPI for PNG
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        filepath.with_suffix('.png'),
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
    )
    fig.savefig(
        filepath.with_suffix('.pdf'),
        bbox_inches='tight',
        facecolor='white',
    )
    print(f"Saved: {filepath.with_suffix('.png')} and .pdf")


def plot_training_curves(
    logs: List[Dict[str, float]],
    output_path: str,
    title: str = "Training Progress",
    show: bool = True,
):
    """
    Plot training loss curves with multiple metrics.

    Args:
        logs: List of log dictionaries with metrics
        output_path: Path to save figure
        title: Plot title
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    steps = list(range(len(logs)))

    # Loss curve
    ax = axes[0, 0]
    if "total_loss" in logs[0] or "loss" in logs[0]:
        losses = [l.get("total_loss", l.get("loss", 0)) for l in logs]
        ax.plot(steps, losses, label="Total Loss", linewidth=2)
    if "kl_loss" in logs[0]:
        kl_losses = [l.get("kl_loss", 0) for l in logs]
        ax.plot(steps, kl_losses, label="KL Loss", linewidth=2, linestyle='--')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[0, 1]
    if "grad_norm" in logs[0]:
        grad_norms = [l.get("grad_norm", 0) for l in logs]
        ax.plot(steps, grad_norms, color='orange', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm")
        ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    if "lr" in logs[0] or "learning_rate" in logs[0]:
        lrs = [l.get("lr", l.get("learning_rate", 0)) for l in logs]
        ax.plot(steps, lrs, color='green', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

    # Spike density (if available)
    ax = axes[1, 1]
    if "spike_density" in logs[0]:
        densities = [l.get("spike_density", 0) for l in logs]
        ax.plot(steps, densities, color='purple', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Spike Density")
        ax.set_title("Spike Density")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    else:
        # Smoothed loss instead
        losses = [l.get("total_loss", l.get("loss", 0)) for l in logs]
        window = min(100, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(len(smoothed)), smoothed, color='red', linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Smoothed Loss")
            ax.set_title(f"Smoothed Loss (window={window})")
            ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_spike_density_over_time(
    densities: List[float],
    output_path: str,
    layer_name: str = "",
    trigger_threshold: Optional[float] = None,
    show: bool = True,
):
    """
    Plot spike density evolution during training/inference.

    Args:
        densities: List of spike density values
        output_path: Path to save figure
        layer_name: Name of layer for title
        trigger_threshold: Optional TTT trigger threshold to show
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = list(range(len(densities)))
    ax.plot(steps, densities, linewidth=2, color='blue', label='Spike Density')

    if trigger_threshold is not None:
        ax.axhline(
            y=trigger_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'TTT Trigger ({trigger_threshold})',
        )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Spike Density", fontsize=12)
    title = "Spike Density Over Time"
    if layer_name:
        title += f" - {layer_name}"
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_firing_map(
    firing_patterns: torch.Tensor,
    output_path: str,
    title: str = "Firing Map",
    max_neurons: int = 256,
    max_timesteps: int = 500,
    show: bool = True,
):
    """
    Heatmap visualization of spike activity across neurons and time.

    Args:
        firing_patterns: Tensor of shape (timesteps, neurons) or list
        output_path: Path to save figure
        title: Plot title
        max_neurons: Maximum neurons to display
        max_timesteps: Maximum timesteps to display
        show: Whether to display plot
    """
    if isinstance(firing_patterns, list):
        firing_patterns = torch.stack(firing_patterns)

    if firing_patterns.dim() == 3:
        # (batch, seq, d_model) -> (seq, d_model)
        firing_patterns = firing_patterns.mean(dim=0)

    # Subsample if too large
    if firing_patterns.shape[0] > max_timesteps:
        indices = np.linspace(0, firing_patterns.shape[0]-1, max_timesteps).astype(int)
        firing_patterns = firing_patterns[indices]

    if firing_patterns.shape[1] > max_neurons:
        indices = np.linspace(0, firing_patterns.shape[1]-1, max_neurons).astype(int)
        firing_patterns = firing_patterns[:, indices]

    fig, ax = plt.subplots(figsize=(12, 8))

    data = firing_patterns.cpu().numpy() if isinstance(firing_patterns, torch.Tensor) else firing_patterns

    im = ax.imshow(
        data.T,
        aspect='auto',
        cmap='RdBu_r',
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Neuron Index", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Firing Probability", fontsize=10)

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_sparsity_structure(
    sparsity_mask: torch.Tensor,
    output_path: str,
    title: str = "Activation Sparsity Pattern",
    max_size: int = 500,
    show: bool = True,
):
    """
    Visualize sparsity structure for kernel analysis.

    Args:
        sparsity_mask: 2D tensor of zeros and ones
        output_path: Path to save figure
        title: Plot title
        max_size: Maximum size per dimension
        show: Whether to display plot
    """
    if sparsity_mask.dim() > 2:
        sparsity_mask = sparsity_mask.view(sparsity_mask.shape[0], -1)

    # Subsample if needed
    if sparsity_mask.shape[0] > max_size:
        indices = np.linspace(0, sparsity_mask.shape[0]-1, max_size).astype(int)
        sparsity_mask = sparsity_mask[indices]
    if sparsity_mask.shape[1] > max_size:
        indices = np.linspace(0, sparsity_mask.shape[1]-1, max_size).astype(int)
        sparsity_mask = sparsity_mask[:, indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.spy(sparsity_mask.cpu().numpy(), markersize=1, aspect='auto')
    ax.set_xlabel("Feature Dimension", fontsize=12)
    ax.set_ylabel("Sequence Position", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_bar(
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Model Comparison",
    show: bool = True,
):
    """
    Bar chart comparing metrics across models.

    Args:
        metrics: {model_name: {metric_name: value}}
        output_path: Path to save figure
        title: Plot title
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))

    for i, model_name in enumerate(model_names):
        values = [metrics[model_name].get(m, 0) for m in metric_names]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=colors[i])

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_ttt_drift_analysis(
    validation_scores: List[float],
    update_points: List[int],
    reversions: List[int],
    output_path: str,
    title: str = "TTT Drift Analysis",
    show: bool = True,
):
    """
    Visualize TTT stability: when updates happen, when reversions occur.

    Args:
        validation_scores: List of validation scores over time
        update_points: Steps where updates were accepted
        reversions: Steps where updates were reverted
        output_path: Path to save figure
        title: Plot title
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = list(range(len(validation_scores)))
    ax.plot(steps, validation_scores, linewidth=2, label='Validation Score', color='blue')

    # Mark update points
    for up in update_points:
        if up < len(validation_scores):
            ax.axvline(x=up, color='green', alpha=0.3, linestyle='--')

    # Mark reversions
    for rev in reversions:
        if rev < len(validation_scores):
            ax.axvline(x=rev, color='red', alpha=0.5, linestyle=':')

    # Add legend entries
    ax.axvline(x=-1, color='green', alpha=0.5, linestyle='--', label='Update Accepted')
    ax.axvline(x=-1, color='red', alpha=0.5, linestyle=':', label='Update Reverted')

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Validation Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_kernel_benchmark(
    results: List[Dict[str, float]],
    output_path: str,
    title: str = "Sparse vs Dense Kernel Performance",
    show: bool = True,
):
    """
    Plot kernel benchmark results comparing sparse and dense performance.

    Args:
        results: List of benchmark result dictionaries
        output_path: Path to save figure
        title: Plot title
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Speedup vs sparsity
    ax = axes[0]
    sparsities = [r["sparsity"] for r in results]
    masked_speedups = [r["masked_speedup"] for r in results]
    coo_speedups = [r["coo_speedup"] for r in results]

    ax.scatter(sparsities, masked_speedups, label='Masked Sparse', alpha=0.7)
    ax.scatter(sparsities, coo_speedups, label='COO Sparse', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', label='No Speedup')
    ax.set_xlabel("Sparsity Level", fontsize=12)
    ax.set_ylabel("Speedup vs Dense", fontsize=12)
    ax.set_title("Speedup vs Sparsity Level", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Time comparison
    ax = axes[1]
    dense_times = [r["dense_time_ms"] for r in results]
    masked_times = [r["masked_time_ms"] for r in results]
    coo_times = [r["coo_sparse_time_ms"] for r in results]

    x = range(len(results))
    ax.plot(x, dense_times, label='Dense', linewidth=2)
    ax.plot(x, masked_times, label='Masked Sparse', linewidth=2)
    ax.plot(x, coo_times, label='COO Sparse', linewidth=2)
    ax.set_xlabel("Benchmark Index", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Execution Time Comparison", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# V15: SpikingBrain Visualization Functions
# =============================================================================


def plot_firing_rate_histogram(
    firing_rates: np.ndarray,
    output_path: str,
    target_rate: float = 0.38,
    healthy_range: Tuple[float, float] = (0.2, 0.6),
    title: str = "Firing Rate Distribution",
    show: bool = True,
):
    """
    Plot histogram of per-channel firing rates for SpikingBrain validation.

    Target: log-normal-like distribution with mean around target_rate.

    Args:
        firing_rates: Array of per-channel firing rates (d_model,)
        output_path: Path to save figure
        target_rate: Expected/target mean firing rate
        healthy_range: Acceptable range for mean firing rate
        title: Plot title
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(
        firing_rates,
        bins=50,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
    )

    # Mean line
    mean_rate = float(firing_rates.mean())
    ax.axvline(
        x=mean_rate,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {mean_rate:.3f}',
    )

    # Target line
    ax.axvline(
        x=target_rate,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f'Target: {target_rate:.2f}',
    )

    # Healthy range shading
    ax.axvspan(
        healthy_range[0],
        healthy_range[1],
        alpha=0.15,
        color='green',
        label=f'Healthy: [{healthy_range[0]}, {healthy_range[1]}]',
    )

    ax.set_xlabel("Firing Rate", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_cka_by_layer(
    cka_values: Dict[str, float],
    output_path: str,
    threshold: float = 0.3,
    title: str = "CKA Similarity by Layer",
    show: bool = True,
):
    """
    Bar chart of CKA similarities between student and teacher layers.

    Args:
        cka_values: Dictionary with keys like 'cka_layer_0_to_2' and values
        output_path: Path to save figure
        threshold: Minimum acceptable CKA for pass
        title: Plot title
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract layer pairs and values
    layer_pairs = []
    values = []
    for key, value in cka_values.items():
        if key.startswith("cka_layer") and "_to_" in key:
            # Parse 'cka_layer_0_to_2' -> 'S0->T2'
            parts = key.replace("cka_layer_", "").split("_to_")
            if len(parts) == 2:
                label = f"S{parts[0]}->T{parts[1]}"
                layer_pairs.append(label)
                values.append(value)

    if not layer_pairs:
        # Fallback if no layer pairs found
        layer_pairs = list(cka_values.keys())
        values = list(cka_values.values())

    x = np.arange(len(layer_pairs))

    # Color bars based on threshold
    colors = ['forestgreen' if v >= threshold else 'coral' for v in values]

    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)

    # Threshold line
    ax.axhline(
        y=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold: {threshold}',
    )

    # Mean line
    if values:
        mean_cka = np.mean(values)
        ax.axhline(
            y=mean_cka,
            color='blue',
            linestyle=':',
            linewidth=2,
            label=f'Mean: {mean_cka:.3f}',
        )

    ax.set_xlabel("Layer Mapping", fontsize=12)
    ax.set_ylabel("CKA Similarity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_pairs, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout()
    save_figure(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close()
