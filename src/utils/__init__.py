"""
ASNN-Goose Utilities

This module contains:
- Visualization: Publication-quality figures
- Logging: JSON Lines structured logging
- Checkpointing: Model save/load
"""

from .visualization import (
    FigureSaver,
    plot_training_curves,
    plot_spike_density_over_time,
    plot_firing_map,
    plot_sparsity_structure,
    plot_comparison_bar,
    plot_ttt_drift_analysis,
    setup_plotting_style,
)
from .logging_utils import (
    JSONLinesLogger,
    MetricsTracker,
    log_metrics,
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager,
)

__all__ = [
    # Visualization
    "FigureSaver",
    "plot_training_curves",
    "plot_spike_density_over_time",
    "plot_firing_map",
    "plot_sparsity_structure",
    "plot_comparison_bar",
    "plot_ttt_drift_analysis",
    "setup_plotting_style",
    # Logging
    "JSONLinesLogger",
    "MetricsTracker",
    "log_metrics",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
]
