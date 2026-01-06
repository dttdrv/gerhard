"""
ASNN-Goose Prototype Configuration
All hyperparameters in one place for reproducibility.

Reference: Lumis-NEXT blueprint
Target: Kaggle T4 GPU (16GB VRAM)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbone dimensions
    d_model: int = 256          # Hidden dimension (small for prototype)
    n_layers: int = 4           # Number of recurrent layers
    vocab_size: int = 32000     # Vocabulary size (GPT-2 tokenizer)
    max_seq_len: int = 1024     # Maximum sequence length

    # Recurrence settings
    expand_factor: int = 2      # FFN expansion factor

    # Ternary activation settings
    ternary_threshold_init: float = 0.5  # Initial threshold for spike formation
    adaptive_threshold: bool = True       # Use data-dependent thresholds (Eq. 4)
    threshold_alpha: float = 1.0          # Scale for adaptive threshold
    learnable_alpha: bool = True          # Make alpha learnable

    # Weight quantization
    weight_bits: int = 8        # INT8 quantization
    quantize_during_training: bool = True  # QAT vs PTQ
    symmetric_quantization: bool = True    # Symmetric INT8 range

    # LoRA settings for TTT
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["key_proj", "value_proj"]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "expand_factor": self.expand_factor,
            "ternary_threshold_init": self.ternary_threshold_init,
            "adaptive_threshold": self.adaptive_threshold,
            "threshold_alpha": self.threshold_alpha,
            "learnable_alpha": self.learnable_alpha,
            "weight_bits": self.weight_bits,
            "quantize_during_training": self.quantize_during_training,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_target_modules": self.lora_target_modules,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000

    # Distillation loss weights
    kl_weight: float = 1.0
    feature_match_weight: float = 0.1  # Optional auxiliary loss
    temperature: float = 2.0           # Distillation temperature

    # STE settings
    ste_estimator: str = "straight_through"  # or "sigmoid_derivative"

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000

    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision for T4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "kl_weight": self.kl_weight,
            "feature_match_weight": self.feature_match_weight,
            "temperature": self.temperature,
            "ste_estimator": self.ste_estimator,
            "max_grad_norm": self.max_grad_norm,
            "use_amp": self.use_amp,
        }


@dataclass
class TTTConfig:
    """Test-Time Training configuration."""
    # Trigger thresholds
    spike_density_trigger: float = 0.8   # Trigger if density > this
    flicker_rate_trigger: float = 0.3    # Trigger if flicker > this

    # Drift controls (Section 7.2)
    max_update_norm: float = 0.1         # Bounded step size
    trust_region_threshold: float = 0.05 # Accept if improvement > this
    reversion_threshold: float = -0.02   # Revert if degradation > this

    # Update frequency
    min_steps_between_updates: int = 100

    # TTT optimizer
    ttt_learning_rate: float = 1e-4
    ttt_weight_decay: float = 0.0

    # Monitoring window
    monitoring_window: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spike_density_trigger": self.spike_density_trigger,
            "flicker_rate_trigger": self.flicker_rate_trigger,
            "max_update_norm": self.max_update_norm,
            "trust_region_threshold": self.trust_region_threshold,
            "reversion_threshold": self.reversion_threshold,
            "min_steps_between_updates": self.min_steps_between_updates,
            "ttt_learning_rate": self.ttt_learning_rate,
            "monitoring_window": self.monitoring_window,
        }


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Benchmarks
    eval_tasks: List[str] = field(default_factory=lambda: [
        "wikitext_perplexity",
        "copy_task",
        "retrieval_task"
    ])

    # Spike analysis
    compute_firing_maps: bool = True
    track_per_layer_stats: bool = True

    # Perplexity evaluation
    eval_batch_size: int = 8
    max_eval_samples: int = 1000

    # Copy task
    copy_seq_lengths: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    # Retrieval task
    retrieval_context_len: int = 512
    retrieval_query_positions: List[int] = field(default_factory=lambda: [128, 256, 384])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_tasks": self.eval_tasks,
            "compute_firing_maps": self.compute_firing_maps,
            "track_per_layer_stats": self.track_per_layer_stats,
            "eval_batch_size": self.eval_batch_size,
            "max_eval_samples": self.max_eval_samples,
            "copy_seq_lengths": self.copy_seq_lengths,
        }


@dataclass
class VisualizationConfig:
    """Visualization settings for publication-quality figures."""
    # Style
    figure_dpi: int = 300
    font_size: int = 12
    title_font_size: int = 14
    color_palette: str = "viridis"

    # Figure sizes (width, height in inches)
    single_col_width: float = 3.5
    double_col_width: float = 7.0
    default_height: float = 4.0

    # Saving
    save_png: bool = True
    save_pdf: bool = True
    figure_dir: str = "outputs/figures"

    # Style presets
    use_latex_fonts: bool = False  # Set True if LaTeX is available
    grid_alpha: float = 0.3
    line_width: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_dpi": self.figure_dpi,
            "font_size": self.font_size,
            "color_palette": self.color_palette,
            "save_png": self.save_png,
            "save_pdf": self.save_pdf,
            "figure_dir": self.figure_dir,
        }


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Tokenizer
    tokenizer_name: str = "gpt2"

    # Sequence length
    max_length: int = 512

    # Data splits
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

    # Preprocessing
    num_workers: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
        }


class Config:
    """
    Global configuration container.
    Access via: from config import get_config; cfg = get_config()
    """
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        ttt: Optional[TTTConfig] = None,
        eval_config: Optional[EvalConfig] = None,
        viz: Optional[VisualizationConfig] = None,
        data: Optional[DataConfig] = None,
        seed: int = 42,
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.ttt = ttt or TTTConfig()
        self.eval = eval_config or EvalConfig()
        self.viz = viz or VisualizationConfig()
        self.data = data or DataConfig()

        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as nested dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "ttt": self.ttt.to_dict(),
            "eval": self.eval.to_dict(),
            "viz": self.viz.to_dict(),
            "data": self.data.to_dict(),
            "seed": self.seed,
            "device": self.device,
        }

    def set_seed(self):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            # Deterministic operations (may be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Global config instance
_CONFIG: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def set_config(config: Config):
    """Set global configuration."""
    global _CONFIG
    _CONFIG = config


def reset_config():
    """Reset to default configuration."""
    global _CONFIG
    _CONFIG = None


# Convenience function for quick access
def cfg() -> Config:
    """Shorthand for get_config()."""
    return get_config()
