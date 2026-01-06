"""
Model checkpointing utilities for ASNN-Goose.

Provides save/load functionality with training state preservation.
Kaggle-compatible with support for Google Drive backup.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
import shutil


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    config: Dict[str, Any],
    filepath: str,
    extra_state: Optional[Dict[str, Any]] = None,
):
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        step: Current training step
        config: Configuration dictionary
        filepath: Output path
        extra_state: Additional state to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    # Save atomically (write to temp, then rename)
    temp_path = filepath.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(filepath)

    print(f"Saved checkpoint to {filepath} (step {step})")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load to
        strict: Whether to require exact key match

    Returns:
        Dictionary with loaded state (step, config, extra_state)
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from {filepath} (step {checkpoint['step']})")

    return {
        "step": checkpoint["step"],
        "config": checkpoint.get("config", {}),
        "extra_state": checkpoint.get("extra_state", {}),
        "timestamp": checkpoint.get("timestamp"),
    }


class CheckpointManager:
    """
    Manage multiple checkpoints with rotation.

    Keeps the N most recent checkpoints and optionally
    the best checkpoint based on a metric.

    Usage:
        manager = CheckpointManager("outputs/checkpoints", max_to_keep=3)
        manager.save(model, optimizer, scheduler, step, config, metrics={"loss": 0.5})
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        keep_best: bool = True,
        best_metric: str = "loss",
        mode: str = "min",  # "min" or "max"
    ):
        """
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_to_keep: Maximum number of recent checkpoints to keep
            keep_best: Whether to keep the best checkpoint separately
            best_metric: Metric name for determining best
            mode: "min" for lower is better, "max" for higher is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.best_metric = best_metric
        self.mode = mode

        self.checkpoints: List[Path] = []
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_path = self.checkpoint_dir / "best.pt"

        # Load existing checkpoints
        self._scan_existing()

    def _scan_existing(self):
        """Scan for existing checkpoints in directory."""
        pattern = "checkpoint_*.pt"
        existing = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )
        self.checkpoints = existing

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        step: int,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            step: Current step
            config: Configuration
            metrics: Optional metrics for best tracking
            extra_state: Additional state

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint path
        filename = f"checkpoint_{step:08d}.pt"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            filepath=str(filepath),
            extra_state=extra_state,
        )

        self.checkpoints.append(filepath)

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_to_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_path:
                old_path.unlink()
                print(f"Removed old checkpoint: {old_path}")

        # Check if this is the best
        if self.keep_best and metrics and self.best_metric in metrics:
            value = metrics[self.best_metric]
            is_best = (
                (self.mode == "min" and value < self.best_value) or
                (self.mode == "max" and value > self.best_value)
            )

            if is_best:
                self.best_value = value
                shutil.copy(filepath, self.best_path)
                print(f"New best checkpoint! {self.best_metric}={value:.4f}")

        return filepath

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Loaded state or None if no checkpoints exist
        """
        if not self.checkpoints:
            print("No checkpoints found")
            return None

        return load_checkpoint(
            str(self.checkpoints[-1]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.

        Returns:
            Loaded state or None if no best checkpoint exists
        """
        if not self.best_path.exists():
            print("No best checkpoint found")
            return None

        return load_checkpoint(
            str(self.best_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def get_latest_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_best_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        return self.best_path if self.best_path.exists() else None


def save_model_only(
    model: nn.Module,
    filepath: str,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Save only model weights (smaller file, for inference).

    Args:
        model: Model to save
        filepath: Output path
        config: Optional config to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if config is not None:
        state["config"] = config

    torch.save(state, filepath)
    print(f"Saved model to {filepath}")


def load_model_only(
    filepath: str,
    model: nn.Module,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load only model weights.

    Args:
        filepath: Path to checkpoint
        model: Model to load into
        device: Device to load to
        strict: Whether to require exact key match

    Returns:
        Config if present
    """
    state = torch.load(filepath, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    print(f"Loaded model from {filepath}")
    return state.get("config", {})
