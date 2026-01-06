"""
ASNN-Goose Training Infrastructure

This module contains:
- DistillationLoss: KL divergence + feature matching
- DistillationTrainer: Training loop for distillation
- TTTController: Test-time training with drift controls
- Training utilities and helpers
"""

from .distillation import DistillationLoss, DistillationTrainer
from .ttt_controller import TTTController, LoRAAdapter
from .trainers import (
    create_optimizer,
    create_scheduler,
    TrainingState,
    train_epoch,
    evaluate,
)

__all__ = [
    "DistillationLoss",
    "DistillationTrainer",
    "TTTController",
    "LoRAAdapter",
    "create_optimizer",
    "create_scheduler",
    "TrainingState",
    "train_epoch",
    "evaluate",
]
