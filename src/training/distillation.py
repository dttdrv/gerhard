"""
Knowledge Distillation from Dense Teacher to ASNN-Goose Student.

Reference: Section 6 of ASNN-Goose blueprint.

This module implements:
1. DistillationLoss: KL divergence + optional feature matching
2. DistillationTrainer: Complete training loop for distillation

The distillation process trains the spiking student model to match
the soft probability outputs of the dense teacher model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from tqdm import tqdm
import time


class DistillationLoss(nn.Module):
    """
    Combined distillation loss with optional feature matching.

    Loss = kl_weight * KL(p_teacher || p_student)
         + feature_weight * MSE(h_teacher, h_student)

    Reference: Hinton et al. "Distilling the Knowledge in a Neural Network"

    Args:
        kl_weight: Weight for KL divergence loss
        feature_weight: Weight for feature matching loss
        temperature: Softmax temperature for soft targets
        feature_layers: List of layer indices for feature matching
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        feature_weight: float = 0.1,
        temperature: float = 2.0,
        feature_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.feature_layers = feature_layers or []

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: Optional[Dict[int, torch.Tensor]] = None,
        teacher_features: Optional[Dict[int, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        label_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits (batch, seq, vocab)
            teacher_logits: Teacher model logits (batch, seq, vocab)
            student_features: Optional dict of student layer activations
            teacher_features: Optional dict of teacher layer activations
            labels: Optional hard labels for auxiliary loss
            label_weight: Weight for hard label cross-entropy loss

        Returns:
            total_loss: Scalar loss
            loss_components: Dict with individual loss values for logging
        """
        loss_components: Dict[str, float] = {}

        # KL Divergence loss (Eq. 3)
        # Use temperature-scaled softmax
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )

        # KL(teacher || student) = sum(p_t * log(p_t / p_s))
        # = sum(p_t * log(p_t)) - sum(p_t * log(p_s))
        # We use F.kl_div which expects log_probs for input
        kl_loss = F.kl_div(
            student_log_probs.view(-1, student_logits.size(-1)),
            teacher_probs.view(-1, teacher_logits.size(-1)),
            reduction="batchmean",
        )

        # Scale by temperature squared (Hinton et al.)
        kl_loss = kl_loss * (self.temperature ** 2)

        loss_components["kl_loss"] = kl_loss.item()
        total_loss = self.kl_weight * kl_loss

        # Feature matching loss (optional)
        if (
            self.feature_weight > 0
            and student_features is not None
            and teacher_features is not None
            and self.feature_layers
        ):
            feature_loss = torch.tensor(0.0, device=student_logits.device)
            matched_layers = 0

            for layer_idx in self.feature_layers:
                if layer_idx in student_features and layer_idx in teacher_features:
                    s_feat = student_features[layer_idx]
                    t_feat = teacher_features[layer_idx]

                    # Normalize features before matching
                    s_feat = F.normalize(s_feat.float(), dim=-1)
                    t_feat = F.normalize(t_feat.float(), dim=-1)

                    # MSE loss on normalized features
                    feature_loss = feature_loss + F.mse_loss(s_feat, t_feat)
                    matched_layers += 1

            if matched_layers > 0:
                feature_loss = feature_loss / matched_layers
                loss_components["feature_loss"] = feature_loss.item()
                total_loss = total_loss + self.feature_weight * feature_loss

        # Hard label loss (optional, for hybrid training)
        if labels is not None and label_weight > 0:
            # Standard cross-entropy with hard labels
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss_components["ce_loss"] = ce_loss.item()
            total_loss = total_loss + label_weight * ce_loss

        loss_components["total_loss"] = total_loss.item()
        return total_loss, loss_components


@dataclass
class DistillationMetrics:
    """Container for distillation training metrics."""
    step: int
    loss: float
    kl_loss: float
    feature_loss: float
    grad_norm: float
    learning_rate: float
    spike_density: float
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "kl_loss": self.kl_loss,
            "feature_loss": self.feature_loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "spike_density": self.spike_density,
            "time_ms": self.time_ms,
        }


class DistillationTrainer:
    """
    Training loop for knowledge distillation.

    Handles:
    - Forward passes through teacher and student
    - Loss computation
    - Gradient clipping
    - Learning rate scheduling
    - Metric logging
    - Checkpointing

    Args:
        teacher: Dense teacher model (frozen)
        student: Spiking student model
        optimizer: Optimizer for student parameters
        scheduler: Learning rate scheduler (optional)
        loss_fn: DistillationLoss instance
        config: Training configuration
        device: Device to train on
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: DistillationLoss,
        config: Any,
        device: torch.device,
    ):
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Training state
        self.step = 0
        self.logs: List[Dict[str, Any]] = []

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> DistillationMetrics:
        """
        Perform single training step.

        Args:
            batch: Dictionary with "input_ids" tensor

        Returns:
            DistillationMetrics with step statistics
        """
        start_time = time.perf_counter()
        self.student.train()

        input_ids = batch["input_ids"].to(self.device)

        # Teacher forward (no grad)
        with torch.no_grad():
            if self.config.training.use_amp:
                with torch.cuda.amp.autocast():
                    teacher_logits, _, teacher_aux = self.teacher(
                        input_ids, return_features=True
                    )
            else:
                teacher_logits, _, teacher_aux = self.teacher(
                    input_ids, return_features=True
                )

        # Student forward
        if self.config.training.use_amp:
            with torch.cuda.amp.autocast():
                student_logits, _, student_aux = self.student(
                    input_ids, return_spike_info=True
                )
                loss, loss_components = self.loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    student_features=student_aux.get("layer_activations"),
                    teacher_features=teacher_aux.get("layer_activations"),
                )
        else:
            student_logits, _, student_aux = self.student(
                input_ids, return_spike_info=True
            )
            loss, loss_components = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_features=student_aux.get("layer_activations"),
                teacher_features=teacher_aux.get("layer_activations"),
            )

        # Backward
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.training.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        self.step += 1

        # Get spike statistics
        spike_stats = self.student.get_spike_stats()
        mean_density = 0.0
        count = 0
        for layer_stats in spike_stats.values():
            if "k_density" in layer_stats:
                mean_density += layer_stats["k_density"]
                count += 1
            if "v_density" in layer_stats:
                mean_density += layer_stats["v_density"]
                count += 1
        mean_density = mean_density / max(count, 1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        metrics = DistillationMetrics(
            step=self.step,
            loss=loss_components.get("total_loss", 0.0),
            kl_loss=loss_components.get("kl_loss", 0.0),
            feature_loss=loss_components.get("feature_loss", 0.0),
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            learning_rate=self.scheduler.get_last_lr()[0] if self.scheduler else self.config.training.learning_rate,
            spike_density=mean_density,
            time_ms=elapsed_ms,
        )

        self.logs.append(metrics.to_dict())
        return metrics

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        progress_bar: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            progress_bar: Whether to show progress bar

        Returns:
            Dictionary of average metrics for the epoch
        """
        epoch_metrics = {
            "loss": 0.0,
            "kl_loss": 0.0,
            "feature_loss": 0.0,
            "grad_norm": 0.0,
            "spike_density": 0.0,
            "time_ms": 0.0,
        }
        num_batches = 0

        iterator = tqdm(dataloader, desc="Training") if progress_bar else dataloader

        for batch in iterator:
            metrics = self.train_step(batch)

            for key in epoch_metrics:
                epoch_metrics[key] += getattr(metrics, key)
            num_batches += 1

            if progress_bar:
                iterator.set_postfix({
                    "loss": f"{metrics.loss:.4f}",
                    "spike": f"{metrics.spike_density:.3f}",
                })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all training logs."""
        return self.logs

    def get_spike_stats(self) -> Dict[str, float]:
        """Collect spike statistics from student model."""
        return self.student.get_spike_stats()

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            "step": self.step,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "logs": self.logs,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.step = checkpoint["step"]
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logs = checkpoint.get("logs", [])


def create_distillation_trainer(
    teacher: nn.Module,
    student: nn.Module,
    config: Any,
    device: torch.device,
) -> DistillationTrainer:
    """
    Factory function to create a DistillationTrainer.

    Args:
        teacher: Dense teacher model
        student: Spiking student model
        config: Configuration object with training settings
        device: Device to train on

    Returns:
        Configured DistillationTrainer
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.max_steps,
        eta_min=config.training.learning_rate * 0.1,
    )

    # Create loss function
    loss_fn = DistillationLoss(
        kl_weight=config.training.kl_weight,
        feature_weight=config.training.feature_match_weight,
        temperature=config.training.temperature,
        feature_layers=list(range(config.model.n_layers)),
    )

    return DistillationTrainer(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )
