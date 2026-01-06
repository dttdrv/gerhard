"""
Memory-locked Test-Time Training with LoRA adapters.

Reference: Section 7 of ASNN-Goose blueprint.

This module implements:
1. TTTController: Manages test-time training updates
2. Trigger logic based on spike behavior anomalies
3. Drift controls with bounded updates
4. Automatic reversion on degradation

The key insight is that anomalous spike patterns (high density or flicker)
indicate the model is struggling, triggering adaptation via LoRA updates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import copy
from dataclasses import dataclass

from ..models.lora_adapter import LoRAAdapter, get_lora_parameters


@dataclass
class TTTEvent:
    """Record of a TTT update event."""
    step: int
    trigger_reason: str
    validation_before: float
    validation_after: float
    improvement: float
    accepted: bool
    reverted: bool
    lora_norms: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "trigger_reason": self.trigger_reason,
            "validation_before": self.validation_before,
            "validation_after": self.validation_after,
            "improvement": self.improvement,
            "accepted": self.accepted,
            "reverted": self.reverted,
            "lora_norms": self.lora_norms,
        }


class TTTController:
    """
    Controller for test-time training updates.

    Implements:
    - Spike-based trigger logic (high density or flicker)
    - Bounded step size via gradient clipping
    - Trust-region validation
    - Automatic reversion on degradation

    Reference: Section 7.2 of blueprint.

    Args:
        model: ASNNGoose model with LoRA adapters
        lora_modules: Dictionary of LoRA adapters by name
        config: TTT configuration
        device: Device to use
    """

    def __init__(
        self,
        model: nn.Module,
        lora_modules: Dict[str, LoRAAdapter],
        config: Any,
        device: torch.device,
    ):
        self.model = model
        self.lora_modules = lora_modules
        self.config = config
        self.device = device

        # Create optimizer for LoRA parameters only
        lora_params = []
        for lora in lora_modules.values():
            lora_params.extend([lora.lora_A, lora.lora_B])

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.ttt.ttt_learning_rate,
            weight_decay=config.ttt.ttt_weight_decay,
        )

        # State tracking
        self.steps_since_update = 0
        self.total_steps = 0
        self.recent_densities: List[float] = []
        self.recent_flickers: List[float] = []
        self.validation_scores: List[float] = []
        self.events: List[TTTEvent] = []

        # Tracking lists for analysis
        self.update_points: List[int] = []
        self.reversions: List[int] = []

        # Checkpoint for reversion
        self.checkpoint_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self.save_checkpoint()

    def save_checkpoint(self):
        """Save current LoRA state for potential reversion."""
        self.checkpoint_state = {
            name: {
                "lora_A": lora.lora_A.data.clone(),
                "lora_B": lora.lora_B.data.clone(),
            }
            for name, lora in self.lora_modules.items()
        }

    def revert_to_checkpoint(self):
        """Revert LoRA modules to last checkpoint."""
        if self.checkpoint_state is None:
            return

        for name, lora in self.lora_modules.items():
            if name in self.checkpoint_state:
                lora.lora_A.data.copy_(self.checkpoint_state[name]["lora_A"])
                lora.lora_B.data.copy_(self.checkpoint_state[name]["lora_B"])

    def update_spike_stats(self, density: float, flicker: float):
        """
        Track recent spike statistics for trigger decision.

        Args:
            density: Current spike density
            flicker: Current flicker rate
        """
        self.recent_densities.append(density)
        self.recent_flickers.append(flicker)

        # Keep only recent window
        window = self.config.ttt.monitoring_window
        self.recent_densities = self.recent_densities[-window:]
        self.recent_flickers = self.recent_flickers[-window:]

    def check_trigger(self) -> Tuple[bool, str]:
        """
        Check if TTT should be triggered.

        Returns:
            (should_trigger, reason)
        """
        # Cooldown check
        if self.steps_since_update < self.config.ttt.min_steps_between_updates:
            return False, "cooldown"

        if not self.recent_densities:
            return False, "no_data"

        # Use recent window for decision
        window = min(20, len(self.recent_densities))
        mean_density = sum(self.recent_densities[-window:]) / window
        mean_flicker = (
            sum(self.recent_flickers[-window:]) / window
            if self.recent_flickers else 0.0
        )

        # Check density trigger
        if mean_density > self.config.ttt.spike_density_trigger:
            return True, f"high_density:{mean_density:.3f}"

        # Check flicker trigger
        if mean_flicker > self.config.ttt.flicker_rate_trigger:
            return True, f"high_flicker:{mean_flicker:.3f}"

        return False, "normal"

    def compute_validation_score(self, input_ids: torch.Tensor) -> float:
        """
        Compute local validation score for trust-region check.

        Uses self-supervised perplexity as the score.
        Higher score = better performance.

        Args:
            input_ids: Token IDs for validation

        Returns:
            Validation score (negative loss, so higher is better)
        """
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(input_ids)

            # Next-token prediction loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

            # Return negative loss (higher is better)
            return -loss.item()

    def get_lora_norms(self) -> Dict[str, float]:
        """Get current LoRA delta norms."""
        return {
            name: lora.get_delta_norm()
            for name, lora in self.lora_modules.items()
        }

    def ttt_step(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Perform one TTT update step with drift controls.

        Args:
            input_ids: Token IDs for self-supervised update

        Returns:
            Dict with update status and metrics
        """
        self.steps_since_update += 1
        self.total_steps += 1

        result = {
            "updated": False,
            "reason": "",
            "validation_delta": 0.0,
            "accepted": False,
            "reverted": False,
        }

        # Check trigger
        should_trigger, reason = self.check_trigger()
        if not should_trigger:
            result["reason"] = reason
            return result

        # Compute validation score before update
        score_before = self.compute_validation_score(input_ids)
        self.validation_scores.append(score_before)

        # Save checkpoint before update
        self.save_checkpoint()

        # Perform update
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits, _, _ = self.model(input_ids)

        # Self-supervised objective: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        loss.backward()

        # Bounded step size (Eq. from Section 7.2)
        total_norm = 0.0
        for lora in self.lora_modules.values():
            for param in [lora.lora_A, lora.lora_B]:
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > self.config.ttt.max_update_norm:
            scale = self.config.ttt.max_update_norm / (total_norm + 1e-8)
            for lora in self.lora_modules.values():
                for param in [lora.lora_A, lora.lora_B]:
                    if param.grad is not None:
                        param.grad.mul_(scale)

        self.optimizer.step()

        # Trust-region check
        score_after = self.compute_validation_score(input_ids)
        improvement = score_after - score_before

        result["validation_delta"] = improvement

        # Decision logic
        if improvement < self.config.ttt.reversion_threshold:
            # Revert on significant degradation
            self.revert_to_checkpoint()
            self.reversions.append(self.total_steps)
            result["reason"] = f"reverted:{improvement:.4f}"
            result["reverted"] = True

        elif improvement < self.config.ttt.trust_region_threshold:
            # Accept but note marginal improvement
            result["updated"] = True
            result["accepted"] = True
            result["reason"] = f"marginal:{improvement:.4f}"
            self.update_points.append(self.total_steps)
            self.steps_since_update = 0

        else:
            # Good update
            result["updated"] = True
            result["accepted"] = True
            result["reason"] = f"good:{improvement:.4f}"
            self.update_points.append(self.total_steps)
            self.steps_since_update = 0

        self.validation_scores.append(score_after)

        # Record event
        event = TTTEvent(
            step=self.total_steps,
            trigger_reason=reason,
            validation_before=score_before,
            validation_after=score_after,
            improvement=improvement,
            accepted=result["accepted"],
            reverted=result["reverted"],
            lora_norms=self.get_lora_norms(),
        )
        self.events.append(event)

        return result

    def step(
        self,
        input_ids: torch.Tensor,
        density: float,
        flicker: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Main entry point for TTT step.

        Args:
            input_ids: Current input tokens
            density: Current spike density
            flicker: Current flicker rate

        Returns:
            TTT step result
        """
        self.update_spike_stats(density, flicker)
        return self.ttt_step(input_ids)

    def get_summary(self) -> Dict[str, Any]:
        """Return TTT statistics summary."""
        return {
            "total_steps": self.total_steps,
            "num_updates": len(self.update_points),
            "num_reversions": len(self.reversions),
            "update_ratio": len(self.update_points) / max(self.total_steps, 1),
            "reversion_ratio": len(self.reversions) / max(len(self.events), 1),
            "update_points": self.update_points,
            "reversions": self.reversions,
            "final_lora_norms": self.get_lora_norms(),
            "mean_recent_density": (
                sum(self.recent_densities) / len(self.recent_densities)
                if self.recent_densities else 0.0
            ),
        }

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all TTT events as dictionaries."""
        return [e.to_dict() for e in self.events]

    def reset(self):
        """Reset controller state (for new evaluation)."""
        self.steps_since_update = 0
        self.total_steps = 0
        self.recent_densities.clear()
        self.recent_flickers.clear()
        self.validation_scores.clear()
        self.events.clear()
        self.update_points.clear()
        self.reversions.clear()
        self.save_checkpoint()


def create_ttt_controller(
    model: nn.Module,
    config: Any,
    device: torch.device,
) -> TTTController:
    """
    Factory function to create a TTTController.

    Args:
        model: ASNNGoose model (should have LoRA applied)
        config: Configuration object
        device: Device to use

    Returns:
        Configured TTTController
    """
    # Ensure LoRA is applied
    if not hasattr(model, "lora_modules") or not model.lora_modules:
        # Apply LoRA if not already applied
        model.apply_lora(
            rank=config.model.lora_rank,
            alpha=config.model.lora_alpha,
            target_modules=config.model.lora_target_modules,
            freeze_base=True,
        )

    return TTTController(
        model=model,
        lora_modules=model.lora_modules,
        config=config,
        device=device,
    )
