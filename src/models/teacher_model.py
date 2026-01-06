"""
Teacher Model wrapper for knowledge distillation.

Reference: Section 6 of ASNN-Goose blueprint.

The teacher is a dense GooseBackbone model with continuous activations.
It provides soft targets for training the spiking student model.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple

from .goose_backbone import GooseBackbone, DeltaRuleState


class TeacherModel(nn.Module):
    """
    Dense teacher model wrapper for distillation.

    Wraps GooseBackbone with additional functionality for:
    - Generating soft targets with temperature scaling
    - Extracting intermediate features for feature matching
    - Freezing weights during distillation

    Args:
        backbone: GooseBackbone instance (or config to create one)
        d_model: Model dimension (if creating new backbone)
        n_layers: Number of layers (if creating new backbone)
        vocab_size: Vocabulary size (if creating new backbone)
        max_seq_len: Maximum sequence length (if creating new backbone)
    """

    def __init__(
        self,
        backbone: Optional[GooseBackbone] = None,
        d_model: int = 256,
        n_layers: int = 4,
        vocab_size: int = 32000,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = GooseBackbone(
                d_model=d_model,
                n_layers=n_layers,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
            )

        self.d_model = self.backbone.d_model
        self.n_layers = self.backbone.n_layers
        self.vocab_size = self.backbone.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[DeltaRuleState]] = None,
        temperature: float = 1.0,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, List[DeltaRuleState], Dict[str, Any]]:
        """
        Forward pass with optional temperature scaling.

        Args:
            input_ids: Token IDs (batch, seq_len)
            states: Optional initial states
            temperature: Temperature for soft targets (default 1.0)
            return_features: Whether to return intermediate features

        Returns:
            logits: (batch, seq_len, vocab_size) - optionally temperature-scaled
            states: Final recurrent states
            aux_outputs: Dict with features for distillation
        """
        logits, states, aux = self.backbone(
            input_ids,
            states=states,
            return_hidden=return_features,
        )

        # Apply temperature scaling to logits
        if temperature != 1.0:
            logits = logits / temperature

        return logits, states, aux

    def get_soft_targets(
        self,
        input_ids: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """
        Get soft probability targets for distillation.

        Args:
            input_ids: Token IDs
            temperature: Distillation temperature

        Returns:
            Soft probabilities (batch, seq_len, vocab_size)
        """
        with torch.no_grad():
            logits, _, _ = self.forward(
                input_ids,
                temperature=temperature,
                return_features=False,
            )
            return torch.softmax(logits, dim=-1)

    def freeze(self):
        """Freeze all teacher parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Unfreeze all teacher parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_config(self) -> Dict[str, Any]:
        """Get teacher model configuration."""
        return self.backbone.get_config()

    def count_parameters(self) -> int:
        """Count total parameters."""
        return self.backbone.count_parameters()

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TeacherModel":
        """
        Load teacher model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load to

        Returns:
            Loaded TeacherModel
        """
        checkpoint = torch.load(path, map_location=device)

        # Extract config
        config = checkpoint.get("config", {})

        # Create model
        model = cls(
            d_model=config.get("d_model", 256),
            n_layers=config.get("n_layers", 4),
            vocab_size=config.get("vocab_size", 32000),
            max_seq_len=config.get("max_seq_len", 1024),
        )

        # Load state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

        return model

    def save(self, path: str):
        """
        Save teacher model to checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "config": self.get_config(),
            "model_state_dict": self.state_dict(),
        }, path)


def create_teacher_student_pair(
    d_model: int = 256,
    n_layers: int = 4,
    vocab_size: int = 32000,
    max_seq_len: int = 1024,
    adaptive_threshold: bool = True,
    threshold_alpha: float = 1.0,
    quantize_student: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> Tuple["TeacherModel", "ASNNGoose"]:
    """
    Create matched teacher-student model pair for distillation.

    Args:
        d_model: Model dimension
        n_layers: Number of layers
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        adaptive_threshold: Use adaptive spiking thresholds
        threshold_alpha: Initial threshold alpha
        quantize_student: Use INT8 quantization for student
        lora_rank: LoRA rank for TTT
        lora_alpha: LoRA alpha for TTT

    Returns:
        Tuple of (teacher, student) models
    """
    from .asnn_goose import ASNNGoose

    # Create teacher
    teacher = TeacherModel(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )

    # Create student with same architecture
    student = ASNNGoose(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        adaptive_threshold=adaptive_threshold,
        threshold_alpha=threshold_alpha,
        quantize_weights=quantize_student,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Copy embeddings from teacher to student
    with torch.no_grad():
        student.embedding.weight.copy_(teacher.backbone.embedding.weight)
        student.pos_embedding.weight.copy_(teacher.backbone.pos_embedding.weight)

    return teacher, student
