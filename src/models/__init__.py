"""
ASNN-Goose Model Components

This module contains:
- GooseBackbone: RWKV-style recurrence with delta-rule updates
- TernaryActivations: Ternary spiking with STE gradients
- BitLinear: BitNet b1.58 linear layer with ternary weights
- QuantizedWeights: INT8 weight quantization
- LoRAAdapter: Low-rank adapters for test-time training
- ASNNGoose: Full student model
- TeacherModel: Dense teacher for distillation
"""

from .goose_backbone import GooseBackbone, GooseRecurrentLayer, DeltaRuleState
from .ternary_activations import (
    TernaryQuantizer,
    AdaptiveTernarySpike,
    FixedThresholdTernarySpike,
    SpikeActivityTracker,
    # BitNet b1.58
    BitLinear,
    weight_quant_absmean,
    weight_quant_absmean_ste,
    activation_quant_absmax,
)
from .quantized_weights import QuantizedLinear, quantize_model_weights
from .lora_adapter import LoRAAdapter, LoRALinear, apply_lora_to_model
from .asnn_goose import ASNNGoose
from .teacher_model import TeacherModel

__all__ = [
    # Backbone
    "GooseBackbone",
    "GooseRecurrentLayer",
    "DeltaRuleState",
    # Ternary / BitNet
    "TernaryQuantizer",
    "AdaptiveTernarySpike",
    "FixedThresholdTernarySpike",
    "SpikeActivityTracker",
    "BitLinear",
    "weight_quant_absmean",
    "weight_quant_absmean_ste",
    "activation_quant_absmax",
    # Quantization
    "QuantizedLinear",
    "quantize_model_weights",
    # LoRA
    "LoRAAdapter",
    "LoRALinear",
    "apply_lora_to_model",
    # Full models
    "ASNNGoose",
    "TeacherModel",
]
