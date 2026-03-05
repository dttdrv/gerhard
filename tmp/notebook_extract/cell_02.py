# =============================================================================
# cell 4: configuration (v14.1 - Hyperparameter Tuning per External LLM)
# =============================================================================
@dataclass
class Config:
    # Version for dynamic labeling (NEVER hardcode versions elsewhere!)
    VERSION: str = 'v15'
    VERSION_DESC: str = 'SpikingBrain information encoding validation'
    
    # gpt-2 teacher (frozen, pre-trained)
    teacher_name: str = "gpt2"

    # student model architecture - v14.1: capacity increase (512d, 5L, ~56M)
    d_model: int = 768      # v14.3: safer scaling (512->768 instead of 512->1024)
    n_layers: int = 5       # v10 value (DO NOT reduce)
    vocab_size: int = 50257
    max_seq_len: int = 256

    # ==========================================================================
    # v14.1: Feature Dynamics Distillation (FDD) with CKA Loss
    # ==========================================================================
    # FDD aligns layer-wise dynamics (Δh) between student and teacher
    # Uses CKA (Centered Kernel Alignment) - projector-free, dimension-agnostic
    use_fdd: bool = True
    fdd_weight: float = 0.1           # v14.1: 100x increase (CKA bounded [0,1], safe)
    fdd_warmup_steps: int = 500       # Don't enable until step 500
    fdd_loss_type: str = "cka"        # Options: "cka" (recommended), "mse"
    fdd_kill_threshold: float = 0.10  # Disable if PPL increases >10%

    # ==========================================================================
    # v14.1: Hard Distillation (CE with ground truth)
    # ==========================================================================
    # Anchors student to correct tokens, not just teacher's soft distribution
    ce_hard_weight: float = 0.5       # Ground truth CE loss weight
    
    # Layer mapping: student_layer -> teacher_layer
    # With 5 student layers and 12 teacher layers:
    # We align early/middle/late semantic representations
    # Default: {0: 2, 2: 6, 4: 10}
    fdd_n_align_layers: int = 3       # Number of layer pairs to align

    # ==========================================================================
    # v14.1: Extended Training (same as v13.1)
    # ==========================================================================
    distill_steps: int = 7000       # v14.3: more steps for larger model
    distill_lr: float = 2e-4       # v14.3: reduced for larger model
    warmup_steps: int = 100
    min_lr: float = 1e-6

    # v14.1: gradient accumulation
    accumulation_steps: int = 2       # effective batch = 8 * 2 = 16

    # ==========================================================================
    # v14.1: Early Stopping (same as v13.1)
    # ==========================================================================
    use_early_stopping: bool = True
    early_stopping_patience: int = 800  # v14.3: more patience for larger model
    min_ppl_delta: float = 1.0

    # ==========================================================================
    # v14.1: POCL DISABLED (failed in v13)
    # ==========================================================================
    use_pocl: bool = False
    pocl_stages: int = 3
    pocl_temp_schedule: tuple = (1.0, 1.5, 2.0)
    pocl_pretrain_steps: int = 100

    # ==========================================================================
    # v14.1: CTKD ENABLED (proven in v12.1, v13.1)
    # ==========================================================================
    use_ctkd: bool = True
    tau_min: float = 1.0
    tau_max: float = 5.0
    tau_init: float = 2.0
    lambda_max: float = 1.0
    lambda_warmup_ratio: float = 0.25  # v14.3: slower CTKD ramp-up

    # Legacy flags (all disabled for v14.1)
    use_learnable_temperature: bool = False
    use_channel_wise_spikes: bool = False
    use_progressive_stages: bool = False
    temperature: float = 2.0

    # Hidden alignment DISABLED (using FDD instead)
    hidden_align_weight: float = 0.0
    teacher_d_model: int = 768
    teacher_n_layers: int = 12
    temperature_lr: float = 0.001

    # lora for ttt
    lora_rank: int = 8
    lora_alpha: float = 16.0
    ttt_lr: float = 1e-4
    ttt_steps: int = 100

    # spiking parameters
    spike_alpha: float = 1.0
    spike_threshold_mix: float = 0.35
    spike_surrogate_temp: float = 0.10

    # v15 spike semantic/health shaping
    use_spike_semantic_loss: bool = True
    spike_semantic_weight: float = 0.08
    spike_semantic_warmup_steps: int = 400
    spike_target_threshold_scale: float = 0.75

    # general training
    batch_size: int = 8
    max_grad_norm: float = 1.0
    eval_interval: int = 300

config = Config()

print(f"configuration (v14.1 - Hyperparameter Tuning per External LLM):")
print(f"  teacher: {config.teacher_name} (124m params)")
print(f"  student: d={config.d_model}, layers={config.n_layers} (~56M params)")
print(f"")
print(f"{config.VERSION} CHANGES (based on external LLM diagnosis):")
print(f"  d_model: 320 -> 512 (capacity increase for ternary compensation)")
print(f"  fdd_weight: 0.001 -> 0.1 (enable alignment gradient signal)")
print(f"  ce_hard_weight: {config.ce_hard_weight} (NEW - ground truth anchoring)")
print(f"")
print(f"{config.VERSION} INNOVATION - Feature Dynamics Distillation (FDD):")
print(f"  use_fdd: {config.use_fdd}")
print(f"  fdd_weight: {config.fdd_weight} (100x increase, CKA bounded [0,1], safe)")
print(f"  fdd_warmup_steps: {config.fdd_warmup_steps}")
print(f"  fdd_loss_type: {config.fdd_loss_type}")
print(f"  fdd_n_align_layers: {config.fdd_n_align_layers}")
print(f"  fdd_kill_threshold: {config.fdd_kill_threshold} (10% PPL increase triggers disable)")
print(f"")
print(f"  FDD Strategy:")
print(f"    - Align layer DYNAMICS (Δh), not just hidden states")
print(f"    - Use CKA loss (projector-free, dimension-agnostic)")
print(f"    - 100x weight increase (was too weak before)")
print(f"    - Safety kill-switch if PPL regresses")
print(f"")
print(f"{config.VERSION}: Hard Distillation:")
print(f"  ce_hard_weight: {config.ce_hard_weight}")
print(f"  Formula: L = KL + 0.5*CE + 0.1*FDD")
print(f"")
print(f"{config.VERSION}: CTKD (proven technique):")
print(f"  use_ctkd: {config.use_ctkd}")
print(f"  Temperature bounds: [{config.tau_min}, {config.tau_max}]")
print(f"  Lambda warmup: {config.lambda_warmup_ratio*100:.0f}%")
print(f"")
print(f"{config.VERSION}: Extended Training:")
print(f"  distill_steps: {config.distill_steps}")
print(f"  warmup_steps: {config.warmup_steps}")
print(f"  min_lr: {config.min_lr}")
print(f"")
print(f"{config.VERSION}: Early Stopping:")
print(f"  use_early_stopping: {config.use_early_stopping}")
print(f"  patience: {config.early_stopping_patience} steps")
print(f"  min_delta: {config.min_ppl_delta} PPL")
print(f"")
print(f"disabled features:")
print(f"  POCL: {config.use_pocl} (failed in v13)")
print(f"  channel-wise spikes: {config.use_channel_wise_spikes}")
print(f"  old hidden alignment: {config.hidden_align_weight}")
print(f"")
print(f"training:")
print(f"  accumulation: {config.accumulation_steps} (effective batch = {config.batch_size * config.accumulation_steps})")
print(f"  spike_threshold_mix: {config.spike_threshold_mix}")
print(f"  spike_surrogate_temp: {config.spike_surrogate_temp}")
print(f"  use_spike_semantic_loss: {config.use_spike_semantic_loss}")
if config.use_spike_semantic_loss:
    print(f"    spike_semantic_weight: {config.spike_semantic_weight}")
    print(f"    spike_semantic_warmup_steps: {config.spike_semantic_warmup_steps}")
    print(f"    spike_target_threshold_scale: {config.spike_target_threshold_scale}")
print(f"")
print(f"targets:")
print(f"  PPL: validate v14.3 (306.89) spike encoding quality")
