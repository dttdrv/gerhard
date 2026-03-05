# =============================================================================
# cell 3: PRE-TRAINING VALIDATION (run before training to catch issues)
# =============================================================================
print("=" * 70)
print("PRE-TRAINING VALIDATION")
print("=" * 70)

validation_errors = []
validation_warnings = []

# 1. Config Sanity Checks
print("")
print("[1] CONFIG SANITY CHECKS")

if config.d_model < 256:
    validation_errors.append(f"d_model={config.d_model} too small (min 256)")
elif config.d_model > 2048:
    validation_warnings.append(f"d_model={config.d_model} very large - check VRAM")
print(f"  d_model: {config.d_model}")

if config.n_layers < 3:
    validation_errors.append(f"n_layers={config.n_layers} too few")
print(f"  n_layers: {config.n_layers}")

print(f"  fdd_weight: {config.fdd_weight}")
print(f"  ce_hard_weight: {config.ce_hard_weight}")
print(f"  spike_threshold_mix: {config.spike_threshold_mix}")
if not (0.0 <= config.spike_threshold_mix <= 1.0):
    validation_errors.append(f"spike_threshold_mix={config.spike_threshold_mix} must be in [0, 1]")
if config.spike_surrogate_temp <= 0:
    validation_errors.append(f"spike_surrogate_temp={config.spike_surrogate_temp} must be > 0")
if config.use_spike_semantic_loss:
    print(f"  spike_semantic_weight: {config.spike_semantic_weight}")
    print(f"  spike_semantic_warmup_steps: {config.spike_semantic_warmup_steps}")
    if config.spike_semantic_weight < 0:
        validation_errors.append(f"spike_semantic_weight={config.spike_semantic_weight} must be >= 0")
print(f"  VERSION: {config.VERSION}")
print(f"  VERSION_DESC: {config.VERSION_DESC}")

# 2. Memory Estimation
print("")
print("[2] MEMORY ESTIMATION")
embed_params = config.vocab_size * config.d_model * 2
layer_params = config.n_layers * config.d_model * config.d_model * 8
total_params_est = embed_params + layer_params
print(f"  Estimated params: ~{total_params_est/1e6:.1f}M")

vram_est_gb = (total_params_est * 4 * 3) / 1e9
print(f"  Estimated VRAM: ~{vram_est_gb:.1f}GB")

if vram_est_gb > 14:
    validation_warnings.append(f"VRAM estimate {vram_est_gb:.1f}GB may exceed 16GB limit")
    print(f"  WARNING: May exceed 16GB VRAM limit!")

# 3. Training Config
print("")
print("[3] TRAINING CONFIG")
print(f"  distill_steps: {config.distill_steps}")
print(f"  distill_lr: {config.distill_lr}")
print(f"  batch_size: {config.batch_size}")
if hasattr(config, 'accumulation_steps'):
    eff_batch = config.batch_size * config.accumulation_steps
    print(f"  effective_batch: {eff_batch}")
print(f"  early_stopping: patience={config.early_stopping_patience}")

# 4. Feature Flags
print("")
print("[4] FEATURE FLAGS")
print(f"  use_fdd: {config.use_fdd}")
print(f"  use_ctkd: {config.use_ctkd}")
print(f"  use_pocl: {config.use_pocl}")

# Summary
print("")
print("=" * 70)
if validation_errors:
    print(f"VALIDATION FAILED - {len(validation_errors)} errors:")
    for e in validation_errors:
        print(f"   - {e}")
    raise RuntimeError("Fix validation errors before training!")
elif validation_warnings:
    print(f"VALIDATION PASSED WITH {len(validation_warnings)} WARNINGS:")
    for w in validation_warnings:
        print(f"   - {w}")
else:
    print("ALL VALIDATIONS PASSED")
print("=" * 70)
