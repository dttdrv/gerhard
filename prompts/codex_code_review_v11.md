# Codex CLI Code Review Prompt: ASNN-Goose V11.1

## Context

You are reviewing a Jupyter notebook implementing knowledge distillation from GPT-2 (124M params) to a Spiking Neural Network student model (~51M params). The notebook recently had critical bugs that caused:
- PPL regression from 526 to 1318 (temperature runaway to 10.0)
- Training time increase from 26 min to 37 min
- Missing variable instantiations causing NameErrors

Seven fixes were applied. Your task is to verify these fixes are correct and identify any remaining issues.

## Files to Review

```
notebooks/asnn_goose_colab_v11.ipynb
```

## Review Checklist

### 1. CRITICAL: Temperature Regularization (Cell 17)

**Verify this exact pattern exists:**
```python
if temp_module is not None:
    # Anchor to stage target (if progressive) or default temp (if not)
    anchor_temp = stage_params['temp_target'] if cfg.use_progressive_stages else cfg.temperature
    temp_reg_loss = 0.1 * (T - anchor_temp) ** 2
else:
    temp_reg_loss = torch.tensor(0.0, device=device)
```

**Check for:**
- [ ] `anchor_temp` variable is defined BEFORE being used
- [ ] Fallback to `cfg.temperature` (should be 2.0) when progressive stages disabled
- [ ] No remaining references to the old buggy pattern: `if temp_module is not None and cfg.use_progressive_stages:`

### 2. Variable Instantiation (Cell 18)

**Verify these lines exist BEFORE the distill_v11() call:**
```python
hw_stats = HardwareStatsCollector()
spike_stats = SpikeStatsCollector(config.n_layers)
```

**Check for:**
- [ ] Both instantiations present
- [ ] `config.n_layers` matches the config (should be 8)
- [ ] No other undefined variables passed to `distill_v11()`

### 3. Speed Optimizations

**Cell 4 (Config):**
- [ ] `eval_interval: int = 300` (NOT 100)

**Cell 11 (Teacher):**
- [ ] `teacher.config.use_cache = False` after model loading
- [ ] `torch.compile(teacher, mode='reduce-overhead')` with try/except fallback

**Cell 15 (Evaluate function):**
- [ ] `torch.inference_mode()` context manager wrapping the loop
- [ ] `non_blocking=True` in `.to(device)` call

**Cell 17 (Distill function):**
- [ ] `torch.inference_mode()` instead of `torch.no_grad()` for teacher forward

### 4. Logical Correctness

**Temperature learning:**
- [ ] `LearnableTemperature` class uses log-parameterization: `torch.exp(self.log_temp)`
- [ ] Clamp range is [1.0, 10.0]
- [ ] Separate learning rate (0.01) in optimizer param groups

**KL Distillation:**
- [ ] Temperature scaling applied to BOTH student and teacher logits
- [ ] KL loss multiplied by T² for gradient magnitude correction
- [ ] Using `reduction='batchmean'`

**Gradient accumulation:**
- [ ] Loss divided by `accumulation_steps` before backward
- [ ] Optimizer step only after `accumulation_steps` iterations
- [ ] Gradient clipping applied after unscaling

### 5. Potential Issues to Flag

**Memory leaks:**
- [ ] No tensors being appended to lists without `.item()` or `.detach()`
- [ ] `optimizer.zero_grad(set_to_none=True)` used

**Numerical stability:**
- [ ] No division by zero risks
- [ ] Gradient clipping with `max_grad_norm`
- [ ] Mixed precision with GradScaler

**Config consistency:**
- [ ] `d_model = 512`, `n_layers = 8` (v10 architecture)
- [ ] `use_learnable_temperature = True`
- [ ] `use_progressive_stages = False`
- [ ] `use_channel_wise_spikes = False`
- [ ] `hidden_align_weight = 0.0`

### 6. Edge Cases

- [ ] What happens if `stage_params` is undefined when progressive stages disabled?
- [ ] Is `cfg.temperature` guaranteed to exist in config?
- [ ] Does `torch.compile` gracefully fallback on unsupported systems?

## Output Format

Provide your review as:

```markdown
## Summary
[One paragraph overall assessment]

## Critical Issues
[List any bugs that would cause incorrect behavior]

## Warnings
[List potential issues or code smells]

## Verified Fixes
[Confirm which of the 7 fixes are correctly implemented]

## Recommendations
[Suggested improvements]
```

## Expected Behavior After Fixes

| Metric | Target |
|--------|--------|
| PPL | < 530 (close to v10's 526.4) |
| Training Time | ~24 minutes |
| Temperature | Stable around 2.0-3.0, NOT 10.0 |
| All 10 tests | PASS |
