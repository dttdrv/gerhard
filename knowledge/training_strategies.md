# Training Strategies

## 1. Mixed Precision Training (Deep Dive)

### Numerical Formats
| Format | Bits | Range | Use Case |
|--------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | Default, stable |
| FP16 | 16 | ±65,504 | Fast, needs GradScaler |
| BF16 | 16 | ±3.4e38 | Same range as FP32, no scaler needed |
| TF32 | 19 | ±3.4e38 | Internal, auto on Ampere+ |

### torch.amp.autocast Rules
| Category | Operations | Precision |
|----------|------------|-----------|
| **Allowlist** | mm, conv2d, linear | FP16/BF16 |
| **Denylist** | sum, mean, softmax, losses | FP32 |
| **Promote** | Mixed inputs | Widest |

### GradScaler Logic
```python
# Required for FP16, OPTIONAL for BF16
scaler = torch.amp.GradScaler("cuda")

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Must unscale BEFORE clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)  # Skips if Inf/NaN detected
scaler.update()
```

**Monitoring**: If `scaler.get_scale()` drops rapidly → instability

### Hardware Recommendations
| Architecture | Recommended | GradScaler |
|--------------|-------------|------------|
| Volta (V100) | FP16 | Required |
| Turing (T4) | FP16 | Required |
| Ampere (A100) | BF16 | Optional |
| Hopper (H100) | BF16/FP8 | Optional/Required |

---

## 2. torch.compile Optimization

### Compilation Stack
1. **TorchDynamo**: Python-level JIT, captures graph
2. **AOTAutograd**: Captures backward graph ahead-of-time
3. **TorchInductor**: Maps to Triton kernels, fuses operators

### Modes
| Mode | Description | Compile Time | Best For |
|------|-------------|--------------|----------|
| `default` | Basic fusion | Fast | Development |
| `reduce-overhead` | CUDA Graphs | Medium | Small models |
| `max-autotune` | Full Triton tuning | Slow | Production |

### Expected Speedup (A100)
- Transformers: **1.3x - 1.8x**
- CNNs: **1.2x - 1.5x**

### Avoiding Graph Breaks
| Cause | Fix |
|-------|-----|
| `print(loss.item())` | Move outside compiled function |
| Data-dependent if/else | Use `torch.where` |
| NumPy/SciPy calls | Use PyTorch equivalents |

**Diagnostic**:
```python
import torch._dynamo as dynamo
explanation = dynamo.explain(compiled_fn, *args)
print(explanation.graph_breaks)
```

---

## 3. torch.inference_mode vs torch.no_grad

### Key Differences
| Feature | no_grad | inference_mode |
|---------|---------|----------------|
| Disables autograd | Yes | Yes |
| Version counter | Maintained | Disabled |
| View tracking | Maintained | Disabled |
| Speed | Baseline | 5-10% faster |

### CRITICAL for KD with Learnable Temperature
```python
# WRONG - inference_mode tensors can't participate in autograd
with torch.inference_mode():
    t_logits = teacher(ids)  # Creates inference tensor
# t_logits / T fails because T requires grad!

# CORRECT - no_grad allows downstream autograd
with torch.no_grad():
    t_logits = teacher(ids)  # Normal tensor, no grad for teacher
# t_logits / T works because t_logits is a regular tensor
```

**Rule**: Use `no_grad` when downstream ops involve learnable parameters.

---

## 4. Data Loading Optimization

### DataLoader Configuration
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,           # 4-8 per GPU
    pin_memory=True,         # Enables async transfer
    persistent_workers=True, # Avoids respawn latency
    prefetch_factor=2,       # Batches per worker to prefetch
)
```

### Non-blocking Transfer
```python
# Overlap transfer with computation
input_ids = batch['input_ids'].to(device, non_blocking=True)
```

### Diagnosing Starvation
If profiler shows gaps between GPU kernels labeled "DataLoader":
1. Increase `num_workers`
2. Increase `prefetch_factor`
3. Check disk I/O bottleneck

---

## 5. Memory Optimization

### Activation Checkpointing
```python
from torch.utils.checkpoint import checkpoint

# Reduces activation memory: O(L) → O(√L)
# Cost: +33% compute (re-forward during backward)
def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x)
    return x
```

**For KD**: Apply only to student model (teacher has no backward).

### Flash Attention
- Standard attention: O(N²) memory and compute
- Flash Attention: O(N) memory, 2x-4x speedup for long sequences

```python
# PyTorch 2.0+ auto-dispatches on supported hardware
output = F.scaled_dot_product_attention(q, k, v)
```

---

## 6. Optimizer Efficiency

### Fused vs Foreach
| Implementation | Description | Speed |
|----------------|-------------|-------|
| Legacy | Python loop, separate kernels | Slow |
| `foreach=True` | Grouped kernels | Medium |
| `fused=True` | Single kernel per group | Fast |

```python
optimizer = torch.optim.AdamW(
    params,
    lr=3e-4,
    fused=True  # Fastest on CUDA
)
```

### zero_grad Performance
```python
optimizer.zero_grad(set_to_none=True)  # Faster than set_to_none=False
```

---

## 7. Gradient Accumulation

### Correct Implementation
```python
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for step, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps  # CRITICAL: scale before backward
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### DDP no_sync Optimization
```python
# Disable gradient sync for first K-1 steps
is_sync_step = (step + 1) % accumulation_steps == 0

context = model.no_sync() if (hasattr(model, 'no_sync') and not is_sync_step) else nullcontext()
with context:
    loss.backward()
```

---

## 8. Learning Rate Schedule

### Current v11.1 Schedule
```python
def lr_lambda(step):
    if step < warmup_steps:  # 50 steps
        return step / warmup_steps  # Linear warmup
    progress = (step - warmup_steps) / (total - warmup_steps)
    return 0.5 * (1 + cos(π * progress))  # Cosine decay to 50%
```

### For Extended Training (v13+)
```python
warmup_steps = 100
total_steps = 5000
min_lr = 1e-6

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max(min_lr / base_lr, 0.5 * (1 + cos(π * progress)))
```

---

## 9. Validation Best Practices

### Initial Loss Verification
```python
# Expected initial loss for C classes
expected_loss = math.log(C)  # e.g., ln(50257) ≈ 10.8 for GPT-2

# For KD with temperature T
expected_kd_loss = expected_loss  # Should scale with T²
```

### Overfit Tiny Subset Test
```python
# Should achieve loss → 0 in ~100 steps on 10-20 samples
# Failure indicates: disconnected graph, bad LR, or preprocessing bug
```

### Gradient Health Checks
```python
# After first backward
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} has no gradient!")
    elif torch.isnan(param.grad).any():
        print(f"WARNING: {name} has NaN gradient!")
    elif param.grad.norm() == 0:
        print(f"WARNING: {name} has zero gradient!")
```

### Key Thresholds
| Metric | Threshold | Action |
|--------|-----------|--------|
| Initial loss | ≈ ln(C) | Check init |
| Gradient clipping freq | < 10% | Reduce LR if exceeded |
| Dead neurons | < 5% | Check thresholds |
| Teacher-Student corr | > 0.9 | Check alignment |

---

## 10. Checkpoint Strategy

### Atomic Writes
```python
import os

def save_checkpoint(state, path):
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.rename(tmp_path, path)  # Atomic on same filesystem
```

### Complete State
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),  # Don't forget!
    'step': step,
    'best_ppl': best_ppl,
}
```

### Resume Verification
```python
# Test: Train N steps, save, resume, train M more
# Compare to: Train N+M continuous
# Should be identical (or statistically indistinguishable)
```

---

## 11. Failure Detection and Recovery

### NaN Detection Hook
```python
def check_nan_hook(module, grad_input, grad_output):
    for g in grad_input:
        if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
            raise RuntimeError(f"NaN/Inf gradient in {module}")

# Register on critical layers
model.attention.register_full_backward_hook(check_nan_hook)
```

### Rollback on Loss Spike
```python
# Detect sudden loss increase
if loss > 3 * loss_moving_avg:
    print("Loss spike detected! Rolling back...")
    model.load_state_dict(last_good_checkpoint)
    for g in optimizer.param_groups:
        g['lr'] *= 0.5  # Reduce LR
```

---

## 12. Distributed Training (DDP vs FSDP)

### Decision Matrix
| Scenario | Teacher | Student | Strategy |
|----------|---------|---------|----------|
| Both fit in VRAM | Any | Any | DDP |
| Teacher > VRAM | Large | Small | FSDP(Teacher) + DDP(Student) |
| Both > VRAM | Large | Large | FSDP(Both) |

### FSDP for torch.compile
```python
fsdp_model = FSDP(
    model,
    use_orig_params=True,  # Required for torch.compile
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

---

## 13. Caching Teacher Logits (Offline Distillation)

For static datasets (no dynamic augmentation):
```python
# Pre-compute teacher logits
teacher_logits = {}
with torch.no_grad():
    for batch in dataloader:
        ids = batch['input_ids']
        logits = teacher(ids)
        teacher_logits[batch['id']] = logits.cpu()

# Save to disk
torch.save(teacher_logits, 'teacher_logits.pt')

# Training uses cached logits - no teacher forward!
```

**Result**: Training speed limited only by student, not teacher.

---

## 14. Eval Interval Tuning

### Trade-off
| eval_interval | Evaluations | Overhead |
|---------------|-------------|----------|
| 100 | 30 (for 3000 steps) | High |
| 300 | 10 | Medium |
| 500 | 6 | Low |

**Recommendation**: Start with 300, increase if training is stable.

---

## Application to ASNN-Goose

### Current v11.1 Settings
| Setting | Value | Notes |
|---------|-------|-------|
| Precision | BF16/FP16 + AMP | GradScaler active |
| eval_interval | 300 | Reduced from 100 |
| Teacher context | `torch.no_grad()` | Not inference_mode! |
| use_cache | False | Disabled for teacher |
| Optimizer | AdamW (fused) | Fastest |
| Accumulation | 2 steps | Effective batch 16 |

### Lessons Learned
1. `torch.inference_mode()` breaks learnable temperature
2. `eval_interval=100` caused 36 min training (now 300 → ~24 min expected)
3. `teacher.config.use_cache = False` required for distillation
4. Always instantiate collectors before distill function

### Performance Targets
| Metric | v11.1 (broken) | v11.1 (fixed) | v10 |
|--------|----------------|---------------|-----|
| PPL | 1318.5 | <530 | 526.4 |
| Time | 36.9 min | ~24 min | 26.3 min |
| Temperature | 10.0 (stuck) | ~2.0-3.0 | N/A |
