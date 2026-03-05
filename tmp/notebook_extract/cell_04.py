# =============================================================================
# cell 6: v13 PROPER CTKD Implementation
# =============================================================================
# References:
# - CTKD Paper: https://arxiv.org/abs/2211.16231
# - GRL Origin: Ganin & Lempitsky (2015) https://arxiv.org/abs/1409.7495
# - torch-gradient-reversal: https://pypi.org/project/torch-gradient-reversal/

# -----------------------------------------------------------------------------
# GradientReversalFunction (Custom Autograd)
# -----------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    Forward: Identity mapping f(x) = x
    Backward: Negates gradient ∂f/∂x = -λ * grad
    
    This enables min-max optimization in a single backward pass:
    - Student minimizes loss (normal gradients)
    - Temperature maximizes loss (reversed gradients via GRL)
    
    Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation"
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        # Save lambda for backward pass
        ctx.lambda_ = lambda_
        # Forward is identity (must clone to avoid in-place issues)
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward negates and scales gradient
        # Returns: (grad for x, grad for lambda_)
        # lambda_ is a hyperparameter, doesn't need gradient
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Module wrapper for GradientReversalFunction.
    
    Usage:
        grl = GradientReversalLayer()
        grl.set_lambda(0.5)  # Set adversarial strength
        y = grl(x)  # Forward: y = x, Backward: grad_x = -0.5 * grad_y
    """
    
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
    
    def set_lambda(self, lambda_: float):
        """Set the adversarial strength (0 = no reversal, 1 = full reversal)."""
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# -----------------------------------------------------------------------------
# Lambda Scheduler (Cosine with Warmup)
# -----------------------------------------------------------------------------
def get_lambda(step: int, total_steps: int, lambda_max: float = 1.0, 
               warmup_ratio: float = 0.2) -> float:
    """
    Cosine schedule for adversarial strength λ.
    
    - During warmup (first warmup_ratio of training): λ = 0
      Temperature learns freely to find reasonable range
    - After warmup: λ increases from 0 to lambda_max via cosine
      Gradually increases adversarial pressure
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        lambda_max: Maximum λ value (default 1.0 = full reversal)
        warmup_ratio: Fraction of training for warmup (default 0.2 = 20%)
    
    Returns:
        Current λ value in [0, lambda_max]
    """
    warmup_steps = int(total_steps * warmup_ratio)
    
    if step < warmup_steps:
        return 0.0
    
    # Progress after warmup [0, 1]
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    # Cosine increase from 0 to lambda_max
    lambda_ = lambda_max * (1 - math.cos(math.pi * progress)) / 2
    return lambda_


# -----------------------------------------------------------------------------
# CTKDTemperature (Proper Implementation with GRL)
# -----------------------------------------------------------------------------
class CTKDTemperature(nn.Module):
    """
    Curriculum Temperature for Knowledge Distillation (CTKD).
    
    Key features:
    1. Adversarial learning via Gradient Reversal Layer
    2. Sigmoid bounding for smooth gradients at boundaries
    3. Proper initialization via logit transform
    
    The temperature module tries to MAXIMIZE the KL loss (via GRL),
    finding the "hardest" temperature for the student.
    The student tries to MINIMIZE the KL loss.
    This adversarial game leads to optimal curriculum difficulty.
    
    Reference: Li et al., "Curriculum Temperature for Knowledge Distillation", AAAI 2023
    """
    
    def __init__(self, tau_min: float = 1.0, tau_max: float = 5.0, init: float = 2.0):
        """
        Args:
            tau_min: Minimum temperature (default 1.0)
            tau_max: Maximum temperature (default 5.0, conservative for LLMs)
            init: Initial temperature (default 2.0)
        """
        super().__init__()
        self.tau_min = tau_min
        self.tau_range = tau_max - tau_min
        
        # Initialize raw parameter so sigmoid outputs init value
        # sigmoid(raw) = (init - tau_min) / tau_range
        # raw = logit((init - tau_min) / tau_range)
        init_normalized = (init - tau_min) / self.tau_range
        init_normalized = max(0.01, min(0.99, init_normalized))  # Clamp for numerical stability
        init_raw = math.log(init_normalized / (1 - init_normalized))  # logit function
        
        self.raw_temp = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        self.grl = GradientReversalLayer()
        
        # Store config for logging
        self.tau_min_val = tau_min
        self.tau_max_val = tau_max
        self.init_val = init
    
    def forward(self, lambda_: float) -> torch.Tensor:
        """
        Compute temperature with GRL applied.
        
        Args:
            lambda_: Current adversarial strength from scheduler
        
        Returns:
            Temperature τ ∈ [tau_min, tau_max]
        """
        # Set GRL strength
        self.grl.set_lambda(lambda_)
        
        # Apply GRL to raw parameter (this is where gradient reversal happens!)
        raw_reversed = self.grl(self.raw_temp)
        
        # Sigmoid bounding (smooth, differentiable at boundaries)
        tau = self.tau_min + self.tau_range * torch.sigmoid(raw_reversed)
        
        return tau
    
    def get_temperature(self) -> float:
        """Get current temperature without GRL (for logging/display)."""
        with torch.no_grad():
            tau = self.tau_min + self.tau_range * torch.sigmoid(self.raw_temp)
            return tau.item()
    
    def get_raw_value(self) -> float:
        """Get raw (unbounded) parameter value (for debugging)."""
        return self.raw_temp.item()


# -----------------------------------------------------------------------------
# Legacy Classes (kept for backward compatibility)
# -----------------------------------------------------------------------------
class LearnableTemperature(nn.Module):
    """
    DEPRECATED: Simple learnable temperature WITHOUT GRL.
    Kept for backward compatibility. Use CTKDTemperature instead.
    
    WARNING: This class caused temperature runaway in v12!
    """
    
    def __init__(self, init: float = 2.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init)))
    
    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(1.0, 10.0)
    
    def get_temperature(self) -> float:
        return self.forward().item()


class ChannelWiseTernarySpike(nn.Module):
    """
    Per-channel learnable alpha and amplitude for ternary spikes.
    DISABLED in v13 due to structural symmetry issue with RWKV.
    """
    
    def __init__(self, d_model: int, alpha_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(d_model) * alpha_init)
        self.amplitude = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_abs_mean = x.abs().mean(dim=(0, 1), keepdim=True)
        threshold = self.alpha * x_abs_mean
        threshold = threshold.clamp(min=0.01, max=10.0)
        
        with torch.no_grad():
            pos_mask = (x > threshold).float()
            neg_mask = (x < -threshold).float()
            spike_signs = pos_mask - neg_mask
        
        spikes = self.amplitude * spike_signs
        return spikes + (x - x.detach())
    
    def get_amplitude(self) -> float:
        return self.amplitude.mean().item()
    
    def get_stats(self) -> dict:
        return {
            'alpha_mean': self.alpha.mean().item(),
            'alpha_std': self.alpha.std().item(),
            'amplitude_mean': self.amplitude.mean().item(),
            'amplitude_std': self.amplitude.std().item(),
        }

    def get_amplitude_stats(self) -> dict:
        return {
            'mean': self.amplitude.mean().item(),
            'std': self.amplitude.std().item(),
            'min': self.amplitude.min().item(),
            'max': self.amplitude.max().item(),
        }


class TrainableTernarySpike(nn.Module):
    """Original trainable ternary spike with scalar amplitude (from v8)."""

    def __init__(
        self,
        alpha: float = 1.0,
        threshold_mix: float = 0.35,
        surrogate_temp: float = 0.10,
    ):
        super().__init__()
        self.alpha = alpha
        self.threshold_mix = threshold_mix
        self.surrogate_temp = surrogate_temp
        self.amplitude = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        token_scale = x.abs().mean(dim=-1, keepdim=True)
        channel_scale = x.abs().mean(dim=(0, 1), keepdim=True)
        threshold = self.alpha * (
            (1.0 - self.threshold_mix) * token_scale + self.threshold_mix * channel_scale
        )
        threshold = threshold.clamp(min=0.01, max=10.0)

        with torch.no_grad():
            pos_mask = (x > threshold).float()
            neg_mask = (x < -threshold).float()
            spike_signs = pos_mask - neg_mask

        amplitude = self.amplitude.clamp(min=0.25, max=4.0)
        spikes = amplitude * spike_signs
        spikes = spikes + (x - x.detach())

        if return_aux:
            soft_activity = torch.sigmoid((x.abs() - threshold) / self.surrogate_temp)
            return spikes, {
                'threshold': threshold.detach(),
                'soft_activity': soft_activity,
            }
        return spikes

    def get_amplitude(self) -> float:
        return self.amplitude.item()


def get_stage_params(step: int, total_steps: int = 3000) -> dict:
    """Progressive training stages (POCL) - kept for infrastructure."""
    if step < total_steps * 0.4:
        return {'stage': 1, 'temp_target': 1.0, 'align_mult': 0.0, 'alpha': 0.9}
    elif step < total_steps * 0.7:
        return {'stage': 2, 'temp_target': 1.5, 'align_mult': 0.5, 'alpha': 0.7}
    else:
        return {'stage': 3, 'temp_target': 2.0, 'align_mult': 1.0, 'alpha': 0.5}


# -----------------------------------------------------------------------------
# Unit Tests for CTKD Components
# -----------------------------------------------------------------------------
print("="*60)
print("v13 CTKD Component Tests")
print("="*60)

# Test 1: GRL Gradient Reversal
print("\n[1] GRL Gradient Reversal Test")
grl = GradientReversalLayer()
grl.set_lambda(1.0)
x_test = torch.tensor([2.0], requires_grad=True)
y_test = grl(x_test)
loss_test = y_test.sum()
loss_test.backward()
expected_grad = -1.0  # GRL should negate: 1 * -1.0 = -1.0
actual_grad = x_test.grad.item()
grl_pass = abs(actual_grad - expected_grad) < 1e-6
print(f"  Input grad without GRL would be: +1.0")
print(f"  Input grad with GRL (λ=1.0): {actual_grad:.4f}")
print(f"  Expected: {expected_grad:.4f}")
print(f"  {'PASS' if grl_pass else 'FAIL'}")
del x_test, y_test, loss_test

# Test 2: Lambda Schedule
print("\n[2] Lambda Schedule Test")
total = 3000
warmup = 0.2
# During warmup
lambda_0 = get_lambda(0, total, warmup_ratio=warmup)
lambda_500 = get_lambda(500, total, warmup_ratio=warmup)
# After warmup
lambda_1500 = get_lambda(1500, total, warmup_ratio=warmup)
lambda_2999 = get_lambda(2999, total, warmup_ratio=warmup)

warmup_pass = lambda_0 == 0.0 and lambda_500 == 0.0
increase_pass = 0 < lambda_1500 < lambda_2999 <= 1.0
lambda_pass = warmup_pass and increase_pass
print(f"  λ(0) = {lambda_0:.4f} (should be 0.0)")
print(f"  λ(500) = {lambda_500:.4f} (should be 0.0, still in warmup)")
print(f"  λ(1500) = {lambda_1500:.4f} (should be > 0)")
print(f"  λ(2999) = {lambda_2999:.4f} (should be ≈ 1.0)")
print(f"  {'PASS' if lambda_pass else 'FAIL'}")

# Test 3: Temperature Bounds
print("\n[3] Temperature Bounds Test")
temp_module = CTKDTemperature(tau_min=1.0, tau_max=5.0, init=2.0).to(DEVICE)
init_temp = temp_module.get_temperature()

# Force extreme raw values
with torch.no_grad():
    temp_module.raw_temp.fill_(-100)
    tau_low = temp_module.get_temperature()
    
    temp_module.raw_temp.fill_(100)
    tau_high = temp_module.get_temperature()
    
    # Reset to init
    init_normalized = (2.0 - 1.0) / 4.0
    init_raw = math.log(init_normalized / (1 - init_normalized))
    temp_module.raw_temp.fill_(init_raw)

bounds_pass = (1.0 <= tau_low <= 1.01) and (4.99 <= tau_high <= 5.0) and (1.9 <= init_temp <= 2.1)
print(f"  Initial temp: {init_temp:.4f} (should be ≈ 2.0)")
print(f"  Min bound test: {tau_low:.4f} (should be ≈ 1.0)")
print(f"  Max bound test: {tau_high:.4f} (should be ≈ 5.0)")
print(f"  {'PASS' if bounds_pass else 'FAIL'}")

# Test 4: End-to-End Gradient Flow
print("\n[4] End-to-End Gradient Flow Test")
temp_module_test = CTKDTemperature(tau_min=1.0, tau_max=5.0, init=2.0).to(DEVICE)
lambda_test = 0.5

# Simulate forward pass
T = temp_module_test(lambda_test)
fake_kl_loss = T * 2.0  # Gradient ∂L/∂T = 2.0

# Without GRL: optimizer would DECREASE T to minimize loss
# With GRL: optimizer should INCREASE T (because grad is reversed)
fake_kl_loss.backward()

raw_grad = temp_module_test.raw_temp.grad.item()
# The gradient through sigmoid and GRL should be negative (reversed)
# Original: ∂L/∂raw > 0 would decrease raw
# With GRL: ∂L/∂raw < 0 (negated), so optimizer increases raw
grad_flow_pass = raw_grad < 0  # Should be negative due to GRL
print(f"  Loss = T * 2.0, so ∂L/∂T = 2.0 (positive)")
print(f"  Without GRL: raw_grad would be positive (decrease T)")
print(f"  With GRL (λ=0.5): raw_grad = {raw_grad:.4f} (should be negative)")
print(f"  {'PASS' if grad_flow_pass else 'FAIL'}")
del temp_module_test

# Summary
print("\n" + "="*60)
all_pass = grl_pass and lambda_pass and bounds_pass and grad_flow_pass
print(f"CTKD Component Tests: {'ALL PASS' if all_pass else 'SOME FAILED'}")
if not all_pass:
    print("WARNING: Fix failing tests before running training!")
print("="*60)
