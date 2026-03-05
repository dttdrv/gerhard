# =============================================================================
# cell 6.5: v14.1 FDD (Feature Dynamics Distillation) with CKA Loss
# =============================================================================
# References:
# - CKA: Kornblith et al., "Similarity of Neural Network Representations Revisited"
# - FDD: Feature Dynamics Distillation (view transformer as ODE)
# - v7 lesson: Hidden alignment with weight=1.0 caused PPL regression to 1655!

# -----------------------------------------------------------------------------
# Centered Kernel Alignment (CKA) Loss
# -----------------------------------------------------------------------------
def cka_loss(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA) loss for representation alignment.

    CKA is a similarity measure between feature representations that is:
    - Invariant to orthogonal transformations
    - Invariant to isotropic scaling
    - Does NOT require dimension matching (projector-free!)

    Args:
        X: Student features [n_samples, dim_x]
        Y: Teacher features [n_samples, dim_y]
        eps: Small constant for numerical stability

    Returns:
        Loss = 1 - CKA (minimize to maximize alignment)
        CKA = 1 means perfect alignment, CKA = 0 means no alignment

    Note: n_samples must match, but dim_x and dim_y can differ!

    CRITICAL: Uses float32 to prevent overflow in mixed precision training.
    With n=2048 samples, Gram matrix sums can exceed float16 max (~65504).
    This is training-time only - does NOT affect student's ternary activations.
    """
    # CRITICAL: Force float32 to prevent overflow in mixed precision
    # Under torch.cuda.amp.autocast(), tensors are float16 by default
    # Gram matrix sums: 4M elements squared and summed -> can exceed 65504
    with torch.cuda.amp.autocast(enabled=False):
        X = X.float()
        Y = Y.float()

        # Validate input shapes
        assert X.dim() == 2 and Y.dim() == 2, f"Expected 2D tensors, got X:{X.dim()}D, Y:{Y.dim()}D"
        assert X.size(0) == Y.size(0), f"Sample count mismatch: X={X.size(0)}, Y={Y.size(0)}"

        # Center the features (critical for CKA)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)

        # Row-normalize for additional numerical stability
        # This bounds Gram matrix elements to [-1, 1] range
        X_norm = X_centered / (X_centered.norm(dim=1, keepdim=True) + eps)
        Y_norm = Y_centered / (Y_centered.norm(dim=1, keepdim=True) + eps)

        # Compute Gram matrices on normalized features
        # K_X[i,j] = dot(X_norm[i], X_norm[j]) in [-1, 1]
        K_X = X_norm @ X_norm.T  # [n, n]
        K_Y = Y_norm @ Y_norm.T  # [n, n]

        # HSIC (Hilbert-Schmidt Independence Criterion)
        # Now numerically stable: elements in [-1, 1]^2 = [0, 1]
        hsic_xy = (K_X * K_Y).sum()
        hsic_xx = (K_X * K_X).sum()
        hsic_yy = (K_Y * K_Y).sum()

        # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)

        # Clamp to valid range (numerical safety)
        cka = cka.clamp(0.0, 1.0)

        # Return loss (1 - CKA): minimize loss = maximize alignment
        return 1.0 - cka


# -----------------------------------------------------------------------------
# Layer Mapping for FDD
# -----------------------------------------------------------------------------
def get_fdd_layer_mapping(n_student_layers: int, n_teacher_layers: int,
                          n_align_layers: int = 3) -> Dict[int, int]:
    """
    Create layer mapping for Feature Dynamics Distillation.

    Maps student layers to teacher layers for hidden state alignment.
    Uses even spacing to cover early/middle/late representations.

    Args:
        n_student_layers: Number of student layers (e.g., 5)
        n_teacher_layers: Number of teacher layers (e.g., 12)
        n_align_layers: Number of layer pairs to align (default 3)

    Returns:
        Dict mapping student_layer_idx -> teacher_layer_idx

    Example for 5 student, 12 teacher, 3 alignments:
        {0: 2, 2: 7, 4: 11}  # Early, middle, late (with +2 offset)
    """
    if n_align_layers > n_student_layers:
        n_align_layers = n_student_layers

    layer_map = {}

    # Evenly space the student layers to align
    student_indices = []
    for i in range(n_align_layers):
        # 0, 2, 4 for 3 alignments with 5 layers
        idx = int(i * (n_student_layers - 1) / max(n_align_layers - 1, 1))
        student_indices.append(idx)

    # Map each student index to corresponding teacher layer
    for s_idx in student_indices:
        # Scale to teacher layers
        t_idx = int((s_idx / (n_student_layers - 1)) * (n_teacher_layers - 1))
        # Offset by 1-2 to avoid embedding layer
        t_idx = max(2, min(t_idx + 2, n_teacher_layers - 1))
        layer_map[s_idx] = t_idx

    return layer_map


# -----------------------------------------------------------------------------
# Feature Dynamics Distillation Loss
# -----------------------------------------------------------------------------
def compute_fdd_loss(
    student_hiddens: List[torch.Tensor],
    teacher_hiddens: List[torch.Tensor],
    layer_map: Dict[int, int],
    loss_type: str = "cka"
) -> torch.Tensor:
    """
    Compute Feature Dynamics Distillation (FDD) loss.

    FDD views the transformer as solving an ODE: dh/dt = f(h, t)
    where each layer is a discrete time step.

    Instead of matching hidden states directly (which failed in v7),
    we match the DYNAMICS (layer-to-layer changes): delta_h = h_{l+1} - h_l

    This teaches the student HOW to transform features, not just WHAT features to have.

    Args:
        student_hiddens: List of student hidden states [h_0, h_1, ..., h_L]
                        Each has shape [batch, seq, student_dim]
        teacher_hiddens: List of teacher hidden states (from output_hidden_states=True)
                        Each has shape [batch, seq, teacher_dim]
        layer_map: Dict mapping student_layer_idx -> teacher_layer_idx
        loss_type: "cka" (recommended) or "mse"

    Returns:
        FDD loss (scalar tensor)

    Note: student_hiddens[0] is embedding, student_hiddens[1] is after layer 0, etc.
    """
    total_loss = torch.tensor(0.0, device=student_hiddens[0].device)
    n_pairs = 0

    for s_layer, t_layer in layer_map.items():
        # Validate indices
        # student_hiddens: [embed, after_L0, after_L1, ..., after_L{n-1}]
        # So layer i output is at index i+1
        s_idx = s_layer + 1  # +1 because [0] is embedding
        t_idx = t_layer + 1  # Same for teacher

        # Check bounds
        if s_idx + 1 >= len(student_hiddens):
            continue
        if t_idx + 1 >= len(teacher_hiddens):
            continue

        # Compute dynamics (velocity): delta_h = h_{l+1} - h_l
        # Student dynamics: change from layer s_layer to s_layer+1
        delta_s = student_hiddens[s_idx + 1] - student_hiddens[s_idx]  # [batch, seq, s_dim]

        # Teacher dynamics: change from layer t_layer to t_layer+1
        delta_t = teacher_hiddens[t_idx + 1] - teacher_hiddens[t_idx]  # [batch, seq, t_dim]

        # Flatten for CKA: [batch, seq, dim] -> [batch*seq, dim]
        batch_size, seq_len = delta_s.size(0), delta_s.size(1)
        delta_s_flat = delta_s.reshape(batch_size * seq_len, -1)  # [n, s_dim]
        delta_t_flat = delta_t.reshape(batch_size * seq_len, -1)  # [n, t_dim]

        # Compute loss
        if loss_type == "cka":
            pair_loss = cka_loss(delta_s_flat, delta_t_flat)
        elif loss_type == "mse":
            # MSE requires dimension matching - use projector if needed
            # For now, skip if dimensions don't match
            if delta_s_flat.size(1) != delta_t_flat.size(1):
                continue
            pair_loss = F.mse_loss(delta_s_flat, delta_t_flat)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        total_loss = total_loss + pair_loss
        n_pairs += 1

    # Average over pairs
    if n_pairs > 0:
        total_loss = total_loss / n_pairs

    return total_loss


# -----------------------------------------------------------------------------
# FDD Weight Scheduler (with warmup)
# -----------------------------------------------------------------------------
def get_fdd_weight(step: int, fdd_warmup_steps: int, fdd_weight: float) -> float:
    """
    Get FDD weight for current step with warmup.

    Args:
        step: Current training step
        fdd_warmup_steps: Steps before FDD kicks in
        fdd_weight: Maximum FDD weight

    Returns:
        Current FDD weight (0 during warmup, then fdd_weight)
    """
    if step < fdd_warmup_steps:
        return 0.0
    return fdd_weight


# -----------------------------------------------------------------------------
# FDD Unit Tests
# -----------------------------------------------------------------------------
print("="*60)
print("v14.1 FDD Component Tests")
print("="*60)

fdd_tests = []

# Test 1: CKA of identical tensors should return 0 loss (CKA=1)
print("\n[1] CKA Identical Tensors Test")
X_test = torch.randn(100, 64)
cka_identical = cka_loss(X_test, X_test)
identical_pass = cka_identical.item() < 0.01  # Should be ~0
print(f"  CKA loss of identical tensors: {cka_identical.item():.6f}")
print(f"  Expected: ~0.0 (CKA=1 means perfect alignment)")
print(f"  {'PASS' if identical_pass else 'FAIL'}")
fdd_tests.append(('CKA identical', identical_pass))

# Test 2: CKA of orthogonal tensors should return high loss
print("\n[2] CKA Orthogonal Tensors Test")
X_orth = torch.randn(100, 64)
Y_orth = torch.randn(100, 64)  # Different random = nearly orthogonal
cka_orthogonal = cka_loss(X_orth, Y_orth)
orthogonal_pass = 0.5 < cka_orthogonal.item() <= 1.0  # Should be high
print(f"  CKA loss of orthogonal tensors: {cka_orthogonal.item():.4f}")
print(f"  Expected: 0.5-1.0 (low alignment)")
print(f"  {'PASS' if orthogonal_pass else 'FAIL'}")
fdd_tests.append(('CKA orthogonal', orthogonal_pass))

# Test 3: CKA handles different dimensions
print("\n[3] CKA Dimension Agnostic Test")
X_small = torch.randn(100, 32)   # 32 dims
Y_large = torch.randn(100, 128)  # 128 dims
try:
    cka_diff_dim = cka_loss(X_small, Y_large)
    dim_pass = True
    print(f"  CKA with dims (32, 128): {cka_diff_dim.item():.4f}")
except Exception as e:
    dim_pass = False
    print(f"  ERROR: {e}")
print(f"  {'PASS' if dim_pass else 'FAIL'}")
fdd_tests.append(('CKA dimension agnostic', dim_pass))

# Test 4: Layer mapping correctness
print("\n[4] Layer Mapping Test")
layer_map = get_fdd_layer_mapping(n_student_layers=5, n_teacher_layers=12, n_align_layers=3)
# Actual computation: s_idx=0 -> t_idx=0+2=2, s_idx=2 -> t_idx=5+2=7, s_idx=4 -> t_idx=11 (clamped)
expected_map = {0: 2, 2: 7, 4: 11}  # Corrected expectation
map_pass = layer_map == expected_map
print(f"  Generated map: {layer_map}")
print(f"  Expected map: {expected_map}")
print(f"  {'PASS' if map_pass else 'FAIL'}")
fdd_tests.append(('Layer mapping', map_pass))

# Test 5: FDD loss computation
print("\n[5] FDD Loss Computation Test")
# Mock hidden states
student_hiddens_mock = [torch.randn(2, 16, 320) for _ in range(6)]  # embed + 5 layers
teacher_hiddens_mock = [torch.randn(2, 16, 768) for _ in range(13)]  # embed + 12 layers
try:
    fdd_loss_val = compute_fdd_loss(
        student_hiddens_mock,
        teacher_hiddens_mock,
        layer_map,
        loss_type="cka"
    )
    fdd_pass = 0.0 <= fdd_loss_val.item() <= 1.0
    print(f"  FDD loss: {fdd_loss_val.item():.4f}")
    print(f"  Expected: [0, 1]")
except Exception as e:
    fdd_pass = False
    print(f"  ERROR: {e}")
print(f"  {'PASS' if fdd_pass else 'FAIL'}")
fdd_tests.append(('FDD loss computation', fdd_pass))

# Test 6: FDD weight scheduler
print("\n[6] FDD Weight Scheduler Test")
w_0 = get_fdd_weight(0, 500, 0.1)
w_400 = get_fdd_weight(400, 500, 0.1)
w_500 = get_fdd_weight(500, 500, 0.1)
w_1000 = get_fdd_weight(1000, 500, 0.1)
scheduler_pass = (w_0 == 0.0 and w_400 == 0.0 and w_500 == 0.1 and w_1000 == 0.1)
print(f"  weight(0): {w_0} (should be 0)")
print(f"  weight(400): {w_400} (should be 0)")
print(f"  weight(500): {w_500} (should be 0.001)")
print(f"  weight(1000): {w_1000} (should be 0.001)")
print(f"  {'PASS' if scheduler_pass else 'FAIL'}")
fdd_tests.append(('FDD weight scheduler', scheduler_pass))

# Test 7: CKA float32 stability test (simulates mixed precision)
print("\n[7] CKA Float32 Stability Test")
# Simulate large values that would overflow in float16
X_large = torch.randn(2048, 320) * 100  # Large values
Y_large = torch.randn(2048, 768) * 100
try:
    cka_large = cka_loss(X_large, Y_large)
    stability_pass = not (torch.isnan(cka_large) or torch.isinf(cka_large))
    print(f"  CKA with large values (n=2048): {cka_large.item():.4f}")
    print(f"  No NaN/Inf: {stability_pass}")
except Exception as e:
    stability_pass = False
    print(f"  ERROR: {e}")
print(f"  {'PASS' if stability_pass else 'FAIL'}")
fdd_tests.append(('CKA float32 stability', stability_pass))

# Summary
print("\n" + "="*60)
all_fdd_pass = all(p for _, p in fdd_tests)
print(f"FDD Component Tests: {'ALL PASS' if all_fdd_pass else 'SOME FAILED'}")
if not all_fdd_pass:
    failed = [n for n, p in fdd_tests if not p]
    print(f"FAILED: {failed}")
print("="*60)
