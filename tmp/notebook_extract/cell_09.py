# =============================================================================
# cell 9: hidden-state projector + v14.1 FDD layer mapping
# =============================================================================
class HiddenStateProjector(nn.Module):
    """
    Project student hidden states to teacher dimension for alignment.

    student: (B, T, 320) -> (B, T, 768)

    Maps student layers to selected teacher layers.

    NOTE: This is kept for infrastructure but hidden alignment is DISABLED.
    v14 uses FDD with CKA loss which is projector-free.
    """

    def __init__(self, student_dim: int, teacher_dim: int, n_student_layers: int):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(student_dim, teacher_dim, bias=False)
            for _ in range(n_student_layers)
        ])
        for proj in self.projectors:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, student_hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.projectors[layer_idx](student_hidden)


def compute_hidden_alignment_loss(
    teacher_hiddens: List[torch.Tensor],
    student_hiddens: List[torch.Tensor],
    projector: HiddenStateProjector,
    teacher_layers: int = 12,
    student_layers: int = 8
) -> torch.Tensor:
    """
    Compute MSE loss between projected student and teacher hidden states.

    NOTE: This is DISABLED in v14 (hidden_align_weight=0.0).
    v14 uses FDD with CKA loss instead.
    """
    # Map student layers to teacher layers
    teacher_indices = [1, 2, 4, 5, 7, 8, 10, 11]

    total_loss = 0.0
    for s_idx, t_idx in enumerate(teacher_indices):
        if s_idx >= len(student_hiddens) - 1:
            break
        if t_idx >= len(teacher_hiddens):
            break

        s_hidden = student_hiddens[s_idx + 1]
        t_hidden = teacher_hiddens[t_idx]

        s_proj = projector(s_hidden, s_idx)
        total_loss = total_loss + F.mse_loss(s_proj, t_hidden)

    return total_loss / len(teacher_indices)


# Create projector (even if disabled, keeps infrastructure)
projector = HiddenStateProjector(
    student_dim=config.d_model,
    teacher_dim=config.teacher_d_model,
    n_student_layers=config.n_layers
).to(DEVICE)

projector_params = sum(p.numel() for p in projector.parameters())
print(f"hidden-state projector: {projector_params:,} params")
print(f"  student dim: {config.d_model}")
print(f"  teacher dim: {config.teacher_d_model}")
print(f"  student layers: {config.n_layers}")
print(f"  hidden_align_weight: {config.hidden_align_weight}")
print(f"  STATUS: DISABLED (v14 uses FDD with CKA instead)")

# =============================================================================
# v14: Create FDD layer mapping
# =============================================================================
fdd_layer_map = get_fdd_layer_mapping(
    n_student_layers=config.n_layers,
    n_teacher_layers=config.teacher_n_layers,
    n_align_layers=config.fdd_n_align_layers
)

print(f"")
print(f"{config.VERSION} FDD Layer Mapping:")
print(f"  Layer pairs to align: {config.fdd_n_align_layers}")
print(f"  Mapping: {fdd_layer_map}")
print(f"  Strategy: Align early/middle/late semantic layers")
