# =============================================================================
# cell 14: create student model and projector (v14 - with compile)
# =============================================================================
print("creating student model (v14 - v14 baseline + POCL)...")

student = StudentSpikingGoose(config, use_checkpointing=USE_GRADIENT_CHECKPOINTING).to(DEVICE)
student_params = sum(p.numel() for p in student.parameters())

# v14: create projector (even if not used, for infrastructure preservation)
projector = HiddenStateProjector(
    student_dim=config.d_model,
    teacher_dim=config.teacher_d_model,
    n_student_layers=config.n_layers
).to(DEVICE)
projector_params = sum(p.numel() for p in projector.parameters())

compression_ratio = teacher_params / student_params

print(f"student: asnn-goose v14 ({student_params:,} params)")
print(f"projector: ({projector_params:,} params)")
print(f"compression ratio: {compression_ratio:.1f}x")
print(f"")
print(f"{config.VERSION} architecture:")
print(f"  d_model: {config.d_model}")
print(f"  n_layers: {config.n_layers}")
print(f"  params: ~{student_params // 1_000_000}M")
print(f"")

# v14: compile model if available and enabled
compile_success = False
if USE_TORCH_COMPILE and TORCH_COMPILE_AVAILABLE:
    try:
        print("compiling student model with torch.compile...")
        # Use the compile() method as recommended by PyTorch docs
        student = torch.compile(student, mode='reduce-overhead')
        compile_success = True
        print("compilation successful!")
    except Exception as e:
        print(f"torch.compile failed: {e}")
        print("continuing without compilation")
else:
    print(f"torch.compile skipped (USE_TORCH_COMPILE={USE_TORCH_COMPILE}, available={TORCH_COMPILE_AVAILABLE})")

print(f"")
print(f"speedups active:")
print(f"  gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
print(f"  torch.compile: {compile_success}")
print(f"  accumulation_steps: {config.accumulation_steps}")
