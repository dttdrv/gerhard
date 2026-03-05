# =============================================================================
# cell 7: v13 POCL (Progressive Overload Curriculum Learning)
# =============================================================================
# Reference: "POCL: Progressive Overload Curriculum Learning" (2025)
# arXiv:2506.05695

# -----------------------------------------------------------------------------
# Sample Difficulty Scoring
# -----------------------------------------------------------------------------
def compute_sample_difficulty(student, teacher, dataloader, device, max_batches=50):
    """
    Compute difficulty scores for each sample using student-teacher divergence.

    Difficulty = average (CE loss + KL divergence) per sample.
    Higher score = harder sample for the student.

    Uses a small pre-trained student to get meaningful gradients.

    Args:
        student: Student model (should be briefly pre-trained)
        teacher: Teacher model (frozen)
        dataloader: Training data loader
        device: Compute device
        max_batches: Limit batches for efficiency

    Returns:
        Dict with sample indices and difficulty scores
    """
    student.eval()
    teacher.eval()

    all_difficulties = []
    all_indices = []
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            ids = batch[0].to(device, non_blocking=True)
            batch_size = ids.size(0)

            # Get logits
            s_logits = student(ids)
            t_logits = teacher(ids).logits

            # Per-sample difficulty (average over sequence)
            # 1. Cross-entropy with teacher as target
            s_probs = F.softmax(s_logits, dim=-1)
            t_probs = F.softmax(t_logits, dim=-1)

            # KL divergence per sample
            kl_div = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                t_probs,
                reduction='none'
            ).sum(dim=-1).mean(dim=-1)  # [batch_size]

            # Cross-entropy per sample (using teacher hard targets)
            t_tokens = t_logits.argmax(dim=-1)
            ce_loss = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)),
                t_tokens.view(-1),
                reduction='none'
            ).view(batch_size, -1).mean(dim=-1)  # [batch_size]

            # Combined difficulty
            difficulty = kl_div + ce_loss  # [batch_size]

            all_difficulties.extend(difficulty.cpu().tolist())
            all_indices.extend(range(sample_idx, sample_idx + batch_size))
            sample_idx += batch_size

            if batch_idx % 10 == 0:
                print(f"  Scoring batch {batch_idx+1}/{max_batches}...")

    student.train()

    return {
        'indices': all_indices,
        'difficulties': all_difficulties,
        'num_samples': len(all_indices)
    }


# -----------------------------------------------------------------------------
# Data Partitioning by Difficulty
# -----------------------------------------------------------------------------
def partition_by_difficulty(difficulties_dict, n_stages=3):
    """
    Partition data into stages by difficulty (easy -> hard).

    Stage 1: Easiest 33%
    Stage 2: Easiest 66% (includes stage 1)
    Stage 3: All 100% (includes stages 1+2)

    Args:
        difficulties_dict: Output from compute_sample_difficulty()
        n_stages: Number of stages (default 3)

    Returns:
        List of index lists, one per stage (cumulative)
    """
    indices = difficulties_dict['indices']
    difficulties = difficulties_dict['difficulties']

    # Sort by difficulty (ascending = easy first)
    sorted_pairs = sorted(zip(indices, difficulties), key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sorted_pairs]

    n = len(sorted_indices)
    stage_indices = []

    for stage in range(n_stages):
        # Cumulative: stage 1 = 33%, stage 2 = 66%, stage 3 = 100%
        end_idx = int(n * (stage + 1) / n_stages)
        stage_indices.append(sorted_indices[:end_idx])

    return stage_indices


# -----------------------------------------------------------------------------
# Brief Pre-training for Difficulty Scoring
# -----------------------------------------------------------------------------
def pretrain_for_difficulty_scoring(student, teacher, train_loader, cfg, device, steps=100):
    """
    Brief pre-training so difficulty scores are meaningful.

    Without pre-training, student predictions are random garbage,
    making all samples appear equally difficult.

    Args:
        student: Student model
        teacher: Teacher model (frozen)
        train_loader: Training data loader
        cfg: Config object
        device: Compute device
        steps: Number of pre-training steps

    Returns:
        Student model (modified in-place)
    """
    print(f"Pre-training student for {steps} steps (for difficulty scoring)...")

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.distill_lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    student.train()
    teacher.eval()

    step = 0
    pbar = tqdm(total=steps, desc='Pre-training')

    for batch in train_loader:
        if step >= steps:
            break

        ids = batch[0].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                t_logits = teacher(ids).logits

            s_logits = student(ids)

            # Simple KL loss (no temperature complexity)
            T = 2.0
            s_log = F.log_softmax(s_logits / T, dim=-1)
            t_prob = F.softmax(t_logits / T, dim=-1)
            loss = F.kl_div(
                s_log.view(-1, s_logits.size(-1)),
                t_prob.view(-1, t_logits.size(-1)),
                reduction='batchmean'
            ) * (T ** 2)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1
        pbar.update(1)
        if step % 20 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    pbar.close()
    print(f"Pre-training complete. Final loss: {loss.item():.3f}")

    return student


# -----------------------------------------------------------------------------
# Get Stage Temperature (Fixed Schedule)
# -----------------------------------------------------------------------------
def get_pocl_temperature(step, total_steps, temp_schedule, n_stages=3):
    """
    Get temperature for current POCL stage.

    Args:
        step: Current training step
        total_steps: Total training steps
        temp_schedule: Tuple of temperatures per stage (e.g., (1.0, 1.5, 2.0))
        n_stages: Number of stages

    Returns:
        Temperature for current stage
    """
    current_stage = get_pocl_stage(step, total_steps, n_stages)
    return temp_schedule[current_stage]


def get_pocl_stage(step, total_steps, n_stages=3):
    """
    Get current POCL stage (0-indexed).

    Uses rounded boundaries to ensure even distribution:
    - 5000 steps, 3 stages: boundaries at 1667, 3333
    - Stage 0: steps 0-1666
    - Stage 1: steps 1667-3332
    - Stage 2: steps 3333-4999
    """
    for i in range(n_stages - 1):
        boundary = round((i + 1) * total_steps / n_stages)
        if step < boundary:
            return i
    return n_stages - 1


# -----------------------------------------------------------------------------
# POCL Unit Tests
# -----------------------------------------------------------------------------
print("="*60)
print("v13 POCL Component Tests")
print("="*60)

# Test 1: Temperature Schedule
print("\n[1] Temperature Schedule Test")
temp_schedule = (1.0, 1.5, 2.0)
total = 5000

t_start = get_pocl_temperature(0, total, temp_schedule)
t_stage1_end = get_pocl_temperature(1666, total, temp_schedule)  # End of stage 1
t_stage2_start = get_pocl_temperature(1667, total, temp_schedule)  # Start of stage 2
t_stage2_end = get_pocl_temperature(3332, total, temp_schedule)
t_stage3 = get_pocl_temperature(4000, total, temp_schedule)

temp_pass = (t_start == 1.0 and t_stage1_end == 1.0 and t_stage2_start == 1.5 and t_stage3 == 2.0)
print(f"  T(0) = {t_start} (should be 1.0)")
print(f"  T(1666) = {t_stage1_end} (should be 1.0, end of stage 1)")
print(f"  T(1667) = {t_stage2_start} (should be 1.5, start of stage 2)")
print(f"  T(4000) = {t_stage3} (should be 2.0, stage 3)")
print(f"  {'PASS' if temp_pass else 'FAIL'}")

# Test 2: Stage Boundaries
print("\n[2] Stage Boundaries Test")
stages = [get_pocl_stage(s, total) for s in [0, 1666, 1667, 3332, 3333, 4999]]
stage_pass = stages == [0, 0, 1, 1, 2, 2]
print(f"  Stages at [0, 1666, 1667, 3332, 3333, 4999]: {stages}")
print(f"  Expected: [0, 0, 1, 1, 2, 2]")
print(f"  {'PASS' if stage_pass else 'FAIL'}")

# Test 3: Partition by Difficulty (mock)
print("\n[3] Partition by Difficulty Test (mock data)")
mock_difficulties = {
    'indices': list(range(9)),
    'difficulties': [0.5, 1.5, 0.3, 2.0, 0.8, 1.2, 2.5, 0.1, 1.8],  # Easy: 7,2,0,4 | Med: 5,1 | Hard: 8,3,6
    'num_samples': 9
}
partitions = partition_by_difficulty(mock_difficulties, n_stages=3)
# After sorting: [7(0.1), 2(0.3), 0(0.5), 4(0.8), 5(1.2), 1(1.5), 8(1.8), 3(2.0), 6(2.5)]
# Stage 1 (33%): indices [7, 2, 0] -> 3 samples
# Stage 2 (66%): indices [7, 2, 0, 4, 5, 1] -> 6 samples
# Stage 3 (100%): all 9 samples
partition_pass = (len(partitions[0]) == 3 and len(partitions[1]) == 6 and len(partitions[2]) == 9)
print(f"  Stage 1 samples: {len(partitions[0])} (should be 3)")
print(f"  Stage 2 samples: {len(partitions[1])} (should be 6)")
print(f"  Stage 3 samples: {len(partitions[2])} (should be 9)")
print(f"  Cumulative check: {partitions[0][0] in partitions[1] and partitions[1][0] in partitions[2]}")
print(f"  {'PASS' if partition_pass else 'FAIL'}")

# Summary
print("\n" + "="*60)
all_pass = temp_pass and stage_pass and partition_pass
print(f"POCL Component Tests: {'ALL PASS' if all_pass else 'SOME FAILED'}")
if not all_pass:
    print("WARNING: Fix failing tests before running training!")
print("="*60)
