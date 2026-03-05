# =============================================================================
# cell 17: distillation training loop (v14.1.1 - FDD+CTKD+HardCE)
# =============================================================================
def get_spike_semantic_weight(step: int, warmup_steps: int, max_weight: float) -> float:
    if step < warmup_steps:
        return 0.0
    ramp_steps = max(warmup_steps, 1)
    ramp = min(1.0, (step - warmup_steps) / ramp_steps)
    return max_weight * ramp


def build_teacher_ternary_target(teacher_hidden: torch.Tensor, threshold_scale: float) -> torch.Tensor:
    centered = teacher_hidden - teacher_hidden.mean(dim=-1, keepdim=True)
    threshold = threshold_scale * centered.abs().mean(dim=-1, keepdim=True)
    pos = centered > threshold
    neg = centered < -threshold
    target = torch.zeros_like(centered)
    target = torch.where(pos, torch.ones_like(target), target)
    target = torch.where(neg, -torch.ones_like(target), target)
    return target


def distill_v14(teacher, student, projector, train_loader, val_loader, cfg, device,
                hw_stats, spike_stats, fdd_layer_map):
    """
    v14 distillation with FDD (Feature Dynamics Distillation) + CTKD.

    Key innovations:
    1. FDD: Align layer dynamics (delta_h) using CKA loss
    2. CTKD: Adversarial temperature learning (proven in v12.1, v13.1)
    3. Safety: FDD kill-switch if PPL regresses

    References:
    - CKA: Kornblith et al., "Similarity of Neural Network Representations"
    - CTKD: https://arxiv.org/abs/2211.16231
    - FDD: Feature Dynamics Distillation (view transformer as ODE)
    """
    training_logs = {
        'loss_history': [],
        'kl_loss_history': [],
        'ce_loss_history': [],  # v14.1: hard distillation
        'fdd_loss_history': [],
        'spike_sem_loss_history': [],
        'align_loss_history': [],
        'ppl_history': [],
        'lr_history': [],
        'temp_history': [],
        'lambda_history': [],
        'fdd_weight_history': [],
        'spike_sem_weight_history': [],
        'stage_history': [],
        'stage_transitions': [],
        'early_stopped': False,
        'early_stop_step': None,
        'fdd_killed': False,
        'fdd_kill_step': None,
    }

    # =========================================================================
    # v14: CTKD Temperature (same as v12.1, v13.1)
    # =========================================================================
    if cfg.use_ctkd:
        temp_module = CTKDTemperature(
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            init=cfg.tau_init
        ).to(device)
        print(f"{config.VERSION}: CTKD with Gradient Reversal Layer")
        print(f"     Temperature bounds: [{cfg.tau_min}, {cfg.tau_max}]")
        print(f"     Initial temp: {cfg.tau_init}")
        print(f"     Lambda warmup: {cfg.lambda_warmup_ratio*100:.0f}%")
    else:
        temp_module = None
        print(f"Using fixed temperature: {cfg.temperature}")

    # =========================================================================
    # v14: FDD Setup
    # =========================================================================
    fdd_enabled = cfg.use_fdd
    fdd_killed = False
    baseline_ppl = None  # Set at fdd_warmup_steps

    if cfg.use_fdd:
        print(f"")
        print(f"{config.VERSION}: Feature Dynamics Distillation (FDD)")
        print(f"     Layer mapping: {fdd_layer_map}")
        print(f"     Weight: {cfg.fdd_weight}")
        print(f"     Warmup: {cfg.fdd_warmup_steps} steps")
        print(f"     Loss type: {cfg.fdd_loss_type}")
        print(f"     Kill threshold: {cfg.fdd_kill_threshold*100:.0f}% PPL increase")

    if cfg.use_spike_semantic_loss:
        print(f"")
        print(f"{config.VERSION}: Spike Semantic Alignment")
        print(f"     Weight: {cfg.spike_semantic_weight}")
        print(f"     Warmup: {cfg.spike_semantic_warmup_steps} steps")
        print(f"     Target threshold scale: {cfg.spike_target_threshold_scale}")

    # =========================================================================
    # v14: Early Stopping Setup
    # =========================================================================
    best_ppl = float('inf')
    best_step = 0
    no_improve_steps = 0

    if cfg.use_early_stopping:
        print(f"")
        print(f"{config.VERSION}: Early Stopping")
        print(f"     Patience: {cfg.early_stopping_patience} steps")
        print(f"     Min delta: {cfg.min_ppl_delta} PPL")

    # =========================================================================
    # Setup optimizer
    # =========================================================================
    param_groups = [
        {'params': list(student.parameters()), 'lr': cfg.distill_lr}
    ]

    if cfg.hidden_align_weight > 0:
        param_groups.append({'params': list(projector.parameters()), 'lr': cfg.distill_lr})

    if temp_module is not None:
        param_groups.append({'params': list(temp_module.parameters()), 'lr': cfg.distill_lr})

    all_params = []
    for group in param_groups:
        all_params.extend(group['params'])

    try:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, fused=True)
        print("Using fused AdamW")
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.distill_steps)
    scaler = torch.cuda.amp.GradScaler()

    hw_stats.start()
    step = 0
    accum_step = 0
    current_stage = 1

    accumulation_steps = cfg.accumulation_steps
    effective_batch = cfg.batch_size * accumulation_steps
    print(f"Gradient accumulation: {accumulation_steps} (effective batch = {effective_batch})")
    print(f"Extended training: {cfg.distill_steps} steps")

    pbar = tqdm(total=cfg.distill_steps, desc='distilling (v14.1 - FDD+CTKD+HardCE)')

    optimizer.zero_grad(set_to_none=True)

    if len(train_loader) == 0:
        pbar.close()
        raise RuntimeError("train_loader is empty; aborting distillation loop.")
    if len(val_loader) == 0:
        pbar.close()
        raise RuntimeError("val_loader is empty; aborting distillation loop.")

    while step < cfg.distill_steps:
        for batch in train_loader:
            if step >= cfg.distill_steps:
                break

            # Check early stopping
            if cfg.use_early_stopping and no_improve_steps >= cfg.early_stopping_patience:
                print(f"\n  [Early Stopping] No improvement for {cfg.early_stopping_patience} steps")
                print(f"     Best PPL: {best_ppl:.2f} at step {best_step}")
                training_logs['early_stopped'] = True
                training_logs['early_stop_step'] = step
                pbar.close()
                return training_logs

            ids = batch[0].to(device, non_blocking=True)

            # Get lambda for CTKD
            if cfg.use_ctkd:
                current_lambda = get_lambda(
                    step, cfg.distill_steps,
                    lambda_max=cfg.lambda_max,
                    warmup_ratio=cfg.lambda_warmup_ratio
                )
            else:
                current_lambda = 0.0

            # Get current FDD weight
            if fdd_enabled and not fdd_killed:
                current_fdd_weight = get_fdd_weight(step, cfg.fdd_warmup_steps, cfg.fdd_weight)
            else:
                current_fdd_weight = 0.0

            if cfg.use_spike_semantic_loss:
                current_spike_sem_weight = get_spike_semantic_weight(
                    step,
                    cfg.spike_semantic_warmup_steps,
                    cfg.spike_semantic_weight,
                )
            else:
                current_spike_sem_weight = 0.0

            with torch.cuda.amp.autocast():
                # Teacher forward (always get hidden states for FDD)
                with torch.no_grad():
                    t_out = teacher(ids, output_hidden_states=True)
                    t_logits = t_out.logits
                    t_hiddens = t_out.hidden_states  # tuple of tensors

                # Student forward (always get hidden states for FDD)
                student.train()
                s_logits, s_hiddens, spike_aux = student(
                    ids,
                    return_hiddens=True,
                    return_spike_info=True,
                    detach_spikes=False,
                )
                spike_info = spike_aux.get('spike_info', {}) if isinstance(spike_aux, dict) else {}

                # Get temperature
                if cfg.use_ctkd and temp_module is not None:
                    T = temp_module(current_lambda)
                elif temp_module is not None:
                    T = temp_module()
                else:
                    T = cfg.temperature

                # KL divergence loss with temperature
                s_log = F.log_softmax(s_logits / T, dim=-1)
                t_prob = F.softmax(t_logits / T, dim=-1)
                kl_loss = F.kl_div(
                    s_log.view(-1, s_logits.size(-1)),
                    t_prob.view(-1, t_logits.size(-1)),
                    reduction='batchmean'
                ) * (T ** 2)

                # FDD loss (v14.1 with 100x weight increase)
                if current_fdd_weight > 0:
                    fdd_loss = compute_fdd_loss(
                        s_hiddens,
                        list(t_hiddens),  # Convert tuple to list
                        fdd_layer_map,
                        loss_type=cfg.fdd_loss_type
                    )
                else:
                    fdd_loss = torch.tensor(0.0, device=device)

                if current_spike_sem_weight > 0 and spike_info:
                    sem_losses = []
                    for s_layer, t_layer in fdd_layer_map.items():
                        layer_spikes = spike_info.get(s_layer)
                        if not isinstance(layer_spikes, dict):
                            continue

                        k_spikes = layer_spikes.get('k_spikes')
                        v_spikes = layer_spikes.get('v_spikes')
                        if k_spikes is None or v_spikes is None:
                            continue

                        spike_repr = 0.5 * (k_spikes + v_spikes)
                        teacher_hidden = t_hiddens[t_layer + 1]

                        if spike_repr.size(-1) != teacher_hidden.size(-1):
                            min_dim = min(spike_repr.size(-1), teacher_hidden.size(-1))
                            spike_repr = spike_repr[..., :min_dim]
                            teacher_hidden = teacher_hidden[..., :min_dim]

                        teacher_target = build_teacher_ternary_target(
                            teacher_hidden,
                            cfg.spike_target_threshold_scale,
                        )
                        sem_losses.append(F.mse_loss(spike_repr, teacher_target))

                    if sem_losses:
                        spike_sem_loss = torch.stack(sem_losses).mean()
                    else:
                        spike_sem_loss = torch.tensor(0.0, device=device)
                else:
                    spike_sem_loss = torch.tensor(0.0, device=device)

                # v14.1: Hard distillation (CE with ground truth)
                if cfg.ce_hard_weight > 0:
                    shift_logits = s_logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    ce_loss = torch.tensor(0.0, device=device)

                # Hidden alignment (usually disabled, kept for infrastructure)
                if cfg.hidden_align_weight > 0:
                    align_loss = compute_hidden_alignment_loss(
                        t_hiddens, s_hiddens, projector,
                        teacher_layers=cfg.teacher_n_layers,
                        student_layers=cfg.n_layers
                    )
                else:
                    align_loss = torch.tensor(0.0, device=device)

                # Total loss (v14.1: added ce_hard_weight * ce_loss)
                loss = (
                    kl_loss
                    + cfg.ce_hard_weight * ce_loss
                    + current_fdd_weight * fdd_loss
                    + current_spike_sem_weight * spike_sem_loss
                    + cfg.hidden_align_weight * align_loss
                )
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)

                if torch.isfinite(gn):
                    scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                hw_stats.record_step(ids.size(0) * accumulation_steps, ids.size(1))
                spike_stats.record(student, step)

                current_lr = optimizer.param_groups[0]['lr']
                current_temp = temp_module.get_temperature() if temp_module is not None else cfg.temperature

                # Log
                training_logs['loss_history'].append({'step': step, 'loss': loss.item() * accumulation_steps})
                training_logs['kl_loss_history'].append({'step': step, 'loss': kl_loss.item()})
                training_logs['ce_loss_history'].append({'step': step, 'loss': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss})  # v14.1
                training_logs['fdd_loss_history'].append({'step': step, 'loss': fdd_loss.item() if isinstance(fdd_loss, torch.Tensor) else fdd_loss})
                training_logs['spike_sem_loss_history'].append({'step': step, 'loss': spike_sem_loss.item() if isinstance(spike_sem_loss, torch.Tensor) else spike_sem_loss})
                training_logs['align_loss_history'].append({'step': step, 'loss': align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss})
                training_logs['lr_history'].append({'step': step, 'lr': current_lr})
                training_logs['temp_history'].append({'step': step, 'temperature': current_temp})
                training_logs['lambda_history'].append({'step': step, 'lambda': current_lambda})
                training_logs['fdd_weight_history'].append({'step': step, 'fdd_weight': current_fdd_weight})
                training_logs['spike_sem_weight_history'].append({'step': step, 'weight': current_spike_sem_weight})
                training_logs['stage_history'].append({'step': step, 'stage': 1})

                # Update progress bar (v14.1: added CE loss)
                fdd_str = f"fdd={fdd_loss.item():.3f}" if current_fdd_weight > 0 else "fdd=of"
                ce_str = f"ce={ce_loss.item():.3f}" if cfg.ce_hard_weight > 0 else "ce=of"
                sem_str = f"sem={spike_sem_loss.item():.3f}" if current_spike_sem_weight > 0 else "sem=of"
                pbar.set_postfix(
                    loss=f"{loss.item() * accumulation_steps:.3f}",
                    kl=f"{kl_loss.item():.3f}",
                    ce=ce_str,
                    sem=sem_str,
                    fdd=fdd_str,
                    T=f"{current_temp:.2f}",
                    lr=f"{current_lr:.1e}"
                )
                pbar.update(1)
                step += 1

                if step % cfg.eval_interval == 0:
                    val_loss = evaluate(student, val_loader, device)
                    val_ppl = get_ppl(val_loss)
                    training_logs['ppl_history'].append({'step': step, 'ppl': val_ppl})

                    amps = student.get_amplitudes()
                    amp_str = ', '.join([f"L{i}:{amps[f'layer_{i}']['k']:.2f}" for i in range(min(4, cfg.n_layers))])

                    # =========================================================
                    # v14.1: FDD Kill Switch
                    # =========================================================
                    if step == cfg.fdd_warmup_steps and fdd_enabled:
                        baseline_ppl = val_ppl
                        print(f"\n  [FDD] Baseline PPL at warmup end: {baseline_ppl:.2f}")

                    if fdd_enabled and not fdd_killed and baseline_ppl is not None:
                        ppl_increase = (val_ppl - baseline_ppl) / baseline_ppl
                        if ppl_increase > cfg.fdd_kill_threshold:
                            fdd_killed = True
                            training_logs['fdd_killed'] = True
                            training_logs['fdd_kill_step'] = step
                            print(f"\n  [FDD KILLED] PPL increased {ppl_increase*100:.1f}% > {cfg.fdd_kill_threshold*100:.0f}%")
                            print(f"     Baseline: {baseline_ppl:.2f}, Current: {val_ppl:.2f}")
                            print(f"     Disabling FDD for remaining training")

                    # Early stopping check
                    if val_ppl < best_ppl - cfg.min_ppl_delta:
                        best_ppl = val_ppl
                        best_step = step
                        no_improve_steps = 0
                        save_dict = {
                            'student': student.state_dict(),
                            'projector': projector.state_dict(),
                            'step': step,
                            'ppl': val_ppl,
                        }
                        if temp_module is not None:
                            save_dict['temp_module'] = temp_module.state_dict()
                        torch.save(save_dict, f'{OUTPUT_DIR}/checkpoints/v15_best.pt')
                        improve_str = " [NEW BEST]"
                    else:
                        no_improve_steps += cfg.eval_interval
                        improve_str = f" (no improve: {no_improve_steps}/{cfg.early_stopping_patience})"

                    lambda_str = f", lambda={current_lambda:.2f}" if cfg.use_ctkd else ""
                    fdd_status = "KILLED" if fdd_killed else f"w={current_fdd_weight:.4f}"
                    sem_status = f"w={current_spike_sem_weight:.4f}" if current_spike_sem_weight > 0 else "off"
                    print(
                        f"\n  step {step}: ppl={val_ppl:.1f}, T={current_temp:.2f}{lambda_str}, "
                        f"FDD:{fdd_status}, SEM:{sem_status}, amps=[{amp_str}...]{improve_str}"
                    )

    pbar.close()
    return training_logs

print("distillation function defined (v14.1.1 - FDD+CTKD+HardCE)")
