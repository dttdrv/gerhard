# =============================================================================
# cell 23: visualization
# =============================================================================
figure_path = None

if MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # distillation loss
    d_steps = [l['step'] for l in distill_logs['loss_history']]
    d_losses = [l['loss'] for l in distill_logs['loss_history']]
    kl_losses = [l['loss'] for l in distill_logs['kl_loss_history']]
    axes[0,0].plot(d_steps, d_losses, label='total', alpha=0.8)
    axes[0,0].plot(d_steps, kl_losses, label='kl', alpha=0.7)

    # CE loss (if available)
    if 'ce_loss_history' in distill_logs and distill_logs['ce_loss_history']:
        ce_losses = [l['loss'] for l in distill_logs['ce_loss_history']]
        axes[0,0].plot(d_steps, ce_losses, label='ce', alpha=0.6)
    axes[0,0].set_xlabel('step')
    axes[0,0].set_ylabel('loss')
    axes[0,0].set_title(f'distillation loss ({config.VERSION})')
    axes[0,0].legend()

    # validation ppl
    p_steps = [l['step'] for l in distill_logs['ppl_history']]
    p_ppls = [l['ppl'] for l in distill_logs['ppl_history']]
    axes[0,1].plot(p_steps, p_ppls, 'orange', marker='o')
    axes[0,1].axhline(y=teacher_ppl, color='green', linestyle='--', label=f'teacher ({teacher_ppl:.1f})')
    axes[0,1].axhline(y=627.3, color='blue', linestyle=':', label='v6 (627.3)')
    axes[0,1].axhline(y=541.7, color='purple', linestyle=':', label='v9 (541.7)')
    axes[0,1].axhline(y=300, color='red', linestyle='--', label=f'{config.VERSION} target')
    axes[0,1].set_xlabel('step')
    axes[0,1].set_ylabel('ppl')
    axes[0,1].set_title('validation ppl')
    axes[0,1].legend()

    # lr schedule
    lr_steps = [l['step'] for l in distill_logs['lr_history']]
    lr_vals = [l['lr'] for l in distill_logs['lr_history']]
    axes[0,2].plot(lr_steps, lr_vals, 'purple')
    axes[0,2].axvline(x=config.warmup_steps, color='gray', linestyle='--', label=f'warmup ({config.warmup_steps})')
    axes[0,2].set_xlabel('step')
    axes[0,2].set_ylabel('lr')
    axes[0,2].set_title('learning rate')
    axes[0,2].legend()

    # spike density + amplitudes (first 4 layers)
    spike_summary = spike_stats.get_summary()
    layers = [f'layer_{i}' for i in range(min(4, config.n_layers))]
    k_dens = [spike_summary['per_layer'][l]['k_final'] for l in layers]
    v_dens = [spike_summary['per_layer'][l]['v_final'] for l in layers]
    k_amps = [spike_summary['per_layer'][l]['k_amp_final'] for l in layers]
    v_amps = [spike_summary['per_layer'][l]['v_amp_final'] for l in layers]

    x = np.arange(len(layers))
    axes[1,0].bar(x - 0.2, k_dens, 0.4, label='k density')
    axes[1,0].bar(x + 0.2, v_dens, 0.4, label='v density')
    ax2 = axes[1,0].twinx()
    ax2.plot(x, k_amps, 'r-o', label='k amp')
    ax2.plot(x, v_amps, 'b-s', label='v amp')
    axes[1,0].set_xlabel('layer')
    axes[1,0].set_ylabel('density')
    ax2.set_ylabel('amplitude')
    axes[1,0].set_title(f'spike density & amps (first 4/{config.n_layers} layers)')
    axes[1,0].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # ttt loss
    t_steps = [l['step'] for l in ttt_logs['loss_history']]
    t_losses = [l['loss'] for l in ttt_logs['loss_history']]
    axes[1,1].plot(t_steps, t_losses, 'red')
    axes[1,1].set_xlabel('step')
    axes[1,1].set_ylabel('ce loss')
    axes[1,1].set_title('ttt with lora')

    # version comparison
    # Historical versions for comparison (must match in length!)
    versions = ['v6', 'v9', 'v10', 'v12.1', 'v13.1', 'v14', 'v14.3', config.VERSION]
    # Teacher PPL is constant
    t_ppls = [44.6] * len(versions)
    # Student PPL history (from changelog)
    s_ppls = [627.3, 541.7, 514.5, 445.61, 434.44, 424.81, 306.89, student_ppl]
    assert len(versions) == len(t_ppls) == len(s_ppls), f"Array length mismatch: {len(versions)}, {len(t_ppls)}, {len(s_ppls)}"
    x = np.arange(len(versions))
    axes[1,2].bar(x - 0.2, t_ppls, 0.4, label='teacher', alpha=0.7)
    axes[1,2].bar(x + 0.2, s_ppls, 0.4, label='student', alpha=0.7)
    axes[1,2].axhline(y=300, color='red', linestyle='--', label=f'{config.VERSION} target', alpha=0.7)
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(versions)
    axes[1,2].set_ylabel('ppl')
    axes[1,2].set_title('version comparison')
    axes[1,2].legend()
    axes[1,2].set_yscale('log')

    plt.tight_layout()
    figure_path = f'{OUTPUT_DIR}/figures/v15_training_{RUN_TIMESTAMP}.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"saved: {figure_path}")
else:
    print("matplotlib unavailable: skipped visualization cell (cell 23).")
