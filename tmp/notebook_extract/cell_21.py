# =============================================================================
# cell 24: build results dict (v15 - spikingbrain validation)
# =============================================================================
print(f"building results ({config.VERSION} - {config.VERSION_DESC})...")

figure_base64 = None
training_plot_filename = None
if 'figure_path' in globals() and figure_path and os.path.exists(figure_path):
    with open(figure_path, 'rb') as f:
        figure_base64 = base64.b64encode(f.read()).decode('utf-8')
    training_plot_filename = os.path.basename(figure_path)
else:
    print("figure not available; continuing without embedded training plot")

# v13: Extract final lambda
final_lambda = distill_logs['temp_history'][-1].get('lambda', config.lambda_max) if distill_logs['temp_history'] else config.lambda_max

results = {
    'version': config.VERSION,
    'timestamp': datetime.now().isoformat(),
    'run_id': RUN_TIMESTAMP,
    'platform': PLATFORM,
    'description': config.VERSION_DESC,

    f'{config.VERSION}_design': {
        'principle': 'CTKD: Adversarial min-max optimization via GRL',
        'innovation': 'Gradient Reversal Layer makes temperature MAXIMIZE KL while student MINIMIZES',
        'rationale': 'Proper CTKD (ArXiv 2211.16231) requires adversarial training, not simple regularization',
        'why_v12_failed': 'v12 used simple regularization - optimizer pushed T to max for easy KL',
        'techniques': {
            'ctkd_with_grl': 'ENABLED - Gradient Reversal Layer for adversarial min-max (v13 KEY)',
            'lambda_scheduling': f'Cosine warmup 0->{config.lambda_max} with {config.lambda_warmup_ratio:.0%} warmup',
            'sigmoid_bounding': f'T bounded to [{config.tau_min}, {config.tau_max}] via sigmoid (smooth gradients)',
            'no_manual_reg': 'GRL eliminates need for manual temperature regularization',
            'progressive_stages': 'DISABLED',
            'channel_wise_spikes': 'DISABLED (structural symmetry issue)',
        },
        'grl_mechanism': {
            'forward_pass': 'Identity: GRL(x) = x',
            'backward_pass': 'Negation: dGRL/dx = -lambda',
            'effect': 'Temperature gradients reversed -> T maximizes KL loss',
        },
        'temperature_config': {
            'tau_min': config.tau_min,
            'tau_max': config.tau_max,
            'tau_init': config.tau_init,
            'lambda_max': config.lambda_max,
            'lambda_warmup_ratio': config.lambda_warmup_ratio,
        },
        'architecture': {
            'd_model': config.d_model,
            'n_layers': 5,
            'params': '~22M',
        },
        'speedups': {
            'gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
            'torch_compile': compile_success,
            'fused_optimizer': True,
            'accumulation_steps': config.accumulation_steps,
        },
        'unchanged': [
            'hidden_align_weight: 0.0',
            'warmup_steps: 50',
            'distill_steps: 3000',
        ],
    },

    'architecture': {
        'teacher': {'name': 'gpt2', 'params': teacher_params},
        'student': {
            'name': f'asnn-goose-{config.VERSION}',
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'params': student_params,
        },
        'projector_params': projector_params,
        'compression_ratio': compression_ratio,
        'vram_peak_gb': vram_peak_gb,
    },

    'training_config': {
        'distill_steps': config.distill_steps,
        'tau_min': config.tau_min,
        'tau_max': config.tau_max,
        'tau_init': config.tau_init,
        'final_temperature': final_temp,
        'lambda_max': config.lambda_max,
        'lambda_warmup_ratio': config.lambda_warmup_ratio,
        'final_lambda': final_lambda,
        'hidden_align_weight': config.hidden_align_weight,
        'warmup_steps': config.warmup_steps,
        'batch_size': config.batch_size,
        'accumulation_steps': config.accumulation_steps,
        'effective_batch': config.batch_size * config.accumulation_steps,
        'distill_lr': config.distill_lr,
        'max_grad_norm': config.max_grad_norm,
    },

    'results': {
        'teacher_ppl': teacher_ppl,
        'student_ppl': student_ppl,
        'ppl_gap': student_ppl - teacher_ppl,
        'spike_density': student.get_avg_spike_density(),
        'amplitudes': student.get_amplitudes(),
        'final_temperature': final_temp,
        'final_lambda': final_lambda,
        'target_met': student_ppl < 500,
    },

    'training_curves': {
        'loss_history': distill_logs['loss_history'],
        'kl_loss_history': distill_logs['kl_loss_history'],
        'ce_loss_history': distill_logs.get('ce_loss_history', []),
        'fdd_loss_history': distill_logs.get('fdd_loss_history', []),
        'spike_sem_loss_history': distill_logs.get('spike_sem_loss_history', []),
        'align_loss_history': distill_logs['align_loss_history'],
        'ppl_history': distill_logs['ppl_history'],
        'lr_history': distill_logs['lr_history'],
        'temp_history': distill_logs['temp_history'],  # v13: includes temperature AND lambda
        'lambda_history': distill_logs.get('lambda_history', []),
        'spike_sem_weight_history': distill_logs.get('spike_sem_weight_history', []),
    },

    'hardware_stats': hw_stats.get_summary(),
    'spike_analysis': spike_stats.get_summary(),

    'ttt': {
        'lora_params': lora_params,
        'pre_ppl': pre_ttt_ppl,
        'post_ppl': post_ttt_ppl,
        'improvement': pre_ttt_ppl - post_ttt_ppl,
        'loss_history': ttt_logs['loss_history'],
    },

    'comparison': {
        'v6': {'student_ppl': 627.3, 'note': 'baseline'},
        'v7': {'student_ppl': 1655, 'note': 'regression (align=1.0, T=4)'},
        'v8': {'student_ppl': 559, 'note': 'fixed defaults (align=0, T=2)'},
        'v9': {'student_ppl': 541.7, 'note': 'capacity increase (320d, 5L)'},
        'v10': {'student_ppl': 514.5, 'note': '320d/5L baseline'},
        'v14': {'student_ppl': 512.67, 'note': 'channel-wise WITH reg (bug)'},
        'v14.1': {'student_ppl': 512.04, 'note': 'channel-wise NO reg (symmetry issue)'},
        'v12': {'student_ppl': 'FAILED', 'note': 'temp runaway without GRL'},
        'v13': {'student_ppl': student_ppl, 'note': f'POCL (T={final_temp:.2f}, L={final_lambda:.3f})'},
    },

    'figures': {
        'training_plot': {
            'filename': training_plot_filename,
            'base64': figure_base64,
        }
    },

    # validation_tests will be added in cell 26
}

print("results dict built (validation_tests pending)")
print(f"  version: {config.VERSION} ({config.VERSION_DESC})")
print(f"  final_temperature: {final_temp:.2f}")
print(f"  final_lambda: {final_lambda:.3f}")
