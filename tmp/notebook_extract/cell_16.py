# =============================================================================
# cell 18: run distillation (v14.1.1 - FDD+CTKD+HardCE)
# =============================================================================
print("="*60)
print("v14.1: FDD (Feature Dynamics Distillation) + CTKD")
print("="*60)
print(f"  Architecture: {config.d_model}d x {config.n_layers}L (~22M params)")
print(f"  Target: PPL < 400 (improve on v13.1's 434.44)")
print(f"")
print(f"{config.VERSION} Configuration:")
print(f"  FDD: {config.use_fdd}")
print(f"    Weight: {config.fdd_weight}")
print(f"    Warmup: {config.fdd_warmup_steps} steps")
print(f"    Layer map: {fdd_layer_map}")
print(f"    Loss type: {config.fdd_loss_type}")
print(f"    Kill threshold: {config.fdd_kill_threshold*100:.0f}%")
print(f"  Spike semantic alignment: {config.use_spike_semantic_loss}")
if config.use_spike_semantic_loss:
    print(f"    Weight: {config.spike_semantic_weight}")
    print(f"    Warmup: {config.spike_semantic_warmup_steps}")
    print(f"    Target threshold scale: {config.spike_target_threshold_scale}")
print(f"  CTKD: {config.use_ctkd} (proven from v12.1, v13.1)")
print(f"  Extended training: {config.distill_steps} steps")
print(f"  Early stopping: patience={config.early_stopping_patience}")
print(f"  POCL: {config.use_pocl} (disabled - caused regression)")
print("")

# Instantiate collectors
hw_stats = HardwareStatsCollector()
spike_stats = SpikeStatsCollector(config.n_layers)
print("Initialized HardwareStatsCollector and SpikeStatsCollector")

# Run distillation (FDD + CTKD)
print(f"\nStarting distillation...")

distill_logs = distill_v14(
    teacher, student, projector,
    train_loader, val_loader,
    config, DEVICE,
    hw_stats, spike_stats,
    fdd_layer_map  # v14: pass FDD layer mapping
)

# Report results
print(f"\n\n" + "="*60)
print("v14.1 Distillation Complete!")
print("="*60)

if distill_logs['ppl_history']:
    final_ppl = distill_logs['ppl_history'][-1]['ppl']
    best_ppl_entry = min(distill_logs['ppl_history'], key=lambda x: x['ppl'])
    print(f"\nFinal PPL: {final_ppl:.2f}")
    print(f"Best PPL: {best_ppl_entry['ppl']:.2f} at step {best_ppl_entry['step']}")

if distill_logs['early_stopped']:
    print(f"\nEarly stopped at step {distill_logs['early_stop_step']}")
else:
    print(f"\nCompleted all {config.distill_steps} steps")

if distill_logs['fdd_killed']:
    print(f"\nFDD was KILLED at step {distill_logs['fdd_kill_step']} (PPL regressed)")
else:
    print(f"\nFDD remained active throughout training")

if distill_logs['temp_history']:
    temps = [h['temperature'] for h in distill_logs['temp_history']]
    print(f"\nTemperature evolution:")
    print(f"  Start: {temps[0]:.2f}")
    print(f"  End: {temps[-1]:.2f}")

if distill_logs['lambda_history']:
    lambdas = [h['lambda'] for h in distill_logs['lambda_history']]
    print(f"\nLambda evolution:")
    print(f"  Start: {lambdas[0]:.2f}")
    print(f"  End: {lambdas[-1]:.2f}")

if distill_logs['fdd_loss_history']:
    fdd_losses = [h['loss'] for h in distill_logs['fdd_loss_history'] if h['loss'] > 0]
    if fdd_losses:
        print(f"\nFDD loss evolution:")
        print(f"  Start (after warmup): {fdd_losses[0]:.4f}")
        print(f"  End: {fdd_losses[-1]:.4f}")

print(f"\n" + "="*60)
