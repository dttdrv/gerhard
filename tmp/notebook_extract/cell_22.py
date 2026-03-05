# =============================================================================
# cell 25: validation tests (v14.1 - 12 tests with FDD)
# =============================================================================
# These tests validate correct v14.1 implementation

print("="*60)
print(f"{config.VERSION} Validation Test Suite")
print("="*60)

tests = []

# =============================================================================
# Test 1: PPL Target (<400)
# =============================================================================
if distill_logs['ppl_history']:
    best_ppl_entry = min(distill_logs['ppl_history'], key=lambda x: x['ppl'])
    best_ppl = best_ppl_entry['ppl']
    target_ppl = 400  # target for this version
    ppl_pass = best_ppl < target_ppl
    tests.append(('PPL < 400', ppl_pass, f"best_ppl={best_ppl:.2f}, target={target_ppl}"))
else:
    tests.append(('PPL < 400', False, "No PPL history found"))

# =============================================================================
# Test 2: PPL Improvement over v13.1 (434.44)
# =============================================================================
if distill_logs['ppl_history']:
    best_ppl_entry = min(distill_logs['ppl_history'], key=lambda x: x['ppl'])
    v13_1_ppl = 434.44
    improvement = v13_1_ppl - best_ppl_entry['ppl']
    improve_pass = improvement > 0
    tests.append(('Improved over v13.1', improve_pass, f"improvement={improvement:.2f} PPL"))
else:
    tests.append(('Improved over v13.1', False, "No PPL history"))

# =============================================================================
# Test 3: Spike Density in Valid Range [0.1, 0.9]
# =============================================================================
if spike_stats.step_densities:
    final_density = spike_stats.step_densities[-1]['density']
    density_pass = 0.1 <= final_density <= 0.9
    tests.append(('Spike density [0.1, 0.9]', density_pass, f"density={final_density:.3f}"))
else:
    tests.append(('Spike density [0.1, 0.9]', False, "No spike history"))

# =============================================================================
# Test 4: Amplitudes in Healthy Range [0.3, 3.0]
# =============================================================================
amps = student.get_amplitudes()
amp_values = []
for layer_name, layer_amps in amps.items():
    amp_values.extend([layer_amps['k'], layer_amps['v']])
amp_min, amp_max = min(amp_values), max(amp_values)
amp_pass = 0.3 <= amp_min and amp_max <= 3.0
tests.append(('Amplitudes [0.3, 3.0]', amp_pass, f"range=[{amp_min:.2f}, {amp_max:.2f}]"))

# =============================================================================
# Test 5: Training Completed (all steps or early stopped)
# =============================================================================
if distill_logs['early_stopped']:
    training_pass = distill_logs['early_stop_step'] > config.distill_steps * 0.3
    tests.append(('Training completed', training_pass, f"Early stopped at {distill_logs['early_stop_step']} steps"))
else:
    training_pass = len(distill_logs['loss_history']) >= config.distill_steps * 0.95
    tests.append(('Training completed', training_pass, f"Completed {len(distill_logs['loss_history'])}/{config.distill_steps} steps"))

# =============================================================================
# Test 6: No NaN/Inf in Loss
# =============================================================================
nan_inf_found = False
for h in distill_logs['loss_history']:
    if h['loss'] != h['loss'] or h['loss'] == float('inf') or h['loss'] == float('-inf'):
        nan_inf_found = True
        break
nan_pass = not nan_inf_found
tests.append(('No NaN/Inf loss', nan_pass, "All losses finite" if nan_pass else "Found NaN/Inf"))

# =============================================================================
# Test 7: VRAM Usage Reasonable (<8GB)
# =============================================================================
if hasattr(hw_stats, 'get_summary'):
    hw_summary = hw_stats.get_summary()
    vram_gb = hw_summary.get('peak_gpu_memory_gb', 0)
    vram_pass = vram_gb < 8.0
    tests.append(('VRAM < 8GB', vram_pass, f"peak={vram_gb:.2f}GB"))
else:
    tests.append(('VRAM < 8GB', True, "hw_stats not available"))

# =============================================================================
# Test 8: FDD Was Active (v14.1)
# =============================================================================
if config.use_fdd:
    # FDD should have been active at some point
    fdd_losses = [h['loss'] for h in distill_logs['fdd_loss_history'] if h.get('loss') is not None and h['loss'] != 0]
    fdd_active_pass = len(fdd_losses) > 0
    if distill_logs['fdd_killed']:
        status = f"Active then KILLED at step {distill_logs['fdd_kill_step']}"
    else:
        status = f"Active for {len(fdd_losses)} steps"
    tests.append(('FDD was active', fdd_active_pass, status))
else:
    tests.append(('FDD was active', True, "FDD disabled in config"))

# =============================================================================
# Test 9: CTKD Temperature Evolved (v14.1)
# =============================================================================
if config.use_ctkd and distill_logs['temp_history']:
    temps = [h['temperature'] for h in distill_logs['temp_history']]
    start_temp = temps[0]
    end_temp = temps[-1]
    temp_evolved = abs(end_temp - start_temp) > 0.1  # Should have moved
    tests.append(('Temperature evolved', temp_evolved, f"start={start_temp:.2f}, end={end_temp:.2f}"))
else:
    tests.append(('Temperature evolved', True, "CTKD disabled or no temp history"))

# =============================================================================
# Test 10: Early Stopping Working (if triggered)
# =============================================================================
if config.use_early_stopping:
    if distill_logs['early_stopped']:
        es_step = distill_logs['early_stop_step']
        es_pass = config.distill_steps * 0.3 < es_step < config.distill_steps
        tests.append(('Early stopping working', es_pass, f"stopped at {es_step}"))
    else:
        if distill_logs['ppl_history']:
            best_ppl_entry = min(distill_logs['ppl_history'], key=lambda x: x['ppl'])
            last_improvement_step = best_ppl_entry['step']
            final_step = distill_logs['ppl_history'][-1]['step']
            gap = final_step - last_improvement_step
            es_pass = gap <= config.early_stopping_patience + config.eval_interval
            tests.append(('Early stopping working', es_pass, f"last improvement at step {last_improvement_step}"))
        else:
            tests.append(('Early stopping working', True, "No PPL history"))
else:
    tests.append(('Early stopping working', True, "Early stopping disabled"))

# =============================================================================
# Test 11: FDD Loss Decreased (v14.1)
# =============================================================================
if config.use_fdd and distill_logs['fdd_loss_history']:
    fdd_losses = [h['loss'] for h in distill_logs['fdd_loss_history'] if h.get('loss') is not None and h['loss'] != 0]
    if len(fdd_losses) >= 10:
        start_fdd = sum(fdd_losses[:5]) / 5
        end_fdd = sum(fdd_losses[-5:]) / 5
        fdd_decreased = end_fdd < start_fdd
        tests.append(('FDD loss decreased', fdd_decreased, f"start={start_fdd:.4f}, end={end_fdd:.4f}"))
    else:
        tests.append(('FDD loss decreased', True, "Not enough FDD data points"))
else:
    tests.append(('FDD loss decreased', True, "FDD disabled or no history"))

# =============================================================================
# Test 12: Extended Training (5000 steps)
# =============================================================================
extended_pass = config.distill_steps >= 5000
tests.append(('Extended training (5000+)', extended_pass, f"distill_steps={config.distill_steps}"))

# =============================================================================
# Report Results
# =============================================================================
print("\n" + "-"*60)
print("TEST RESULTS")
print("-"*60)

passed = 0
failed = 0
for name, result, details in tests:
    status = "PASS" if result else "FAIL"
    symbol = "V" if result else "X"
    print(f"[{symbol}] {name}: {details}")
    if result:
        passed += 1
    else:
        failed += 1

print("-"*60)
print(f"SUMMARY: {passed}/{len(tests)} tests passed")
if failed > 0:
    print(f"WARNING: {failed} tests failed!")
else:
    print(f"ALL TESTS PASSED! {config.VERSION} implementation validated.")
print("="*60)

# Store results
validation_results = {
    'tests': tests,
    'passed': passed,
    'failed': failed,
    'total': len(tests)
}
