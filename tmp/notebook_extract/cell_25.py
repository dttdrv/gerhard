# =============================================================================
# cell 29: V15 Success Criteria Check
# =============================================================================

print('='*60)
print('V15 SUCCESS CRITERIA')
print('='*60)

tests = []

# Test 1: Dead neurons < 5%
dead_pct = v15_results.health.dead_neuron_pct
test1 = dead_pct < 0.05
tests.append(('Dead neurons < 5%', test1, f'{dead_pct*100:.1f}%'))

# Test 2: Saturated neurons < 10%
sat_pct = v15_results.health.saturated_neuron_pct
test2 = sat_pct < 0.10
tests.append(('Saturated neurons < 10%', test2, f'{sat_pct*100:.1f}%'))

# Test 3: MI > 0.1
mi_val = v15_results.mutual_information.get('mutual_information', 0)
test3 = mi_val > 0.1
tests.append(('MI > 0.1', test3, f'{mi_val:.4f}'))

# Test 4: CKA mean > 0.3
cka_mean = v15_results.cka.get('cka_mean', 0)
test4 = cka_mean > 0.3
tests.append(('CKA mean > 0.3', test4, f'{cka_mean:.4f}'))

# Test 5: Firing rate in healthy range [0.2, 0.6]
fr_mean = v15_results.health.firing_rate_mean
test5 = 0.2 <= fr_mean <= 0.6
tests.append(('Firing rate [0.2, 0.6]', test5, f'{fr_mean:.3f}'))

# Print results
for name, passed, value in tests:
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name}: {value}')

all_pass = all(t[1] for t in tests)
print(f'\nOverall: {"ALL PASS - Ready for v16 (sparse ops)" if all_pass else "NEEDS ATTENTION"}')
print('='*60)

# Store in results
results['v15_spiking_brain'] = {
    'validation': v15_results.to_dict(),
    'tests': {name: {'passed': passed, 'value': value} for name, passed, value in tests},
    'all_pass': all_pass,
}

print('\nV15 results added to results dict')
