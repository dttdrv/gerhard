# =============================================================================
# cell 22: final evaluation (v14.1.1 - FDD+CTKD+HardCE)
# =============================================================================
print("="*60)
print("final evaluation (v14.1.1 - FDD+CTKD+HardCE)")
print("="*60)

teacher_loss = evaluate(teacher, val_loader, DEVICE, is_gpt2=True)
teacher_ppl = get_ppl(teacher_loss)
student_loss = evaluate(student, val_loader, DEVICE)
student_ppl = get_ppl(student_loss)

# v14: Get final temperature and lambda from CTKD
final_temp = distill_logs['temp_history'][-1]['temperature'] if distill_logs['temp_history'] else config.tau_init
final_lambda = distill_logs['temp_history'][-1].get('lambda', config.lambda_max) if distill_logs['temp_history'] else config.lambda_max

# VRAM logging
vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

print(f"")
print(f"{'model':<30} {'ppl':>10} {'params':>15}")
print("-" * 55)
print(f"{'gpt-2 (teacher)':<30} {teacher_ppl:>10.2f} {teacher_params:>15,}")
print(f"{'asnn-goose v14.1 (student)':<30} {student_ppl:>10.2f} {student_params:>15,}")
print("-" * 55)
print(f"{'compression':<30} {compression_ratio:>10.1f}x")
print(f"{'ppl gap':<30} {student_ppl - teacher_ppl:>10.2f}")
print(f"{'spike density':<30} {student.get_avg_spike_density():>10.3f}")
print(f"{'VRAM peak':<30} {vram_peak_gb:>10.2f}GB")
print(f"{'final temperature':<30} {final_temp:>10.2f}")
print(f"{'final lambda (GRL)':<30} {final_lambda:>10.3f}")
print("")
print("CTKD Implementation:")
print(f"  tau range: [{config.tau_min:.1f}, {config.tau_max:.1f}]")
print(f"  lambda warmup ratio: {config.lambda_warmup_ratio:.0%}")
print(f"  GRL: Gradient Reversal Layer for adversarial min-max")
print("")
print("version comparison:")
print(f"  v6: 627.3 PPL (baseline)")
print(f"  v7: 1655 PPL (regression!)")
print(f"  v8: 559 PPL (fixed)")
print(f"  v9: 541.7 PPL (capacity increase)")
print(f"  v10: 514.5 PPL (320d/5L baseline)")
print(f"  v14: 512.67 PPL (channel-wise, WITH reg)")
print(f"  v14.1: 512.04 PPL (channel-wise, NO reg)")
print(f"  v12: FAILED (temp runaway without GRL)")
print(f"  v14: {student_ppl:.2f} PPL (POCL, T={final_temp:.2f}, λ={final_lambda:.3f})")
if student_ppl < 500:
    print(f"  {config.VERSION} TARGET MET! PPL < 500")
elif student_ppl < 512.04:
    print(f"  v14.1 beats v14 by {424.81 - student_ppl:.1f} PPL")
else:
    print(f"  WARNING: v14.1 did not improve over v14")
