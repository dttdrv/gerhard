# =============================================================================
# cell 21: ttt with lora (same as v9)
# =============================================================================
print("="*60)
print("phase 2: test-time training with lora")
print("="*60)

for p in student.parameters():
    p.requires_grad = False

lora_modules = apply_lora(student, config.lora_rank, config.lora_alpha)
lora_params = sum(p.numel() for m in lora_modules.values() for p in m.parameters())

pre_ttt_loss = evaluate(student, val_loader, DEVICE)
pre_ttt_ppl = get_ppl(pre_ttt_loss)
print(f"\npre-ttt ppl: {pre_ttt_ppl:.2f}")

lora_opt = torch.optim.AdamW([p for m in lora_modules.values() for p in m.parameters()], lr=config.ttt_lr)
ttt_logs = {'loss_history': []}
student.train()

for step, batch in enumerate(val_loader):
    if step >= config.ttt_steps:
        break
    ids = batch[0].to(DEVICE)
    with torch.cuda.amp.autocast():
        loss = F.cross_entropy(student(ids)[:, :-1].reshape(-1, config.vocab_size), ids[:, 1:].reshape(-1))
    lora_opt.zero_grad()
    loss.backward()
    lora_opt.step()
    ttt_logs['loss_history'].append({'step': step, 'loss': loss.item()})
    if step % 20 == 0:
        print(f"  ttt {step}: loss={loss.item():.4f}")

post_ttt_loss = evaluate(student, val_loader, DEVICE)
post_ttt_ppl = get_ppl(post_ttt_loss)
print(f"\npost-ttt ppl: {post_ttt_ppl:.2f}")
print(f"ttt improvement: {pre_ttt_ppl - post_ttt_ppl:.1f} ppl")