# =============================================================================
# cell 15: evaluation functions (same as v9)
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, is_gpt2=False):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.inference_mode():
      for batch in loader:
        ids = batch[0].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(ids).logits if is_gpt2 else model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += ids[:, 1:].numel()
    if total_tokens == 0:
        raise RuntimeError(
            "Evaluation loader produced zero tokens; cannot compute loss/PPL."
        )
    return total_loss / total_tokens

def get_ppl(loss):
    return math.exp(min(loss, 10))

print("evaluation functions defined")
