# =============================================================================
# cell 20: lora implementation (same as v9)
# =============================================================================
class LoRALinear(nn.Module):
    """lora adapter for linear layers."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model, rank=8, alpha=16.0, targets=['key_proj', 'value_proj']):
    """apply lora adapters to specified modules."""
    lora_modules = {}
    for name, module in model.named_modules():
        if any(t in name for t in targets) and isinstance(module, nn.Linear):
            lora = LoRALinear(module.in_features, module.out_features, rank, alpha).to(next(module.parameters()).device)
            lora_modules[name] = lora
            orig_forward = module.forward
            def make_forward(orig, lora_mod):
                def forward(x):
                    return orig(x) + lora_mod(x)
                return forward
            module.forward = make_forward(orig_forward, lora)
    print(f"lora: {len(lora_modules)} modules, rank={rank}")
    return lora_modules

print("lora defined")