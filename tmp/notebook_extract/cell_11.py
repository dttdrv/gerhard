# =============================================================================
# cell 11: load gpt-2 teacher (same as v9)
# =============================================================================
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("loading gpt-2 teacher...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
teacher.config.use_cache = False  # Disable KV caching (not needed for distillation)

# Compile teacher for faster inference (PyTorch 2.0+)
try:
    teacher = torch.compile(teacher, mode='reduce-overhead')
    print('teacher compiled with torch.compile')
except Exception as e:
    print(f'torch.compile not available: {e}')
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"teacher: gpt-2 ({teacher_params:,} params)")
