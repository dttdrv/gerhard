# =============================================================================
# cell 13: data loading (v14 - efficient DataLoader)
# =============================================================================
from datasets import load_dataset

print("loading wikitext-2...")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

def pre_tokenize(texts, max_len):
    all_tokens = []
    for text in tqdm(texts, desc="tokenizing", leave=False):
        if text.strip():
            all_tokens.extend(tokenizer.encode(text, max_length=max_len*2, truncation=True))
    chunks = [all_tokens[i:i+max_len] for i in range(0, len(all_tokens)-max_len+1, max_len//2) if len(all_tokens[i:i+max_len]) == max_len]
    print(f"created {len(chunks)} sequences")
    return torch.tensor(chunks, dtype=torch.long)

train_tokens = pre_tokenize(dataset['train']['text'], config.max_seq_len)
val_tokens = pre_tokenize(dataset['validation']['text'], config.max_seq_len)

# v14: efficient DataLoader with workers and prefetch
# Note: num_workers=0 for Kaggle/Colab compatibility, but prefetch still helps
dataloader_kwargs = {
    'batch_size': config.batch_size,
    'pin_memory': True,
    'num_workers': 0 if IS_KAGGLE or IS_COLAB else 2,  # workers disabled on cloud platforms
    'prefetch_factor': None if IS_KAGGLE or IS_COLAB else 2,
    'persistent_workers': False if IS_KAGGLE or IS_COLAB else True,
}

train_loader = DataLoader(TensorDataset(train_tokens), shuffle=True, **dataloader_kwargs)
val_loader = DataLoader(TensorDataset(val_tokens), shuffle=False, **dataloader_kwargs)

print(f"train: {len(train_loader)} batches, val: {len(val_loader)} batches")
print(f"DataLoader: num_workers={dataloader_kwargs['num_workers']}, pin_memory={dataloader_kwargs['pin_memory']}")
if len(train_loader) == 0:
    raise RuntimeError(
        "train_loader is empty after tokenization. "
        "Check dataset availability and max_seq_len/chunking settings."
    )
if len(val_loader) == 0:
    raise RuntimeError(
        "val_loader is empty after tokenization. "
        "Check dataset availability and max_seq_len/chunking settings."
    )
