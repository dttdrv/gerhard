"""
Training utilities for ASNN-Goose.

This module provides:
1. Optimizer and scheduler creation utilities
2. TrainingState dataclass for tracking
3. Generic training/evaluation loops
4. Data loading utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import time
from tqdm import tqdm
import math


@dataclass
class TrainingState:
    """
    Container for training state.
    Tracks step counts, losses, and other metrics.
    """
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    best_step: int = 0
    losses: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, loss: float, **kwargs):
        """Update state with new loss and metrics."""
        self.step += 1
        self.losses.append(loss)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = self.step

        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "recent_loss": self.losses[-100:] if self.losses else [],
            "metrics": {k: v[-100:] for k, v in self.metrics.items()},
        }


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to biases and layer norms
        if "bias" in name or "ln" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    max_steps: int = 10000,
    warmup_steps: int = 500,
    min_lr_ratio: float = 0.1,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
        max_steps: Total training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of initial LR
        **kwargs: Additional scheduler arguments

    Returns:
        Configured scheduler
    """
    if scheduler_type == "cosine":
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        # Linear decay with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
                return max(min_lr_ratio, 1 - progress * (1 - min_lr_ratio))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "constant":
        # Constant LR with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    state: TrainingState,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 100,
    progress_bar: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        state: Training state tracker
        max_grad_norm: Gradient clipping threshold
        use_amp: Use automatic mixed precision
        log_interval: How often to log metrics
        progress_bar: Show progress bar

    Returns:
        Dictionary of epoch metrics
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    epoch_loss = 0.0
    epoch_tokens = 0
    start_time = time.perf_counter()

    iterator = tqdm(dataloader, desc=f"Epoch {state.epoch}") if progress_bar else dataloader

    for batch_idx, batch in enumerate(iterator):
        input_ids = batch["input_ids"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _, _ = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _, _ = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update state
        state.update(
            loss=loss.item(),
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            lr=scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"],
        )

        epoch_loss += loss.item() * input_ids.size(0)
        epoch_tokens += input_ids.numel()

        if progress_bar:
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{math.exp(min(loss.item(), 10)):.2f}",
            })

    elapsed = time.perf_counter() - start_time
    avg_loss = epoch_loss / max(len(dataloader), 1)

    state.epoch += 1

    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 10)),
        "tokens_per_sec": epoch_tokens / elapsed,
        "time_sec": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
    progress_bar: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        device: Device to evaluate on
        use_amp: Use automatic mixed precision
        max_batches: Maximum batches to evaluate (None for all)
        progress_bar: Show progress bar

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    iterator = tqdm(dataloader, desc="Evaluating") if progress_bar else dataloader

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits, _, _ = model(input_ids)
        else:
            logits, _, _ = model(input_ids)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()
        num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 10)),
        "num_tokens": total_tokens,
        "num_batches": num_batches,
    }


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.

    Tokenizes text and creates fixed-length sequences.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []

        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, max(1, len(tokens) - max_length + 1), stride):
                self.examples.append(tokens[i : i + max_length])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}


def create_wikitext_dataloader(
    split: str = "train",
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2",
) -> DataLoader:
    """
    Create DataLoader for WikiText-2 dataset.

    Args:
        split: Dataset split ("train", "validation", "test")
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        tokenizer_name: Name of tokenizer to use

    Returns:
        Configured DataLoader
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Filter empty texts
    texts = [t for t in dataset["text"] if t.strip()]

    # Create dataset
    text_dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=max_length // 2,
    )

    return DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


def get_tokenizer(name: str = "gpt2"):
    """Get tokenizer by name."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
