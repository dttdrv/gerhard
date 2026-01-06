"""
Evaluation benchmarks for ASNN-Goose.

Reference: Section 10 of blueprint.

This module implements:
1. Perplexity evaluation on WikiText-2
2. Copy task for testing sequence memory
3. Retrieval task for testing long-range dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
import math
from tqdm import tqdm


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Evaluate perplexity on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        device: Device to evaluate on
        max_batches: Maximum batches to evaluate (None for all)
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary with perplexity and loss metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Perplexity")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits, _, _ = model(input_ids)
        else:
            logits, _, _ = model(input_ids)

        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_copy_task(
    model: nn.Module,
    device: torch.device,
    vocab_size: int = 100,
    seq_lengths: List[int] = [16, 32, 64, 128],
    num_samples: int = 100,
    delimiter_token: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate copy task: model must reproduce a sequence after a delimiter.

    Task format: [random tokens] [delimiter] [copy of random tokens]

    This tests the model's ability to store and retrieve sequences.

    Args:
        model: Model to evaluate
        device: Device to evaluate on
        vocab_size: Vocabulary size for random tokens
        seq_lengths: List of sequence lengths to test
        num_samples: Number of samples per length
        delimiter_token: Token ID for delimiter

    Returns:
        Dictionary mapping sequence lengths to accuracy metrics
    """
    model.eval()
    results = {}

    for seq_len in seq_lengths:
        correct = 0
        total = 0
        exact_matches = 0

        for _ in range(num_samples):
            # Generate random sequence (avoid delimiter token)
            source = torch.randint(
                2, vocab_size, (1, seq_len), device=device
            )

            # Create input: source + delimiter + source (for copying)
            delimiter = torch.full((1, 1), delimiter_token, device=device)
            input_ids = torch.cat([source, delimiter, source], dim=1)

            # Forward pass
            logits, _, _ = model(input_ids[:, :-1])

            # Get predictions for copy region
            copy_start = seq_len + 1  # After delimiter
            pred_logits = logits[:, copy_start-1:-1, :]  # Predict copy region
            predictions = pred_logits.argmax(dim=-1)

            target = source.squeeze(0)
            pred = predictions.squeeze(0)[:seq_len]

            # Count correct tokens
            correct += (pred == target).sum().item()
            total += seq_len

            # Count exact matches
            if (pred == target).all():
                exact_matches += 1

        results[f"len_{seq_len}"] = {
            "token_accuracy": correct / total,
            "exact_match": exact_matches / num_samples,
            "num_samples": num_samples,
        }

    return results


@torch.no_grad()
def evaluate_retrieval_task(
    model: nn.Module,
    device: torch.device,
    context_length: int = 512,
    query_positions: List[int] = [128, 256, 384],
    vocab_size: int = 100,
    num_samples: int = 100,
    key_token: int = 2,
    value_range: tuple = (10, 50),
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate retrieval task: find value associated with key at varying distances.

    Task format: Context with (key, value) pairs, query key, predict value.
    Tests long-range dependency modeling.

    Args:
        model: Model to evaluate
        device: Device to evaluate on
        context_length: Total context length
        query_positions: Positions to place the key-value pair
        vocab_size: Vocabulary size
        num_samples: Number of samples per position
        key_token: Special token for key
        value_range: Range for value tokens

    Returns:
        Dictionary mapping query positions to accuracy metrics
    """
    model.eval()
    results = {}

    for query_pos in query_positions:
        if query_pos >= context_length - 2:
            continue

        correct = 0
        total = 0

        for _ in range(num_samples):
            # Generate random context (avoid special tokens)
            context = torch.randint(
                value_range[1] + 1, vocab_size, (1, context_length), device=device
            )

            # Insert key-value pair at query_pos
            value = torch.randint(value_range[0], value_range[1], (1,), device=device)
            context[:, query_pos] = key_token
            context[:, query_pos + 1] = value

            # Query at end: place key and predict value
            context[:, -2] = key_token
            context[:, -1] = value  # Target

            # Forward pass
            logits, _, _ = model(context[:, :-1])

            # Predict value after final key
            pred = logits[:, -1, :].argmax(dim=-1)

            if pred.item() == value.item():
                correct += 1
            total += 1

        results[f"pos_{query_pos}"] = {
            "accuracy": correct / total,
            "distance": context_length - query_pos,
            "num_samples": num_samples,
        }

    return results


def run_all_benchmarks(
    model: nn.Module,
    device: torch.device,
    dataloader: Optional[DataLoader] = None,
    vocab_size: int = 32000,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run all evaluation benchmarks.

    Args:
        model: Model to evaluate
        device: Device to evaluate on
        dataloader: Optional dataloader for perplexity
        vocab_size: Vocabulary size
        config: Optional configuration

    Returns:
        Dictionary with all benchmark results
    """
    results = {}

    # Perplexity
    if dataloader is not None:
        results["perplexity"] = evaluate_perplexity(
            model, dataloader, device,
            max_batches=config.eval.max_eval_samples // config.eval.eval_batch_size if config else 100,
        )

    # Copy task
    copy_lengths = config.eval.copy_seq_lengths if config else [16, 32, 64, 128]
    results["copy_task"] = evaluate_copy_task(
        model, device,
        vocab_size=min(vocab_size, 100),
        seq_lengths=copy_lengths,
    )

    # Retrieval task
    query_positions = config.eval.retrieval_query_positions if config else [128, 256, 384]
    results["retrieval_task"] = evaluate_retrieval_task(
        model, device,
        context_length=config.eval.retrieval_context_len if config else 512,
        query_positions=query_positions,
        vocab_size=min(vocab_size, 100),
    )

    return results


def compute_summary_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute summary metrics from benchmark results.

    Args:
        results: Full benchmark results

    Returns:
        Dictionary of summary metrics
    """
    summary = {}

    # Perplexity
    if "perplexity" in results:
        summary["perplexity"] = results["perplexity"]["perplexity"]

    # Copy task average
    if "copy_task" in results:
        accuracies = [
            v["token_accuracy"] for v in results["copy_task"].values()
        ]
        summary["copy_task_avg_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0

    # Retrieval task average
    if "retrieval_task" in results:
        accuracies = [
            v["accuracy"] for v in results["retrieval_task"].values()
        ]
        summary["retrieval_task_avg_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0

    return summary
