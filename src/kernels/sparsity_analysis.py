"""
Kernel-level sparsity analysis for ASNN-Goose.

Reference: Sections 5.4, 8.1-8.2 of blueprint.

This module analyzes whether activation sparsity can translate
to real computational savings. Key insight: naive sparse operations
are often SLOWER than dense on GPUs due to indexing overhead.

Provides:
1. Dense vs sparse matmul benchmarks
2. Structured sparsity pattern analysis
3. Memory bandwidth analysis
4. Warp-level efficiency analysis
"""
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class SparsityKernelAnalyzer:
    """
    Analyze whether activation sparsity translates to real speedups.

    GPU sparsity is only beneficial when:
    1. Sparsity is structured (warp-aligned, block-aligned)
    2. Sparsity level is very high (>90%)
    3. Using specialized sparse kernels

    Reference: Section 5.4 kernel realism.
    """

    def __init__(self, device: torch.device, warp_size: int = 32):
        self.device = device
        self.warp_size = warp_size
        self.results: List[Dict[str, Any]] = []

    def benchmark_dense_vs_sparse_matmul(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Compare dense matmul vs various sparse approaches.

        Tests:
        1. Dense: Standard matmul
        2. Masked: Sparse-as-dense (no actual speedup expected)
        3. COO sparse: PyTorch sparse tensor (often slower!)
        4. Block-masked: Skip computation for zero blocks

        Args:
            weight: Weight matrix (out_features, in_features)
            activation: Activation tensor (batch, in_features)
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing results and analysis
        """
        weight = weight.to(self.device)
        activation = activation.to(self.device)

        # Warmup
        for _ in range(warmup_runs):
            _ = torch.matmul(activation, weight.T)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 1. Dense baseline
        start = time.perf_counter()
        for _ in range(num_runs):
            dense_result = torch.matmul(activation, weight.T)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        dense_time = (time.perf_counter() - start) / num_runs

        # 2. Masked sparse (simulate structured sparsity)
        mask = (activation != 0).float()
        masked_activation = activation * mask

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            sparse_result = torch.matmul(masked_activation, weight.T)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        masked_time = (time.perf_counter() - start) / num_runs

        # 3. True sparse (COO format) - usually slower on GPU!
        sparse_activation = activation.to_sparse()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            coo_result = torch.sparse.mm(sparse_activation, weight.T)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        coo_time = (time.perf_counter() - start) / num_runs

        # Compute sparsity level
        sparsity = (activation == 0).float().mean().item()

        result = {
            "sparsity": sparsity,
            "dense_time_ms": dense_time * 1000,
            "masked_time_ms": masked_time * 1000,
            "coo_sparse_time_ms": coo_time * 1000,
            "masked_speedup": dense_time / masked_time if masked_time > 0 else 0,
            "coo_speedup": dense_time / coo_time if coo_time > 0 else 0,
            "is_sparse_faster": masked_time < dense_time or coo_time < dense_time,
        }

        self.results.append(result)
        return result

    def analyze_structured_sparsity_potential(
        self,
        activations: torch.Tensor,
        block_sizes: List[int] = [4, 8, 16, 32],
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze whether activations could benefit from structured sparsity kernels.

        Structured sparsity (like N:M sparsity) requires specific patterns.
        NVIDIA Ampere+ supports 2:4 sparsity (2 zeros per 4 elements).

        Args:
            activations: Activation tensor
            block_sizes: Block sizes to analyze

        Returns:
            Dictionary with block-level sparsity statistics
        """
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        activations = activations.view(activations.shape[0], -1)
        batch_size, dim = activations.shape

        results = {}

        for block_size in block_sizes:
            if dim < block_size:
                continue

            # Reshape into blocks
            num_blocks = dim // block_size
            truncated = activations[:, :num_blocks * block_size]
            blocks = truncated.view(batch_size, num_blocks, block_size)

            # Count zeros per block
            zeros_per_block = (blocks == 0).sum(dim=-1).float()

            # N:M pattern analysis
            nm_patterns = {}
            for n in [1, 2, 3]:
                # Check if we have at least n zeros per 4 elements
                target_zeros = n * (block_size // 4)
                matches_pattern = (zeros_per_block >= target_zeros).float().mean().item()
                nm_patterns[f"{n}:{4}"] = matches_pattern

            # Compute effective sparsity
            fully_zero_blocks = (zeros_per_block == block_size).float().mean().item()

            results[f"block_{block_size}"] = {
                "mean_zeros_per_block": zeros_per_block.mean().item(),
                "std_zeros_per_block": zeros_per_block.std().item(),
                "fully_zero_blocks": fully_zero_blocks,
                "nm_pattern_matches": nm_patterns,
            }

        return results

    def compute_warp_efficiency(
        self,
        activations: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Analyze warp-level efficiency for GPU execution.

        In GPU warps (32 threads), if any thread has work, all threads
        must execute. Perfect efficiency requires all-zero or all-nonzero
        warp lanes.

        Args:
            activations: Activation tensor

        Returns:
            Warp efficiency metrics
        """
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        flat = activations.view(activations.shape[0], -1)
        batch_size, dim = flat.shape

        # Pad to warp boundary
        if dim % self.warp_size != 0:
            pad_size = self.warp_size - (dim % self.warp_size)
            flat = torch.nn.functional.pad(flat, (0, pad_size))

        num_warps = flat.shape[1] // self.warp_size
        warps = flat.view(batch_size, num_warps, self.warp_size)

        # Classify warps
        all_zero = (warps == 0).all(dim=-1)
        all_nonzero = (warps != 0).all(dim=-1)
        mixed = ~(all_zero | all_nonzero)

        # Efficiency: work done / work scheduled
        # For mixed warps, we do full warp work but only some elements are useful
        active_elements = (warps != 0).float()
        work_per_warp = active_elements.sum(dim=-1) / self.warp_size

        # For all-zero warps, we can skip (efficiency = 1)
        # For all-nonzero, all work is useful (efficiency = 1)
        # For mixed, efficiency = active_elements / warp_size
        efficiency = torch.where(
            all_zero | all_nonzero,
            torch.ones_like(work_per_warp),
            work_per_warp
        )

        return {
            "all_zero_warps": all_zero.float().mean().item(),
            "all_nonzero_warps": all_nonzero.float().mean().item(),
            "mixed_warps": mixed.float().mean().item(),
            "mean_warp_efficiency": efficiency.mean().item(),
            "skippable_warps": all_zero.float().mean().item(),  # Can skip computation
        }

    def memory_bandwidth_analysis(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Estimate memory bandwidth utilization.

        Many operations are memory-bound on modern GPUs.
        Sparsity can reduce memory traffic if implemented correctly.

        Args:
            model: Model to analyze
            input_shape: Shape of input tensor
            num_runs: Number of profiling runs

        Returns:
            Memory bandwidth metrics
        """
        model = model.to(self.device)
        model.eval()

        dummy_input = torch.randint(0, 1000, input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Profile
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / num_runs

        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        else:
            peak_memory = 0

        # Count parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        return {
            "inference_time_ms": elapsed * 1000,
            "peak_memory_gb": peak_memory,
            "param_memory_mb": param_bytes / 1e6,
            "estimated_bandwidth_gbps": (param_bytes / 1e9) / elapsed if elapsed > 0 else 0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.results:
            return {}

        return {
            "num_benchmarks": len(self.results),
            "mean_sparsity": np.mean([r["sparsity"] for r in self.results]),
            "mean_dense_time_ms": np.mean([r["dense_time_ms"] for r in self.results]),
            "mean_masked_time_ms": np.mean([r["masked_time_ms"] for r in self.results]),
            "mean_coo_time_ms": np.mean([r["coo_sparse_time_ms"] for r in self.results]),
            "sparse_faster_count": sum(1 for r in self.results if r["is_sparse_faster"]),
        }


def run_kernel_ablation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, Any]:
    """
    Comprehensive kernel analysis across different sparsity levels.

    Args:
        model: Model to analyze
        dataloader: Data loader
        device: Device to use
        max_batches: Maximum batches to analyze

    Returns:
        Complete kernel analysis results
    """
    analyzer = SparsityKernelAnalyzer(device)

    all_results = {
        "matmul_benchmarks": [],
        "structured_sparsity": [],
        "warp_efficiency": [],
        "memory_analysis": None,
    }

    # Collect activations and analyze
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            _, _, aux = model(input_ids, return_spike_info=True)

            spike_info = aux.get("spike_info", {})

            for layer_idx, layer_spikes in spike_info.items():
                for step_info in layer_spikes:
                    k_spikes = step_info["k_spikes"]

                    # Flatten for analysis
                    flat_act = k_spikes.view(-1, k_spikes.shape[-1])

                    # Sample for benchmarking (limit size)
                    sample_act = flat_act[:32]
                    weight = torch.randn(
                        flat_act.shape[-1], flat_act.shape[-1], device=device
                    )

                    # Matmul benchmark
                    matmul_result = analyzer.benchmark_dense_vs_sparse_matmul(
                        weight, sample_act, num_runs=50
                    )
                    matmul_result["layer"] = layer_idx
                    matmul_result["batch_idx"] = batch_idx
                    all_results["matmul_benchmarks"].append(matmul_result)

                    # Structured sparsity
                    struct_result = analyzer.analyze_structured_sparsity_potential(flat_act)
                    struct_result["layer"] = layer_idx
                    struct_result["batch_idx"] = batch_idx
                    all_results["structured_sparsity"].append(struct_result)

                    # Warp efficiency
                    warp_result = analyzer.compute_warp_efficiency(flat_act)
                    warp_result["layer"] = layer_idx
                    warp_result["batch_idx"] = batch_idx
                    all_results["warp_efficiency"].append(warp_result)

    # Memory analysis
    all_results["memory_analysis"] = analyzer.memory_bandwidth_analysis(
        model, (1, 128), num_runs=20
    )

    # Add summary
    all_results["summary"] = analyzer.get_summary()

    return all_results
