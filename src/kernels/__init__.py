"""
ASNN-Goose Kernel Analysis

This module contains:
- SparsityKernelAnalyzer: Benchmark dense vs sparse operations
- Structured sparsity pattern analysis
- Memory bandwidth analysis
"""

from .sparsity_analysis import (
    SparsityKernelAnalyzer,
    run_kernel_ablation,
)

__all__ = [
    "SparsityKernelAnalyzer",
    "run_kernel_ablation",
]
