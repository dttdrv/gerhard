"""
ASNN-Goose Evaluation and Analysis

This module contains:
- SpikeAnalyzer: Spike density, flicker rate, warp alignment
- SpikingBrainValidator: V15 information encoding validation
- Benchmarks: Perplexity, copy task, retrieval task
- StabilityTests: TTT drift analysis
"""

from .spike_analysis import SpikeAnalyzer, SpikeStatistics
from .spiking_brain import (
    SpikingBrainValidator,
    SpikingBrainValidation,
    SpikeHealthMetrics,
    MutualInformationEstimator,
    RepresentationAnalyzer,
)
from .benchmarks import (
    evaluate_perplexity,
    evaluate_copy_task,
    evaluate_retrieval_task,
    run_all_benchmarks,
)
from .stability_tests import TTTStabilityAnalyzer

__all__ = [
    "SpikeAnalyzer",
    "SpikeStatistics",
    "SpikingBrainValidator",
    "SpikingBrainValidation",
    "SpikeHealthMetrics",
    "MutualInformationEstimator",
    "RepresentationAnalyzer",
    "evaluate_perplexity",
    "evaluate_copy_task",
    "evaluate_retrieval_task",
    "run_all_benchmarks",
    "TTTStabilityAnalyzer",
]
