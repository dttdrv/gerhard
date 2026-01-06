"""
Structured logging utilities for ASNN-Goose.

Uses JSON Lines format for easy parsing and analysis.
Kaggle-compatible without external dependencies.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys


class JSONLinesLogger:
    """
    Logger that writes metrics to JSON Lines format.

    Each line is a valid JSON object, making it easy to
    parse incrementally and analyze with pandas.

    Usage:
        logger = JSONLinesLogger("outputs/logs/training.jsonl")
        logger.log({"step": 1, "loss": 0.5, "lr": 1e-4})
    """

    def __init__(self, filepath: str, overwrite: bool = False):
        """
        Args:
            filepath: Path to log file
            overwrite: If True, overwrite existing file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        mode = 'w' if overwrite else 'a'
        self.file = open(self.filepath, mode, buffering=1)  # Line buffered

        # Log start
        self.log({
            "_event": "log_start",
            "_timestamp": datetime.now().isoformat(),
        })

    def log(self, data: Dict[str, Any]):
        """
        Log a dictionary as a JSON line.

        Args:
            data: Dictionary to log
        """
        # Add timestamp if not present
        if "_timestamp" not in data:
            data["_timestamp"] = datetime.now().isoformat()

        line = json.dumps(data, default=str)
        self.file.write(line + '\n')

    def close(self):
        """Close the log file."""
        self.log({"_event": "log_end", "_timestamp": datetime.now().isoformat()})
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def load(filepath: str) -> List[Dict[str, Any]]:
        """
        Load all log entries from a file.

        Args:
            filepath: Path to log file

        Returns:
            List of log entries
        """
        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries


class MetricsTracker:
    """
    Track metrics over training with running statistics.

    Provides:
    - Running mean and standard deviation
    - Best value tracking
    - Automatic logging to JSONLinesLogger
    """

    def __init__(
        self,
        logger: Optional[JSONLinesLogger] = None,
        window_size: int = 100,
    ):
        """
        Args:
            logger: Optional JSONLinesLogger for persistence
            window_size: Window size for running statistics
        """
        self.logger = logger
        self.window_size = window_size

        self.metrics: Dict[str, List[float]] = {}
        self.best_values: Dict[str, float] = {}
        self.best_steps: Dict[str, int] = {}
        self.step = 0

    def update(self, **kwargs):
        """
        Update metrics with new values.

        Args:
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.best_values[name] = float('inf')
                self.best_steps[name] = 0

            self.metrics[name].append(value)

            # Keep only recent window
            if len(self.metrics[name]) > self.window_size:
                self.metrics[name] = self.metrics[name][-self.window_size:]

            # Track best (assuming lower is better for loss-like metrics)
            if value < self.best_values[name]:
                self.best_values[name] = value
                self.best_steps[name] = self.step

        self.step += 1

        # Log if logger is set
        if self.logger is not None:
            log_data = {"step": self.step - 1}
            log_data.update(kwargs)
            self.logger.log(log_data)

    def get_mean(self, name: str) -> float:
        """Get running mean for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_std(self, name: str) -> float:
        """Get running standard deviation for a metric."""
        import math
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        values = self.metrics[name]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def get_best(self, name: str) -> tuple:
        """Get best value and step for a metric."""
        return self.best_values.get(name, float('inf')), self.best_steps.get(name, 0)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all tracked metrics."""
        summary = {}
        for name in self.metrics:
            summary[name] = {
                "current": self.metrics[name][-1] if self.metrics[name] else 0,
                "mean": self.get_mean(name),
                "std": self.get_std(name),
                "best": self.best_values[name],
                "best_step": self.best_steps[name],
            }
        return summary

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.best_values.clear()
        self.best_steps.clear()
        self.step = 0


def log_metrics(
    filepath: str,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
):
    """
    Convenience function to append metrics to a log file.

    Args:
        filepath: Path to log file
        metrics: Dictionary of metrics
        step: Optional step number
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    entry = metrics.copy()
    entry["_timestamp"] = datetime.now().isoformat()
    if step is not None:
        entry["step"] = step

    with open(filepath, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def create_experiment_log(
    experiment_name: str,
    config: Dict[str, Any],
    output_dir: str = "outputs/logs",
) -> JSONLinesLogger:
    """
    Create a logger for an experiment with config recorded.

    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        output_dir: Output directory for logs

    Returns:
        Configured JSONLinesLogger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.jsonl"
    filepath = Path(output_dir) / filename

    logger = JSONLinesLogger(str(filepath), overwrite=True)

    # Log experiment metadata
    logger.log({
        "_event": "experiment_start",
        "experiment_name": experiment_name,
        "config": config,
    })

    return logger


class PrintLogger:
    """
    Simple logger that prints to console and optionally to file.
    Useful for Kaggle notebooks where you want visible output.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        print_to_console: bool = True,
    ):
        self.filepath = filepath
        self.print_to_console = print_to_console

        if filepath:
            self.filepath = Path(filepath)
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.filepath, 'a')
        else:
            self.file = None

    def log(self, message: str):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"

        if self.print_to_console:
            print(formatted)

        if self.file:
            self.file.write(formatted + '\n')
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
