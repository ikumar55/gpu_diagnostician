"""Stage 1 — Collector.

Wraps a user's training step function, runs torch.profiler for a fixed number
of steps (skipping warmup), and samples GPU/CPU utilisation alongside the trace.

CUDA-gated: GPU util sampling (pynvml) and CUDA kernel events from torch.profiler
only activate when torch.cuda.is_available(). Everything else runs on any device.
"""

from typing import Callable
from .types import RawTrace


class Collector:
    """Wrap a training step and collect a RawTrace.

    Args:
        steps: Number of steps to profile (after warmup).
        warmup: Number of leading steps to discard (avoids one-time JIT costs).
    """

    def __init__(self, steps: int = 20, warmup: int = 5) -> None:
        self.steps = steps
        self.warmup = warmup

    def collect(self, train_fn: Callable[[], None]) -> RawTrace:
        """Run train_fn for (warmup + steps) iterations and return a RawTrace.

        Args:
            train_fn: A zero-argument callable that executes one training step.

        Returns:
            RawTrace populated with step timings, utilisation samples, and — on
            CUDA — kernel events and sync counts.

        Raises:
            NotImplementedError: Stage 1 not yet implemented.
        """
        raise NotImplementedError("Stage 1 (Collector) not yet implemented")
