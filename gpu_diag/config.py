"""Diagnosis thresholds — the single source of truth for every rule's numeric limits.

All values are v0 starting points from the taxonomy spec. Tune them against the
validation suite (validation/runner.py) until each broken script trips its
intended rule and nothing else.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # ── Rule 1: Dataloader starvation ────────────────────────────────────────
    # GPU util below gpu_util_low → underutilised; CPU util above cpu_util_high
    # → CPU is the bottleneck; idle fraction above gpu_idle_high → stalled.
    gpu_util_low: float = 60.0
    cpu_util_high: float = 85.0
    gpu_idle_high: float = 0.20

    # ── Rule 2: Work too small (launch overhead) ──────────────────────────────
    # Kernel shorter than tiny_kernel_us (µs) counts as "tiny".
    # If more than tiny_kernel_fraction_high of all kernels are tiny → rule fires.
    tiny_kernel_us: float = 10.0
    tiny_kernel_fraction_high: float = 0.50
    batch_size_small: int = 4           # corroborating signal only

    # ── Rule 3: Host synchronisation stall ───────────────────────────────────
    # Any device→host sync inside the hot loop is a sync event.
    # sync_per_step_threshold: minimum events/step to trigger the rule.
    sync_per_step_threshold: int = 1

    # ── Rule 4: Memory-bandwidth-bound ───────────────────────────────────────
    # GPU IS busy (above gpu_util_busy) but time is dominated by memory-bound kernels.
    # Tuned: healthy.py (wide Linear + Adam) measures ~0.54 memory-bound because
    # Adam weight-update kernels are elementwise. Rule 4 threshold must be above
    # that so a mixed-but-healthy model doesn't falsely trigger.
    # broken_memory.py (ALL elementwise, no matmul) measures ~0.85+.
    gpu_util_busy: float = 70.0
    mem_bound_fraction_high: float = 0.75

    # ── Rule 5: Compute-bound / healthy ──────────────────────────────────────
    gpu_util_healthy: float = 80.0
    # Must be < mem_bound_fraction_high. Healthy model at ~0.54; threshold
    # gives headroom above that while staying below the Rule 4 threshold.
    mem_bound_fraction_healthy: float = 0.65


# Module-level sentinel; import this everywhere instead of constructing instances.
THRESHOLDS = Thresholds()
