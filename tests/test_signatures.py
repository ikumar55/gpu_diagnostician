"""CUDA-gated signature tests.

These tests verify that each broken/healthy script produces the expected
feature fingerprint on a real CUDA GPU. They are SKIPPED locally on machines
without CUDA (including this Mac) and run for real on the GPU box.

They do NOT test diagnosis rules — only that the capture harness records
the right raw numbers. Rule firing is Milestone 2.

Prerequisites:
  Run `python validation/runner.py` first to populate artifacts/*.json.
  The tests read those JSON files; they do not re-run the scripts.
"""

import json
from pathlib import Path

import pytest
import torch

# Skip the entire module when no CUDA device is present.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required — run on the GPU box after `python validation/runner.py`",
)

_ARTIFACTS = Path(__file__).parent.parent / "artifacts"


def _load(name: str) -> dict:
    path = _ARTIFACTS / f"{name}_features.json"
    if not path.exists():
        pytest.skip(
            f"Artifact not found: {path}. "
            "Run `python validation/runner.py` first."
        )
    with open(path) as f:
        return json.load(f)


# ── Rule 1: Dataloader starvation ─────────────────────────────────────────────

class TestBrokenDataloader:
    """broken_dataloader should show low GPU util + high CPU util + high idle."""

    def setup_method(self):
        self.f = _load("broken_dataloader")

    def test_gpu_util_is_low(self):
        assert self.f["gpu_util_pct"] is not None, "gpu_util_pct not captured"
        assert self.f["gpu_util_pct"] < 60, (
            f"Expected gpu_util_pct < 60, got {self.f['gpu_util_pct']:.1f}. "
            "Tune broken_dataloader.py CPU burn or threshold."
        )

    def test_cpu_util_is_high(self):
        assert self.f["cpu_util_pct"] is not None, "cpu_util_pct not captured"
        assert self.f["cpu_util_pct"] > 85, (
            f"Expected cpu_util_pct > 85, got {self.f['cpu_util_pct']:.1f}"
        )

    def test_gpu_idle_is_high(self):
        assert self.f["gpu_idle_fraction"] is not None, "gpu_idle_fraction not captured"
        assert self.f["gpu_idle_fraction"] > 0.20, (
            f"Expected gpu_idle_fraction > 0.20, got {self.f['gpu_idle_fraction']:.3f}"
        )


# ── Rule 2: Work too small ────────────────────────────────────────────────────

class TestBrokenBatchsize:
    """broken_batchsize should show low GPU util + high tiny-kernel fraction."""

    def setup_method(self):
        self.f = _load("broken_batchsize")

    def test_gpu_util_is_low(self):
        assert self.f["gpu_util_pct"] is not None
        assert self.f["gpu_util_pct"] < 60, (
            f"Expected gpu_util_pct < 60, got {self.f['gpu_util_pct']:.1f}"
        )

    def test_tiny_kernel_fraction_is_high(self):
        assert self.f["tiny_kernel_fraction"] is not None, (
            "tiny_kernel_fraction not captured"
        )
        assert self.f["tiny_kernel_fraction"] > 0.50, (
            f"Expected tiny_kernel_fraction > 0.50, got "
            f"{self.f['tiny_kernel_fraction']:.3f}"
        )

    def test_batch_size_is_one(self):
        assert self.f["batch_size"] == 1


# ── Rule 3: Host synchronisation stall ───────────────────────────────────────

class TestBrokenSync:
    """broken_sync should show ≥1 sync event per step + elevated idle fraction."""

    def setup_method(self):
        self.f = _load("broken_sync")

    def test_sync_events_per_step_ge_one(self):
        assert self.f["sync_events_per_step"] is not None, (
            "sync_events_per_step not captured"
        )
        assert self.f["sync_events_per_step"] >= 1, (
            f"Expected sync_events_per_step >= 1, got {self.f['sync_events_per_step']:.2f}"
        )

    def test_gpu_idle_is_elevated(self):
        assert self.f["gpu_idle_fraction"] is not None
        assert self.f["gpu_idle_fraction"] > 0.20, (
            f"Expected gpu_idle_fraction > 0.20, got {self.f['gpu_idle_fraction']:.3f}"
        )


# ── Rule 4: Memory-bandwidth-bound ───────────────────────────────────────────

class TestBrokenMemory:
    """broken_memory should show high GPU util but dominated by memory-bound kernels."""

    def setup_method(self):
        self.f = _load("broken_memory")

    def test_gpu_util_is_high(self):
        assert self.f["gpu_util_pct"] is not None
        assert self.f["gpu_util_pct"] > 70, (
            f"Expected gpu_util_pct > 70 (GPU IS busy), got {self.f['gpu_util_pct']:.1f}"
        )

    def test_mem_bound_fraction_is_high(self):
        assert self.f["mem_bound_fraction"] is not None, (
            "mem_bound_fraction not captured"
        )
        # All ops are elementwise — should be overwhelmingly memory-bound.
        assert self.f["mem_bound_fraction"] > 0.75, (
            f"Expected mem_bound_fraction > 0.75, got {self.f['mem_bound_fraction']:.3f}"
        )


# ── Rule 5: Compute-bound / healthy ──────────────────────────────────────────

class TestHealthy:
    """healthy should show high GPU util + low memory-bound fraction (good state)."""

    def setup_method(self):
        self.f = _load("healthy")

    def test_gpu_util_is_high(self):
        assert self.f["gpu_util_pct"] is not None
        assert self.f["gpu_util_pct"] > 80, (
            f"Expected gpu_util_pct > 80, got {self.f['gpu_util_pct']:.1f}"
        )

    def test_mem_bound_fraction_is_low(self):
        assert self.f["mem_bound_fraction"] is not None
        # Threshold tuned from real run: Adam weight-update kernels are
        # elementwise (memory-bound by heuristic), so even a healthy model
        # sits around 0.50-0.55. We assert it's clearly below broken_memory's
        # ~0.85+ rather than chasing an unreachable ideal.
        assert self.f["mem_bound_fraction"] < 0.65, (
            f"Expected mem_bound_fraction < 0.65 (healthy baseline with Adam), "
            f"got {self.f['mem_bound_fraction']:.3f}"
        )

    def test_no_per_step_syncs(self):
        seps = self.f.get("sync_events_per_step")
        if seps is not None:
            assert seps < 1, (
                f"Healthy script should have no per-step syncs, got {seps:.2f}"
            )
