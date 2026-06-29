"""Tests for gpu_diag/config.py — always runs, no CUDA required."""

import dataclasses
import pytest
from gpu_diag.config import Thresholds, THRESHOLDS


class TestThresholdsDefaults:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(Thresholds)

    def test_is_frozen(self):
        t = Thresholds()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            t.gpu_util_low = 99.0

    def test_sentinel_is_thresholds_instance(self):
        assert isinstance(THRESHOLDS, Thresholds)

    # Rule 1
    def test_gpu_util_low_default(self):
        assert THRESHOLDS.gpu_util_low == 60.0

    def test_cpu_util_high_default(self):
        assert THRESHOLDS.cpu_util_high == 85.0

    def test_gpu_idle_high_default(self):
        assert THRESHOLDS.gpu_idle_high == 0.20

    # Rule 2
    def test_tiny_kernel_us_default(self):
        assert THRESHOLDS.tiny_kernel_us == 10.0

    def test_tiny_kernel_fraction_high_default(self):
        assert THRESHOLDS.tiny_kernel_fraction_high == 0.50

    def test_batch_size_small_default(self):
        assert THRESHOLDS.batch_size_small == 4

    # Rule 3
    def test_sync_per_step_threshold_default(self):
        assert THRESHOLDS.sync_per_step_threshold == 1

    # Rule 4
    def test_gpu_util_busy_default(self):
        assert THRESHOLDS.gpu_util_busy == 70.0

    def test_mem_bound_fraction_high_default(self):
        assert THRESHOLDS.mem_bound_fraction_high == 0.40

    # Rule 5
    def test_gpu_util_healthy_default(self):
        assert THRESHOLDS.gpu_util_healthy == 80.0

    def test_mem_bound_fraction_healthy_default(self):
        assert THRESHOLDS.mem_bound_fraction_healthy == 0.25


class TestThresholdsOverride:
    def test_custom_thresholds(self):
        custom = Thresholds(gpu_util_low=50.0, cpu_util_high=90.0)
        assert custom.gpu_util_low == 50.0
        assert custom.cpu_util_high == 90.0
        # Other fields keep defaults
        assert custom.gpu_idle_high == 0.20

    def test_custom_does_not_affect_sentinel(self):
        Thresholds(gpu_util_low=1.0)
        assert THRESHOLDS.gpu_util_low == 60.0

    def test_all_float_fields_are_float(self):
        float_fields = [
            "gpu_util_low", "cpu_util_high", "gpu_idle_high",
            "tiny_kernel_us", "tiny_kernel_fraction_high",
            "gpu_util_busy", "mem_bound_fraction_high",
            "gpu_util_healthy", "mem_bound_fraction_healthy",
        ]
        for field in float_fields:
            assert isinstance(getattr(THRESHOLDS, field), float), (
                f"{field} should be float, got {type(getattr(THRESHOLDS, field))}"
            )

    def test_batch_size_small_is_int(self):
        assert isinstance(THRESHOLDS.batch_size_small, int)

    def test_sync_per_step_threshold_is_int(self):
        assert isinstance(THRESHOLDS.sync_per_step_threshold, int)


class TestThresholdsSanity:
    """Values should be in taxonomy-consistent ranges."""

    def test_gpu_util_low_lt_busy_lt_healthy(self):
        t = THRESHOLDS
        assert t.gpu_util_low < t.gpu_util_busy < t.gpu_util_healthy

    def test_mem_bound_healthy_lt_high(self):
        assert THRESHOLDS.mem_bound_fraction_healthy < THRESHOLDS.mem_bound_fraction_high

    def test_idle_fraction_is_fraction(self):
        assert 0 < THRESHOLDS.gpu_idle_high < 1

    def test_tiny_kernel_fraction_is_fraction(self):
        assert 0 < THRESHOLDS.tiny_kernel_fraction_high < 1

    def test_mem_bound_fractions_are_fractions(self):
        assert 0 < THRESHOLDS.mem_bound_fraction_high < 1
        assert 0 < THRESHOLDS.mem_bound_fraction_healthy < 1
