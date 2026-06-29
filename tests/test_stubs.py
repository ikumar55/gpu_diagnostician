"""Tests for the 5 pipeline stage stubs — always runs, no CUDA required.

Verifies that every stage:
  1. Is importable without errors.
  2. Can be instantiated with no arguments.
  3. Raises NotImplementedError when its primary method is called.

Also verifies that the shared type containers in gpu_diag/types.py are
importable dataclasses with the expected fields.
"""

import dataclasses
import pytest

from gpu_diag.types import RawTrace, Diagnosis, Recommendation, KernelEvent
from gpu_diag.collector import Collector
from gpu_diag.features import FeatureExtractor
from gpu_diag.engine import DiagnosisEngine
from gpu_diag.recommender import Recommender
from gpu_diag.reporter import Reporter


# ── Type containers ───────────────────────────────────────────────────────────

class TestTypes:
    def test_raw_trace_is_dataclass(self):
        assert dataclasses.is_dataclass(RawTrace)

    def test_raw_trace_instantiable(self):
        t = RawTrace()
        assert t.step_times_ms == []
        assert t.cuda_available is False

    def test_raw_trace_has_expected_fields(self):
        fields = {f.name for f in dataclasses.fields(RawTrace)}
        assert fields >= {
            "step_times_ms", "cpu_util_samples", "gpu_util_samples",
            "kernel_events", "sync_event_count", "num_steps",
            "batch_size", "num_workers", "peak_memory_mb", "cuda_available",
        }

    def test_diagnosis_is_dataclass(self):
        assert dataclasses.is_dataclass(Diagnosis)

    def test_diagnosis_instantiable(self):
        d = Diagnosis(
            rule_name="test", confidence_pct=90.0,
            signals_matched=3, signals_total=4,
        )
        assert d.rule_name == "test"

    def test_recommendation_is_dataclass(self):
        assert dataclasses.is_dataclass(Recommendation)

    def test_recommendation_instantiable(self):
        d = Diagnosis(rule_name="x", confidence_pct=50.0,
                      signals_matched=1, signals_total=2)
        r = Recommendation(diagnosis=d, fix_summary="do something")
        assert r.fix_summary == "do something"

    def test_kernel_event_is_dataclass(self):
        assert dataclasses.is_dataclass(KernelEvent)


# ── Stage 1: Collector ────────────────────────────────────────────────────────

class TestCollector:
    def test_instantiable_no_args(self):
        c = Collector()
        assert c.steps == 20
        assert c.warmup == 5

    def test_instantiable_with_args(self):
        c = Collector(steps=10, warmup=2)
        assert c.steps == 10

    def test_collect_raises_not_implemented(self):
        c = Collector()
        with pytest.raises(NotImplementedError):
            c.collect(lambda: None)


# ── Stage 2: FeatureExtractor ─────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_instantiable_no_args(self):
        FeatureExtractor()

    def test_extract_raises_not_implemented(self):
        fe = FeatureExtractor()
        with pytest.raises(NotImplementedError):
            fe.extract(RawTrace())


# ── Stage 3: DiagnosisEngine ──────────────────────────────────────────────────

class TestDiagnosisEngine:
    def test_instantiable_no_args(self):
        DiagnosisEngine()

    def test_uses_default_thresholds(self):
        from gpu_diag.config import THRESHOLDS
        engine = DiagnosisEngine()
        assert engine.thresholds is THRESHOLDS

    def test_instantiable_with_custom_thresholds(self):
        from gpu_diag.config import Thresholds
        custom = Thresholds(gpu_util_low=50.0)
        engine = DiagnosisEngine(thresholds=custom)
        assert engine.thresholds.gpu_util_low == 50.0

    def test_diagnose_raises_not_implemented(self):
        engine = DiagnosisEngine()
        with pytest.raises(NotImplementedError):
            engine.diagnose({})


# ── Stage 4: Recommender ─────────────────────────────────────────────────────

class TestRecommender:
    def test_instantiable_no_args(self):
        Recommender()

    def test_recommend_raises_not_implemented(self):
        r = Recommender()
        with pytest.raises(NotImplementedError):
            r.recommend([])


# ── Stage 5: Reporter ─────────────────────────────────────────────────────────

class TestReporter:
    def test_instantiable_no_args(self):
        Reporter()

    def test_report_raises_not_implemented(self):
        r = Reporter()
        with pytest.raises(NotImplementedError):
            r.report([])
