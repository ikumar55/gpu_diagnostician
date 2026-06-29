"""Lightweight data containers shared across pipeline stages."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class KernelEvent:
    name: str
    duration_us: float
    category: str  # "compute" | "memory" | "other"


@dataclass
class RawTrace:
    """Output of Stage 1 (Collector). Holds everything captured during the run."""
    step_times_ms: List[float] = field(default_factory=list)
    cpu_util_samples: List[float] = field(default_factory=list)
    gpu_util_samples: List[float] = field(default_factory=list)  # empty if no CUDA
    kernel_events: List[KernelEvent] = field(default_factory=list)  # empty if no CUDA
    sync_event_count: int = 0       # D2H memcpy events; 0 if no CUDA
    num_steps: int = 0
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    peak_memory_mb: Optional[float] = None  # GPU memory; None if no CUDA
    cuda_available: bool = False


@dataclass
class Diagnosis:
    """Output of Stage 3 (DiagnosisEngine). One entry per fired rule."""
    rule_name: str
    confidence_pct: float           # 0–100
    signals_matched: int
    signals_total: int
    evidence: Dict[str, str] = field(default_factory=dict)
    estimated_impact_pct: Optional[float] = None


@dataclass
class Recommendation:
    """Output of Stage 4 (Recommender). One entry per diagnosis."""
    diagnosis: Diagnosis
    fix_summary: str
    patch_before: Optional[str] = None
    patch_after: Optional[str] = None
    impact_note: str = ""
