"""Stage 2 — Feature Extractor.

Boils a RawTrace down to the ~10 scalar features the diagnosis rules consume
(taxonomy §1). GPU-derived features are None when the trace contains no CUDA data.

Kernel categorisation heuristic (v0 approximation, not a true roofline):
  compute-bound  →  kernels whose names contain: gemm, matmul, conv, bmm
  memory-bound   →  everything else (elementwise, norm, activation, copy, etc.)
This approximation is documented in the README; exact FLOPs-per-byte is future work.
"""

from typing import Dict, Optional
from .types import RawTrace


Features = Dict[str, Optional[float]]


class FeatureExtractor:
    """Reduce a RawTrace to a flat feature dict.

    Returns:
        dict with keys matching taxonomy §1:
          gpu_util_pct, cpu_util_pct, gpu_idle_fraction,
          tiny_kernel_fraction, mem_bound_fraction,
          sync_events_per_step, batch_size, num_workers.
        GPU-derived keys are None when trace.cuda_available is False.
    """

    def extract(self, trace: RawTrace) -> Features:
        """Extract features from a RawTrace.

        Raises:
            NotImplementedError: Stage 2 not yet implemented.
        """
        raise NotImplementedError("Stage 2 (FeatureExtractor) not yet implemented")
