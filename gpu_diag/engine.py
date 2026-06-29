"""Stage 3 — Diagnosis Engine.

Applies the five deterministic rules from taxonomy §3 to the feature dict.
Each rule produces a Diagnosis with a confidence score and evidence checklist.
Results are returned ranked by estimated recoverable impact (taxonomy §4).
"""

from typing import Dict, List, Optional
from .types import Diagnosis
from .config import THRESHOLDS, Thresholds


class DiagnosisEngine:
    """Apply all five rules and return ranked diagnoses.

    Args:
        thresholds: Override the default THRESHOLDS for tuning. Defaults to the
                    module-level sentinel, which uses taxonomy v0 values.
    """

    def __init__(self, thresholds: Thresholds = THRESHOLDS) -> None:
        self.thresholds = thresholds

    def diagnose(self, features: Dict[str, Optional[float]]) -> List[Diagnosis]:
        """Run all rules against features and return ranked Diagnosis list.

        Args:
            features: Output of FeatureExtractor.extract().

        Returns:
            List of Diagnosis objects, highest estimated impact first.
            May be empty if no rule fires.

        Raises:
            NotImplementedError: Stage 3 not yet implemented.
        """
        raise NotImplementedError("Stage 3 (DiagnosisEngine) not yet implemented")
