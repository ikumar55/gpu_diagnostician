"""Stage 4 — Recommender.

Maps each Diagnosis to a concrete, actionable Recommendation: a fix summary,
a before/after code patch where applicable, and a plain-English impact note.
"""

from typing import List
from .types import Diagnosis, Recommendation


class Recommender:
    """Produce a Recommendation for every fired Diagnosis."""

    def recommend(self, diagnoses: List[Diagnosis]) -> List[Recommendation]:
        """Map diagnoses to recommendations in the same rank order.

        Args:
            diagnoses: Output of DiagnosisEngine.diagnose().

        Returns:
            List of Recommendation objects, preserving input rank order.

        Raises:
            NotImplementedError: Stage 4 not yet implemented.
        """
        raise NotImplementedError("Stage 4 (Recommender) not yet implemented")
