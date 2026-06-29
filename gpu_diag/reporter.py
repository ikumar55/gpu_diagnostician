"""Stage 5 — Reporter.

Formats the ranked Recommendation list into a clean terminal report and
optionally saves an HTML/Markdown version. The report must be fully usable
without any LLM; the LLM layer is optional gift-wrap.
"""

from typing import List
from .types import Recommendation


class Reporter:
    """Render a ranked list of Recommendations as a human-readable report."""

    def report(self, recommendations: List[Recommendation]) -> str:
        """Format recommendations into a plain-text terminal report.

        Args:
            recommendations: Output of Recommender.recommend().

        Returns:
            A formatted string ready for print() or file output.

        Raises:
            NotImplementedError: Stage 5 not yet implemented.
        """
        raise NotImplementedError("Stage 5 (Reporter) not yet implemented")
