"""Shared building blocks of the stage-1 arms: the evidence types and the Temporal Filter's score.

The realigned study (adapted RAG² + evidence-criteria verification) lives in ``experiments/medchange/``.
This package holds only what that code still imports: ``src.common.evidence`` (the ``Candidate`` and
``Evidence`` types) and ``src.temporal_filter`` (the recency term and admission score of stage 1, a result of
record). The original three-arm framework built around them was removed on 2026-10-06 (Git history, commit 5e03540).
"""

from .common import Candidate, Evidence, ExperimentResult

__all__ = ["Candidate", "Evidence", "ExperimentResult"]
