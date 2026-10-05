"""Archived: the threshold-and-budget form of the Temporal Filter (``admission.py``).

Its score and recency term stay active in ``src/temporal_filter/{scorer,temporal}.py``.
"""

from .admission import (
    AdmissionConfig,
    OutputState,
    PassageDecision,
    TemporalFilterPolicy,
    TemporalFilterSystem,
)

__all__ = ["AdmissionConfig", "OutputState", "PassageDecision", "TemporalFilterPolicy", "TemporalFilterSystem"]
