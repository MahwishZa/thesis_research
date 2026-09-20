"""The Temporal Filter: the thesis's proposed method.

This package has one job: admit evidence using a temporal score alongside
relevance, instead of relevance alone.

    temporal.py   T(s, q, t_q), the temporal score (evidence age vs. the
                  question date)
    scorer.py     A(s) = (1 - lambda) * rho(s) + lambda * T(s), the combined
                  admission score
    admission.py  threshold, context budget, output state - ties the score
                  to an admit/reject decision

Earlier, secondary machinery (contested-evidence detection, answer
verification) that was never part of this experiment has been moved to
``_archive/`` - see ``_archive/README.md``. Nothing here imports it.
"""

from .admission import (
    AdmissionConfig,
    OutputState,
    PassageDecision,
    TemporalFilterPolicy,
    TemporalFilterSystem,
)
from .temporal import TemporalPolicy, TemporalResult, TemporalState
from .scorer import AdmissionScore, AdmissionScorer

__all__ = [
    "AdmissionConfig",
    "AdmissionScore",
    "AdmissionScorer",
    "OutputState",
    "PassageDecision",
    "TemporalFilterPolicy",
    "TemporalFilterSystem",
    "TemporalPolicy",
    "TemporalResult",
    "TemporalState",
]
