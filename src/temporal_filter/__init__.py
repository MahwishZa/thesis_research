"""The Temporal Filter of stage 1 (a result of record: it did not beat standard retrieval on dev).

Admission by a relevance score plus a recency term, instead of relevance alone.

    temporal.py   T(s, q, t_q), the temporal score (evidence age against the question date)
    scorer.py     A(s) = (1 - lambda) * rho(s) + lambda * T(s), the combined admission score

``experiments/medchange/arms.py`` calls both, so the formula exists in exactly one place. The threshold-and-budget
form of the policy (``admission.py``) belonged to the archived framework and lives in
``_archive/alzheimers_framework/src/temporal_filter/``. Since 2026-10-05 the proposed system of the thesis is the
evidence-criteria verification of ``experiments/medchange/rag2.py``, not this package.
"""

from .scorer import AdmissionScore, AdmissionScorer
from .temporal import TemporalPolicy, TemporalResult, TemporalState

__all__ = ["AdmissionScore", "AdmissionScorer", "TemporalPolicy", "TemporalResult", "TemporalState"]
