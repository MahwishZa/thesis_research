"""Stage 1 of the study (a result of record: recency-aware admission did not beat standard retrieval on dev).

Admission by a relevance score plus a recency term, instead of relevance alone.

    evidence.py   the ``Evidence`` and ``Candidate`` records that carry a passage and its dates to the admission score
    temporal.py   T(s, q, t_q), the temporal score (evidence age against the question date)
    scorer.py     A(s) = (1 - lambda) * rho(s) + lambda * T(s), the combined admission score

``experiments/medchange/arms.py`` calls them for the stage-1 arms, so the formula exists in exactly one place. The current systems
(``experiments/medchange/rag2.py``) work on plain dictionaries and do not use this package. The threshold-and-budget form of the policy
(``admission.py``) belonged to the original framework, removed on 2026-10-06 (Git history, commit 5e03540). Since 2026-10-05 the proposed
system of the thesis is the evidence-criteria verification of ``experiments/medchange/rag2.py``, not this package. Until 2026-10-12 the
modules lived in two packages named ``src.common`` and ``src.temporal_filter``.
"""

from .evidence import Candidate, Evidence
from .scorer import AdmissionScore, AdmissionScorer
from .temporal import TemporalPolicy, TemporalResult, TemporalState

__all__ = ["AdmissionScore", "AdmissionScorer", "Candidate", "Evidence", "TemporalPolicy", "TemporalResult", "TemporalState"]
