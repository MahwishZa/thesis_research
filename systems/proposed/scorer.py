"""Admission score for the Temporal Filter.

    A(s) = (1 - lambda) * rho(s) + lambda * T(s, q, t_q)

    admit if A(s) >= theta

``rho`` is the rank-normalised reranker score the baseline already uses, and
``T`` is the temporal score (temporal.py). Three decisions are worth
stating, because each removes a parameter or a confound rather than just
simplifying code:

1.  **One weight, not two.** A pair of free weights has a redundant degree
    of freedom: scaling both moves the score without changing the ranking,
    and the threshold then absorbs the difference. A single ``lambda`` in
    [0, 1] says exactly one thing - how much of the score is temporal - and
    ``lambda = 0`` recovers pure relevance, which is the built-in ablation
    that isolates the Temporal Filter's contribution.

2.  **No extra normalisation.** ``rho`` is rank-normalised to [0, 1] by
    construction and ``T`` is an exponential decay in (0, 1], so A(s) is in
    [0, 1] and theta is directly interpretable. Rescaling either would put a
    second, hidden knob next to lambda.

3.  **Two components only.** Nothing beyond relevance and temporal weight
    is part of this experiment. (Earlier exploratory work looked at a
    four-component score with an entailment-based "support" signal and a
    source-authority term; both needed extra models the thesis does not
    have, and are archived - see ``_archive/README.md``.)

``lambda``, ``theta`` and the half-life ``H`` are the only tunable
quantities, and all three are fitted on the validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..interfaces.evidence import Candidate


@dataclass(frozen=True)
class AdmissionScore:
    """Components and total for one candidate."""

    total: float
    relevance: float
    temporal: float
    temporal_weight: float


class AdmissionScorer:
    """Combine the relevance and temporal signals into one admission score."""

    def __init__(self, temporal_weight: Optional[float]) -> None:
        """
        Args:
            temporal_weight: lambda in [0, 1]. Unresolved by default and
                fitted on the validation split; 0.0 is a legitimate setting
                (the pure-relevance ablation) and is distinct from "unset".
        """

        if temporal_weight is None:
            raise ValueError(
                "temporal_weight (lambda) is unresolved. Fit it on the "
                "validation split; it is not given a default. Pass 0.0 "
                "explicitly for the pure-relevance ablation."
            )

        if not 0.0 <= float(temporal_weight) <= 1.0:
            raise ValueError("temporal_weight must be in [0, 1].")

        self.temporal_weight = float(temporal_weight)

    @staticmethod
    def normalize_rank(rank: int, candidate_count: int) -> float:
        """Map a reranker rank onto [0, 1], best rank scoring 1.0.

        Rank-based rather than min-max over raw reranker scores, so that one
        threshold is comparable across queries whose score ranges differ.

        Two requirements follow from this, and evidence freezing must satisfy both when
        it builds the cached candidate set:

        * **Ranks are 1..N contiguous.** Injecting the evaluation pair into a
          retrieved set means the whole set has to be re-ranked afterwards,
          not given the pair's original ranks. A gap or a duplicate raises
          here rather than silently distorting rho.
        * **N is the same for every item.** rho is a within-set rank, so the
          best passage always scores 1.0 and the worst 0.0 regardless of how
          relevant either actually is. theta is therefore only comparable
          across items when the candidate-set size is fixed - which
          specification §15 already requires across arms, and which must
          also hold across items.
        """

        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")

        if not 1 <= rank <= candidate_count:
            raise ValueError("rank must be between 1 and candidate_count.")

        if candidate_count == 1:
            return 1.0

        return 1.0 - ((rank - 1) / (candidate_count - 1))

    def score(
        self,
        candidate: Candidate,
        *,
        temporal: float,
        candidate_count: int,
    ) -> AdmissionScore:

        if not 0.0 <= temporal <= 1.0:
            raise ValueError("temporal must be in [0, 1].")

        relevance = self.normalize_rank(
            candidate.rerank_rank,
            candidate_count,
        )

        total = (
            (1.0 - self.temporal_weight) * relevance
            + self.temporal_weight * temporal
        )

        return AdmissionScore(
            total=total,
            relevance=relevance,
            temporal=temporal,
            temporal_weight=self.temporal_weight,
        )
