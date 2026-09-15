"""SCAF evidence-admission score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..interfaces.evidence import Candidate


@dataclass(frozen=True)
class SCAFWeights:
    """Weights for the four SCAF components."""

    support: Optional[float] = None
    currency: Optional[float] = None
    rerank: Optional[float] = None
    authority: Optional[float] = None

    def validate(self) -> None:

        values = {
            "support": self.support,
            "currency": self.currency,
            "rerank": self.rerank,
            "authority": self.authority,
        }

        missing = [
            name
            for name, value in values.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                "SCAF weights are unresolved: "
                + ", ".join(missing)
            )


@dataclass(frozen=True)
class SCAFScore:
    """Individual SCAF components and total score."""

    total: float
    support: float
    currency: float
    rerank: float
    authority: float


class SCAFScorer:
    """Calculate the SCAF admission score."""

    def __init__(
        self,
        weights: SCAFWeights,
    ) -> None:

        weights.validate()
        self.weights = weights

    @staticmethod
    def normalize_rank(
        rank: int,
        candidate_count: int,
    ) -> float:

        if candidate_count <= 0:
            raise ValueError(
                "candidate_count must be positive."
            )

        if not 1 <= rank <= candidate_count:
            raise ValueError(
                "rank must be between 1 and candidate_count."
            )

        if candidate_count == 1:
            return 1.0

        return 1.0 - (
            (rank - 1)
            / (candidate_count - 1)
        )

    def score(
        self,
        candidate: Candidate,
        *,
        support: float,
        currency: float,
        authority: float,
        candidate_count: int,
    ) -> SCAFScore:

        for name, value in (
            ("support", support),
            ("currency", currency),
            ("authority", authority),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1]."
                )

        rerank = self.normalize_rank(
            candidate.rerank_rank,
            candidate_count,
        )

        total = (
            self.weights.support * support
            + self.weights.currency * currency
            + self.weights.rerank * rerank
            + self.weights.authority * authority
        )

        return SCAFScore(
            total=total,
            support=support,
            currency=currency,
            rerank=rerank,
            authority=authority,
        )