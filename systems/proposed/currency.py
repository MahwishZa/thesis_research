"""Temporal currency component of SCAF."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from math import pow
from typing import Callable, Optional

from ..interfaces.evidence import Evidence


class CurrencyState(str, Enum):
    HARD_INVALID = "hard_invalid"
    TIME_INVARIANT = "time_invariant"
    SUPERSEDED = "superseded"
    CURRENT = "current"
    UNKNOWN_DATE = "unknown_date"


@dataclass(frozen=True)
class CurrencyResult:
    score: float
    state: CurrencyState


class CurrencyPolicy:
    """Calculate the SCAF currency function."""

    def __init__(
        self,
        *,
        half_life_days: Optional[float],
        superseded_factor: Optional[float],
        is_time_invariant: Callable[[str], bool],
    ) -> None:

        self.half_life_days = half_life_days
        self.superseded_factor = superseded_factor
        self.is_time_invariant = is_time_invariant

    def score(
        self,
        evidence: Evidence,
        *,
        question: str,
        question_date: date,
    ) -> CurrencyResult:

        if evidence.is_hard_invalid():
            return CurrencyResult(
                score=0.0,
                state=CurrencyState.HARD_INVALID,
            )

        if self.is_time_invariant(question):
            return CurrencyResult(
                score=1.0,
                state=CurrencyState.TIME_INVARIANT,
            )

        if evidence.publication_date is None:
            raise ValueError(
                "Non-time-invariant evidence has no publication date."
            )

        if self.half_life_days is None:
            raise ValueError(
                "half_life_days has not been configured."
            )

        if self.half_life_days <= 0:
            raise ValueError(
                "half_life_days must be positive."
            )

        age_days = max(
            0,
            (
                question_date
                - evidence.publication_date
            ).days,
        )

        decay = pow(
            2.0,
            -age_days / self.half_life_days,
        )

        if evidence.supersession_pointer is not None:

            if self.superseded_factor is None:
                raise ValueError(
                    "superseded_factor has not been configured."
                )

            return CurrencyResult(
                score=(
                    self.superseded_factor
                    * decay
                ),
                state=CurrencyState.SUPERSEDED,
            )

        return CurrencyResult(
            score=decay,
            state=CurrencyState.CURRENT,
        )