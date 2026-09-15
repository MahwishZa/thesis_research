"""Temporal currency component of SCAF.

Implements gamma(s, q, t_q) from the experimental specification (section 25):

    gamma = 0                               if retracted or withdrawn
          = 1                               if psi(q) = 0  (time-invariant)
          = delta * 2^(-(t_q - date(s))/H)  if superseded
          = 2^(-(t_q - date(s))/H)          otherwise

Two behaviours here are specification requirements rather than implementation
convenience, and both are load-bearing:

1.  **Contested is evaluated before supersession** (specification section 29).
    A claim under live dispute must not be treated as resolved by the newer
    side. The caller therefore passes ``contested=True`` for evidence whose
    claim class is contested, and the supersession discount is skipped.
    Reversing this order would present an unresolved controversy as settled.

2.  **Superseded evidence is down-weighted, never deleted.** Hard exclusion is
    restricted to objectively established retraction or withdrawal.

Unresolved parameters (H, delta, psi) are ``None`` by default and raise on use.
They are not given invented values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Optional

from ..interfaces.evidence import Evidence


class CurrencyState(str, Enum):
    """Which branch of gamma produced the score."""

    HARD_INVALID = "hard_invalid"
    TIME_INVARIANT = "time_invariant"
    SUPERSEDED = "superseded"
    CURRENT = "current"
    UNKNOWN_DATE = "unknown_date"
    CONTESTED_NOT_SUPERSEDED = "contested_not_superseded"


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
        unknown_date_score: Optional[float] = None,
    ) -> None:
        """
        Args:
            half_life_days: H. Unresolved in the specification; must be set
                by the experiment configuration before use.
            superseded_factor: delta. Unresolved; must be set before use.
            is_time_invariant: psi(q). Returns True when the question is not
                temporally sensitive.
            unknown_date_score: score for evidence carrying no usable
                publication date. ``None`` means the caller has not decided,
                and such evidence is scored 0.0 and reported as UNKNOWN_DATE
                so it is visible in the decision record rather than silently
                treated as current. This is a REQUIRES-DECISION parameter:
                the Alzheimer's corpus contains records whose date could not
                be resolved, and how they are treated affects the temporal
                result.
        """
        self.half_life_days = half_life_days
        self.superseded_factor = superseded_factor
        self.is_time_invariant = is_time_invariant
        self.unknown_date_score = unknown_date_score

    def score(
        self,
        evidence: Evidence,
        *,
        question: str,
        question_date: date,
        contested: bool = False,
    ) -> CurrencyResult:
        """Return gamma for one passage.

        Args:
            contested: True when this passage's claim class is under an
                active, unresolved dispute. Set by the admission policy,
                which evaluates the contested condition BEFORE calling this
                method. When True the supersession discount is not applied.
        """

        # Branch 1: retraction / withdrawal is the only hard exclusion.
        if evidence.is_hard_invalid():
            return CurrencyResult(
                score=0.0,
                state=CurrencyState.HARD_INVALID,
            )

        # Branch 2: psi(q) = 0 -> currency does not apply.
        if self.is_time_invariant(question):
            return CurrencyResult(
                score=1.0,
                state=CurrencyState.TIME_INVARIANT,
            )

        # No usable date: recorded, never fatal. The corpus contains such
        # records, and aborting the run on one of them would make Stage 5
        # impossible to complete.
        if evidence.publication_date is None:
            return CurrencyResult(
                score=(
                    0.0
                    if self.unknown_date_score is None
                    else float(self.unknown_date_score)
                ),
                state=CurrencyState.UNKNOWN_DATE,
            )

        if self.half_life_days is None:
            raise ValueError(
                "half_life_days (H) is unresolved. Set it in the "
                "experiment configuration; it is not given a default."
            )

        if self.half_life_days <= 0:
            raise ValueError(
                "half_life_days must be positive."
            )

        age_days = max(
            0,
            (question_date - evidence.publication_date).days,
        )

        decay = 2.0 ** (-age_days / self.half_life_days)

        superseded = evidence.supersession_pointer is not None

        # Specification section 29: contested is evaluated BEFORE
        # supersession. A contested claim is not treated as resolved.
        if superseded and contested:
            return CurrencyResult(
                score=decay,
                state=CurrencyState.CONTESTED_NOT_SUPERSEDED,
            )

        if superseded:

            if self.superseded_factor is None:
                raise ValueError(
                    "superseded_factor (delta) is unresolved. Set it in "
                    "the experiment configuration; it is not given a "
                    "default."
                )

            return CurrencyResult(
                score=self.superseded_factor * decay,
                state=CurrencyState.SUPERSEDED,
            )

        return CurrencyResult(
            score=decay,
            state=CurrencyState.CURRENT,
        )
