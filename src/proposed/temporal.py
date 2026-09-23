"""Temporal score T(s, q, t_q): how old a passage is, relative to the question.

This is the signal the Temporal Filter adds to evidence admission. RAG² does
not use it at all. The whole thesis question is whether adding it helps, so
the formula is kept deliberately small:

    T(s, q, t_q) = 2 ** (-age_days / half_life_days)

where ``age_days = t_q - publication_date(s)``, clamped at zero. T is 1.0 for
a passage published on the question date, and halves every ``half_life_days``
after that. It is always in (0, 1], which is what lets it be combined with
the relevance score without any extra scaling.

**Secondary rules are off by default.** Retraction handling, a "superseded"
discount, and a time-invariance flag are implemented here but are NOT part
of the thesis experiment: enabling one changes what the score measures, so
each is an explicit opt-in, and the result records which were used. Leave
them off unless you are running a secondary analysis and say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Optional

from ..common.evidence import Evidence


class TemporalState(str, Enum):
    """Which branch produced the score. Recorded on every decision."""

    #: The only branch the thesis experiment should ever produce: plain
    #: age decay from a real publication date.
    DATED = "dated"

    #: No usable publication date. Scored with ``undated_score``.
    UNDATED = "undated"

    # --- secondary branches, reached only when explicitly enabled ---
    HARD_INVALID = "hard_invalid"
    TIME_INVARIANT = "time_invariant"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class TemporalResult:
    score: float
    state: TemporalState


class TemporalPolicy:
    """Compute the temporal score for one passage."""

    def __init__(
        self,
        *,
        half_life_days: Optional[float],
        undated_score: Optional[float] = None,
        # --- secondary, off by default ---------------------------------
        exclude_retracted: bool = False,
        is_time_invariant: Optional[Callable[[str], bool]] = None,
        superseded_factor: Optional[float] = None,
    ) -> None:
        """
        Args:
            half_life_days: H, the number of days over which the score
                halves. Data-dependent and unresolved; fit it on the
                validation split. No default is invented.
            undated_score: T for a passage with no usable publication date.
                REQUIRED: there is no defensible default, and scoring undated
                evidence 0.0 is a de-facto exclusion. Real evaluation items
                already exclude undated evidence before this ever runs
                (specification §15.3), so in the main experiment this branch
                should never fire; it exists for secondary analyses only.
            exclude_retracted: SECONDARY. Score retracted/withdrawn evidence
                0.0 instead of by age.
            is_time_invariant: SECONDARY. A rule for "this question's answer
                doesn't change over time"; when it returns True the passage
                scores 1.0 regardless of age.
            superseded_factor: SECONDARY. Multiplier applied when the passage
                is marked as superseded by a newer one.

        Raises:
            ValueError: if ``undated_score`` is missing, so a run fails
                immediately rather than part-way through.
        """

        if undated_score is None:
            raise ValueError(
                "undated_score is required and has no default. It sets T for "
                "evidence with no usable publication date; scoring such "
                "evidence 0.0 amounts to excluding it, which is a research "
                "decision that must be made and recorded explicitly."
            )

        if not 0.0 <= float(undated_score) <= 1.0:
            raise ValueError("undated_score must be in [0, 1].")

        self.half_life_days = half_life_days
        self.undated_score = float(undated_score)

        self.exclude_retracted = exclude_retracted
        self.is_time_invariant = is_time_invariant
        self.superseded_factor = superseded_factor

    @property
    def secondary_rules_enabled(self) -> tuple[str, ...]:
        """Secondary rules in force. Empty in the main experiment.

        Recorded in the run metadata so a result is never read as though it
        came from the plain temporal score when it did not.
        """
        enabled = []
        if self.exclude_retracted:
            enabled.append("exclude_retracted")
        if self.is_time_invariant is not None:
            enabled.append("time_invariance")
        if self.superseded_factor is not None:
            enabled.append("supersession_discount")
        return tuple(enabled)

    def score(
        self,
        evidence: Evidence,
        *,
        question: str,
        question_date: date,
    ) -> TemporalResult:
        """Return T for one passage."""

        if self.exclude_retracted and evidence.is_hard_invalid():
            return TemporalResult(0.0, TemporalState.HARD_INVALID)

        if self.is_time_invariant is not None and self.is_time_invariant(question):
            return TemporalResult(1.0, TemporalState.TIME_INVARIANT)

        if evidence.publication_date is None:
            return TemporalResult(self.undated_score, TemporalState.UNDATED)

        if self.half_life_days is None:
            raise ValueError(
                "half_life_days (H) is unresolved. Set it in the experiment "
                "configuration from the validation split; it is not given a "
                "default."
            )

        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive.")

        age_days = max(0, (question_date - evidence.publication_date).days)
        decay = 2.0 ** (-age_days / self.half_life_days)

        if (
            self.superseded_factor is not None
            and evidence.supersession_pointer is not None
        ):
            return TemporalResult(
                self.superseded_factor * decay,
                TemporalState.SUPERSEDED,
            )

        return TemporalResult(decay, TemporalState.DATED)
