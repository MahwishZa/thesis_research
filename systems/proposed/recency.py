"""Recency score R(s, q, t_q): how recent a passage is, relative to the question.

This is the single signal the thesis adds to evidence admission. The whole
hypothesis rests on it, so it is kept deliberately small:

    R(s, q, t_q) = 2 ** (-age_days / half_life_days)

where ``age_days = t_q - publication_date(s)``, clamped at zero. R is 1.0 for
a passage published on the question date and halves every ``half_life_days``.
It is always in (0, 1], which is what lets it be combined with the
rank-normalised relevance score without further scaling.

**Why "recency" and not "currency".** The earlier design called this the
currency score, after a three-state currency model (current / superseded /
contested). Two of those states are no longer part of the primary experiment,
and "currency" reads as money on first encounter. The phenomenon under study
is recency asymmetry, so the signal is the recency score. One word, one
meaning, used everywhere.

**Secondary rules are off by default.** Retraction handling, the supersession
discount and time-invariance (psi) belong to the wider design and are
implemented here because they were already written and may support a later
qualitative analysis. They are NOT part of the primary experiment: enabling
one changes what the score measures, so each is an explicit opt-in and the
result records which were applied. Leave them off unless you are running a
secondary analysis and say so in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Optional

from ..interfaces.evidence import Evidence


class RecencyState(str, Enum):
    """Which branch produced the score. Recorded on every decision."""

    #: The primary branch: plain age decay. The only one the primary
    #: experiment should ever produce.
    DATED = "dated"

    #: No usable publication date. Scored with ``undated_score``.
    UNDATED = "undated"

    # --- secondary branches, reached only when explicitly enabled ---
    HARD_INVALID = "hard_invalid"
    TIME_INVARIANT = "time_invariant"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class RecencyResult:
    score: float
    state: RecencyState


class RecencyPolicy:
    """Compute the recency score for one passage."""

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
                halves. Data-dependent and unresolved; tune it on the
                validation split. No default is invented.
            undated_score: R for a passage with no usable publication date.
                REQUIRED, because there is no defensible default and the
                value materially affects the temporal result: scoring undated
                evidence 0.0 is a de-facto exclusion. Stage-2 eligibility
                already removes undated pairs from the primary evaluation, so
                in the primary experiment this branch should never fire; it
                exists for secondary analyses over the corpus.
            exclude_retracted: SECONDARY. Score retracted/withdrawn evidence
                0.0 instead of by age.
            is_time_invariant: SECONDARY. psi(q); when it returns True the
                passage scores 1.0 regardless of age.
            superseded_factor: SECONDARY. Multiplier applied when the passage
                carries a supersession pointer.

        Raises:
            ValueError: if ``undated_score`` is missing, so a run fails
                deterministically at construction rather than part-way
                through.
        """

        if undated_score is None:
            raise ValueError(
                "undated_score is required and has no default. It sets R for "
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
        """Secondary rules in force. Empty in the primary experiment.

        Recorded in the run metadata so a result is never read as though it
        came from the plain recency score when it did not.
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
    ) -> RecencyResult:
        """Return R for one passage."""

        if self.exclude_retracted and evidence.is_hard_invalid():
            return RecencyResult(0.0, RecencyState.HARD_INVALID)

        if self.is_time_invariant is not None and self.is_time_invariant(question):
            return RecencyResult(1.0, RecencyState.TIME_INVARIANT)

        if evidence.publication_date is None:
            return RecencyResult(self.undated_score, RecencyState.UNDATED)

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
            return RecencyResult(
                self.superseded_factor * decay,
                RecencyState.SUPERSEDED,
            )

        return RecencyResult(decay, RecencyState.DATED)
