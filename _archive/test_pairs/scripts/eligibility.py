"""Stage-2 eligibility rules.

These rules decide whether a candidate pair can support a *temporal* claim.
They are not SCAF's runtime behaviour and must not be confused with it:

* Stage 2 asks "can this item measure recency bias at all?" and answers by
  excluding the item, with a recorded reason, when it cannot.
* SCAF later asks "what score does undated evidence get at inference time?"
  That is a separate, still-unresolved question (ledger: ``unknown_date_score``
  remains a supervisor decision) and nothing here settles it.

The central rule needs no invented tolerance parameter. A publication date is
an interval, so the older passage is *unambiguously* older exactly when its
interval ends before the newer passage's interval begins. Two passages both
dated ``"2023"`` are not orderable and are excluded; ``"2023-01"`` versus
``"2024-06"`` is orderable even though neither day is known.

``min_separation_days`` is deliberately unset by default. The pilot exists to
produce the separation distribution from which a defensible threshold is
chosen; enforcing a guessed value beforehand would discard the evidence needed
to choose it. When it is None the rule is not applied, and
:func:`enforced_rules` records that, following the same convention as the
contested detector in ``systems/proposed/contested.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Optional

from .schema import EvidenceRef, TestPair, _parse_partial_date


@dataclass(frozen=True)
class EligibilityConfig:
    """Stage-2 inclusion rules. Every value is explicit and recorded."""

    #: Minimum days between the two publication dates. None = not enforced;
    #: set it from the pilot's separation histogram before the final set.
    min_separation_days: Optional[int] = None

    #: Require both sides to carry a persistent identifier (PMID/PMCID/DOI).
    #: Off by default because the current corpus chunks do not carry one;
    #: the reason is still recorded, so the cost of turning it on is visible.
    require_persistent_id: bool = False

    #: Require both sides to share a source tier (specification 10.1
    #: matching constraint). Recorded either way.
    require_same_source_tier: bool = False

    #: Maximum ratio between the longer and shorter passage. None = not
    #: enforced. Specification 10.1 asks for a "tolerance band" but does not
    #: give one; the pilot measures the observed distribution instead.
    max_length_ratio: Optional[float] = None

    #: Exclude pairs where either side is retracted or withdrawn. On by
    #: default: a retracted passage cannot represent a defensible position in
    #: an evaluation item. This is a Stage-2 sampling rule and is unrelated
    #: to SCAF's runtime hard-exclusion rule.
    exclude_retracted: bool = True

    def enforced_rules(self) -> tuple[str, ...]:
        """Which optional rules this configuration actually applies."""
        applied = ["usable_dates", "unambiguous_temporal_order", "question_date"]
        if self.min_separation_days is not None:
            applied.append("min_separation_days")
        if self.require_persistent_id:
            applied.append("persistent_identifier")
        if self.require_same_source_tier:
            applied.append("same_source_tier")
        if self.max_length_ratio is not None:
            applied.append("length_ratio")
        if self.exclude_retracted:
            applied.append("exclude_retracted")
        return tuple(applied)


def _interval(ref: EvidenceRef) -> tuple[Optional[date], Optional[date]]:
    return ref.date_interval


def separation(older: EvidenceRef, newer: EvidenceRef) -> tuple[Optional[int], Optional[int]]:
    """Return ``(point_separation, guaranteed_separation)`` in days.

    The point separation uses the earliest day consistent with each date.
    The guaranteed separation is what the intervals alone establish, and is
    negative when the ordering is not established at all.
    """

    older_start, older_end = _interval(older)
    newer_start, newer_end = _interval(newer)

    if older_start is None or newer_start is None:
        return (None, None)

    point = (newer_start - older_start).days
    guaranteed = (newer_start - older_end).days

    return (point, guaranteed)


def evaluate(pair: TestPair, config: EligibilityConfig) -> TestPair:
    """Return ``pair`` with temporal eligibility and exclusion reasons set.

    Pure and deterministic: the same pair and config always yield the same
    result, and every rejection carries a reason from the closed vocabulary.
    """

    # Reasons already recorded by the builder (for example an unverified
    # contradiction) are kept: eligibility narrows a pair, never rehabilitates
    # one.
    reasons: list[str] = list(pair.exclusion_reasons)

    if pair.question_date is None or _parse_partial_date(pair.question_date)[0] is None:
        reasons.append("missing_question_date")

    if not pair.older.has_usable_date or not pair.newer.has_usable_date:
        reasons.append("missing_publication_date")

    point, guaranteed = separation(pair.older, pair.newer)

    if point is not None:
        if guaranteed is not None and guaranteed <= 0:
            # Intervals overlap or run the wrong way: which passage is older
            # is not established by the recorded dates.
            reasons.append("ambiguous_temporal_order")
        elif (
            config.min_separation_days is not None
            and point < config.min_separation_days
        ):
            reasons.append("insufficient_temporal_separation")

    if config.exclude_retracted and (
        pair.older.retracted
        or pair.older.withdrawn
        or pair.newer.retracted
        or pair.newer.withdrawn
    ):
        reasons.append("retracted_or_withdrawn")

    if config.require_persistent_id and not (
        pair.older.persistent_id and pair.newer.persistent_id
    ):
        reasons.append("missing_persistent_identifier")

    if config.require_same_source_tier and (
        pair.older.source_tier != pair.newer.source_tier
    ):
        reasons.append("source_tier_mismatch")

    if config.max_length_ratio is not None:
        lengths = (pair.older.length_tokens, pair.newer.length_tokens)
        if all(value for value in lengths):
            ratio = max(lengths) / min(lengths)
            if ratio > config.max_length_ratio:
                reasons.append("length_mismatch")

    updated = replace(
        pair,
        separation_days=point,
        min_separation_days=guaranteed,
        temporal_eligible=False,
        exclusion_reasons=(),
    )

    if reasons:
        return updated.excluded(*reasons)

    return replace(updated, temporal_eligible=True)
