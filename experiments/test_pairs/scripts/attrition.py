"""Attrition accounting for Stage-2 candidate pairs.

The point of the pilot is not the pairs it yields but the bottleneck it
exposes. A candidate can fail several rules at once, so two counts are kept
and they answer different questions:

* ``pairs_with_reason`` - how many candidates carry this reason at all.
  Sums to more than the number excluded when reasons co-occur.
* ``sole_reason`` - how many candidates this reason alone removed. This is
  the number that would be recovered by fixing that one constraint, and is
  the only one of the two that supports "if we fix X we gain Y pairs".

Reporting only the first would systematically overstate what any single fix
is worth.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .schema import EXCLUSION_REASONS, TestPair


@dataclass(frozen=True)
class AttritionReport:
    candidates: int
    eligible: int
    excluded: int
    pairs_with_reason: dict[str, int]
    sole_reason: dict[str, int]

    @property
    def yield_rate(self) -> float:
        if self.candidates == 0:
            return 0.0
        return self.eligible / self.candidates

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "exclusion_reason": reason,
                "pairs_with_reason": self.pairs_with_reason.get(reason, 0),
                "sole_reason": self.sole_reason.get(reason, 0),
                "share_of_candidates": (
                    round(self.pairs_with_reason.get(reason, 0) / self.candidates, 4)
                    if self.candidates
                    else 0.0
                ),
            }
            for reason in EXCLUSION_REASONS
            if self.pairs_with_reason.get(reason, 0)
        ]

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "exclusion_reason",
                    "pairs_with_reason",
                    "sole_reason",
                    "share_of_candidates",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows())


def compute(pairs: Sequence[TestPair]) -> AttritionReport:
    with_reason: Counter[str] = Counter()
    sole: Counter[str] = Counter()

    eligible = 0

    for pair in pairs:
        if pair.is_eligible:
            eligible += 1
            continue

        reasons = tuple(pair.exclusion_reasons)
        with_reason.update(reasons)

        if len(reasons) == 1:
            sole.update(reasons)

    return AttritionReport(
        candidates=len(pairs),
        eligible=eligible,
        excluded=len(pairs) - eligible,
        pairs_with_reason=dict(with_reason),
        sole_reason=dict(sole),
    )


def separation_summary(pairs: Sequence[TestPair]) -> dict[str, object]:
    """Distribution of temporal separation among eligible pairs.

    This is the evidence from which ``min_separation_days`` is chosen. It is
    reported, never used to set the value automatically: tuning an inclusion
    rule from the data it will be applied to is the circularity the
    specification's freeze conditions exist to prevent.
    """

    values = sorted(
        pair.separation_days
        for pair in pairs
        if pair.is_eligible and pair.separation_days is not None
    )

    if not values:
        return {"n": 0}

    def percentile(fraction: float) -> int:
        index = min(len(values) - 1, int(round(fraction * (len(values) - 1))))
        return values[index]

    return {
        "n": len(values),
        "min_days": values[0],
        "p25_days": percentile(0.25),
        "median_days": percentile(0.50),
        "p75_days": percentile(0.75),
        "max_days": values[-1],
    }
