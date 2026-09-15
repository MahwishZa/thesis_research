"""Contested-evidence detection for SCAF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..interfaces.evidence import Candidate


@dataclass(frozen=True)
class ClaimConflict:
    """Conflict between two evidence passages."""

    evidence_a: str
    evidence_b: str
    claim_class: str


class ContestedDetector:
    """Detect semantic conflicts among evidence candidates."""

    def __init__(
        self,
        conflict_fn: Callable[
            [Candidate, Candidate],
            bool,
        ],
    ) -> None:
        self._conflict_fn = conflict_fn

    def find_conflicts(
        self,
        candidates: Sequence[Candidate],
    ) -> tuple[ClaimConflict, ...]:

        conflicts: list[ClaimConflict] = []

        for index, left in enumerate(candidates):

            left_classes = set(
                left.evidence.claim_classes
            )

            for right in candidates[index + 1:]:

                shared_classes = (
                    left_classes
                    & set(right.evidence.claim_classes)
                )

                if not shared_classes:
                    continue

                if not self._conflict_fn(
                    left,
                    right,
                ):
                    continue

                claim_class = sorted(
                    shared_classes
                )[0]

                conflicts.append(
                    ClaimConflict(
                        evidence_a=(
                            left.evidence.evidence_id
                        ),
                        evidence_b=(
                            right.evidence.evidence_id
                        ),
                        claim_class=claim_class,
                    )
                )

        return tuple(conflicts)