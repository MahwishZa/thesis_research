"""SECONDARY - not part of the primary experiment.

Contested-evidence detection needs claim classes (which exist in the corpus
only as an unvalidated keyword heuristic, ledger A8/D-22), a contest window
and a source-tier gate (neither resolved), and it is not required by the
three current objectives - see docs/current_objectives.md's "Removed from
the primary pipeline". The admission policy takes
``contested=None`` by default; enabling it changes which passages are
admitted, so such a run is a secondary analysis and must say so.

Contested-evidence detection for the recency-aware admission policy.

Specification section 29 defines a claim as contested when ALL FOUR of the
following hold:

1. the passages belong to the same claim class;
2. the passages support opposing conclusions;
3. the passages fall within the defined contest window;
4. both satisfy the minimum source-quality requirement.

All four are enforced here. Conditions 3 and 4 were previously absent, which
made the detector strictly more permissive than the specification.

Two parameters are ``[TO BE SPECIFIED]`` in the research documents and are
therefore configurable with no invented default:

* ``contest_window_days`` - how close in time two opposing passages must be
  for the disagreement to count as contemporaneous rather than as one
  superseding the other. This choice materially changes whether the state
  fires at all, so it must be frozen before evaluation.
* ``minimum_source_tiers`` - which source tiers are credible enough to
  establish a dispute. Note that this is a *gate*, not a ranking: source
  authority is a tested experimental variable (specification section 27), so
  no ordering over tiers is defined or implied here.

Specification section 30: the caller's ``conflict_fn`` is responsible for
distinguishing substantive contradiction from differences of scope
(population, stage, intervention, outcome, setting). That distinction is
identified in the research documents as a technical limitation, so contested
results describe the mechanism unless validation establishes reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Optional, Sequence

from ..interfaces.evidence import Candidate


@dataclass(frozen=True)
class ClaimConflict:
    """A detected conflict between two evidence passages."""

    evidence_a: str
    evidence_b: str
    claim_class: str

    # Recorded so a conflict can be audited after the run.
    separation_days: Optional[int] = None


class ContestedDetector:
    """Detect unresolved evidential disagreement among candidates."""

    def __init__(
        self,
        conflict_fn: Callable[[Candidate, Candidate], bool],
        *,
        contest_window_days: Optional[int] = None,
        minimum_source_tiers: Optional[Collection[str]] = None,
    ) -> None:
        """
        Args:
            conflict_fn: returns True when the two passages assert opposing
                conclusions (specification condition 2). Scope differences
                must not be reported as conflicts.
            contest_window_days: specification condition 3. ``None`` means
                the window is unresolved and NOT enforced; the detector then
                records that the condition was not applied rather than
                silently assuming an unbounded window.
            minimum_source_tiers: specification condition 4. ``None`` means
                unresolved and NOT enforced. Membership only - no ordering
                over tiers is defined, because authority is a tested
                variable.
        """
        self._conflict_fn = conflict_fn
        self.contest_window_days = contest_window_days
        self.minimum_source_tiers = (
            None
            if minimum_source_tiers is None
            else frozenset(minimum_source_tiers)
        )

    @property
    def enforced_conditions(self) -> tuple[str, ...]:
        """Which specification conditions this instance actually applies.

        Recorded in the run metadata so a result is never read as though a
        condition had been checked when it was left unresolved.
        """
        applied = ["same_claim_class", "opposing_conclusions"]
        if self.contest_window_days is not None:
            applied.append("contest_window")
        if self.minimum_source_tiers is not None:
            applied.append("minimum_source_tier")
        return tuple(applied)

    def _within_window(
        self,
        left: Candidate,
        right: Candidate,
    ) -> tuple[bool, Optional[int]]:

        left_date = left.evidence.publication_date
        right_date = right.evidence.publication_date

        if left_date is None or right_date is None:
            # Cannot establish contemporaneity without both dates.
            return (self.contest_window_days is None, None)

        separation = abs((left_date - right_date).days)

        if self.contest_window_days is None:
            return (True, separation)

        return (separation <= self.contest_window_days, separation)

    def _tier_ok(self, candidate: Candidate) -> bool:
        if self.minimum_source_tiers is None:
            return True
        return candidate.evidence.source_tier in self.minimum_source_tiers

    def find_conflicts(
        self,
        candidates: Sequence[Candidate],
    ) -> tuple[ClaimConflict, ...]:
        """Return every conflicting pair satisfying the enforced conditions.

        The caller must pass the FULL candidate set, not only admitted
        candidates: specification section 29 requires representative evidence
        for both positions to be preserved, which is impossible if one side
        was already discarded.
        """

        conflicts: list[ClaimConflict] = []

        for index, left in enumerate(candidates):

            if not self._tier_ok(left):
                continue

            left_classes = set(left.evidence.claim_classes)

            if not left_classes:
                continue

            for right in candidates[index + 1:]:

                if not self._tier_ok(right):
                    continue

                shared_classes = left_classes & set(
                    right.evidence.claim_classes
                )

                if not shared_classes:
                    continue

                in_window, separation = self._within_window(left, right)

                if not in_window:
                    continue

                if not self._conflict_fn(left, right):
                    continue

                conflicts.append(
                    ClaimConflict(
                        evidence_a=left.evidence.evidence_id,
                        evidence_b=right.evidence.evidence_id,
                        claim_class=sorted(shared_classes)[0],
                        separation_days=separation,
                    )
                )

        return tuple(conflicts)

    def contested_claim_classes(
        self,
        conflicts: Sequence[ClaimConflict],
    ) -> frozenset[str]:
        """Claim classes under dispute, for the recency policy."""
        return frozenset(conflict.claim_class for conflict in conflicts)
