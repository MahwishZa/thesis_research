"""SCAF admission policy and proposed system.

Order of operations, which follows the experimental specification and is not
interchangeable:

    1. detect contested claim classes over the FULL candidate set
    2. score each candidate, passing its contested status into gamma
       (specification section 29: contested is evaluated BEFORE supersession)
    3. apply the admission threshold
    4. preserve representative evidence for both sides of any contested claim
    5. apply the matched context budget by score rank
    6. derive the system output state

Step 1 must see every candidate, not only admitted ones: a contested claim
whose opposing side was already discarded cannot be represented, and
specification section 29 requires both positions to survive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import Callable, Optional, Sequence

from ..interfaces.evidence import Candidate, ExperimentResult
from ..interfaces.generator import Generator
from ..interfaces.system import System
from .contested import ClaimConflict, ContestedDetector
from .currency import CurrencyPolicy
from .scorer import SCAFScore, SCAFScorer
from .verifier import ClaimVerifier


class SCAFState(str, Enum):
    """SCAF output states (specification section 31).

    These are SYSTEM output states describing the answer as a whole. They are
    not per-passage labels: a rejected candidate carries ``state=None``, not
    ABSTAIN, because ABSTAIN means "the available evidence is insufficient for
    a supported answer" and is a property of the item, not of one passage.
    """

    GROUNDED = "GROUNDED"
    FLAGGED = "FLAGGED"
    CONTESTED = "CONTESTED"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class SCAFConfig:
    """Configuration explicitly supplied by the experiment.

    Every value the specification leaves ``[TO BE SPECIFIED]`` defaults to
    ``None`` and raises rather than taking an invented value.
    """

    admit_threshold: Optional[float] = None
    question_date: Optional[date] = None
    max_admitted_passages: Optional[int] = None

    # Specification section 29: when a contested claim is detected, both
    # positions must survive admission. Leaving this False would let the
    # threshold silence one side of a dispute the system exists to surface.
    preserve_contested_positions: bool = True

    # Sections 16 and 29 can conflict: section 16 makes the context budget
    # INVARIANT across arms, while section 29 requires both positions of a
    # contested claim to be preserved. The research documents do not say
    # which wins, so the resolution is explicit here rather than silent.
    #
    #   "cap"    - the budget is never exceeded. Contested passages are
    #              prioritised within it, and any position that still had to
    #              be dropped is recorded. Arm comparability is preserved.
    #   "exempt" - both positions are always kept, so the budget may be
    #              exceeded. This BREAKS the section 16 invariant and must be
    #              declared in the write-up; the overrun is recorded.
    #
    # Default is "cap" because silently breaking an invariant that Stage 5
    # comparability rests on is the worse failure.
    contested_budget_policy: str = "cap"

    def validate(self) -> None:

        if self.admit_threshold is None:
            raise ValueError(
                "SCAF admit_threshold is unresolved. It is "
                "[TO BE SPECIFIED] in the experimental specification and "
                "must be set on validation data before any test run."
            )

        if self.question_date is None:
            raise ValueError(
                "SCAF question_date (t_q) is required: gamma is undefined "
                "without an as-of date."
            )

        if self.max_admitted_passages is not None:
            if self.max_admitted_passages <= 0:
                raise ValueError(
                    "max_admitted_passages must be positive."
                )

        if self.contested_budget_policy not in ("cap", "exempt"):
            raise ValueError(
                "contested_budget_policy must be 'cap' or 'exempt'; got "
                f"{self.contested_budget_policy!r}."
            )


@dataclass(frozen=True)
class SCAFDecision:
    """Decision for one candidate."""

    candidate: Candidate
    score: SCAFScore

    admitted: bool

    # None for rejected candidates - see SCAFState.
    state: Optional[SCAFState]

    currency_state: str

    # True when this passage was retained to represent one side of a
    # contested claim despite scoring below the admission threshold.
    preserved_for_contest: bool = False

    # True when this passage belonged to a contested pair but was dropped to
    # stay inside the context budget. Recorded so a contested result is never
    # read as complete when one position was removed.
    dropped_for_budget: bool = False

    conflicts: tuple[ClaimConflict, ...] = ()


StateFunction = Callable[[SCAFScore, Candidate], SCAFState]


class SCAFAdmissionPolicy:
    """Compute SCAF evidence-admission decisions."""

    def __init__(
        self,
        *,
        scorer: SCAFScorer,
        currency: CurrencyPolicy,
        contested: ContestedDetector,
        support_fn: Callable[[str, Candidate], float],
        authority_fn: Callable[[Candidate], float],
        state_fn: StateFunction,
        config: SCAFConfig,
    ) -> None:

        config.validate()

        self.scorer = scorer
        self.currency = currency
        self.contested = contested

        self.support_fn = support_fn
        self.authority_fn = authority_fn
        self.state_fn = state_fn

        self.config = config

    def decide(
        self,
        question: str,
        candidates: Sequence[Candidate],
    ) -> tuple[SCAFDecision, ...]:

        if not candidates:
            return ()

        question_date = self.config.question_date

        if question_date is None:
            # Defensive: validate() already enforces this. Not an assert,
            # because assertions are stripped under `python -O`.
            raise ValueError("SCAF question_date is required.")

        # --- 1. contested detection, over the FULL candidate set ----------
        conflicts = self.contested.find_conflicts(candidates)
        contested_classes = self.contested.contested_claim_classes(conflicts)

        conflict_ids: set[str] = set()
        for conflict in conflicts:
            conflict_ids.add(conflict.evidence_a)
            conflict_ids.add(conflict.evidence_b)

        # --- 2. score, with contested status feeding gamma ----------------
        decisions: list[SCAFDecision] = []

        for candidate in candidates:

            is_contested = bool(
                contested_classes
                & set(candidate.evidence.claim_classes)
            )

            currency_result = self.currency.score(
                candidate.evidence,
                question=question,
                question_date=question_date,
                contested=is_contested,
            )

            support = self.support_fn(question, candidate)
            authority = self.authority_fn(candidate)

            score = self.scorer.score(
                candidate,
                support=support,
                currency=currency_result.score,
                authority=authority,
                candidate_count=len(candidates),
            )

            # --- 3. threshold -------------------------------------------
            admitted = (
                score.total >= self.config.admit_threshold
                and not candidate.evidence.is_hard_invalid()
            )

            state: Optional[SCAFState]
            if not admitted:
                state = None
            elif candidate.evidence.evidence_id in conflict_ids:
                state = SCAFState.CONTESTED
            else:
                state = self.state_fn(score, candidate)

            decisions.append(
                SCAFDecision(
                    candidate=candidate,
                    score=score,
                    admitted=admitted,
                    state=state,
                    currency_state=currency_result.state.value,
                    conflicts=conflicts,
                )
            )

        # --- 4. preserve both sides of any contested claim ----------------
        # Retraction and withdrawal remain the only hard exclusions, so a
        # retracted passage is never resurrected to represent a position.
        if conflicts and self.config.preserve_contested_positions:
            decisions = [
                (
                    replace(
                        decision,
                        admitted=True,
                        state=SCAFState.CONTESTED,
                        preserved_for_contest=True,
                    )
                    if (
                        not decision.admitted
                        and decision.candidate.evidence.evidence_id
                        in conflict_ids
                        and not decision.candidate.evidence.is_hard_invalid()
                    )
                    else decision
                )
                for decision in decisions
            ]

        # --- 5. matched context budget ------------------------------------
        # Specification section 16 makes the context budget invariant across
        # arms; section 29 requires both positions of a contested claim to
        # survive. Under "cap" the invariant wins and contested passages are
        # merely prioritised inside the budget; under "exempt" section 29
        # wins and the overrun is recorded. Neither is chosen silently.
        #
        # Ordering is A(s) descending with evidence_id as a deterministic
        # tie-break, so two runs over identical inputs select identically.
        limit = self.config.max_admitted_passages

        if limit is not None:

            admitted_now = [d for d in decisions if d.admitted]

            by_score = lambda d: (
                -d.score.total,
                d.candidate.evidence.evidence_id,
            )

            contested_admitted = sorted(
                (
                    d for d in admitted_now
                    if d.candidate.evidence.evidence_id in conflict_ids
                ),
                key=by_score,
            )
            other_admitted = sorted(
                (
                    d for d in admitted_now
                    if d.candidate.evidence.evidence_id not in conflict_ids
                ),
                key=by_score,
            )

            if self.config.contested_budget_policy == "exempt":
                # Every contested position survives; the budget may be
                # exceeded, and that overrun is visible in the result.
                keep = {
                    d.candidate.evidence.evidence_id
                    for d in contested_admitted
                }
                for decision in other_admitted:
                    if len(keep) >= limit:
                        break
                    keep.add(decision.candidate.evidence.evidence_id)
            else:
                # "cap": contested passages are taken first so that
                # representation survives as far as the budget allows, but
                # the budget is never exceeded.
                keep = set()
                for decision in contested_admitted + other_admitted:
                    if len(keep) >= limit:
                        break
                    keep.add(decision.candidate.evidence.evidence_id)

            decisions = [
                (
                    decision
                    if (
                        not decision.admitted
                        or decision.candidate.evidence.evidence_id in keep
                    )
                    else replace(
                        decision,
                        admitted=False,
                        state=None,
                        preserved_for_contest=False,
                        dropped_for_budget=(
                            decision.candidate.evidence.evidence_id
                            in conflict_ids
                        ),
                    )
                )
                for decision in decisions
            ]

        return tuple(decisions)


class SCAFSystem(System):
    """Complete SCAF experimental system."""

    name = "P_SCAF"

    def __init__(
        self,
        *,
        answer_generator: Generator,
        admission_policy: SCAFAdmissionPolicy,
        verifier: Optional[ClaimVerifier] = None,
        context_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            context_prompt: answer-prompt template with ``{question}`` and
                ``{context}`` fields. Specification section 16 requires
                prompt parity across arms, so the experiment runner should
                pass the SAME template here and to the baseline.
        """

        self.answer_generator = answer_generator
        self.admission_policy = admission_policy
        self.verifier = verifier
        self.context_prompt = context_prompt

    def build_prompt(
        self,
        question: str,
        decisions: Sequence[SCAFDecision],
    ) -> str:

        admitted = [
            decision for decision in decisions if decision.admitted
        ]

        context = "\n\n".join(
            (
                f"[{decision.candidate.evidence.evidence_id}] "
                f"{decision.candidate.evidence.text}"
            )
            for decision in admitted
        )

        if self.context_prompt is not None:
            return self.context_prompt.format(
                question=question,
                context=context,
            )

        return (
            "Answer the question using the provided evidence.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context}"
        )

    def run(
        self,
        *,
        sample_id: str,
        experiment_id: str,
        question: str,
        candidates: Sequence[Candidate],
        rationale: Optional[str] = None,
    ) -> ExperimentResult:

        decisions = self.admission_policy.decide(question, candidates)

        admitted = tuple(d for d in decisions if d.admitted)
        rejected = tuple(d for d in decisions if not d.admitted)

        limit = self.admission_policy.config.max_admitted_passages

        score_metadata = {
            decision.candidate.evidence.evidence_id: {
                "total": decision.score.total,
                "support": decision.score.support,
                "currency": decision.score.currency,
                "rerank": decision.score.rerank,
                "authority": decision.score.authority,
                "currency_state": decision.currency_state,
                "state": (
                    decision.state.value
                    if decision.state is not None
                    else None
                ),
                "admitted": decision.admitted,
                "preserved_for_contest": decision.preserved_for_contest,
                "dropped_for_budget": decision.dropped_for_budget,
            }
            for decision in decisions
        }

        conflicts = decisions[0].conflicts if decisions else ()

        base_metadata = {
            "system": "SCAF",
            "scores": score_metadata,
            "conflicts": [
                {
                    "evidence_a": c.evidence_a,
                    "evidence_b": c.evidence_b,
                    "claim_class": c.claim_class,
                    "separation_days": c.separation_days,
                }
                for c in conflicts
            ],
            # Records which specification conditions were actually applied,
            # so a contested result is never read as though an unresolved
            # condition had been checked.
            "contested_conditions_enforced": list(
                self.admission_policy.contested.enforced_conditions
            ),
            # Specification section 16 requires the context budget to be
            # invariant across arms. Recorded explicitly so a Stage-5
            # comparison can verify it held rather than assume it.
            "context_budget": {
                "limit": limit,
                "admitted": len(admitted),
                "policy": (
                    self.admission_policy.config.contested_budget_policy
                ),
                "exceeded": (
                    limit is not None and len(admitted) > limit
                ),
                "contested_positions_dropped": [
                    d.candidate.evidence.evidence_id
                    for d in decisions
                    if d.dropped_for_budget
                ],
                "preserved_for_contest": [
                    d.candidate.evidence.evidence_id
                    for d in decisions
                    if d.preserved_for_contest
                ],
            },
        }

        if not admitted:
            return ExperimentResult(
                sample_id=sample_id,
                experiment_id=experiment_id,
                variant=self.name,
                prediction=None,
                output_state=SCAFState.ABSTAIN.value,
                admitted_evidence_ids=(),
                rejected_evidence_ids=tuple(
                    d.candidate.evidence.evidence_id for d in rejected
                ),
                candidate_count=len(candidates),
                rationale=rationale,
                metadata={
                    **base_metadata,
                    "reason": "no_admitted_evidence",
                },
            )

        evidence = tuple(d.candidate.evidence for d in admitted)
        prompt = self.build_prompt(question, decisions)

        answer = self.answer_generator.generate(
            question,
            evidence,
            prompt=prompt,
        )

        states = {d.state for d in admitted}

        if SCAFState.CONTESTED in states:
            output_state = SCAFState.CONTESTED
        elif SCAFState.FLAGGED in states:
            output_state = SCAFState.FLAGGED
        else:
            output_state = SCAFState.GROUNDED

        verification_metadata = None

        if self.verifier is not None:
            verification = self.verifier.verify(answer.text, evidence)
            verification_metadata = {
                "supported": verification.supported,
                "confidence": verification.confidence,
                "claims_checked": verification.claims_checked,
                "metadata": dict(verification.metadata),
            }

        return ExperimentResult(
            sample_id=sample_id,
            experiment_id=experiment_id,
            variant=self.name,
            prediction=answer.text,
            output_state=output_state.value,
            admitted_evidence_ids=tuple(
                d.candidate.evidence.evidence_id for d in admitted
            ),
            rejected_evidence_ids=tuple(
                d.candidate.evidence.evidence_id for d in rejected
            ),
            candidate_count=len(candidates),
            rationale=rationale,
            generated_text=answer.text,
            confidence=answer.confidence,
            metadata={
                **base_metadata,
                "verification": verification_metadata,
            },
        )
