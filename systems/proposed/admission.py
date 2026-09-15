"""SCAF admission policy and proposed system."""

from __future__ import annotations

from dataclasses import dataclass
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
    """SCAF output states."""

    GROUNDED = "GROUNDED"
    FLAGGED = "FLAGGED"
    CONTESTED = "CONTESTED"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class SCAFConfig:
    """Configuration explicitly supplied by the experiment."""

    admit_threshold: Optional[float] = None
    question_date: Optional[date] = None
    max_admitted_passages: Optional[int] = None

    def validate(self) -> None:

        if self.admit_threshold is None:
            raise ValueError(
                "SCAF admit_threshold is unresolved."
            )

        if self.question_date is None:
            raise ValueError(
                "SCAF question_date is required."
            )

        if self.max_admitted_passages is not None:
            if self.max_admitted_passages <= 0:
                raise ValueError(
                    "max_admitted_passages must be positive."
                )


@dataclass(frozen=True)
class SCAFDecision:
    """Decision for one candidate."""

    candidate: Candidate
    score: SCAFScore

    admitted: bool
    state: SCAFState

    currency_state: str

    conflicts: tuple[ClaimConflict, ...] = ()


StateFunction = Callable[
    [SCAFScore, Candidate],
    SCAFState,
]


class SCAFAdmissionPolicy:
    """Compute SCAF evidence-admission decisions."""

    def __init__(
        self,
        *,
        scorer: SCAFScorer,
        currency: CurrencyPolicy,
        contested: ContestedDetector,
        support_fn: Callable[
            [str, Candidate],
            float,
        ],
        authority_fn: Callable[
            [Candidate],
            float,
        ],
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

        assert question_date is not None

        decisions: list[SCAFDecision] = []

        for candidate in candidates:

            currency_result = self.currency.score(
                candidate.evidence,
                question=question,
                question_date=question_date,
            )

            support = self.support_fn(
                question,
                candidate,
            )

            authority = self.authority_fn(
                candidate,
            )

            score = self.scorer.score(
                candidate,
                support=support,
                currency=currency_result.score,
                authority=authority,
                candidate_count=len(candidates),
            )

            admitted = (
                score.total
                >= self.config.admit_threshold
                and not candidate.evidence.is_hard_invalid()
            )

            if admitted:
                state = self.state_fn(
                    score,
                    candidate,
                )
            else:
                state = SCAFState.ABSTAIN

            decisions.append(
                SCAFDecision(
                    candidate=candidate,
                    score=score,
                    admitted=admitted,
                    state=state,
                    currency_state=(
                        currency_result.state.value
                    ),
                )
            )

        admitted_candidates = tuple(
            decision.candidate
            for decision in decisions
            if decision.admitted
        )

        conflicts = self.contested.find_conflicts(
            admitted_candidates
        )

        if conflicts:

            conflict_ids = {
                conflict.evidence_a
                for conflict in conflicts
            }

            conflict_ids.update(
                conflict.evidence_b
                for conflict in conflicts
            )

            decisions = [
                SCAFDecision(
                    candidate=decision.candidate,
                    score=decision.score,
                    admitted=decision.admitted,
                    state=(
                        SCAFState.CONTESTED
                        if (
                            decision.candidate.evidence.evidence_id
                            in conflict_ids
                        )
                        else decision.state
                    ),
                    currency_state=decision.currency_state,
                    conflicts=conflicts,
                )
                for decision in decisions
            ]

        if self.config.max_admitted_passages is not None:

            limited: list[SCAFDecision] = []
            admitted_count = 0

            for decision in decisions:

                if not decision.admitted:
                    limited.append(decision)
                    continue

                if (
                    admitted_count
                    >= self.config.max_admitted_passages
                ):
                    limited.append(
                        SCAFDecision(
                            candidate=decision.candidate,
                            score=decision.score,
                            admitted=False,
                            state=SCAFState.ABSTAIN,
                            currency_state=(
                                decision.currency_state
                            ),
                            conflicts=decision.conflicts,
                        )
                    )
                else:
                    limited.append(decision)
                    admitted_count += 1

            decisions = limited

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
    ) -> None:

        self.answer_generator = answer_generator
        self.admission_policy = admission_policy
        self.verifier = verifier

    @staticmethod
    def build_prompt(
        question: str,
        decisions: Sequence[SCAFDecision],
    ) -> str:

        admitted = [
            decision
            for decision in decisions
            if decision.admitted
        ]

        context = "\n\n".join(
            (
                f"[{decision.candidate.evidence.evidence_id}] "
                f"{decision.candidate.evidence.text}"
            )
            for decision in admitted
        )

        return (
            "Answer the question using the selected evidence.\n\n"
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
    ) -> ExperimentResult:

        decisions = self.admission_policy.decide(
            question,
            candidates,
        )

        admitted = tuple(
            decision
            for decision in decisions
            if decision.admitted
        )

        rejected = tuple(
            decision
            for decision in decisions
            if not decision.admitted
        )

        if not admitted:

            return ExperimentResult(
                sample_id=sample_id,
                experiment_id=experiment_id,
                variant=self.name,
                prediction=None,
                output_state=SCAFState.ABSTAIN.value,
                admitted_evidence_ids=(),
                rejected_evidence_ids=tuple(
                    decision.candidate.evidence.evidence_id
                    for decision in rejected
                ),
                candidate_count=len(candidates),
                metadata={
                    "system": "SCAF",
                    "reason": "no_admitted_evidence",
                },
            )

        evidence = tuple(
            decision.candidate.evidence
            for decision in admitted
        )

        prompt = self.build_prompt(
            question,
            decisions,
        )

        answer = self.answer_generator.generate(
            question,
            evidence,
            prompt=prompt,
        )

        states = {
            decision.state
            for decision in admitted
        }

        if SCAFState.CONTESTED in states:
            output_state = SCAFState.CONTESTED

        elif SCAFState.FLAGGED in states:
            output_state = SCAFState.FLAGGED

        else:
            output_state = SCAFState.GROUNDED

        verification_metadata = None

        if self.verifier is not None:

            verification = self.verifier.verify(
                answer.text,
                evidence,
            )

            verification_metadata = {
                "supported": verification.supported,
                "confidence": verification.confidence,
                "claims_checked": (
                    verification.claims_checked
                ),
                "metadata": dict(
                    verification.metadata
                ),
            }

        score_metadata = {
            decision.candidate.evidence.evidence_id: {
                "total": decision.score.total,
                "support": decision.score.support,
                "currency": decision.score.currency,
                "rerank": decision.score.rerank,
                "authority": decision.score.authority,
                "currency_state": (
                    decision.currency_state
                ),
                "state": decision.state.value,
            }
            for decision in decisions
        }

        return ExperimentResult(
            sample_id=sample_id,
            experiment_id=experiment_id,
            variant=self.name,
            prediction=answer.text,
            output_state=output_state.value,
            admitted_evidence_ids=tuple(
                decision.candidate.evidence.evidence_id
                for decision in admitted
            ),
            rejected_evidence_ids=tuple(
                decision.candidate.evidence.evidence_id
                for decision in rejected
            ),
            candidate_count=len(candidates),
            generated_text=answer.text,
            confidence=answer.confidence,
            metadata={
                "system": "SCAF",
                "scores": score_metadata,
                "verification": verification_metadata,
            },
        )