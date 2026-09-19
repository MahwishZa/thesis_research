"""Recency-aware evidence admission: the proposed method.

This is an admission policy, not a framework. It scores each candidate in a
frozen candidate set, admits those at or above a threshold, and caps the
admitted set at the shared context budget:

    1. score each candidate: A(s) = (1 - lambda) * rho(s) + lambda * R(s)
    2. admit those with A(s) >= theta
    3. cap at the context budget by score rank, deterministically
    4. report the output state

Everything that made the earlier design a "framework" - entailment-derived
support, source authority, contested-state handling, supersession, answer
verification - is outside the primary experiment. Contested handling can
still be switched on for a secondary analysis (pass ``contested``), and it
stays off unless it is.

The point of keeping this small is attribution, not tidiness. The baseline
and this policy receive the same questions, the same candidate set, the same
reranker scores and the same context budget; the ONLY difference is that this
policy can see publication dates. With one added signal and one weight, a
difference in what gets admitted is attributable to the temporal signal. With
four weighted components it would not be.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import Optional, Sequence

from ..interfaces.evidence import Candidate, ExperimentResult
from ..interfaces.generator import Generator
from ..interfaces.system import System
from .recency import RecencyPolicy
from .scorer import AdmissionScore, AdmissionScorer


class OutputState(str, Enum):
    """System-level output states.

    These describe the answer as a whole, not individual passages: a rejected
    candidate carries ``state=None``, because ABSTAIN means "the available
    evidence is insufficient" and is a property of the item.

    GROUNDED and ABSTAIN are the primary states. CONTESTED is produced only
    when the secondary contested detector is enabled.
    """

    GROUNDED = "GROUNDED"
    ABSTAIN = "ABSTAIN"
    #: Answered with an empty evidence block, under ANSWER_ALWAYS. Not
    #: GROUNDED - there was nothing to ground on - and not ABSTAIN, because
    #: an answer was produced and must be annotated like any other.
    UNGROUNDED = "UNGROUNDED"
    CONTESTED = "CONTESTED"


class AbstentionPolicy(str, Enum):
    """What the proposed system does when no passage clears theta.

    The baseline does NOT abstain: when its filter admits nothing it still
    generates, from an empty evidence block. If this arm abstains in the same
    situation, the two arms behave differently for a reason that has nothing
    to do with the intervention, and the hallucination rate stops being
    comparable - an abstention makes no claims, so it can never be labelled
    hallucinated.

    ``ANSWER_ALWAYS`` is the default because it restores parity by
    construction: both arms generate from whatever they admitted, including
    nothing. ``ABSTAIN_WHEN_EMPTY`` preserves the earlier behaviour and stays
    available as a declared secondary condition, never as the primary
    comparison.
    """

    #: Generate from the admitted evidence, even when that set is empty.
    ANSWER_ALWAYS = "answer_always"
    #: Decline to answer when nothing clears theta.
    ABSTAIN_WHEN_EMPTY = "abstain_when_empty"


@dataclass(frozen=True)
class AdmissionConfig:
    """Experiment configuration. Unresolved values raise rather than default."""

    #: theta. Fitted on the validation split, never on the test set.
    admit_threshold: Optional[float] = None

    #: t_q, the information state the decision is evaluated against. Set per
    #: item during question construction; never derived from the evidence
    #: being scored.
    question_date: Optional[date] = None

    #: The shared context budget. Must be identical across all three arms,
    #: or an admission difference is confounded with context volume.
    max_admitted_passages: Optional[int] = None

    #: What to do when no passage clears theta. Defaults to answering, which
    #: is what the baseline does in the same situation.
    abstention_policy: AbstentionPolicy = AbstentionPolicy.ANSWER_ALWAYS

    def validate(self) -> None:

        if self.admit_threshold is None:
            raise ValueError(
                "admit_threshold (theta) is unresolved. Fit it on the "
                "validation split before any test run."
            )

        # A(s) is a convex combination of two quantities in [0, 1], so it is
        # itself in [0, 1] and theta is only meaningful on that scale. Out of
        # range, theta is not a wrong setting but a silently degenerate one:
        # above 1 nothing ever clears it, so the arm answers every question
        # from an empty context; below 0 everything clears it, so the arm
        # stops admitting at all. Either reads as a real (terrible or
        # indiscriminate) result rather than as a misconfiguration, and a
        # validation-split sweep that strays outside [0, 1] would produce
        # exactly that with no error.
        if not 0.0 <= float(self.admit_threshold) <= 1.0:
            raise ValueError("admit_threshold (theta) must be in [0, 1].")

        if self.question_date is None:
            raise ValueError(
                "question_date (t_q) is required: the recency score is "
                "undefined without an as-of date."
            )

        if (
            self.max_admitted_passages is not None
            and self.max_admitted_passages <= 0
        ):
            raise ValueError("max_admitted_passages must be positive.")


@dataclass(frozen=True)
class PassageDecision:
    """The decision for one candidate passage."""

    candidate: Candidate
    score: AdmissionScore

    admitted: bool
    state: Optional[OutputState]
    recency_state: str

    #: True when the passage cleared the threshold but was dropped to stay
    #: inside the context budget.
    dropped_for_budget: bool = False

    #: Secondary: set only when the contested detector is enabled.
    contested: bool = False


class RecencyAwareAdmissionPolicy:
    """Decide which candidates to admit."""

    def __init__(
        self,
        *,
        scorer: AdmissionScorer,
        recency: RecencyPolicy,
        config: AdmissionConfig,
        contested: Optional[object] = None,
    ) -> None:
        """
        Args:
            contested: SECONDARY, default None (off). A
                ``ContestedDetector`` for the qualitative contested-evidence
                analysis. It plays no part in the primary experiment, and
                enabling it changes which passages are admitted, so it is
                recorded in the run metadata.
        """

        config.validate()

        self.scorer = scorer
        self.recency = recency
        self.config = config
        self.contested = contested

    def decide(
        self,
        question: str,
        candidates: Sequence[Candidate],
    ) -> tuple[PassageDecision, ...]:

        if not candidates:
            return ()

        question_date = self.config.question_date

        if question_date is None:
            # Defensive: validate() enforces this. Not an assert, because
            # assertions are stripped under `python -O`.
            raise ValueError("question_date is required.")

        contested_ids: set[str] = set()

        if self.contested is not None:
            conflicts = self.contested.find_conflicts(candidates)
            for conflict in conflicts:
                contested_ids.add(conflict.evidence_a)
                contested_ids.add(conflict.evidence_b)

        decisions: list[PassageDecision] = []

        for candidate in candidates:

            recency_result = self.recency.score(
                candidate.evidence,
                question=question,
                question_date=question_date,
            )

            score = self.scorer.score(
                candidate,
                recency=recency_result.score,
                candidate_count=len(candidates),
            )

            admitted = score.total >= self.config.admit_threshold
            is_contested = candidate.evidence.evidence_id in contested_ids

            state: Optional[OutputState]
            if not admitted:
                state = None
            elif is_contested:
                state = OutputState.CONTESTED
            else:
                state = OutputState.GROUNDED

            decisions.append(
                PassageDecision(
                    candidate=candidate,
                    score=score,
                    admitted=admitted,
                    state=state,
                    recency_state=recency_result.state.value,
                    contested=is_contested,
                )
            )

        return self._apply_budget(decisions)

    def _apply_budget(
        self,
        decisions: Sequence[PassageDecision],
    ) -> tuple[PassageDecision, ...]:
        """Keep the best-scoring admitted passages, up to the budget.

        Ordering is A(s) descending with ``evidence_id`` as a deterministic
        tie-break, so two runs over identical inputs select identically.
        """

        limit = self.config.max_admitted_passages

        if limit is None:
            return tuple(decisions)

        ranked = sorted(
            (d for d in decisions if d.admitted),
            key=lambda d: (-d.score.total, d.candidate.evidence.evidence_id),
        )

        keep = {
            d.candidate.evidence.evidence_id for d in ranked[:limit]
        }

        return tuple(
            decision
            if (
                not decision.admitted
                or decision.candidate.evidence.evidence_id in keep
            )
            else replace(
                decision,
                admitted=False,
                state=None,
                dropped_for_budget=True,
            )
            for decision in decisions
        )


class RecencyAwareSystem(System):
    """The proposed arm: recency-aware admission plus answer generation."""

    name = "P_RECENCY"

    def __init__(
        self,
        *,
        answer_generator: Generator,
        admission_policy: RecencyAwareAdmissionPolicy,
        context_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            context_prompt: answer-prompt template with ``{question}`` and
                ``{context}`` fields. Prompt parity is required across arms,
                so the runner must pass the SAME template here, to the
                baseline and to the no-filter control.
        """

        self.answer_generator = answer_generator
        self.admission_policy = admission_policy
        self.context_prompt = context_prompt

    def build_prompt(
        self,
        question: str,
        decisions: Sequence[PassageDecision],
    ) -> str:

        context = "\n\n".join(
            (
                f"[{decision.candidate.evidence.evidence_id}] "
                f"{decision.candidate.evidence.text}"
            )
            for decision in decisions
            if decision.admitted
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

        config = self.admission_policy.config

        base_metadata = {
            "system": self.name,
            "scores": {
                d.candidate.evidence.evidence_id: {
                    "total": d.score.total,
                    "relevance": d.score.relevance,
                    "recency": d.score.recency,
                    "recency_state": d.recency_state,
                    "admitted": d.admitted,
                    "dropped_for_budget": d.dropped_for_budget,
                    "contested": d.contested,
                }
                for d in decisions
            },
            "recency_weight": self.admission_policy.scorer.recency_weight,
            "admit_threshold": config.admit_threshold,
            "abstention_policy": config.abstention_policy.value,
            "half_life_days": self.admission_policy.recency.half_life_days,
            "question_date": config.question_date.isoformat(),
            # Empty in the primary experiment. Recorded so a result is never
            # read as plain recency scoring when a secondary rule was on.
            "recency_secondary_rules": list(
                self.admission_policy.recency.secondary_rules_enabled
            ),
            "contested_detection_enabled": (
                self.admission_policy.contested is not None
            ),
            # Must match the other two arms exactly (fairness control).
            "context_budget": {
                "limit": config.max_admitted_passages,
                "budget_configured": config.max_admitted_passages is not None,
                "admitted": len(admitted),
                "dropped_for_budget": [
                    d.candidate.evidence.evidence_id
                    for d in decisions
                    if d.dropped_for_budget
                ],
            },
        }

        abstains = (
            config.abstention_policy
            is AbstentionPolicy.ABSTAIN_WHEN_EMPTY
        )
        if not admitted and abstains:
            return ExperimentResult(
                sample_id=sample_id,
                experiment_id=experiment_id,
                variant=self.name,
                prediction=None,
                output_state=OutputState.ABSTAIN.value,
                admitted_evidence_ids=(),
                rejected_evidence_ids=tuple(
                    d.candidate.evidence.evidence_id for d in rejected
                ),
                candidate_count=len(candidates),
                rationale=rationale,
                metadata={**base_metadata, "reason": "no_admitted_evidence"},
            )

        evidence = tuple(d.candidate.evidence for d in admitted)
        answer = self.answer_generator.generate(
            question,
            evidence,
            prompt=self.build_prompt(question, decisions),
        )

        if not admitted:
            output_state = OutputState.UNGROUNDED
        elif any(d.state is OutputState.CONTESTED for d in admitted):
            output_state = OutputState.CONTESTED
        else:
            output_state = OutputState.GROUNDED

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
            metadata=base_metadata,
        )
