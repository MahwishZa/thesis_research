"""RAG² baseline downstream system.

Scope of this module. The full RAG² pipeline is four stages:

    question -> [1] rationale generation -> [2] retrieval + reranking
             -> [3] filtering -> [4] answer generation

Stages 1 and 2 are performed ONCE by the experiment runner and serialised, so
that every arm replays a byte-identical candidate set (specification section
16). This module implements stages 3 and 4 over that frozen set. The rationale
produced by stage 1 is passed in only to be recorded, never regenerated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..interfaces.evidence import Candidate, ExperimentResult
from ..interfaces.generator import Generator
from ..interfaces.system import System
from .admission import AdmissionDecision, AdmissionFilter


@dataclass(frozen=True)
class RAG2Config:
    """Downstream RAG² configuration."""

    # Matched context budget. RAG² itself keeps every [HELPFUL] passage; this
    # cap is a thesis-level control so both arms receive the same amount of
    # evidence (specification section 16). None means "keep all [HELPFUL]",
    # which is faithful RAG² behaviour.
    max_admitted_passages: Optional[int] = None

    # Answer-prompt template with {question} and {context} fields.
    # Specification section 16 requires prompt parity, so the runner should
    # pass the SAME template to this system and to SCAF.
    context_prompt: Optional[str] = None


class RAG2System(System):
    """RAG² filtering and answer-generation stages."""

    name = "B2_RAG2"

    def __init__(
        self,
        *,
        answer_generator: Generator,
        admission_filter: AdmissionFilter,
        config: RAG2Config,
    ) -> None:

        self.answer_generator = answer_generator
        self.admission_filter = admission_filter
        self.config = config

    def filter_candidates(
        self,
        question: str,
        candidates: Sequence[Candidate],
    ) -> tuple[tuple[Candidate, ...], tuple[AdmissionDecision, ...]]:
        """Label every candidate, then apply the context budget.

        The budget keeps the best-ranked [HELPFUL] passages. RAG²'s filter
        emits a binary label, not a ranking, so reranker rank is the only
        ordering available; ``evidence_id`` breaks ties so that two runs over
        identical inputs select identically.
        """

        decisions = tuple(
            self.admission_filter.predict(question, candidate)
            for candidate in candidates
        )

        helpful = tuple(
            candidate
            for candidate, decision in zip(candidates, decisions)
            if decision.helpful
        )

        if self.config.max_admitted_passages is None:
            return helpful, decisions

        ordered = sorted(
            helpful,
            key=lambda c: (c.rerank_rank, c.evidence.evidence_id),
        )
        keep = {
            c.evidence.evidence_id
            for c in ordered[: self.config.max_admitted_passages]
        }

        admitted = tuple(
            c for c in helpful if c.evidence.evidence_id in keep
        )

        return admitted, decisions

    def build_answer_prompt(
        self,
        question: str,
        evidence: Sequence[Candidate],
    ) -> str:

        context = "\n\n".join(
            (
                f"[{candidate.evidence.evidence_id}] "
                f"{candidate.evidence.text}"
            )
            for candidate in evidence
        )

        if self.config.context_prompt is not None:
            return self.config.context_prompt.format(
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

        admitted, decisions = self.filter_candidates(question, candidates)

        admitted_ids = {c.evidence.evidence_id for c in admitted}

        rejected_ids = tuple(
            candidate.evidence.evidence_id
            for candidate in candidates
            if candidate.evidence.evidence_id not in admitted_ids
        )

        evidence = tuple(candidate.evidence for candidate in admitted)

        prompt = self.build_answer_prompt(question, admitted)

        answer = self.answer_generator.generate(
            question,
            evidence,
            prompt=prompt,
        )

        return ExperimentResult(
            sample_id=sample_id,
            experiment_id=experiment_id,
            variant=self.name,
            prediction=answer.text,
            # RAG² has no output-state policy; only SCAF does.
            output_state=None,
            admitted_evidence_ids=tuple(
                candidate.evidence.evidence_id for candidate in admitted
            ),
            rejected_evidence_ids=rejected_ids,
            candidate_count=len(candidates),
            rationale=rationale,
            generated_text=answer.text,
            confidence=answer.confidence,
            metadata={
                "system": "RAG2",
                "retrieval_external": True,
                "filter_decisions": {
                    decision.evidence_id: {
                        "label": decision.label,
                        "probability_helpful": (
                            decision.probability_helpful
                        ),
                    }
                    for decision in decisions
                },
            },
        )
