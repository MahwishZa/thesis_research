"""RAG² baseline downstream system.

Retrieval and reranking are intentionally external. The experiment runner
constructs the frozen candidate set before invoking this system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..interfaces.evidence import Candidate, ExperimentResult
from ..interfaces.generator import Generator
from ..interfaces.system import System
from .admission import (
    AdmissionDecision,
    AdmissionFilter,
)


@dataclass(frozen=True)
class RAG2Config:
    """Downstream RAG² configuration."""

    max_admitted_passages: Optional[int] = None
    context_prompt: Optional[str] = None


class RAG2System(System):
    """RAG² filtering and answer-generation stage."""

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
    ) -> tuple[
        tuple[Candidate, ...],
        tuple[AdmissionDecision, ...],
    ]:

        decisions = tuple(
            self.admission_filter.predict(
                question,
                candidate,
            )
            for candidate in candidates
        )

        admitted = tuple(
            candidate
            for candidate, decision in zip(
                candidates,
                decisions,
            )
            if decision.helpful
        )

        if self.config.max_admitted_passages is not None:
            admitted = admitted[
                : self.config.max_admitted_passages
            ]

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
    ) -> ExperimentResult:

        admitted, decisions = self.filter_candidates(
            question,
            candidates,
        )

        rejected_ids = tuple(
            candidate.evidence.evidence_id
            for candidate, decision in zip(
                candidates,
                decisions,
            )
            if not decision.helpful
        )

        evidence = tuple(
            candidate.evidence
            for candidate in admitted
        )

        prompt = self.build_answer_prompt(
            question,
            admitted,
        )

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
            output_state=None,
            admitted_evidence_ids=tuple(
                candidate.evidence.evidence_id
                for candidate in admitted
            ),
            rejected_evidence_ids=rejected_ids,
            candidate_count=len(candidates),
            generated_text=answer.text,
            confidence=answer.confidence,
            metadata={
                "system": "RAG2",
                "retrieval_external": True,
            },
        )