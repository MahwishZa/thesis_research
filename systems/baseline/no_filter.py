"""No-filter control: the unfiltered condition.

The third arm of the primary experiment, and the cheapest one. It applies no
admission policy at all - every candidate in the frozen set is passed to the
generator, up to the shared context budget.

It is not a strawman, it is the reference point that makes the other two arms
interpretable. Without it, RAG2 and the recency-aware policy can only be
compared to each other, and a difference between them says nothing about
whether filtering helps at all. With it:

* RAG2 vs. no-filter says what confidence-derived filtering is worth;
* recency-aware vs. no-filter says the same for the proposed policy;
* recency-aware vs. RAG2 is the thesis's central comparison, and the control
  is what rules out "the proposed policy only looks better because it admits
  more evidence".

Budget ordering is by reranker rank, which is the only ordering available
here and the same one RAG2 uses when its budget binds.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..interfaces.evidence import Candidate, ExperimentResult
from ..interfaces.generator import Generator
from ..interfaces.system import System


class NoFilterSystem(System):
    """Admit every candidate, up to the shared context budget."""

    name = "B1_NO_FILTER"

    def __init__(
        self,
        *,
        answer_generator: Generator,
        max_admitted_passages: Optional[int] = None,
        context_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            max_admitted_passages: the shared context budget. Must be the
                same value the other two arms receive.
            context_prompt: the SAME answer-prompt template as the other
                arms; prompt parity is a fairness control.
        """

        if max_admitted_passages is not None and max_admitted_passages <= 0:
            raise ValueError("max_admitted_passages must be positive.")

        self.answer_generator = answer_generator
        self.max_admitted_passages = max_admitted_passages
        self.context_prompt = context_prompt

    def select(
        self,
        candidates: Sequence[Candidate],
    ) -> tuple[Candidate, ...]:
        """Best-ranked candidates up to the budget, deterministically."""

        ordered = sorted(
            candidates,
            key=lambda c: (c.rerank_rank, c.evidence.evidence_id),
        )

        if self.max_admitted_passages is None:
            return tuple(ordered)

        return tuple(ordered[: self.max_admitted_passages])

    def build_prompt(
        self,
        question: str,
        admitted: Sequence[Candidate],
    ) -> str:

        context = "\n\n".join(
            (
                f"[{candidate.evidence.evidence_id}] "
                f"{candidate.evidence.text}"
            )
            for candidate in admitted
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

        admitted = self.select(candidates)
        admitted_ids = {c.evidence.evidence_id for c in admitted}

        evidence = tuple(candidate.evidence for candidate in admitted)

        answer = self.answer_generator.generate(
            question,
            evidence,
            prompt=self.build_prompt(question, admitted),
        )

        return ExperimentResult(
            sample_id=sample_id,
            experiment_id=experiment_id,
            variant=self.name,
            prediction=answer.text,
            # No admission policy, so no output-state policy either.
            output_state=None,
            admitted_evidence_ids=tuple(
                c.evidence.evidence_id for c in admitted
            ),
            rejected_evidence_ids=tuple(
                c.evidence.evidence_id
                for c in candidates
                if c.evidence.evidence_id not in admitted_ids
            ),
            candidate_count=len(candidates),
            rationale=rationale,
            generated_text=answer.text,
            confidence=answer.confidence,
            metadata={
                "system": self.name,
                "filtering": "none",
                "context_budget": {
                    "limit": self.max_admitted_passages,
                    "budget_configured": (
                        self.max_admitted_passages is not None
                    ),
                    "admitted": len(admitted),
                },
            },
        )
