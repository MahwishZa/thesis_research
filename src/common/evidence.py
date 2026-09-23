"""Shared evidence and experiment data structures.

These classes define the interface between the upstream corpus/retrieval
pipeline and the three experimental arms (no-filter, RAG², Temporal Filter
 admission).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Evidence:
    """Normalized evidence passage.

    An Evidence object represents one corpus passage after normalization
    and chunking. It contains provenance and temporal metadata required by
    the downstream admission systems.
    """

    evidence_id: str
    text: str

    # Source provenance.
    source_tier: str
    persistent_id: Optional[str] = None
    journal: Optional[str] = None
    section: Optional[str] = None

    # Temporal information.
    publication_date: Optional[date] = None
    version: Optional[str] = None
    supersession_pointer: Optional[str] = None

    # Publication status.
    retracted: bool = False
    withdrawn: bool = False

    # Thesis-specific metadata.
    claim_classes: tuple[str, ...] = ()
    guideline_family: Optional[str] = None

    # Optional character offsets within the source document.
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def is_hard_invalid(self) -> bool:
        """Return True if the source is explicitly retracted/withdrawn."""
        return self.retracted or self.withdrawn


@dataclass(frozen=True)
class Candidate:
    """Evidence candidate after upstream retrieval/reranking.

    Candidate sets should be generated once and then replayed across
    experimental arms to preserve upstream identity.
    """

    evidence: Evidence

    # Reranking information.
    rerank_score: float
    rerank_rank: int

    # Optional original retrieval information.
    retrieval_score: Optional[float] = None
    retrieval_rank: Optional[int] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ExperimentResult:
    """Standardized output from an experimental system."""

    sample_id: str
    experiment_id: str
    variant: str

    prediction: Optional[str]
    output_state: Optional[str]

    admitted_evidence_ids: tuple[str, ...] = ()
    rejected_evidence_ids: tuple[str, ...] = ()

    candidate_count: int = 0

    rationale: Optional[str] = None
    generated_text: Optional[str] = None
    confidence: Optional[float] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "sample_id": self.sample_id,
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "prediction": self.prediction,
            "output_state": self.output_state,
            "admitted_evidence_ids": list(
                self.admitted_evidence_ids
            ),
            "rejected_evidence_ids": list(
                self.rejected_evidence_ids
            ),
            "candidate_count": self.candidate_count,
            "rationale": self.rationale,
            "generated_text": self.generated_text,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }