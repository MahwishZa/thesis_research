"""The evidence records of the stage-1 arms.

These classes carry a candidate passage and its dates to the admission score of the stage-1
arms (``experiments/medchange/stage1/scorer.py``, called by ``experiments/medchange/arms.py``). The
current systems (``experiments/medchange/rag2.py``) work on plain dictionaries and do not use them.
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
