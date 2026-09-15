"""Common interface for experimental systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from .evidence import Candidate, ExperimentResult


class System(ABC):
    """Experimental system operating on a frozen candidate set.

    Retrieval and reranking happen once, upstream. Every arm receives the
    same serialised candidate set, so differences between arms are
    attributable to admission policy alone (specification section 16).
    """

    name: str

    @abstractmethod
    def run(
        self,
        *,
        sample_id: str,
        experiment_id: str,
        question: str,
        candidates: Sequence[Candidate],
        rationale: Optional[str] = None,
    ) -> ExperimentResult:
        """Run the system over one evaluation item.

        Args:
            rationale: the rationale used upstream as the retrieval query.
                Specification section 16 lists the rationale among the
                quantities that must be invariant across arms; it is passed
                in and recorded so that invariance can be verified after the
                run. The system does not regenerate it.
        """
        raise NotImplementedError
