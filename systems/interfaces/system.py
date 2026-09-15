"""Common interface for experimental systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .evidence import Candidate, ExperimentResult


class System(ABC):
    """Experimental system operating on a candidate set."""

    name: str

    @abstractmethod
    def run(
        self,
        *,
        sample_id: str,
        experiment_id: str,
        question: str,
        candidates: Sequence[Candidate],
    ) -> ExperimentResult:
        """Run the system."""
        raise NotImplementedError