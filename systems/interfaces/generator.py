"""Common text-generation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .evidence import Evidence


@dataclass(frozen=True)
class GenerationResult:
    """Normalized generator output."""

    text: str
    confidence: Optional[float] = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


class Generator(ABC):
    """Interface for rationale and answer generation."""

    @abstractmethod
    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence] = (),
        *,
        prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from a question and optional evidence."""
        raise NotImplementedError


class CallableGenerator(Generator):
    """Adapter around an arbitrary Python callable.

    Useful for development, tests, Transformers, vLLM wrappers, or
    other model backends.
    """

    def __init__(self, function: Any) -> None:
        self._function = function

    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence] = (),
        *,
        prompt: Optional[str] = None,
    ) -> GenerationResult:

        result = self._function(
            question,
            list(evidence),
            prompt,
        )

        if isinstance(result, GenerationResult):
            return result

        return GenerationResult(
            text=str(result)
        )