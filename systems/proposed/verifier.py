"""Post-hoc claim verification interface for SCAF."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..interfaces.evidence import Evidence


@dataclass(frozen=True)
class VerificationResult:
    """Result of post-hoc claim verification."""

    supported: Optional[bool]
    confidence: Optional[float] = None
    claims_checked: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


class ClaimVerifier(ABC):
    """Interface for post-generation claim verification."""

    @abstractmethod
    def verify(
        self,
        answer: str,
        evidence: Sequence[Evidence],
    ) -> VerificationResult:
        raise NotImplementedError


class NoOpVerifier(ClaimVerifier):
    """Development-only verifier.

    This does not constitute scientific verification.
    """

    def verify(
        self,
        answer: str,
        evidence: Sequence[Evidence],
    ) -> VerificationResult:

        return VerificationResult(
            supported=None,
            confidence=None,
            claims_checked=0,
            metadata={
                "verifier": "noop",
                "development_only": True,
            },
        )