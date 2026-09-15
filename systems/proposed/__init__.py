"""SCAF proposed-system implementation."""

from .admission import (
    SCAFAdmissionPolicy,
    SCAFConfig,
    SCAFDecision,
    SCAFState,
    SCAFSystem,
)
from .contested import (
    ClaimConflict,
    ContestedDetector,
)
from .currency import (
    CurrencyPolicy,
    CurrencyResult,
    CurrencyState,
)
from .scorer import (
    SCAFScore,
    SCAFScorer,
    SCAFWeights,
)
from .verifier import (
    ClaimVerifier,
    NoOpVerifier,
    VerificationResult,
)

__all__ = [
    "SCAFAdmissionPolicy",
    "SCAFConfig",
    "SCAFDecision",
    "SCAFState",
    "SCAFSystem",
    "ClaimConflict",
    "ContestedDetector",
    "CurrencyPolicy",
    "CurrencyResult",
    "CurrencyState",
    "SCAFScore",
    "SCAFScorer",
    "SCAFWeights",
    "ClaimVerifier",
    "NoOpVerifier",
    "VerificationResult",
]