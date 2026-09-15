"""SCAF proposed-system implementation."""

from .admission import (
    CONTESTED_BUDGET_POLICY,
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
    "CONTESTED_BUDGET_POLICY",
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