"""Recency-aware evidence admission: the proposed method.

PRIMARY - the reduced admission policy under test:

    recency.py   R(s, q, t_q), the recency score
    scorer.py    A(s) = (1 - lambda) * rho(s) + lambda * R(s)
    admission.py threshold, context budget, output state

SECONDARY - kept because the code is written and may support a qualitative
analysis or future work. NOT part of the primary experiment, off by default,
and enabling either changes what is measured:

    contested.py contested-evidence detection
    verifier.py  answer-claim verification
"""

from .admission import (
    AdmissionConfig,
    OutputState,
    PassageDecision,
    RecencyAwareAdmissionPolicy,
    RecencyAwareSystem,
)
from .recency import RecencyPolicy, RecencyResult, RecencyState
from .scorer import AdmissionScore, AdmissionScorer

# Secondary. Imported so they remain reachable, listed apart so it is clear
# they are not part of the primary path.
from .contested import ClaimConflict, ContestedDetector
from .verifier import ClaimVerifier, NoOpVerifier, VerificationResult

__all__ = [
    # primary
    "AdmissionConfig",
    "AdmissionScore",
    "AdmissionScorer",
    "OutputState",
    "PassageDecision",
    "RecencyAwareAdmissionPolicy",
    "RecencyAwareSystem",
    "RecencyPolicy",
    "RecencyResult",
    "RecencyState",
    # secondary
    "ClaimConflict",
    "ContestedDetector",
    "ClaimVerifier",
    "NoOpVerifier",
    "VerificationResult",
]
