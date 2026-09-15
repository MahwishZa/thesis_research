"""RAG² baseline implementation."""

from .admission import (
    HELPFUL,
    NOT_HELPFUL,
    RAG2_FILTER_TEMPLATE,
    AdmissionDecision,
    AdmissionFilter,
    FlanT5RAG2Filter,
    MockRAG2Filter,
)
from .rag2 import (
    RAG2Config,
    RAG2System,
)

__all__ = [
    "HELPFUL",
    "NOT_HELPFUL",
    "RAG2_FILTER_TEMPLATE",
    "AdmissionDecision",
    "AdmissionFilter",
    "FlanT5RAG2Filter",
    "MockRAG2Filter",
    "RAG2Config",
    "RAG2System",
]
