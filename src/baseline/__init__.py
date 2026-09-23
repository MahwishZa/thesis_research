"""Baseline arms: the unfiltered control and the RAG2 reproduction."""

from .admission import (
    HELPFUL,
    NOT_HELPFUL,
    RAG2_FILTER_TEMPLATE,
    AdmissionDecision,
    AdmissionFilter,
    FlanT5RAG2Filter,
    MockRAG2Filter,
)
from .no_filter import NoFilterSystem
from .rag2 import RAG2Config, RAG2System

__all__ = [
    "HELPFUL",
    "NOT_HELPFUL",
    "RAG2_FILTER_TEMPLATE",
    "AdmissionDecision",
    "AdmissionFilter",
    "FlanT5RAG2Filter",
    "MockRAG2Filter",
    "NoFilterSystem",
    "RAG2Config",
    "RAG2System",
]
