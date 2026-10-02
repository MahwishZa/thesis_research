"""Retriever and reranker interfaces, archived from ``src/common/retriever.py`` on 2026-10-02.

Superseded by ``experiments/shared/retrieval/``. These ABCs and their ``Passthrough*`` dev stubs were
written before the real MedCPT retrieval stage existed and were never invoked by any ``System`` or by
the runner: the frozen candidate set is built upstream, once, by
``experiments.shared.retrieval.RetrievalPipeline``, and every arm only replays it. They were kept in
``src.common`` for backward compatibility until an audit found no use of them in the repository, its
tests or its documentation. New code should use ``experiments.shared.retrieval`` (``Encoder``,
``CrossEncoderReranker``, ``RetrievalPipeline``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.common.evidence import Candidate


class Retriever(ABC):
    """Interface for evidence retrieval."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Sequence[Candidate]:
        raise NotImplementedError


class Reranker(ABC):
    """Interface for evidence reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> Sequence[Candidate]:
        raise NotImplementedError


class PassthroughRetriever(Retriever):
    """Development-only deterministic retriever."""

    def __init__(
        self,
        candidates: Sequence[Candidate],
    ) -> None:
        self._candidates = tuple(candidates)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Sequence[Candidate]:
        return self._candidates[:top_k]


class PassthroughReranker(Reranker):
    """Development-only identity reranker."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> Sequence[Candidate]:
        return tuple(candidates)