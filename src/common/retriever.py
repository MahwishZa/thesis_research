"""Retriever and reranker interfaces.

**Superseded by `experiments/shared/retrieval/`.** These ABCs and their
``Passthrough*`` dev stubs were written before the real MedCPT retrieval
stage existed and are not invoked by any ``System`` or by the runner - the
frozen candidate set is built upstream, once, by
``experiments.shared.retrieval.RetrievalPipeline``, and every arm only ever replays
it. Kept for backward compatibility with anything importing
``src.common``; new code should use ``experiments.shared.retrieval``
(``Encoder``, ``CrossEncoderReranker``, ``RetrievalPipeline``) instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .evidence import Candidate


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