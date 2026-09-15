"""Retriever and reranker interfaces."""

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