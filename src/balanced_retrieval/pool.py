"""Joining the candidates of the four corpora and ordering the pool (paper p.5; the authors' ``rerank.py``).

Both functions work on plain lists and never touch a model: the search and the cross-encoder are passed in elsewhere.
A candidate is a mapping with at least ``corpus`` (one of ``CORPUS_ORDER``), ``doc_key`` and ``text``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: The four corpora, in the order in which the authors' code joins their candidates.
CORPUS_ORDER = ("pubmed", "pmc", "guidelines", "textbooks")


def check_top_k(top_k: object) -> int:
    """``top_k`` as a whole number of at least 1; a float, a bool or anything else is refused."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be a whole number, got {type(top_k).__name__}")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    return top_k


def check_corpus_names(names: Sequence[str]) -> None:
    unknown = [name for name in names if name not in CORPUS_ORDER]
    if unknown:
        raise ValueError(f"unknown corpus {', '.join(map(repr, unknown))}; the corpora are {', '.join(CORPUS_ORDER)}")


def pool_candidates(per_corpus: Mapping[str, Sequence[Any]], top_k: int) -> list[Any]:
    """The best ``top_k`` of every corpus, joined in the order PubMed, PMC, guidelines, textbooks.

    Nothing is removed: a document that two corpora both return stays twice, and a repeat inside one corpus stays.
    A corpus with fewer than ``top_k`` candidates adds fewer; the gap is never filled from another corpus. A subset of
    the corpora is allowed (for ablations); an empty mapping or an unknown corpus name is an error. The candidates are
    the very objects that came in, in a new list.
    """
    check_top_k(top_k)
    if not per_corpus:
        raise ValueError("no corpus was given")
    check_corpus_names(list(per_corpus))
    pooled: list[Any] = []
    for name in CORPUS_ORDER:
        if name in per_corpus:
            pooled.extend(list(per_corpus[name])[:top_k])
    return pooled


def rerank(candidates: Sequence[Any], scores: Sequence[float], top_k: int) -> list[Any]:
    """The ``top_k`` candidates with the highest scores, best first; equal scores keep the order of the pool.

    This is the authors' ``sorted(range(n), key=score, reverse=True)[:top_k]``, which is a stable sort. No quota per corpus
    is applied afterwards: the final list is not balanced. A score that is not a finite number is an error.
    """
    check_top_k(top_k)
    candidates = list(candidates)
    scores = list(scores)
    if len(candidates) != len(scores):
        raise ValueError(f"{len(candidates)} candidates but {len(scores)} scores")
    for position, score in enumerate(scores):
        if not math.isfinite(score):
            raise ValueError(f"the score of candidate {position} is not a finite number: {score!r}")
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return [candidates[i] for i in order[:top_k]]
