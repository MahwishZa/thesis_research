"""One question through balanced retrieval: the rationale searches every corpus, one cross-encoder orders the pool.

This is the wiring of Part 1 and Part 2 (paper section 3.4). The rationale alone is the search text; the question is not
added. Every corpus is asked for the same ``top_k`` candidates, the lists are pooled, the pool is scored once, and the best
``top_k`` stay. The searches and the scorer are passed in, so the function is tested without a model or an index.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rationale_query.search_text import dense_query, rerank_query

from .pool import CORPUS_ORDER, check_corpus_names, check_top_k, pool_candidates, rerank

RERANK_AGAINST = ("question", "rationale")

Searcher = Callable[[str, int], Sequence[Mapping[str, Any]]]
Scorer = Callable[[str, Sequence[Mapping[str, Any]]], Sequence[float]]


def balanced_retrieve(
    *,
    question: str,
    rationale: str,
    searchers: Mapping[str, Searcher],
    scorer: Scorer,
    top_k: int,
    rerank_against: str = "question",
) -> list[Mapping[str, Any]]:
    """The final evidence list of one question.

    ``searchers`` maps a corpus name to ``search(query_text, k)``, which returns that corpus's best candidates, best first.
    ``scorer(text, candidates)`` returns one relevance score per candidate. The re-ranker reads the question, as the paper
    says; ``rerank_against="rationale"`` is the variant of the authors' released code, run as a labelled sensitivity check.
    Everything is checked before the first search. A search that fails is an error, never a corpus that is left out, and
    an empty pool gives an empty list without calling the scorer.
    """
    check_top_k(top_k)
    if rerank_against not in RERANK_AGAINST:
        raise ValueError(f"rerank_against must be one of {', '.join(RERANK_AGAINST)}, got {rerank_against!r}")
    if not searchers:
        raise ValueError("no corpus was given")
    check_corpus_names(list(searchers))
    search_text = dense_query(rationale)
    score_text = rerank_query(question=question, rationale=rationale, against=rerank_against)

    per_corpus: dict[str, list[Mapping[str, Any]]] = {}
    for name in CORPUS_ORDER:
        if name not in searchers:
            continue
        hits = list(searchers[name](search_text, top_k))
        for hit in hits:
            if hit.get("corpus") != name:
                raise ValueError(f"the search of {name} returned a candidate of corpus {hit.get('corpus')!r}")
        per_corpus[name] = hits

    pooled = pool_candidates(per_corpus, top_k)
    if not pooled:
        return []
    scores = list(scorer(score_text, pooled))
    if len(scores) != len(pooled):
        raise ValueError(f"the scorer returned {len(scores)} scores for {len(pooled)} candidates")
    return rerank(pooled, scores, top_k)
