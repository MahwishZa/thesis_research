"""Question → retrieval → reranking → frozen candidate set.

This is the upstream stage, and it runs **once per question**. Its output is
frozen and replayed byte-identically to every arm, so that a difference in
answers is attributable to admission and not to what was retrieved. Neither
arm ever calls anything in this module.

The stage mirrors RAG²'s: dense retrieval with the MedCPT query encoder, then
MedCPT cross-encoder reranking of the retrieved set. One deviation is recorded
in ``RetrievalConfig``: RAG² retrieves with the generated *rationale* as query
because the original question is too long for the encoder; whether a rationale
is used here is explicit, because it changes what was retrieved and must not
be an accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from evaluation.freezing import FrozenCandidate

from .corpus import CorpusPassage
from .index import ORDERING_PRECISION, DenseIndex, Hit

#: Engineering constants, decided in
#: _archive/docs_legacy/research_experimental_specification.md §8.2.
DEFAULT_RETRIEVAL_DEPTH = 50
DEFAULT_CANDIDATE_COUNT = 20


class RetrievalError(RuntimeError):
    """Raised when a candidate set cannot be built safely."""


@dataclass(frozen=True)
class RetrievalConfig:
    """Everything that decides what a question retrieves.

    Recorded alongside the frozen evidence: two candidate sets built under
    different settings are not interchangeable, and the only defence against
    mixing them is writing the settings down next to the output.
    """

    #: How deep the dense retriever goes before reranking.
    retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH
    #: How many survive reranking to become the candidate set. Must be the
    #: same for every question: rho is a within-set rank, so theta is only
    #: comparable across questions at fixed N.
    candidate_count: int = DEFAULT_CANDIDATE_COUNT
    #: RAG² queries with the generated rationale, not the question. Opt-in
    #: and recorded, because it changes the retrieved set.
    query_with_rationale: bool = False
    #: Retracted passages stay retrievable in the corpus by design. Whether
    #: they may enter a candidate set is an experiment-level decision.
    exclude_retracted: bool = True

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise RetrievalError("candidate_count must be positive")
        if self.retrieval_depth < self.candidate_count:
            raise RetrievalError(
                f"retrieval_depth ({self.retrieval_depth}) is below "
                f"candidate_count ({self.candidate_count}); reranking cannot "
                "produce more candidates than were retrieved"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_depth": self.retrieval_depth,
            "candidate_count": self.candidate_count,
            "query_with_rationale": self.query_with_rationale,
            "exclude_retracted": self.exclude_retracted,
        }


@dataclass(frozen=True)
class RetrievedSet:
    """One question's candidate set, ready to freeze."""

    question_id: str
    query_text: str
    candidates: tuple[FrozenCandidate, ...]
    corpus_snapshot: str
    provenance: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    """Dense retrieval plus cross-encoder reranking over one index."""

    def __init__(
        self,
        *,
        index: DenseIndex,
        passages: Sequence[CorpusPassage],
        query_encoder,
        reranker,
        config: RetrievalConfig,
    ) -> None:
        if len(passages) != len(index.passage_ids):
            raise RetrievalError(
                f"{len(passages)} passages given but the index holds "
                f"{len(index.passage_ids)}; they must be the same set in the "
                "same order"
            )
        for passage, indexed_id in zip(passages, index.passage_ids):
            if passage.chunk_id != indexed_id:
                raise RetrievalError(
                    f"passage order does not match the index at "
                    f"{indexed_id!r}; row order decides retrieval ties"
                )
        self.index = index
        self.passages = tuple(passages)
        self.query_encoder = query_encoder
        self.reranker = reranker
        self.config = config

    def retrieve(self, query: str) -> list[Hit]:
        """Dense retrieval, deepest first, before reranking."""
        vector = self.query_encoder.encode([query])
        if vector.shape[0] != 1:
            raise RetrievalError("query encoder must return exactly one vector")
        hits = []
        for rank, (row, score) in enumerate(
            self.index.search(vector[0], top_k=self.config.retrieval_depth), 1
        ):
            passage = self.passages[row]
            if self.config.exclude_retracted and passage.retracted:
                continue
            hits.append(Hit(passage=passage, score=score, rank=rank))
        return hits

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[tuple[Hit, float]]:
        """Score every retrieved passage against the query, jointly.

        RAG² reranks with the **original question**, even when retrieval used
        the rationale (fact E6). That asymmetry is deliberate in the paper and
        is preserved here: this method is always called with the question.
        """
        if not hits:
            return []
        scores = self.reranker.score(query, [h.passage.text for h in hits])
        if len(scores) != len(hits):
            raise RetrievalError(
                f"reranker returned {len(scores)} scores for {len(hits)} hits")
        paired = list(zip(hits, (float(s) for s in scores)))
        # Descending score, quantised for the same reason the index quantises
        # (float noise must not decide order), with chunk_id as the tie-break
        # so two runs agree exactly. The reported score is the raw one.
        paired.sort(key=lambda pair: (-round(pair[1], ORDERING_PRECISION),
                                      pair[0].passage.chunk_id))
        return paired

    def build_candidate_set(
        self,
        *,
        question_id: str,
        question: str,
        rationale: Optional[str] = None,
    ) -> RetrievedSet:
        """Produce one question's frozen-ready candidate set.

        Raises rather than return a short set: ``rho`` is a within-set rank,
        so a question with fewer candidates than the others would sit on a
        different threshold scale, and ``theta`` would no longer mean the same
        thing across items.
        """
        if self.config.query_with_rationale:
            if not rationale:
                raise RetrievalError(
                    f"{question_id}: config sets query_with_rationale but no "
                    "rationale was supplied"
                )
            query_text = rationale
        else:
            query_text = question

        hits = self.retrieve(query_text)
        # Reranking always uses the original question, per RAG².
        reranked = self.rerank(question, hits)[: self.config.candidate_count]

        if len(reranked) < self.config.candidate_count:
            raise RetrievalError(
                f"{question_id}: only {len(reranked)} candidates survived "
                f"retrieval, but candidate_count is "
                f"{self.config.candidate_count}. Every question must have the "
                "same candidate-set size or theta is not comparable across "
                "items."
            )

        candidates = tuple(
            FrozenCandidate(
                evidence_id=hit.passage.chunk_id,
                text=hit.passage.text,
                retrieval_rank=hit.rank,
                retrieval_score=hit.score,
                rerank_rank=position,
                rerank_score=rerank_score,
                publication_date=hit.passage.publication_date,
                source_metadata={
                    "source_tier": hit.passage.source_tier,
                    "persistent_id": hit.passage.document_id,
                    "section": hit.passage.section,
                },
            )
            for position, (hit, rerank_score) in enumerate(reranked, 1)
        )

        return RetrievedSet(
            question_id=question_id,
            query_text=query_text,
            candidates=candidates,
            corpus_snapshot=self.index.corpus_snapshot,
            provenance={
                "query_encoder": getattr(self.query_encoder, "name",
                                         type(self.query_encoder).__name__),
                "article_encoder": self.index.encoder_name,
                "reranker": getattr(self.reranker, "name",
                                    type(self.reranker).__name__),
                "config": self.config.to_dict(),
            },
        )
