"""A flat, exact dense index over the corpus.

**Why exact search and not FAISS HNSW/IVF.** An approximate index returns
neighbours that depend on build order, graph parameters and, for IVF, a
training sample. Two builds of the "same" index can then answer the same query
differently, which would put a reproducibility hazard directly upstream of the
frozen candidate set - the one thing the whole comparison rests on. Exact
inner-product search has no such freedom: for a given matrix and query, the
top-k is determined.

The affordability argument: RAG² indexes 116.7M passages, where approximation
is unavoidable. This thesis indexes an Alzheimer's slice - **checked against
the actual corpus, not assumed: 4,377,041 chunks** (100% dated, per the
2026-09-21 corpus date-coverage measurement), not "a few hundred thousand" as
an earlier version of this note estimated. At MedCPT's 768 dimensions that is
a ``vectors.npy`` of **~12.8 GB in float32** (4,377,041 × 768 × 4 bytes).
Per-query search is still one matrix-vector product - fast, milliseconds -
so the *scan* is not the concern the earlier estimate implied. What the
earlier estimate got wrong is memory: a 12.8 GB array is a large fraction of
the documented hardware's 16 GB RAM (`status_and_decisions.md` §5), and
*building* it (`encoders.py::MedCPTEncoder.encode`) used to briefly hold two
copies at once - fixed 2026-09-21 by writing each batch into a preallocated
array instead of collecting a list and ``np.vstack``-ing it, which was the
actually-risky part, not the exact-vs-approximate choice this section
argues for. The approximation still buys nothing for correctness/
reproducibility and is not used; if 12.8 GB turns out not to fit alongside
everything else running, the documented mitigation is `--include-undated`
being off by default already narrows the set, and reducing further (e.g. by
source tier) is the kind of change this docstring says to record, not make
quietly.

Nothing here writes to ``corpus/``. Index artifacts live under
``experiments/results/index/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .corpus import CorpusPassage

INDEX_FORMAT = 1

#: Decimal places a similarity score is rounded to before ordering. float32
#: inner products carry noise around 1e-7, which is far below any meaningful
#: relevance difference but large enough to reorder equally-relevant passages
#: differently on different hardware. Five places is well clear of the noise
#: and well below anything the reranker would distinguish.
ORDERING_PRECISION = 5


class IndexError_(RuntimeError):
    """Raised when an index cannot be built, loaded or queried safely."""


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so inner product equals cosine similarity.

    MedCPT scores by inner product over unnormalised [CLS] vectors; the
    magnitude then reflects passage length as much as relevance. Normalising
    both sides makes the ranking a pure angle comparison and makes scores
    comparable across queries, which matters because ``rho`` is computed
    per query set.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass(frozen=True)
class Hit:
    """One retrieved passage, before reranking."""

    passage: CorpusPassage
    score: float
    rank: int


@dataclass
class DenseIndex:
    """Exact inner-product index over normalised passage vectors."""

    passage_ids: tuple[str, ...]
    vectors: np.ndarray
    encoder_name: str
    corpus_snapshot: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.passage_ids) != self.vectors.shape[0]:
            raise IndexError_(
                f"index has {len(self.passage_ids)} ids but "
                f"{self.vectors.shape[0]} vectors"
            )

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[tuple[int, float]]:
        """Return ``(row, score)`` for the top-k rows, best first.

        Ties break by row order, which is corpus file order, so a query
        hitting two identical passages resolves the same way on every run.
        """
        if top_k <= 0:
            raise IndexError_("top_k must be positive")
        query = l2_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if query.shape[1] != self.dim:
            raise IndexError_(
                f"query vector has dimension {query.shape[1]} but the index "
                f"was built at {self.dim}; the query encoder does not match "
                f"the article encoder ({self.encoder_name})"
            )
        scores = (self.vectors @ query.T).reshape(-1)
        k = min(top_k, scores.shape[0])
        # Order on scores rounded to ORDERING_PRECISION, with row index as the
        # tie-break. Without the rounding, two passages with identical text
        # score 1.0 and 1.0000001 - float32 noise from normalisation, not a
        # relevance difference - and the order between them would depend on
        # accumulation order, i.e. on the platform's BLAS. The reported score
        # stays the raw one; only the comparison is quantised.
        order = np.lexsort((
            np.arange(scores.shape[0]),
            -np.round(scores, ORDERING_PRECISION),
        ))[:k]
        return [(int(row), float(scores[row])) for row in order]

    # -- persistence -----------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        manifest = {
            "index_format": INDEX_FORMAT,
            "encoder_name": self.encoder_name,
            "corpus_snapshot": self.corpus_snapshot,
            "n_passages": len(self.passage_ids),
            "dim": self.dim,
            "passage_ids": list(self.passage_ids),
            "metadata": self.metadata,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex":
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise IndexError_(f"no index manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("index_format") != INDEX_FORMAT:
            raise IndexError_(
                f"index format {manifest.get('index_format')!r} was written by "
                f"a different version of this code (expected {INDEX_FORMAT})"
            )
        return cls(
            passage_ids=tuple(manifest["passage_ids"]),
            vectors=np.load(directory / "vectors.npy"),
            encoder_name=manifest["encoder_name"],
            corpus_snapshot=manifest["corpus_snapshot"],
            metadata=manifest.get("metadata", {}),
        )


def build_index(
    passages: Sequence[CorpusPassage],
    encoder,
    *,
    corpus_snapshot: str,
    metadata: Optional[dict[str, Any]] = None,
    on_progress: Optional[Any] = None,
) -> DenseIndex:
    """Embed every passage and build the index.

    The article encoder is applied to ``retrieval_text`` - the corpus builds
    that field with section headers attached, which is what MedCPT's article
    side is meant to receive.

    ``on_progress``, if given, is passed straight through to the encoder's
    own ``encode(on_progress=...)`` (currently only ``MedCPTEncoder``
    supports it; other encoders ignore an unknown keyword at their own
    call site, not here, so this stays a thin pass-through).
    """
    if not passages:
        raise IndexError_("cannot build an index over zero passages")
    texts = [p.retrieval_text for p in passages]
    if on_progress is not None:
        vectors = encoder.encode(texts, on_progress=on_progress)
    else:
        vectors = encoder.encode(texts)
    if vectors.shape[0] != len(passages):
        raise IndexError_(
            f"encoder returned {vectors.shape[0]} vectors for "
            f"{len(passages)} passages"
        )
    return DenseIndex(
        passage_ids=tuple(p.chunk_id for p in passages),
        vectors=l2_normalize(vectors),
        encoder_name=getattr(encoder, "name", type(encoder).__name__),
        corpus_snapshot=corpus_snapshot,
        metadata=metadata or {},
    )
