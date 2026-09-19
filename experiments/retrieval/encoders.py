"""MedCPT encoders and the cross-encoder reranker.

RAG² uses MedCPT at both stages: a dual-encoder for retrieval and a
cross-encoder for reranking (`docs/status_and_decisions.md`, fact E6). The
thesis keeps both, over its own Alzheimer's corpus rather than RAG²'s 564 GB
general-medical one. That substitution of *corpus* is the thesis's declared
adaptation; the *method* is unchanged.

``torch`` and ``transformers`` are imported inside the methods, not at module
import. The test suite and every offline tool must keep working on a machine
with neither installed, and a top-level import would make retrieval code
unimportable there.

Determinism: all three models run in ``eval`` mode under ``torch.no_grad()``
with no sampling anywhere. Encoding the same text twice on the same machine
and build yields the same vector.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

#: The published MedCPT checkpoints RAG² uses.
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"

#: MedCPT's published input limits.
QUERY_MAX_TOKENS = 64
ARTICLE_MAX_TOKENS = 512
PAIR_MAX_TOKENS = 512


class Encoder(ABC):
    """Turns text into a matrix of row vectors, one per input."""

    #: Recorded into the index manifest so an index cannot be silently
    #: queried with vectors from a different encoder.
    name: str

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class CrossEncoderReranker(ABC):
    """Scores (query, passage) pairs jointly."""

    name: str

    @abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


def _load(model_id: str, kind: str):
    """Import torch/transformers lazily and load one checkpoint."""
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"{model_id} needs torch and transformers installed. Retrieval "
            "runs on the machine holding the corpus; this environment does "
            "not have them."
        ) from exc

    if kind == "sequence_classification":
        from transformers import AutoModelForSequenceClassification as Model
    else:
        from transformers import AutoModel as Model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Model.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


class MedCPTEncoder(Encoder):
    """Dense encoder for queries or articles.

    MedCPT is a dual-encoder: queries and articles go through *different*
    checkpoints and are compared in a shared space. Using one checkpoint for
    both would silently degrade retrieval, so which one this is must be
    explicit at construction.
    """

    def __init__(self, model_id: str, *, max_tokens: int,
                 batch_size: int = 32, device: Optional[str] = None) -> None:
        self.name = model_id
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = device
        self._tokenizer = None
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._tokenizer, self._model = _load(self.model_id, "encoder")
            if self.device:
                self._model.to(self.device)
        return self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        tokenizer, model = self._ensure()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                encoded = tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=self.max_tokens, return_tensors="pt",
                )
                if self.device:
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                # MedCPT reads the [CLS] representation, per its model card.
                hidden = model(**encoded).last_hidden_state[:, 0, :]
                out.append(hidden.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def medcpt_query_encoder(**kwargs) -> MedCPTEncoder:
    return MedCPTEncoder(QUERY_ENCODER, max_tokens=QUERY_MAX_TOKENS, **kwargs)


def medcpt_article_encoder(**kwargs) -> MedCPTEncoder:
    return MedCPTEncoder(ARTICLE_ENCODER, max_tokens=ARTICLE_MAX_TOKENS,
                         **kwargs)


class MedCPTReranker(CrossEncoderReranker):
    """MedCPT cross-encoder: one forward pass per (query, passage) pair.

    Cost scales with candidates per question, not corpus size, so this stays
    cheap even on CPU: retrieval depth times the question count.
    """

    def __init__(self, model_id: str = CROSS_ENCODER, *, batch_size: int = 16,
                 device: Optional[str] = None) -> None:
        self.name = model_id
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self._tokenizer = None
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._tokenizer, self._model = _load(
                self.model_id, "sequence_classification")
            if self.device:
                self._model.to(self.device)
        return self._tokenizer, self._model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        import torch

        tokenizer, model = self._ensure()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(passages), self.batch_size):
                batch = list(passages[start:start + self.batch_size])
                encoded = tokenizer(
                    [query] * len(batch), batch, truncation=True,
                    padding=True, max_length=PAIR_MAX_TOKENS,
                    return_tensors="pt",
                )
                if self.device:
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = model(**encoded).logits.squeeze(-1)
                scores.append(
                    logits.reshape(-1).cpu().numpy().astype(np.float32))
        return (np.concatenate(scores) if scores
                else np.zeros((0,), dtype=np.float32))


class HashingEncoder(Encoder):
    """Deterministic stand-in for tests. **Never a thesis result.**

    Produces a stable pseudo-random vector per text by hashing. It has no
    semantics whatsoever: it exists so the index, retrieval and freezing code
    can be exercised on a machine without torch. Any retrieval quality it
    appears to show is an artifact of hashing.
    """

    name = "hashing-stub"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return (np.asarray(rows, dtype=np.float32) if rows
                else np.zeros((0, self.dim), dtype=np.float32))


class LexicalOverlapReranker(CrossEncoderReranker):
    """Deterministic stand-in reranker for tests. **Never a thesis result.**

    Scores by token overlap with the query. Unlike the hashing encoder this
    is weakly meaningful, which is useful for asserting that reranking
    reorders a candidate list at all - but it is not MedCPT and must never
    stand in for it in a run that produces numbers.
    """

    name = "lexical-overlap-stub"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        wanted = set(query.lower().split())
        return np.asarray(
            [len(wanted & set(p.lower().split())) / (len(wanted) or 1)
             for p in passages],
            dtype=np.float32,
        )
