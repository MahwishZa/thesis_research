"""MedCPT encoders and the cross-encoder reranker.

RAG² uses MedCPT at both stages: a dual-encoder for retrieval and a
cross-encoder for reranking (`_archive/docs_legacy/status_and_decisions.md`, fact E6). The
thesis keeps both, over its own Alzheimer's corpus rather than RAG²'s 564 GB
general-medical one. That substitution of *corpus* is the thesis's declared
adaptation; the *method* is unchanged.

``torch`` and ``transformers`` are imported inside the methods, not at module
import. The test suite and every offline tool must keep working on a machine
with neither installed, and a top-level import would make retrieval code
unimportable there.

Determinism: all three models run in ``eval`` mode under
``torch.inference_mode()`` (strictly disables autograd bookkeeping, unlike
``no_grad()`` which only disables gradient tracking - faster, same numbers)
with no sampling anywhere. Encoding the same text twice on the same machine
and build yields the same vector.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

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

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = Model.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def _move_to_device(model, device: str, model_id: str):
    """``model.to(device)``, but with an error a person can act on.

    The raw failure mode this guards against is real and was hit on a real
    run (2026-09-21): asking for ``cuda`` when torch itself has no CUDA
    support raises ``AssertionError: Torch not compiled with CUDA enabled``
    eight stack frames deep inside torch's own ``Module._apply`` - which
    says nothing about *why*, and nothing about the actual, very common
    cause on Colab/Kaggle: a later ``pip install`` (typically pulled in by
    ``accelerate`` or ``bitsandbytes``) silently replacing the platform's
    preinstalled CUDA-enabled torch with a CPU-only wheel. Checking
    ``torch.cuda.is_available()`` first turns a multi-minute failure (after
    downloading and tokenizing everything) into an immediate, specific one.
    """
    import torch

    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested device={device!r} for {model_id}, but "
            f"torch.cuda.is_available() is False (torch {torch.__version__}). "
            "This usually means a `pip install` after the notebook started "
            "replaced the platform's preinstalled CUDA-enabled torch with a "
            "CPU-only build (a common Colab/Kaggle gotcha when installing "
            "accelerate/bitsandbytes) - check `torch.__version__` for a "
            "'+cpu' suffix. Fix: reinstall torch from the CUDA wheel index "
            "matching this machine's CUDA version (check `!nvidia-smi`), "
            "e.g. `pip install --index-url "
            "https://download.pytorch.org/whl/cu121 torch --force-reinstall`, "
            "then verify with `torch.cuda.is_available()` BEFORE re-running "
            "anything expensive."
        )
    model.to(device)


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
                _move_to_device(self._model, self.device, self.model_id)
        return self._tokenizer, self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        progress_every_batches: int = 200,
    ) -> np.ndarray:
        """Encode every text, batched.

        Writes each batch straight into a preallocated output array instead
        of collecting per-batch arrays in a list and ``np.vstack``-ing them
        at the end. For a small test corpus the difference is invisible; for
        the real ~4.3M-chunk corpus at MedCPT's 768 dims, the vector matrix
        alone is ~13 GB in float32, and the list-then-vstack approach holds
        *two* copies (the growing list plus the freshly concatenated array)
        at its peak - on the order of 26 GB, comfortably past a 16 GB
        laptop. Preallocating keeps peak memory to roughly one matrix.

        ``on_progress(batches_done, batches_total, elapsed_seconds)`` is
        called every ``progress_every_batches`` batches (and always on the
        last one) - encoding hundreds of thousands of batches on CPU can run
        for hours with nothing else to show it is still working.
        """
        import torch

        tokenizer, model = self._ensure()
        n = len(texts)
        if n == 0:
            return np.zeros((0, 768), dtype=np.float32)

        n_batches = (n + self.batch_size - 1) // self.batch_size
        vectors: Optional[np.ndarray] = None
        start_time = time.monotonic()

        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, n, self.batch_size), 1):
                batch = list(texts[start:start + self.batch_size])
                encoded = tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=self.max_tokens, return_tensors="pt",
                )
                if self.device:
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                # MedCPT reads the [CLS] representation, per its model card.
                hidden = model(**encoded).last_hidden_state[:, 0, :]
                batch_vectors = hidden.cpu().numpy().astype(np.float32)

                if vectors is None:
                    # Dimension is only known once the model has actually
                    # run, so the array is allocated after the first batch
                    # rather than guessed up front.
                    vectors = np.empty((n, batch_vectors.shape[1]), dtype=np.float32)
                vectors[start:start + len(batch)] = batch_vectors

                if on_progress is not None and (
                    batch_index % progress_every_batches == 0
                    or batch_index == n_batches
                ):
                    on_progress(batch_index, n_batches, time.monotonic() - start_time)

        assert vectors is not None  # n > 0 guarantees at least one batch ran
        return vectors


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
                _move_to_device(self._model, self.device, self.model_id)
        return self._tokenizer, self._model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        import torch

        tokenizer, model = self._ensure()
        scores: list[np.ndarray] = []
        with torch.inference_mode():
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
