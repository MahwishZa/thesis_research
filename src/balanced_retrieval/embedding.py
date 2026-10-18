"""The MedCPT article encoder and the resumable writer of a corpus's embeddings.

The article encoder is used exactly like the query encoder of Part 1: the tokenizer cuts at 512 tokens, the model runs in
evaluation mode without gradients, and the vector of the first token is kept, as 32-bit numbers and not normalised.
MedCPT's own example passes a title and an abstract as a pair; the passages here have no separate title, so a passage is
encoded as one text. How the authors embedded their snippets is not stated (an open item).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from rationale_query.encoder import QueryEncoder
from rationale_query.store import write_atomic

from .corpus import EMBEDDINGS_FILE, EMBEDDINGS_STATE, Corpus, CorpusError

ARTICLE_ENCODER_NAME = "ncbi/MedCPT-Article-Encoder"
DTYPES = ("float32", "float16")

#: Fields of the encoder's ``info`` that decide what a vector is; a half-written file is refused if any of them differs.
ENCODER_FIELDS = ("encoder", "revision", "max_length", "pooling", "normalised")


class ArticleEncoder(QueryEncoder):
    """The same code as the query encoder, loaded with the article model."""

    def __init__(self, tokenizer: Any, model: Any, **options: Any) -> None:
        options.setdefault("name", ARTICLE_ENCODER_NAME)
        super().__init__(tokenizer, model, **options)

    @classmethod
    def from_pretrained(cls, name: str = ARTICLE_ENCODER_NAME, **options: Any) -> ArticleEncoder:
        return super().from_pretrained(name, **options)


def embed_corpus(
    corpus: Corpus,
    encoder: Any,
    *,
    batch_size: int = 32,
    save_every: int = 2048,
    dtype: str = "float32",
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write ``embeddings.npy`` for a corpus, row by row in passage order; run it again to continue after an interruption.

    The file is created at its full size on the first batch, filled in place, and flushed together with a small state
    file every ``save_every`` rows. A run that finds a state written with other encoder settings or another dtype refuses
    to go on instead of mixing two kinds of vector. Returns the final state.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(DTYPES)}, got {dtype!r}")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(save_every, int) or save_every < batch_size:
        raise ValueError("save_every must be a whole number of at least batch_size")
    total = len(corpus)
    settings = {key: encoder.info.get(key) for key in ENCODER_FIELDS} | {"dtype": dtype}
    if encoder.info.get("compute_dtype", "float32") != "float32":
        settings["compute_dtype"] = encoder.info["compute_dtype"]  # named only when it differs: earlier embeddings stay valid
    state_path = corpus.directory / EMBEDDINGS_STATE
    matrix_path = corpus.directory / EMBEDDINGS_FILE
    state = corpus.embedding_state()
    if state is None:
        if matrix_path.exists():
            raise CorpusError(f"{matrix_path} exists without {EMBEDDINGS_STATE}, so it is not known how it was made; remove it")
        state = {"settings": settings, "rows": total, "rows_done": 0, "complete": False}
    else:
        if state["settings"] != settings:
            differing = {k: (state["settings"].get(k), settings.get(k)) for k in settings | state["settings"] if state["settings"].get(k) != settings.get(k)}
            raise CorpusError(f"{corpus.name} was embedded with other settings (recorded, now): {differing}")
        if state["rows"] != total:
            raise CorpusError(f"{corpus.name} has {total} passages but its embeddings were started for {state['rows']}")
        if state["complete"]:
            return state

    matrix = None
    done = int(state["rows_done"])
    since_save = 0
    batch: list[str] = []

    def flush_batch() -> None:
        nonlocal matrix, done, since_save
        vectors = encoder.encode(batch)
        if matrix is None:
            width = vectors.shape[1]
            if done == 0:
                matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.dtype(dtype), shape=(total, width))
                write_atomic(state_path, json.dumps(state, indent=2) + "\n")  # so a kill before the first save can be resumed
            else:
                matrix = np.load(matrix_path, mmap_mode="r+")
                if matrix.shape != (total, width) or matrix.dtype != np.dtype(dtype):
                    raise CorpusError(f"{matrix_path} has shape {matrix.shape} and type {matrix.dtype}, not what the state records")
        matrix[done : done + len(batch)] = vectors.astype(matrix.dtype, copy=False)
        done += len(batch)
        since_save += len(batch)
        batch.clear()
        if since_save >= save_every:
            matrix.flush()
            write_atomic(state_path, json.dumps(dict(state, rows_done=done), indent=2) + "\n")
            since_save = 0
        if progress is not None:
            progress(done, total)

    for record in corpus.iter_passages(done):
        batch.append(record["text"])
        if len(batch) == batch_size:
            flush_batch()
    if batch:
        flush_batch()
    if matrix is None:  # every row was done already, but the state was never marked complete
        matrix = np.load(matrix_path, mmap_mode="r+")
    if done != total:
        raise CorpusError(f"{corpus.name}: {done} vectors for {total} passages")
    matrix.flush()
    state = dict(state, rows_done=done, complete=True)
    write_atomic(state_path, json.dumps(state, indent=2) + "\n")
    return state


def compare_precision(texts: list[str], full: Any, half: Any, *, queries: int = 200, top: int = 10, clock: Callable[[], float] | None = None) -> dict[str, Any]:
    """How much half-precision arithmetic changes the vectors of the same texts, and how much faster it is.

    ``full`` and ``half`` are two encoders over the same model. Reported: the cosine between the two vectors of each text
    (mean and lowest), and, for the first ``queries`` texts used as queries against all texts by inner product, the share of
    the ``top`` nearest texts that both kinds of vector agree on and how often the single nearest text is the same.
    """
    import time

    clock = clock or time.monotonic
    started = clock()
    a = full.encode(texts).astype(np.float64)
    middle = clock()
    b = half.encode(texts).astype(np.float64)
    ended = clock()
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    count = min(queries, len(texts))
    top = min(top, len(texts))
    first = np.argsort(-(a[:count] @ a.T), axis=1, kind="stable")[:, :top]
    second = np.argsort(-(b[:count] @ b.T), axis=1, kind="stable")[:, :top]
    overlap = [len(set(x) & set(y)) / top for x, y in zip(first, second, strict=True)]
    return {
        "texts": len(texts),
        "queries": count,
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        f"top{top}_overlap": float(np.mean(overlap)),
        "nearest_same": float(np.mean(first[:, 0] == second[:, 0])),
        "seconds_full": middle - started,
        "seconds_half": ended - middle,
    }
