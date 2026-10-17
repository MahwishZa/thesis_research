"""Exact inner-product search over the embeddings of one corpus, block by block.

The authors build a ``faiss.IndexFlatIP`` that holds a whole corpus in memory. This does the same arithmetic, the exact
inner product of every passage vector with every query vector, but reads the vectors from a memory-mapped file one block
at a time, so a corpus larger than the memory of the laptop can be searched. Scores are 32-bit. Results are best first;
equal scores are ordered by the lower row number, except that which of several equal scores is cut off at the edge of
the list is decided by the selection inside a block (the same input always gives the same output).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pool import check_top_k


class ExactIndex:
    """Searches a 2-D array of passage vectors (an ``np.memmap`` is fine) for the best inner products."""

    def __init__(self, vectors: Any, *, block_rows: int = 32768, query_chunk: int = 512) -> None:
        if getattr(vectors, "ndim", None) != 2:
            raise ValueError("the vectors must be a 2-D array")
        for name, value in (("block_rows", block_rows), ("query_chunk", query_chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.vectors = vectors
        self.block_rows = block_rows
        self.query_chunk = query_chunk

    @property
    def rows(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, queries: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Row numbers and scores of the ``k`` best passages for each query, both of shape (queries, min(k, rows))."""
        check_top_k(k)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"the queries must have shape (n, {self.dim}), got {queries.shape}")
        if not np.isfinite(queries).all():
            raise ValueError("a query vector holds a value that is not a finite number")
        keep = min(k, self.rows)
        indices = np.empty((len(queries), keep), dtype=np.int64)
        scores = np.empty((len(queries), keep), dtype=np.float32)
        if keep == 0:
            return indices, scores
        for first in range(0, len(queries), self.query_chunk):
            chunk = queries[first : first + self.query_chunk]
            best_scores = best_rows = None
            for start in range(0, self.rows, self.block_rows):
                block = np.asarray(self.vectors[start : start + self.block_rows], dtype=np.float32)
                block_scores = block @ chunk.T  # (rows in the block, queries in the chunk)
                take = min(keep, block_scores.shape[0])
                if take < block_scores.shape[0]:
                    chosen = np.argpartition(-block_scores, take - 1, axis=0)[:take]
                else:
                    chosen = np.broadcast_to(np.arange(take)[:, None], block_scores.shape)
                candidate_scores = np.take_along_axis(block_scores, chosen, axis=0)
                candidate_rows = chosen.astype(np.int64) + start
                if best_scores is not None:
                    candidate_scores = np.concatenate([best_scores, candidate_scores], axis=0)
                    candidate_rows = np.concatenate([best_rows, candidate_rows], axis=0)
                order = np.lexsort((candidate_rows, -candidate_scores), axis=0)[:keep]
                best_scores = np.take_along_axis(candidate_scores, order, axis=0)
                best_rows = np.take_along_axis(candidate_rows, order, axis=0)
            indices[first : first + len(chunk)] = best_rows.T
            scores[first : first + len(chunk)] = best_scores.T
        return indices, scores
