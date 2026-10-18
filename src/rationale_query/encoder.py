"""The MedCPT query encoder, used exactly as in the authors' retriever (``retriever/query_encode.py``).

Each query goes through the tokenizer with truncation at 512 tokens, the special tokens included, through the
model in evaluation mode without gradients, and the vector of the first token (``last_hidden_state[:, 0, :]``) is
kept, as 32-bit numbers and without normalising. The authors encode one query at a time; ``batch_size=1`` does the
same, and a larger batch gives the same vectors up to rounding.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

ENCODER_NAME = "ncbi/MedCPT-Query-Encoder"
MAX_LENGTH = 512  # the authors' value for rationales; MedCPT's own example uses 64 for short queries
POOLING = "first token: last_hidden_state[:, 0, :]"


class QueryEncoder:
    """Texts in, one 768-number vector per text out. Build it with ``from_pretrained``, or inject a tokenizer and model."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        device: str = "cpu",
        batch_size: int = 1,
        max_length: int = MAX_LENGTH,
        no_grad: Callable[[], Any] | None = None,
        name: str = ENCODER_NAME,
        revision: str | None = None,
        compute_dtype: str = "float32",
    ) -> None:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_length, int) or max_length < 3:
            raise ValueError("max_length must be an integer of at least 3")
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._no_grad = no_grad or contextlib.nullcontext
        self.name = name
        self.revision = revision
        self.compute_dtype = compute_dtype

    @classmethod
    def from_pretrained(
        cls,
        name: str = ENCODER_NAME,
        *,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 1,
        local_files_only: bool = False,
        compute_dtype: str = "float32",
    ) -> QueryEncoder:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(name, revision=revision, local_files_only=local_files_only)
        if compute_dtype not in ("float32", "float16"):
            raise ValueError(f"compute_dtype must be float32 or float16, got {compute_dtype!r}")
        if compute_dtype == "float16":
            model.half()  # the arithmetic of the encoder only; the vectors that come out are still returned as float32
        model.eval()
        model.to(device)
        resolved = getattr(model.config, "_commit_hash", None) or revision
        return cls(tokenizer, model, device=device, batch_size=batch_size, no_grad=torch.no_grad, name=name, revision=resolved, compute_dtype=compute_dtype)

    @property
    def info(self) -> dict[str, Any]:
        info = {
            "encoder": self.name,
            "revision": self.revision,
            "max_length": self.max_length,
            "pooling": POOLING,
            "normalised": False,
            "dtype": "float32",
            "device": self.device,
            "batch_size": self.batch_size,
        }
        if self.compute_dtype != "float32":  # named only when it differs, so that every earlier description stays as it was
            info["compute_dtype"] = self.compute_dtype
        return info

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """Number of tokens each text would have, special tokens included, before truncation."""
        extra = self.tokenizer.num_special_tokens_to_add(pair=False)
        return [len(self.tokenizer.tokenize(text)) + extra for text in texts]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One float32 vector per text, in the order given. An empty text is an error."""
        texts = list(texts)
        if not texts:
            raise ValueError("no texts to encode")
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"text {position} is empty or not a string; an empty query is never encoded")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=self.max_length)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._no_grad():
                embeds = self.model(**encoded).last_hidden_state[:, 0, :]
            vectors.append(embeds.detach().cpu().numpy())
        return np.vstack(vectors).astype(np.float32, copy=False)
