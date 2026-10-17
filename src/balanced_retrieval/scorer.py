"""The MedCPT cross-encoder, which scores a query against each pooled passage (paper p.5; the authors' ``rerank.py``).

Each pair [query, passage] goes through the tokenizer with truncation at 512 tokens and through the model in evaluation
mode without gradients; the single logit of each pair is its score. The authors score a whole pool in one call; this does
it in batches, which gives the same scores up to rounding because the padding is masked.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

CROSS_ENCODER_NAME = "ncbi/MedCPT-Cross-Encoder"
MAX_LENGTH = 512


class CrossEncoderScorer:
    """``scorer(query_text, candidates)`` returns one float per candidate, in the order of the candidates."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = MAX_LENGTH,
        no_grad: Callable[[], Any] | None = None,
        name: str = CROSS_ENCODER_NAME,
        revision: str | None = None,
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

    @classmethod
    def from_pretrained(
        cls,
        name: str = CROSS_ENCODER_NAME,
        *,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> CrossEncoderScorer:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, local_files_only=local_files_only)
        model.eval()
        model.to(device)
        resolved = getattr(model.config, "_commit_hash", None) or revision
        return cls(tokenizer, model, device=device, batch_size=batch_size, no_grad=torch.no_grad, name=name, revision=resolved)

    @property
    def info(self) -> dict[str, Any]:
        return {
            "cross_encoder": self.name,
            "revision": self.revision,
            "max_length": self.max_length,
            "device": self.device,
            "batch_size": self.batch_size,
        }

    def __call__(self, query: str, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("the query text is empty or not a string")
        scores: list[float] = []
        for start in range(0, len(candidates), self.batch_size):
            pairs = [[query, candidate["text"]] for candidate in candidates[start : start + self.batch_size]]
            encoded = self.tokenizer(pairs, truncation=True, padding=True, return_tensors="pt", max_length=self.max_length)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._no_grad():
                logits = self.model(**encoded).logits.squeeze(dim=1)
            scores.extend(float(value) for value in logits.detach().cpu().numpy())
        return scores
