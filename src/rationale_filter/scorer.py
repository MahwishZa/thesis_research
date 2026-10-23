"""The log-probability of each token of a text under a Llama-3 prompt, with llama.cpp.

``LlamaLogprobScorer`` loads the model with ``logits_all`` so that the model's output at every position is kept; llama.cpp
then holds ``n_ctx`` times the vocabulary size of 32-bit numbers (about 2.1 GB at 4096 positions for Llama-3, whose vocabulary
has 128,256 tokens), which is why the scorer has its own, smaller context and is a separate object from the generator. The
weights are memory-mapped, so a generator and a scorer on the same file share them (not the GPU layers, if any).

The prompt is tokenised as in ``rationale_query.generator`` (the Llama-3 chat format, one start-of-text token, the message text
not parsed for markers). The text is tokenised on its own and follows the assistant header; the end-of-turn token is not
scored. A text is scored under two prompts in the same way, so any quirk of the tokenisation is the same on both sides.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rationale_query.generator import (
    ContextOverflowError,
    GenerationError,
    ModelMismatchError,
    check_llama3_vocabulary,
    file_sha256,
    llama3_prompt_tokens,
)


def _import_llama() -> Callable[..., Any]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise GenerationError("llama-cpp-python is not installed; install it (pip install llama-cpp-python)") from exc
    return Llama


class LlamaLogprobScorer:
    """Scores the tokens of a text, one call at a time. Call ``close`` when done."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        expected_sha256: str | None = None,
        llama_factory: Callable[..., Any] | None = None,
        verbose: bool = False,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        for name, value in (("n_ctx", n_ctx), ("n_batch", n_batch)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        sha = file_sha256(path)
        if expected_sha256 and expected_sha256.strip().lower() != sha:
            raise ModelMismatchError(f"{path.name} has SHA-256 {sha}, expected {expected_sha256.strip().lower()}")
        factory = llama_factory or _import_llama()
        self._llm = factory(
            model_path=str(path), n_ctx=n_ctx, n_batch=n_batch, n_threads=n_threads, n_gpu_layers=n_gpu_layers,
            logits_all=True, verbose=verbose,
        )
        check_llama3_vocabulary(self._llm)
        self.n_ctx = n_ctx
        self.config: dict[str, Any] = {
            "model_file": path.name, "model_sha256": sha, "n_ctx": n_ctx, "n_batch": n_batch, "n_gpu_layers": n_gpu_layers,
            "prompt_format": "llama-3 user turn, no system prompt", "scored": "the text's tokens, without the end-of-turn token",
        }

    def token_logprobs(self, user_message: str, completion: str) -> list[float]:
        """The log-probability of every token of ``completion`` given the Llama-3 prompt for ``user_message``."""
        import numpy as np

        if not isinstance(completion, str) or not completion.strip():
            raise ValueError("the text to score is empty")
        prompt = llama3_prompt_tokens(self._llm, user_message)
        text = list(self._llm.tokenize(completion.encode("utf-8"), add_bos=False, special=False))
        tokens = prompt + text
        if len(tokens) > self.n_ctx:
            raise ContextOverflowError(f"prompt of {len(prompt)} tokens plus {len(text)} tokens of text does not fit into {self.n_ctx}")
        self._llm.reset()
        self._llm.eval(tokens)
        scores = np.asarray(self._llm.scores[: len(tokens)], dtype=np.float64)
        result = []
        for position in range(len(prompt), len(tokens)):
            row = scores[position - 1]  # the output after token position - 1 predicts token position
            peak = row.max()
            result.append(float(row[tokens[position]] - peak - math.log(np.exp(row - peak).sum())))
        return result

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if close is not None:
            close()
