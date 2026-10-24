"""The language model that writes the rationale: Meerkat-7B (Kim et al., 2025) run with llama.cpp.

What the paper fixes, and what this file does about it:

* The paper's prompt is sent as one user message and nothing else is added: no system message (section 3.3 shows only
  the prompt text). Meerkat's own chat template, from its ``tokenizer_config.json``, turns one user message into
  ``USER: <message> ASSISTANT: ``; the tokenizer then adds the start token once.
* Decoding is greedy: the most likely next word every time, no randomness (the paper: temperature 0, appendix A.3).
* The paper gives no limit for the length of a rationale. This file sets none of its own: the model writes until it stops
  by itself. The only hard stop is the memory window. Meerkat-7B was trained with a maximum length of 2048 tokens
  (section 5.1 of the paper, and the Methods of Kim et al., 2025), so the window is 2048 tokens by default. A rationale that
  fills the window is reported as ``finish_reason == "length"``; it is kept, never shortened or replaced.

Not stated in the paper: whether the authors applied a chat format at all, and whether the box was sent as one message.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHAT_FORMAT = "meerkat"

#: Every sampling value is written out, because the library's own defaults are not greedy.
SAMPLING = {
    "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
}


class GenerationError(RuntimeError):
    """The model could not write a rationale."""


class ContextOverflowError(GenerationError):
    """The prompt alone does not fit in the memory window. It is never shortened."""


class ModelMismatchError(GenerationError):
    """The model file is not the one the researcher pinned."""


@dataclass(frozen=True)
class GenerationSettings:
    n_ctx: int = 2048
    n_threads: int | None = None
    n_gpu_layers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("n_ctx", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a whole number from 1, got {value!r}")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str  # "stop": the model ended its answer; "length": the memory window was full
    prompt_tokens: int
    completion_tokens: int


def render_user_message(prompt: str) -> str:
    """Meerkat's chat template for one user message and no system message: ``USER: <message> ASSISTANT: ``."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("the prompt must be non-empty text")
    return f"USER: {prompt} ASSISTANT: "


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _import_llama() -> Callable[..., Any]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError("the package llama-cpp-python is needed to run the model (pip install -e \".[rationale]\")") from exc
    return Llama


def _library_version() -> str:
    try:
        import llama_cpp
    except ImportError:  # pragma: no cover
        return "not installed"
    return getattr(llama_cpp, "__version__", "unknown")


class LlamaGenerator:
    """One question in, one rationale out. ``llama_factory`` lets a test put a stand-in in place of llama.cpp."""

    def __init__(
        self,
        model_path: str | Path,
        settings: GenerationSettings | None = None,
        *,
        expected_sha256: str | None = None,
        verbose: bool = False,
        llama_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings or GenerationSettings()
        path = Path(model_path)
        if not path.is_file():
            raise GenerationError(f"the model file {path} does not exist")
        sha = file_sha256(path)
        if expected_sha256 is not None and sha != expected_sha256.lower():
            raise ModelMismatchError(f"{path.name} has SHA-256 {sha}, not the pinned {expected_sha256}")
        s = self.settings
        self.llm = (llama_factory or _import_llama())(
            model_path=str(path), n_ctx=s.n_ctx, n_threads=s.n_threads, n_gpu_layers=s.n_gpu_layers, seed=s.seed,
            logits_all=False, verbose=verbose,
        )
        self.config: dict[str, Any] = {
            "model_file": path.name, "model_sha256": sha, "chat_format": CHAT_FORMAT, "n_ctx": s.n_ctx, "seed": s.seed,
            "sampling": dict(SAMPLING), "n_threads": s.n_threads, "n_gpu_layers": s.n_gpu_layers,
            "llama_cpp_python": _library_version(),
        }

    def prompt_tokens(self, prompt: str) -> list[int]:
        """The token list the model reads: the start token once, then the text of the chat template."""
        text = render_user_message(prompt)
        tokens = list(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=False))
        bos = self.llm.token_bos()
        if not tokens or tokens[0] != bos or bos in tokens[1:]:
            raise GenerationError("the start token must stand once, at the beginning of the prompt")
        return tokens

    def generate(self, prompt: str) -> GenerationResult:
        tokens = self.prompt_tokens(prompt)
        window = self.settings.n_ctx
        if len(tokens) >= window:
            raise ContextOverflowError(f"the prompt has {len(tokens)} tokens; the window holds {window}")
        self._empty_cache()
        out = self.llm.create_completion(prompt=tokens, max_tokens=window - len(tokens), stop=None, **SAMPLING)
        choice = out["choices"][0]
        text = choice["text"]
        if not isinstance(text, str):
            raise GenerationError("the model returned no text")
        reason = choice.get("finish_reason")
        if reason not in ("stop", "length"):
            raise GenerationError(f"unexpected finish reason {reason!r}")
        return GenerationResult(text=text, finish_reason=reason, prompt_tokens=len(tokens), completion_tokens=int(out["usage"]["completion_tokens"]))

    def _empty_cache(self) -> None:
        """Forget the previous question, so that no answer depends on what was asked before it."""
        self.llm.reset()
        context = getattr(self.llm, "_ctx", None)
        clear = getattr(context, "kv_cache_clear", None)
        if clear is not None:
            clear()

    def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if close is not None:
            close()
