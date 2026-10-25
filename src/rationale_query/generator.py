"""The language model that writes the rationale: Meerkat-7B (Kim et al., 2025) run with Hugging Face Transformers in BF16.

What the paper fixes, and what this file does about it:

* The paper ran the model in BF16 with greedy decoding (temperature 0) and vLLM, on one H100 (appendix A.3 of the paper;
  Methods of Kim et al., 2025). This file loads the weights in BF16 and decodes greedily. Only the engine differs: both
  vLLM and Transformers run the model with PyTorch.
* The model reads the paper's prompt text and nothing else: no system message and no chat wrapper such as Meerkat's
  ``USER: ... ASSISTANT:``, because the paper shows only the prompt text (section 3.3) and never mentions a chat format.
  The only thing added is the start-of-text token that the tokenizer adds, once, at the beginning. Meerkat's own chat
  format stays available as a named setting for a later, labelled check; it is off by default.
* The paper gives no limit for the length of a rationale. This file sets none of its own: the model writes until it stops
  by itself. The only hard stop is the memory window. Meerkat-7B was trained with a maximum length of 2048 tokens
  (section 5.1 of the paper, and the Methods of Kim et al., 2025), so the window is 2048 tokens by default. A rationale that
  fills the window is reported as ``finish_reason == "length"``; it is kept, never shortened or replaced.

A computer whose graphics card cannot hold the model runs the rest on the CPU and, if needed, from disk. That is slow but
does not change a single number.

Not stated in the paper: whether the authors applied a chat format at all, and whether the box was sent as one message.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: "none": the prompt text alone, as the paper shows it. "meerkat": Meerkat's own chat template for one user message.
CHAT_FORMATS = ("none", "meerkat")
DEFAULT_CHAT_FORMAT = "none"
DTYPE = "bfloat16"

#: Greedy decoding, every value written out because a library's own defaults need not be greedy.
DECODING = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}

#: The files of the model folder that decide what the model is; their hashes make up the model's identity.
IDENTITY_SUFFIXES = (".safetensors", ".model")
IDENTITY_NAMES = ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "model.safetensors.index.json")


class GenerationError(RuntimeError):
    """The model could not write a rationale."""


class ContextOverflowError(GenerationError):
    """The prompt alone does not fit in the memory window. It is never shortened."""


class ModelMismatchError(GenerationError):
    """The model folder is not the one the researcher pinned."""


@dataclass(frozen=True)
class GenerationSettings:
    n_ctx: int = 2048
    n_threads: int | None = None
    max_gpu_gib: float | None = None  # memory the graphics card may use; the rest of the model goes to the CPU
    max_cpu_gib: float | None = None  # memory the CPU side may use; anything above that goes to the disk
    offload_dir: str | None = None  # where parts of the model wait if they fit nowhere else
    chat_format: str = DEFAULT_CHAT_FORMAT

    def __post_init__(self) -> None:
        if isinstance(self.n_ctx, bool) or not isinstance(self.n_ctx, int) or self.n_ctx < 1:
            raise ValueError(f"n_ctx must be a whole number from 1, got {self.n_ctx!r}")
        if self.n_threads is not None and (isinstance(self.n_threads, bool) or not isinstance(self.n_threads, int) or self.n_threads < 1):
            raise ValueError(f"n_threads must be a whole number from 1, got {self.n_threads!r}")
        for name in ("max_gpu_gib", "max_cpu_gib"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if self.chat_format not in CHAT_FORMATS:
            raise ValueError(f"chat_format must be one of {CHAT_FORMATS}, got {self.chat_format!r}")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str  # "stop": the model ended its answer; "length": the memory window was full
    prompt_tokens: int
    completion_tokens: int


def render_model_input(prompt: str, chat_format: str = DEFAULT_CHAT_FORMAT) -> str:
    """The text the model reads. ``"none"``: the prompt itself. ``"meerkat"``: ``USER: <prompt> ASSISTANT: `` (Meerkat's template)."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("the prompt must be non-empty text")
    if chat_format == "none":
        return prompt
    if chat_format == "meerkat":
        return f"USER: {prompt} ASSISTANT: "
    raise ValueError(f"chat_format must be one of {CHAT_FORMATS}, got {chat_format!r}")


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def folder_sha256(folder: str | Path) -> str:
    """One hash for a model folder: the hashes of its weight, tokenizer and configuration files, in name order."""
    root = Path(folder)
    names = sorted(p.name for p in root.iterdir() if p.is_file() and (p.name in IDENTITY_NAMES or p.suffix in IDENTITY_SUFFIXES))
    if not any(name.endswith(".safetensors") for name in names):
        raise GenerationError(f"{root} holds no .safetensors weight files")
    lines = "".join(f"{name} {file_sha256(root / name)}\n" for name in names)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def load_with_transformers(folder: Path, settings: GenerationSettings) -> tuple[Any, Any]:
    """The model in BF16, spread over the graphics card, the CPU and (if needed) the disk, and its tokenizer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError("the packages torch, transformers and accelerate are needed (pip install -e \".[rationale]\")") from exc
    if settings.n_threads:
        torch.set_num_threads(settings.n_threads)
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "device_map": "auto"}
    memory: dict[Any, str] = {}
    if settings.max_gpu_gib is not None and torch.cuda.is_available():
        memory[0] = f"{settings.max_gpu_gib}GiB"
    if settings.max_cpu_gib is not None:
        memory["cpu"] = f"{settings.max_cpu_gib}GiB"
    if memory:
        kwargs["max_memory"] = memory
    if settings.offload_dir:
        Path(settings.offload_dir).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = settings.offload_dir
    model = AutoModelForCausalLM.from_pretrained(str(folder), **kwargs)
    model.eval()
    return model, tokenizer


class TransformersGenerator:
    """One question in, one rationale out. ``loader`` lets a test put a stand-in in place of the real model."""

    def __init__(
        self,
        model_dir: str | Path,
        settings: GenerationSettings | None = None,
        *,
        expected_sha256: str | None = None,
        loader: Callable[[Path, GenerationSettings], tuple[Any, Any]] | None = None,
    ) -> None:
        self.settings = settings or GenerationSettings()
        folder = Path(model_dir)
        if not (folder / "config.json").is_file():
            raise GenerationError(f"{folder} is not a model folder (no config.json)")
        sha = folder_sha256(folder)
        if expected_sha256 is not None and sha != expected_sha256.lower():
            raise ModelMismatchError(f"{folder.name} has hash {sha}, not the pinned {expected_sha256}")
        self.model, self.tokenizer = (loader or load_with_transformers)(folder, self.settings)
        s = self.settings
        self.config: dict[str, Any] = {
            "model_folder": folder.name, "model_sha256": sha, "dtype": DTYPE, "chat_format": s.chat_format, "n_ctx": s.n_ctx,
            "decoding": dict(DECODING), "n_threads": s.n_threads, "max_gpu_gib": s.max_gpu_gib, "max_cpu_gib": s.max_cpu_gib,
            "engine": _engine_versions(),
        }

    # The three steps below are the only ones that touch the model or its tokenizer; a test replaces them one by one.
    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text)["input_ids"])  # the tokenizer adds the start token

    def _complete(self, ids: list[int], max_new_tokens: int) -> list[int]:
        import torch

        device = getattr(self.model, "device", None)
        inputs = torch.tensor([ids], device=device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_new_tokens, temperature=None, top_p=None,
                top_k=None, pad_token_id=self.tokenizer.eos_token_id, **DECODING,
            )
        return [int(token) for token in out[0, len(ids):]]

    def _decode(self, ids: list[int]) -> str:
        return str(self.tokenizer.decode(ids, skip_special_tokens=True))

    def prompt_tokens(self, prompt: str) -> list[int]:
        """The token list the model reads: the start token once, then the text."""
        ids = self._encode(render_model_input(prompt, self.settings.chat_format))
        bos = self.tokenizer.bos_token_id
        if not ids or ids[0] != bos or bos in ids[1:]:
            raise GenerationError("the start token must stand once, at the beginning of the prompt")
        return ids

    def generate(self, prompt: str) -> GenerationResult:
        ids = self.prompt_tokens(prompt)
        window = self.settings.n_ctx
        if len(ids) >= window:
            raise ContextOverflowError(f"the prompt has {len(ids)} tokens; the window holds {window}")
        new = self._complete(ids, window - len(ids))
        stopped = bool(new) and new[-1] == self.tokenizer.eos_token_id
        return GenerationResult(
            text=self._decode(new), finish_reason="stop" if stopped else "length", prompt_tokens=len(ids), completion_tokens=len(new)
        )

    def close(self) -> None:
        self.model = None


def _engine_versions() -> str:
    try:
        import torch
        import transformers
    except ImportError:  # pragma: no cover
        return "not installed"
    return f"transformers {transformers.__version__}, torch {torch.__version__}"
