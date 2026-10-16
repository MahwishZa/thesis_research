"""The language model that writes the rationales, behind one small interface.

The paper uses Llama-3-8B-Instruct with greedy decoding (temperature 0), served with vLLM (A.3, p.13). Here the same
model runs from a GGUF file with llama.cpp, so the weights are 4-bit unless a larger file is given; that is a declared
deviation. Everything that can change the text of a rationale is recorded in ``config`` and bound to the output
folder by the store, so two files made with different settings are never mixed.

Two ways to build the prompt are offered:

* ``llama-3``: this module formats one user turn exactly as the official Llama-3-Instruct chat template does and
  hands llama.cpp the token list, so the start-of-text token appears once. The file's own template is rendered and
  compared when the library allows it.
* ``gguf``: the chat template stored in the GGUF file is applied by llama-cpp-python. This is for other models.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompt import RATIONALE_PROMPT, build_rationale_prompt

CHAT_FORMATS = ("llama-3", "gguf")

# The special tokens of the Llama-3 chat format.
BOS = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"
USER_HEAD = START_HEADER + "user" + END_HEADER + "\n\n"
ASSISTANT_HEAD = START_HEADER + "assistant" + END_HEADER + "\n\n"
LLAMA3_TEMPLATE_ID = "llama-3-instruct: one user turn, no system prompt"
GGUF_TEMPLATE_FORMAT = "chat_template.default"  # llama-cpp-python's name for the template stored in the file

# Greedy decoding written out so that no library default can change it. With temperature 0 llama.cpp takes the
# most likely token; top_k 1 gives the same result under any sampler order. The penalties are neutral values.
SAMPLING = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

#: Fields of ``config`` that can change what a rationale says. A folder whose recorded values differ is refused.
#: The thread counts and the seed are recorded but not listed: greedy decoding does not use the seed, and the number of
#: threads is not expected to change the numbers (llama.cpp gives each output value to one thread). The real-file check
#: in tests/integration/test_real_models.py compares one thread with the default on the real model.
RESULT_RELEVANT = (
    "model_sha256",
    "chat_format",
    "chat_template",
    "prompt_sha256",
    "max_new_tokens",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_gpu_layers",
    "llama_cpp_python",
    *SAMPLING,
)

# Room kept for a chat template's own tokens when the prompt length is estimated before generation (gguf format).
TEMPLATE_MARGIN_TOKENS = 64


class GenerationError(RuntimeError):
    """The model could not produce a rationale."""


class ContextOverflowError(GenerationError):
    """Prompt and the allowed new tokens do not fit into the context window."""


class ModelMismatchError(GenerationError):
    """The model file is not the one expected."""


class TemplateMismatchError(GenerationError):
    """The file's own chat template renders a different prompt than the Llama-3 format used here."""


@dataclass(frozen=True)
class GenerationSettings:
    """Settings of one generation run. ``model_name`` is a label for people and is not bound to the output."""

    model_name: str = ""
    chat_format: str = "llama-3"
    max_new_tokens: int = 512
    n_ctx: int = 8192  # Llama-3's own window; a question of realistic length uses under a tenth of it
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int | None = None
    n_gpu_layers: int = 0
    seed: int = 42
    allow_template_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.chat_format not in CHAT_FORMATS:
            raise ValueError(f"chat_format must be one of {', '.join(CHAT_FORMATS)}, got {self.chat_format!r}")
        for name in ("max_new_tokens", "n_ctx", "n_batch", "n_ubatch"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_ctx <= self.max_new_tokens:
            raise ValueError(f"n_ctx ({self.n_ctx}) must be larger than max_new_tokens ({self.max_new_tokens})")
        if self.n_threads is not None and (not isinstance(self.n_threads, int) or self.n_threads < 1):
            raise ValueError("n_threads must be a positive integer or None")
        if not isinstance(self.n_gpu_layers, int) or self.n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be an integer of at least -1")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str  # "stop": the model ended its turn; "length": cut off at max_new_tokens
    prompt_tokens: int
    completion_tokens: int


def file_sha256(
    path: str | Path,
    *,
    chunk_size: int = 1 << 20,
    progress: Callable[[int, int], None] | None = None,
    report_every: int = 1 << 30,
) -> str:
    """SHA-256 of a file, read in chunks. ``progress(done_bytes, total_bytes)`` is called about once per GiB."""
    digest = hashlib.sha256()
    total = Path(path).stat().st_size
    done = 0
    next_report = report_every
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            done += len(chunk)
            if progress is not None and done >= next_report:
                progress(done, total)
                next_report += report_every
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def llama3_prompt_text(user_message: str) -> str:
    """One user turn as the official Llama-3-Instruct chat template renders it (the content is trimmed)."""
    return BOS + USER_HEAD + user_message.strip() + EOT + ASSISTANT_HEAD


def llama3_prompt_tokens(llm: Any, user_message: str) -> list[int]:
    """Token ids of the Llama-3 prompt: the special markers are parsed, the message text is not.

    A message that happens to contain a marker string such as "<|eot_id|>" therefore stays plain text and cannot
    end the turn early. The start-of-text token appears exactly once, and llama.cpp adds none of its own because
    the list is passed as it is.
    """
    head = llm.tokenize((BOS + USER_HEAD).encode("utf-8"), add_bos=False, special=True)
    body = llm.tokenize(user_message.strip().encode("utf-8"), add_bos=False, special=False)
    tail = llm.tokenize((EOT + ASSISTANT_HEAD).encode("utf-8"), add_bos=False, special=True)
    return list(head) + list(body) + list(tail)


def check_llama3_vocabulary(llm: Any) -> None:
    """Each special marker must be a single token, and the start-of-text marker must be the model's BOS token."""
    for marker in (BOS, START_HEADER, END_HEADER, EOT):
        ids = llm.tokenize(marker.encode("utf-8"), add_bos=False, special=True)
        if len(ids) != 1:
            raise ModelMismatchError(
                f"{marker} is not one token in this model's vocabulary; the file does not look like Llama-3-Instruct"
            )
        if marker == BOS and ids[0] != llm.token_bos():
            raise ModelMismatchError(f"{BOS} is not this model's start-of-text token")


def gguf_template_renderer(llm: Any) -> Callable[[str], str] | None:
    """A function that renders one user turn with the chat template stored in the GGUF file, or None if there is none."""
    template = (getattr(llm, "metadata", None) or {}).get("tokenizer.chat_template")
    if not template:
        return None
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    bos = llm.detokenize([llm.token_bos()], special=True).decode("utf-8", errors="replace")
    eos = llm.detokenize([llm.token_eos()], special=True).decode("utf-8", errors="replace")
    formatter = Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)

    def render(user_message: str) -> str:
        return formatter(messages=[{"role": "user", "content": user_message}]).prompt

    return render


def _import_llama() -> Callable[..., Any]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise GenerationError(
            "llama-cpp-python is not installed; install it (pip install llama-cpp-python) or use a machine that has it"
        ) from exc
    return Llama


def _library_version() -> str:
    try:
        return importlib.metadata.version("llama_cpp_python")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class LlamaCppGenerator:
    """Writes one rationale per call from a GGUF model with llama.cpp. Call ``close`` when done."""

    def __init__(
        self,
        model_path: str | Path,
        settings: GenerationSettings | None = None,
        *,
        expected_sha256: str | None = None,
        llama_factory: Callable[..., Any] | None = None,
        template_renderer: Callable[[str], str] | None = None,
        library_version: str | None = None,
        progress: Callable[[int, int], None] | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings or GenerationSettings()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        sha = file_sha256(path, progress=progress)
        if expected_sha256 and expected_sha256.strip().lower() != sha:
            raise ModelMismatchError(f"{path.name} has SHA-256 {sha}, expected {expected_sha256.strip().lower()}")
        factory = llama_factory or _import_llama()
        s = self.settings
        options: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": s.n_ctx,
            "n_batch": s.n_batch,
            "n_ubatch": s.n_ubatch,
            "n_threads": s.n_threads,
            "n_gpu_layers": s.n_gpu_layers,
            "seed": s.seed,
            "verbose": verbose,
        }
        if s.chat_format == "gguf":
            # The template stored in the file, rendered by Jinja2. Without this the library may swap in one of its own
            # built-in formats for a template it believes it recognises.
            options["chat_format"] = GGUF_TEMPLATE_FORMAT
        self._llm = factory(**options)
        metadata = getattr(self._llm, "metadata", None) or {}
        gguf_template = metadata.get("tokenizer.chat_template")
        if s.chat_format == "llama-3":
            check_llama3_vocabulary(self._llm)
            template_id = LLAMA3_TEMPLATE_ID
            template_check = self._verify_llama3_template(template_renderer)
        else:
            if not gguf_template:
                raise GenerationError(
                    f"{path.name} has no chat template, so the 'gguf' format cannot be used; "
                    "llama-cpp-python would fall back to a Llama-2 format"
                )
            template_id = "gguf-template-sha256:" + text_sha256(gguf_template)
            template_check = "not applicable: the file's own template is applied"
        params = getattr(self._llm, "context_params", None)
        self.config: dict[str, Any] = {
            "model_name": s.model_name,
            "model_file": path.name,
            "model_sha256": sha,
            "chat_format": s.chat_format,
            "chat_format_resolved": str(getattr(self._llm, "chat_format", None)),
            "chat_template": template_id,
            "template_check": template_check,
            "system_prompt": None,
            "prompt_sha256": text_sha256(RATIONALE_PROMPT),
            "max_new_tokens": s.max_new_tokens,
            "n_ctx": s.n_ctx,
            # What the library actually uses; it lowers n_batch to n_ctx and n_ubatch to n_batch when they are larger.
            "n_batch": getattr(self._llm, "n_batch", None) or s.n_batch,
            "n_ubatch": getattr(params, "n_ubatch", None) or s.n_ubatch,
            "n_threads": getattr(self._llm, "n_threads", None) or s.n_threads,
            "n_threads_batch": getattr(self._llm, "n_threads_batch", None),
            "n_gpu_layers": s.n_gpu_layers,
            "seed": s.seed,
            "decoding": "greedy",
            "fresh_cache_per_question": True,
            "flash_attn_type": getattr(params, "flash_attn_type", None),
            "type_k": getattr(params, "type_k", None),
            "type_v": getattr(params, "type_v", None),
            "llama_cpp_python": library_version or _library_version(),
            **SAMPLING,
        }

    def _verify_llama3_template(self, renderer: Callable[[str], str] | None) -> str:
        try:
            render = renderer or gguf_template_renderer(self._llm)
            if render is None:
                return "no chat template in the file"
            sample = build_rationale_prompt("Which of the following is correct? A) one B) two C) three D) four")
            theirs = render(sample)
        except Exception as exc:  # noqa: BLE001 - the check itself failed, which says nothing about the template
            return f"not checkable: {type(exc).__name__}: {exc}"
        if theirs != llama3_prompt_text(sample):
            if self.settings.allow_template_mismatch:
                return "MISMATCH (allowed by setting)"
            raise TemplateMismatchError(
                "the chat template stored in the model file renders a different prompt than the Llama-3 format "
                "used here (for example a Llama-3.1 file); use chat_format 'gguf' or allow the mismatch knowingly"
            )
        return "matches the file's own chat template"

    def generate(self, user_message: str) -> GenerationResult:
        """Write the rationale for one user message. Every call starts from an empty cache."""
        self._llm.reset()  # otherwise llama.cpp reuses the cache of the previous prompt and results depend on order
        if self.settings.chat_format == "llama-3":
            return self._generate_llama3(user_message)
        return self._generate_gguf(user_message)

    def _generate_llama3(self, user_message: str) -> GenerationResult:
        s = self.settings
        tokens = llama3_prompt_tokens(self._llm, user_message)
        bos = self._llm.token_bos()
        if tokens[0] != bos or tokens.count(bos) != 1:
            raise GenerationError("the prompt must start with exactly one start-of-text token")
        if len(tokens) + s.max_new_tokens > s.n_ctx:
            raise ContextOverflowError(
                f"prompt of {len(tokens)} tokens plus {s.max_new_tokens} new tokens does not fit into a context of {s.n_ctx}"
            )
        out = self._llm.create_completion(prompt=tokens, max_tokens=s.max_new_tokens, seed=s.seed, **SAMPLING)
        choice = out["choices"][0]
        return self._result(choice["text"], choice["finish_reason"], out.get("usage"), len(tokens))

    def _generate_gguf(self, user_message: str) -> GenerationResult:
        s = self.settings
        text = user_message.strip()
        estimate = len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)) + TEMPLATE_MARGIN_TOKENS
        if estimate + s.max_new_tokens > s.n_ctx:
            raise ContextOverflowError(
                f"prompt of about {estimate} tokens plus {s.max_new_tokens} new tokens does not fit into a context of {s.n_ctx}"
            )
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": text}], max_tokens=s.max_new_tokens, seed=s.seed, **SAMPLING
        )
        choice = out["choices"][0]
        usage = out.get("usage") or {}
        if usage.get("prompt_tokens", 0) + s.max_new_tokens > s.n_ctx:
            # llama-cpp-python lowers max_tokens silently when the window is too small; that is not this setting.
            raise ContextOverflowError(
                f"prompt of {usage['prompt_tokens']} tokens plus {s.max_new_tokens} new tokens does not fit into a context of {s.n_ctx}"
            )
        return self._result(choice["message"]["content"], choice["finish_reason"], usage, 0)

    @staticmethod
    def _result(text: str | None, finish_reason: str, usage: Mapping[str, int] | None, prompt_tokens: int) -> GenerationResult:
        if finish_reason not in ("stop", "length"):
            raise GenerationError(f"unexpected finish reason {finish_reason!r}")
        usage = usage or {}
        return GenerationResult(
            text=text or "",
            finish_reason=finish_reason,
            prompt_tokens=int(usage.get("prompt_tokens", prompt_tokens)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if close is not None:
            close()
