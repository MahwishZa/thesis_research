"""Shared helpers and stand-ins for the tests of src/rationale_query and src/balanced_retrieval. No model, no network,
no third-party package is needed except numpy for the encoder stand-ins, which a test skips without.

    FakeLlama          the part of llama_cpp.Llama that the generator uses, with the Llama-3 special tokens
    ScriptedGenerator  a generator whose answers are listed in the test
    FakeTokenizer / FakeModel / FakeTensor   a tokenizer and encoder over numpy arrays (words as tokens)
    FakePairTokenizer / FakeClassifier       a tokenizer of [query, passage] pairs and a cross-encoder over numpy arrays
    TempDirTestCase    a test case with a fresh temporary folder whose name contains spaces, as on the researcher's laptop
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from rationale_query.generator import (
    BOS,
    END_HEADER,
    EOT,
    SAMPLING,
    START_HEADER,
    GenerationResult,
    llama3_prompt_text,
)
from rationale_query.store import Question
from tests import ROOT

try:
    import numpy as np
except ImportError:  # the encoder stand-ins and their tests need numpy; everything else does not
    np = None

EXPERIMENT_FOLDER = ROOT / "experiments" / "01_rationale_writers"


def pinned_sha256(file_name):
    """The SHA-256 that the study settings pin for this model file name, or None if it is not listed."""
    try:
        writers = json.loads((EXPERIMENT_FOLDER / "settings.json").read_text(encoding="utf-8")).get("writers", [])
    except (OSError, ValueError):
        return None
    return next((w.get("sha256") for w in writers if w.get("model_file") == file_name), None)


def load_experiment_runner():
    """experiments/01_rationale_writers/run_experiment.py as a module (the folder name is not a name that can be imported)."""
    spec = importlib.util.spec_from_file_location("run_experiment", EXPERIMENT_FOLDER / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUESTIONS = [
    Question("q1", "A 72-year-old woman has progressive memory loss. Which is most likely? A) Delirium B) Alzheimer disease C) Hypothyroidism D) Depression"),
    Question("q2", "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"),
    Question("q3", "Which protein forms plaques in Alzheimer disease? A) Beta-amyloid B) Albumin C) Insulin D) Keratin"),
]


def question_lines(questions=QUESTIONS) -> str:
    return "".join(json.dumps({"id": q.id, "question": q.text}) + "\n" for q in questions)


class TempDirTestCase(unittest.TestCase):
    """``self.dir`` is a fresh folder with a space in its name; it is removed after the test."""

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory(prefix="rq test ")
        self.addCleanup(holder.cleanup)
        self.dir = Path(holder.name)

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path


def run_cli(argv: list[str], **factories: Any) -> tuple[int, str, str]:
    """Run ``rationale_query.cli.main`` in this process; returns (exit status, stdout, stderr)."""
    from rationale_query.cli import Streams, main

    out, err = io.StringIO(), io.StringIO()
    status = main(argv, streams=Streams(out, err), **factories)
    return status, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------------------------- generator stand-ins


class FakeLlama:
    """Stands in for ``llama_cpp.Llama``. Bytes are tokens 0-255; the four chat markers, BOS and the end token follow.

    ``outputs`` is a list of ``(text, finish_reason)`` taken one per generation call, the last one repeating.
    ``events`` lists every call in order, so a test can check that ``reset`` comes before a completion.
    """

    IDS: ClassVar[dict[str, int]] = {BOS: 300, START_HEADER: 301, END_HEADER: 302, EOT: 303}
    EOS_ID = 304

    def __init__(
        self,
        *,
        outputs: list[tuple[str, str]] | None = None,
        template: str | None = None,
        split_marker: str | None = None,
        bos_id: int = 300,
        add_bos_when_told_not_to: bool = False,
        prompt_tokens_in_usage: int | None = None,
        **options: Any,
    ) -> None:
        self.options = options
        self.outputs = outputs or [(" The answer is B.", "stop")]
        self.metadata = {"tokenizer.chat_template": template} if template is not None else {}
        self.chat_format = options.get("chat_format")
        self.n_threads = options.get("n_threads") or 4
        self.n_threads_batch = 8
        # Like the library, lower the batch sizes to what the context allows and expose the context parameters.
        self.n_batch = min(options.get("n_ctx", 512), options.get("n_batch", 512))
        self.context_params = SimpleNamespace(
            n_ubatch=min(self.n_batch, options.get("n_ubatch", 512)), flash_attn_type=0, type_k=1, type_v=1
        )
        self.split_marker = split_marker
        self.bos_id = bos_id
        self.add_bos_when_told_not_to = add_bos_when_told_not_to
        self.prompt_tokens_in_usage = prompt_tokens_in_usage
        self.events: list[str] = []
        self.tokenize_calls: list[tuple[str, bool, bool]] = []
        self.completion_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.closed = False
        self._served = 0

    # --- tokens
    def token_bos(self) -> int:
        return self.bos_id

    def token_eos(self) -> int:
        return self.EOS_ID

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        self.events.append("tokenize")
        self.tokenize_calls.append((text.decode("utf-8"), add_bos, special))
        if self.split_marker is not None and special and text.decode("utf-8") == self.split_marker:
            return [1, 2]  # a marker that is not a single token in this vocabulary
        tokens: list[int] = []
        if add_bos or self.add_bos_when_told_not_to:
            tokens.append(self.bos_id)
        remaining = text.decode("utf-8")
        while remaining:
            marker = next((m for m in self.IDS if special and remaining.startswith(m)), None)
            if marker is not None:
                tokens.append(self.IDS[marker])
                remaining = remaining[len(marker):]
            else:
                head = remaining[0]
                tokens.extend(head.encode("utf-8"))
                remaining = remaining[1:]
        return tokens

    def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
        names = {value: key for key, value in self.IDS.items()}
        names[self.EOS_ID] = EOT
        pieces = []
        for token in tokens:
            pieces.append(names[token].encode("utf-8") if token in names else bytes([token]))
        return b"".join(pieces)

    # --- generation
    def reset(self) -> None:
        self.events.append("reset")

    def _next_output(self) -> tuple[str, str]:
        output = self.outputs[min(self._served, len(self.outputs) - 1)]
        self._served += 1
        return output

    def create_completion(self, **kwargs: Any) -> dict:
        self.events.append("create_completion")
        self.completion_calls.append(kwargs)
        text, finish = self._next_output()
        prompt = kwargs["prompt"]
        used = self.prompt_tokens_in_usage if self.prompt_tokens_in_usage is not None else len(prompt)
        return {
            "choices": [{"text": text, "finish_reason": finish}],
            "usage": {"prompt_tokens": used, "completion_tokens": max(1, len(text.split())) if text else 0},
        }

    def create_chat_completion(self, **kwargs: Any) -> dict:
        self.events.append("create_chat_completion")
        self.chat_calls.append(kwargs)
        text, finish = self._next_output()
        used = self.prompt_tokens_in_usage if self.prompt_tokens_in_usage is not None else 50
        return {
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": used, "completion_tokens": max(1, len(text.split())) if text else 0},
        }

    def close(self) -> None:
        self.closed = True


#: The official Llama-3-Instruct chat template, as stored in the model files' tokenizer_config.
LLAMA3_JINJA = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def official_renderer(user_message: str) -> str:
    """What the official Llama-3-Instruct template renders for one user turn."""
    return llama3_prompt_text(user_message)


def llama_factory(**fake_options: Any) -> Callable[..., FakeLlama]:
    """A ``llama_factory`` for ``LlamaCppGenerator`` that builds FakeLlama and remembers the instance it made."""
    made: list[FakeLlama] = []

    def factory(**options: Any) -> FakeLlama:
        llm = FakeLlama(**fake_options, **options)
        made.append(llm)
        return llm

    factory.made = made  # type: ignore[attr-defined]
    return factory


class ScriptedGenerator:
    """A generator whose answers are listed: a ``GenerationResult``, an exception to raise, or a function of the prompt."""

    def __init__(self, answers: Any = None, config: dict | None = None) -> None:
        self.answers = answers if answers is not None else [ok("The answer is B.")]
        self.config = config if config is not None else default_config()
        self.prompts: list[str] = []
        self.closed = False

    def generate(self, user_message: str) -> GenerationResult:
        self.prompts.append(user_message)
        answer = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)] if isinstance(self.answers, list) else self.answers
        if callable(answer):
            answer = answer(user_message)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def close(self) -> None:
        self.closed = True


def ok(text: str, *, finish: str = "stop", prompt_tokens: int = 100, completion_tokens: int = 7) -> GenerationResult:
    return GenerationResult(text=text, finish_reason=finish, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def default_config(**changes: Any) -> dict:
    """A config with every result-relevant field of ``RESULT_RELEVANT``, so the store has something to compare."""
    from rationale_query.generator import RESULT_RELEVANT

    config = {name: "x" for name in RESULT_RELEVANT}
    config.update(SAMPLING)
    config.update(model_name="fake", seed=42, max_new_tokens=512, n_ctx=4096)
    config.update(changes)
    return config


# ---------------------------------------------------------------------------------------------- encoder stand-ins


class FakeTensor:
    """The few tensor methods the encoder uses, over a numpy array."""

    def __init__(self, array: Any) -> None:
        self.array = array
        self.moved_to: list[str] = []

    def to(self, device: str) -> FakeTensor:
        self.moved_to.append(device)
        return self

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> Any:
        return self.array

    def __getitem__(self, index: Any) -> FakeTensor:
        return FakeTensor(self.array[index])

    def squeeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.squeeze(self.array, axis=dim))


class FakeTokenizer:
    """Words are tokens; [CLS] and [SEP] are added. Records every call so a test can check the arguments."""

    PAD, CLS, SEP = 0, 1, 2

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def _word_id(word: str) -> int:
        return 10 + sum(ord(c) for c in word) % 1000

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, texts: list[str], **kwargs: Any) -> dict:
        self.calls.append({"texts": list(texts), **kwargs})
        limit = kwargs.get("max_length")
        rows = []
        for text in texts:
            ids = [self.CLS] + [self._word_id(w) for w in text.split()] + [self.SEP]
            if kwargs.get("truncation") and limit is not None and len(ids) > limit:
                ids = ids[: limit - 1] + [self.SEP]
            rows.append(ids)
        width = max(len(r) for r in rows) if kwargs.get("padding") else None
        if width is None and len({len(r) for r in rows}) > 1:
            raise ValueError("unequal lengths need padding")
        ids = np.array([r + [self.PAD] * (width - len(r)) for r in rows] if width else rows)
        mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows] if width else [[1] * len(r) for r in rows])
        return {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(mask)}


class FakeOutput:
    def __init__(self, last_hidden_state: FakeTensor) -> None:
        self.last_hidden_state = last_hidden_state


class FakeModel:
    """The first-token vector depends on all real tokens of its text (as a real encoder's does), not on padding."""

    DIM = 6

    def __init__(self) -> None:
        self.calls = 0
        self.last_inputs: tuple[FakeTensor, FakeTensor] | None = None

    def __call__(self, input_ids: FakeTensor, attention_mask: FakeTensor) -> FakeOutput:
        self.calls += 1
        self.last_inputs = (input_ids, attention_mask)
        ids, mask = input_ids.array, attention_mask.array
        batch, width = ids.shape
        hidden = np.zeros((batch, width, self.DIM), dtype=np.float64)
        for b in range(batch):
            real = ids[b][mask[b] == 1]
            first = np.array([real.sum(), real.size, real.max(), real.min(), real[1], real[-2]], dtype=np.float64)
            hidden[b, 0, :] = first
            for t in range(1, width):  # other positions differ, so taking the wrong position would show
                hidden[b, t, :] = -1.0 * (t + 1)
        return FakeOutput(FakeTensor(hidden))


class FakePairTokenizer:
    """Tokenizes [query, passage] pairs: words are tokens, and [CLS] query [SEP] passage [SEP] is cut at ``max_length``."""

    PAD, CLS, SEP = 0, 1, 2

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, pairs: list[list[str]], **kwargs: Any) -> dict:
        self.calls.append({"pairs": [list(p) for p in pairs], **kwargs})
        rows = []
        for query, passage in pairs:
            ids = [self.CLS] + [FakeTokenizer._word_id(w) for w in query.split()] + [self.SEP]
            ids += [FakeTokenizer._word_id(w) for w in passage.split()] + [self.SEP]
            limit = kwargs.get("max_length")
            if kwargs.get("truncation") and limit is not None and len(ids) > limit:
                ids = ids[: limit - 1] + [self.SEP]
            rows.append(ids)
        width = max(len(r) for r in rows)
        ids = np.array([r + [self.PAD] * (width - len(r)) for r in rows])
        mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(mask)}


class FakeClassifierOutput:
    def __init__(self, logits: FakeTensor) -> None:
        self.logits = logits


class FakeClassifier:
    """The score of a pair is the sum of its real token ids divided by 1000, so it does not depend on padding."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids: FakeTensor, attention_mask: FakeTensor) -> FakeClassifierOutput:
        self.calls += 1
        ids, mask = input_ids.array, attention_mask.array
        scores = np.array([[float(ids[b][mask[b] == 1].sum()) / 1000.0] for b in range(ids.shape[0])])
        return FakeClassifierOutput(FakeTensor(scores))


class RecordingNoGrad:
    """A ``torch.no_grad`` stand-in that counts how often it was entered and left."""

    def __init__(self) -> None:
        self.entered = 0
        self.left = 0

    def __call__(self) -> RecordingNoGrad:
        return self

    def __enter__(self) -> None:
        self.entered += 1

    def __exit__(self, *exc_info: object) -> None:
        self.left += 1
