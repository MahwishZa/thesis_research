"""The loop that writes a rationale for every question that has none yet and saves each one at once."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .generator import ContextOverflowError, GenerationResult
from .prompt import build_rationale_prompt
from .store import (
    Question,
    QuestionChanged,
    RationaleRecord,
    RationaleStore,
    question_hash,
)

#: A run stops after this many failures in a row: something is wrong with the model or the machine, not with one question.
MAX_CONSECUTIVE_ERRORS = 5


class RunAborted(RuntimeError):
    """Too many failures in a row."""


class Generator(Protocol):
    config: dict

    def generate(self, prompt: str) -> GenerationResult: ...


@dataclass
class RunSummary:
    total: int = 0
    already_done: int = 0
    written: int = 0
    cut_off: int = 0  # rationales that filled the memory window
    errors: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0


def generate_rationales(
    questions: Sequence[Question],
    generator: Generator,
    store: RationaleStore,
    config: dict,
    relevant: Sequence[str],
    *,
    progress: Callable[[int, int, RationaleRecord], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunSummary:
    """Write the rationale of every question that has no finished one, saving each as soon as it exists.

    A question that failed before is tried again. A saved rationale whose question text has changed stops the run.
    A failure is saved as a failure, with the reason; the question text is never used in place of a rationale.
    """
    store.bind(config, relevant)
    saved = store.load(repair=True)
    summary = RunSummary(total=len(questions))
    started = clock()
    in_a_row = 0
    for position, question in enumerate(questions, start=1):
        digest = question_hash(question.text)
        earlier = saved.get(question.id)
        if earlier is not None and earlier.status == "ok":
            if earlier.question_sha256 != digest:
                raise QuestionChanged(f"{question.id}: the saved rationale was written for another text than this question")
            summary.already_done += 1
            continue
        begun = clock()
        about_this_question_only = False
        try:
            result = generator.generate(build_rationale_prompt(question.text))
            if not result.text.strip():
                raise ValueError("the model wrote an empty rationale")
            record = RationaleRecord(
                id=question.id, question_sha256=digest, status="ok", rationale=result.text, finish_reason=result.finish_reason,
                prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens, seconds=round(clock() - begun, 3),
            )
        except Exception as exc:  # noqa: BLE001 - saved per question; the run goes on
            message = f"{type(exc).__name__}: {exc}"
            about_this_question_only = isinstance(exc, ContextOverflowError)
            record = RationaleRecord(id=question.id, question_sha256=digest, status="error", seconds=round(clock() - begun, 3), error=message)
        store.append(record)
        if record.status == "ok":
            in_a_row = 0
            summary.written += 1
            summary.cut_off += int(record.finish_reason == "length")
        else:
            in_a_row += 0 if about_this_question_only else 1  # a prompt that is too long says nothing about the model or the machine
            summary.errors.append((question.id, record.error or ""))
        if progress is not None:
            progress(position, len(questions), record)
        if in_a_row >= MAX_CONSECUTIVE_ERRORS:
            summary.seconds = clock() - started
            raise RunAborted(f"{MAX_CONSECUTIVE_ERRORS} failures in a row; the last was: {summary.errors[-1][1]}")
    summary.seconds = clock() - started
    return summary
