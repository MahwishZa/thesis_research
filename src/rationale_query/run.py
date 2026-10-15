"""The loop that turns questions into rationale records."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .generator import RESULT_RELEVANT, GenerationResult
from .prompt import build_rationale_prompt
from .store import (
    Question,
    QuestionChanged,
    RationaleRecord,
    RationaleStore,
    question_hash,
)

#: A run stops after this many failed questions in a row: the cause is then in the setup, not in one question.
MAX_CONSECUTIVE_ERRORS = 5


class Generator(Protocol):
    config: dict

    def generate(self, user_message: str) -> GenerationResult: ...


class RunAborted(RuntimeError):
    """Too many questions in a row failed."""


@dataclass(frozen=True)
class Plan:
    pending: list[Question]
    already_done: int


@dataclass
class RunSummary:
    total: int = 0
    already_done: int = 0
    written: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    removed_cut_off_line: bool = False  # the file ended in a half-written line of an interrupted run, now removed


def plan(questions: Sequence[Question], existing: Mapping[str, RationaleRecord], directory: str | Path = "") -> Plan:
    """Which questions still need a rationale. Used before the model is loaded and again by the run itself.

    A question with a successful record is done; one with an error record, or no record, is pending. A question whose
    text differs from the text its record was written for raises ``QuestionChanged`` before anything is generated,
    so a run never spends hours on a folder that does not belong to this questions file.
    """
    pending: list[Question] = []
    done = 0
    for question in questions:
        record = existing.get(question.id)
        if record is not None and record.question_sha256 != question_hash(question.text):
            raise QuestionChanged(f"{question.id} is in {directory} with a different question text; use a new output folder")
        if record is not None and record.status == "ok":
            done += 1
        else:
            pending.append(question)
    return Plan(pending, done)


def generate_rationales(
    questions: Sequence[Question],
    generator: Generator,
    store: RationaleStore,
    relevant: Sequence[str] = RESULT_RELEVANT,
    *,
    progress: Callable[[int, int, RationaleRecord], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunSummary:
    """Write a rationale for every question that has none yet, appending one record per question.

    A question with a successful record is skipped (and must still have the same text). A question that failed
    before is tried again. A failure, including an empty answer, is recorded as an error and never replaced by the
    question, because that would change the method. Returns what was done; the caller decides what a non-empty
    ``errors`` list means.
    """
    store.bind(generator.config, relevant)
    todo = plan(questions, store.load(repair=True), store.directory)
    summary = RunSummary(
        total=len(questions), already_done=todo.already_done, removed_cut_off_line=store.recovered_partial_line
    )
    place = {question.id: number for number, question in enumerate(questions, start=1)}
    started = clock()
    consecutive = 0
    for question in todo.pending:
        record = _write_one(question, question_hash(question.text), generator, clock)
        store.append(record)
        summary.written += 1
        if record.status == "error":
            summary.errors.append((question.id, record.error or "error"))
            consecutive += 1
        else:
            consecutive = 0
        if progress is not None:
            progress(place[question.id], len(questions), record)
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            summary.seconds = clock() - started
            raise RunAborted(
                f"{consecutive} questions in a row failed (last: {record.error}). "
                "Fix the cause and run the same command again."
            )
    summary.seconds = clock() - started
    return summary


def _write_one(question: Question, digest: str, generator: Generator, clock: Callable[[], float]) -> RationaleRecord:
    begun = clock()
    try:
        result = generator.generate(build_rationale_prompt(question.text))
    except Exception as exc:  # noqa: BLE001 - recorded per question; the run goes on
        return RationaleRecord(
            id=question.id,
            question_sha256=digest,
            status="error",
            seconds=round(clock() - begun, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
    seconds = round(clock() - begun, 3)
    if not result.text.strip():
        return RationaleRecord(
            id=question.id,
            question_sha256=digest,
            status="error",
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            seconds=seconds,
            error="EmptyRationale: the model returned no text (the question is never used instead)",
        )
    return RationaleRecord(
        id=question.id,
        question_sha256=digest,
        status="ok",
        rationale=result.text,
        finish_reason=result.finish_reason,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        seconds=seconds,
    )
