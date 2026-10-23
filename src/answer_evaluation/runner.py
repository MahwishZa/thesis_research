"""The loop that answers every question with every system and records the answers (decisions D12 to D16 and D24).

A system is an arm and a k: ``no_retrieval`` answers with the final option of the rationale of Part 1 (no new text is written;
k is 0), any other arm gets the snippets that ``evidence(question, arm)`` returns, at most k of them, and the model answers from
the answer prompt. Each answer is read strictly; an unreadable answer is recorded as wrong. The run can be stopped and started
again. On a test split it must be given a ledger: the ledger refuses a second scoring of the same system on the same test set,
and the claim is made before the first answer is written.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rationale_query.batch import MAX_CONSECUTIVE_ERRORS, RunAborted
from rationale_query.generator import GenerationResult, text_sha256
from rationale_query.store import QuestionChanged

from .ledger import Ledger
from .prompt import build_answer_prompt
from .reader import reading
from .records import AnswerRecord, AnswerStore, answer_id

NO_RETRIEVAL = "no_retrieval"


class Generator(Protocol):
    config: dict

    def generate(self, user_message: str) -> GenerationResult: ...


@dataclass(frozen=True)
class EvalQuestion:
    """A question with its options inline, its answer key and the rationale Part 1 wrote for it."""

    id: str
    text: str
    gold: str
    rationale: str


@dataclass(frozen=True)
class Arm:
    name: str
    k: int

    def __post_init__(self) -> None:
        if not self.name or "|" in self.name:
            raise ValueError(f"an arm name must be non-empty and without '|', got {self.name!r}")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 0:
            raise ValueError(f"k must be a whole number from 0, got {self.k!r}")
        if (self.name == NO_RETRIEVAL) != (self.k == 0):
            raise ValueError("k is 0 for no_retrieval and above 0 for every other arm")


@dataclass
class RunSummary:
    total: int = 0
    already_done: int = 0
    written: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0


def question_sha256(question: EvalQuestion) -> str:
    return text_sha256(f"{question.text.strip()}\x00{question.gold}")


def settings_sha256(config: dict, relevant: Sequence[str]) -> str:
    """A hash of the result-relevant settings, for the ledger."""
    return text_sha256(json.dumps({name: config.get(name) for name in relevant}, sort_keys=True, ensure_ascii=False))


def answer_one(
    question: EvalQuestion, arm: Arm, snippets: Sequence[str], generator: Generator | None, clock: Callable[[], float] = time.perf_counter
) -> AnswerRecord:
    """The record of one answer. A failure is recorded as an error, never as a wrong answer."""
    begun = clock()
    base = {
        "id": answer_id(arm.name, arm.k, question.id), "question_id": question.id, "arm": arm.name, "k": arm.k,
        "question_sha256": question_sha256(question),
    }
    try:
        if arm.name == NO_RETRIEVAL:
            text, finish, prompt_tokens, completion_tokens = question.rationale, None, None, None
        else:
            if len(snippets) > arm.k:
                raise ValueError(f"{len(snippets)} snippets for k = {arm.k}")
            if generator is None:
                raise ValueError("this arm needs a generator")
            result = generator.generate(build_answer_prompt(question.text, snippets))
            text, finish, prompt_tokens, completion_tokens = result.text, result.finish_reason, result.prompt_tokens, result.completion_tokens
        if not text.strip():
            raise ValueError("the answer is empty")
        letter, why = reading(text)
        return AnswerRecord(
            **base, status="ok", evidence_count=len(snippets), evidence_sha256=text_sha256("\x00".join(s.strip() for s in snippets)),
            text=text, letter=letter, reading=why, correct=letter == question.gold, finish_reason=finish,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, seconds=round(clock() - begun, 3),
        )
    except Exception as exc:  # noqa: BLE001 - recorded per answer; the run goes on
        return AnswerRecord(**base, status="error", seconds=round(clock() - begun, 3), error=f"{type(exc).__name__}: {exc}")


def run_arms(
    questions: Sequence[EvalQuestion],
    arms: Sequence[Arm],
    evidence: Callable[[EvalQuestion, Arm], Sequence[str]],
    generator: Generator,
    store: AnswerStore,
    config: dict,
    relevant: Sequence[str],
    *,
    ledger: Ledger | None = None,
    system: str | None = None,
    progress: Callable[[int, int, AnswerRecord], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunSummary:
    """Answer every question with every arm that has no answer yet, appending one record per answer.

    ``config`` holds ``dataset`` and ``split`` and the other settings in ``relevant``. With ``split`` equal to ``test`` a
    ``ledger`` and a ``system`` name are required. Stops after five failures in a row.
    """
    split = config.get("split")
    if split == "test":
        if ledger is None or not system:
            raise ValueError("a test run needs a ledger and a system name, so that a second scoring of the test set is refused")
        ledger.claim(dataset=config["dataset"], split=split, system=system, settings_sha256=settings_sha256(config, relevant))
    if len({(arm.name, arm.k) for arm in arms}) != len(arms):
        raise ValueError("an arm appears twice")
    store.bind(config, relevant)
    existing = store.load(repair=True)
    tasks = [(question, arm) for question in questions for arm in arms]
    pending = []
    for question, arm in tasks:
        record = existing.get(answer_id(arm.name, arm.k, question.id))
        if record is not None and record.question_sha256 != question_sha256(question):
            raise QuestionChanged(f"{record.id} is in {store.directory} with another question or answer key; use a new output folder")
        if record is None or record.status == "error":
            pending.append((question, arm))
    summary = RunSummary(total=len(tasks), already_done=len(tasks) - len(pending))
    place = {answer_id(arm.name, arm.k, question.id): number for number, (question, arm) in enumerate(tasks, start=1)}
    started = clock()
    consecutive = 0
    for question, arm in pending:
        snippets: Sequence[str] = ()
        try:
            snippets = [] if arm.name == NO_RETRIEVAL else list(evidence(question, arm))
        except Exception as exc:  # noqa: BLE001 - recorded like any other failure of this answer
            record = AnswerRecord(
                id=answer_id(arm.name, arm.k, question.id), question_id=question.id, arm=arm.name, k=arm.k,
                question_sha256=question_sha256(question), status="error", error=f"{type(exc).__name__}: {exc}",
            )
        else:
            record = answer_one(question, arm, snippets, generator, clock)
        store.append(record)
        summary.written += 1
        if record.status == "error":
            summary.errors.append((record.id, record.error or "error"))
            consecutive += 1
        else:
            consecutive = 0
        if progress is not None:
            progress(place[record.id], len(tasks), record)
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            summary.seconds = clock() - started
            raise RunAborted(f"{consecutive} answers in a row failed (last: {record.error}). Fix the cause and run the same command again.")
    summary.seconds = clock() - started
    if split == "test" and not summary.errors and ledger is not None and system:
        ledger.finish(dataset=config["dataset"], split=split, system=system)
    return summary
