"""The loop that turns question-snippet pairs into pair records (decisions D14 to D20 of docs/methodology.md).

For a pair the answer without the snippet is the rationale Part 1 wrote (its final option; the rationale prompt asks for one).
The answer with the snippet is written from the answer prompt with a one-snippet Evidence section. Both are read strictly; a
pair with an unreadable answer is skipped before any more work is spent on it. For the others the rationale is scored under the
prompt without and the prompt with the snippet, and the two perplexities go into the record. The label is made later, from all
records together, because τ is a quantile of the whole pool.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from answer_evaluation.prompt import build_answer_prompt
from answer_evaluation.reader import labelling_correct
from rationale_query.batch import MAX_CONSECUTIVE_ERRORS, RunAborted
from rationale_query.generator import GenerationResult, text_sha256
from rationale_query.store import QuestionChanged

from .label_records import PairRecord, PairStore, pair_id
from .perplexity import perplexity


class Generator(Protocol):
    config: dict

    def generate(self, user_message: str) -> GenerationResult: ...


class Scorer(Protocol):
    config: dict

    def token_logprobs(self, user_message: str, completion: str) -> list[float]: ...


@dataclass(frozen=True)
class PairTask:
    """One pair to label: the question with its options inline, its answer key, the rationale of Part 1, and one snippet."""

    question_id: str
    question: str
    gold: str
    rationale: str
    rank: int
    snippet: str

    @property
    def id(self) -> str:
        return pair_id(self.question_id, self.rank)


@dataclass
class RunSummary:
    total: int = 0
    already_done: int = 0
    written: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0


def hashes(task: PairTask) -> tuple[str, str]:
    """What a record was made from: the question with its answer key, and the snippet."""
    return text_sha256(f"{task.question.strip()}\x00{task.gold}"), text_sha256(task.snippet.strip())


def plan(tasks: Sequence[PairTask], existing: dict[str, PairRecord], directory: object = "") -> list[PairTask]:
    """The tasks still to do. A task whose question or snippet differs from the record's raises ``QuestionChanged``."""
    pending = []
    for task in tasks:
        record = existing.get(task.id)
        if record is not None and (record.question_sha256, record.snippet_sha256) != hashes(task):
            raise QuestionChanged(f"{task.id} is in {directory} with another question or snippet; use a new output folder")
        if record is None or record.status == "error":
            pending.append(task)
    return pending


def score_pair(
    task: PairTask,
    generator: Generator,
    scorer: Scorer,
    *,
    without_cache: dict[str, list[float]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PairRecord:
    """The record of one pair. ``without_cache`` keeps the scoring without a snippet, which is the same for every snippet of a question."""
    begun = clock()
    question_hash, snippet_hash = hashes(task)
    base = {"id": task.id, "question_id": task.question_id, "rank": task.rank, "question_sha256": question_hash, "snippet_sha256": snippet_hash}
    try:
        correct_without = labelling_correct(task.rationale, task.gold)
        if correct_without is None:
            return PairRecord(**base, status="skipped", reason="without_unreadable", seconds=round(clock() - begun, 3))
        answer = generator.generate(build_answer_prompt(task.question, [task.snippet]))
        if not answer.text.strip():
            raise ValueError("the model returned no text")
        correct_with = labelling_correct(answer.text, task.gold)
        if correct_with is None:
            return PairRecord(
                **base, status="skipped", reason="with_unreadable", correct_without=correct_without,
                generation_with=answer.text, finish_reason_with=answer.finish_reason, seconds=round(clock() - begun, 3),
            )
        cache = without_cache if without_cache is not None else {}
        if task.question_id not in cache:
            cache[task.question_id] = scorer.token_logprobs(build_answer_prompt(task.question), task.rationale)
        with_snippet = scorer.token_logprobs(build_answer_prompt(task.question, [task.snippet]), task.rationale)
        if len(with_snippet) != len(cache[task.question_id]):
            raise ValueError("the rationale has a different number of tokens under the two prompts")
        return PairRecord(
            **base, status="ok", correct_without=correct_without, correct_with=correct_with,
            ppl_without=perplexity(cache[task.question_id]), ppl_with=perplexity(with_snippet),
            generation_with=answer.text, finish_reason_with=answer.finish_reason, seconds=round(clock() - begun, 3),
        )
    except Exception as exc:  # noqa: BLE001 - recorded per pair; the run goes on
        return PairRecord(**base, status="error", seconds=round(clock() - begun, 3), error=f"{type(exc).__name__}: {exc}")


def label_pairs(
    tasks: Sequence[PairTask],
    generator: Generator,
    scorer: Scorer,
    store: PairStore,
    config: dict,
    relevant: Sequence[str],
    *,
    progress: Callable[[int, int, PairRecord], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunSummary:
    """Write a record for every pair that has none yet, appending one record per pair; stops after five failures in a row."""
    store.bind(config, relevant)
    existing = store.load(repair=True)
    pending = plan(tasks, existing, store.directory)
    summary = RunSummary(total=len(tasks), already_done=len(tasks) - len(pending))
    place = {task.id: number for number, task in enumerate(tasks, start=1)}
    cache: dict[str, list[float]] = {}
    started = clock()
    consecutive = 0
    for task in pending:
        record = score_pair(task, generator, scorer, without_cache=cache, clock=clock)
        store.append(record)
        summary.written += 1
        if record.status == "error":
            summary.errors.append((task.id, record.error or "error"))
            consecutive += 1
        else:
            consecutive = 0
        if progress is not None:
            progress(place[task.id], len(tasks), record)
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            summary.seconds = clock() - started
            raise RunAborted(f"{consecutive} pairs in a row failed (last: {record.error}). Fix the cause and run the same command again.")
    summary.seconds = clock() - started
    return summary
