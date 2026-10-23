"""One resumable record per answer: a question, a system and a number of snippets k.

It is the store of Part 1 with another kind of record. A folder holds ``config.json`` (every setting that shapes the answers)
and ``answers.jsonl`` (one JSON line per answer, appended and flushed as soon as it is written). The text the model wrote is kept,
so a reading rule that is widened after the validation check (decision D16) can be applied again without writing the answers again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rationale_query.store import RationaleStore

ANSWERS_FILE = "answers.jsonl"
STATUSES = ("ok", "error")

#: The fields of the run's configuration that decide what an answer is. A folder whose recorded values differ is refused.
RELEVANT = ("dataset", "split", "generator", "answer_prompt_sha256", "reader_pattern")


def answer_id(arm: str, k: int, question_id: str) -> str:
    return f"{arm}|k{k}|{question_id}"


@dataclass(frozen=True)
class AnswerRecord:
    """The answer of one system (``arm`` and ``k``) to one question. ``correct`` is False for an unreadable answer (decision D16)."""

    id: str
    question_id: str
    arm: str
    k: int
    question_sha256: str
    status: str
    evidence_count: int = 0
    evidence_sha256: str | None = None
    text: str | None = None
    letter: str | None = None
    reading: str | None = None
    correct: bool | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnswerRecord:
        if not isinstance(data, Mapping):
            raise TypeError("not an answer record")
        known = {name: data.get(name) for name in cls.__dataclass_fields__}
        if not isinstance(known["id"], str) or known["status"] not in STATUSES:
            raise ValueError("not an answer record")
        if not isinstance(known["question_id"], str) or not isinstance(known["arm"], str):
            raise ValueError("not an answer record")
        if isinstance(known["k"], bool) or not isinstance(known["k"], int):
            raise ValueError("not an answer record")
        if known["id"] != answer_id(known["arm"], known["k"], known["question_id"]):
            raise ValueError("the id of an answer record is its arm, k and question id")
        if known["status"] == "ok" and not isinstance(known["correct"], bool):
            raise ValueError("an ok record needs correct as true or false")
        return cls(**known)


class AnswerStore(RationaleStore):
    """One output folder of an evaluation run."""

    records_file = ANSWERS_FILE
    record_type = AnswerRecord
