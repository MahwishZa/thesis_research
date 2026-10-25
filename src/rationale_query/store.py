"""Questions in, rationales out, saved so that a long run can be stopped and started again.

A run over thousands of questions takes days on a laptop, so every rationale is saved the moment it is written. The folder
of a run holds ``config.json`` (the settings that decide what a rationale is) and ``rationales.jsonl`` (one line per
answer). A run started again with other settings, or on a question whose text has changed, is refused, so that one folder
never mixes two setups. A failure is saved as a failure; it is never replaced by anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_FILE = "config.json"
RECORDS_FILE = "rationales.jsonl"
STATUSES = ("ok", "error")


class StoreError(RuntimeError):
    """Something is wrong with the questions or with the saved results."""


class ConfigMismatch(StoreError):
    """The folder was made with other settings."""


class QuestionChanged(StoreError):
    """A saved rationale belongs to another text than the question it is now asked for."""


class CorruptStore(StoreError):
    """A saved line cannot be read."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str


def question_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_questions(path: str | Path) -> list[Question]:
    """Questions from a JSON-lines file of objects with ``id`` and ``question``. The question text carries its options."""
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise StoreError(f"cannot read {path}: {exc}") from exc
    questions: list[Question] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StoreError(f"{path}, line {number} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("question"), str):
            raise StoreError(f"{path}, line {number}: every row needs a text 'id' and a text 'question'")
        if not row["question"].strip():
            raise StoreError(f"{path}, line {number}: the question of {row['id']} is empty")
        if row["id"] in seen:
            raise StoreError(f"{path}, line {number}: the id {row['id']} appears twice")
        seen.add(row["id"])
        questions.append(Question(row["id"], row["question"]))
    if not questions:
        raise StoreError(f"{path} holds no questions")
    return questions


@dataclass(frozen=True)
class RationaleRecord:
    """The result for one question. ``rationale`` is the model's complete text, unchanged."""

    id: str
    question_sha256: str
    status: str
    rationale: str | None = None
    finish_reason: str | None = None  # "stop", or "length" when the memory window was full
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RationaleRecord:
        if not isinstance(data, Mapping) or not isinstance(data.get("id"), str) or data.get("status") not in STATUSES:
            raise ValueError("not a rationale record")
        if not isinstance(data.get("question_sha256"), str):
            raise ValueError("not a rationale record")
        if data["status"] == "ok" and not (isinstance(data.get("rationale"), str) and data["rationale"].strip()):
            raise ValueError("an ok record needs a rationale")
        return cls(
            id=data["id"], question_sha256=data["question_sha256"], status=data["status"], rationale=data.get("rationale"),
            finish_reason=data.get("finish_reason"), prompt_tokens=data.get("prompt_tokens"), completion_tokens=data.get("completion_tokens"),
            seconds=data.get("seconds"), error=data.get("error"),
        )


def write_atomic(path: str | Path, text: str) -> None:
    """Write a file so that a crash leaves the old file or the new one, never half of either."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RationaleStore:
    """One output folder of a run."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.config_path = self.directory / CONFIG_FILE
        self.records_path = self.directory / RECORDS_FILE

    def bind(self, config: Mapping[str, Any], relevant: Sequence[str]) -> None:
        """Tie the folder to its settings: record them the first time, refuse a run with other values afterwards."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            write_atomic(self.config_path, json.dumps(dict(config), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            return
        recorded = self.read_config()
        differing = [name for name in relevant if recorded.get(name) != config.get(name)]
        if differing:
            details = "; ".join(f"{name}: recorded {recorded.get(name)!r}, now {config.get(name)!r}" for name in differing)
            raise ConfigMismatch(f"{self.directory} was made with other settings ({details}). Use a new output folder for the new settings.")

    def read_config(self) -> dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read {self.config_path}: {exc}") from exc

    def load(self, *, repair: bool = False) -> dict[str, RationaleRecord]:
        """The latest record of each question. A last line cut short by a stop is ignored (removed, with ``repair=True``)."""
        if not self.records_path.exists():
            return {}
        raw = self.records_path.read_bytes().decode("utf-8")
        lines = raw.split("\n")
        cut_short = lines[-1] != ""
        records: dict[str, RationaleRecord] = {}
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = RationaleRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                if cut_short and number == len(lines):
                    if repair:
                        keep = raw[: raw.rfind("\n") + 1]
                        write_atomic(self.records_path, keep)
                    continue
                raise CorruptStore(f"{self.records_path}, line {number} cannot be read: {exc}") from exc
            records[record.id] = record
        return records

    def append(self, record: RationaleRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.records_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
