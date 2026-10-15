"""Questions in, one resumable record per question out.

An output folder holds ``config.json`` (every setting that shapes the rationales) and ``rationales.jsonl`` (one JSON
line per question, appended and flushed as soon as the question is done). A run that is stopped can be started again
with the same command and continues where it stopped. A folder is bound to its first configuration; a run with a
different result-relevant setting is refused instead of mixing two kinds of rationale in one file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generator import text_sha256

RATIONALES_FILE = "rationales.jsonl"
CONFIG_FILE = "config.json"


class StoreError(RuntimeError):
    """The output folder or a question file cannot be used."""


class ConfigMismatch(StoreError):
    """The folder was made with other settings than the current run."""


class QuestionChanged(StoreError):
    """A question id is in the folder with a different question text."""


class CorruptStore(StoreError):
    """A line of the records file is damaged somewhere other than the end."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str


@dataclass(frozen=True)
class RationaleRecord:
    id: str
    question_sha256: str
    status: str  # "ok" or "error"
    rationale: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RationaleRecord:
        if not isinstance(data, Mapping):
            raise TypeError("not a rationale record")
        known = {name: data.get(name) for name in cls.__dataclass_fields__}
        if not isinstance(known["id"], str) or known["status"] not in ("ok", "error"):
            raise ValueError("not a rationale record")
        return cls(**known)


def write_atomic(path: str | Path, content: str | bytes) -> None:
    """Write a whole file so that a reader finds the old file or the new one, never half of it. Text is UTF-8 with LF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if isinstance(content, str):
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    else:
        with open(temp, "wb") as handle:
            handle.write(content)
    os.replace(temp, path)


def question_hash(text: str) -> str:
    """Hash of a question without its surrounding whitespace, which the prompt ignores anyway."""
    return text_sha256(text.strip())


def read_questions(path: str | Path) -> list[Question]:
    """Questions from a JSON-lines file (or a JSON list) of objects with ``id`` and ``question``.

    Every id must be unique and every question non-empty. The question text carries its answer options inline.
    A byte-order mark, which some Windows editors add, is accepted.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise StoreError(f"cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise StoreError(f"{path} is not UTF-8 text: {exc}") from exc
    rows: list[tuple[str, Any]] = []
    if raw.lstrip().startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(f"{path} is not valid JSON: {exc}") from exc
        rows = [(f"item {index}", item) for index, item in enumerate(items, start=1)]
    else:
        # Split on the line feed only: a question may contain other line-break characters (U+2028 and the like).
        for number, line in enumerate(raw.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                rows.append((f"line {number}", json.loads(line)))
            except json.JSONDecodeError as exc:
                raise StoreError(f"{path}, line {number} is not valid JSON: {exc}") from exc
    questions: list[Question] = []
    seen: set[str] = set()
    for where, item in rows:
        if not isinstance(item, dict) or "id" not in item or "question" not in item:
            raise StoreError(f"{path}, {where}: every question needs an 'id' and a 'question'")
        raw_id, text = item["id"], item["question"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, str | int):
            raise StoreError(f"{path}, {where}: the id must be text or a whole number")
        question_id = str(raw_id).strip()
        if not question_id:
            raise StoreError(f"{path}, {where}: the id is empty")
        if not isinstance(text, str) or not text.strip():
            raise StoreError(f"{path}, {where}: the question of {question_id} is empty")
        if question_id in seen:
            raise StoreError(f"{path}, {where}: the id {question_id} appears twice")
        seen.add(question_id)
        questions.append(Question(question_id, text.strip()))
    if not questions:
        raise StoreError(f"{path} holds no questions")
    return questions


class RationaleStore:
    """One output folder."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.records_path = self.directory / RATIONALES_FILE
        self.config_path = self.directory / CONFIG_FILE
        self.recovered_partial_line = False
        self._size: int | None = None  # size of the records file after this object last read or wrote it
        self._unrepaired_tail = False

    def bind(self, config: Mapping[str, Any], relevant: Sequence[str]) -> None:
        """Record ``config`` on first use; afterwards require that the ``relevant`` fields are unchanged."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            if self.records_path.exists() and self.records_path.stat().st_size > 0:
                raise StoreError(
                    f"{self.records_path} holds records but {self.config_path} is missing, so it is not known how they "
                    "were made; use a new output folder"
                )
            write_atomic(self.config_path, json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            return
        recorded = self.read_config()
        differing = [name for name in relevant if recorded.get(name) != config.get(name)]
        if differing:
            details = "; ".join(f"{n}: recorded {recorded.get(n)!r}, now {config.get(n)!r}" for n in differing)
            raise ConfigMismatch(
                f"{self.directory} was made with other settings ({details}). Use a new output folder for the new settings."
            )

    def read_config(self) -> dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read {self.config_path}: {exc}") from exc

    def load(self, *, repair: bool = False) -> dict[str, RationaleRecord]:
        """The latest record per id.

        A last line without its line break was cut short by an interrupted write and is ignored. With
        ``repair=True`` (the writer only) it is also removed from the file, so the next record starts on a new
        line; a reader never changes the file, because the tail may be a write in progress.
        """
        self.recovered_partial_line = False
        self._unrepaired_tail = False
        if not self.records_path.exists():
            self._size = 0
            return {}
        data = self.records_path.read_bytes()
        self._size = len(data)
        if data and not data.endswith(b"\n"):
            keep = data.rfind(b"\n") + 1
            data = data[:keep]
            self.recovered_partial_line = True
            if repair:
                with open(self.records_path, "r+b") as handle:
                    handle.truncate(keep)
                self._size = keep
            else:
                self._unrepaired_tail = True
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CorruptStore(f"{self.records_path} is not valid UTF-8: {exc}") from exc
        records: dict[str, RationaleRecord] = {}
        for number, line in enumerate(text.split("\n"), start=1):
            if not line:
                continue
            try:
                record = RationaleRecord.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise CorruptStore(f"{self.records_path}, line {number} is damaged: {exc}") from exc
            records[record.id] = record
        return records

    def append(self, record: RationaleRecord) -> None:
        """Add one record. Refuses if the file is not as this object left it (another run writing to the same folder)."""
        if self._unrepaired_tail:
            raise StoreError(f"{self.records_path} ends in a cut-off line; read it with load(repair=True) before writing")
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.records_path, "ab") as handle:
            now = os.fstat(handle.fileno()).st_size
            if self._size is not None and now != self._size:
                raise StoreError(
                    f"{self.records_path} changed while this run was writing to it ({self._size} bytes expected, "
                    f"{now} found). Is another run using the same output folder?"
                )
            handle.write((record.to_json() + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            self._size = os.fstat(handle.fileno()).st_size


def load_rationales(directory: str | Path, questions: Iterable[Question]) -> dict[str, str]:
    """The rationale of every question, in a dict. Raises unless every question has a successful record.

    A question without a rationale is never skipped or filled in: a missing entry would shift every later query
    against its question (the authors' retriever matches queries to questions by position).
    """
    records = RationaleStore(directory).load()
    rationales: dict[str, str] = {}
    problems: list[str] = []
    for question in questions:
        record = records.get(question.id)
        if record is None:
            problems.append(f"{question.id}: no record")
        elif record.status != "ok":
            problems.append(f"{question.id}: {record.error or 'error'}")
        elif record.question_sha256 != question_hash(question.text):
            problems.append(f"{question.id}: the question text differs from the one the rationale was written for")
        else:
            rationales[question.id] = record.rationale or ""
    if problems:
        more = f" ... ({len(problems)} problems in all)" if len(problems) > 5 else ""
        raise StoreError(f"not every question has a usable rationale in {directory}: {'; '.join(problems[:5])}{more}")
    return rationales
