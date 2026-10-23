"""The questions of the three benchmarks as one kind of record, and how a question is written out.

Every benchmark here is four-option multiple choice. A reader takes the rows of one split, as dictionaries, and returns
``Item`` records with the stem, the four options, the answer letter and an id that stays the same between runs. The
readers check what the dataset cards promise and refuse anything else, so a wrong assumption is a visible error and not
a silent shift of every answer:

* MedQA (``awinml/medqa``, configuration ``questions``): ``question``, ``options`` (A to D), ``answer_idx``.
* MedMCQA (``openlifescienceai/medmcqa``): ``question``, ``opa`` to ``opd``, ``cop`` as 0 to 3 (A is 0; checked by the
  researcher on 2026-10-23, although the card says 1 to 4). The answers of the test split are withheld (``cop`` is -1 in
  every row), so that split is refused.
* MMLU-Med (``Layered-Labs/benchbase-mmlu-medical``): ``question``, ``options`` (a list of ``original_key`` and ``text``)
  and ``answer`` (the matching option).

The question text of a record is written with its options inline, ``<stem> A) <one> B) <two> C) <three> D) <four>``, the form
of the released training examples of RAG² (docs/methodology.md, section 5).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LETTERS = "ABCD"
DATASETS = ("medqa", "medmcqa", "mmlu_med")


class QuestionDataError(ValueError):
    """A row does not have the form its dataset card promises."""


class AnswersWithheld(QuestionDataError):
    """The split has no answer key (the MedMCQA test split); its questions cannot be scored."""


@dataclass(frozen=True)
class Item:
    """One question. ``answer`` is a letter from A to D; ``id`` is unique inside one dataset and split."""

    id: str
    dataset: str
    split: str
    stem: str
    options: tuple[str, ...]
    answer: str
    subject: str | None = None
    choice_type: str | None = None


def render_question(item: Item) -> str:
    """The question with its options inline, as in the released samples: one space between the parts."""
    options = " ".join(f"{letter}) {text}" for letter, text in zip(LETTERS, item.options, strict=True))
    return f"{item.stem} {options}"


def question_sha256(item: Item) -> str:
    """A hash of what the model sees (the rendered question) and of the answer, to notice a changed dataset."""
    return hashlib.sha256(f"{render_question(item)}\x00{item.answer}".encode()).hexdigest()


def _text(where: str, name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuestionDataError(f"{where}: {name} must be non-empty text, got {value!r}")
    return value.strip()


def _row(where: str, row: Any) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise QuestionDataError(f"{where}: a row must be a mapping, got {type(row).__name__}")
    return row


def _unique(items: list[Item], dataset: str, split: str) -> list[Item]:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise QuestionDataError(f"{dataset} {split}: the id {item.id} appears twice")
        seen.add(item.id)
    return items


def _letter_options(where: str, options: Any) -> tuple[str, ...]:
    """Four option texts in the order A to D, from a mapping keyed A to D or from a list of four."""
    if isinstance(options, Mapping):
        if sorted(options) != list(LETTERS):
            raise QuestionDataError(f"{where}: the options must be keyed A, B, C and D, got {sorted(options)!r}")
        values = [options[letter] for letter in LETTERS]
    elif isinstance(options, Sequence) and not isinstance(options, str):
        values = list(options)
        if len(values) != 4:
            raise QuestionDataError(f"{where}: there must be four options, got {len(values)}")
    else:
        raise QuestionDataError(f"{where}: the options must be a mapping or a list, got {type(options).__name__}")
    return tuple(_text(where, f"option {letter}", value) for letter, value in zip(LETTERS, values, strict=True))


def read_medqa(rows: Iterable[Mapping[str, Any]], split: str) -> list[Item]:
    """MedQA rows. The id is ``medqa:<split>:<position>``; the dataset has no id of its own, so the position in the split
    is the id and the dataset revision must be pinned (the manifest records the hash of every question)."""
    items = []
    for index, raw in enumerate(rows):
        where = f"medqa {split} row {index}"
        row = _row(where, raw)
        options = _letter_options(where, row.get("options"))
        letter = row.get("answer_idx")
        if letter not in tuple(LETTERS):
            raise QuestionDataError(f"{where}: answer_idx must be one of A, B, C, D, got {letter!r}")
        answer_text = row.get("answer")
        if answer_text is not None and _text(where, "answer", answer_text) != options[LETTERS.index(letter)]:
            raise QuestionDataError(f"{where}: the answer text is not the text of option {letter}")
        items.append(Item(f"medqa:{split}:{index}", "medqa", split, _text(where, "question", row.get("question")), options, letter))
    return _unique(items, "medqa", split)


def read_medmcqa(rows: Iterable[Mapping[str, Any]], split: str) -> list[Item]:
    """MedMCQA rows, with ``cop`` as 0 to 3. A split whose ``cop`` is -1 (the test split) raises ``AnswersWithheld``."""
    items = []
    withheld = 0
    for index, raw in enumerate(rows):
        where = f"medmcqa {split} row {index}"
        row = _row(where, raw)
        cop = row.get("cop")
        if isinstance(cop, bool) or not isinstance(cop, int):
            raise QuestionDataError(f"{where}: cop must be a whole number, got {cop!r}")
        if cop == -1:
            withheld += 1
            continue
        if not 0 <= cop <= 3:
            raise QuestionDataError(f"{where}: cop must be 0 to 3 (A is 0), got {cop}")
        options = tuple(_text(where, name, row.get(name)) for name in ("opa", "opb", "opc", "opd"))
        question_id = _text(where, "id", row.get("id"))
        subject = row.get("subject_name")
        choice_type = row.get("choice_type")
        items.append(
            Item(
                question_id, "medmcqa", split, _text(where, "question", row.get("question")), options, LETTERS[cop],
                subject if isinstance(subject, str) else None, choice_type if isinstance(choice_type, str) else None,
            )
        )
    if withheld:
        if items:
            raise QuestionDataError(f"medmcqa {split}: {withheld} rows have cop = -1 and {len(items)} do not; the split is mixed")
        raise AnswersWithheld(f"the answers of the medmcqa {split} split are withheld (cop = -1 in all {withheld} rows)")
    return _unique(items, "medmcqa", split)


def read_mmlu_med(rows: Iterable[Mapping[str, Any]], split: str) -> list[Item]:
    """MMLU-Med rows of the BenchBase layout. The id is ``mmlu_med:<split>:<position>``."""
    items = []
    for index, raw in enumerate(rows):
        where = f"mmlu_med {split} row {index}"
        row = _row(where, raw)
        listed = row.get("options")
        if not isinstance(listed, Sequence) or isinstance(listed, str) or len(listed) != 4:
            raise QuestionDataError(f"{where}: there must be four options")
        keys, texts = [], []
        for position, option in enumerate(listed):
            if not isinstance(option, Mapping):
                raise QuestionDataError(f"{where}: option {position} must be a mapping with original_key and text")
            keys.append(option.get("original_key", LETTERS[position]))
            texts.append(_text(where, f"option {LETTERS[position]}", option.get("text")))
        if keys != list(LETTERS):
            raise QuestionDataError(f"{where}: the options must come in the order A, B, C, D, got {keys!r}")
        answer = row.get("answer")
        if not isinstance(answer, Mapping) or answer.get("original_key") not in tuple(LETTERS):
            raise QuestionDataError(f"{where}: the answer must name one of the options A to D")
        letter = answer["original_key"]
        if _text(where, "answer text", answer.get("text")) != texts[LETTERS.index(letter)]:
            raise QuestionDataError(f"{where}: the answer text is not the text of option {letter}")
        metadata = row.get("metadata")
        subject = None
        if isinstance(metadata, Mapping):
            subject = next((metadata[name] for name in ("subject", "subject_name") if isinstance(metadata.get(name), str)), None)
        items.append(Item(f"mmlu_med:{split}:{index}", "mmlu_med", split, _text(where, "question", row.get("question")), tuple(texts), letter, subject))
    return _unique(items, "mmlu_med", split)


READERS = {"medqa": read_medqa, "medmcqa": read_medmcqa, "mmlu_med": read_mmlu_med}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Rows from a local file: a JSON list, JSON lines, or (if pyarrow is installed) a Parquet file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise QuestionDataError(f"reading {path} needs pyarrow; export the split as JSON lines instead") from exc
        return parquet.read_table(path).to_pylist()
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise QuestionDataError(f"cannot read {path}: {exc}") from exc
    try:
        if raw.lstrip().startswith("["):
            rows = json.loads(raw)
        else:
            rows = [json.loads(line) for line in raw.split("\n") if line.strip()]
    except json.JSONDecodeError as exc:
        raise QuestionDataError(f"{path} is not valid JSON or JSON lines: {exc}") from exc
    if not isinstance(rows, list):
        raise QuestionDataError(f"{path} must hold a list of rows")
    return rows


def hub_rows(repo: str, split: str, *, config: str | None = None, revision: str | None = None, loader: Any = None) -> list[dict[str, Any]]:
    """The rows of one split of a Hugging Face dataset. ``loader`` is ``datasets.load_dataset`` (imported only here)."""
    if loader is None:
        try:
            from datasets import load_dataset as loader
        except ImportError as exc:
            raise QuestionDataError("loading from the hub needs the datasets package; or give a local JSON-lines file") from exc
    arguments: dict[str, Any] = {"split": split}
    if revision:
        arguments["revision"] = revision
    data = loader(repo, config, **arguments) if config else loader(repo, **arguments)
    return [dict(row) for row in data]


def manifest(items: Sequence[Item]) -> dict[str, Any]:
    """What a later run needs to see that the questions are the same: the count and a hash for every question."""
    ordered = sorted(items, key=lambda item: item.id)
    return {
        "dataset": ordered[0].dataset if ordered else None,
        "split": ordered[0].split if ordered else None,
        "count": len(ordered),
        "questions": {item.id: question_sha256(item) for item in ordered},
    }


def summarise(items: Sequence[Item]) -> dict[str, Any]:
    """Counts for a check by eye: how many, how the answers are spread, how many stems repeat, which subjects."""
    answers = {letter: sum(1 for item in items if item.answer == letter) for letter in LETTERS}
    stems = [" ".join(item.stem.lower().split()) for item in items]
    subjects: dict[str, int] = {}
    for item in items:
        if item.subject:
            subjects[item.subject] = subjects.get(item.subject, 0) + 1
    choice_types: dict[str, int] = {}
    for item in items:
        if item.choice_type:
            choice_types[item.choice_type] = choice_types.get(item.choice_type, 0) + 1
    return {
        "count": len(items),
        "answers": answers,
        "repeated_stems": len(stems) - len(set(stems)),
        "subjects": dict(sorted(subjects.items())),
        "choice_types": dict(sorted(choice_types.items())),
    }
