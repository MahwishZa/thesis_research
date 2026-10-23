"""The training file of the filter, in the format of the released sample (docs/methodology.md, section 5).

A record has the four fields of the authors' ``5%-train.json``: ``id``, ``answer`` (``[HELPFUL]`` or ``[NOT_HELPFUL]``),
``dataset_name`` and ``question`` (the filter input: the instruction, the snippet and the question). Discarded pairs, and pairs
that were skipped or failed, have no record. The file is a JSON list; writing refuses to replace an existing file, because the
labels cost many hours and a rerun that replaced them silently would destroy the provenance of a trained filter.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rationale_query.store import write_atomic

from .inference import build_filter_input
from .label_records import LabelledPair
from .labelling import PairTask, hashes
from .labels import DISCARD, TOKEN_HELPFUL, TOKEN_NOT_HELPFUL, to_training_token

FIELDS = ("id", "answer", "dataset_name", "question")


class TrainingDataError(ValueError):
    """The labels and the pairs they came from do not belong together, or the file cannot be written."""


def training_records(labelled: Sequence[LabelledPair], tasks: Sequence[PairTask], dataset_name: str) -> list[dict[str, str]]:
    """One record per labelled pair that is not discarded, in the order of ``labelled``.

    Each pair is matched to its task by id, and the hashes of the question and the snippet must agree with the record, so a
    label can never be attached to a different question or snippet than the one it was made from.
    """
    if not isinstance(dataset_name, str) or not dataset_name:
        raise TrainingDataError("the dataset name must be non-empty text")
    by_id = {task.id: task for task in tasks}
    records: list[dict[str, str]] = []
    for item in labelled:
        if item.label == DISCARD:
            continue
        record = item.record
        task = by_id.get(record.id)
        if task is None:
            raise TrainingDataError(f"{record.id} has a label but no task to take its question and snippet from")
        if (record.question_sha256, record.snippet_sha256) != hashes(task):
            raise TrainingDataError(f"{record.id}: the question or the snippet is not the one the label was made from")
        records.append(
            {
                "id": f"{dataset_name}_{record.id}",
                "answer": to_training_token(item.label),
                "dataset_name": dataset_name,
                "question": build_filter_input(evidence=task.snippet, question=task.question),
            }
        )
    return records


def write_training_file(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write the JSON list. Refuses an existing file and a record that does not have exactly the four fields."""
    target = Path(path)
    if target.exists():
        raise TrainingDataError(f"refusing to overwrite {target}; labels are expensive, so choose a new file name")
    ids = set()
    for position, record in enumerate(records):
        if tuple(record) != FIELDS:
            raise TrainingDataError(f"record {position} must have the fields {', '.join(FIELDS)} in that order, got {', '.join(record)}")
        if record["answer"] not in (TOKEN_HELPFUL, TOKEN_NOT_HELPFUL):
            raise TrainingDataError(f"record {position}: the answer must be {TOKEN_HELPFUL} or {TOKEN_NOT_HELPFUL}, got {record['answer']!r}")
        if record["id"] in ids:
            raise TrainingDataError(f"the id {record['id']} appears twice")
        ids.add(record["id"])
    write_atomic(target, json.dumps(list(records), indent=2, ensure_ascii=False) + "\n")
    return target


def read_training_file(path: str | Path) -> list[dict[str, Any]]:
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"cannot read {path}: {exc}") from exc
    if not isinstance(records, list):
        raise TrainingDataError(f"{path} must hold a JSON list")
    return records


def label_balance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many of each token, and the share of the larger class. A filter trained on a set that is almost all one label learns the prior."""
    counts = Counter(record["answer"] for record in records)
    total = sum(counts.values())
    return {
        TOKEN_HELPFUL: counts[TOKEN_HELPFUL],
        TOKEN_NOT_HELPFUL: counts[TOKEN_NOT_HELPFUL],
        "majority_share": round(max(counts.values()) / total, 4) if total else None,
    }
