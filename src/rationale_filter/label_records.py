"""One resumable record per question-snippet pair, and the labels that follow from the records.

A folder holds ``config.json`` (every setting that shapes the numbers) and ``pairs.jsonl`` (one JSON line per pair, appended
and flushed as soon as the pair is done), exactly like the stores of Parts 1 and 2: a run that is stopped is started again with
the same command, and a run with other result-relevant settings is refused. The answer written with the snippet is kept as
text, so a reading rule that is widened after the validation check (decision D16) can be applied again without writing it again.

τ is not stored: it is the top 25% of the perplexity differences of all pairs that went through the rule (decision D18), so it
is computed from the records when the labels are made, and it changes as pairs are added.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rationale_query.store import RationaleStore

from .labels import DISCARD, HELPFUL, NOT_HELPFUL, TOP_FRACTION, label_pair, perplexity_threshold

PAIRS_FILE = "pairs.jsonl"
STATUSES = ("ok", "skipped", "error")
SKIP_REASONS = ("without_unreadable", "with_unreadable")

#: The fields of the run's configuration that decide what a record means. A folder whose recorded values differ is refused.
RELEVANT = (
    "dataset",
    "split",
    "generator",
    "scorer",
    "answer_prompt_sha256",
    "reader_pattern",
    "snippets_per_question",
)


def pair_id(question_id: str, rank: int) -> str:
    return f"{question_id}#{rank}"


@dataclass(frozen=True)
class PairRecord:
    """Everything known about one pair.

    ``status`` is ``ok`` (both answers readable, both perplexities known), ``skipped`` (an answer was unreadable, so the pair has
    no label and is not used; ``reason`` says which) or ``error`` (a failure, tried again by the next run).
    """

    id: str
    question_id: str
    rank: int
    question_sha256: str
    snippet_sha256: str
    status: str
    reason: str | None = None
    correct_without: bool | None = None
    correct_with: bool | None = None
    ppl_without: float | None = None
    ppl_with: float | None = None
    generation_with: str | None = None
    finish_reason_with: str | None = None
    seconds: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PairRecord:
        if not isinstance(data, Mapping):
            raise TypeError("not a pair record")
        known = {name: data.get(name) for name in cls.__dataclass_fields__}
        if not isinstance(known["id"], str) or known["status"] not in STATUSES:
            raise ValueError("not a pair record")
        if not isinstance(known["question_id"], str) or isinstance(known["rank"], bool) or not isinstance(known["rank"], int):
            raise ValueError("not a pair record")
        if known["id"] != pair_id(known["question_id"], known["rank"]):
            raise ValueError("the id of a pair record is its question id and rank")
        if known["status"] == "ok":
            for name in ("correct_without", "correct_with"):
                if not isinstance(known[name], bool):
                    raise ValueError(f"an ok record needs {name} as true or false")
            for name in ("ppl_without", "ppl_with"):
                value = known[name]
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"an ok record needs {name} above zero")
        if known["status"] == "skipped" and known["reason"] not in SKIP_REASONS:
            raise ValueError("a skipped record needs a reason")
        return cls(**known)


class PairStore(RationaleStore):
    """One output folder of the labelling run."""

    records_file = PAIRS_FILE
    record_type = PairRecord


@dataclass(frozen=True)
class LabelledPair:
    record: PairRecord
    label: str


def label_records(
    records: Sequence[PairRecord] | Mapping[str, PairRecord], top_fraction: float = TOP_FRACTION
) -> tuple[float, list[LabelledPair]]:
    """``(τ, the labelled pairs)`` for the ``ok`` records, in the order given (sorted by id for a mapping).

    τ is taken over the differences of all ``ok`` records before anything is discarded (D18). Skipped and error records have
    no label. Raises ``ValueError`` when no record is ``ok``.
    """
    if isinstance(records, Mapping):
        records = [records[key] for key in sorted(records)]
    usable = [record for record in records if record.status == "ok"]
    if not usable:
        raise ValueError("no pair is ok, so there is no pool for τ")
    tau = perplexity_threshold([record.ppl_without - record.ppl_with for record in usable], top_fraction)
    labelled = [
        LabelledPair(
            record,
            label_pair(
                correct_without=record.correct_without,
                correct_with=record.correct_with,
                ppl_without=record.ppl_without,
                ppl_with=record.ppl_with,
                tau=tau,
            ),
        )
        for record in usable
    ]
    return tau, labelled


def summarise(records: Sequence[PairRecord], labelled: Sequence[LabelledPair] | None = None) -> dict[str, Any]:
    """What a labelling run holds: records by status, why pairs were skipped, and the labels if they were made."""
    summary: dict[str, Any] = {
        "records": len(records),
        "status": {name: sum(1 for r in records if r.status == name) for name in STATUSES},
        "skipped_reasons": dict(Counter(r.reason for r in records if r.status == "skipped")),
        "questions": len({r.question_id for r in records}),
    }
    if labelled is not None:
        counts = Counter(item.label for item in labelled)
        summary["labels"] = {name: counts[name] for name in (HELPFUL, NOT_HELPFUL, DISCARD)}
        kept = counts[HELPFUL] + counts[NOT_HELPFUL]
        summary["helpful_share_of_kept"] = round(counts[HELPFUL] / kept, 4) if kept else None
    return summary
