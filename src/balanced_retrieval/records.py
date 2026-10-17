"""One resumable record per question, bound to the settings of the run.

It is the store of Part 1 with another kind of record: a folder holds ``config.json`` (every setting that shapes the
evidence) and ``retrieved.jsonl`` (one JSON line per question, appended and flushed as soon as the question is done). A run
that is stopped is started again with the same command; a run with other result-relevant settings, or a question whose
text changed, is refused instead of mixing two kinds of evidence in one file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rationale_query.store import RationaleStore

RECORDS_FILE = "retrieved.jsonl"


@dataclass(frozen=True)
class RetrievalRecord:
    """The final evidence of one question. ``status`` is always ``"ok"``: a failing run stops instead of recording a gap.

    The hash of the question and the hash of the rationale it was searched with let a later run notice that either changed.
    """

    id: str
    question_sha256: str
    rationale_sha256: str
    pool_size: int
    evidence: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    def to_json(self) -> str:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalRecord:
        if not isinstance(data, Mapping):
            raise TypeError("not a retrieval record")
        if not isinstance(data.get("id"), str) or data.get("status") != "ok" or not isinstance(data.get("evidence"), list):
            raise ValueError("not a retrieval record")
        return cls(
            id=data["id"],
            question_sha256=str(data.get("question_sha256")),
            rationale_sha256=str(data.get("rationale_sha256")),
            pool_size=int(data.get("pool_size", 0)),
            evidence=data["evidence"],
        )


class RetrievalStore(RationaleStore):
    """One output folder of the ``retrieve`` command."""

    records_file = RECORDS_FILE
    record_type = RetrievalRecord
