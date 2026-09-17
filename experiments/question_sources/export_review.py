"""Regenerate the reviewer-facing file from the candidate pool.

Reads ``candidates.jsonl`` and writes ``review.csv``. It reads only - the pool
is never rewritten here - so the reviewer file can be regenerated at any time
without risk to the candidates or their provenance.

The export is neutral by construction: it emits exactly
``questions.REVIEW_COLUMNS`` and refuses to run if any withheld field would
reach the file. The point is that a reviewer should judge each candidate from
its source, not re-confirm a classification this pipeline already assigned.

    python -m experiments.question_sources.export_review
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from ..evaluation.questions import (
    REVIEW_COLUMNS, WITHHELD_FROM_REVIEW, EvaluationQuestion, review_export,
)

DEFAULT_POOL = Path("experiments/question_sources/pool")

#: Only candidates in this state go to review. Rejected candidates stay in the
#: pool with their reason, but are not put in front of a reviewer.
REVIEWABLE_STATUS = "validated"


class ExportError(RuntimeError):
    """Raised when the export would be unsafe or incomplete."""


def load_pool(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ExportError(f"candidate pool not found: {path}")
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ExportError(f"{path}:{number}: {exc}") from exc
    return records


def to_questions(records: Sequence[dict[str, Any]]) -> list[EvaluationQuestion]:
    """Rebuild question objects without altering anything.

    Round-tripping through the schema re-runs its validation, so a record that
    lost a required field on the way in or out fails here rather than reaching
    a reviewer as a blank cell.
    """
    questions = []
    for record in records:
        questions.append(EvaluationQuestion(
            question_id=record["question_id"],
            question=record["question"],
            topic=record["topic"],
            subtopic=record.get("subtopic"),
            reference_answer=record["reference_answer"],
            reference_source=record["reference_source"],
            reference_locator=record.get("reference_locator"),
            reference_date=record["reference_date"],
            AD_anchor=bool(record.get("AD_anchor")),
            determinate=bool(record.get("determinate")),
            corpus_support_expected=bool(record.get("corpus_support_expected")),
            temporal_candidate=bool(record.get("temporal_candidate")),
            ambiguity_candidate=bool(record.get("ambiguity_candidate")),
            status=record.get("status", "candidate"),
            rejection_reason=record.get("rejection_reason"),
            metadata=record.get("metadata", {}),
        ))
    return questions


def assert_neutral(rows: Sequence[dict[str, Any]]) -> None:
    """Fail if any withheld classification would reach the reviewer.

    Checks the column names, not a sample: a leak that only appears in some
    rows is still a leak.
    """
    for row in rows:
        leaked = sorted(set(row) & set(WITHHELD_FROM_REVIEW))
        if leaked:
            raise ExportError(
                f"refusing to write a review file exposing internal "
                f"classifications: {leaked}"
            )
        unexpected = sorted(set(row) - set(REVIEW_COLUMNS))
        if unexpected:
            raise ExportError(
                f"unexpected columns in the review export: {unexpected}"
            )


def assert_complete(
    rows: Sequence[dict[str, Any]],
    expected_ids: Sequence[str],
) -> None:
    """Every reviewable candidate appears exactly once, and nothing extra."""
    ids = [row["question_id"] for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ExportError(f"duplicate candidates in the export: {duplicates}")
    missing = sorted(set(expected_ids) - set(ids))
    extra = sorted(set(ids) - set(expected_ids))
    if missing or extra:
        raise ExportError(
            f"export does not match the pool; missing={missing} extra={extra}"
        )


def missing_provenance(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows lacking something a reviewer needs to trace the answer.

    Reported, never filled in: a fabricated locator is worse than an absent
    one, because it looks checkable.
    """
    required = ("reference_source", "reference_date", "source_locator",
                "reference_answer")
    gaps = []
    for row in rows:
        absent = [f for f in required if not str(row.get(f, "")).strip()]
        if absent:
            gaps.append({"question_id": row["question_id"], "missing": absent})
    return gaps


def run(pool_dir: Path = DEFAULT_POOL) -> dict[str, Any]:
    pool_dir = Path(pool_dir)
    records = load_pool(pool_dir / "candidates.jsonl")

    reviewable = [r for r in records if r.get("status") == REVIEWABLE_STATUS]
    questions = to_questions(reviewable)
    rows = review_export(questions)

    assert_neutral(rows)
    assert_complete(rows, [r["question_id"] for r in reviewable])
    gaps = missing_provenance(rows)

    out = pool_dir / "review.csv"
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "pool_records": len(records),
        "reviewable": len(rows),
        "columns": list(REVIEW_COLUMNS),
        "withheld": list(WITHHELD_FROM_REVIEW),
        "missing_provenance": gaps,
        "written": str(out),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.pool_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
