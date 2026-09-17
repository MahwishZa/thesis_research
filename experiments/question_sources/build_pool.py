"""Build the candidate question pool from the inspected sources.

Deterministic: same inputs, same pool, same ids. Nothing is marked final here
- every record leaves as ``candidate`` or ``rejected``, and a human decides
what reaches the evaluation set.

    python -m experiments.question_sources.build_pool \
        --medrevqa path/to/DS_MedRevQA.csv \
        --medquad path/to/MedQuAD \
        --retrieved-on 2026-09
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from dataclasses import replace as _replace
from typing import Any, Optional, Sequence

from ..evaluation.questions import (
    REVIEW_COLUMNS, EvaluationQuestion, find_duplicates, looks_indeterminate,
    review_export, summarise_pool, validate, validation_failures,
)
from .export_review import assert_neutral
from . import cochrane, nih_medquad

DEFAULT_OUT = Path("experiments/question_sources/pool")


def collect(
    medrevqa: Optional[Path],
    medquad: Optional[Path],
    retrieved_on: str,
) -> list[EvaluationQuestion]:
    pool: list[EvaluationQuestion] = []
    if medrevqa:
        pool.extend(cochrane.candidates(Path(medrevqa)))
    if medquad:
        pool.extend(nih_medquad.candidates(Path(medquad),
                                           retrieved_on=retrieved_on))
    return pool


def screen(
    pool: Sequence[EvaluationQuestion],
) -> tuple[list[EvaluationQuestion], dict[str, Any]]:
    """Validate, flag indeterminate wording, and drop duplicates.

    Duplicate handling keeps the first occurrence and rejects the rest with a
    reason, rather than deleting them: a reviewer can see what collided and
    why, and the counts still add up.
    """
    seen_ids: set[str] = set()
    screened: list[EvaluationQuestion] = []
    id_collisions = 0

    for question in pool:
        if question.question_id in seen_ids:
            id_collisions += 1
            continue
        seen_ids.add(question.question_id)
        screened.append(question)

    duplicates = find_duplicates(screened, threshold=0.85)
    drop_after_first = {b for _, b, _ in duplicates}

    final: list[EvaluationQuestion] = []
    for question in screened:
        question = validate(question)
        if question.status == "rejected":
            final.append(question)
            continue
        if question.question_id in drop_after_first:
            question = _replace(
                question,
                status="rejected",
                rejection_reason="near_duplicate_of_earlier_candidate",
            )
        elif looks_indeterminate(question.question):
            question = _replace(question, ambiguity_candidate=True)
        final.append(question)

    stats = {
        "sourced": len(pool),
        "identical_id_collisions_dropped": id_collisions,
        "near_duplicate_pairs": len(duplicates),
        "auto_rejected": sum(1 for q in final if q.status == "rejected"),
        "auto_validated": sum(1 for q in final if q.status == "validated"),
        "rejection_reasons": _reason_counts(final),
    }
    return final, stats


def _reason_counts(pool: Sequence[EvaluationQuestion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in pool:
        if question.status != "rejected" or not question.rejection_reason:
            continue
        for reason in question.rejection_reason.split("; "):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def write_pool(pool: Sequence[EvaluationQuestion], out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl = out_dir / "candidates.jsonl"
    with open(jsonl, "w", encoding="utf-8") as handle:
        for question in pool:
            handle.write(json.dumps(question.to_dict(), sort_keys=True,
                                    ensure_ascii=False) + "\n")

    reviewable = [q for q in pool if q.status == "validated"]
    rows = review_export(reviewable)
    # Same neutrality gate the standalone exporter applies: the reviewer file
    # must not carry a classification this pipeline assigned.
    assert_neutral(rows)

    review = out_dir / "review.csv"
    with open(review, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    return {"candidates": str(jsonl), "review": str(review)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medrevqa", type=Path, default=None)
    parser.add_argument("--medquad", type=Path, default=None)
    parser.add_argument("--retrieved-on", default="2026-09")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.medrevqa and not args.medquad:
        parser.error("at least one source path is required")

    pool = collect(args.medrevqa, args.medquad, args.retrieved_on)
    pool, screen_stats = screen(pool)
    paths = write_pool(pool, args.out_dir)

    summary = {
        **screen_stats,
        **summarise_pool(pool),
        "source_types": _source_types(pool),
        "written": paths,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _source_types(pool: Sequence[EvaluationQuestion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in pool:
        key = str(question.metadata.get("reference_source_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
