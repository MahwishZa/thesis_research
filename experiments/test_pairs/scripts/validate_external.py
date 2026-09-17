"""Check a prepared external evaluation file before it is used.

This is the acquisition boundary. It answers one question - *is this file
usable, and what will it cost?* - before any pairs are built, so a dataset
problem shows up as a report rather than as a surprising attrition table
three steps later.

It deliberately does **not** convert MedChangeQA's own distribution format.
Nobody here has seen that format, and a converter written against a guessed
schema would fail silently on the real thing or, worse, quietly mis-assign
which side is newer. Preparing the JSONL is a small manual step, documented
in ``docs/external_evaluation_data.md``; this validator is what
makes that step verifiable.

It records the raw file's SHA-256 and the dataset name, so the evaluation set
can be traced back to the exact bytes it came from (specification 50).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from .build_pairs import EXTERNAL_FIELDS, EXTERNAL_OPTIONAL_FIELDS, _read_jsonl


#: Fields whose absence does not stop the run but costs items. Reported with
#: the exclusion each one would trigger, so the cost is visible in advance.
COST_OF_ABSENCE = {
    "older_text": "missing_evidence_text",
    "newer_text": "missing_evidence_text",
    "older_publication_date": "missing_publication_date",
    "newer_publication_date": "missing_publication_date",
    "question_date": "falls back to evaluation_as_of_date",
    "change_point_date": "Check A not interpretable for this item",
}


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Field coverage and the exclusions the gaps would cause."""

    total = len(records)

    present: Counter[str] = Counter()
    for record in records:
        for field in EXTERNAL_FIELDS + EXTERNAL_OPTIONAL_FIELDS:
            if str(record.get(field, "")).strip():
                present[field] += 1

    missing_required = {
        field: total - present[field]
        for field in EXTERNAL_FIELDS
        if present[field] < total
    }

    anticipated: dict[str, str] = {}
    for field, consequence in COST_OF_ABSENCE.items():
        absent = total - present[field]
        if absent:
            anticipated[field] = (
                f"{absent}/{total} absent -> {consequence}"
            )

    duplicate_questions = [
        question
        for question, count in Counter(
            str(record.get("question_id", "")) for record in records
        ).items()
        if count > 1 and question
    ]

    return {
        "records": total,
        "field_coverage": {
            field: present[field]
            for field in EXTERNAL_FIELDS + EXTERNAL_OPTIONAL_FIELDS
        },
        "missing_required": missing_required,
        "anticipated_exclusions": anticipated,
        "duplicate_question_ids": sorted(duplicate_questions)[:20],
        "usable": not missing_required,
    }


def run(
    *,
    input_path: Path,
    dataset_name: str,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:

    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"External evaluation file not found: {input_path}\n"
            "It is not generated here. See "
            "docs/external_evaluation_data.md for what to "
            "acquire and the JSONL contract to convert it to."
        )

    records = _read_jsonl(Path(input_path))

    record = {
        "dataset": dataset_name,
        "input_path": str(input_path),
        "sha256": file_digest(Path(input_path)),
        **audit(records),
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a prepared external evaluation file.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)

    report = run(
        input_path=args.input,
        dataset_name=args.dataset,
        output_path=args.output,
    )

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    if not report["usable"]:
        sys.stderr.write(
            "\nFile is NOT usable: required fields are missing from some "
            "records. Fix the conversion before building pairs; the builder "
            "will refuse the file rather than fill the gaps.\n"
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
