"""From rationale to retrieval query: the rationale alone is the query (paper, section 3.3).

The question is not added: the paper says both together exceed the retriever's maximum length and gave worse results in the
authors' tests. The query is the rationale as the model wrote it, with only the whitespace at its two ends removed. The
authors' retriever later cuts it to 512 tokens when it encodes it (``query_encode.py``); that belongs to the next component,
so nothing is shortened here. The retriever reads a JSON list of query strings, in the order of its questions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .store import Question, RationaleRecord, StoreError, question_hash, write_atomic


def rationale_to_query(rationale: str) -> str:
    """The retrieval query for one rationale. An empty rationale is an error; it is never replaced by the question."""
    if not isinstance(rationale, str):
        raise TypeError(f"the rationale must be a string, got {type(rationale).__name__}")
    query = rationale.strip()
    if not query:
        raise ValueError("the rationale is empty")
    return query


def queries_for(questions: Sequence[Question], records: Mapping[str, RationaleRecord]) -> list[str]:
    """One query per question, in the order of ``questions``. Every question needs a finished rationale for its own text."""
    problems: list[str] = []
    queries: list[str] = []
    for question in questions:
        record = records.get(question.id)
        if record is None:
            problems.append(f"{question.id}: no rationale yet")
        elif record.status != "ok":
            problems.append(f"{question.id}: {record.error or 'failed'}")
        elif record.question_sha256 != question_hash(question.text):
            problems.append(f"{question.id}: the rationale was written for another text than this question")
        else:
            queries.append(rationale_to_query(record.rationale or ""))
    if problems:
        more = f" ... ({len(problems)} problems in all)" if len(problems) > 5 else ""
        raise StoreError(f"not every question has a usable rationale: {'; '.join(problems[:5])}{more}")
    return queries


def export_queries(queries: Sequence[str], path: str | Path) -> None:
    """Write the queries as a JSON list for the retriever. An existing file is never overwritten.

    Non-ASCII characters are written as escapes, so that the authors' script (which opens the file with the computer's own
    encoding) reads the same text on any computer.
    """
    target = Path(path)
    if target.exists():
        raise StoreError(f"{target} exists; this command does not overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(target, json.dumps(list(queries), ensure_ascii=True, indent=2) + "\n")
