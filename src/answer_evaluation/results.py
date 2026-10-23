"""Accuracy per system, the paired comparison of two systems, and the choice of k on validation data (decision D13).

Only ``ok`` records count. The unreadable answers are inside the accuracy as wrong answers (decision D16) and are also counted on
their own, so that the share can be reported. Two systems can be compared only on the same questions; a question that one system
failed to answer (an error record) makes the comparison refuse and name the problem instead of dropping the question quietly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .paired_tests import StatsError, compare
from .records import AnswerRecord


def outcomes(records: Iterable[AnswerRecord], arm: str, k: int) -> dict[str, int]:
    """``{question id: 1 or 0}`` for one system, from its ``ok`` records."""
    return {r.question_id: int(bool(r.correct)) for r in records if r.status == "ok" and r.arm == arm and r.k == k}


def accuracy_table(records: Iterable[AnswerRecord]) -> list[dict[str, Any]]:
    """One row per system: how many questions, how many correct, the accuracy and the unreadable answers. Sorted by arm and k."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if record.status != "ok":
            continue
        row = rows.setdefault((record.arm, record.k), {"arm": record.arm, "k": record.k, "n": 0, "correct": 0, "unreadable": 0})
        row["n"] += 1
        row["correct"] += int(bool(record.correct))
        row["unreadable"] += int(record.letter is None)
    for row in rows.values():
        row["accuracy"] = round(row["correct"] / row["n"], 4)
    return [rows[key] for key in sorted(rows)]


def compare_systems(
    records: Iterable[AnswerRecord], baseline: tuple[str, int], proposed: tuple[str, int], *, seed: str = "bootstrap", iterations: int = 10000
) -> dict[str, Any]:
    """The paired comparison of two systems. Raises ``StatsError`` unless both answered exactly the same questions."""
    records = list(records)
    first, second = outcomes(records, *baseline), outcomes(records, *proposed)
    if not first or not second:
        raise StatsError(f"no answers for {baseline if not first else proposed}")
    return compare(first, second, seed=seed, iterations=iterations)


def best_k(records: Iterable[AnswerRecord], arm: str, candidates: Iterable[int]) -> dict[str, Any]:
    """The k with the highest accuracy for ``arm`` among ``candidates``; on a tie the smaller k. Use validation records only.

    Every candidate must have answered the same questions, or the accuracies are not comparable and ``StatsError`` is raised.
    """
    records = list(records)
    table: dict[int, Mapping[str, int]] = {k: outcomes(records, arm, k) for k in candidates}
    if not table:
        raise StatsError("no candidate k")
    empty = [k for k, found in table.items() if not found]
    if empty:
        raise StatsError(f"no answers for {arm} at k = {empty}")
    question_sets = {frozenset(found) for found in table.values()}
    if len(question_sets) != 1:
        raise StatsError(f"the values of k did not answer the same questions for {arm}")
    scores = {k: sum(found.values()) / len(found) for k, found in table.items()}
    chosen = min(scores, key=lambda k: (-scores[k], k))
    return {"arm": arm, "k": chosen, "accuracy": round(scores[chosen], 4), "n": len(table[chosen]), "all": {k: round(v, 4) for k, v in sorted(scores.items())}}
