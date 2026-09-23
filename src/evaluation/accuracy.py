"""QA accuracy judgement schema.

This module is the human-judged correctness track (secondary confirmation
under the current objectives - see _archive/docs_legacy/current_objectives.md; it was
primary under an earlier, superseded scope). It deliberately does not
decide what "correct" means - it does not run exact match, fuzzy match, or
any automatic scorer against the reference answer. Automatic string-overlap
scoring (ROUGE/BLEU/BERTScore and similar) belongs in
``src/evaluation/rag_metrics.py``, the current objectives' ablation
study metrics track, not here - inventing a second, bespoke automatic judge
in this module would duplicate that with an undeclared scoring methodology.

Instead, ``correct`` is supplied by whatever the thesis's QA protocol decides
- a human judge reading the generated answer against the reference, or an
exact-match rule for closed-form questions where one is appropriate - and
this module only gives that judgement a validated, traceable shape. The
aggregation into totals and accuracy lives in ``stats.qa_accuracy``, next to
the hallucination-rate aggregation it mirrors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional


class AccuracyError(ValueError):
    """Raised when a QA judgement record is not usable."""


@dataclass(frozen=True)
class QAJudgment:
    """One correctness judgement for one system's answer to one question."""

    question_id: str
    system: str
    question: str
    generated_answer: str
    reference_answer: str
    correct: int
    #: Who or what decided correctness - an annotator id, or a named rule
    #: such as "exact_match". Never left implicit: a reader must be able to
    #: tell a human judgement from a rule-based one.
    judge: str
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.correct not in (0, 1):
            raise AccuracyError("correct must be 0 or 1")
        if not self.judge.strip():
            raise AccuracyError(
                "judge is required - a correctness judgement must say who "
                "or what made it"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "system": self.system,
            "question": self.question,
            "generated_answer": self.generated_answer,
            "reference_answer": self.reference_answer,
            "correct": self.correct,
            "judge": self.judge,
            "note": self.note,
        }


def read_judgments(path: str) -> tuple[QAJudgment, ...]:
    """Read completed QA judgements back, validated the same way they wrote."""
    judgments = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            judgments.append(QAJudgment(
                question_id=row["question_id"],
                system=row["system"],
                question=row["question"],
                generated_answer=row["generated_answer"],
                reference_answer=row["reference_answer"],
                correct=row["correct"],
                judge=row["judge"],
                note=row.get("note"),
            ))
    return tuple(judgments)


def write_judgments(judgments: Iterable[QAJudgment], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for judgment in judgments:
            handle.write(json.dumps(judgment.to_dict(), sort_keys=True,
                                    ensure_ascii=False) + "\n")


def by_question(
    judgments: Iterable[QAJudgment],
) -> dict[str, dict[str, QAJudgment]]:
    """Reshape into ``{system: {question_id: QAJudgment}}``.

    Mirrors ``runner.group_by_system``: the same shape every downstream
    comparison expects, so accuracy and hallucination results line up on the
    same key without separate glue code at each call site.
    """
    grouped: dict[str, dict[str, QAJudgment]] = {}
    for judgment in judgments:
        by_q = grouped.setdefault(judgment.system, {})
        if judgment.question_id in by_q:
            raise AccuracyError(
                f"duplicate judgment for system={judgment.system!r} "
                f"question_id={judgment.question_id!r}"
            )
        by_q[judgment.question_id] = judgment
    return grouped
