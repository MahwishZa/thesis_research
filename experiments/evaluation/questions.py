"""Evaluation-question records and their validation.

A question enters this module from an authoritative Alzheimer's source and
leaves it only after a human has approved it. The schema exists to make the
two failure modes visible: a question with no traceable reference, and a
question that duplicates one already in the set.

Nothing here generates a reference answer. A reference answer is transcribed
from a named source with a date, because an answer an LLM invented for a
question the same LLM invented is not evidence of anything.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional, Sequence

STATUSES = (
    "candidate",
    "validated",
    "human_reviewed",
    "approved",
    "evidence_pending",
    "evidence_validated",
    "final",
    "rejected",
)

#: Statuses a question may hold and still reach the experiment.
USABLE_STATUSES = ("approved", "evidence_pending", "evidence_validated", "final")

_DATE = re.compile(r"^\d{4}(-\d{2}){0,2}$")
_WORD = re.compile(r"[a-z0-9]+")


class QuestionError(ValueError):
    """Raised when a record cannot represent an evaluation question."""


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Used only for duplicate detection, never for storage: the stored question
    keeps its original wording.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_WORD.findall(stripped))


def token_set(text: str) -> frozenset[str]:
    return frozenset(normalise(text).split())


def jaccard(a: str, b: str) -> float:
    left, right = token_set(a), token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def question_id(question_text: str, reference_source: str) -> str:
    """Stable id from the question and its source.

    Deriving it from content rather than a counter means re-running generation
    over the same inputs reproduces the same ids, so a partially reviewed set
    survives a rebuild.
    """
    digest = hashlib.sha256(
        (normalise(question_text) + "\x00" + reference_source.strip()).encode()
    ).hexdigest()
    return f"ADQ-{digest[:12]}"


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    topic: str
    reference_answer: str
    reference_source: str
    reference_date: str

    subtopic: Optional[str] = None
    question_id: Optional[str] = None

    AD_anchor: bool = False
    determinate: bool = False
    corpus_support_expected: bool = True
    temporal_candidate: bool = False
    #: Flagged as potentially ambiguous. Ambiguity is a cause of
    #: hallucination, so these are kept and tagged rather than discarded -
    #: dropping them would remove the cases most likely to expose the
    #: behaviour under study.
    ambiguity_candidate: bool = False

    status: str = "candidate"

    #: Free-text reason, set when status is "rejected".
    rejection_reason: Optional[str] = None
    #: Where in the source the reference answer came from (section, page, DOI).
    reference_locator: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("question", "topic", "reference_answer",
                     "reference_source", "reference_date"):
            if not str(getattr(self, name) or "").strip():
                raise QuestionError(f"{name} is required and must be non-empty")

        if self.status not in STATUSES:
            raise QuestionError(
                f"unknown status {self.status!r}; expected one of {STATUSES}"
            )

        if not _DATE.match(self.reference_date):
            raise QuestionError(
                f"reference_date must be YYYY[-MM[-DD]], got "
                f"{self.reference_date!r}"
            )

        if self.status == "rejected" and not self.rejection_reason:
            raise QuestionError("a rejected question must carry a reason")

        if self.question_id is None:
            object.__setattr__(
                self, "question_id",
                question_id(self.question, self.reference_source),
            )

    @property
    def is_usable(self) -> bool:
        return self.status in USABLE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "reference_answer": self.reference_answer,
            "reference_source": self.reference_source,
            "reference_locator": self.reference_locator,
            "reference_date": self.reference_date,
            "AD_anchor": self.AD_anchor,
            "determinate": self.determinate,
            "corpus_support_expected": self.corpus_support_expected,
            "temporal_candidate": self.temporal_candidate,
            "ambiguity_candidate": self.ambiguity_candidate,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "metadata": dict(self.metadata),
        }


def validation_failures(question: EvaluationQuestion) -> tuple[str, ...]:
    """Automatic checks. Passing them makes a question reviewable, not final.

    A human still has to approve it; these only catch what a machine can.
    """
    failures = []
    if not question.AD_anchor:
        failures.append("not_ad_anchored")
    if not question.determinate:
        failures.append("not_determinate")
    if len(token_set(question.question)) < 4:
        failures.append("question_too_short")
    if not question.question.strip().endswith("?"):
        failures.append("not_interrogative")
    if question.reference_source.strip().lower().startswith("http") and \
            len(question.reference_source.strip()) < 12:
        failures.append("reference_source_not_identifiable")
    if len(token_set(question.reference_answer)) < 2:
        failures.append("reference_answer_too_thin")
    if not question.corpus_support_expected:
        failures.append("no_corpus_support_expected")
    return tuple(failures)


def validate(question: EvaluationQuestion) -> EvaluationQuestion:
    """Promote to ``validated`` when the automatic checks pass.

    Failures are recorded rather than raised: the reviewer needs to see why a
    candidate did not advance.
    """
    failures = validation_failures(question)
    if failures:
        return replace(
            question,
            status="rejected",
            rejection_reason="; ".join(failures),
        )
    return replace(question, status="validated")


def find_duplicates(
    questions: Sequence[EvaluationQuestion],
    *,
    threshold: float = 0.85,
) -> tuple[tuple[str, str, float], ...]:
    """Exact and near-duplicate pairs, as (id_a, id_b, similarity).

    Exact duplicates report 1.0. Near duplicates use Jaccard over normalised
    tokens - deliberately crude and deterministic, because a reviewer confirms
    every pair and an embedding model would make the decision unreproducible.
    """
    found = []
    for i, a in enumerate(questions):
        for b in questions[i + 1:]:
            score = jaccard(a.question, b.question)
            if score >= threshold:
                found.append((a.question_id, b.question_id, round(score, 4)))
    return tuple(found)


def review_export(questions: Iterable[EvaluationQuestion]) -> list[dict[str, Any]]:
    """Rows for human review, with the decision columns left blank.

    The reviewer fills ``reviewer_decision`` and ``reviewer_note``; nothing
    else in the row should change.
    """
    rows = []
    for q in questions:
        row = q.to_dict()
        row["automatic_failures"] = "; ".join(validation_failures(q))
        row["reviewer_decision"] = ""
        row["reviewer_note"] = ""
        rows.append(row)
    return rows


#: Wording that usually signals a question with no determinate answer.
_INDETERMINATE = (
    "what do you think", "in your opinion", "how do you feel",
    "should i", "what would you", "is it better",
)


def looks_indeterminate(text: str) -> bool:
    """Cheap answerability screen, for triage only.

    A hit means a reviewer should look, not that the question is unusable. It
    is deliberately a keyword list rather than a model: a reviewer confirms
    every flag, and a classifier here would make the evaluation set depend on
    an unreproducible judgement.
    """
    lowered = " " + normalise(text) + " "
    return any(f" {p} " in lowered for p in (normalise(x) for x in _INDETERMINATE))


def summarise_pool(questions: Sequence[EvaluationQuestion]) -> dict[str, Any]:
    """Counts a reviewer needs before deciding the pool is large enough.

    Reports what is there; it does not decide a sample size. The ~100-question
    target is a practical budget, not a powered calculation, and the thesis
    should say so.
    """
    by_status: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    for q in questions:
        by_status[q.status] = by_status.get(q.status, 0) + 1
        by_topic[q.topic] = by_topic.get(q.topic, 0) + 1
    usable = [q for q in questions if q.is_usable]
    return {
        "total": len(questions),
        "by_status": dict(sorted(by_status.items())),
        "by_topic": dict(sorted(by_topic.items(), key=lambda kv: -kv[1])),
        "usable": len(usable),
        "ad_anchored": sum(1 for q in questions if q.AD_anchor),
        "determinate": sum(1 for q in questions if q.determinate),
        "temporal_candidates": sum(1 for q in questions if q.temporal_candidate),
        "ambiguity_candidates": sum(1 for q in questions
                                    if q.ambiguity_candidate),
        "duplicate_pairs": len(find_duplicates(questions)),
        "distinct_sources": len({q.reference_source for q in questions}),
    }
