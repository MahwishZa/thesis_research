"""Frozen candidate evidence, and the checks that make the comparison valid.

The experiment's one load-bearing claim is that both systems saw the same
evidence. This module makes that claim checkable rather than assumed: the
candidate set is hashed, the hash travels with the manifest, and a run that
would compare two different candidate sets fails instead of producing a
number.

It also enforces the provenance firewall. The passage used to establish a
question's reference answer must not silently become one of the passages the
system is evaluated on - that is circular, and it is the kind of error that
produces a good-looking result and an indefensible thesis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .questions import EvaluationQuestion


class FreezeError(RuntimeError):
    """Raised when a frozen-evidence invariant is violated."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenCandidate:
    """One passage as it will be replayed to every arm."""

    evidence_id: str
    text: str
    retrieval_rank: int
    retrieval_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    publication_date: Optional[str] = None
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "rerank_rank": self.rerank_rank,
            "rerank_score": self.rerank_score,
            "publication_date": self.publication_date,
            "source_metadata": dict(self.source_metadata),
        }

    def hash_payload(self) -> dict[str, Any]:
        """Exactly the fields that define "the same evidence".

        Ranks and scores are included because they decide which passages
        survive the context budget in every arm; two sets with identical text
        but different ranks are not interchangeable. ``source_metadata`` is
        excluded: it is provenance for the write-up and must not make an
        otherwise identical set compare unequal.
        """
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "rerank_rank": self.rerank_rank,
            "rerank_score": self.rerank_score,
            "publication_date": self.publication_date,
        }


def candidate_set_hash(candidates: Sequence[FrozenCandidate]) -> str:
    """Order-sensitive SHA-256 over the candidate set.

    Order is part of the identity: context position affects what a generator
    attends to, so a reordered set is a different experimental condition even
    when it holds the same passages.
    """
    return _sha256(_canonical([c.hash_payload() for c in candidates]))


def config_hash(config: dict[str, Any]) -> str:
    """SHA-256 over a run configuration, so a silent settings change shows up."""
    return _sha256(_canonical(config))


@dataclass(frozen=True)
class FrozenItem:
    """One evaluation item: a question plus the evidence both arms will see."""

    question_id: str
    question: str
    reference_answer: str
    reference_source: str
    reference_date: str
    candidates: tuple[FrozenCandidate, ...]

    #: Identifies which corpus build these candidates were retrieved from
    #: (e.g. "corpus@2026-10-03"). Required, not folded into
    #: ``metadata``: a manifest frozen against one corpus snapshot is not
    #: comparable to one frozen against another, so the identifier has to be
    #: present on every item, not an optional afterthought.
    corpus_snapshot: str

    evaluation_timestamp: Optional[str] = None
    configuration_hash: Optional[str] = None
    #: Evidence ids that established the reference answer. Firewalled.
    reference_evidence_ids: tuple[str, ...] = ()
    #: Carried from ``EvaluationQuestion.temporal_candidate`` (a Cochrane
    #: review cited at .pub2 or higher - its conclusion has been revisited
    #: at least once). Diagnostic only, exactly as it is at the question
    #: stage: it does not affect admission, generation or scoring. Its only
    #: use is `run_end_to_end.py`'s subgroup breakdown, which asks whether
    #: the Temporal Filter's effect concentrates on questions whose evidence
    #: base has actually been revised over time - the question a temporal
    #: filter exists to answer - rather than being flat across the pool.
    temporal_candidate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise FreezeError(
                f"{self.question_id}: a frozen item needs at least one candidate"
            )
        ids = [c.evidence_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            duplicated = sorted({i for i in ids if ids.count(i) > 1})
            raise FreezeError(
                f"{self.question_id}: duplicate evidence ids {duplicated}"
            )
        if not self.corpus_snapshot.strip():
            raise FreezeError(
                f"{self.question_id}: corpus_snapshot is required - which "
                "corpus build these candidates came from must be traceable "
                "before evidence is frozen"
            )

    @property
    def candidate_set_hash(self) -> str:
        return candidate_set_hash(self.candidates)

    @property
    def candidate_evidence_ids(self) -> tuple[str, ...]:
        return tuple(c.evidence_id for c in self.candidates)

    def firewall_violations(self) -> tuple[str, ...]:
        """Reference evidence that also appears among the candidates.

        Non-empty means the system is being asked a question whose answer was
        derived from a passage it is being shown - the reference is no longer
        independent of the evaluation.
        """
        return tuple(
            sorted(set(self.reference_evidence_ids)
                   & set(self.candidate_evidence_ids))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "reference_source": self.reference_source,
            "reference_date": self.reference_date,
            "corpus_snapshot": self.corpus_snapshot,
            "candidate_evidence_ids": list(self.candidate_evidence_ids),
            "candidate_evidence": [c.to_dict() for c in self.candidates],
            "candidate_set_hash": self.candidate_set_hash,
            "evaluation_timestamp": self.evaluation_timestamp,
            "configuration_hash": self.configuration_hash,
            "reference_evidence_ids": list(self.reference_evidence_ids),
            "metadata": dict(self.metadata),
        }


def assert_firewall(items: Sequence[FrozenItem]) -> None:
    """Fail the run if any item's reference evidence is also candidate evidence."""
    breaches = {
        item.question_id: item.firewall_violations()
        for item in items
        if item.firewall_violations()
    }
    if breaches:
        raise FreezeError(
            "provenance firewall breached - reference evidence appears in the "
            f"candidate set for: {breaches}"
        )


def assert_same_candidate_sets(
    baseline: dict[str, str],
    proposed: dict[str, str],
) -> None:
    """Fail unless both arms were given byte-identical candidate sets.

    Takes {question_id: candidate_set_hash} per arm. Raises on any missing,
    extra or differing item. There is deliberately no tolerance and no warning
    path: a mismatch invalidates the comparison, so the run must stop.
    """
    missing = sorted(set(baseline) - set(proposed))
    extra = sorted(set(proposed) - set(baseline))
    differing = sorted(
        qid for qid in set(baseline) & set(proposed)
        if baseline[qid] != proposed[qid]
    )
    if missing or extra or differing:
        raise FreezeError(
            "candidate-set mismatch between arms; the comparison is invalid. "
            f"missing_from_proposed={missing} extra_in_proposed={extra} "
            f"differing_hashes={differing}"
        )


def write_manifest(items: Sequence[FrozenItem], path: str) -> str:
    """Write the frozen manifest as JSONL and return its file digest."""
    assert_firewall(items)
    payload = "".join(
        json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        for item in items
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return _sha256(payload)


def read_manifest(path: str) -> tuple[FrozenItem, ...]:
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            items.append(FrozenItem(
                question_id=record["question_id"],
                question=record["question"],
                reference_answer=record["reference_answer"],
                reference_source=record["reference_source"],
                reference_date=record["reference_date"],
                corpus_snapshot=record["corpus_snapshot"],
                candidates=tuple(
                    FrozenCandidate(**c) for c in record["candidate_evidence"]
                ),
                evaluation_timestamp=record.get("evaluation_timestamp"),
                configuration_hash=record.get("configuration_hash"),
                reference_evidence_ids=tuple(
                    record.get("reference_evidence_ids", ())
                ),
                metadata=record.get("metadata", {}),
            ))
    return tuple(items)


def verify_manifest(path: str, expected_digest: str) -> tuple[FrozenItem, ...]:
    """Read a manifest only if its bytes still match the recorded digest."""
    with open(path, encoding="utf-8") as handle:
        actual = _sha256(handle.read())
    if actual != expected_digest:
        raise FreezeError(
            f"frozen manifest changed since it was recorded: {path}\n"
            f"  expected {expected_digest}\n  actual   {actual}"
        )
    return read_manifest(path)


def from_question(
    question: EvaluationQuestion,
    candidates: Sequence[FrozenCandidate],
    *,
    corpus_snapshot: str,
    reference_evidence_ids: Sequence[str] = (),
    evaluation_timestamp: Optional[str] = None,
    configuration_hash: Optional[str] = None,
) -> FrozenItem:
    """Build a ``FrozenItem`` from an approved question plus its evidence.

    This is the one place Step 2's output (an ``EvaluationQuestion``) meets
    Step 3's input (a ``FrozenItem``). Copying the shared fields by hand at
    each call site is how they drift - a stale ``reference_date`` surviving a
    question revision, for instance - so the copy happens once, here, from
    the question's own fields.

    Only ``question.is_usable`` questions should reach this function; that is
    a caller-side check (it depends on the review outcome, which this module
    has no reason to know about), not enforced here.
    """
    return FrozenItem(
        question_id=question.question_id,
        question=question.question,
        reference_answer=question.reference_answer,
        reference_source=question.reference_source,
        reference_date=question.reference_date,
        corpus_snapshot=corpus_snapshot,
        candidates=tuple(candidates),
        reference_evidence_ids=tuple(reference_evidence_ids),
        temporal_candidate=question.temporal_candidate,
        evaluation_timestamp=evaluation_timestamp,
        configuration_hash=configuration_hash,
    )
