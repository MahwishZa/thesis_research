"""Hallucination annotation: schema, blinding, and agreement.

The primary label is deliberately coarse. An answer is hallucinated when it
contains at least one claim unsupported by, or contradicted by, the evidence
that was actually supplied to the system that produced it. That is a
reading-comprehension judgement against a fixed passage set, not a clinical
one, which is what makes a single medical student a defensible primary
annotator.

Blinding is not optional here. An annotator who can see which system produced
an answer cannot produce an unbiased label, and the primary outcome of the
thesis is that label.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

SUBTYPES = ("faithfulness", "factuality", "temporal",
            "misinterpretation", "ambiguity", "other")


class AnnotationError(ValueError):
    """Raised when an annotation record is not usable."""


@dataclass(frozen=True)
class Annotation:
    """One annotator's judgement of one answer."""

    answer_key: str
    annotator_id: str
    hallucinated: int
    n_hallucinated_claims: int = 0
    subtype: Optional[str] = None
    abstained: int = 0
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.hallucinated not in (0, 1):
            raise AnnotationError("hallucinated must be 0 or 1")
        if self.abstained not in (0, 1):
            raise AnnotationError("abstained must be 0 or 1")
        if self.n_hallucinated_claims < 0:
            raise AnnotationError("n_hallucinated_claims cannot be negative")
        if self.hallucinated == 1 and self.n_hallucinated_claims < 1:
            raise AnnotationError(
                "an answer labelled hallucinated must have at least one "
                "hallucinated claim"
            )
        if self.hallucinated == 0 and self.n_hallucinated_claims:
            raise AnnotationError(
                "claims were counted on an answer labelled not hallucinated"
            )
        if self.subtype is not None and self.subtype not in SUBTYPES:
            raise AnnotationError(
                f"unknown subtype {self.subtype!r}; expected one of {SUBTYPES}"
            )
        if self.hallucinated == 1 and self.subtype is None:
            raise AnnotationError("a hallucinated answer needs a subtype")
        if self.abstained == 1 and self.hallucinated == 1:
            raise AnnotationError(
                "an abstention makes no claims and cannot be hallucinated"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_key": self.answer_key,
            "annotator_id": self.annotator_id,
            "hallucinated": self.hallucinated,
            "n_hallucinated_claims": self.n_hallucinated_claims,
            "subtype": self.subtype,
            "abstained": self.abstained,
            "note": self.note,
        }


def _answer_key(run_id: str, question_id: str, system: str) -> str:
    digest = hashlib.sha256(
        f"{run_id}\x00{question_id}\x00{system}".encode()
    ).hexdigest()
    return f"ANS-{digest[:16]}"


def build_blinded_packet(
    answers: Sequence[dict[str, Any]],
    *,
    seed: str,
    label_map: Sequence[str] = ("System A", "System B"),
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Split answers into an annotator packet and a separate unblinding key.

    Returns ``(packet, key)``. The packet carries no system identity; the key
    maps each opaque answer key back to its system and question. They are
    returned separately so the caller can store the key somewhere the
    annotator does not read.

    Order is shuffled under ``seed`` so the run is reproducible without being
    predictable to the annotator.
    """
    systems = sorted({a["system"] for a in answers})
    if len(systems) > len(label_map):
        raise AnnotationError(
            f"{len(systems)} systems but only {len(label_map)} blind labels"
        )
    blinded = {s: label_map[i] for i, s in enumerate(systems)}

    packet, key = [], {}
    for answer in answers:
        akey = _answer_key(answer["run_id"], answer["question_id"],
                           answer["system"])
        packet.append({
            "answer_key": akey,
            "question": answer["question"],
            "evidence": answer.get("admitted_evidence_text", []),
            "answer": answer.get("generated_answer"),
            "shown_as": blinded[answer["system"]],
            "hallucinated": "",
            "n_hallucinated_claims": "",
            "subtype": "",
            "abstained": "",
            "note": "",
        })
        key[akey] = {
            "system": answer["system"],
            "question_id": answer["question_id"],
            "run_id": answer["run_id"],
            "shown_as": blinded[answer["system"]],
        }

    random.Random(seed).shuffle(packet)
    return packet, key


def sample_for_second_annotator(
    answer_keys: Sequence[str],
    *,
    fraction: float,
    seed: str,
) -> tuple[str, ...]:
    """A reproducible random subset for the independent reviewer.

    Random rather than chosen, so the agreement estimate is not computed on a
    subset selected for being easy or interesting.
    """
    if not 0 < fraction <= 1:
        raise AnnotationError("fraction must be in (0, 1]")
    ordered = sorted(answer_keys)
    n = max(1, round(len(ordered) * fraction))
    return tuple(sorted(random.Random(seed).sample(ordered, n)))


def agreement(
    first: Sequence[Annotation],
    second: Sequence[Annotation],
) -> dict[str, Any]:
    """Raw agreement and Cohen's kappa on the binary hallucination label.

    Computed only over answers both annotators labelled. Returns ``kappa:
    None`` when it is undefined rather than substituting a number - which
    happens when one annotator used a single label throughout, a real and
    reportable situation.
    """
    a = {x.answer_key: x.hallucinated for x in first}
    b = {x.answer_key: x.hallucinated for x in second}
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if not n:
        return {"n": 0, "raw_agreement": None, "kappa": None,
                "note": "no overlapping annotations"}

    observed = sum(1 for k in shared if a[k] == b[k]) / n
    pa1 = sum(a[k] for k in shared) / n
    pb1 = sum(b[k] for k in shared) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)

    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {
        "n": n,
        "raw_agreement": round(observed, 4),
        "kappa": None if kappa is None else round(kappa, 4),
        "disagreements": [k for k in shared if a[k] != b[k]],
        "note": None if kappa is not None else
                "kappa undefined: no expected disagreement",
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True,
                                    ensure_ascii=False) + "\n")


def read_annotations(path: str) -> tuple[Annotation, ...]:
    """Read a completed annotation file back into validated records.

    Each row is passed through ``Annotation``'s own validation, so a
    malformed row - a hallucinated answer with no subtype, an abstention
    marked hallucinated - is refused here rather than reaching statistics.
    """
    annotations = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            annotations.append(Annotation(
                answer_key=row["answer_key"],
                annotator_id=row["annotator_id"],
                hallucinated=row["hallucinated"],
                n_hallucinated_claims=row.get("n_hallucinated_claims", 0),
                subtype=row.get("subtype"),
                abstained=row.get("abstained", 0),
                note=row.get("note"),
            ))
    return tuple(annotations)


def unblind_annotations(
    annotations: Sequence[Annotation],
    key: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Annotation]]:
    """Recover ``{system: {question_id: Annotation}}`` after review.

    ``key`` is the unblinding map ``build_blinded_packet`` returned
    separately from the packet the annotator saw. This is the only place
    annotation identity and system identity are brought back together, and it
    happens after annotation is complete - never before, and never inside the
    file an annotator reads.
    """
    grouped: dict[str, dict[str, Annotation]] = {}
    for annotation in annotations:
        mapping = key.get(annotation.answer_key)
        if mapping is None:
            raise AnnotationError(
                f"answer_key {annotation.answer_key!r} is not in the "
                "unblinding key"
            )
        system = mapping["system"]
        question_id = mapping["question_id"]
        by_question = grouped.setdefault(system, {})
        if question_id in by_question:
            raise AnnotationError(
                f"duplicate annotation for system={system!r} "
                f"question_id={question_id!r}"
            )
        by_question[question_id] = annotation
    return grouped
