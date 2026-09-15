"""Stage-2 test-pair data model.

A test pair is one clinical question, two evidence passages taking opposing
positions on it, a reference answer as of a stated question date, and enough
temporal metadata to ask whether the newer or the older passage was admitted.

Three properties of this schema are load-bearing rather than decorative:

1.  **Provenance is a hard partition, not a field to filter on later.**
    Specification section 33.2 and ledger decision D-2 require the primary
    evaluation claim to rest on externally-authored material. The two pools
    are therefore written to separate files, and :func:`assert_firewall`
    refuses a collection containing both. Pooling them is the failure mode
    the firewall exists to prevent, and it is silent if it is only a column.

2.  **Dates are intervals, not points.** A publication date of ``"2024"``
    means some day in 2024. Stage 2 has to decide whether the older passage
    is *unambiguously* older, and a point estimate cannot answer that. Each
    date therefore carries the earliest and latest day consistent with its
    stated precision (see :class:`EvidenceRef`).

3.  **Every label records where it came from.** ``claim_class_source``,
    ``contradiction_source`` and ``question_date_source`` exist because the
    corpus currently carries automatic keyword labels that must never be read
    as ground truth (ledger A8).

Serialisation is JSONL with sorted keys, so two runs over identical inputs
produce byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


#: The two evaluation pools. They are never merged into one file.
POOL_PRIMARY_EXTERNAL = "primary_external"
POOL_SECONDARY_CURATED = "secondary_curated"

POOLS = (POOL_PRIMARY_EXTERNAL, POOL_SECONDARY_CURATED)


#: Where a passage's text comes from. ``external_item`` passages travel with
#: the evaluation item itself (the two systematic-review records the external
#: dataset is built from) and are injected into the Stage-3 candidate set.
#: ``corpus`` passages are references into the frozen Alzheimer's corpus.
#: Both carry stable ids; the distinction is recorded so a reviewer can see
#: which passages the retriever had to find and which were placed there.
ORIGIN_EXTERNAL_ITEM = "external_item"
ORIGIN_CORPUS = "corpus"

EVIDENCE_ORIGINS = (ORIGIN_EXTERNAL_ITEM, ORIGIN_CORPUS)


#: ``question_date`` values derived from the pair's own evidence are barred:
#: setting t_q to the newer passage's publication date gives that passage
#: gamma = 1 by construction, inflating the currency contrast SCAF is being
#: measured on. See ledger decision D-21.
FORBIDDEN_QUESTION_DATE_SOURCES = (
    "derived:newer_publication_date",
    "derived:older_publication_date",
)


class PairCategory(str, Enum):
    """Specification section 11."""

    CHANGED = "changed"
    UNCHANGED = "unchanged"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


class ContradictionStatus(str, Enum):
    """How the relationship between the two passages was determined.

    ``EXTERNALLY_ESTABLISHED`` is reserved for pairs whose opposition comes
    from the external dataset's own verdict change: the thesis did not judge
    it, so it does not inherit the thesis's judgement error.
    """

    EXTERNALLY_ESTABLISHED = "externally_established"
    GENUINE_CONTRADICTION = "genuine_contradiction"
    APPARENT_CONTEXT_DIFFERENCE = "apparent_context_difference"
    SCIENTIFIC_PROGRESSION = "scientific_progression"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class ValidationStatus(str, Enum):
    """How far a pair has been checked."""

    UNVALIDATED = "unvalidated"
    MACHINE_CHECKED = "machine_checked"
    MANUALLY_VALIDATED = "manually_validated"
    REJECTED = "rejected"


#: Exclusion vocabulary. Specification section 35 lists the required reasons;
#: the rest are the mechanical failures the builder can actually detect.
#: A closed vocabulary keeps the attrition table addable across runs.
EXCLUSION_REASONS = (
    "missing_question_date",
    "missing_reference_answer",
    "missing_publication_date",
    "missing_evidence_text",
    "ambiguous_temporal_order",
    "insufficient_temporal_separation",
    "insufficient_contradiction",
    "unverified_claim_equivalence",
    "missing_persistent_identifier",
    "source_tier_mismatch",
    "length_mismatch",
    "unsupported_claim_class",
    "duplicate_evidence",
    "non_comparable_context",
    "retracted_or_withdrawn",
    "insufficient_provenance",
    "question_date_not_independent",
    "other",
)


class SchemaError(ValueError):
    """Raised when a record violates a Stage-2 invariant."""


def _parse_partial_date(value: Optional[str]) -> tuple[Optional[date], Optional[date], str]:
    """Return ``(earliest, latest, precision)`` for a partial date string.

    Accepts ``YYYY``, ``YYYY-MM`` and ``YYYY-MM-DD``. A date is an interval:
    ``"2024"`` spans 2024-01-01 to 2024-12-31. Returning both ends is what
    lets :mod:`.eligibility` decide whether an ordering is unambiguous
    without an invented tolerance parameter.
    """

    if value is None or not str(value).strip():
        return (None, None, "unknown")

    text = str(value).strip()
    parts = text.split("-")

    try:
        if len(parts) == 1:
            year = int(parts[0])
            return (date(year, 1, 1), date(year, 12, 31), "year")

        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            if month == 12:
                last = date(year, 12, 31)
            else:
                last = date(year, month + 1, 1).toordinal() - 1
                last = date.fromordinal(last)
            return (date(year, month, 1), last, "month")

        if len(parts) == 3:
            day = date(int(parts[0]), int(parts[1]), int(parts[2]))
            return (day, day, "day")

    except (ValueError, TypeError):
        return (None, None, "unparseable")

    return (None, None, "unparseable")


@dataclass(frozen=True)
class EvidenceRef:
    """One side of a pair, as a reference into the frozen corpus.

    This is deliberately a *reference*, not a copy of the passage: the
    frozen-candidate-set principle (specification section 16) requires every
    arm to replay the same corpus objects, so Stage 2 stores identifiers and
    the metadata needed for matching, and the text stays in the corpus.
    """

    evidence_id: str
    document_id: str

    publication_date: Optional[str] = None
    source_tier: Optional[str] = None
    persistent_id: Optional[str] = None

    #: Passage length, used for the length-tolerance matching constraint
    #: (specification section 10.1). Whitespace tokens, matching the corpus
    #: chunker's own unit, so the two are comparable.
    length_tokens: Optional[int] = None

    retracted: bool = False
    withdrawn: bool = False

    #: Passage text. Required for ``external_item`` passages, which do not
    #: exist anywhere else and would otherwise be unrecoverable; omitted for
    #: ``corpus`` passages, whose text lives in the frozen corpus.
    text: Optional[str] = None

    origin: str = ORIGIN_CORPUS

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise SchemaError("evidence_id is required.")
        if not self.document_id:
            raise SchemaError("document_id is required.")
        if self.origin not in EVIDENCE_ORIGINS:
            raise SchemaError(
                f"origin must be one of {EVIDENCE_ORIGINS}; "
                f"got {self.origin!r}."
            )

    @property
    def date_interval(self) -> tuple[Optional[date], Optional[date]]:
        earliest, latest, _ = _parse_partial_date(self.publication_date)
        return (earliest, latest)

    @property
    def date_precision(self) -> str:
        return _parse_partial_date(self.publication_date)[2]

    @property
    def has_usable_date(self) -> bool:
        return self.date_interval[0] is not None


@dataclass(frozen=True)
class Provenance:
    """Where the question, the answer and the evidence each came from.

    Recorded per pair so a reviewer can audit any single item without
    trusting the file it happens to live in.
    """

    pool: str
    question_source: str
    answer_source: str
    evidence_source: str
    extraction_date: str

    def __post_init__(self) -> None:
        if self.pool not in POOLS:
            raise SchemaError(
                f"pool must be one of {POOLS}; got {self.pool!r}."
            )
        for name in ("question_source", "answer_source", "evidence_source"):
            if not getattr(self, name):
                raise SchemaError(f"{name} is required for provenance audit.")


@dataclass(frozen=True)
class TestPair:
    """One temporal-counterfactual candidate pair."""

    pair_id: str
    question_id: str
    question_text: str

    provenance: Provenance

    older: EvidenceRef
    newer: EvidenceRef

    question_date: Optional[str] = None
    question_date_source: str = "unknown"

    reference_answer: Optional[str] = None
    reference_answer_source: str = "unknown"

    change_point_date: Optional[str] = None
    change_point_source: str = "unknown"

    claim_class: Optional[str] = None
    #: external | manual | heuristic | unknown. A ``heuristic`` value must
    #: never be used as a matching criterion in the primary pool (ledger A8).
    claim_class_source: str = "unknown"

    pair_category: str = PairCategory.UNKNOWN.value
    contradiction_status: str = ContradictionStatus.INSUFFICIENT_INFORMATION.value
    contradiction_source: str = "unknown"

    temporal_eligible: bool = False
    separation_days: Optional[int] = None
    #: Separation guaranteed by the date intervals alone. Smaller than
    #: ``separation_days`` whenever either date is imprecise.
    min_separation_days: Optional[int] = None

    exclusion_reasons: tuple[str, ...] = ()
    validation_status: str = ValidationStatus.UNVALIDATED.value
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise SchemaError("pair_id is required.")
        if not self.question_id:
            raise SchemaError("question_id is required.")
        if not self.question_text:
            raise SchemaError("question_text is required.")

        if self.older.evidence_id == self.newer.evidence_id:
            raise SchemaError(
                f"{self.pair_id}: older and newer are the same passage "
                f"({self.older.evidence_id!r})."
            )

        unknown = set(self.exclusion_reasons) - set(EXCLUSION_REASONS)
        if unknown:
            raise SchemaError(
                f"{self.pair_id}: unknown exclusion reason(s) "
                f"{sorted(unknown)}. Add to EXCLUSION_REASONS rather than "
                "inventing one at the call site, so attrition stays addable."
            )

        if self.temporal_eligible and self.exclusion_reasons:
            raise SchemaError(
                f"{self.pair_id}: marked temporally eligible while carrying "
                f"exclusion reasons {list(self.exclusion_reasons)}."
            )

        if self.question_date_source in FORBIDDEN_QUESTION_DATE_SOURCES:
            raise SchemaError(
                f"{self.pair_id}: question_date_source "
                f"{self.question_date_source!r} derives t_q from the evidence "
                "under test, which makes that passage maximally current by "
                "construction (ledger D-21). Use the dataset's own date or "
                "the experiment-wide evaluation_as_of_date."
            )

        for name, enum_type in (
            ("pair_category", PairCategory),
            ("contradiction_status", ContradictionStatus),
            ("validation_status", ValidationStatus),
        ):
            value = getattr(self, name)
            if value not in {member.value for member in enum_type}:
                raise SchemaError(f"{self.pair_id}: invalid {name}={value!r}.")

    @property
    def is_eligible(self) -> bool:
        """Usable in the primary temporal evaluation set."""
        return self.temporal_eligible and not self.exclusion_reasons

    @property
    def evidence_ids(self) -> tuple[str, str]:
        return (self.older.evidence_id, self.newer.evidence_id)

    def excluded(self, *reasons: str) -> "TestPair":
        """Return a copy carrying additional exclusion reasons."""
        merged = tuple(
            sorted(set(self.exclusion_reasons) | set(reasons))
        )
        return replace(
            self,
            exclusion_reasons=merged,
            temporal_eligible=False,
        )

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["exclusion_reasons"] = list(self.exclusion_reasons)
        # Derived, recorded so downstream consumers need not re-parse dates.
        record["older"]["date_precision"] = self.older.date_precision
        record["newer"]["date_precision"] = self.newer.date_precision
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "TestPair":
        data = dict(record)

        def _ref(key: str) -> EvidenceRef:
            raw = dict(data.pop(key))
            raw.pop("date_precision", None)
            return EvidenceRef(**raw)

        older = _ref("older")
        newer = _ref("newer")
        provenance = Provenance(**dict(data.pop("provenance")))
        data["exclusion_reasons"] = tuple(data.get("exclusion_reasons", ()))

        return cls(
            older=older,
            newer=newer,
            provenance=provenance,
            **data,
        )


def assert_firewall(pairs: Sequence[TestPair]) -> str:
    """Return the single pool of ``pairs``, or raise.

    Specification section 33.2 / ledger D-2: externally-authored evaluation
    material and thesis-curated material support different claims and must
    not be analysed as one set. This is checked on write and on read so a
    mixed file cannot reach an experiment at all.
    """

    pools = {pair.provenance.pool for pair in pairs}

    if not pools:
        raise SchemaError("No pairs: cannot determine pool.")

    if len(pools) > 1:
        raise SchemaError(
            "Provenance firewall violation: a single collection contains "
            f"pools {sorted(pools)}. Externally-authored and thesis-curated "
            "material must be kept in separate files (specification 33.2)."
        )

    return pools.pop()


def write_pairs(pairs: Iterable[TestPair], path: Path) -> int:
    """Write pairs as deterministic JSONL. Returns the count written."""

    ordered = sorted(pairs, key=lambda p: p.pair_id)

    if ordered:
        assert_firewall(ordered)

    seen: set[str] = set()
    for pair in ordered:
        if pair.pair_id in seen:
            raise SchemaError(f"Duplicate pair_id {pair.pair_id!r}.")
        seen.add(pair.pair_id)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for pair in ordered:
            handle.write(
                json.dumps(pair.to_dict(), sort_keys=True, ensure_ascii=False)
            )
            handle.write("\n")

    return len(ordered)


def content_hash(path: Path) -> str:
    """SHA-256 of a serialised pair file.

    Recorded in the Stage-2 manifest and asserted by Stage 3, so a pair file
    edited after the evaluation set was frozen cannot be used without the
    mismatch being visible (specification 33.6). Writing is deterministic, so
    the hash is stable across regenerations of identical content.
    """

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(path: Path, expected_sha256: str) -> None:
    """Raise unless ``path`` still matches the hash recorded at freeze time."""

    actual = content_hash(path)

    if actual != expected_sha256:
        raise SchemaError(
            f"{path} has changed since it was frozen (expected "
            f"{expected_sha256[:12]}..., found {actual[:12]}...). Stage 3 "
            "must run against the frozen evaluation set; a pair file edited "
            "after Stage 3 results were seen invalidates the experiment."
        )


def read_pairs(path: Path) -> list[TestPair]:
    """Read a JSONL pair file, enforcing the firewall."""

    pairs: list[TestPair] = []

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(TestPair.from_dict(json.loads(line)))

    if pairs:
        assert_firewall(pairs)

    return pairs
