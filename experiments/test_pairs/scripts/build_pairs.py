"""Build and audit a Stage-2 pilot of candidate test pairs.

Two input modes, kept separate because they feed different pools and support
different claims (specification 33.2, ledger D-2):

``--pool primary_external``
    Reads an externally-authored evaluation file whose questions, reference
    answers and change points were not produced by this thesis. The field
    contract is documented in :data:`EXTERNAL_FIELDS` and validated on load.
    If the file is absent the command fails and names the dependency; it
    never falls back to thesis-created questions.

``--pool secondary_curated``
    Reads the Alzheimer's corpus chunk file and enumerates candidate pairs
    within each claim class. The corpus carries no questions, no reference
    answers and no contradiction labels, so every candidate it produces is
    excluded for exactly those reasons. That is the intended outcome: the
    run measures the size of the gap rather than papering over it.

Nothing here retrieves, ranks, or scores. Stage 2 emits references into the
frozen corpus so that Stage 3 can build one cached candidate set and replay
it across arms (specification 16).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from . import attrition as attrition_module
from . import check_a, split as split_module
from .eligibility import EligibilityConfig, evaluate as evaluate_eligibility
from .schema import (
    POOL_PRIMARY_EXTERNAL,
    POOL_SECONDARY_CURATED,
    ContradictionStatus,
    EvidenceRef,
    PairCategory,
    Provenance,
    SchemaError,
    TestPair,
    ValidationStatus,
    write_pairs,
)

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None


#: Required fields of an external evaluation file, one JSON object per line.
#: Documented here so the file can be prepared before the dataset arrives.
EXTERNAL_FIELDS = (
    "question_id",
    "question_text",
    "reference_answer",
    "older_evidence_id",
    "older_document_id",
    "newer_evidence_id",
    "newer_document_id",
)

#: Optional fields. Absence is recorded, never filled in.
EXTERNAL_OPTIONAL_FIELDS = (
    "question_date",
    "change_point_date",
    "claim_class",
    "pair_category",
    "contradiction_status",
    "older_publication_date",
    "older_source_tier",
    "older_persistent_id",
    "older_length_tokens",
    "newer_publication_date",
    "newer_source_tier",
    "newer_persistent_id",
    "newer_length_tokens",
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "experiments" / "configs" / "stage2_pilot.yaml"
)


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{number}: invalid JSON - {exc}") from exc
    return records


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to read the Stage-2 config.")
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# --------------------------------------------------------------------------
# primary_external
# --------------------------------------------------------------------------


def build_external_pairs(
    records: Sequence[dict[str, Any]],
    *,
    dataset_name: str,
    evidence_source: str,
    extraction_date: str,
) -> list[TestPair]:
    """Convert external evaluation records into candidate pairs."""

    pairs: list[TestPair] = []

    for index, record in enumerate(records):

        missing = [field for field in EXTERNAL_FIELDS if not record.get(field)]
        if missing:
            raise SchemaError(
                f"external record {index}: missing required field(s) "
                f"{missing}. The loader does not supply defaults, because a "
                "silently defaulted question date or reference answer would "
                "be indistinguishable from a real one."
            )

        reasons: list[str] = []

        question_date = record.get("question_date")
        if question_date:
            question_date_source = f"{dataset_name}:question_date"
        else:
            # Item-level fallback: the question is posed as of the date the
            # newer evidence appeared. Never a single global cutoff.
            question_date = record.get("newer_publication_date")
            question_date_source = "derived:newer_publication_date"

        contradiction = record.get(
            "contradiction_status",
            ContradictionStatus.EXTERNALLY_ESTABLISHED.value,
        )
        if contradiction == ContradictionStatus.INSUFFICIENT_INFORMATION.value:
            reasons.append("insufficient_contradiction")

        claim_class = record.get("claim_class")

        older = EvidenceRef(
            evidence_id=str(record["older_evidence_id"]),
            document_id=str(record["older_document_id"]),
            publication_date=record.get("older_publication_date"),
            source_tier=record.get("older_source_tier"),
            persistent_id=record.get("older_persistent_id"),
            length_tokens=record.get("older_length_tokens"),
        )
        newer = EvidenceRef(
            evidence_id=str(record["newer_evidence_id"]),
            document_id=str(record["newer_document_id"]),
            publication_date=record.get("newer_publication_date"),
            source_tier=record.get("newer_source_tier"),
            persistent_id=record.get("newer_persistent_id"),
            length_tokens=record.get("newer_length_tokens"),
        )

        # Identity is derived from content, never from the input line index:
        # a pair id that changed when the source file was reordered would
        # break every downstream reference to it.
        pair_key = hashlib.sha256(
            "|".join(
                (
                    str(record["question_id"]),
                    str(record["older_evidence_id"]),
                    str(record["newer_evidence_id"]),
                )
            ).encode("utf-8")
        ).hexdigest()[:10]

        pairs.append(
            TestPair(
                pair_id=f"EXT-{record['question_id']}-{pair_key}",
                question_id=str(record["question_id"]),
                question_text=str(record["question_text"]),
                question_date=question_date,
                question_date_source=question_date_source,
                reference_answer=str(record["reference_answer"]),
                reference_answer_source=f"{dataset_name}:reference_answer",
                change_point_date=record.get("change_point_date"),
                change_point_source=(
                    f"{dataset_name}:change_point"
                    if record.get("change_point_date")
                    else "derived:newer_publication_date"
                ),
                claim_class=claim_class,
                claim_class_source="external" if claim_class else "unknown",
                pair_category=record.get(
                    "pair_category", PairCategory.CHANGED.value
                ),
                contradiction_status=contradiction,
                contradiction_source=dataset_name,
                provenance=Provenance(
                    pool=POOL_PRIMARY_EXTERNAL,
                    question_source=dataset_name,
                    answer_source=dataset_name,
                    evidence_source=evidence_source,
                    extraction_date=extraction_date,
                ),
                older=older,
                newer=newer,
                exclusion_reasons=tuple(reasons),
                validation_status=ValidationStatus.UNVALIDATED.value,
            )
        )

    return pairs


# --------------------------------------------------------------------------
# secondary_curated
# --------------------------------------------------------------------------


def _chunk_ref(chunk: dict[str, Any]) -> EvidenceRef:
    text = chunk.get("text") or ""
    return EvidenceRef(
        evidence_id=str(chunk["chunk_id"]),
        document_id=str(chunk["document_id"]),
        publication_date=chunk.get("publication_date") or None,
        source_tier=chunk.get("source_tier") or None,
        persistent_id=chunk.get("pmid") or chunk.get("doi") or None,
        length_tokens=len(text.split()) or None,
        retracted=bool(chunk.get("retracted")),
        withdrawn=bool(chunk.get("withdrawn")),
    )


def build_curated_candidates(
    chunks: Sequence[dict[str, Any]],
    *,
    corpus_source: str,
    extraction_date: str,
) -> list[TestPair]:
    """Enumerate within-claim-class candidate pairs from corpus chunks.

    The corpus supplies passages and dates but no question, no reference
    answer and no verified contradiction, so each candidate carries those
    exclusions from the start. Claim classes in the current corpus come from
    an automatic keyword heuristic and are marked ``heuristic``; they group
    candidates for counting and must not be treated as verified equivalence.
    """

    by_class: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        for claim_class in chunk.get("claim_classes") or []:
            by_class.setdefault(str(claim_class), []).append(chunk)

    pairs: list[TestPair] = []

    for claim_class in sorted(by_class):

        members = sorted(
            by_class[claim_class], key=lambda c: str(c["chunk_id"])
        )

        for left, right in itertools.combinations(members, 2):

            left_ref = _chunk_ref(left)
            right_ref = _chunk_ref(right)

            left_start = left_ref.date_interval[0]
            right_start = right_ref.date_interval[0]

            if left_start and right_start and right_start < left_start:
                older_ref, newer_ref = right_ref, left_ref
            else:
                older_ref, newer_ref = left_ref, right_ref

            reasons = [
                # No question or answer exists for corpus-only material.
                "missing_reference_answer",
                # Nothing has established that these two passages oppose
                # each other; shared claim class is not contradiction.
                "insufficient_contradiction",
                # The claim class is an unvalidated automatic label, so
                # claim equivalence is asserted by nothing.
                "unverified_claim_equivalence",
                # Questions are placeholders, not externally authored.
                "insufficient_provenance",
            ]

            if older_ref.document_id == newer_ref.document_id:
                reasons.append("duplicate_evidence")

            pair_id = (
                f"CUR-{claim_class}-"
                + hashlib.sha256(
                    f"{older_ref.evidence_id}|{newer_ref.evidence_id}".encode()
                ).hexdigest()[:10]
            )

            pairs.append(
                TestPair(
                    pair_id=pair_id,
                    question_id=f"PLACEHOLDER-{claim_class}",
                    question_text=(
                        f"[NO QUESTION: corpus-only candidate for claim class "
                        f"{claim_class}]"
                    ),
                    question_date=None,
                    question_date_source="unavailable",
                    reference_answer=None,
                    reference_answer_source="unavailable",
                    change_point_date=None,
                    change_point_source="unavailable",
                    claim_class=claim_class,
                    claim_class_source="heuristic",
                    pair_category=PairCategory.UNKNOWN.value,
                    contradiction_status=(
                        ContradictionStatus.INSUFFICIENT_INFORMATION.value
                    ),
                    contradiction_source="none",
                    provenance=Provenance(
                        pool=POOL_SECONDARY_CURATED,
                        question_source="placeholder:claim_class",
                        answer_source="unavailable",
                        evidence_source=corpus_source,
                        extraction_date=extraction_date,
                    ),
                    older=older_ref,
                    newer=newer_ref,
                    exclusion_reasons=tuple(sorted(set(reasons))),
                    validation_status=ValidationStatus.UNVALIDATED.value,
                    notes=(
                        "Corpus-only candidate. Not a test pair: it has no "
                        "question, no reference answer and no verified "
                        "contradiction."
                    ),
                )
            )

    return pairs


# --------------------------------------------------------------------------
# pilot sampling
# --------------------------------------------------------------------------


def sample_pilot(
    pairs: Sequence[TestPair],
    *,
    size: Optional[int],
    seed: str,
) -> list[TestPair]:
    """Deterministically take up to ``size`` candidates.

    Ordering is by a seeded hash of the pair id, so the sample does not
    depend on input order and adding candidates later does not reshuffle the
    ones already chosen.
    """

    ordered = sorted(
        pairs,
        key=lambda p: hashlib.sha256(
            f"{seed}:{p.pair_id}".encode("utf-8")
        ).hexdigest(),
    )

    if size is None:
        return sorted(ordered, key=lambda p: p.pair_id)

    return sorted(ordered[:size], key=lambda p: p.pair_id)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run(
    *,
    pool: str,
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the pilot, audit it, and write pairs plus a manifest."""

    extraction_date = _utc_today()

    pilot_cfg = config.get("pilot", {})
    eligibility_cfg = EligibilityConfig(**(config.get("eligibility") or {}))
    split_cfg = split_module.SplitConfig(**(config.get("split") or {}))
    split_cfg.validate()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Stage-2 input not found: {input_path}\n"
            + (
                "The primary evaluation pool requires an externally-authored "
                "dataset. It is not generated here and no substitute is "
                "created: see docs/RESEARCH_LEDGER.md for the outstanding "
                "dependency."
                if pool == POOL_PRIMARY_EXTERNAL
                else "Run the Stage-1 chunking script first."
            )
        )

    records = _read_jsonl(input_path)

    if pool == POOL_PRIMARY_EXTERNAL:
        candidates = build_external_pairs(
            records,
            dataset_name=str(config.get("external_dataset", "UNSPECIFIED")),
            evidence_source=str(config.get("corpus_snapshot", "UNSPECIFIED")),
            extraction_date=extraction_date,
        )
    elif pool == POOL_SECONDARY_CURATED:
        candidates = build_curated_candidates(
            records,
            corpus_source=str(config.get("corpus_snapshot", "UNSPECIFIED")),
            extraction_date=extraction_date,
        )
    else:
        raise ValueError(f"unknown pool {pool!r}.")

    seed = str(pilot_cfg.get("seed", split_cfg.seed))
    pilot = sample_pilot(
        candidates,
        size=pilot_cfg.get("size"),
        seed=seed,
    )

    assessed = [evaluate_eligibility(pair, eligibility_cfg) for pair in pilot]

    report = attrition_module.compute(assessed)
    separation = attrition_module.separation_summary(assessed)
    partitions = split_module.split_pairs(assessed, split_cfg)

    cutoffs = config.get("label_model_cutoffs") or {}
    check_a_rows = check_a.evaluate(assessed, cutoffs) if cutoffs else []

    # Check A is only a Check A result when the change points are real. When
    # every pair falls back to a publication date, the strata describe the
    # corpus, not the evidence-change timeline, and must not be reported as
    # Check A.
    explicit_change_points = sum(
        1 for pair in assessed if pair.change_point_date
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / f"{pool}_pilot.jsonl"
    write_pairs(assessed, pairs_path)
    report.write_csv(output_dir / f"{pool}_attrition.csv")

    manifest = {
        "stage": 2,
        "pool": pool,
        "input_path": str(input_path),
        "extraction_date": extraction_date,
        "candidates_enumerated": len(candidates),
        "pilot_size_requested": pilot_cfg.get("size"),
        "pilot_size_actual": len(assessed),
        "pilot_seed": seed,
        "eligibility_config": asdict(eligibility_cfg),
        "eligibility_rules_enforced": list(eligibility_cfg.enforced_rules()),
        "split_config": asdict(split_cfg),
        "attrition": {
            "candidates": report.candidates,
            "eligible": report.eligible,
            "excluded": report.excluded,
            "yield_rate": round(report.yield_rate, 4),
            "pairs_with_reason": report.pairs_with_reason,
            "sole_reason": report.sole_reason,
        },
        "separation_summary": separation,
        "check_a": check_a_rows,
        "check_a_interpretable": bool(assessed) and explicit_change_points == len(assessed),
        "check_a_explicit_change_points": explicit_change_points,
        "split_summary": split_module.summarise(partitions),
        "outputs": {
            "pairs": str(pairs_path),
            "attrition": str(output_dir / f"{pool}_attrition.csv"),
        },
    }

    manifest_path = output_dir / f"{pool}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Stage-2 pilot of candidate test pairs.",
    )
    parser.add_argument(
        "--pool",
        required=True,
        choices=[POOL_PRIMARY_EXTERNAL, POOL_SECONDARY_CURATED],
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    args = parser.parse_args(argv)

    manifest = run(
        pool=args.pool,
        input_path=args.input,
        output_dir=args.output_dir,
        config=load_config(args.config),
    )

    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
