"""Validation / test split over the reviewed evaluation-question pool.

Why two splits, not three. Step 3 of the roadmap is titled "train/validation/
test split", but nothing in this thesis *trains* on the 123-question pool:
the RAG2 filter is trained on general-medical MedQA (specification SS10),
never on these Alzheimer's questions, and the generator is used off the
shelf, never fine-tuned. The only thing this pool feeds is the small,
manual fit of three scalar parameters -- lambda, theta, H (specification
SS8.1) -- which needs a held-out set to be fit *on*, and a separate,
untouched set to report the final result *on*. That is a validation/test
split, not a train/validation/test split. Forcing a third, unused "training"
partition out of 113 questions would only shrink the two splits that matter
for no scientific reason, so this module produces exactly two.

Inputs, and what counts as usable. A question enters this split only if it
survived the completed human review in ``review.csv``:

* ``REJECT`` -- excluded. It never reaches this module's output.
* ``HOLD`` -- excluded (none exist in the current review, but the rule holds
  if one ever appears): the reviewer has not yet decided the question is
  usable at all.
* ``ACCEPT`` -- included, no caveat.
* ``REVISE`` -- included, but flagged ``pending_revision: true``. The
  reviewer's own taxonomy (``docs/question_review.md``) is explicit that a
  REVISE item's underlying claim is sound and only its wording needs fixing
  -- it is not a REJECT-in-waiting. Excluding all 23 of them here would
  throw away defensible questions over a wording problem that is separate,
  already-scoped work (correcting ``reference_answer``/question text, never
  done by guessing at unseen source text). What this module will not do is
  pretend they are already clean: every split consumer can see the flag and
  a run should refuse to treat an unrevised REVISE item as final evidence.

Reproducibility. The split is a seeded, deterministic function of the
reviewed pool: same inputs, same seed, same assignment, byte for byte. There
is no clock, no file-system ordering dependency (ids are sorted before the
seeded shuffle), and no manual choice per question.

    python -m experiments.questions.split
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

DEFAULT_POOL = Path("experiments/questions")
DEFAULT_OUTPUT = DEFAULT_POOL / "splits.json"

#: review_decision values that make a question usable in an experimental split.
USABLE_DECISIONS = ("ACCEPT", "REVISE")

#: Fraction of the usable pool held out for parameter fitting. A validation
#: split exists to fit three scalar values (lambda, theta, H) by a small grid
#: or line search, not to train a model -- it does not need a large sample.
#: 20% keeps every topic stratum (the smallest has 2 usable questions)
#: representable in both splits while leaving the majority of the pool, and
#: every topic's dominant share, in the test set the thesis actually reports.
DEFAULT_VALIDATION_FRACTION = 0.20

#: Fixed so the split is reproducible without anyone having to remember a
#: number. Changing it would change the split, so it is not exposed as a
#: casual CLI default -- callers who must change it pass --seed explicitly.
DEFAULT_SEED = 20260921

SPLIT_NAMES = ("validation", "test")


class SplitError(RuntimeError):
    """Raised when the pool cannot be split safely, or an existing split
    on disk no longer matches what the reviewed pool would produce."""


def load_reviewed_pool(pool_dir: Path = DEFAULT_POOL) -> list[dict[str, Any]]:
    """Join ``candidates.jsonl`` and ``review.csv`` on ``question_id``.

    Returns one record per row of ``review.csv`` (the human-reviewed set),
    carrying the reviewer's decision plus the topic/temporal metadata that
    only exists in ``candidates.jsonl`` and was deliberately withheld from
    the reviewer (``questions.WITHHELD_FROM_REVIEW``).
    """
    pool_dir = Path(pool_dir)
    candidates_path = pool_dir / "candidates.jsonl"
    review_path = pool_dir / "review.csv"
    if not candidates_path.exists():
        raise SplitError(f"no candidate pool at {candidates_path}")
    if not review_path.exists():
        raise SplitError(f"no review file at {review_path}")

    by_id: dict[str, dict[str, Any]] = {}
    with open(candidates_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            by_id[record["question_id"]] = record

    joined = []
    with open(review_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = row["question_id"]
            if qid not in by_id:
                raise SplitError(
                    f"{qid} is in review.csv but not in candidates.jsonl"
                )
            candidate = by_id[qid]
            joined.append({
                "question_id": qid,
                "topic": candidate["topic"],
                "subtopic": candidate.get("subtopic"),
                "temporal_candidate": bool(candidate.get("temporal_candidate", False)),
                "review_decision": row["review_decision"].strip(),
                "reviewer_note": row["reviewer_note"].strip(),
            })
    return joined


def usable_records(reviewed: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """REJECT and HOLD (and anything unreviewed) never reach the output."""
    return [r for r in reviewed if r["review_decision"] in USABLE_DECISIONS]


def _largest_remainder(counts: dict[str, int], fraction: float) -> dict[str, int]:
    """How many validation items each stratum gets, summing to round(total*fraction).

    Proportional (Hamilton/largest-remainder) allocation: every stratum's
    share is proportional to its size, and the few leftover seats from
    integer rounding go to the strata with the largest fractional remainder.
    Deterministic given the input order, which the caller fixes by sorting.
    """
    exact = {k: v * fraction for k, v in counts.items()}
    base = {k: int(v) for k, v in exact.items()}
    remainder_order = sorted(counts, key=lambda k: (-(exact[k] - base[k]), k))
    total_target = round(sum(counts.values()) * fraction)
    seats_left = total_target - sum(base.values())
    for k in remainder_order[:max(seats_left, 0)]:
        base[k] += 1
    return base


def assign_splits(
    usable: Sequence[dict[str, Any]],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """{question_id: "validation" | "test"}, stratified by topic, then by
    ``temporal_candidate`` within each topic.

    Topic first, so every topic keeps its usual share in both splits (a
    topic's validation quota is proportional to its size in the pool). Then,
    within each topic, the same proportional-rounding rule splits that
    topic's quota between its temporal and non-temporal questions, so
    validation's temporal/non-temporal mix tracks test's rather than
    drifting -- an early version of this function stratified by topic alone
    and validation ended up 74% temporal against test's 54%, which would
    have made a lambda fit on validation look better than it generalises.
    A topic with only one question on one side of that split still lands
    wherever the topic-level quota puts it; there is no data left to
    stratify further at that size, and that is reported, not hidden.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise SplitError("validation_fraction must be between 0 and 1")

    by_topic: dict[str, list[dict[str, Any]]] = {}
    for r in usable:
        by_topic.setdefault(r["topic"], []).append(r)

    topic_counts = {topic: len(rs) for topic, rs in by_topic.items()}
    topic_quota = _largest_remainder(topic_counts, validation_fraction)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for topic in sorted(by_topic):
        records = by_topic[topic]
        temporal_ids = sorted(r["question_id"] for r in records if r["temporal_candidate"])
        non_temporal_ids = sorted(r["question_id"] for r in records if not r["temporal_candidate"])

        sub_counts = {"temporal": len(temporal_ids), "non_temporal": len(non_temporal_ids)}
        sub_quota = _largest_remainder(sub_counts, topic_quota[topic] / len(records))
        # Rounding both levels independently can miss the topic's exact
        # quota by one seat; correct it deterministically against whichever
        # subgroup still has room, largest subgroup first.
        drift = topic_quota[topic] - sum(sub_quota.values())
        for name in sorted(sub_counts, key=lambda n: -sub_counts[n]):
            if drift == 0:
                break
            if drift > 0 and sub_quota[name] < sub_counts[name]:
                sub_quota[name] += 1
                drift -= 1
            elif drift < 0 and sub_quota[name] > 0:
                sub_quota[name] -= 1
                drift += 1

        for name, ids in (("temporal", temporal_ids), ("non_temporal", non_temporal_ids)):
            ids = list(ids)
            rng.shuffle(ids)
            n_val = sub_quota[name]
            for qid in ids[:n_val]:
                assignment[qid] = "validation"
            for qid in ids[n_val:]:
                assignment[qid] = "test"
    return assignment


def split_summary(
    reviewed: Sequence[dict[str, Any]], assignment: dict[str, str],
) -> dict[str, Any]:
    """Counts a reader needs to judge whether the split is defensible."""
    by_id = {r["question_id"]: r for r in reviewed}
    summary: dict[str, Any] = {name: {
        "count": 0, "by_topic": {}, "temporal_candidate": 0,
        "non_temporal": 0, "pending_revision": 0,
    } for name in SPLIT_NAMES}
    for qid, split in assignment.items():
        r = by_id[qid]
        bucket = summary[split]
        bucket["count"] += 1
        bucket["by_topic"][r["topic"]] = bucket["by_topic"].get(r["topic"], 0) + 1
        if r["temporal_candidate"]:
            bucket["temporal_candidate"] += 1
        else:
            bucket["non_temporal"] += 1
        if r["review_decision"] == "REVISE":
            bucket["pending_revision"] += 1
    for bucket in summary.values():
        bucket["by_topic"] = dict(sorted(bucket["by_topic"].items()))
    return summary


def build_split(
    pool_dir: Path = DEFAULT_POOL,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    reviewed = load_reviewed_pool(pool_dir)
    usable = usable_records(reviewed)
    excluded = [r for r in reviewed if r not in usable]
    assignment = assign_splits(
        usable, validation_fraction=validation_fraction, seed=seed
    )
    assert_valid(reviewed, assignment)
    by_id = {r["question_id"]: r for r in reviewed}
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "usable_decisions": list(USABLE_DECISIONS),
        "total_reviewed": len(reviewed),
        "total_usable": len(usable),
        "excluded": [
            {"question_id": r["question_id"], "review_decision": r["review_decision"]}
            for r in sorted(excluded, key=lambda r: r["question_id"])
        ],
        "summary": split_summary(reviewed, assignment),
        "items": [
            {
                "question_id": qid,
                "split": split,
                "topic": by_id[qid]["topic"],
                "temporal_candidate": by_id[qid]["temporal_candidate"],
                "review_decision": by_id[qid]["review_decision"],
                "pending_revision": by_id[qid]["review_decision"] == "REVISE",
            }
            for qid, split in sorted(assignment.items())
        ],
    }


def assert_valid(
    reviewed: Sequence[dict[str, Any]], assignment: dict[str, str],
) -> None:
    """Mutual exclusivity, completeness, and no rejected/held item leaking in.

    Raises ``SplitError`` naming exactly what is wrong; callers (including
    the test suite) should never need to re-derive these checks by hand.
    """
    usable = usable_records(reviewed)
    usable_ids = {r["question_id"] for r in usable}
    excluded_ids = {r["question_id"] for r in reviewed} - usable_ids

    assigned_ids = set(assignment)
    if assigned_ids != usable_ids:
        missing = sorted(usable_ids - assigned_ids)
        extra = sorted(assigned_ids - usable_ids)
        raise SplitError(
            "split does not exactly cover the usable pool: "
            f"missing={missing} extra={extra}"
        )

    leaked = sorted(excluded_ids & assigned_ids)
    if leaked:
        raise SplitError(
            f"rejected/held question(s) present in the split: {leaked}"
        )

    values = set(assignment.values())
    if not values <= set(SPLIT_NAMES):
        raise SplitError(f"unknown split name(s): {sorted(values - set(SPLIT_NAMES))}")

    if len(set(assignment.items())) != len(assignment):
        raise SplitError("assignment is not a well-formed mapping")  # unreachable via dict, defensive


def write_split(
    pool_dir: Path = DEFAULT_POOL,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    """Write ``splits.json``. Refuses to silently change an existing split.

    A frozen split changing without anyone deciding that is exactly the
    failure mode this guards against: if the pool changed (a question was
    re-reviewed, added or removed) since the file was last written, the
    freshly computed split will differ from what is on disk, and this
    raises rather than overwriting -- pass ``force=True`` once that is a
    deliberate choice, not an accident.
    """
    output_path = Path(output_path)
    result = build_split(
        pool_dir, validation_fraction=validation_fraction, seed=seed
    )
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("items") != result["items"]:
            raise SplitError(
                f"{output_path} already exists and would change "
                "(the reviewed pool has changed since it was written). "
                "Pass force=True / --force once that is intended."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float,
                         default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = write_split(
        args.pool_dir, args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed, force=args.force,
    )
    print(f"total_reviewed={result['total_reviewed']} "
          f"total_usable={result['total_usable']} "
          f"validation={result['summary']['validation']['count']} "
          f"test={result['summary']['test']['count']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    _main()
