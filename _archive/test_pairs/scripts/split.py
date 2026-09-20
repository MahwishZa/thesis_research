"""Development / validation / final-test partitioning (specification 32).

Two properties matter more than the split algorithm:

1.  **The unit is the question, not the pair.** Two pairs sharing a question
    share its answer and usually one of its passages. Splitting on pairs
    would put near-duplicates of a test item into the tuning partition,
    which is the leakage section 33.1 forbids.

2.  **The assignment is a pure function of the question id and a recorded
    seed.** No RNG state, no shuffling, no dependence on input order or on
    how many questions happen to exist. Re-running after adding questions
    leaves existing assignments untouched, so the test partition cannot
    quietly change composition between the pilot and the final set.

Hash-based assignment gives proportions that are approximate at small n.
That is the honest trade for stability, and the realised counts are reported
rather than forced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .schema import TestPair


SPLITS = ("dev", "val", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Partition proportions and seed. Both are recorded in the manifest."""

    seed: str
    dev: float = 0.2
    val: float = 0.2
    test: float = 0.6

    def validate(self) -> None:
        if not self.seed:
            raise ValueError(
                "split seed is required: an unrecorded split is not "
                "reproducible (specification 50)."
            )
        total = self.dev + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split proportions must sum to 1.0; got {total}.")
        if min(self.dev, self.val, self.test) < 0:
            raise ValueError("split proportions must be non-negative.")


def assign(question_id: str, config: SplitConfig) -> str:
    """Deterministically assign one question to a partition."""

    config.validate()

    digest = hashlib.sha256(
        f"{config.seed}:{question_id}".encode("utf-8")
    ).digest()

    # 53 bits keeps the ratio exactly representable as a float.
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)

    if position < config.dev:
        return "dev"
    if position < config.dev + config.val:
        return "val"
    return "test"


def split_pairs(
    pairs: Sequence[TestPair],
    config: SplitConfig,
) -> dict[str, list[TestPair]]:
    """Group pairs into partitions by their question id."""

    result: dict[str, list[TestPair]] = {name: [] for name in SPLITS}

    for pair in pairs:
        result[assign(pair.question_id, config)].append(pair)

    for name in SPLITS:
        result[name].sort(key=lambda p: p.pair_id)

    return result


def summarise(partitions: dict[str, list[TestPair]]) -> dict[str, dict[str, int]]:
    """Realised counts per partition, for the manifest."""

    return {
        name: {
            "pairs": len(items),
            "questions": len({pair.question_id for pair in items}),
            "eligible_pairs": sum(1 for pair in items if pair.is_eligible),
        }
        for name, items in partitions.items()
    }
