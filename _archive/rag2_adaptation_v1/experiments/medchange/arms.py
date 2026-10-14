"""Admission rules of the two standard arms (pure functions over a frozen pool).

B0 admits nothing (the model answers from its own knowledge). B1 is standard retrieval: the top ``BUDGET`` passages of the frozen pool by
cross-encoder rank. Both see the same frozen pool and answer with the same prompt (``prompts.py``). The arms of the adapted RAG² and of the
verification step are in ``rag2.py``.

Pool candidates carry ``rank`` (1 = best cross-encoder rerank) and ``pmid``; the date bounds ``lower`` and ``upper`` are not used by B0 or B1.

``SETTINGS`` is hashed into every B0 and B1 answer record and into the configuration recorded beside the answers files
(``generator.run_config``), so a silent change shows. It is kept exactly as it was when those answers were made: it still lists the
parameters of the recency arms of stage 1 (B2, B3, P and C1), whose code left the tree on 2026-10-13 (``docs/methodology.md`` §10), because the
hash in the committed records is the hash of this dictionary. Do not edit it; ``tests/test_arms.py`` compares the hash with those records.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

BUDGET = 5
ARMS = ("B0", "B1")
SETTINGS = {"budget": BUDGET, "half_life_days": 1095.0, "lambdas": {"B3": 0.5, "P": 0.5, "C1": 0.5},
            "date_point": "midpoint of availability bounds", "norm": "rank-normalised to [0,1]"}


def settings_hash() -> str:
    return hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:12]


def _top(pool: Sequence[dict], scores: Sequence[float], k: int) -> list[dict]:
    idx = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]["rank"], pool[i]["pmid"]))
    return [pool[i] for i in idx[:k]]


def admit(arm: str, pool: Sequence[dict], *, budget: int = BUDGET) -> list[dict]:
    """The passages (in context order) the arm gives the generator."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "B0" or not pool:
        return []
    return _top(pool, [-c["rank"] for c in pool], budget)
