"""Admission rules for the experimental arms (pure functions over a frozen pool).

Every arm sees the SAME frozen pool and differs only in which <= BUDGET passages
it admits and in what order. Settings are declared here, once, before any result
exists; ``SETTINGS`` is hashed into every answer record so a silent change shows.

Pool candidates carry ``rank`` (1 = best cross-encoder rerank), ``lower``/``upper``
(ISO bounds on first public availability) and, once computed, ``helpful`` (a
zero-shot P(helpful) from ``helpfulness.py``).

Relevance signal x recency:

    | relevance signal         | no recency | recency        |
    | MedCPT cross-encoder     | B1         | B3 (TempRALM-style) |
    | zero-shot helpfulness    | B2         | P (proposed)   |

C1 = P with dates shuffled within the pool (falsification control).
B0 admits nothing (the model's own knowledge).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from typing import Optional, Sequence

BUDGET = 5
HALF_LIFE_DAYS = 1095.0          # ~3 years, a typical Cochrane update horizon
LAMBDAS = {"B3": 0.5, "P": 0.5, "C1": 0.5}
ARMS = ("B0", "B1", "B2", "B3", "P", "C1")
SETTINGS = {"budget": BUDGET, "half_life_days": HALF_LIFE_DAYS, "lambdas": LAMBDAS,
            "date_point": "midpoint of availability bounds", "norm": "rank-normalised to [0,1]"}


def settings_hash() -> str:
    return hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:12]


def _date(iso: str) -> dt.date:
    return dt.date.fromisoformat(iso)


def point_date(c: dict) -> dt.date:
    lo, hi = _date(c["lower"]), _date(c["upper"])
    return lo + (hi - lo) / 2


def recency(c: dict, cutoff: str, half_life_days: float = HALF_LIFE_DAYS) -> float:
    """T = 2 ** (-age_days / H), age measured to the question date, clamped at 0."""
    age = max(0, (_date(cutoff) - point_date(c)).days)
    return 2.0 ** (-age / half_life_days)


def rank_normalise(values: Sequence[float]) -> list[float]:
    """Highest value -> 1.0, lowest -> 0.0, by rank (ties share the better rank)."""
    n = len(values)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: -values[i])
    out = [0.0] * n
    pos = 0
    for k, i in enumerate(order):
        if k > 0 and values[i] == values[order[k - 1]]:
            out[i] = out[order[k - 1]]
        else:
            pos = k
            out[i] = 1.0 - pos / (n - 1)
    return out


def _top(pool: Sequence[dict], scores: Sequence[float], k: int) -> list[dict]:
    idx = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]["rank"], pool[i]["pmid"]))
    return [pool[i] for i in idx[:k]]


def shuffle_dates(pool: Sequence[dict], seed: str) -> list[dict]:
    """Same pool, dates permuted among passages (relevance signals untouched)."""
    rng = random.Random(seed)
    dates = [(c["lower"], c["upper"]) for c in pool]
    rng.shuffle(dates)
    return [dict(c, lower=lo, upper=hi) for c, (lo, hi) in zip(pool, dates)]


def admit(arm: str, pool: Sequence[dict], cutoff: str, *, item_id: str = "",
          budget: int = BUDGET) -> list[dict]:
    """The passages (in context order) the arm gives the generator."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "B0" or not pool:
        return []
    if arm == "B1":
        return _top(pool, [-c["rank"] for c in pool], budget)
    if arm in ("B2", "P", "C1") and any("helpful" not in c for c in pool):
        raise ValueError(f"arm {arm} needs helpfulness scores on every candidate")
    if arm == "B2":
        return _top(pool, [c["helpful"] for c in pool], budget)
    if arm == "B3":
        rel = rank_normalise([-c["rank"] for c in pool])
        use = pool
    else:  # P, C1
        rel = rank_normalise([c["helpful"] for c in pool])
        use = shuffle_dates(pool, f"C1|{item_id}") if arm == "C1" else pool
    lam = LAMBDAS[arm]
    scores = [(1 - lam) * r + lam * recency(c, cutoff) for r, c in zip(rel, use)]
    chosen = _top(pool, scores, budget)
    return chosen
