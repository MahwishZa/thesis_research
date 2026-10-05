"""Admission rules for the experimental arms (pure functions over a frozen pool).

Every arm sees the SAME frozen pool and differs only in which <= BUDGET passages
it admits and in what order. Settings are declared here, once, before any result
exists; ``SETTINGS`` is hashed into every answer record so a silent change shows.

Pool candidates carry ``rank`` (1 = best cross-encoder rerank), ``lower``/``upper``
(ISO bounds on first public availability) and, once computed, ``helpful`` (a
zero-shot P(helpful) from ``helpfulness.py``).

The recency term T and the admission score A = (1 - lambda) * rho + lambda * T are NOT
re-implemented here: they are computed by the project's reference implementation of the
proposed system, ``src.proposed.temporal.TemporalPolicy`` and
``src.proposed.scorer.AdmissionScorer``, so the formula exists in exactly one place.
Here rho is the within-pool rank normalisation of the arm's relevance signal (ties broken
by cross-encoder rank), and passages are admitted by top-``BUDGET`` score (no theta
threshold: the MedChange arms are fixed-budget, see the stage-1 protocol, the file experiment_plan.md at commit 92e3aaf).

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
from typing import Sequence

from src.common.evidence import Candidate, Evidence
from src.proposed.scorer import AdmissionScorer
from src.proposed.temporal import TemporalPolicy

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


def _evidence(c: dict) -> Evidence:
    return Evidence(evidence_id=c["pmid"], text="", source_tier="pubmed_abstract",
                    publication_date=point_date(c))


def recency(c: dict, cutoff: str, half_life_days: float = HALF_LIFE_DAYS) -> float:
    """T = 2 ** (-age_days / H), age from the passage to the question date, clamped at 0
    (``src.proposed.temporal.TemporalPolicy``)."""
    policy = TemporalPolicy(half_life_days=half_life_days, undated_score=0.0)
    return policy.score(_evidence(c), question="", question_date=_date(cutoff)).score


def relevance_ranks(values: Sequence[float], pool: Sequence[dict]) -> list[int]:
    """1-based ranks of ``values`` (highest = 1), ties broken by cross-encoder rank then
    PMID, so ranks are contiguous 1..N as ``AdmissionScorer`` requires."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], pool[i]["rank"], pool[i]["pmid"]))
    ranks = [0] * len(values)
    for r, i in enumerate(order, 1):
        ranks[i] = r
    return ranks


def admission_scores(pool: Sequence[dict], relevance: Sequence[float], dates_from: Sequence[dict],
                     cutoff: str, lam: float) -> list[float]:
    """A(s) for every passage: relevance rank from ``relevance``, dates from ``dates_from``."""
    scorer = AdmissionScorer(temporal_weight=lam)
    ranks = relevance_ranks(relevance, pool)
    out = []
    for c, d, r in zip(pool, dates_from, ranks):
        cand = Candidate(evidence=_evidence(c), rerank_score=0.0, rerank_rank=r)
        out.append(scorer.score(cand, temporal=recency(d, cutoff), candidate_count=len(pool)).total)
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
        rel = [-c["rank"] for c in pool]
        dates = pool
    else:  # P, C1
        rel = [c["helpful"] for c in pool]
        dates = shuffle_dates(pool, f"C1|{item_id}") if arm == "C1" else pool
    scores = admission_scores(pool, rel, dates, cutoff, LAMBDAS[arm])
    return _top(pool, scores, budget)
