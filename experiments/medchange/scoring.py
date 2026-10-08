"""Scoring helpers shared by the analysis and the report: accuracy with intervals, per-class recall and macro-F1,
paired comparisons (exact McNemar + paired bootstrap, ``evaluation/stats.py``) and Holm-adjusted families.

``paired(items, answers, a, b, kinds)`` compares arm ``a`` with arm ``b`` over the same questions; a result in a
family is *confirmed* only if its Holm-adjusted p is below ``ALPHA``, its 95% interval excludes 0 and the difference
is positive (``docs/evaluation.md``). The functions are unchanged from the earlier stages' analysis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

from evaluation.stats import holm, mcnemar, paired_bootstrap_ci

from .benchmark import LABELS
from .generate_answers import load_jsonl

ALPHA = 0.05
#: Decimals kept in stored rates and intervals. Six, not four: a rate stored to four decimals and then shown as a
#: percentage to one decimal can round the wrong way at a boundary (259/528 = 49.053% was shown as 49.0%).
DIGITS = 6


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, DIGITS), round(centre + half, DIGITS)]


def detectable_range(n: int) -> tuple[int, int]:
    """The smallest paired difference, in percentage points, that n questions can confirm with 80% power at
    alpha = .05 (normal approximation, 2.8 standard errors), when exactly one of the two systems is right on
    between 10% and 25% of the questions: (528) -> (4, 6), (208) -> (6, 10) (``docs/evaluation.md`` §4)."""
    return round(280 * math.sqrt(0.10 / n)), round(280 * math.sqrt(0.25 / n))


def load_answers(data: Path, split: str) -> dict:
    """{(item_id, arm): record} from ``answers_<split>.jsonl`` (the B0/B1 answers of the earlier stages)."""
    return {(r["item_id"], r["arm"]): r for r in load_jsonl(data / f"answers_{split}.jsonl")}


def correct_map(items: dict, answers: dict, arm: str, kinds: Sequence[str]) -> dict:
    return {i: int(answers[(i, arm)]["verdict"] == it["newest"]["label"])
            for i, it in items.items() if it["kind"] in kinds and (i, arm) in answers}


def class_stats(items: dict, answers: dict, arm: str, kinds: Sequence[str]) -> dict:
    ids = [i for i, it in items.items() if it["kind"] in kinds and (i, arm) in answers]
    recall, f1s = {}, []
    for label in LABELS:
        gold = [i for i in ids if items[i]["newest"]["label"] == label]
        predicted = [i for i in ids if answers[(i, arm)]["verdict"] == label]
        hit = sum(answers[(i, arm)]["verdict"] == label for i in gold)
        recall[label] = round(hit / len(gold), DIGITS) if gold else None
        precision = hit / len(predicted) if predicted else 0.0
        r = hit / len(gold) if gold else 0.0
        f1s.append(2 * precision * r / (precision + r) if precision + r else 0.0)
    return {"recall": recall, "macro_f1": round(sum(f1s) / len(f1s), DIGITS),
            "predicted_share": {label: round(sum(answers[(i, arm)]["verdict"] == label for i in ids) / len(ids), DIGITS)
                                for label in LABELS}}


def summarize(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    out = {}
    for arm in arms:
        row = {}
        for name, kinds in (("all", ("changed", "unchanged")), ("changed", ("changed",)),
                            ("unchanged", ("unchanged",))):
            c = correct_map(items, answers, arm, kinds)
            if c:
                row[name] = {"n": len(c), "accuracy": round(sum(c.values()) / len(c), DIGITS),
                             "wilson95": wilson(sum(c.values()), len(c))}
        if row:
            row.update(class_stats(items, answers, arm, ("changed", "unchanged")))
            ids = [i for i, it in items.items() if it["kind"] == "changed" and (i, arm) in answers]
            row["outdated_verdict_rate_changed"] = (
                round(sum(answers[(i, arm)]["verdict"] == items[i]["previous"]["label"] for i in ids) / len(ids), DIGITS)
                if ids else None)
            row["unparsed"] = sum(answers[(i, arm)]["verdict"] is None
                                  for i in items if (i, arm) in answers)
            out[arm] = row
    return out


def paired(items: dict, answers: dict, a: str, b: str, kinds: Sequence[str]) -> Optional[dict]:
    ca, cb = correct_map(items, answers, a, kinds), correct_map(items, answers, b, kinds)
    keys = sorted(set(ca) & set(cb))
    if len(keys) < 5:
        return None
    A_ = {k: ca[k] for k in keys}
    B_ = {k: cb[k] for k in keys}
    m = mcnemar(B_, A_)                       # baseline = b, proposed = a
    ci = paired_bootstrap_ci({k: float(v) for k, v in B_.items()}, {k: float(v) for k, v in A_.items()},
                             seed=f"{a}-{b}-{'+'.join(kinds)}", iterations=4000)
    return {"a": a, "b": b, "items": "+".join(kinds), "n": len(keys), "diff_a_minus_b": ci["difference"],
            "ci95": [ci["ci_low"], ci["ci_high"]], "a_only_correct": m.proposed_only,
            "b_only_correct": m.baseline_only, "mcnemar_p": round(m.p_value, 4)}


def holm_family(results: dict) -> dict:
    results = {k: v for k, v in results.items() if v}
    if not results:
        return {}
    adjusted = holm({k: v["mcnemar_p"] for k, v in results.items()})
    out = {}
    for k, v in results.items():
        confirmed = adjusted[k] < ALPHA and v["ci95"][0] > 0 and v["diff_a_minus_b"] > 0
        out[k] = dict(v, holm_p=round(adjusted[k], 4), confirmed=bool(confirmed))
    return out


def stable_difference(items: dict, answers: dict, a: str, b: str, stable_ids: Sequence[str]) -> Optional[dict]:
    """Accuracy of arm ``a`` minus arm ``b`` over the label-stable items (descriptive; no test)."""
    keep = {i: it for i, it in items.items() if i in set(stable_ids)}
    ca = correct_map(keep, answers, a, ("changed", "unchanged"))
    cb = correct_map(keep, answers, b, ("changed", "unchanged"))
    ids = sorted(set(ca) & set(cb))
    if not ids:
        return None
    return {"n": len(ids), "diff_a_minus_b": round(sum(ca[i] - cb[i] for i in ids) / len(ids), DIGITS)}
