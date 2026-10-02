"""Analysis of generated answers: per-arm accuracy, paired tests, and the dev gates.

    python -m experiments.medchange.analyze --split dev

Primary outcome: parsed verdict == gold NEWEST verdict (unparsed counts wrong and is
reported). Key secondary: verdict == PREVIOUS (outdated) verdict, changed items only.
Safety: accuracy on unchanged items. Paired comparisons use the exact McNemar test and a
question-resampled bootstrap CI (``evaluation/stats.py``); only arms present in the
answers file are compared. Gates are the pre-stated ones in docs/next_phase_plan.md:

  G2  B1 changes the verdict of >= 20% of dev items relative to B0.
  G3  P - B2 >= +5 pp on changed items AND P - C1 >= +2.5 pp (C1 must not reproduce it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from evaluation.stats import mcnemar, paired_bootstrap_ci

from .generate_answers import load_jsonl

HERE = Path(__file__).resolve().parent
G2_MIN_CHANGE = 0.20
G3_MIN_GAIN = 0.05
G3_MIN_OVER_C1 = 0.025
PAIRS = (("B1", "B0"), ("B2", "B1"), ("B3", "B1"), ("P", "B2"), ("P", "B3"), ("P", "B1"), ("P", "C1"))


def correctness(items: dict, answers: dict, arm: str, kind: str, which: str = "newest") -> dict:
    """{item_id: 0/1} for items of ``kind`` that have an answer from ``arm``."""
    out = {}
    for iid, it in items.items():
        if it["kind"] != kind or (iid, arm) not in answers:
            continue
        gold = it[which]["label"]
        out[iid] = int(answers[(iid, arm)]["verdict"] == gold)
    return out


def summarize(items: dict, answers: dict, arms: list[str]) -> dict:
    out = {}
    for arm in arms:
        row = {}
        for kind in ("changed", "unchanged"):
            c = correctness(items, answers, arm, kind)
            if not c:
                continue
            ids = list(c)
            row[kind] = {
                "n": len(ids),
                "accuracy": round(sum(c.values()) / len(ids), 4),
                "unparsed": sum(answers[(i, arm)]["verdict"] is None for i in ids),
            }
            if kind == "changed":
                o = correctness(items, answers, arm, kind, "previous")
                row[kind]["outdated_match"] = round(sum(o.values()) / len(o), 4)
        if row:
            row["mean_seconds"] = round(sum(answers[k]["seconds"] for k in answers if k[1] == arm)
                                        / sum(1 for k in answers if k[1] == arm), 1)
            out[arm] = row
    return out


def paired(items: dict, answers: dict, a: str, b: str, kind: str = "changed") -> Optional[dict]:
    ca, cb = correctness(items, answers, a, kind), correctness(items, answers, b, kind)
    keys = sorted(set(ca) & set(cb))
    if len(keys) < 5:
        return None
    A_ = {k: ca[k] for k in keys}
    B_ = {k: cb[k] for k in keys}
    m = mcnemar(B_, A_)           # baseline = b, proposed = a
    ci = paired_bootstrap_ci({k: float(v) for k, v in B_.items()}, {k: float(v) for k, v in A_.items()},
                             seed=f"{a}-{b}-{kind}", iterations=4000)
    return {"a": a, "b": b, "kind": kind, "n": len(keys), "diff_a_minus_b": ci["difference"],
            "ci95": [ci["ci_low"], ci["ci_high"]], "a_only_correct": m.proposed_only,
            "b_only_correct": m.baseline_only, "mcnemar_p": round(m.p_value, 4)}


def verdict_change_rate(items: dict, answers: dict, a: str, b: str) -> Optional[float]:
    keys = [i for i in items if (i, a) in answers and (i, b) in answers]
    if not keys:
        return None
    return sum(answers[(i, a)]["verdict"] != answers[(i, b)]["verdict"] for i in keys) / len(keys)


def gates(items: dict, answers: dict) -> dict:
    out = {}
    r = verdict_change_rate(items, answers, "B1", "B0")
    if r is not None:
        out["G2_B1_vs_B0_verdict_change_rate"] = round(r, 4)
        out["G2"] = "PASS" if r >= G2_MIN_CHANGE else "FAIL"
    pb, pc = paired(items, answers, "P", "B2"), paired(items, answers, "P", "C1")
    if pb and pc:
        ok = pb["diff_a_minus_b"] >= G3_MIN_GAIN and pc["diff_a_minus_b"] >= G3_MIN_OVER_C1
        out["G3_P_minus_B2"] = pb["diff_a_minus_b"]
        out["G3_P_minus_C1"] = pc["diff_a_minus_b"]
        out["G3"] = "PASS" if ok else "FAIL"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--answers", default=None)
    args = ap.parse_args(argv)
    d = HERE / "data"
    items = {r["item_id"]: r for r in load_jsonl(d / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]}
    rows = load_jsonl(Path(args.answers or d / f"answers_{args.split}.jsonl"))
    answers = {(r["item_id"], r["arm"]): r for r in rows}
    arms = sorted({a for _, a in answers}, key=lambda a: ("B0", "B1", "B2", "B3", "P", "C1").index(a))
    report = {"split": args.split, "n_answers": len(rows), "per_arm": summarize(items, answers, arms),
              "paired_changed": [p for a, b in PAIRS if (p := paired(items, answers, a, b))],
              "paired_unchanged": [p for a, b in PAIRS if (p := paired(items, answers, a, b, "unchanged"))],
              "gates": gates(items, answers)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
