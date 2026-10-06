"""Analysis of generated answers: per-arm accuracy, paired tests, and the dev gates.

    python -m experiments.medchange.analyze --split dev

Primary outcome: parsed verdict == gold NEWEST verdict (unparsed counts wrong and is
reported). Key secondary: verdict == PREVIOUS (outdated) verdict, changed items only.
Safety: accuracy on unchanged items. Paired comparisons use the exact McNemar test and a
question-resampled bootstrap CI (``evaluation/stats.py``); only arms present in the
answers file are compared. Gates are the pre-stated ones of the stage-1 protocol (the file experiment_plan.md at commit 92e3aaf):

Retrieval-level metrics (what each arm admitted; manipulation checks, never outcomes):
share of admitted passages that surely first appeared inside the update window (after the
previous review version, on or before the newest), items with any such passage, mean
passage age, and overlap with B1's admitted set.

  G2  B1 changes the verdict of >= 20% of dev items relative to B0.
  G3  P - B2 >= +5 pp on changed items AND P - C1 >= +2.5 pp (C1 must not reproduce it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from evaluation.stats import holm, mcnemar, paired_bootstrap_ci

from .generate_answers import config_path, load_jsonl

HERE = Path(__file__).resolve().parent
G2_MIN_CHANGE = 0.20
G3_MIN_GAIN = 0.05
G3_MIN_OVER_C1 = 0.025
CONFIRMATORY = (("P", "B1"), ("P", "B2"), ("P", "B3"))
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


def confirmatory_family(items: dict, answers: dict) -> dict:
    """Primary outcome, changed items, P vs B1/B2/B3: raw exact-McNemar p and Holm-adjusted p."""
    results = {f"{a}-{b}": paired(items, answers, a, b) for a, b in CONFIRMATORY}
    results = {k: v for k, v in results.items() if v}
    if not results:
        return {}
    adjusted = holm({k: v["mcnemar_p"] for k, v in results.items()})
    return {k: dict(v, holm_p=round(adjusted[k], 4)) for k, v in results.items()}


def verdict_change_rate(items: dict, answers: dict, a: str, b: str) -> Optional[float]:
    keys = [i for i in items if (i, a) in answers and (i, b) in answers]
    if not keys:
        return None
    return sum(answers[(i, a)]["verdict"] != answers[(i, b)]["verdict"] for i in keys) / len(keys)


def _in_window(c: dict, previous: str, newest: str) -> bool:
    """Surely first public after the previous version and on or before the newest."""
    return previous < c["lower"] and c["upper"] <= newest


def retrieval_metrics(items: dict, pools: dict, answers: dict, arms: list[str]) -> dict:
    """Per arm and item kind: what the arm admitted, judged against the frozen pools."""
    from datetime import date
    from .arms import point_date
    out: dict = {}
    for arm in arms:
        row = {}
        for kind in ("changed", "unchanged"):
            ids = [i for i, it in items.items()
                   if it["kind"] == kind and (i, arm) in answers and i in pools]
            if not ids:
                continue
            shares, any_window, ages, jac = [], 0, [], []
            for i in ids:
                it, cands = items[i], {c["pmid"]: c for c in pools[i]["candidates"]}
                admitted = [cands[p] for p in answers[(i, arm)]["admitted"] if p in cands]
                flags = [_in_window(c, it["previous"]["date"], it["newest"]["date"]) for c in admitted]
                if admitted:
                    shares.append(sum(flags) / len(admitted))
                any_window += any(flags)
                cutoff = date.fromisoformat(it["newest"]["date"])
                ages += [max(0, (cutoff - point_date(c)).days) / 365.25 for c in admitted]
                if arm != "B1" and (i, "B1") in answers:
                    a, b = set(answers[(i, arm)]["admitted"]), set(answers[(i, "B1")]["admitted"])
                    if a | b:
                        jac.append(len(a & b) / len(a | b))
            row[kind] = {
                "n": len(ids),
                "mean_admitted": round(sum(len(answers[(i, arm)]["admitted"]) for i in ids) / len(ids), 2),
                "update_window_share": round(sum(shares) / len(shares), 4) if shares else None,
                "items_with_update_window_evidence": round(any_window / len(ids), 4),
                "mean_age_years": round(sum(ages) / len(ages), 2) if ages else None,
                "jaccard_with_B1": round(sum(jac) / len(jac), 4) if jac else None,
            }
        if row:
            out[arm] = row
    return out


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
    ap.add_argument("--frozen", default=None, help="frozen pools (default data/frozen_<split>.jsonl)")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=None,
                    help="write the JSON report to this file (UTF-8, LF) instead of stdout")
    args = ap.parse_args(argv)
    d = Path(args.data_dir)
    items = {r["item_id"]: r for r in load_jsonl(d / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]}
    rows = load_jsonl(Path(args.answers or d / f"answers_{args.split}.jsonl"))
    answers = {(r["item_id"], r["arm"]): r for r in rows}
    config = config_path(Path(args.answers or d / f"answers_{args.split}.jsonl"))
    arms = sorted({a for _, a in answers}, key=lambda a: ("B0", "B1", "B2", "B3", "P", "C1").index(a))
    report = {"split": args.split, "n_answers": len(rows), "per_arm": summarize(items, answers, arms),
              "paired_changed": [p for a, b in PAIRS if (p := paired(items, answers, a, b))],
              "paired_unchanged": [p for a, b in PAIRS if (p := paired(items, answers, a, b, "unchanged"))],
              "confirmatory_family_changed": confirmatory_family(items, answers),
              "gates": gates(items, answers)}
    if config.exists():
        report["generation_config"] = json.loads(config.read_text(encoding="utf-8"))
    frozen = Path(args.frozen or d / f"frozen_{args.split}.jsonl")
    if frozen.exists():
        pools = {r["item_id"]: r for r in load_jsonl(frozen)}
        report["retrieval"] = retrieval_metrics(items, pools, answers, arms)
    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
