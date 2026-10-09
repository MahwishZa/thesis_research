"""Exploratory subgroup of the dementia run: the questions whose text names Alzheimer's disease.

    python -m experiments.medchange.subgroup_ad [--data-dir <benchmark folder>] [--results-dir <answers folder>]

Uses the answers already on file (no model is run). The subgroup is defined by the wording of the question alone ("alzheimer",
any case); it was chosen after the dementia run and is exploratory: 48 questions can confirm only very large differences, so
nothing here is read as meeting the requirement. Writes ``ad_subgroup_alzheimer.json`` and ``.md`` to the results folder."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from . import analyze_rag2 as A
from . import class_balance as CB
from .generate_answers import load_jsonl
from .scoring import DIGITS, detectable_range, summarize

HERE = Path(__file__).resolve().parent
NAMES_ALZHEIMER = re.compile(r"alzheimer", re.IGNORECASE)
COMPARISONS = (("R2V", "R2"), ("R2C", "R2"), ("R2V", "R2C"), ("R2", "B1"), ("R2V", "B1"), ("R2V", "B0"))


def subgroup(data: Path, results: Path) -> dict:
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl") if r["split"] == "ad" and not r["likely_label_noise"]}
    answers = {}
    for name in ("answers_ad.jsonl", "rag2_answers_ad.jsonl"):
        for r in load_jsonl(results / name):
            if r["item_id"] in items and r["arm"] in A.ARM_ORDER:
                answers[(r["item_id"], r["arm"])] = r
    sub = {i: it for i, it in items.items() if NAMES_ALZHEIMER.search(it["question"])}
    arms = A.present_arms(answers)
    gold = Counter(it["newest"]["label"] for it in sub.values())
    constant_label, constant_n = gold.most_common(1)[0]
    gen = summarize(sub, answers, arms)
    comparisons = {}
    for a, b in COMPARISONS:
        r = A.paired(sub, answers, a, b, A.BOTH)
        if r:
            comparisons[f"{a} vs {b}"] = r
    pred = {a: {i: answers[(i, a)]["verdict"] for i in sub if (i, a) in answers} for a in arms}
    gold_v, vec = CB.vectors(sub, {(i, a): v for a, m in pred.items() for i, v in m.items()}, arms)
    macro = CB.paired_interval(gold_v, vec["R2V"], vec["R2"], CB.macro_f1, seed="ad-subgroup-macro-f1") if {"R2V", "R2"} <= set(vec) else None
    return {"split": "ad", "subgroup": "question text names Alzheimer's disease", "dementia_set_items": len(items), "n": len(sub),
            "gold": dict(gold), "constant_answer": {"label": constant_label, "accuracy": round(constant_n / len(sub), DIGITS)},
            "accuracy": {a: gen[a]["all"] for a in arms}, "recall": {a: gen[a]["recall"] for a in arms},
            "macro_f1": {a: gen[a]["macro_f1"] for a in arms}, "predicted_share": {a: gen[a]["predicted_share"] for a in arms},
            "macro_f1_R2V_minus_R2": macro,
            "comparisons": comparisons, "detectable_range_points": list(detectable_range(len(sub))),
            "status": "exploratory; chosen after the run; the requirement is not read from it"}


def to_markdown(rep: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}%"
    pp = lambda x: f"{100 * x:+.1f}".replace("-", "−")
    L = [f"# Exploratory subgroup of the dementia run: questions naming Alzheimer's disease (n = {rep['n']} of {rep['dementia_set_items']})", "",
         "Gold verdicts: " + ", ".join(f"{k} {v}" for k, v in rep["gold"].items()) +
         f". Constant answer ({rep['constant_answer']['label']}): {pct(rep['constant_answer']['accuracy'])}.", "",
         "| System | Accuracy | 95% interval |", "|---|---|---|"]
    for a, v in rep["accuracy"].items():
        L.append(f"| {a} | {pct(v['accuracy'])} | {pct(v['wilson95'][0])} to {pct(v['wilson95'][1])} |")
    L += ["", "| Paired difference | Points | 95% interval | Only first right | Only second right | p (exact) |", "|---|---|---|---|---|---|"]
    for k, r in rep["comparisons"].items():
        L.append(f"| {k} | {pp(r['diff_a_minus_b'])} | {pp(r['ci95'][0])} to {pp(r['ci95'][1])} | {r['a_only_correct']} | "
                 f"{r['b_only_correct']} | {r['mcnemar_p']:.2f} |")
    lo, hi = rep["detectable_range_points"]
    if rep.get("macro_f1_R2V_minus_R2"):
        m = rep["macro_f1_R2V_minus_R2"]
        L += ["", "| System | Macro-F1 | Predicted SUPPORTED | REFUTED recall |", "|---|---|---|---|"]
        for a in rep["accuracy"]:
            L.append(f"| {a} | {rep['macro_f1'][a]:.3f} | {pct(rep['predicted_share'][a]['SUPPORTED'])} | {pct(rep['recall'][a]['REFUTED'])} |")
        L += ["", f"Macro-F1, R2V − R2: {m['difference']:+.3f} (95% interval {m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f}). Exploratory: the metric was chosen after the accuracy results were known."]
    L += ["", f"With {rep['n']} questions only paired differences of about {lo} to {hi} points or more could be confirmed with 80% power "
          f"(`evaluation.md` §4). {rep['status'].capitalize()}. The 1-point requirement is therefore neither met nor failed here; "
          "the point estimates are descriptive."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    rep = subgroup(Path(a.data_dir), Path(a.results_dir))
    out = Path(a.results_dir)
    (out / "ad_subgroup_alzheimer.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = to_markdown(rep)
    (out / "ad_subgroup_alzheimer.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
