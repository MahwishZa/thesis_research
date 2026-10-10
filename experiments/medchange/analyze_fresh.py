"""Blinded analysis of the pre-registered test on fresh questions (``docs/protocol.md`` §10).

    python -m experiments.medchange.analyze_fresh [--data-dir <folder>] [--results-dir <folder>]

**Blinded:** until every one of the N questions has an answer for R2, R2C and R2V, this prints only how many are answered and computes nothing
else, so that no partial result can be seen. When the run is complete it computes, once and exactly as pre-registered:

* the primary outcome: the paired difference of the macro-F1 of R2V and R2 with a 95% interval (10,000 bootstrap resamples of questions,
  seed ``fresh-macro-f1``); **confirmed if the lower end is above 0**, otherwise "positive but not confirmed" (point estimate above 0) or "no
  evidence"; the upper end is stated (below 0.02 excludes a gain of 0.02 or more);
* the secondary outcomes: macro-F1 of R2V − R2C; REFUTED recall and the predicted-SUPPORTED share; verdict accuracy R2V − R2 (exact McNemar,
  bootstrap interval); how many answers R2V changed and how many it fixed or broke; the label audit when it exists;
* the descriptive Alzheimer's/dementia comparison: the macro-F1 difference of the 208 dementia questions from ``class_balance.json`` and
  whether its sign agrees (no confirmatory claim).

It writes ``rag2_analysis_fresh.json`` and ``RAG2_FINDINGS_FRESH.md`` to the results folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import analyze_rag2 as A
from .benchmark import LABELS
from .class_balance import macro_f1, recall
from .scoring import paired, summarize

HERE = Path(__file__).resolve().parent
ARMS = ("R2", "R2C", "R2V")
EXPECTED = 850
ITERATIONS = 10000
PRIMARY_SEED = "fresh-macro-f1"


class Blinded(Exception):
    """The run is not complete: no result may be computed or shown."""


def completeness(items: dict, answers: dict) -> dict:
    answered = {a: sum((i, a) in answers for i in items) for a in ARMS}
    return {"items": len(items), "answered": answered, "complete": bool(items) and all(v == len(items) for v in answered.values())}


def _rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16))


def paired_interval(gold, pa, pb, metric, seed: str, iterations: int = ITERATIONS) -> dict:
    rng = _rng(seed)
    n = len(gold)
    point = metric(gold, pa) - metric(gold, pb)
    diffs = np.empty(iterations)
    for k in range(iterations):
        s = rng.integers(0, n, n)
        diffs[k] = metric(gold[s], pa[s]) - metric(gold[s], pb[s])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def read_primary(d: dict) -> dict:
    lo, hi = d["ci95"]
    if lo > 0:
        reading = "confirmed"
    elif d["difference"] > 0:
        reading = "positive but not confirmed"
    else:
        reading = "no evidence"
    return {"reading": reading, "gain_of_0.02_or_more_excluded": hi < 0.02}


def analyse(items: dict, answers: dict, *, expected: int = EXPECTED, iterations: int = ITERATIONS,
            class_balance: Optional[dict] = None, audit: Optional[dict] = None) -> dict:
    comp = completeness(items, answers)
    if not comp["complete"] or len(items) != expected:
        raise Blinded(f"not complete: {comp['answered']} of {len(items)} answered (expected {expected} questions); no result is computed")
    ids = sorted(items)
    idx = {l: k for k, l in enumerate(LABELS)}
    gold = np.array([idx[items[i]["newest"]["label"]] for i in ids])
    pred = {a: np.array([idx.get(answers[(i, a)]["verdict"], len(LABELS)) for i in ids]) for a in ARMS}      # unparsed: a fourth code, wrong everywhere
    f1 = lambda a: round(macro_f1(gold, pred[a]), 4)
    primary = paired_interval(gold, pred["R2V"], pred["R2"], macro_f1, PRIMARY_SEED, iterations)
    primary.update(read_primary(primary), metric="macro-F1, R2V minus R2")
    sec = {
        "macro_f1_R2V_minus_R2C": paired_interval(gold, pred["R2V"], pred["R2C"], macro_f1, "fresh-macro-f1-R2V-R2C", iterations),
        "refuted_recall_R2V_minus_R2": paired_interval(gold, pred["R2V"], pred["R2"], lambda g, p: recall(g, p, 1), "fresh-refuted-R2V-R2", iterations),
        "accuracy_R2V_minus_R2": paired(items, answers, "R2V", "R2", A.BOTH),
        "accuracy_R2V_minus_R2C": paired(items, answers, "R2V", "R2C", A.BOTH)}
    verification = A.verifier_rows(items, answers, ARMS)
    arms = {a: {"macro_f1": f1(a), "predicted_share": {l: round(float((pred[a] == k).mean()), 4) for k, l in enumerate(LABELS)},
                "unparsed": int((pred[a] == len(LABELS)).sum()),
                "recall": {l: round(recall(gold, pred[a], k), 4) for k, l in enumerate(LABELS)}} for a in ARMS}
    rep = {"split": "fresh", "n": len(ids), "iterations": iterations, "primary": primary, "secondary": sec, "arms": arms,
           "accuracy": {a: v["all"] for a, v in summarize(items, answers, ARMS).items()},
           "gold_share": {l: round(float((gold == k).mean()), 4) for k, l in enumerate(LABELS)}, "verification": verification}
    if class_balance:
        ad = class_balance["splits"]["ad"]["paired"]["R2V vs R2"]["macro_f1"]
        rep["alzheimers_dementia_descriptive"] = {
            "n": class_balance["splits"]["ad"]["n"], "macro_f1_R2V_minus_R2": ad,
            "sign_agrees_with_fresh": (ad["difference"] > 0) == (primary["difference"] > 0),
            "note": "seen before the hypothesis was formed: description, not confirmation"}
    if audit:
        rep["label_audit"] = {k: audit.get(k) for k in ("n_items", "n_unparsed", "agreement", "kappa", "per_gold_class")}
    return rep


def to_markdown(rep: dict) -> str:
    p = rep["primary"]
    f = lambda x: f"{x:+.3f}"
    L = ["# Pre-registered test on fresh questions (run once)", "",
         f"{rep['n']} fresh questions; arms R2, R2C and R2V; decision rule fixed in `docs/protocol.md` §10 before any question was answered.", "",
         "## Primary outcome", "",
         f"Macro-F1, R2V − R2 = **{f(p['difference'])}** (95% interval {f(p['ci95'][0])} to {f(p['ci95'][1])}; {rep['iterations']} bootstrap resamples).",
         f"Reading: **{p['reading']}**." + (" A gain of 0.02 or more is excluded." if p["gain_of_0.02_or_more_excluded"] else ""), "",
         "## Secondary outcomes (no confirmatory claim)", ""]
    s = rep["secondary"]
    L.append(f"* Macro-F1, R2V − R2C: {f(s['macro_f1_R2V_minus_R2C']['difference'])} ({f(s['macro_f1_R2V_minus_R2C']['ci95'][0])} to {f(s['macro_f1_R2V_minus_R2C']['ci95'][1])}).")
    r = s["refuted_recall_R2V_minus_R2"]
    L.append(f"* REFUTED recall, R2V − R2: {100 * r['difference']:+.1f} points ({100 * r['ci95'][0]:+.1f} to {100 * r['ci95'][1]:+.1f}).")
    for k, lab in (("accuracy_R2V_minus_R2", "R2V − R2"), ("accuracy_R2V_minus_R2C", "R2V − R2C")):
        a = s[k]
        if a:
            L.append(f"* Verdict accuracy, {lab}: {100 * a['diff_a_minus_b']:+.1f} points (95% interval {100 * a['ci95'][0]:+.1f} to {100 * a['ci95'][1]:+.1f}; "
                     f"exact McNemar p = {a['mcnemar_p']:.4f}).")
    L += ["", "| System | Accuracy | Macro-F1 | Predicted SUPPORTED | REFUTED recall |", "|---|---|---|---|---|"]
    for a, v in rep["arms"].items():
        L.append(f"| {a} | {100 * rep['accuracy'][a]['accuracy']:.1f}% | {v['macro_f1']:.3f} | {100 * v['predicted_share']['SUPPORTED']:.1f}% | "
                 f"{100 * v['recall']['REFUTED']:.1f}% |")
    unp = {a: v["unparsed"] for a, v in rep["arms"].items() if v["unparsed"]}
    if unp:
        L += ["", "Unparsed answers (counted wrong): " + ", ".join(f"{a} {n}" for a, n in unp.items()) + "."]
    v = rep["verification"].get("R2V")
    if v:
        L += ["", f"R2V changed {100 * v['changed_rate']:.1f}% of the answers it verified: {v['changes_fixed']} fixed, {v['changes_broke']} broken."]
    d = rep.get("alzheimers_dementia_descriptive")
    if d:
        m = d["macro_f1_R2V_minus_R2"]
        L += ["", "## Alzheimer's disease and related dementias (descriptive)", "",
              f"The {d['n']} dementia questions: macro-F1, R2V − R2 = {f(m['difference'])} ({f(m['ci95'][0])} to {f(m['ci95'][1])}). "
              f"The sign {'agrees' if d['sign_agrees_with_fresh'] else 'does not agree'} with the fresh result. {d['note'].capitalize()}."]
    au = rep.get("label_audit")
    if au:
        L += ["", "## Label audit (sample)", "", f"Agreement with an independent model {100 * au['agreement']:.1f}%, kappa {au['kappa']} ({au['n_items']} questions)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--expected", type=int, default=EXPECTED)
    a = ap.parse_args(argv)
    data, results = Path(a.data_dir), Path(a.results_dir)
    items, answers = A.load(data, "fresh")
    try:
        cbp, aup = results / "class_balance.json", results / "label_audit_fresh.json"
        rep = analyse(items, answers, expected=a.expected,
                      class_balance=json.loads(cbp.read_text(encoding="utf-8")) if cbp.is_file() else None,
                      audit=json.loads(aup.read_text(encoding="utf-8")) if aup.is_file() else None)
    except Blinded as why:
        print(f"BLINDED: {why}")
        return 3
    results.mkdir(parents=True, exist_ok=True)
    (results / "rag2_analysis_fresh.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = to_markdown(rep)
    (results / "RAG2_FINDINGS_FRESH.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
