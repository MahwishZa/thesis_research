"""Dev audit: the numbers behind the decision to design stage 2, recomputed from the committed answers.

    python -m experiments.medchange.dev_audit --out-dir experiments\\medchange\\results

Reads ``results/answers_dev.jsonl`` (committed) and ``data/benchmark.jsonl`` (rebuilt by
``build_benchmark``; gitignored) and reports, for the dev items only (no generation, no model):

* what a constant answer scores (the local model's lean towards SUPPORTED is the baseline to beat);
* recall of each gold class and the share of NOT ENOUGH INFORMATION answers, per arm;
* order sensitivity: pairs of answers whose admitted lists hold the same papers in another order;
* the mean age of B1's evidence by gold class;
* whether recalibrating B1's hard verdict, or adding evidence-age features, helps (repeated
  cross-validation with the same fitting as stage 2);
* a majority vote of three evidence arms.

This documents why stage 2 was designed as it was (``docs/log.md`` Phase 27). It is descriptive: no
number here tests the proposed system, and it refuses to run on the confirmatory split.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import itertools
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .generate_answers import load_jsonl
from .synthesis import LABELS, cv_accuracy, gold_indices, repeated_cv

HERE = Path(__file__).resolve().parent
EVIDENCE_ARMS = ("B1", "B2", "B3", "P", "C1")
ARMS = ("B0",) + EVIDENCE_ARMS
CV_FOLDS, CV_REPEATS, CV_SEED = 5, 50, 20261003


def gold(item: dict) -> str:
    return item["newest"]["label"]


def _kind_ids(items: dict, kind: Optional[str]) -> list[str]:
    return sorted(i for i, it in items.items() if kind is None or it["kind"] == kind)


def constant_baseline(items: dict, kind: Optional[str] = "changed") -> dict:
    """Share of each gold class: accuracy of always giving that verdict."""
    ids = _kind_ids(items, kind)
    counts = collections.Counter(gold(items[i]) for i in ids)
    return {"n": len(ids), "accuracy_if_always": {lab: counts.get(lab, 0) / len(ids) for lab in LABELS},
            "counts": {lab: counts.get(lab, 0) for lab in LABELS}}


def class_stats(items: dict, answers: dict, arm: str, kind: Optional[str] = "changed") -> dict:
    """Accuracy, recall of each gold class and the share of NOT ENOUGH INFORMATION answers."""
    ids = _kind_ids(items, kind)
    verdict = {i: answers[(i, arm)]["verdict"] for i in ids}
    recall = {}
    for lab in LABELS:
        sel = [i for i in ids if gold(items[i]) == lab]
        recall[lab] = sum(verdict[i] == lab for i in sel) / len(sel) if sel else None
    return {"n": len(ids), "accuracy": sum(verdict[i] == gold(items[i]) for i in ids) / len(ids),
            "recall": recall,
            "nei_share": sum(verdict[i] == "NOT ENOUGH INFORMATION" for i in ids) / len(ids)}


def order_sensitivity(answers: dict, item_ids: Sequence[str], arms: Sequence[str] = EVIDENCE_ARMS) -> dict:
    """Agreement of two arms' verdicts on the same question, by how their admitted lists relate:
    identical list, the same papers in another order, or partial overlap (Jaccard bands)."""
    cats: dict = collections.defaultdict(lambda: {"pairs": 0, "same_verdict": 0})
    identical_text = [0, 0]
    for i in item_ids:
        for a, b in itertools.combinations(arms, 2):
            x, y = answers[(i, a)], answers[(i, b)]
            sa, sb = x["admitted"], y["admitted"]
            if sa == sb:
                key = "identical list"
                identical_text[1] += 1
                identical_text[0] += x["text"] == y["text"]
            elif set(sa) == set(sb):
                key = "same papers, different order"
            else:
                union = set(sa) | set(sb)
                jaccard = len(set(sa) & set(sb)) / len(union) if union else 0.0
                key = ("overlap 0.67 or more" if jaccard >= 0.67 else
                       "overlap 0.25 to 0.66" if jaccard >= 0.25 else "overlap below 0.25")
            cats[key]["pairs"] += 1
            cats[key]["same_verdict"] += x["verdict"] == y["verdict"]
    out = {k: {"pairs": v["pairs"], "same_verdict": v["same_verdict"] / v["pairs"]} for k, v in cats.items()}
    return {"categories": out,
            "identical_list_identical_text": {"identical": identical_text[0], "pairs": identical_text[1]}}


def evidence_ages(item: dict, answer: dict) -> Optional[list[float]]:
    """Age in years of each admitted passage at the question date (upper bound of when it became public)."""
    cutoff = dt.date.fromisoformat(item["newest"]["date"])
    ages = [(cutoff - dt.date.fromisoformat(u)).days / 365.25 for u in answer.get("admitted_upper") or [] if u]
    return ages or None


def age_by_gold(items: dict, answers: dict, arm: str = "B1") -> dict:
    """Mean age of the arm's admitted evidence, by the item's gold class."""
    by: dict = {lab: [] for lab in LABELS}
    for i, it in items.items():
        ages = evidence_ages(it, answers[(i, arm)])
        if ages:
            by[gold(it)].append(float(np.mean(ages)))
    return {lab: {"n": len(v), "mean_age_years": (float(np.mean(v)) if v else None)} for lab, v in by.items()}


def _age_features(item: dict, answer: dict) -> list[float]:
    ages = np.array(evidence_ages(item, answer) or [0.0])
    return [float(ages.mean()), float(ages.min()), float((ages <= 3).mean()), float((ages <= 1).mean()),
            float(ages.std())]


def _onehot(answers: dict, ids: Sequence[str], arm: str) -> np.ndarray:
    return np.array([[answers[(i, arm)]["verdict"] == lab for lab in LABELS] for i in ids], dtype=float)


def recalibration_cv(items: dict, answers: dict, folds: int = CV_FOLDS, repeats: int = CV_REPEATS) -> dict:
    """Cross-validated accuracy (all dev items, and changed items) of fitting on B1's hard verdict and on
    evidence-age features, against the constant-prior fit and B1's raw accuracy."""
    ids = _kind_ids(items, None)
    y = gold_indices([items[i] for i in ids])
    changed = np.array([items[i]["kind"] == "changed" for i in ids])
    v1 = _onehot(answers, ids, "B1")
    ages = np.array([_age_features(items[i], answers[(i, "B1")]) for i in ids])
    designs = {"constant prior": np.zeros((len(ids), 0)), "B1 verdict": v1,
               "B1 verdict + evidence age": np.c_[v1, ages], "evidence age only": ages}
    out = {"n": len(ids), "n_changed": int(changed.sum()),
           "raw_B1_accuracy": float((v1.argmax(axis=1) == y).mean()),
           "raw_B1_accuracy_changed": float((v1.argmax(axis=1) == y)[changed].mean()), "variants": {}}
    for name, X in designs.items():
        pred = repeated_cv(X, y, folds, repeats, CV_SEED)
        out["variants"][name] = {"accuracy": cv_accuracy(pred, y), "accuracy_changed": cv_accuracy(pred, y, changed)}
    return out


def majority_vote(items: dict, answers: dict, arms: Sequence[str] = ("B1", "B2", "B3"),
                  kind: Optional[str] = "changed") -> dict:
    """Accuracy of the majority verdict of several arms (a tie goes to the first arm listed)."""
    ids = _kind_ids(items, kind)
    correct = 0
    for i in ids:
        votes = collections.Counter(answers[(i, a)]["verdict"] for a in arms)
        top = votes.most_common()
        verdict = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else answers[(i, arms[0])]["verdict"]
        correct += verdict == gold(items[i])
    return {"arms": list(arms), "n": len(ids), "accuracy": correct / len(ids)}


def audit(items: dict, answers: dict, folds: int = CV_FOLDS, repeats: int = CV_REPEATS) -> dict:
    changed = _kind_ids(items, "changed")
    return {
        "constant": {"changed": constant_baseline(items, "changed"), "all": constant_baseline(items, None)},
        "arms": {arm: {"changed": class_stats(items, answers, arm, "changed"),
                       "unchanged": class_stats(items, answers, arm, "unchanged")} for arm in ARMS},
        "order": order_sensitivity(answers, changed + _kind_ids(items, "unchanged")),
        "age_by_gold_B1": age_by_gold(items, answers, "B1"),
        "recalibration": recalibration_cv(items, answers, folds, repeats),
        "vote": {kind: majority_vote(items, answers, kind=kind) for kind in ("changed", "unchanged")},
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(rep: dict) -> str:
    c = rep["constant"]
    lines = ["# Dev audit (descriptive; recomputed from the committed answers)", "",
             "## Constant answers", "",
             "| Always answer | accuracy, changed items | accuracy, all dev items |", "|---|---|---|"]
    for lab in LABELS:
        lines.append(f"| {lab} | {_pct(c['changed']['accuracy_if_always'][lab])} | "
                     f"{_pct(c['all']['accuracy_if_always'][lab])} |")
    lines += ["", f"Changed items: n = {c['changed']['n']}; all dev items: n = {c['all']['n']}.", "",
              "## Per-class recall and abstention, changed items", "",
              "| Arm | accuracy | recall SUPPORTED | recall REFUTED | recall NOT ENOUGH INFORMATION | "
              "share answering NOT ENOUGH INFORMATION |", "|---|---|---|---|---|---|"]
    for arm, v in rep["arms"].items():
        s = v["changed"]
        lines.append(f"| {arm} | {_pct(s['accuracy'])} | {_pct(s['recall']['SUPPORTED'])} | "
                     f"{_pct(s['recall']['REFUTED'])} | {_pct(s['recall']['NOT ENOUGH INFORMATION'])} | "
                     f"{_pct(s['nei_share'])} |")
    lines += ["", "## Order sensitivity (pairs of evidence arms, same question, all dev items)", "",
              "| Admitted lists | pairs | same verdict |", "|---|---|---|"]
    for key, v in sorted(rep["order"]["categories"].items()):
        lines.append(f"| {key} | {v['pairs']} | {_pct(v['same_verdict'])} |")
    t = rep["order"]["identical_list_identical_text"]
    lines += ["", f"Identical lists in identical order give identical generated text in {t['identical']} of "
              f"{t['pairs']} pairs (greedy decoding is not bitwise reproducible).", "",
              "## Mean age of B1's evidence by gold class (years)", "", "| Gold | items | mean age |", "|---|---|---|"]
    for lab, v in rep["age_by_gold_B1"].items():
        age = "n/a" if v["mean_age_years"] is None else f"{v['mean_age_years']:.1f}"
        lines.append(f"| {lab} | {v['n']} | {age} |")
    r = rep["recalibration"]
    lines += ["", f"## Does refitting help? ({CV_FOLDS}-fold cross-validation, {CV_REPEATS} repeats, seed {CV_SEED}, "
              "the stage-2 fitting)", "",
              f"B1 raw accuracy: {_pct(r['raw_B1_accuracy'])} on all {r['n']} dev items, "
              f"{_pct(r['raw_B1_accuracy_changed'])} on {r['n_changed']} changed items.", "",
              "| Fitted on | accuracy, all | accuracy, changed |", "|---|---|---|"]
    for name, v in r["variants"].items():
        lines.append(f"| {name} | {_pct(v['accuracy'])} | {_pct(v['accuracy_changed'])} |")
    v = rep["vote"]
    lines += ["", f"Majority vote of {', '.join(v['changed']['arms'])}: {_pct(v['changed']['accuracy'])} on changed "
              f"items, {_pct(v['unchanged']['accuracy'])} on unchanged items.", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev",),
                    help="dev only: the confirmatory labels are not analysed before the frozen model exists")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--answers", default=str(HERE / "results" / "answers_dev.jsonl"))
    ap.add_argument("--out-dir", default=None, help="write dev_audit.json/.md here")
    ap.add_argument("--repeats", type=int, default=CV_REPEATS)
    args = ap.parse_args(argv)
    bench = Path(args.data_dir) / "benchmark.jsonl"
    if not bench.is_file():
        print(f"{bench} not found: run build_benchmark first", file=sys.stderr)
        return 2
    items = {r["item_id"]: r for r in load_jsonl(bench) if r["split"] == args.split and not r["likely_label_noise"]}
    answers = {(r["item_id"], r["arm"]): r for r in load_jsonl(Path(args.answers))}
    missing = [(i, a) for i in items for a in ARMS if (i, a) not in answers]
    if missing:
        print(f"{len(missing)} (item, arm) answers are missing, e.g. {missing[:3]}", file=sys.stderr)
        return 2
    rep = audit(items, answers, repeats=args.repeats)
    text = to_markdown(rep)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, body in (("dev_audit.json", json.dumps(rep, indent=2) + "\n"), ("dev_audit.md", text)):
            with open(out / name, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
        print(f"wrote {out / 'dev_audit.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
