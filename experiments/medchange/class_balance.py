"""Per-verdict behaviour of the systems: predicted shares, recall per verdict and macro-F1, with paired bootstrap intervals.

    python -m experiments.medchange.class_balance [--data-dir <benchmark folder>] [--results-dir <answers folder>]

Uses the answers already on file (no model is run) for the held-out ("confirm") and Alzheimer's disease ("ad") splits. **Exploratory:** the
metric was chosen after the accuracy results were known (``docs/protocol.md`` §10.1), so nothing here is a confirmatory result;
``protocol.md`` §10 fixes how it is tested on fresh questions. Macro-F1 is the unweighted mean of the F1 of the three verdicts (an F1 is 0
when a verdict is never predicted correctly; an unparsed answer is a fourth code that is never right). Intervals: 10,000 bootstrap resamples
of questions, seeded. The generator is seeded from the first 8 bytes of the seed text (``scoring.prefix_rng``), so every comparison of this file
shares the stream of resamples that begins "class-ba"; the intervals are valid bootstrap intervals and are kept as committed. The
pre-registered analysis (``analyze_fresh``) hashes the whole seed text instead (``scoring.hashed_rng``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import analysis as A
from .benchmark import LABELS
from .scoring import DIGITS, ITERATIONS, macro_f1, paired_interval, recall, vectors

HERE = Path(__file__).resolve().parent
SPLITS = ("confirm", "ad")
PAIRS = (("R2V", "R2"), ("R2C", "R2"), ("R2V", "R2C"))
SEED = "class-balance-v1"


def load(data: Path, results: Path, split: str):
    """(items, {(item id, arm): verdict}): the benchmark from ``data``, the answers from ``results``."""
    items, records = A.load_split(data, split, results)
    return items, {k: r["verdict"] for k, r in records.items()}


def analyse(data: Path, results: Path) -> dict:
    out = {"status": "exploratory; metric chosen after the accuracy results were known; see protocol §10", "iterations": ITERATIONS,
           "seed": SEED, "splits": {}}
    for split in SPLITS:
        items, answers = load(data, results, split)
        arms = [a for a in A.ARM_ORDER if any((i, a) in answers for i in items)]
        gold, pred = vectors(items, answers, arms)
        n = len(gold)
        rep = {"n": n, "gold_share": {l: round(float((gold == k).mean()), DIGITS) for k, l in enumerate(LABELS)}, "arms": {}, "paired": {}}
        for a, p in pred.items():
            rep["arms"][a] = {"macro_f1": round(macro_f1(gold, p), DIGITS),
                              "predicted_share": {l: round(float((p == k).mean()), DIGITS) for k, l in enumerate(LABELS)},
                              "recall": {l: round(recall(gold, p, k), DIGITS) for k, l in enumerate(LABELS)}}
        for a, b in PAIRS:
            if a in pred and b in pred:
                rep["paired"][f"{a} vs {b}"] = {
                    "macro_f1": paired_interval(gold, pred[a], pred[b], macro_f1, seed=f"{SEED}|{split}|{a}|{b}|f1", iterations=ITERATIONS),
                    "refuted_recall": paired_interval(gold, pred[a], pred[b], lambda g, p: recall(g, p, 1),
                                                      seed=f"{SEED}|{split}|{a}|{b}|rec", iterations=ITERATIONS)}
        out["splits"][split] = rep
    return out


def to_markdown(rep: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}%"
    L = ["# Per-verdict behaviour (exploratory)", "", f"{rep['status']}. Intervals: {rep['iterations']} bootstrap resamples of questions.", ""]
    for split, r in rep["splits"].items():
        L += [f"## {split} split (n = {r['n']})", "",
              "Gold shares: " + ", ".join(f"{l} {pct(v)}" for l, v in r["gold_share"].items()), "",
              "| System | Macro-F1 | Predicted SUPPORTED | REFUTED recall | NOT ENOUGH INFORMATION recall |", "|---|---|---|---|---|"]
        for a, v in r["arms"].items():
            L.append(f"| {a} | {v['macro_f1']:.3f} | {pct(v['predicted_share']['SUPPORTED'])} | {pct(v['recall']['REFUTED'])} | "
                     f"{pct(v['recall']['NOT ENOUGH INFORMATION'])} |")
        L += ["", "| Paired difference | Macro-F1 (95% interval) | REFUTED recall, points (95% interval) |", "|---|---|---|"]
        for k, v in r["paired"].items():
            f1, rc = v["macro_f1"], v["refuted_recall"]
            L.append(f"| {k} | {f1['difference']:+.3f} ({f1['ci95'][0]:+.3f} to {f1['ci95'][1]:+.3f}) | "
                     f"{100 * rc['difference']:+.1f} ({100 * rc['ci95'][0]:+.1f} to {100 * rc['ci95'][1]:+.1f}) |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    rep = analyse(Path(a.data_dir), Path(a.results_dir))
    out = Path(a.results_dir)
    (out / "class_balance.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = to_markdown(rep)
    (out / "class_balance.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
