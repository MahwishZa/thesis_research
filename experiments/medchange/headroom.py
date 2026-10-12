"""Headroom from the MedChange authors' own released LLM answers (no RAG).

Before spending compute on retrieval arms: how often does a model WITHOUT
retrieval already give the current verdict, and how often the outdated one?
Uses ``Code/GeneratedAnswers/<model>_answers.txt`` (one line per MedRevQA row,
for the single-line files) and the benchmark built by ``build_benchmark``.

    python -m experiments.medchange.headroom --medchange-dir <clone>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .released_answers import MODELS, read_answers

HERE = Path(__file__).resolve().parent
#: The splits of the main benchmark; the questions of the ``ad`` and ``fresh`` splits appended to the same file are not part of this headroom check.
MAIN_SPLITS = ("dev", "confirm")


def score(items: list[dict], answers: list) -> dict:
    out = {}
    for kind in ("changed", "unchanged"):
        sub = [i for i in items if i["kind"] == kind and not i["likely_label_noise"]]
        if not sub:
            continue
        preds = [answers[i["newest"]["row"]] for i in sub]
        out[kind] = {
            "n": len(sub),
            "newest_verdict_accuracy": round(sum(p == i["newest"]["label"]
                                                 for p, i in zip(preds, sub)) / len(sub), 4),
            "previous_verdict_match": round(sum(p == i["previous"]["label"]
                                                for p, i in zip(preds, sub)) / len(sub), 4),
            "unparsed": sum(p is None for p in preds),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--medchange-dir", required=True)
    ap.add_argument("--benchmark", default=str(HERE / "data" / "benchmark.jsonl"))
    args = ap.parse_args(argv)
    with open(args.benchmark, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    items = [i for i in items if i["split"] in MAIN_SPLITS]
    ans_dir = Path(args.medchange_dir) / "Code" / "GeneratedAnswers"
    report = {}
    for m in MODELS:
        answers = read_answers(ans_dir, m)
        if answers is not None:
            report[m] = score(items, answers)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
