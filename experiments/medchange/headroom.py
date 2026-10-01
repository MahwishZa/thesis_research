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
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Files with exactly one answer per MedRevQA row (the three-line-per-answer
#: files need the authors' own slicing and are left out rather than guessed).
MODELS = ("qwen25-7b", "llama33-70b", "mistral-24b", "gpt4o-mini", "deepsek-v3")
_LABEL = re.compile(r"NOT ENOUGH INFORMATION|SUPPORTED|REFUTED")


def parse_label(text: str):
    m = _LABEL.search(text.upper())
    return m.group(0) if m else None


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
    items = [json.loads(l) for l in open(args.benchmark, encoding="utf-8")]
    ans_dir = Path(args.medchange_dir) / "Code" / "GeneratedAnswers"
    report = {}
    for m in MODELS:
        p = ans_dir / f"{m}_answers.txt"
        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        report[m] = score(items, [parse_label(x) for x in lines])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
