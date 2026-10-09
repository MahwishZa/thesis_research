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
from typing import Optional

HERE = Path(__file__).resolve().parent
#: The eight models whose closed-book answers the MedChange release contains, by the key used here.
FILES = {"qwen25-7b": "qwen25-7b_answers.txt", "llama33-70b": "llama33-70b_answers.txt",
         "mistral-24b": "mistral-24b_answers.txt", "gpt4o-mini": "gpt4o-mini_answers.txt",
         "deepsek-v3": "deepsek-v3_answers.txt", "biomistral": "biomistral_answers.txt",
         "pmcllama": "pmcllama_answers.txt", "olmo-13b": "olmo_gguf_answers_13b.txt"}
#: These three files hold three lines per answer ("Question: ...", "Answer: ...", "---"); the authors' own notebook
#: (Code/experiments_results.ipynb) reads the answers as ``lines[1::3]``. The others hold one line per answer.
THREE_LINE = ("biomistral", "pmcllama", "olmo-13b")
MODELS = tuple(FILES)
_LABEL = re.compile(r"NOT ENOUGH INFORMATION|SUPPORTED|REFUTED")


def parse_label(text: str):
    m = _LABEL.search(text.upper())
    return m.group(0) if m else None


def read_answers(folder: Path, model: str) -> Optional[list]:
    """One verdict (or None) per MedRevQA row from the release's answers of ``model``; None if the file is absent."""
    path = Path(folder) / FILES[model]
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    if model in THREE_LINE:
        if lines and lines[-1] == "":
            lines = lines[:-1]
        lines = lines[1::3]
    return [parse_label(x) for x in lines]


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
        answers = read_answers(ans_dir, m)
        if answers is not None:
            report[m] = score(items, answers)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
