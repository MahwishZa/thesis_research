"""RETIRED 2026-10-04 (archived): G1 human check; is the stated VERDICT consistent with the justification?

Exports a seeded sample of answers (arm hidden) to a CSV; a human reads each answer and
fills ``consistent`` with Y or N; ``score`` reports the agreement and the parse rate.
A verdict line that contradicts its own justification (or a justification that hedges
between verdicts) is N. Pre-stated G1 pass: parse rate >= 95% and consistency >= 90%.

    python -m _archive.medchange_human_checks.consistency export --n 50
    python -m _archive.medchange_human_checks.consistency score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from experiments.medchange.generate_answers import load_jsonl

HERE = Path(__file__).resolve().parents[2] / "experiments" / "medchange"
MIN_PARSE, MIN_CONSISTENT = 0.95, 0.90


def sample_rows(answers: list[dict], n: int, seed: int) -> list[dict]:
    rows = sorted(answers, key=lambda r: (r["item_id"], r["arm"]))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def score_sheet(sheet: list[dict], parse_rate: float) -> dict:
    marked = [r for r in sheet if r["consistent"].strip().upper() in ("Y", "N")]
    if len(marked) < len(sheet):
        raise ValueError(f"{len(sheet) - len(marked)} rows are not marked Y or N")
    ok = sum(r["consistent"].strip().upper() == "Y" for r in marked) / len(marked)
    return {"n_checked": len(marked), "consistency": round(ok, 4), "parse_rate": round(parse_rate, 4),
            "G1": "PASS" if ok >= MIN_CONSISTENT and parse_rate >= MIN_PARSE else "FAIL"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("export", "score"))
    ap.add_argument("--answers", default=str(HERE / "data" / "answers_dev.jsonl"))
    ap.add_argument("--sheet", default=str(HERE / "data" / "consistency_sheet.csv"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    answers = load_jsonl(Path(args.answers))
    if not answers:
        print("no answers yet", file=sys.stderr)
        return 2
    if args.mode == "export":
        picked = sample_rows(answers, args.n, args.seed)
        with open(args.sheet, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "item_id", "answer_text", "consistent"])
            for i, r in enumerate(picked, 1):
                w.writerow([i, r["item_id"], r["text"], ""])
        print(f"wrote {len(picked)} rows to {args.sheet}; fill the 'consistent' column with Y or N")
        return 0
    with open(args.sheet, encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    parse_rate = sum(r["verdict"] is not None for r in answers) / len(answers)
    try:
        print(json.dumps(score_sheet(sheet, parse_rate), indent=2))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
