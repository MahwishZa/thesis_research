"""Pilot checks for the stance step (gate 1): machine checks only, no human labelling.

After ``stance --pilot`` has scored 40 dev items (both wordings plus the irrelevant-paper control):

    python -m experiments.medchange.stance_check report

Gate 1 (all must hold): the two wordings agree on >= 80% of papers; >= 70% of the irrelevant control
papers are rated "neither"; <= 2% invalid outputs on REAL papers; <= 10 s per paper (llama) or <= 4 s
(flan). The original gate also required a 40-paper hand check by the researcher (>= 70% accurate); it was
removed on 2026-10-03 because the researcher is not a domain expert and the quantity that matters, whether
stance predicts the gold verdict, is tested objectively in gate 2 (``synthesis fit``). The invalid-rate
criterion is applied to real papers because control papers never enter any analysis and an invalid real
paper is treated as "no clear stance"; the pooled rate is still reported (§13 of the stage-2 protocol, the file ``experiment_plan.md`` at commit 92e3aaf).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .generate_answers import load_jsonl
from .stance import CLASSES

HERE = Path(__file__).resolve().parent
MIN_AGREEMENT, MIN_CONTROL_NEITHER = 0.80, 0.70
MAX_INVALID, MAX_SECONDS_LLAMA, MAX_SECONDS_FLAN = 0.02, 10.0, 4.0


def _real(records: list[dict], wording: str) -> list[dict]:
    return [r for r in records if not r.get("control") and r["wording"] == wording]


def pilot_report(records: list[dict]) -> dict:
    """Machine checks on a pilot file: validity, speed, wording agreement, irrelevant control."""
    a, b = _real(records, "A"), _real(records, "B")
    first = a or b
    control = [r for r in records if r.get("control")]
    if not first:
        raise ValueError("no pilot records: run the stance pilot first")
    allr = a + b + control
    invalid = sum(r["probs"] is None for r in allr) / len(allr)
    by_b = {(r["item_id"], r["pmid"]): r for r in b}
    pairs = [(r, by_b[(r["item_id"], r["pmid"])]) for r in a if (r["item_id"], r["pmid"]) in by_b]
    agreement = (sum(x["argmax"] == y["argmax"] for x, y in pairs) / len(pairs)) if pairs else None

    def strength(r):                       # how clearly a paper takes a side
        return 0.0 if r["probs"] is None else abs(r["probs"][0] - r["probs"][1])

    real_a = a or b
    out = {
        "backend": first[0]["backend"], "n_real_papers": len(real_a), "n_control_papers": len(control),
        "invalid_rate": round(invalid, 4),
        "invalid_rate_real": round(sum(r["probs"] is None for r in a + b) / len(a + b), 4),
        "invalid_by_group": {name: (round(sum(r["probs"] is None for r in g) / len(g), 4) if g else None)
                             for name, g in (("wording_A", a), ("wording_B", b), ("control", control))},
        "seconds_per_paper_mean": round(statistics.mean(r["seconds"] for r in allr), 2),
        "seconds_per_paper_median": round(statistics.median(r["seconds"] for r in allr), 2),
        "wording_agreement": None if agreement is None else round(agreement, 4),
        "n_wording_pairs": len(pairs),
        "real_class_share": {c: round(sum(r["argmax"] == c for r in real_a) / len(real_a), 4) for c in CLASSES},
        "control_neither_share": (round(sum(r["argmax"] == "neither" for r in control) / len(control), 4)
                                  if control else None),
        "mean_side_strength_real": round(statistics.mean(strength(r) for r in real_a), 4),
        "mean_side_strength_control": (round(statistics.mean(strength(r) for r in control), 4)
                                       if control else None),
    }
    return out


def gate1(report: dict) -> dict:
    """The pre-stated pilot gate, machine checks only."""
    limit = MAX_SECONDS_FLAN if report["backend"].startswith("flan") else MAX_SECONDS_LLAMA
    checks = {
        "wording_agreement>=0.80": None if report["wording_agreement"] is None
        else report["wording_agreement"] >= MIN_AGREEMENT,
        "control_neither_share>=0.70": None if report["control_neither_share"] is None
        else report["control_neither_share"] >= MIN_CONTROL_NEITHER,
        "invalid_rate_real_papers<=0.02": report["invalid_rate_real"] <= MAX_INVALID,
        f"seconds_per_paper<={limit:g}": report["seconds_per_paper_mean"] <= limit,
    }
    if any(v is None for v in checks.values()):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if all(checks.values()) else "FAIL"
    return {"checks": checks, "gate1": verdict}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("report",))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=None, help="also write the report here (JSON)")
    args = ap.parse_args(argv)
    records = load_jsonl(Path(args.data_dir) / "stance_pilot.jsonl")
    if not records:
        print("no pilot records yet: run the stance pilot first", file=sys.stderr)
        return 2
    report = pilot_report(records)
    result = {"pilot": report, "gate": gate1(report)}
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
