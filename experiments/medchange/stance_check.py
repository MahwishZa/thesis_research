"""Pilot checks for the stance step (gate 1) and the hand check you do yourself.

After ``stance --pilot`` has scored 40 dev items (both wordings plus the irrelevant-paper control):

    python -m experiments.medchange.stance_check report
    python -m experiments.medchange.stance_check export        # writes the sheet you label
    python -m experiments.medchange.stance_check score         # after you filled the sheet

``report`` prints the machine checks; ``export`` writes ``stance_handcheck.csv`` (question, title
and the text the model read, with the model's answer hidden) and a key file; ``score`` compares your
labels with both wordings, picks the better wording (ties: A), writes ``stance_choice.json`` and
prints the gate-1 verdict. Pre-stated gate 1 (all must hold): hand-check accuracy >= 70% on 40
papers; the two wordings agree on >= 80% of papers; >= 70% of the irrelevant control papers are
rated "neither"; <= 2% invalid outputs; <= 10 s per paper (llama) or <= 4 s (flan).

How to label (S, C or N): read the question as a claim. S = the study text reports a benefit or
effect consistent with the claim; C = it reports no benefit, harm or the opposite; N = unrelated,
inconclusive or mixed. Judge only the text shown, not your own medical knowledge.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Optional

from .generate_answers import load_jsonl
from .stance import CLASSES, study_snippet

HERE = Path(__file__).resolve().parent
MIN_HANDCHECK, MIN_AGREEMENT, MIN_CONTROL_NEITHER = 0.70, 0.80, 0.70
MAX_INVALID, MAX_SECONDS_LLAMA, MAX_SECONDS_FLAN = 0.02, 10.0, 4.0
LABEL_WORDS = {"S": "supports", "C": "contradicts", "N": "neither",
               "SUPPORTS": "supports", "CONTRADICTS": "contradicts", "NEITHER": "neither"}


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


def gate1(report: dict, handcheck: Optional[dict]) -> dict:
    """The pre-stated pilot gate; ``handcheck`` is the result of ``score_handcheck`` (or None)."""
    limit = MAX_SECONDS_FLAN if report["backend"].startswith("flan") else MAX_SECONDS_LLAMA
    checks = {
        "hand_check_accuracy>=0.70": None if handcheck is None else handcheck["chosen_accuracy"] >= MIN_HANDCHECK,
        "wording_agreement>=0.80": None if report["wording_agreement"] is None
        else report["wording_agreement"] >= MIN_AGREEMENT,
        "control_neither_share>=0.70": None if report["control_neither_share"] is None
        else report["control_neither_share"] >= MIN_CONTROL_NEITHER,
        "invalid_rate<=0.02": report["invalid_rate"] <= MAX_INVALID,
        f"seconds_per_paper<={limit:g}": report["seconds_per_paper_mean"] <= limit,
    }
    if any(v is None for v in checks.values()):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if all(checks.values()) else "FAIL"
    return {"checks": checks, "gate1": verdict}


def export_handcheck(records: list[dict], pools: dict, items: dict, n: int, seed: int,
                     sheet: Path, key: Path) -> int:
    """Write the sheet to label (model answers hidden) and the key file; returns the rows written."""
    a = {(r["item_id"], r["pmid"]): r for r in _real(records, "A")}
    b = {(r["item_id"], r["pmid"]): r for r in _real(records, "B")}
    ids = sorted(a)
    random.Random(seed).shuffle(ids)
    picked = ids[:n]
    rows, keys = [], {}
    for number, (item_id, pmid) in enumerate(picked, 1):
        cand = next(c for c in pools[item_id]["candidates"] if c["pmid"] == pmid)
        rows.append([number, items[item_id]["question"], study_snippet(cand), ""])
        keys[str(number)] = {"item_id": item_id, "pmid": pmid, "A": a[(item_id, pmid)]["argmax"],
                             "B": b[(item_id, pmid)]["argmax"] if (item_id, pmid) in b else None}
    with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "question", "study_text", "your_label"])
        writer.writerows(rows)
    key.write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8", newline="\n")
    return len(rows)


def score_handcheck(sheet_rows: list[dict], key: dict) -> dict:
    """Compare the researcher's S/C/N labels with both wordings' answers."""
    labels = {}
    for row in sheet_rows:
        raw = row["your_label"].strip().upper()
        if raw not in LABEL_WORDS:
            raise ValueError(f"row {row['row']}: '{row['your_label']}' is not S, C or N")
        labels[row["row"]] = LABEL_WORDS[raw]
    acc, confusion = {}, {}
    for wording in ("A", "B"):
        pairs = [(labels[r], key[r][wording]) for r in labels if key[r].get(wording)]
        if not pairs:
            continue
        acc[wording] = sum(x == y for x, y in pairs) / len(pairs)
        confusion[wording] = {f"{t}->{m}": sum(1 for x, y in pairs if x == t and y == m)
                              for t in CLASSES for m in CLASSES}
    chosen = "A" if acc.get("A", -1) >= acc.get("B", -1) else "B"
    return {"n_checked": len(labels), "accuracy": {k: round(v, 4) for k, v in acc.items()},
            "chosen_wording": chosen, "chosen_accuracy": round(acc[chosen], 4),
            "human_label_counts": {c: sum(v == c for v in labels.values()) for c in CLASSES},
            "confusion_true_to_model": confusion}


def _load_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("report", "export", "score"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    data = Path(args.data_dir)
    pilot = data / "stance_pilot.jsonl"
    sheet, key = data / "stance_handcheck.csv", data / "stance_handcheck_key.json"
    records = load_jsonl(pilot)
    if not records:
        print("no pilot records yet: run the stance pilot first", file=sys.stderr)
        return 2

    if args.mode == "export":
        pools = {r["item_id"]: r for r in load_jsonl(data / "frozen_dev.jsonl")}
        items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")}
        n = export_handcheck(records, pools, items, args.n, args.seed, sheet, key)
        print(f"wrote {n} rows to {sheet}\nFill the 'your_label' column with S, C or N "
              "(rules: top of experiments/medchange/stance_check.py), save, then run: stance_check score")
        return 0

    report = pilot_report(records)
    if args.mode == "report":
        print(json.dumps({"pilot": report, "gate": gate1(report, None)}, indent=2))
        return 0

    result = score_handcheck(_load_sheet(sheet), json.loads(key.read_text(encoding="utf-8")))
    verdict = gate1(report, result)
    (data / "stance_choice.json").write_text(
        json.dumps({"wording": result["chosen_wording"], "gate1": verdict["gate1"],
                    "hand_check": result, "pilot": report}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"hand_check": result, "pilot": report, "gate": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
