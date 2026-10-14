"""How many unused as-of questions are left in MedRevQA? (counts only: no network, no model, no answers)

    python -m experiments.medchange.fresh_supply --medchange-dir ..\\MedChange

A question is *fresh* when neither its study group nor its Cochrane ID nor its wording appears in any split of
``data/benchmark.jsonl`` (dev, confirm, ad). The tool applies the exclusion rules of ``selection`` (the ones that built the Alzheimer's disease set) but no topic filter,
and reports how many fresh questions there are at several earliest review dates, with their verdicts and kinds. It selects
nothing and writes only counts to ``results/fresh_supply.json`` (``docs/protocol.md`` §10 fixes the selection rule before any
question is chosen)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from .benchmark import LABELS, load_groups, read_csv
from .files import text_file_sha256
from .selection import fresh_items, used_review_ids

HERE = Path(__file__).resolve().parent
EARLIEST = ("2005-01-01", "2010-01-01", "2015-01-01", "2023-04-01")      # last: after the generator's stated knowledge cutoff


def report(items: list[dict]) -> dict:
    out = {"fresh_questions_all_dates": len(items), "by_earliest_date": {}}
    for since in EARLIEST:
        sel = [i for i in items if i["date"] >= since]
        labels = Counter(i["label"] for i in sel)
        out["by_earliest_date"][since] = {
            "questions": len(sel), "reviews": len({i["cochrane_id"] for i in sel}),
            "kinds": dict(Counter(i["kind"] for i in sel)),
            "labels": {l: labels.get(l, 0) for l in LABELS},
            "label_share": {l: round(labels.get(l, 0) / len(sel), 3) if sel else None for l in LABELS},
            "year_only_dates": sum(i["precision"] == "year" for i in sel),
            "dementia_wording": sum(i["dementia_wording"] for i in sel),
            "dementia_in_review_text": {
                "questions": sum(i["dementia_in_review_text"] for i in sel),
                "labels": {l: sum(i["dementia_in_review_text"] and i["label"] == l for i in sel) for l in LABELS}},
            "neurodegenerative_wording": {
                "questions": sum(i["neuro_wording"] for i in sel), "reviews": len({i["cochrane_id"] for i in sel if i["neuro_wording"]}),
                "labels": {l: sum(i["neuro_wording"] and i["label"] == l for i in sel) for l in LABELS}},
            "neurodegenerative_in_review_text": {
                "questions": sum(i["neuro_in_review_text"] for i in sel),
                "labels": {l: sum(i["neuro_in_review_text"] and i["label"] == l for i in sel) for l in LABELS}},
            "ids_sha256": hashlib.sha256(json.dumps(sorted(i["row"] for i in sel)).encode()).hexdigest()}
    years = Counter(i["date"][:4] for i in items)
    out["by_year"] = {y: years[y] for y in sorted(years)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--medchange-dir", required=True)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    bench = Path(a.data_dir) / "benchmark.jsonl"
    base = Path(a.medchange_dir) / "Datasets"
    paths = {n: base / n for n in ("MedRevQA.csv", "AllStudyGroups.csv")}
    missing = [str(p) for p in [bench, *paths.values()] if not p.is_file()]
    if missing:
        print(f"missing files: {missing}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    medrev = {int(r[""]): r for r in read_csv(paths["MedRevQA.csv"])}
    items = fresh_items(medrev, load_groups(read_csv(paths["AllStudyGroups.csv"])),
                        {r["group_id"] for r in rows if r.get("group_id") is not None}, used_review_ids(rows),
                        frozenset(r["question"].strip().lower() for r in rows))
    rep = report(items)
    rep.update(existing_items=len(rows), medrevqa_rows=len(medrev), inputs={n: text_file_sha256(p) for n, p in paths.items()},
               note="counts only; nothing is selected; the selection rule is fixed in docs/protocol.md §10")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fresh_supply.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
