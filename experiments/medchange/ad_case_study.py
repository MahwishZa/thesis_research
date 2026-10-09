"""Case study of the Alzheimer's-named questions of the dementia run (descriptive; from the answers on file; no model is run).

    python -m experiments.medchange.ad_case_study --out-dir <folder outside the repository>

Writes, for the 48 questions whose text names Alzheimer's disease: ``ad_cases_all.csv`` (every question with the verdict of every system and what
R2V did), ``ad_case_study.md`` (12 questions chosen by a seeded hash, then the first fixed and the first broken answer of R2V in that order, so that
a failure is shown as well as a success) and ``validation_sheet.csv`` (the first 20 questions in the same order, with the reference verdict and empty
columns for a clinician's check). The outputs contain question text and model answers: they are **not** committed (the questions come from the MedChange
release, which states no licence). Nothing here is a confirmatory result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

from .generate_answers import load_jsonl

HERE = Path(__file__).resolve().parent
SEED = "ad-case-v1"
ARMS = ("B0", "B1", "R2", "R2C", "R2V")
NAMES_ALZHEIMER = re.compile(r"alzheimer", re.IGNORECASE)
SHORT = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT ENOUGH INFORMATION": "NOT ENOUGH INFO"}


def hash_order(ids) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{SEED}|{i}".encode("utf-8")).hexdigest())


def load(data: Path, results: Path) -> tuple[dict, dict]:
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == "ad" and not r["likely_label_noise"] and NAMES_ALZHEIMER.search(r["question"])}
    rec = {}
    for name in ("answers_ad.jsonl", "rag2_answers_ad.jsonl"):
        for r in load_jsonl(results / name):
            if r["item_id"] in items and r["arm"] in ARMS:
                rec[(r["item_id"], r["arm"])] = r
    return items, rec


def case(item: dict, rec: dict) -> dict:
    i, gold = item["item_id"], item["newest"]["label"]
    verdict = {a: rec[(i, a)]["verdict"] for a in ARMS if (i, a) in rec}
    r2v = rec.get((i, "R2V"), {})
    changed = bool(r2v.get("changed"))
    change = "unchanged"
    if changed:
        draft_ok, final_ok = r2v.get("draft_verdict") == gold, r2v.get("verdict") == gold
        change = "fixed" if final_ok and not draft_ok else "broke" if draft_ok and not final_ok else "changed, same correctness"
    admitted = rec.get((i, "R2"), {}).get("admitted", [])
    return {"item_id": i, "question": item["question"], "gold": gold, "verdict": verdict, "correct": {a: v == gold for a, v in verdict.items()},
            "change": change, "r2v_draft": r2v.get("draft_verdict"), "r2_admitted": len(admitted),
            "r2_strata": rec.get((i, "R2"), {}).get("admitted_stratum", []), "b1_admitted": len(rec.get((i, "B1"), {}).get("admitted", [])),
            "review": item["newest"].get("cochrane_id"), "review_date": item["newest"]["date"], "medrev_row": item["newest"]["row"],
            "r2v_text": " ".join((r2v.get("text") or "").split())}


def build(items: dict, rec: dict) -> list[dict]:
    return [case(items[i], rec) for i in hash_order(items)]


def to_markdown(cases: list[dict], n: int = 12) -> str:
    first = lambda kind: next((c for c in cases if c["change"] == kind), None)
    chosen = cases[:n]
    extra = [c for c in (first("fixed"), first("broke")) if c and c not in chosen]
    L = ["# Alzheimer's-named questions: a case study (descriptive)", "",
         f"{len(cases)} questions of the dementia run name Alzheimer's disease. Shown: the first {n} in the order of a seeded hash (`{SEED}`), "
         "then the first answer R2V fixed and the first it broke in that order, so that a failure is shown as well as a success. Nothing was chosen by outcome "
         "except those two. Nothing here is a confirmatory result.", ""]
    for k, c in enumerate(chosen + extra, 1):
        tag = "" if c in chosen else f" ({'fixed' if c['change'] == 'fixed' else 'broken'} by R2V: added)"
        L += [f"## {k}. {c['question']}{tag}", "",
              f"Reference verdict: **{c['gold']}** (review {c['review']}, {c['review_date']}).", "",
              "| System | Verdict | Correct |", "|---|---|---|"]
        for a in ARMS:
            if a in c["verdict"]:
                L.append(f"| {a} | {SHORT.get(c['verdict'][a], c['verdict'][a])} | {'yes' if c['correct'][a] else 'no'} |")
        L += ["", f"Evidence: R2 admitted {c['r2_admitted']} abstract(s) {c['r2_strata'] or ''}; B1 read {c['b1_admitted']}. "
              f"R2V's draft was {SHORT.get(c['r2v_draft'], c['r2v_draft'])}; its action: **{c['change']}**.", "",
              f"> R2V's check: {c['r2v_text'][:420]}{'…' if len(c['r2v_text']) > 420 else ''}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--out-dir", required=True, help="a folder outside the repository (the files hold question text)")
    a = ap.parse_args(argv)
    items, rec = load(Path(a.data_dir), Path(a.results_dir))
    cases = build(items, rec)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ad_cases_all.csv", "w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(["item_id", "question", "reference_verdict", *ARMS, "r2v_action", "r2_abstracts_admitted", "review", "review_date", "medrev_row"])
        for c in cases:
            w.writerow([c["item_id"], c["question"], c["gold"], *[c["verdict"].get(x, "") for x in ARMS], c["change"], c["r2_admitted"],
                        c["review"], c["review_date"], c["medrev_row"]])
    with open(out / "validation_sheet.csv", "w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(["item_id", "question", "reference_verdict", "review", "review_date", "medrev_row", "clinician_verdict", "agrees_with_reference", "comment"])
        for c in cases[:20]:
            w.writerow([c["item_id"], c["question"], c["gold"], c["review"], c["review_date"], c["medrev_row"], "", "", ""])
    (out / "ad_case_study.md").write_text(to_markdown(cases), encoding="utf-8", newline="\n")
    print(f"{len(cases)} Alzheimer's-named questions; wrote ad_cases_all.csv, ad_case_study.md and validation_sheet.csv to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
