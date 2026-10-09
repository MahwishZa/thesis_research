"""The fresh confirmatory split (``split = "fresh"``): questions of MedRevQA that no other split uses (``docs/protocol.md`` §10).

    python -m experiments.medchange.fresh_benchmark --medchange-dir ..\\MedChange

Selection rule, fixed before any question is answered: *fresh* questions (``fresh_supply``) whose review is dated from 2010-01-01 and whose
wording does not name dementia or Alzheimer's disease, in the order of the SHA-256 of ``fresh-v1|<MedRevQA row>``; the first N (1,500). No
label, topic or answer is used. The items are appended to ``data/benchmark.jsonl`` with ``split = "fresh"`` (earlier "fresh" rows are
replaced, so a rerun gives the same file) and summarised in ``experiments/medchange/manifest_fresh.json`` (tracked: counts, labels, the
selected MedRevQA rows, the hash of the item list and of the input files). The manifest is committed and pushed before any answer exists
(``rag2_pipeline fresh`` refuses to start otherwise)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .ad_benchmark import used_review_ids
from .benchmark import LABELS, _version, file_sha256, load_groups, read_csv
from .fresh_supply import fresh_items

HERE = Path(__file__).resolve().parent
SELECTION_SEED = "fresh-v1"
EARLIEST = "2010-01-01"
N_DEFAULT = 1500


def order_key(row: int) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{row}".encode("utf-8")).hexdigest()


def select(candidates: list[dict], n: int = N_DEFAULT, earliest: str = EARLIEST) -> list[dict]:
    """The first ``n`` qualifying candidates in hash order; nothing but the date and the wording is looked at."""
    ok = [c for c in candidates if c["date"] >= earliest and not c["dementia_wording"]]
    return sorted(ok, key=lambda c: order_key(c["row"]))[:n]


def make_item(medrev: dict, c: dict) -> dict:
    newest, previous = _version(medrev, c["row"]), _version(medrev, c["previous"])
    return {"item_id": f"FR-{c['row']:05d}", "group_id": c["group_id"], "kind": c["kind"], "split": "fresh",
            "question": medrev[c["row"]]["Question"].strip(), "newest": newest.__dict__, "previous": previous.__dict__,
            "change_type": f"{previous.label} -> {newest.label}" if c["kind"] == "changed" else None,
            "decisive_flip": False, "conclusion_similarity": None, "likely_label_noise": False, "ad_related": False,
            "notes": ["single version: no earlier review; update-window metrics do not apply"] if c["row"] == c["previous"] else []}


def summary(items: list[dict], qualifying: int, inputs: dict) -> dict:
    ids = [i["item_id"] for i in items]
    labels = Counter(i["newest"]["label"] for i in items)
    return {"split": "fresh", "items": len(items), "kinds": dict(Counter(i["kind"] for i in items)),
            "labels": {l: labels.get(l, 0) for l in LABELS}, "reviews": len({i["newest"]["cochrane_id"] for i in items}),
            "single_version": sum(i["previous"]["row"] == i["newest"]["row"] for i in items),
            "selection": {"seed": SELECTION_SEED, "order": "sha256(<seed>|<MedRevQA row>)", "earliest_review_date": EARLIEST,
                          "qualifying_questions": qualifying, "selected": len(items),
                          "dates": [min(i["newest"]["date"] for i in items), max(i["newest"]["date"] for i in items)]},
            "rows": [i["newest"]["row"] for i in items],
            "item_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "inputs": inputs}


def build(medrev: dict, groups: dict, rows: list[dict], n: int = N_DEFAULT) -> tuple[list[dict], dict]:
    main_rows = [r for r in rows if r["split"] != "fresh"]
    used_groups = {r["group_id"] for r in main_rows if r.get("group_id") is not None}
    cands = fresh_items(medrev, groups, used_groups, used_review_ids(main_rows),
                        frozenset(r["question"].strip().lower() for r in main_rows))
    chosen = select(cands, n)
    return [make_item(medrev, c) for c in chosen], {"qualifying": sum(c["date"] >= EARLIEST and not c["dementia_wording"] for c in cands)}


DRAFT_STATUS = re.compile(r"\*\*Status\.\*\* This section is a draft.*?(?=\n\n)", re.DOTALL)


def freeze(protocol: Path, manifest: dict, today: str) -> bool:
    """Mark protocol §10 as in force: replaces the draft status paragraph; False when there is none (already frozen or edited)."""
    text = protocol.read_text(encoding="utf-8")
    line = (f"**Status.** This section is IN FORCE since {today}. The selection is the first {manifest['items']} qualifying questions in the order of "
            f"the SHA-256 of `fresh-v1|<row>` (item list hash `{manifest['item_ids_sha256'][:16]}`, manifest committed before any answer). "
            "From this date the hypothesis, the primary metric and decision rule, the sample size, the selection rule and seed, the arms, every "
            "setting and every prompt are fixed (§10.7). The sections below are those of the draft.")
    new, n = DRAFT_STATUS.subn(lambda m: line, text, count=1)
    if n:
        protocol.write_text(new, encoding="utf-8", newline="\n")
    return bool(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--medchange-dir", required=True)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--manifest", default=str(HERE / "manifest_fresh.json"))
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--freeze", action="store_true",
                    help="mark docs/protocol.md section 10 as IN FORCE (needs manifest_fresh.json committed and pushed first)")
    a = ap.parse_args(argv)
    if a.freeze:
        from .runner import frozen_is_pushed
        rel = "experiments/medchange/manifest_fresh.json"
        ok, why = frozen_is_pushed(rel)
        if not ok:
            print(f"cannot freeze: {why}", file=sys.stderr)
            return 2
        protocol = HERE.parents[1] / "docs" / "protocol.md"
        if not freeze(protocol, json.loads(Path(a.manifest).read_text(encoding="utf-8")), dt.date.today().isoformat()):
            print("cannot freeze: section 10 has no draft status paragraph (already frozen?)", file=sys.stderr)
            return 2
        print("docs/protocol.md section 10 is now IN FORCE: commit and push it, then start the run")
        return 0
    bench = Path(a.data_dir) / "benchmark.jsonl"
    base = Path(a.medchange_dir) / "Datasets"
    paths = {n: base / n for n in ("MedRevQA.csv", "AllStudyGroups.csv")}
    missing = [str(p) for p in [bench, *paths.values()] if not p.is_file()]
    if missing:
        print(f"missing files: {missing}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    medrev = {int(r[""]): r for r in read_csv(paths["MedRevQA.csv"])}
    items, meta = build(medrev, load_groups(read_csv(paths["AllStudyGroups.csv"])), rows, a.n)
    if len(items) < a.n:
        print(f"only {len(items)} qualifying questions for N = {a.n}: the pre-registration needs a decision", file=sys.stderr)
        return 3
    rep = summary(items, meta["qualifying"], {n: file_sha256(p) for n, p in paths.items()})
    mpath = Path(a.manifest)
    if mpath.is_file():
        old = json.loads(mpath.read_text(encoding="utf-8"))
        if old.get("item_ids_sha256") != rep["item_ids_sha256"]:
            print("refusing to replace manifest_fresh.json: the selection differs from the recorded one "
                  "(the list is frozen once it is committed)", file=sys.stderr)
            return 2
    with open(bench, "w", encoding="utf-8", newline="\n") as h:
        for r in [r for r in rows if r["split"] != "fresh"] + items:
            h.write(json.dumps(r) + "\n")
    mpath.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rep[k] for k in ("items", "kinds", "labels", "reviews", "single_version", "selection", "item_ids_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
