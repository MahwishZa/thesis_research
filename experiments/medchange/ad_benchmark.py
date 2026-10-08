"""Alzheimer's/dementia benchmark: a fresh, held-out secondary test set (split "ad").

    python -m experiments.medchange.ad_benchmark --medchange-dir ..\\MedChange

Every MedRevQA question whose text names dementia, Alzheimer's disease, mild cognitive impairment or
cognitive decline, from a Cochrane review that is NOT already in the dev or confirmatory split, by study
group AND by Cochrane ID (``data/benchmark.jsonl`` must exist: run ``build_benchmark`` first). Exact duplicate
questions are kept once.
Each question is asked as of its review's publication date, exactly like the main benchmark; the gold label is
the release's label for that review. Reviews with one version have no earlier version: their ``previous``
field repeats ``newest`` and update-window metrics do not apply to them.

The items are appended to ``data/benchmark.jsonl`` with ``split = "ad"`` (earlier "ad" rows are replaced, so a
rerun gives the same file) and summarised in ``experiments/medchange/manifest_ad.json`` (tracked): counts,
labels, the item-id hash and the input-file hashes. No label is used to choose items; the set is used once,
after the realigned design is frozen (docs/protocol.md §8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .benchmark import _version, file_sha256, load_groups, read_csv

HERE = Path(__file__).resolve().parent
AD_QUESTION = re.compile(r"dementia|alzheimer|mild cognitive impairment|cognitive decline|cognitively impaired|"
                         r"cognitive impairment", re.I)


def _item(medrev: dict, newest_row: int, previous_row: int, kind: str) -> dict:
    newest, previous = _version(medrev, newest_row), _version(medrev, previous_row)
    single = newest_row == previous_row
    return {"item_id": f"AD-{newest_row:05d}", "group_id": None, "kind": kind, "split": "ad",
            "question": medrev[newest_row]["Question"].strip(),
            "newest": newest.__dict__, "previous": previous.__dict__,
            "change_type": f"{previous.label} -> {newest.label}" if kind == "changed" else None,
            "decisive_flip": False, "conclusion_similarity": None, "likely_label_noise": False,
            "ad_related": True,
            "notes": ["single version: no earlier review; update-window metrics do not apply"] if single else []}


def build_ad_items(medrev: dict, groups: dict, used_groups: set, used_reviews: frozenset = frozenset()) -> list[dict]:
    """Questions about dementia/Alzheimer's from reviews outside dev and confirm; one item per distinct question.

    A review is "outside" when neither its study group (``used_groups``) nor its Cochrane ID (``used_reviews``)
    appears in those splits. The ID check matters: an older version of a review that dev or confirm already
    holds can sit in ``MedRevQA`` as an ungrouped row of its own, and the group check alone lets it through
    (four such questions were found on 2026-10-05, before any use of the set).
    """
    grouped = {k for keys in groups.values() for k in keys}
    candidates = []                                   # (newest row, previous row, kind)
    for gid in sorted(groups):
        if gid in used_groups:
            continue
        keys = sorted(set(groups[gid]))
        if len(keys) >= 2:
            same = medrev[keys[0]]["Label"].strip() == medrev[keys[1]]["Label"].strip()
            candidates.append((keys[0], keys[1], "unchanged" if same else "changed"))
        else:
            candidates.append((keys[0], keys[0], "unchanged"))
    candidates += [(k, k, "unchanged") for k in sorted(medrev) if k not in grouped]
    seen, out = set(), []
    for newest, previous, kind in sorted(candidates):
        question = medrev[newest]["Question"].strip()
        if not AD_QUESTION.search(question) or question.lower() in seen:
            continue
        if {_version(medrev, newest).cochrane_id, _version(medrev, previous).cochrane_id} & used_reviews:
            continue
        seen.add(question.lower())
        out.append(_item(medrev, newest, previous, kind))
    return out


def used_review_ids(main_rows: list[dict]) -> frozenset:
    """Cochrane IDs of every review (either version) that an item of the main benchmark is built from."""
    return frozenset(v["cochrane_id"] for r in main_rows for v in (r["newest"], r["previous"])
                     if v.get("cochrane_id"))


def summary(items: list[dict], inputs: dict) -> dict:
    ids = [i["item_id"] for i in items]
    return {"split": "ad", "items": len(items), "kinds": dict(Counter(i["kind"] for i in items)),
            "labels": dict(sorted(Counter(i["newest"]["label"] for i in items).items())),
            "reviews": len({i["newest"]["cochrane_id"] for i in items}),
            "single_version": sum(i["previous"]["row"] == i["newest"]["row"] for i in items),
            "question_filter": AD_QUESTION.pattern,
            "item_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "inputs": inputs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--medchange-dir", required=True)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--manifest", default=str(HERE / "manifest_ad.json"))
    a = ap.parse_args(argv)
    bench = Path(a.data_dir) / "benchmark.jsonl"
    if not bench.is_file():
        print(f"{bench} not found: run build_benchmark first", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    main_rows = [r for r in rows if r["split"] != "ad"]
    used = {r["group_id"] for r in main_rows}
    base = Path(a.medchange_dir) / "Datasets"
    paths = {n: base / n for n in ("MedRevQA.csv", "AllStudyGroups.csv")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"missing MedChange files: {missing}", file=sys.stderr)
        return 2
    medrev = {int(r[""]): r for r in read_csv(paths["MedRevQA.csv"])}
    items = build_ad_items(medrev, load_groups(read_csv(paths["AllStudyGroups.csv"])), used,
                           used_review_ids(main_rows))
    with open(bench, "w", encoding="utf-8", newline="\n") as h:
        for r in main_rows + items:
            h.write(json.dumps(r) + "\n")
    rep = summary(items, {n: file_sha256(p) for n, p in paths.items()})
    with open(a.manifest, "w", encoding="utf-8", newline="\n") as h:
        h.write(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("items", "kinds", "labels", "reviews", "single_version")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
