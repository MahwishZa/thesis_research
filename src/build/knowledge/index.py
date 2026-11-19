"""The knowledge index: every item open-book runs may retrieve, in one table, and what each class has. Run it after the text dataset
and the library:

    python -m src.build.knowledge.index

It writes, from committed tables only (and, for the licence of each case taken from a PMC full text, the text dataset's download
in data/text/sources/pubmed/abstracts.jsonl):
- data/knowledge/index.csv: the training cases of the text dataset and the passages of the library. No validation or test item of
  any dataset is in it. An item is linked to a question only by its disease (``link`` = disease).
- data/knowledge/diseases.csv: one row per class: the names and MeSH headings of the text dataset, and how many cases and passages
  it has.
- the index section of data/knowledge/dataset_report.md.
The step stops before writing anything if a validation or test item would enter the index.

(Version 1 also indexed the training photos of the ISIC image dataset. They were removed on 2026-11-19 with the Version 1 photo
datasets; the Git history keeps them, commit a1c9209. Precedent photos for Version 2 will come from the training splits of its own
datasets, and are not built yet.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from ...common import write_section
from ..text.prepare import CLASSES, DATA as TEXT
from .library import KNOWLEDGE, REPORT, REPORT_HEAD, STEPS

INDEX_TABLE, DISEASES_TABLE = "index.csv", "diseases.csv"
INDEX_COLUMNS = ["item_id", "kind", "diseases", "link", "path", "sha256", "source", "licence"]
DISEASE_COLUMNS = ["class", "text_terms", "mesh_headings", "cases_train", "cases_val", "cases_test", "passages"]
CASE_LICENCES = {"PMC-Patients": "CC BY-NC-SA 4.0 (PMC-Patients)", "PubMed abstract": "publisher's copyright (abstract in PubMed)"}
FULL_TEXTS = Path("sources") / "pubmed" / "abstracts.jsonl"  # in the text dataset's folder (on this computer): each full text's licence
NO_LICENCE = "(no licence given by PMC)"
SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: the text dataset and the library must be prepared first")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def full_text_licences(text: Path) -> dict:
    """Each PMC full text's own licence, as the text dataset downloaded it, by PubMed number."""
    path = text / FULL_TEXTS
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: it holds the licence of each case taken from a PMC full text (the "
                                "text dataset's abstracts step writes it)")
    licences = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            licences[str(record["pmid"])] = record.get("licence") or ""
    return licences


def held_out_items(items: list[dict], cases: list[dict]) -> list[str]:
    """The ids of the index items that are a validation or test case of the text dataset (there must be none)."""
    held_out = {c["path"] for c in cases if c["split"] != "train"}
    return [k["item_id"] for k in items if k["path"].split("/dataset/", 1)[-1] in held_out]


def index(text: Path = TEXT, data: Path = KNOWLEDGE) -> dict:
    """Writes index.csv, diseases.csv and the index section of the report (see the module's description)."""
    text, data = Path(text), Path(data)
    cases = _read_csv(text / "dataset.csv")
    passages = _read_csv(data / "library.csv")
    outside = sorted({c["label"] for c in cases} - set(CLASSES))
    if outside:
        raise ValueError(f"the text dataset has classes that are not among the eight of the dataset: {', '.join(outside)}")

    items, licences = [], full_text_licences(text) if any(c["source"] == "PMC full text" for c in cases) else {}
    for c in sorted(cases, key=lambda c: c["path"]):
        if c["split"] == "train":
            items.append({"item_id": f"case:{c['patient_uid']}", "kind": "case", "diseases": c["label"], "link": "disease",
                          "path": f"data/text/dataset/{c['path']}", "sha256": c["sha256"], "source": f"PMID {c['pmid']}",
                          "licence": CASE_LICENCES.get(c["source"]) or licences.get(c["pmid"]) or NO_LICENCE})
    for p in passages:
        items.append({"item_id": f"passage:{p['passage_id']}", "kind": "passage", "diseases": p["topics"], "link": "disease",
                      "path": f"data/knowledge/texts/passages.jsonl#{p['passage_id']}", "sha256": p["sha256"],
                      "source": p["source"], "licence": p["licence"] or "(not recorded)"})
    leaked = held_out_items(items, cases)
    if leaked:  # cannot happen by construction; a guard against a later change
        raise ValueError(f"validation or test items in the knowledge index: {leaked[:3]}")

    terms = _read_csv(text / "build" / "class_terms.csv")
    mesh = _read_csv(text / "build" / "term_check.csv")
    diseases = []
    for label in sorted(CLASSES):
        headings = sorted({h for row in mesh if row["class"] == label and row["counts"] == "yes"
                           for h in row["mesh_headings"].split("; ") if h})
        diseases.append({"class": label,
                         "text_terms": "; ".join(t["term"] for t in terms if t["class"] == label and t["counts"] == "yes"),
                         "mesh_headings": "; ".join(headings),
                         **{f"cases_{s}": sum(c["label"] == label and c["split"] == s for c in cases) for s in SPLITS},
                         "passages": sum(label in p["topics"].split("; ") for p in passages)})

    _write_csv(data / INDEX_TABLE, INDEX_COLUMNS, items)
    _write_csv(data / DISEASES_TABLE, DISEASE_COLUMNS, diseases)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "index", _summary(items, diseases))
    return {"items": Counter(k["kind"] for k in items), "table": (data / INDEX_TABLE).as_posix(),
            "diseases": (data / DISEASES_TABLE).as_posix(), "report": (data / REPORT).as_posix()}


def _summary(items: list[dict], diseases: list[dict]) -> list[str]:
    kinds = Counter(k["kind"] for k in items)
    lines = ["# Index: what open-book runs may retrieve", "",
             "Written by `python -m src.build.knowledge.index`, from the committed tables of the text dataset and the library.", "",
             f"- `index.csv`: {len(items)} items: {kinds['case']} training cases and {kinds['passage']} library passages. No validation "
             "or test item of any dataset is among them. Each is linked to the questions only by its disease.",
             f"- `diseases.csv`: {len(diseases)} classes, with the text names and MeSH headings, and the counts.", "",
             "| Class | Training cases | Passages |", "|---|---|---|"]
    for d in diseases:
        lines.append(f"| {d['class']} | {d['cases_train']} | {d['passages']} |")
    lines += ["", "Each item's licence is in the table, as its source gives it.", ""]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.build.knowledge.index", description=__doc__.split("\n\n")[0])
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    try:
        done = index()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print("Items: " + ", ".join(f"{k} {n}" for k, n in sorted(done["items"].items())))
    print(f"Written: {done['table']}, {done['diseases']} and {done['report']} (the index section). Commit all three.")


if __name__ == "__main__":
    main(sys.argv[1:])
