"""The knowledge index: every item open-book runs may retrieve, in one table, and what each class has. Run it after the library
and the image + metadata build:

    python -m src.build.knowledge.index

It writes, from committed tables only (and, for the licence of each case taken from a PMC full text, the text dataset's download
in data/text/sources/pubmed/abstracts.jsonl):
- data/knowledge/index.csv: the training images of the image dataset, the training cases of the text dataset and the passages of
  the library. No validation or test item of any dataset is in it. An item is linked to a question only by its disease
  (``link`` = disease): the photos and the texts share no patient.
- data/knowledge/diseases.csv: one row per class: the ISIC diagnoses its images have, the names and MeSH headings of the text
  dataset, and how many image + metadata questions, cases and passages it has.
- the index section of data/knowledge/dataset_report.md.
The step stops before writing anything if a validation or test item would enter the index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from ...common import write_section
from ..image_metadata.prepare import DATA as IMAGE_METADATA
from ..image_metadata.prepare import IMAGE, TEXT
from .library import KNOWLEDGE, REPORT, REPORT_HEAD, STEPS

INDEX_TABLE, DISEASES_TABLE = "index.csv", "diseases.csv"
INDEX_COLUMNS = ["item_id", "kind", "diseases", "link", "path", "sha256", "source", "licence"]
DISEASE_COLUMNS = ["class", "isic_diagnoses", "text_terms", "mesh_headings", "questions_train", "questions_val", "questions_test",
                   "complete_test", "cases_train", "cases_val", "cases_test", "passages"]
CASE_LICENCES = {"PMC-Patients": "CC BY-NC-SA 4.0 (PMC-Patients)", "PubMed abstract": "publisher's copyright (abstract in PubMed)"}
FULL_TEXTS = Path("sources") / "pubmed" / "abstracts.jsonl"  # in the text dataset's folder (on this computer): each full text's licence
NO_LICENCE = "(no licence given by PMC)"
SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: the image and text datasets, the image + metadata build and the "
                                "library must be prepared first")
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


def index(image: Path = IMAGE, text: Path = TEXT, metadata: Path = IMAGE_METADATA, data: Path = KNOWLEDGE) -> dict:
    """Writes index.csv, diseases.csv and the index section of the report (see the module's description)."""
    image, text, metadata, data = Path(image), Path(text), Path(metadata), Path(data)
    table = {row["path"]: row for row in _read_csv(image / "dataset.csv")}
    photos = {row["photo_id"]: row for row in _read_csv(image / "build" / "raw_photos.csv")}
    questions = sorted(_read_csv(metadata / "dataset.csv"), key=lambda q: (q["split"], q["label"], q["path"]))
    cases = _read_csv(text / "dataset.csv")
    passages = _read_csv(data / "library.csv")
    labels = sorted({q["label"] for q in questions})
    if sorted({c["label"] for c in cases}) != labels:
        raise ValueError("the image and text datasets do not have the same classes")
    if {q["path"] for q in questions} != set(table):
        raise ValueError(f"{(metadata / 'dataset.csv').as_posix()} does not hold exactly the image dataset's photos: build it again")

    items, licences = [], full_text_licences(text) if any(c["source"] == "PMC full text" for c in cases) else {}
    for q in questions:
        if q["split"] == "train":
            items.append({"item_id": f"image:{q['path']}", "kind": "image", "diseases": q["label"], "link": "disease",
                          "path": f"data/image/dataset/{q['path']}", "sha256": q["sha256"], "source": f"ISIC {q['isic_id']}",
                          "licence": photos[table[q["path"]]["photo_id"]]["licence"]})
    for c in sorted(cases, key=lambda c: c["path"]):
        if c["split"] == "train":
            items.append({"item_id": f"case:{c['patient_uid']}", "kind": "case", "diseases": c["label"], "link": "disease",
                          "path": f"data/text/dataset/{c['path']}", "sha256": c["sha256"], "source": f"PMID {c['pmid']}",
                          "licence": CASE_LICENCES.get(c["source"]) or licences.get(c["pmid"]) or NO_LICENCE})
    for p in passages:
        items.append({"item_id": f"passage:{p['passage_id']}", "kind": "passage", "diseases": p["topics"], "link": "disease",
                      "path": f"data/knowledge/texts/passages.jsonl#{p['passage_id']}", "sha256": p["sha256"],
                      "source": p["source"], "licence": p["licence"] or "(not recorded)"})
    held_out = {q["path"] for q in questions if q["split"] != "train"} | {c["path"] for c in cases if c["split"] != "train"}
    leaked = [k["item_id"] for k in items if k["path"].split("/dataset/", 1)[-1] in held_out]
    if leaked:  # cannot happen by construction; a guard against a later change
        raise ValueError(f"validation or test items in the knowledge index: {leaked[:3]}")

    terms = _read_csv(text / "build" / "class_terms.csv")
    mesh = _read_csv(text / "build" / "term_check.csv")
    diseases = []
    for label in labels:
        diagnoses = Counter(photos[table[q["path"]]["photo_id"]]["isic_diagnosis"] for q in questions if q["label"] == label)
        headings = sorted({h for row in mesh if row["class"] == label and row["counts"] == "yes"
                           for h in row["mesh_headings"].split("; ") if h})
        diseases.append({"class": label,
                         "isic_diagnoses": "; ".join(f"{d} ({n})" for d, n in sorted(diagnoses.items(), key=lambda x: (-x[1], x[0]))),
                         "text_terms": "; ".join(t["term"] for t in terms if t["class"] == label and t["counts"] == "yes"),
                         "mesh_headings": "; ".join(headings),
                         **{f"questions_{s}": sum(q["label"] == label and q["split"] == s for q in questions) for s in SPLITS},
                         "complete_test": sum(q["label"] == label and q["split"] == "test" and q["complete"] == "yes" for q in questions),
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
             "Written by `python -m src.build.knowledge.index`, from the committed tables of the image, image + metadata and text "
             "datasets and the library.", "",
             f"- `index.csv`: {len(items)} items: {kinds['image']} training images, {kinds['case']} training cases and "
             f"{kinds['passage']} library passages. No validation or test item of any dataset is among them. Each is linked to the "
             "questions only by its disease: the photos and the texts share no patient.",
             f"- `diseases.csv`: {len(diseases)} classes, with the ISIC diagnoses, the text names and MeSH headings, and the counts.", "",
             "| Class | Training images | Training cases | Passages |", "|---|---|---|---|"]
    for d in diseases:
        lines.append(f"| {d['class']} | {d['questions_train']} | {d['cases_train']} | {d['passages']} |")
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
