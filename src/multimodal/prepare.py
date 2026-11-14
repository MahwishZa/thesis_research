"""Makes the multimodal dataset from the two frozen datasets, one command per step, with no new image and no new text.

    python -m src.multimodal.prepare records
    python -m src.multimodal.prepare build

The image and text datasets share no patient: the images come from the ISIC Archive, the texts from published case reports. So a
multimodal question is one photo of the image dataset with the same patient's own recorded facts (age, sex, body site), from that
photo's ISIC record; case texts and library passages are linked to the photos only by their disease, as knowledge to retrieve.
Pairing a photo with a case text as "one patient" would invent a case, and choosing a text for a test photo by its label would put
the answer into the question, so neither is ever done.

``records`` reads the ISIC record of every image of the image dataset (downloaded by ``python -m src.image.prepare isic`` into
data/image/downloads/isic/, on this computer) and reports in data/multimodal/dataset_report.md which patient facts the records
hold and how often: by class and by photo collection, since the classes come from two collections (melanoma and nevus almost
all from one, the other six almost all from the other) and a fact recorded differently by each collection would give the class
away. A field whose name points to the diagnosis (a diagnosis, benign or malignant, melanocytic, a melanoma or nevus detail, a
biopsy, a history of melanoma) or that is an identifier is listed but never offered as a fact. The step checks that each record
is the very one the image dataset was made from (its fingerprint in data/image/raw_photos.csv), changes no file of the image
dataset and asks no service.

``build`` writes the three tables, from the frozen tables and the ISIC records only (no file is copied; every row names its file
and fingerprint):
- questions.csv: one question per image of the image dataset, with its split, class, file and lesion group exactly as there, and
  the facts the researcher chose (FACTS: age, sex and body site at its broadest level, the three that both photo collections
  record with the same values), written as one sentence the model is shown; a fact that is missing, or on which a photo's two
  ISIC records disagree, is written "not recorded" in the same way for every image. ``complete`` marks the questions with all
  three facts.
- knowledge.csv: what open-book runs may search: the training images, the training cases of the text dataset and the passages
  of the text dataset's library. No validation or test item of either dataset is in it. An item is linked to a question only
  by its disease (``link`` = disease): the two datasets share no patient.
- diseases.csv: one row per class: the ISIC diagnoses its images have, the names and MeSH headings of the text dataset, and how
  many questions, cases and passages it has.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from ..common import flatten, read_rows, write_section

IMAGE = Path("data/image")  # the frozen image dataset: dataset.csv, raw_photos.csv and, on this computer, downloads/isic/
DATA = Path("data/multimodal")  # this dataset's folder: its committed tables and reports
RECORDS = Path("downloads") / "isic"  # inside the image dataset's folder: one ISIC record per photo, as downloaded
REPORT = "dataset_report.md"
STEPS = ("records", "build", "audit")  # the report's sections, in the order the steps run (the audit is src/multimodal/audit.py)
TEXT = Path("data/text")  # the frozen text dataset: dataset.csv, class_terms.csv, term_check.csv and library.csv
FACTS = (("age_approx", "Age"), ("sex", "Sex"), ("anatom_site_1", "Body site"))  # chosen from the records report (2026-11-14)
NOT_RECORDED = "not recorded"
QUESTION_COLUMNS = ["split", "label", "path", "sha256", "lesion_group", "isic_id", "collection", "age", "sex", "site", "complete",
                    "facts"]
KNOWLEDGE_COLUMNS = ["item_id", "kind", "diseases", "link", "path", "sha256", "source", "licence"]
DISEASE_COLUMNS = ["class", "isic_diagnoses", "text_terms", "mesh_headings", "questions_train", "questions_val", "questions_test",
                   "complete_test", "cases_train", "cases_val", "cases_test", "passages"]
CASE_LICENCES = {"PMC-Patients": "CC BY-NC-SA 4.0 (PMC-Patients)", "PubMed abstract": "publisher's copyright (abstract in PubMed)"}
FULL_TEXTS = Path("sources") / "pubmed" / "abstracts.jsonl"  # in the text dataset's folder (on this computer): each full text's licence
NO_LICENCE = "(no licence given by PMC)"
REPORT_HEAD = ["# Preparation of the multimodal dataset", "",
               "Made by `python -m src.multimodal.prepare <step>`, one section per step, in the order the steps run. Each step "
               "rewrites only its own section.", ""]
PATIENT_GROUPS = ("metadata.clinical.", "metadata.patient.")  # where ISIC keeps what it knows about the patient and the lesion
HINTS = re.compile(r"diagnos|benign|malignan|melan|(?:^|_)mel(?:_|$)|nev|biops|(?:^|_)hx(?:_|$)", re.I)  # point to the diagnosis
IDENTIFIER = re.compile(r"(?:^|_)id$", re.I)  # a lesion or patient number: links photos, says nothing about the patient
TOP_VALUES = 8  # values listed per fact and collection


def _record_hash(record) -> str:
    """The fingerprint the image dataset gave the record (src/image/prepare.py): its JSON with sorted keys."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def kind_of(field: str) -> str:
    """How a patient field may be used: "fact" (may be offered), "hint" (points to the diagnosis) or "identifier"."""
    name = field.split(".")[-1]
    if HINTS.search(name):
        return "hint"
    if IDENTIFIER.search(name):
        return "identifier"
    return "fact"


def _read_csv(path: Path) -> list[dict]:
    return read_rows(path, "the image and text datasets must be prepared first")


def read_images(image: Path = IMAGE) -> list[dict]:
    """Every image of the image dataset with its split, class, photo collection and patient fields, from its own ISIC record.

    A photo uploaded to ISIC more than once has one record per upload (raw_photos.csv lists their numbers and fingerprints in the
    same order). A field counts only when every record that has it gives the same value; a disagreement is kept in ``conflicts``
    and the field is left out for that photo, never guessed."""
    photos = {row["photo_id"]: row for row in _read_csv(image / "raw_photos.csv")}
    rows, missing, changed = [], [], []
    for row in _read_csv(image / "dataset.csv"):
        photo = photos.get(row["photo_id"])
        numbers, fingerprints = (photo["isic_ids"].split(), photo["isic_record_sha256"].split()) if photo else ([], [])
        if not numbers or len(numbers) != len(fingerprints):
            raise ValueError(f"{row['path']}: raw_photos.csv gives no ISIC number, or not one fingerprint per number, for photo "
                             f"{row['photo_id']!r}")
        values = {}
        for number, fingerprint in zip(numbers, fingerprints):
            path = image / RECORDS / f"{number}.json"
            if not path.is_file():
                missing.append(path.as_posix())
                continue
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("status") != "found" or _record_hash(entry.get("record")) != fingerprint:
                changed.append(path.as_posix())
                continue
            for k, v in flatten(entry["record"]).items():
                if k.startswith(PATIENT_GROUPS) and v not in (None, ""):
                    values.setdefault(k, []).append(v)
        fields = {k: v[0] for k, v in values.items() if all(x == v[0] for x in v)}
        conflicts = {k: v for k, v in values.items() if k not in fields}
        rows.append({"path": row["path"], "split": row["split"], "label": row["label"], "isic_id": " ".join(numbers),
                     "collection": photo["attribution"] or "(not recorded)", "fields": fields, "conflicts": conflicts})
    if missing or changed:
        raise ValueError(f"{len(missing)} ISIC records missing and {len(changed)} differing from the ones the image dataset was made "
                         f"from (for example {(missing + changed)[0]}). Run python -m src.image.prepare isic to download the missing "
                         "ones; a differing record means the downloads changed after the image dataset was frozen")
    return rows


def _share(k: int, n: int) -> str:
    return f"{k} of {n} ({100 * k / n:.0f}%)" if n else "–"


def _value(value) -> str:
    return str(value).replace("|", "/")


def records(image: Path = IMAGE, data: Path = DATA) -> dict:
    """The records step: what the ISIC records hold about each patient, by class and by photo collection."""
    rows = read_images(image)
    labels = sorted({r["label"] for r in rows})
    collections = sorted({r["collection"] for r in rows})
    by_label, by_collection = Counter(r["label"] for r in rows), Counter(r["collection"] for r in rows)
    present = Counter(field for r in rows for field in r["fields"])
    fields = sorted(present, key=lambda f: (-present[f], f))
    facts = [f for f in fields if kind_of(f) == "fact"]

    def has(field, key, value):
        return sum(1 for r in rows if r[key] == value and field in r["fields"])

    lines = ["## Records: the patient facts in the ISIC records", "",
             f"Read: the ISIC record of each of the {len(rows)} images of the image dataset, each one checked against the fingerprint the "
             "image dataset recorded for it. No file of the image dataset was changed and no service was asked.", "",
             "### Images by class and photo collection", "",
             "The collection is the record's attribution. A fact recorded differently by each collection gives the class away.", "",
             "| Class | " + " | ".join(collections) + " | All |", "|---|" + "---|" * (len(collections) + 1)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(sum(1 for r in rows if r["label"] == label and r["collection"] == c))
                                                  for c in collections) + f" | {by_label[label]} |")
    lines += ["", "### Every patient field the records hold", "",
              "Facts may be offered to the model. Hints point to the diagnosis and identifiers only link photos: both are never offered.",
              "", "| Field | Kind | Images with it | " + " | ".join(collections) + " |", "|---|---|---|" + "---|" * len(collections)]
    for field in fields:
        lines.append(f"| `{field}` | {kind_of(field)} | {_share(present[field], len(rows))} | "
                     + " | ".join(_share(has(field, "collection", c), by_collection[c]) for c in collections) + " |")
    if not fields:
        lines.append("| (none) | | | " + " | " * len(collections))
    twice = [r for r in rows if " " in r["isic_id"]]
    lines += ["", f"Photos with more than one ISIC record (uploaded more than once): {len(twice)}"
              + (f" ({', '.join(r['isic_id'] for r in twice)}). A field counts only where their records agree." if twice else "."),
              ""]
    for r in rows:
        for field, values in sorted((f, v) for f, v in r["conflicts"].items() if kind_of(f) == "fact"):
            lines.append(f"- Records disagree, so left out: `{r['isic_id']}` `{field}`: {', '.join(map(_value, values))}")
    lines += ["", "### How often each fact is recorded, by class", "",
              "| Fact | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for field in facts:
        lines.append(f"| `{field.split('.')[-1]}` | " + " | ".join(_share(has(field, "label", label), by_label[label]) for label in labels) + " |")
    lines += ["", "### The values of each fact, by collection", ""]
    for field in facts:
        lines += [f"`{field}`:", "", "| Collection | Values (images) |", "|---|---|"]
        for c in collections:
            values = Counter(_value(r["fields"][field]) for r in rows if r["collection"] == c and field in r["fields"])
            shown = ", ".join(f"{v} ({n})" for v, n in sorted(values.items(), key=lambda x: (-x[1], x[0]))[:TOP_VALUES])
            more = len(values) - TOP_VALUES
            lines.append(f"| {c} | {shown or 'not recorded'}{f', and {more} more values' if more > 0 else ''} |")
        numbers = {label: [float(r["fields"][field]) for r in rows if r["label"] == label and field in r["fields"]
                           and _is_number(r["fields"][field])] for label in labels}
        if sum(map(len, numbers.values())) >= present[field] * 0.9:  # a number: its median and range by class
            lines += ["", "| Class | Median | Range |", "|---|---|---|"]
            lines += [f"| {label} | {statistics.median(v):g} | {min(v):g} to {max(v):g} |" if v else f"| {label} | – | – |"
                      for label, v in numbers.items()]
        lines.append("")
    lines += ["### Not done in this step", "",
              "- No fact is chosen yet: the researcher chooses which facts the questions show, from this report.",
              "- No table is written: the build writes them once the facts are chosen.",
              "- Whether a fact gives the class away is measured in the audit, like the image dataset's shortcut check.", ""]
    write_section(data / REPORT, REPORT_HEAD, STEPS, "records", lines)
    return {"images": len(rows), "fields": {f: (kind_of(f), present[f]) for f in fields}, "report": (data / REPORT).as_posix()}


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def fact_values(fields: dict) -> dict:
    """The chosen facts of one photo, as shown: age in years ("about 65 years"), sex and body site in lower case, or "not recorded"."""
    out = {}
    for name, _ in FACTS:
        value = fields.get(f"metadata.clinical.{name}")
        if value is None:
            out[name] = NOT_RECORDED
        elif name == "age_approx":
            number = float(value)
            out[name] = f"about {number:g} years"
        else:
            out[name] = str(value).strip().lower()
    return out


def fact_sentence(values: dict) -> str:
    """What the model is shown next to the photo: "Age: about 65 years. Sex: male. Body site: trunk." """
    return " ".join(f"{label}: {values[name]}." for name, label in FACTS)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def build(image: Path = IMAGE, text: Path = TEXT, data: Path = DATA) -> dict:
    """The build step: questions.csv, knowledge.csv and diseases.csv, from the frozen tables and the ISIC records."""
    images = read_images(image)
    table = {row["path"]: row for row in _read_csv(image / "dataset.csv")}
    photos = {row["photo_id"]: row for row in _read_csv(image / "raw_photos.csv")}
    cases = _read_csv(text / "dataset.csv")
    passages = _read_csv(text / "library.csv")
    labels = sorted({r["label"] for r in images})
    if sorted({c["label"] for c in cases}) != labels:
        raise ValueError("the image and text datasets do not have the same classes")

    questions = []
    for r in sorted(images, key=lambda r: (r["split"], r["label"], r["path"])):
        source = table[r["path"]]
        values = fact_values(r["fields"])
        questions.append({"split": r["split"], "label": r["label"], "path": r["path"], "sha256": source["sha256"],
                          "lesion_group": source["lesion_group"], "isic_id": r["isic_id"], "collection": r["collection"],
                          "age": values["age_approx"], "sex": values["sex"], "site": values["anatom_site_1"],
                          "complete": "yes" if NOT_RECORDED not in values.values() else "no", "facts": fact_sentence(values)})

    knowledge, licences = [], _full_text_licences(text) if any(c["source"] == "PMC full text" for c in cases) else {}
    for q in questions:
        if q["split"] == "train":
            photo = photos[table[q["path"]]["photo_id"]]
            knowledge.append({"item_id": f"image:{q['path']}", "kind": "image", "diseases": q["label"], "link": "disease",
                              "path": f"data/image/dataset/{q['path']}", "sha256": q["sha256"], "source": f"ISIC {q['isic_id']}",
                              "licence": photo["licence"]})
    for c in sorted(cases, key=lambda c: c["path"]):
        if c["split"] == "train":
            knowledge.append({"item_id": f"case:{c['patient_uid']}", "kind": "case", "diseases": c["label"], "link": "disease",
                              "path": f"data/text/dataset/{c['path']}", "sha256": c["sha256"], "source": f"PMID {c['pmid']}",
                              "licence": CASE_LICENCES.get(c["source"]) or licences.get(c["pmid"]) or NO_LICENCE})
    for p in passages:
        knowledge.append({"item_id": f"passage:{p['passage_id']}", "kind": "passage", "diseases": p["topics"], "link": "disease",
                          "path": f"data/text/library/passages.jsonl#{p['passage_id']}", "sha256": p["sha256"],
                          "source": p["source"], "licence": p["licence"] or "(not recorded)"})
    held_out = {q["path"] for q in questions if q["split"] != "train"} | {c["path"] for c in cases if c["split"] != "train"}
    leaked = [k["item_id"] for k in knowledge if k["path"].split("/dataset/", 1)[-1] in held_out]
    if leaked:  # cannot happen by construction; a guard against a later change
        raise ValueError(f"validation or test items in the knowledge table: {leaked[:3]}")

    terms = _read_csv(text / "class_terms.csv")
    mesh = _read_csv(text / "term_check.csv")
    diseases = []
    for label in labels:
        diagnoses = Counter(photos[table[q["path"]]["photo_id"]]["isic_diagnosis"] for q in questions if q["label"] == label)
        headings = sorted({h for row in mesh if row["class"] == label and row["counts"] == "yes"
                           for h in row["mesh_headings"].split("; ") if h})
        diseases.append({"class": label,
                         "isic_diagnoses": "; ".join(f"{d} ({n})" for d, n in sorted(diagnoses.items(), key=lambda x: (-x[1], x[0]))),
                         "text_terms": "; ".join(t["term"] for t in terms if t["class"] == label and t["counts"] == "yes"),
                         "mesh_headings": "; ".join(headings),
                         **{f"questions_{s}": sum(q["label"] == label and q["split"] == s for q in questions) for s in ("train", "val", "test")},
                         "complete_test": sum(q["label"] == label and q["split"] == "test" and q["complete"] == "yes" for q in questions),
                         **{f"cases_{s}": sum(c["label"] == label and c["split"] == s for c in cases) for s in ("train", "val", "test")},
                         "passages": sum(label in p["topics"].split("; ") for p in passages)})

    _write_csv(data / "questions.csv", QUESTION_COLUMNS, questions)
    _write_csv(data / "knowledge.csv", KNOWLEDGE_COLUMNS, knowledge)
    _write_csv(data / "diseases.csv", DISEASE_COLUMNS, diseases)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "build", _build_summary(questions, knowledge, diseases, labels))
    return {"questions": len(questions), "knowledge": Counter(k["kind"] for k in knowledge),
            "complete": Counter(q["split"] for q in questions if q["complete"] == "yes"),
            "splits": Counter(q["split"] for q in questions), "report": (data / REPORT).as_posix()}


def _full_text_licences(text: Path) -> dict:
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


def _build_summary(questions: list[dict], knowledge: list[dict], diseases: list[dict], labels: list[str]) -> list[str]:
    splits = ("train", "val", "test")
    kinds = Counter(k["kind"] for k in knowledge)
    lines = ["## Build: the three tables", "",
             f"- `questions.csv`: {len(questions)} questions, one per image of the image dataset, with its split, class, file and "
             "lesion group unchanged: " + ", ".join(f"{s} {sum(q['split'] == s for q in questions)}" for s in splits) + ".",
             "- Facts shown with each photo: " + ", ".join(label.lower() for _, label in FACTS) + " (chosen from the records "
             f"report). A missing fact, or one the photo's two ISIC records disagree on, is shown as \"{NOT_RECORDED}\".",
             f"- `knowledge.csv`: {len(knowledge)} items open-book runs may search: {kinds['image']} training images, "
             f"{kinds['case']} training cases and {kinds['passage']} library passages. No validation or test item of either "
             "dataset is among them. Each is linked to the questions only by its disease: the two datasets share no patient.",
             f"- `diseases.csv`: {len(diseases)} classes, with the ISIC diagnoses, the text names and MeSH headings, and the counts.",
             "", "### Questions with all three facts", "",
             "A missing fact points to melanoma or nevus (their photo collection records fewer facts), so each run is also scored on "
             "the questions with all three facts.", "",
             "| Class | " + " | ".join(splits) + " | Missing age | Missing sex | Missing site |", "|---|" + "---|" * 6]
    for label in labels:
        mine = [q for q in questions if q["label"] == label]
        lines.append(f"| {label} | " + " | ".join(
            f"{sum(q['split'] == s and q['complete'] == 'yes' for q in mine)} of {sum(q['split'] == s for q in mine)}" for s in splits)
            + " | " + " | ".join(str(sum(q[f] == NOT_RECORDED for q in mine)) for f in ("age", "sex", "site")) + " |")
    lines += ["", "### Examples of what the model is shown next to the photo", ""]
    seen = []
    for q in questions:
        if q["facts"] not in seen and len(seen) < 4 and (q["complete"] == "yes") == (len(seen) % 2 == 0):
            seen.append(q["facts"])
    lines += [f"- \"{s}\"" for s in seen]
    lines += ["", "### Not done in this step", "",
              "- Whether retrieval may see the facts of training images is still to be decided: knowledge.csv names the training "
              "images, and their facts are in questions.csv, so either choice needs no new build.",
              "- The audit checks the tables against their sources and measures how much each fact, and its absence, gives the class "
              "away.", ""]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.multimodal.prepare", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("records", help="report which patient facts the ISIC records of the image dataset hold (no file is changed)")
    sub.add_parser("build", help="write questions.csv, knowledge.csv and diseases.csv from the frozen datasets and the ISIC records")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        done = records() if args.step == "records" else build()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if args.step == "records":
        print(f"ISIC records read and checked: {done['images']}")
        for field, (kind, n) in done["fields"].items():
            print(f"  {field}: {kind}, in {n} records")
    else:
        print(f"Questions: {done['questions']} (" + ", ".join(f"{s} {done['splits'][s]}" for s in ("train", "val", "test"))
              + "); with all three facts: " + ", ".join(f"{s} {done['complete'][s]}" for s in ("train", "val", "test")))
        print("Knowledge items: " + ", ".join(f"{k} {n}" for k, n in sorted(done["knowledge"].items())))
    print(f"Written: {done['report']} (the {args.step} section)")


if __name__ == "__main__":
    main(sys.argv[1:])
