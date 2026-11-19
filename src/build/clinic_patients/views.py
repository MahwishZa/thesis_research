"""Step ``views`` of the clinic patients cohort: the same questions in three views, and the final tables.

One question is one photo of one patient's lesion, with the record of the same clinic visit. The views differ only in what a model is
shown, so that the same question can be asked three times and the answers paired:
- **photo**: the standardised photo (step ``images``) alone;
- **record**: the patient's record alone, as text (no photo);
- **photo + record**: the photo, with the same record text in the prompt.

**The record** is six fields in fixed wording, one numbered line each (``TEMPLATE``, version ``record-v1``): age, body site, and whether
the lesion itches, hurts, has bled and has an elevation (the authors' own words for these fields, Pacheco et al. 2020). A field with no
entry reads ``unknown``: in the authors' words the patient did not know the answer (``UNK``). Nothing is guessed or filled in. Age and body
site are always present according to the authors; a gap in one of them stops the step. The record is the same words in the record view
and in the photo + record view, to the byte.

**What the record does not show:** whether the lesion grew or changed (more often unknown for cancers than for benign lesions: a
hint, see step ``shortcuts``) and the block of fields added later to the questionnaire (gender, smoking, drinking, pesticides,
histories, water, sewage, skin type, diameters, the parents' origin: empty for 804 early photos, all benign). They stay in dataset.csv
for a separate secondary analysis and are not in any view. The wording of the lines is provisional until the authors' paper confirms
what each field means; a change of wording is a new template version and a rebuild of the views.

A field a line cannot print (an age that is not a whole number from 0 to 120, a value outside yes, no and a gap, a gap in the age or the
body site) stops the step: a decision for the researcher, not for the code.

Writes files/records/<split>/<class>/<img_id>.txt (not committed; one record per question, no trailing newline), dataset.csv (the
photo view: build/split.csv with the photo's ``path`` and ``sha256``, the original's ``raw_path`` and ``raw_sha256``, and the record
text as ``facts``; the photo + record view reads the same table), record.csv (the record view: the same questions, ``path`` and
``sha256`` are the record file, ``photo_path`` and ``photo_sha256`` the photo it belongs to) and the views section of dataset_report.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...common import write_section
from .clean import AGE_RANGE
from .split import SPLITS
from .table import (BUILD, DATA, DATASET, NOT_RECORDED, QUESTION_COLUMNS, RECORDS, REPORT, REPORT_HEAD, SPLIT, STEPS, names, read_csv,
                    remove_stale, write_csv)

RECORD = "record.csv"  # in DATA: the record view's table
UNKNOWN = "unknown"  # how a gap reads in a symptom line: the patient did not know the answer (the authors' UNK)
TEMPLATE_VERSION = "record-v1"


@dataclass(frozen=True)
class Field:
    column: str  # the column of the cleaned table
    line: str  # the words before the value
    kind: str  # "age", "site" or "yes_no": how the value is written


FIELDS = (Field("age", "Age in years", "age"), Field("region", "Body site", "site"),
          Field("itch", "The lesion itches", "yes_no"), Field("hurt", "The lesion hurts", "yes_no"),
          Field("bleed", "The lesion has bled", "yes_no"), Field("elevation", "The lesion has an elevation", "yes_no"))
CORE = tuple(f.column for f in FIELDS)  # the fields of the record
SYMPTOMS = tuple(f.column for f in FIELDS if f.kind == "yes_no")  # the only fields that can be a gap
LEFT_OUT = ("grew", "changed")  # not in any view: unknown more often for cancers (step shortcuts)
YES_NO = {"TRUE": "yes", "FALSE": "no"}
TEMPLATE = "\n".join(f"{number}. {f.line}: <{f.column}>" for number, f in enumerate(FIELDS, start=1))  # the record with its blanks
FINGERPRINT = hashlib.sha256(f"{TEMPLATE_VERSION}\n{TEMPLATE}\nnot recorded: {NOT_RECORDED}".encode()).hexdigest()
RECORD_COLUMNS = ["split", "label", "path", "sha256", "img_id", "patient_id", "lesion_id", "split_group", "biopsed", "photo_path",
                  "photo_sha256"]
DATASET_FIRST = ["split", "label", "path", "sha256", "img_id", "patient_id", "lesion_id", "split_group", "biopsed", "raw_path", "raw_sha256",
                 "facts"]
assert tuple(DATASET_FIRST) == QUESTION_COLUMNS  # step shortcuts takes every other column of dataset.csv for the record


def value_of(field: Field, row: dict) -> str:
    """How one field of a question's cleaned record is written in the record. Raises ValueError for a value the line cannot print."""
    value = row[field.column]
    if value == NOT_RECORDED:
        if field.kind != "yes_no":
            raise ValueError(f"{row['img_id']}: {field.column} is not recorded, but the authors say it is always present")
        return UNKNOWN
    if field.kind == "age":
        if not (value.isascii() and value.isdigit()) or not AGE_RANGE[0] <= int(value) <= AGE_RANGE[1]:
            raise ValueError(f"{row['img_id']}: {field.column} = {value!r} is not a whole number from {AGE_RANGE[0]} to {AGE_RANGE[1]}")
        return str(int(value))
    if field.kind == "site":
        return value.lower()
    if value not in YES_NO:
        raise ValueError(f"{row['img_id']}: {field.column} = {value!r} is neither yes, no nor a gap")
    return YES_NO[value]


def record_text(row: dict) -> str:
    """The record of one question: six numbered lines, no trailing newline."""
    return "\n".join(f"{number}. {f.line}: {value_of(f, row)}" for number, f in enumerate(FIELDS, start=1))


def views(data: Path = DATA) -> dict:
    """Writes files/records/, dataset.csv, record.csv and the views section of the report (see the module's description)."""
    data = Path(data)
    questions = read_csv(data / BUILD / SPLIT, "run the split step first")
    made = {r["img_id"]: r for r in read_csv(data / BUILD / "files.csv", "run the images step first")}
    raw = {r["img_id"]: r for r in read_csv(data / BUILD / "raw_images.csv", "run the inventory step first")}
    absent = [c for c in CORE if c not in questions[0]]
    if absent:
        raise ValueError(f"the cleaned records have no column {', '.join(absent)} (build/split.csv has {', '.join(questions[0])})")
    missing = sorted({q["img_id"] for q in questions} - set(made))
    if missing:
        raise ValueError(f"no standardised photo for {names(missing)}: run the images step again")
    texts = [record_text(q) for q in questions]  # every record is checked before anything is written
    root = data / RECORDS
    photo_rows, record_rows, written = [], [], 0
    for q, text in zip(questions, texts):
        path = f"{q['split']}/{q['label']}/{Path(q['img_id']).stem}.txt"
        content = text.encode("utf-8")
        file = root / path
        if not file.is_file() or file.read_bytes() != content:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
            written += 1
        photo = made[q["img_id"]]
        base = {k: q[k] for k in ("split", "label", "img_id", "patient_id", "lesion_id", "split_group", "biopsed")}
        photo_rows.append({**base, "path": photo["path"], "sha256": photo["sha256"], "raw_path": raw[q["img_id"]]["file"],
                           "raw_sha256": raw[q["img_id"]]["sha256"], "facts": text,
                           **{c: q[c] for c in q if c not in DATASET_FIRST and c not in base}})
        record_rows.append({**base, "path": path, "sha256": hashlib.sha256(content).hexdigest(), "photo_path": photo["path"],
                            "photo_sha256": photo["sha256"]})
    removed = remove_stale(root, {r["path"] for r in record_rows}, ".txt")
    columns = DATASET_FIRST + [c for c in photo_rows[0] if c not in DATASET_FIRST]  # then the rest of the clinical record
    write_csv(data / DATASET, columns, photo_rows)
    write_csv(data / RECORD, RECORD_COLUMNS, record_rows)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "views", _report(photo_rows))
    shared = sum(len(g) for g in _identical(photo_rows))
    return {"summary": f"{len(photo_rows)} questions in three views (template {TEMPLATE_VERSION}); {written} records written, {removed} "
                       f"removed; {shared} questions share their record with another question",
            "questions": len(photo_rows), "written": written, "removed": removed}


def _identical(rows: list[dict]) -> list[list[dict]]:
    """The groups of two or more questions whose record text is the same."""
    by_text: dict[str, list[dict]] = {}
    for r in rows:
        by_text.setdefault(r["facts"], []).append(r)
    return [group for _, group in sorted(by_text.items()) if len(group) > 1]


def _report(rows: list[dict]) -> list[str]:
    classes = sorted({r["label"] for r in rows})
    lines = ["# Views: the same questions as photo, record, and photo + record", "",
             "Written by `python -m src.build.clinic_patients.prepare views`. One question is one photo of one patient's lesion with the "
             "record of the same visit. `dataset.csv` lists the photo view and the photo + record view (the record text is its `facts` "
             "column); `record.csv` lists the record view; the records are in `files/records/` (not committed).", "",
             f"- Questions: {len(rows)} ({', '.join(f'{s} {sum(r['split'] == s for r in rows)}' for s in SPLITS)}). Each is in all three views.",
             f"- Record template `{TEMPLATE_VERSION}` (fingerprint `{FINGERPRINT[:12]}`); the lines use the authors' own words for each field "
             f"(Pacheco et al. 2020). A gap in a symptom line is written `{UNKNOWN}`: the patient did not know the answer. Age and body site "
             "are always present (a gap would stop the step). Nothing is filled in.", "",
             "```", TEMPLATE, "```", "",
             f"- Not in any view: {', '.join(f'`{c}`' for c in LEFT_OUT)} (unknown more often for cancers: a hint), and the later "
             "questionnaire fields. They stay in `dataset.csv` for a separate secondary analysis.", "",
             f"## `{UNKNOWN}` in the record, by class", "",
             f"| Class | Questions | No `{UNKNOWN}` in the record | " + " | ".join(f"`{c}` {UNKNOWN}" for c in SYMPTOMS) + " |",
             "|---|---|---|" + "---|" * len(SYMPTOMS)]
    for label in [*classes, None]:
        mine = [r for r in rows if label is None or r["label"] == label]
        full = sum(all(r[c] != NOT_RECORDED for c in SYMPTOMS) for r in mine)
        lines.append(f"| {label or 'all'} | {len(mine)} | {full} | " + " | ".join(str(sum(r[c] == NOT_RECORDED for r in mine)) for c in SYMPTOMS) + " |")
    same = _identical(rows)
    within = sum(any(o is not r and o["patient_id"] == r["patient_id"] for o in g) for g in same for r in g)
    across = sum(any(o["patient_id"] != r["patient_id"] for o in g) for g in same for r in g)
    mixed = sum(len(g) for g in same if len({r["label"] for r in g}) > 1)
    lines += ["", "## How much a record can say", "",
              f"- Distinct records: {len({r['facts'] for r in rows})} for {len(rows)} questions. Questions whose record is the same as "
              f"another question's: {sum(len(g) for g in same)} (in {len(same)} groups).",
              f"  - with another photo of the same patient: {within} (several photos of one visit share one record);",
              f"  - with a photo of a different patient: {across};",
              f"  - in a group of more than one class: {mixed} (the record alone cannot tell these apart).",
              "- Identical records are kept: the record view asks every question, so that it pairs with the other two views. Photos of "
              "one patient are never in two splits, and the intervals resample whole patients.", "",
              "## Examples (the first question of each class in the train split)", ""]
    for label in classes:
        first = min((r for r in rows if r["label"] == label and r["split"] == "train"), key=lambda r: r["img_id"], default=None)
        if first:
            lines += [f"**{label}** (`{first['img_id']}`)", "", "```", first["facts"], "```", ""]
    return lines
