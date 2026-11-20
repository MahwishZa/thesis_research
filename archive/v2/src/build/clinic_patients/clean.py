"""Step ``clean`` of the clinic patients cohort: the clinical record, checked and normalised by fixed rules, with every decision written
down. It reads raw/metadata.csv and changes no original file.

What it does, and what it never does:
- A gap (blank, ``UNK``, ``NaN``, ...) is written as ``not recorded`` in the cleaned table, in the same cells as in the original: which
  fields are empty is information (see step ``shortcuts``), so no gap is filled in and no row is left out.
- A value that differs only in case or spaces is written in its standard form; the one known spelling variant (``BRASIL`` for
  ``BRAZIL``) is unified. Each such change is a row of build/cleaning.csv. Nothing else is rewritten.
- A value that is not in the known vocabulary of a controlled column (the diagnosis, the body region, gender, skin type, the yes/no
  columns) stops the step with a message: a new kind of value is a decision for the researcher, not for the code.
- A value that is implausible, or that disagrees inside one patient or one lesion, is **flagged** in build/cleaning.csv and kept as
  recorded. The later steps and the researcher decide what a flag means.

Writes build/cleaned.csv (the same columns and rows as metadata.csv, cleaned), build/cleaning.csv (every change and flag) and the clean
section of dataset_report.md.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ...common import write_section
from .table import (BUILD, DATA, EXAMPLES, NOT_RECORDED, RAW, REPORT, REPORT_HEAD, STEPS, clinical_columns, find_metadata, is_empty,
                    lesion_key, names, read_metadata, write_csv)

CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")  # the six diagnoses of the authors (Bowen's disease counted as SCC)
REGIONS = ("ABDOMEN", "ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT", "FOREARM", "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH")
GENDERS = ("FEMALE", "MALE")
FITZPATRICK = ("1", "2", "3", "4", "5", "6")
YES_NO = ("smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system", "itch",
          "grew", "hurt", "changed", "bleed", "elevation")  # TRUE or FALSE, or a gap
ORIGIN = ("background_father", "background_mother")  # a country or region, free text: only the spelling is unified
SPELLING = {"BRASIL": "BRAZIL"}  # the only variant found in the download (inventory, 2026-11-19)
PATIENT_FIXED = ("smoke", "drink", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
                 "has_sewage_system", "fitspatrick")  # fields of the patient, not of the lesion or the visit: one value per patient
AGE_RANGE = (0, 120)
CLEANING_COLUMNS = ["img_id", "patient_id", "column", "old", "new", "action", "reason"]


def _flag(rows: list[dict], row: dict, column: str, reason: str) -> None:
    rows.append({"img_id": row["img_id"], "patient_id": row["patient_id"], "column": column, "old": row[column], "new": row[column],
                 "action": "flagged", "reason": reason})


def clean(data: Path = DATA) -> dict:
    """Writes build/cleaned.csv, build/cleaning.csv and the clean section of the report (see the module's description)."""
    data = Path(data)
    columns, rows = read_metadata(find_metadata(data / RAW))
    clinical = clinical_columns(columns)
    cleaned, log, unknown = [], [], []
    gaps = Counter()  # column -> how many cells were gaps, now written as NOT_RECORDED

    def controlled(column: str, value: str, allowed: tuple[str, ...], row: dict, gap_allowed: bool) -> str:
        if is_empty(value):
            if gap_allowed:
                gaps[column] += 1
                return NOT_RECORDED
            unknown.append((row["img_id"], column, value))
            return value
        standard = value.strip().upper()
        if standard not in allowed:
            unknown.append((row["img_id"], column, value))
        return standard

    for row in rows:
        out = {}
        for column in columns:
            value = (row[column] or "").strip()
            if column == "diagnostic":
                new = controlled(column, value, CLASSES, row, gap_allowed=False)
            elif column == "biopsed":
                new = controlled(column, value, ("TRUE", "FALSE"), row, gap_allowed=False)
            elif column == "region":
                new = controlled(column, value, REGIONS, row, gap_allowed=False)
            elif column == "gender":
                new = controlled(column, value, GENDERS, row, gap_allowed=True)
            elif column == "fitspatrick":
                new = controlled(column, value, FITZPATRICK, row, gap_allowed=True)
            elif column in YES_NO:
                new = controlled(column, value, ("TRUE", "FALSE"), row, gap_allowed=True)
            elif column in ORIGIN:
                if is_empty(value):
                    gaps[column] += 1
                    new = NOT_RECORDED
                else:
                    new = SPELLING.get(value.upper(), value.upper())
            elif column in clinical:  # age and the diameters: a number, or a gap
                if is_empty(value):
                    gaps[column] += 1
                    new = NOT_RECORDED
                else:
                    new = value
            else:  # identifiers
                new = value
            out[column] = new
            original = row[column] or ""
            if new != original and not (new == NOT_RECORDED and is_empty(original)):
                reason = ("spelling variant unified" if column in ORIGIN and value.upper() in SPELLING
                          else "case or spaces standardised")
                log.append({"img_id": row["img_id"], "patient_id": row["patient_id"], "column": column, "old": original, "new": new,
                            "action": "normalised", "reason": reason})
        cleaned.append(out)
    if unknown:
        shown = "; ".join(f"{img}: {column} = {value!r}" for img, column, value in unknown[:EXAMPLES])
        raise ValueError(f"{len(unknown)} value(s) outside the known vocabulary (first {min(EXAMPLES, len(unknown))}: {shown}). "
                         "Decide how to handle them (src/build/clinic_patients/clean.py) before going on")

    flags = []
    for row in cleaned:
        if "age" in row and row["age"] != NOT_RECORDED:
            try:
                age = int(row["age"])
                if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
                    _flag(flags, row, "age", f"age outside {AGE_RANGE[0]} to {AGE_RANGE[1]}")
            except ValueError:
                _flag(flags, row, "age", "age is not a whole number")
        for column in ("diameter_1", "diameter_2"):
            if column in row and row[column] != NOT_RECORDED:
                try:
                    size = float(row[column])
                    if size == 0:
                        _flag(flags, row, column, "diameter is 0: implausible for a lesion")
                    elif size < 0:
                        _flag(flags, row, column, "diameter is negative")
                except ValueError:
                    _flag(flags, row, column, "diameter is not a number")
    by_patient, by_lesion = defaultdict(lambda: defaultdict(set)), defaultdict(set)
    for row in cleaned:
        for column in PATIENT_FIXED:
            if column in row and row[column] != NOT_RECORDED:
                by_patient[row["patient_id"]][column].add(row[column])
        if "region" in row:
            by_lesion[lesion_key(row)].add(row["region"])
    patient_conflicts = [(patient, column, sorted(values)) for patient, fixed in sorted(by_patient.items())
                         for column, values in sorted(fixed.items()) if len(values) > 1]
    for patient, column, values in patient_conflicts:
        flags.append({"img_id": "", "patient_id": patient, "column": column, "old": "|".join(values), "new": "|".join(values),
                      "action": "flagged", "reason": "differs between photos of one patient"})
    lesion_conflicts = [(key, sorted(values)) for key, values in sorted(by_lesion.items()) if len(values) > 1]
    for (patient, lesion), values in lesion_conflicts:
        flags.append({"img_id": "", "patient_id": patient, "column": "region", "old": "|".join(values), "new": "|".join(values),
                      "action": "flagged", "reason": f"lesion {lesion} is on different body regions"})

    write_csv(data / BUILD / "cleaned.csv", columns, cleaned)
    write_csv(data / BUILD / "cleaning.csv", CLEANING_COLUMNS, log + flags)
    label = {row["img_id"]: row["diagnostic"] for row in cleaned}
    write_section(data / REPORT, REPORT_HEAD, STEPS, "clean", _report(len(cleaned), gaps, log, flags, label, patient_conflicts,
                                                                       lesion_conflicts))
    return {"summary": f"{len(cleaned)} rows; {sum(gaps.values())} gaps written as '{NOT_RECORDED}'; {len(log)} values normalised; "
                       f"{len(flags)} flags", "rows": len(cleaned), "gaps": dict(gaps), "normalised": len(log), "flags": len(flags)}


def _report(n: int, gaps: Counter, log: list[dict], flags: list[dict], label: dict, patient_conflicts, lesion_conflicts) -> list[str]:
    lines = ["# Clean: the clinical record, checked and normalised", "",
             "Written by `python -m src.build.clinic_patients.prepare clean`. The original `metadata.csv` is not changed; the result is "
             "`build/cleaned.csv`, and every change and flag is a row of `build/cleaning.csv`.", "",
             f"- Rows: {n}. No row is left out, and no gap is filled in.",
             f"- Gaps (blank, `UNK` and the like) written as `{NOT_RECORDED}`: {sum(gaps.values())} cells in {len(gaps)} columns. Which "
             "cells are gaps does not change: the pattern of gaps is checked in step `shortcuts`.",
             f"- Values standardised: {sum(r['reason'] == 'case or spaces standardised' for r in log)} (case or spaces); "
             f"{sum(r['reason'] == 'spelling variant unified' for r in log)} (spelling: "
             + ", ".join(f"`{a}` to `{b}`" for a, b in SPELLING.items()) + ")",
             "- Every value of the diagnosis, body region, gender, skin type and yes/no columns is in the known vocabulary "
             "(otherwise this step stops).", "",
             "## Flags (kept as recorded)", "", "| Flag | Count | Where |", "|---|---|---|"]
    by_reason = Counter(r["reason"] for r in flags if r["img_id"])
    for reason, count in sorted(by_reason.items()):
        mine = [r for r in flags if r["reason"] == reason]
        classes = Counter(label.get(r["img_id"], "") for r in mine)
        lines.append(f"| {reason} | {count} | " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())) + " |")
    lines.append(f"| differs between photos of one patient (a patient-level field) | {len(patient_conflicts)} | "
                 + (names({f"{p}: {c}" for p, c, _ in patient_conflicts}) if patient_conflicts else "none") + " |")
    lines.append(f"| one lesion on different body regions | {len(lesion_conflicts)} | "
                 + (names({f"{p}/{lesion}" for (p, lesion), _ in lesion_conflicts}) if lesion_conflicts else "none") + " |")
    if not flags:
        lines.append("| (none) | 0 | |")
    lines += ["", "A flag is not a correction. A flagged value stays as recorded; a view that shows a field decides what to do with its "
              "flags.", ""]
    return lines
