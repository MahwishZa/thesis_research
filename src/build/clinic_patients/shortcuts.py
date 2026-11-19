"""Step ``shortcuts`` of the clinic patients cohort: how much of the class can be told from things that are not the lesion.

A model that sees the clinical record or the photo could use a hint that has nothing to do with the lesion: which fields are empty
(the questionnaire grew over the years of collection, and the early records are almost all benign lesions), or the size of the photo
(different phones, different years). This step measures such hints with a deliberately simple learner, so that the thesis can report
them next to the results, and so that no field that gives the class away by its gaps is shown as clinical evidence without saying so.

The learner is a nearest-neighbour vote (``K`` neighbours, features standardised on the training part of each fold). It is a lower
bound on what a better learner could find, not a model of any LLM. It is tried on **the train split only** (the validation and test
splits are never looked at here), in ``FOLDS`` folds that never divide a split group, so no patient is in a fold's training and
testing part at once. Scores are balanced accuracy over the six classes (chance 1/6) and over cancer against not cancer
(BCC, SCC, MEL against ACK, NEV, SEK; chance 1/2), with a 95% interval from resampling whole split groups.

Feature sets: ``gaps`` (is each clinical field empty?), ``photo properties`` (width, height, file size, bytes per pixel), of the original
files and of the standardised files (step ``images``; those are what a model receives, so the second is the one that matters),
``core record`` (age, body site, itch, hurt, bleed, elevation: what a view would show), ``full record`` (every clinical field, gaps
as a value). The last two are information, not hints: they measure how informative the record is.

Writes build/shortcuts.csv and the shortcuts section of dataset_report.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ... import metrics
from ...common import write_section
from .split import SEED
from .table import BUILD, DATA, DATASET, NOT_RECORDED, QUESTION_COLUMNS, REPORT, REPORT_HEAD, STEPS, read_csv, write_csv

FOLDS, K = 5, 15
CANCER = ("BCC", "MEL", "SCC")
CORE = ("age", "region", "itch", "hurt", "bleed", "elevation")
RESULT_COLUMNS = ["features", "task", "balanced_accuracy", "ci_low", "ci_high", "chance", "recall_by_class"]


def encode(rows: list[dict], columns: list[str]) -> np.ndarray:
    """The rows' values in ``columns`` as numbers: a column of numbers stays one (a gap is 0, and a 0/1 column says it was a gap); any
    other column becomes one 0/1 column per value seen."""
    parts = []
    for column in columns:
        values = [r[column] for r in rows]
        recorded = [v for v in values if v != NOT_RECORDED]
        try:
            numbers = [float(v) for v in recorded]
        except ValueError:
            numbers = None
        if numbers is not None and recorded:
            parts.append(np.array([0.0 if v == NOT_RECORDED else float(v) for v in values])[:, None])
            if len(recorded) != len(values):
                parts.append(np.array([1.0 if v == NOT_RECORDED else 0.0 for v in values])[:, None])
        else:
            for value in sorted(set(values)):
                parts.append(np.array([1.0 if v == value else 0.0 for v in values])[:, None])
    return np.hstack(parts) if parts else np.zeros((len(rows), 0))


def photo_properties(entry: dict) -> list[float]:
    """Width, height, file size and bytes per pixel of a photo, from its row of build/raw_images.csv."""
    width, height, size = float(entry["width"]), float(entry["height"]), float(entry["bytes"])
    return [width, height, size, size / (width * height)]


def gaps(rows: list[dict], columns: list[str]) -> np.ndarray:
    return np.array([[1.0 if r[c] == NOT_RECORDED else 0.0 for c in columns] for r in rows])


def vote(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, classes: list[str], k: int | None = None) -> list[str]:
    """Nearest-neighbour vote after standardising on the training rows. A tie goes to the class that comes first in ``classes``."""
    k = K if k is None else k
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std == 0] = 1.0
    a, b = (train_x - mean) / std, (test_x - mean) / std
    dist = (b ** 2).sum(axis=1)[:, None] + (a ** 2).sum(axis=1)[None, :] - 2 * b @ a.T
    nearest = np.argsort(dist, axis=1, kind="stable")[:, :k]
    out = []
    for row in nearest:
        counts = np.bincount([classes.index(train_y[i]) for i in row], minlength=len(classes))
        out.append(classes[int(np.argmax(counts))])
    return out


def folds(groups: list[str]) -> list[int]:
    """The fold of each row: whole groups go to one fold, by a fixed hash of the seed and the group."""
    return [int(hashlib.sha256(f"{SEED}:fold:{g}".encode()).hexdigest()[:8], 16) % FOLDS for g in groups]


def cross_validate(x: np.ndarray, y: list[str], groups: list[str], classes: list[str]) -> list[str]:
    """The prediction for every row, each made by a vote of rows of other folds."""
    fold = np.array(folds(groups))
    out = [""] * len(y)
    labels = np.array(y)
    for f in range(FOLDS):
        test, train = np.nonzero(fold == f)[0], np.nonzero(fold != f)[0]
        if len(test) == 0 or len(train) == 0:
            continue
        for i, p in zip(test, vote(x[train], labels[train], x[test], classes)):
            out[i] = p
    return out


def shortcuts(data: Path = DATA) -> dict:
    """Writes build/shortcuts.csv and the shortcuts section of the report (see the module's description)."""
    data = Path(data)
    table = [r for r in read_csv(data / DATASET, "run the images step first") if r["split"] == "train"]
    originals = {r["img_id"]: r for r in read_csv(data / BUILD / "raw_images.csv", "run the inventory step first")}
    made = {r["img_id"]: r for r in read_csv(data / BUILD / "files.csv", "run the images step first")}
    if not table:
        raise ValueError("the train split is empty")
    clinical = [c for c in table[0] if c not in QUESTION_COLUMNS]
    classes = sorted({r["label"] for r in table})
    y, groups = [r["label"] for r in table], [r["split_group"] for r in table]
    original = np.array([photo_properties(originals[r["img_id"]]) for r in table])
    standardised = np.array([photo_properties(made[r["img_id"]]) for r in table])
    core = [c for c in CORE if c in clinical]
    sets = {"gaps": gaps(table, clinical), "photo properties (originals)": original, "photo properties (standardised)": standardised,
            "core record": encode(table, core), "full record": encode(table, clinical)}
    results = []
    for name, x in sets.items():
        for task, labels, order in (("6 classes", y, classes), ("cancer or not", ["cancer" if v in CANCER else "not" for v in y],
                                                                  ["cancer", "not"])):
            predicted = cross_validate(x, labels, groups, order)
            m = metrics.summarize(labels, predicted, order, groups, group_name="split group")
            low, high = m["balanced_accuracy_ci95"] or (None, None)
            results.append({"features": name, "task": task, "balanced_accuracy": round(m["balanced_accuracy"], 4),
                            "ci_low": "" if low is None else round(low, 4), "ci_high": "" if high is None else round(high, 4),
                            "chance": round(1 / len(order), 4),
                            "recall_by_class": "; ".join(f"{c} {m['per_class'][c]['recall']:.2f}" for c in order
                                                         if m["per_class"][c]["recall"] is not None)})
    write_csv(data / BUILD / "shortcuts.csv", RESULT_COLUMNS, results)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "shortcuts", _report(results, len(table), len({r["patient_id"] for r in table}),
                                                                           len(set(groups)), len(clinical)))
    best = {r["features"]: r["balanced_accuracy"] for r in results if r["task"] == "6 classes"}
    both = {r["features"]: r["balanced_accuracy"] for r in results if r["task"] == "cancer or not"}
    return {"summary": "train split only; balanced accuracy, 6 classes (chance 16.7%) / cancer or not (chance 50%): "
                       + "; ".join(f"{k} {best[k]:.1%} / {both[k]:.1%}" for k in best), "results": results}


def _report(results: list[dict], photos: int, patients: int, groups: int, fields: int) -> list[str]:
    lines = ["# Shortcuts: what the gaps and the photo properties alone give away", "",
             "Written by `python -m src.build.clinic_patients.prepare shortcuts`. A nearest-neighbour vote "
             f"({K} neighbours) on the **train split only** ({photos} photos, {patients} patients, {groups} split groups), in {FOLDS} "
             "folds that never divide a split group. The validation and test splits are not used. The score is balanced accuracy with "
             "a 95% interval from resampling whole split groups; the learner is simple, so a real hint can be larger than shown.", "",
             "| Features | 6 classes (chance 16.7%) | Cancer or not (chance 50%) | Recall by class (6 classes) |", "|---|---|---|---|"]
    by = {(r["features"], r["task"]): r for r in results}
    for name in dict.fromkeys(r["features"] for r in results):
        six, two = by[(name, "6 classes")], by[(name, "cancer or not")]

        def cell(r):
            if r["ci_low"] == "":
                return f"{r['balanced_accuracy']:.1%}"
            return f"{r['balanced_accuracy']:.1%} ({r['ci_low']:.1%} to {r['ci_high']:.1%})"
        lines.append(f"| {name} | {cell(six)} | {cell(two)} | {six['recall_by_class']} |")
    lines += ["", f"- `gaps`: {fields} clinical fields, each 1 if empty. `photo properties`: width, height, file size, bytes per pixel, of the original files and of the standardised files (the ones a model receives). "
              "`core record`: " + ", ".join(CORE) + " (what a view would show). `full record`: every clinical field.",
              "- `gaps` and `photo properties` are hints that are not about the lesion: a score well above chance means a model could "
              "use them. The records are information: how much the record says about the class, legitimately or not.", ""]
    return lines
