"""The dataset's files and the three views: where the images, case tables and question files are, the balanced test and
validation splits (used by preprocessing.py), and the question files of each view (python -m src.main views). Only the Python
standard library is used, so the split rule and the file layout are tested and checked without the image packages."""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict

from src.utils import (
    ALL_CASES,
    CLASSES,
    LETTERS,
    OPTION_TEXT,
    PROCESSED_IMAGES,
    SEED,
    SPLIT_FILES,
    SPLITS,
    VIEWS,
    VIEWS_DIR,
    get_logger,
    read_csv,
    write_csv,
    write_jsonl,
)


# =============================================================================================================================
# Where the files are: the case images, the case tables and the question files of each view
# =============================================================================================================================


def image_path(isic_id):
    """Where a cleaned image is, relative to the data folder, with "/" on every system (never "\\" from Windows)."""
    return f"{PROCESSED_IMAGES}/{isic_id}.jpg"


def images_not_in_table(data, isic_ids):
    """The .jpg files in data/processed/images/ whose name is not a case in the table (left over from an earlier run), sorted."""
    folder = os.path.join(data, PROCESSED_IMAGES)
    if not os.path.isdir(folder):
        return []
    keep = {f"{i}.jpg" for i in isic_ids}
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg") and f not in keep)


def view_table(view):
    return f"{VIEWS_DIR}/{view}/cases.csv"


def question_file(view, split):
    return f"{VIEWS_DIR}/{view}/{SPLIT_FILES[split]}.jsonl"


def moved_files(data):
    """data/views/provenance.json: for each file that the 2026-11-25 move changed (new image paths, a new file name), its earlier
    path and its SHA-256 before and after, keyed by its path now (relative to the data folder). Empty when there is no record."""
    path = os.path.join(data, VIEWS_DIR, "provenance.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["files"]


def recorded_question_path(root, recorded):
    """The question file that a run recorded (a path relative to the repository folder, e.g. data/views/text_only/val.jsonl), at
    its place now: a file renamed by the 2026-11-25 move is found through data/views/provenance.json."""
    path = os.path.join(root, recorded)
    if os.path.isfile(path):
        return path
    data, _, rest = recorded.replace("\\", "/").partition("/")
    for now, record in moved_files(os.path.join(root, data)).items():
        if record["previous_path"] == rest:
            return os.path.join(root, data, now)
    return path


# =============================================================================================================================
# The balanced test and validation splits
# =============================================================================================================================

TEST_PER_CLASS = 30                   # images of each class in the balanced test split
VAL_PER_CLASS = 20                    # images of each class in the balanced validation split


# ---- the balanced test and validation splits (used by preprocessing.py step 8, checked by main.py status) -----------------------------
# The balanced splits (test decided 2026-11-22: 30 per class; validation decided 2026-11-22: 20 per class): the split holds exactly
# the same number of images of every class, so every class weighs the same in its accuracy. Used by preprocessing.py (step 8)
# after the lesion-grouped split: first for "test", then for "val" (the test split is not touched by the second pass). Train keeps
# the natural class mix (the imbalance is handled when a model is trained: class weights, the oversampled and undersampled lists).
#
# The fixed rule, for each class on its own, for the split being balanced (call it S):
#   1. Whole lesion groups only: a group (the photos of one lesion) is moved as a whole, so no lesion is ever in two splits.
#   2. Order: the groups are taken in a fixed order, the SHA-256 of "<seed>:<group_id>" (no hand choice, the same on every system).
#   3. Keep: the class's S groups, in that order, as long as they fit within the target (a group that would go over it is
#      skipped); the S groups not kept move to train.
#   4. Fill: if the class still has fewer than the target, train groups of the class are added in the same order, again skipping a
#      group that would go over it.
#   5. Every split other than S and train is never changed.
# The rule stops with an error when a class cannot reach the target exactly. Only the Python standard library is used, so the
# rule is tested on the committed table without the image packages.
def group_order(seed, group_id):
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def balanced_splits(cases, per_class=TEST_PER_CLASS, seed=SEED, split="test"):
    """cases: [{"isic_id", "group_id", "label", "split"}, ...] after the lesion-grouped split.
    Returns {isic_id: new split} for every case (the other splits unchanged; train and `split` changed only by whole groups)."""
    groups = defaultdict(list)
    for c in cases:
        groups[c["group_id"]].append(c)
    for g, members in groups.items():
        if len({m["label"] for m in members}) != 1 or len({m["split"] for m in members}) != 1:
            raise ValueError(f"lesion group {g} has more than one class or split")
    new = {c["isic_id"]: c["split"] for c in cases}
    for label in sorted({c["label"] for c in cases}):
        def ordered(which):
            gs = [g for g, m in groups.items() if m[0]["label"] == label and m[0]["split"] == which]
            return sorted(gs, key=lambda g: group_order(seed, g))
        n = 0
        for g in ordered(split):
            if n + len(groups[g]) <= per_class:
                n += len(groups[g])
            else:
                for m in groups[g]:
                    new[m["isic_id"]] = "train"
        for g in ordered("train"):
            if n + len(groups[g]) <= per_class:
                n += len(groups[g])
                for m in groups[g]:
                    new[m["isic_id"]] = split
        if n != per_class:
            raise ValueError(f"{label}: only {n} {split} images can be reached by whole lesion groups, not {per_class}")
    return new


# ----------------------------------------------------------------------------------------------------------------------
# views: step 2 of the data pipeline (python -m src.main views)
# ----------------------------------------------------------------------------------------------------------------------
# Step 2 of the data pipeline: turns the cleaned cases (data/processed/cases/all_cases.csv) into the three views of the same cases.
#
#   data/views/image_only/   the image only
#   data/views/text_only/    the patient's recorded facts only (age, sex, body site in one fixed sentence)
#   data/views/multimodal/   the image and the same sentence
#
# Each view folder holds
#   cases.csv                one row per case: id, split, fold, lesion group, label, and only the inputs of that view
#   train.jsonl, validation.jsonl, test.jsonl
#                            one 8-option multiple-choice question per case (MedQA style) for the LLM and RAG systems
#
# The questions use one fixed template and real data only; nothing is generated. The 8 options are shuffled for each case with a
# fixed seed (the same order for a case in all three views), so a letter never stands for one class. The question never names a
# class, and a text-only question never mentions an image.
#
# Usage: python -m src.main views --data data

OPTION_SEED = SEED
CORE = ["isic_id", "split", "cv_fold", "group_id", "label", "label_idx", "label_name"]
VIEW_INPUTS = {"image_only": ["image_path"],
               "text_only": ["clinical_text", "age", "sex", "site"],
               "multimodal": ["image_path", "clinical_text", "age", "sex", "site"]}
Q_IMAGE = "Look at the dermoscopic image of the skin lesion. What is the most likely diagnosis?"
Q_TEXT = "{vignette} What is the most likely diagnosis?"
Q_MM = "{vignette} Look at the dermoscopic image of the lesion. What is the most likely diagnosis?"


def options_for(case_id, seed=OPTION_SEED):
    """The options of one case: the eight classes in an order fixed by the seed and the case id (the same on every computer).
    Returns {letter: option text} and {class: letter}."""
    order = random.Random(f"{seed}:{case_id}").sample(CLASSES, len(CLASSES))
    return {LETTERS[i]: OPTION_TEXT[c] for i, c in enumerate(order)}, {c: LETTERS[i] for i, c in enumerate(order)}


def question(case, view):
    """The question record of one case in one view."""
    options, letter = options_for(case["isic_id"])
    q = {"id": case["isic_id"], "split": case["split"], "options": options, "answer": letter[case["label"]],
         "answer_text": OPTION_TEXT[case["label"]], "gold_label": case["label"], "group_id": case["group_id"]}
    if view == "image_only":
        return {**q, "question": Q_IMAGE, "image_path": case["image_path"]}
    if view == "text_only":
        return {**q, "question": Q_TEXT.format(vignette=case["clinical_text"])}
    return {**q, "question": Q_MM.format(vignette=case["clinical_text"]), "image_path": case["image_path"]}


def build_views(data):
    cases = read_csv(os.path.join(data, ALL_CASES))
    unknown = sorted({c["label"] for c in cases} - set(CLASSES))
    if unknown:
        raise ValueError(f"{ALL_CASES} holds classes that are not among the 8: {unknown}")
    counts = {}
    for view in VIEWS:
        write_csv(os.path.join(data, view_table(view)), cases, CORE + VIEW_INPUTS[view])
        for split in SPLITS:
            items = [question(c, view) for c in cases if c["split"] == split]
            write_jsonl(os.path.join(data, question_file(view, split)), items)
            counts[(view, split)] = len(items)
    return counts


def views_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main views", description="Builds the three views and their question files from the cleaned cases.")
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    log = get_logger("build_views")
    for (view, split), n in build_views(a.data).items():
        log.info(f"{view:11s} {split:5s} {n:5d} questions")
