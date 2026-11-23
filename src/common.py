"""Shared definitions for every script in src/: the classes, the folder layout of data/, safe file writing and logging.
Only the Python standard library is used here, so every script (and every test) can import it."""
import contextlib
import csv
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SEED = 42
TEST_PER_CLASS = 30                   # images of each class in the balanced test split
VAL_PER_CLASS = 20                    # images of each class in the balanced validation split

# The 8 ISIC 2019 classes, in the order of their index (label_idx).
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
# The class name in the tables (label_name, label_map.json).
CLASS_NAMES = {
    "MEL": "melanoma", "NV": "melanocytic nevus", "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis", "BKL": "benign keratosis-like lesion (pigmented benign keratosis)",
    "DF": "dermatofibroma", "VASC": "vascular lesion", "SCC": "squamous cell carcinoma",
}
# The answer option shown in a question.
OPTION_TEXT = {
    "MEL": "Melanoma", "NV": "Melanocytic nevus", "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis", "BKL": "Benign keratosis-like lesion (seborrheic keratosis / solar lentigo / lichen planus-like keratosis)",
    "DF": "Dermatofibroma", "VASC": "Vascular lesion", "SCC": "Squamous cell carcinoma",
}
LETTERS = "ABCDEFGH"
SPLITS = ("train", "val", "test")
VIEWS = ("image_only", "text_only", "multimodal")

# Layout of data/ (all paths relative to the data folder; image paths in the tables are relative to it too).
RAW = "raw"                           # the Kaggle download, untouched (laptop only)
OFFICIAL_LABELS = "official_labels"   # the official ISIC 2019, HAM10000 and ISIC Archive label and metadata files
PROCESSED_IMAGES = "processed_images" # the cleaned, standardised images (laptop only)
PROCESSED_CASES = "processed_cases"   # the cleaned, de-duplicated and split cases, and the tables made with them
VIEWS_DIR = "views"                   # one folder per view: cases.csv and the question files
KNOWLEDGE_CORPUS = "knowledge_corpus" # the medical text library for RAG
ALL_CASES = f"{PROCESSED_CASES}/all_cases.csv"


def image_path(isic_id):
    """Where a cleaned image is, relative to the data folder, with "/" on every system (never "\\" from Windows)."""
    return f"{PROCESSED_IMAGES}/{isic_id}.jpg"


def images_not_in_table(data, isic_ids):
    """The .jpg files in data/processed_images/ whose name is not a case in the table (left over from an earlier run), sorted."""
    folder = os.path.join(data, PROCESSED_IMAGES)
    if not os.path.isdir(folder):
        return []
    keep = {f"{i}.jpg" for i in isic_ids}
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg") and f not in keep)


def view_table(view):
    return f"{VIEWS_DIR}/{view}/cases.csv"


def question_file(view, split):
    return f"{VIEWS_DIR}/{view}/{split}.jsonl"


@contextlib.contextmanager
def atomic_path(path):
    """Yields a temporary path next to `path`; when the block ends without an error it replaces `path` in one step.
    An interrupted run therefore never leaves a half-written file behind (the old file stays as it was)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_env(path=".env"):
    """Puts the NAME=value lines of a local .env file into the environment. A variable already set in the terminal keeps its value
    (one that is set but empty counts as not set). Blank lines and # comments are skipped; quotes around a value are removed; a
    missing file is fine. Returns the line numbers (never the lines: they may hold a secret) of the lines that could not be used:
    no "=", no name, or no value."""
    unusable = []
    if not os.path.isfile(path):
        return unusable
    with open(path, encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, value = (part.strip() for part in line.partition("="))
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not (name and value):
                unusable.append(number)
            elif not os.environ.get(name, "").strip():
                os.environ[name] = value
    return unusable


def write_text(path, text):
    with atomic_path(path) as tmp, open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(path, obj):
    """The same format as json.dump(obj, f, indent=2), which made the committed .json files (no newline at the end)."""
    write_text(path, json.dumps(obj, indent=2))


def write_jsonl(path, records):
    write_text(path, "".join(json.dumps(r) + "\n" for r in records))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    """Writes rows (dicts) with the given columns, LF line endings and minimal quoting (the same format pandas writes)."""
    with atomic_path(path) as tmp, open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        w.writerows([r[c] for c in columns] for r in rows)


class _Screen(logging.StreamHandler):
    """Writes to the screen as it is when a message is logged (so a test or a caller can redirect it)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def get_logger(name, log_file=None):
    """A logger that prints to the screen and, when log_file is given, also appends to that file."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        screen = _Screen()
        screen.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(screen)
        log.propagate = False
    if log_file and not any(getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in log.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file, encoding="utf-8")
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(file)
    return log


# ---- the balanced test and validation splits (used by preprocess.py step 8, checked by check.py) -----------------------------
# The balanced splits (test decided 2026-11-22: 30 per class; validation decided 2026-11-22: 20 per class): the split holds exactly
# the same number of images of every class, so every class weighs the same in its accuracy. Used by src/preprocess.py (step 8)
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
