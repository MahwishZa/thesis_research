"""Shared definitions for every script in src/: the classes, the folder layout of data/, safe file writing and logging.
Only the Python standard library is used here, so every script (and every test) can import it."""
import contextlib
import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

SEED = 42

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


def get_logger(name, log_file=None):
    """A logger that prints to the screen and, when log_file is given, also appends to that file."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        screen = logging.StreamHandler(sys.stdout)
        screen.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(screen)
        log.propagate = False
    if log_file and not any(getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in log.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file, encoding="utf-8")
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(file)
    return log
