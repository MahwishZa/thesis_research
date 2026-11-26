"""The shared helpers of the whole project: the constants and the layout of data/, safe file reading and writing, loggers, and the
fixed seed and file fingerprints."""
import contextlib
import csv
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path


# =============================================================================================================================
# Constants: the 8 classes, their option texts, the splits, the views and the folder layout of data/ (relative to the data folder)
# =============================================================================================================================

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
# The file name of each split's question file (the split itself is still called "val" in the tables and the records).
SPLIT_FILES = {"train": "train", "val": "validation", "test": "test"}
VIEWS = ("image_only", "text_only", "multimodal")

# Layout of data/ (all paths relative to the data folder; image paths in the tables are relative to it too).
RAW = "raw"                           # the Kaggle download, untouched (laptop only)
OFFICIAL_LABELS = "annotations"            # the official ISIC 2019, HAM10000 and ISIC Archive label and metadata files
PROCESSED_IMAGES = "processed/images"            # the cleaned, standardised images (laptop only)
PROCESSED_CASES = "processed/cases"              # the cleaned, de-duplicated and split cases, and the tables made with them
VIEWS_DIR = "views"                              # one folder per view: cases.csv and the question files
CORPUS_DIR = "corpus"            # the knowledge library of the RAG systems: one folder per view (text_only, image_only, multimodal)
REFERENCE_TEXTS = "annotations/reference_texts.json"  # the reference texts of the evaluation (never retrieved)
ALL_CASES = f"{PROCESSED_CASES}/all_cases.csv"


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


TEXT_LIBRARY = f"{CORPUS_DIR}/text_only"         # what the text-only view searches: passages, manifest, cards, download cache, text index
IMAGE_LIBRARY = f"{CORPUS_DIR}/image_only"       # what the image-only view searches: picture-and-caption records, images, image index
MULTIMODAL_LIBRARY = f"{CORPUS_DIR}/multimodal"  # no records of its own: library.json names the two libraries it combines
CORPUS_FOLDERS = {"text_only": TEXT_LIBRARY, "image_only": IMAGE_LIBRARY, "multimodal": MULTIMODAL_LIBRARY}     # the library of each view
INDEX_DIR = f"{TEXT_LIBRARY}/index"              # the text index (embeddings.f32, index.json), laptop only


def hamming(a, b):
    """The number of bits in which two picture hashes differ."""
    return (a ^ b).bit_count()


# =============================================================================================================================
# Files: safe reading and writing (JSON, JSON Lines, CSV); a half-written file never replaces a good one
# =============================================================================================================================


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


# =============================================================================================================================
# Logging: to the screen and, when asked, also to a log file
# =============================================================================================================================


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


class _LogFile(logging.Handler):
    """Appends each message to a log file and closes the file again at once, so no file is held open: on Windows an open file
    cannot be deleted or have its folder renamed. A message for a folder that no longer exists is skipped (never re-creates it)."""

    def __init__(self, path):
        super().__init__()
        self.baseFilename = os.path.abspath(path)

    def emit(self, record):
        if not os.path.isdir(os.path.dirname(self.baseFilename)):
            return
        try:
            with open(self.baseFilename, "a", encoding="utf-8", newline="\n") as f:
                f.write(self.format(record) + "\n")
        except Exception:      # a log line must never stop a run  # noqa: BLE001
            self.handleError(record)


def get_logger(name, log_file=None):
    """A logger that prints to the screen and, when log_file is given, also appends to that file (one file per logger: a new
    log_file replaces the earlier one, so a command never writes into the log of an earlier command)."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        screen = _Screen()
        screen.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(screen)
        log.propagate = False
    if log_file and not any(getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in log.handlers):
        for h in [h for h in log.handlers if getattr(h, "baseFilename", None)]:
            log.removeHandler(h)
            h.close()
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file = _LogFile(log_file)
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(file)
    return log


# =============================================================================================================================
# Reproducibility: the fixed seed and file fingerprints (SHA-256), so a run can be repeated and its inputs checked
# =============================================================================================================================

SEED = 42


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# The fingerprint of a text file (questions, predictions) is taken with LF line endings, so the same file has the same fingerprint
# on Windows (CRLF when an older version of the code wrote it) and in Git (always LF, .gitattributes). Before 2026-11-25 the reports
# recorded the fingerprint of the file as Windows saved it (CRLF); text_matches accepts that form too, so an approved report stays valid.
def _lf_bytes(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def text_sha256(path):
    """SHA-256 of a text file with its line endings written as LF."""
    return hashlib.sha256(_lf_bytes(path)).hexdigest()


def text_matches(path, recorded):
    """Does a recorded fingerprint belong to this text file, with LF (now) or CRLF (Windows, reports before 2026-11-25) endings?"""
    lf = _lf_bytes(path)
    return recorded in (hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())
