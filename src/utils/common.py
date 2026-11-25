"""The constants shared by the whole project: the 8 classes, their option texts, the splits, the views and the folder layout
of data/ (relative to the data folder)."""
import os

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
OFFICIAL_LABELS = "annotations/official_labels"   # the official ISIC 2019, HAM10000 and ISIC Archive label and metadata files
PROCESSED_IMAGES = "processed/images"            # the cleaned, standardised images (laptop only)
PROCESSED_CASES = "processed/cases"              # the cleaned, de-duplicated and split cases, and the tables made with them
VIEWS_DIR = "views"                              # one folder per view: cases.csv and the question files
CORPUS_DIR = "corpus"            # the knowledge library for the RAG systems (text and images)
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


INDEX_DIR = f"{CORPUS_DIR}/index"           # the text index (embeddings.f32, index.json), laptop only
