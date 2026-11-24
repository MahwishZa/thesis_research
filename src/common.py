"""Shared definitions for every script in src/: the classes, the folder layout of data/, safe file writing and logging.
Only the Python standard library is used here (the encoders import torch and transformers only when they are called), so every
script and every test can import it."""
import contextlib
import csv
import hashlib
import heapq
import json
import logging
import math
import operator
import os
import re
import sys
import tempfile
from array import array
from collections import Counter, defaultdict
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


# ---------------------------------------------------------------------------------------------------------------------------
# Dense retrieval for the RAG systems (standard RAG, 2026-11-24). Passages are embedded once with the MedCPT article encoder, a
# question is embedded with the MedCPT query encoder (Jin et al. 2023, Bioinformatics 39:btad651), and the passages with the highest
# cosine similarity come first. The vectors have unit length, so the cosine is a dot product. Nothing here reads a label or a tag.
# ---------------------------------------------------------------------------------------------------------------------------
MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"
INDEX_DIR = f"{KNOWLEDGE_CORPUS}/index"
INDEX_FILES = ("embeddings.f32", "index.json")


def medcpt_encoder(model_name, max_length):
    """A function that turns a list of texts (a list of [title, text] pairs for the article encoder) into unit-length vectors, using the
    [CLS] embedding of the model. The model is downloaded on first use."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name).eval()

    @torch.no_grad()
    def encode(items, batch=16):
        out = []
        for start in range(0, len(items), batch):
            x = tok(items[start:start + batch], truncation=True, padding=True, return_tensors="pt", max_length=max_length)
            out += torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0], dim=-1).tolist()
        return out
    return encode


MEDCPT_CROSS = "ncbi/MedCPT-Cross-Encoder"


def medcpt_cross_encoder(model_name=MEDCPT_CROSS, max_length=512):
    """A function score(query, [[title, text], ...]) -> one relevance score per passage (higher = more relevant), from the MedCPT
    cross-encoder (Jin et al. 2023), which reads the query and a passage together. The model is downloaded on first use."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(model_name), AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    @torch.no_grad()
    def score(query, passages, batch=16):
        out = []
        for start in range(0, len(passages), batch):
            pairs = [[query, f"{t} {x}".strip()] for t, x in passages[start:start + batch]]
            enc = tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
            out += model(**enc).logits.squeeze(dim=-1).reshape(-1).tolist()
        return out
    return score


# ---- images: encoder and the check against the Kaggle images --------------------------------------------------------------
IMAGE_ENCODERS = ("biomedclip",)       # "monet" is added once its loading code is confirmed from its model card
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def image_encoder(kind):
    """A function that turns a list of image file paths into unit-length vectors. BiomedCLIP (Zhang et al. 2023) is loaded the way its
    model card does (open_clip, `create_model_from_pretrained`); it needs the `open_clip_torch` and `Pillow` packages and downloads
    the model on first use."""
    if kind != "biomedclip":
        raise ValueError(f"unknown image encoder {kind!r}; choose one of {IMAGE_ENCODERS}")
    import open_clip
    import torch
    from PIL import Image
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP)
    model.eval()

    @torch.no_grad()
    def encode(paths, batch=16):
        out = []
        for start in range(0, len(paths), batch):
            images = [preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch]]
            out += torch.nn.functional.normalize(model.encode_image(torch.stack(images)), dim=-1).tolist()
        return out
    return encode


def dhash_bits(pixels, size=8):
    """64 bits from a size x (size + 1) grid of grey values, row by row: one bit per pair of neighbours (left brighter than right)."""
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | int(pixels[r * (size + 1) + c] > pixels[r * (size + 1) + c + 1])
    return bits


def _dhash(img, size=8):
    """The difference hash of a PIL image: shrunk to 9 x 8 grey pixels."""
    from PIL import Image
    return dhash_bits(list(img.convert("L").resize((size + 1, size), Image.LANCZOS).getdata()), size)


def dhash_variants(path):
    """The hash of an image file and of its mirror images and half turn (a figure may be shown flipped): 4 integers."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        return [_dhash(v) for v in (im, im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM),
                                    im.transpose(Image.ROTATE_180))]


def hamming(a, b):
    return (a ^ b).bit_count()


def max_similarities(vectors, reference_vectors):
    """For each vector, its largest dot product (cosine, as all vectors have unit length) with any reference vector, and which one."""
    try:
        import numpy as np
        sims = np.asarray(vectors, dtype="float32") @ np.asarray(reference_vectors, dtype="float32").T
        best = sims.argmax(axis=1)
        return [(float(sims[i, j]), int(j)) for i, j in enumerate(best)]
    except ImportError:                                     # small inputs only (the tests); the laptop has numpy
        out = []
        for v in vectors:
            scores = [sum(map(operator.mul, v, r)) for r in reference_vectors]
            j = max(range(len(scores)), key=lambda i: (scores[i], -i))
            out.append((scores[j], j))
        return out


def drop_near_duplicates(vectors, threshold):
    """Greedy, in order: a vector is kept unless its cosine with an already kept vector is at least `threshold`.
    Returns (indices kept, [(index dropped, index of the kept look-alike, similarity), ...])."""
    kept, dropped = [], []
    try:
        import numpy as np
        matrix = np.asarray(vectors, dtype="float32")
        for i in range(len(matrix)):
            if kept:
                sims = matrix[kept] @ matrix[i]
                j = int(sims.argmax())
                if float(sims[j]) >= threshold:
                    dropped.append((i, kept[j], float(sims[j])))
                    continue
            kept.append(i)
    except ImportError:                                     # small inputs only (the tests); the laptop has numpy
        for i, v in enumerate(vectors):
            best = max(((sum(map(operator.mul, v, vectors[j])), j) for j in kept), default=(-2.0, -1))
            if best[0] >= threshold:
                dropped.append((i, best[1], best[0]))
            else:
                kept.append(i)
    return kept, dropped


# ---- keyword search and rank fusion (standard library only) ---------------------------------------------------------------
WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD.findall((text or "").lower())


class BM25Index:
    """Okapi BM25 (Robertson and Zaragoza 2009) over [title, text] with the usual k1 = 1.5 and b = 0.75; `search` gives
    [(passage, score), ...], best first, a tie broken by the order of the passages."""

    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages, self.k1, self.b = passages, k1, b
        tf = [Counter(words(f"{p.get('title') or ''} {p['text']}")) for p in passages]
        self.length = [sum(t.values()) for t in tf]
        self.avg = (sum(self.length) / len(tf)) if tf else 0.0
        df = Counter(w for t in tf for w in t)
        n = len(tf)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.postings = defaultdict(list)
        for i, t in enumerate(tf):
            for w, c in t.items():
                self.postings[w].append((i, c))

    def search(self, query, k):
        scores = defaultdict(float)
        for w in set(words(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, c in self.postings[w]:
                scores[i] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.length[i] / self.avg))
        best = heapq.nlargest(k, scores.items(), key=lambda t: (t[1], -t[0]))
        return [(self.passages[i], round(score, 6)) for i, score in best]


def reciprocal_rank_fusion(rankings, k=60, top=10):
    """Reciprocal rank fusion (Cormack et al. 2009): each passage scores the sum of 1 / (k + rank) over the lists it is in (k = 60 as
    in the paper). `rankings` are lists of passages, best first. Returns [(passage, score), ...]; a tie keeps the order first seen."""
    score, item = {}, {}
    for ranking in rankings:
        for rank, p in enumerate(ranking, 1):
            score[p["passage_id"]] = score.get(p["passage_id"], 0.0) + 1.0 / (k + rank)
            item.setdefault(p["passage_id"], p)
    order = sorted(score, key=lambda pid: -score[pid])
    return [(item[pid], round(score[pid], 6)) for pid in order[:top]]


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_index(folder, passages, vectors, meta):
    """Saves the vectors (float32, one row per passage, in the order of `passages`) and index.json (the passage ids and how the index
    was made). The vectors are derived from licensed text, so the folder stays on the laptop (see .gitignore)."""
    if len(vectors) != len(passages) or not passages:
        raise ValueError("an index needs one vector per passage, and at least one passage")
    dim = len(vectors[0])
    flat = array("f")
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors of an index must have the same length")
        flat.extend(v)
    os.makedirs(folder, exist_ok=True)
    with atomic_path(os.path.join(folder, "embeddings.f32")) as tmp, open(tmp, "wb") as f:
        flat.tofile(f)
    write_json(os.path.join(folder, "index.json"),
               {**meta, "dim": dim, "n_passages": len(passages), "passage_ids": [p["passage_id"] for p in passages]})


class DenseIndex:
    """The passages and their vectors; `search` gives the k best passages for one query vector."""

    def __init__(self, passages, flat, dim, meta=None):
        self.passages, self.flat, self.dim, self.meta = passages, flat, dim, meta or {}
        self.rows = memoryview(flat)

    def search(self, query_vector, k):
        """[(passage, score), ...], best first. A tie is broken by the order of the passages, so a search always gives the same list."""
        if len(query_vector) != self.dim:
            raise ValueError(f"the query vector has {len(query_vector)} numbers, the index {self.dim}")
        scores = (sum(map(operator.mul, query_vector, self.rows[i * self.dim:(i + 1) * self.dim])) for i in range(len(self.passages)))
        best = heapq.nlargest(k, enumerate(scores), key=lambda t: (t[1], -t[0]))
        return [(self.passages[i], round(score, 6)) for i, score in best]


def read_index(folder, passages_path):
    """Loads an index written by write_index with the passages it was made from; stops if the passage file has changed since."""
    meta_path = os.path.join(folder, "index.json")
    if not os.path.isfile(meta_path) or not os.path.isfile(os.path.join(folder, "embeddings.f32")):
        raise FileNotFoundError(f"no index in {folder}: run python src/build.py index")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("passages_sha256") != file_sha256(passages_path):
        raise ValueError(f"{passages_path} has changed since the index was built: run python src/build.py index again")
    by_id = {p["passage_id"]: p for p in read_jsonl(passages_path)}
    flat = array("f")
    with open(os.path.join(folder, "embeddings.f32"), "rb") as f:
        flat.fromfile(f, meta["n_passages"] * meta["dim"])
    return DenseIndex([by_id[i] for i in meta["passage_ids"]], flat, meta["dim"], meta)
