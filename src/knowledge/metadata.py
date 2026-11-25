"""Step 4 of the knowledge corpus: the tags, topics and card of the passages (tags are for evaluation only, never retrieval)."""
import re
from collections import Counter

from src.utils.common import CLASSES

LIBRARIES = ("guidelines", "pmc", "pubmed", "textbook")     # guidelines first, so a guideline paper is not counted again as pmc
SYNONYMS = {
    "MEL": ["melanoma", "lentigo maligna", "melanoma in situ", "cutaneous melanoma"],
    "NV": ["nevus", "naevus", "nevi", "naevi", "melanocytic nevus"],
    "BCC": ["basal cell carcinoma", "basal-cell carcinoma", "basalioma", "basal cell epithelioma"],
    "AK": ["actinic keratosis", "actinic keratoses", "solar keratosis", "solar keratoses"],
    "BKL": ["seborrheic keratosis", "seborrhoeic keratosis", "seborrheic keratoses", "solar lentigo", "solar lentigines",
            "lichen planus-like keratosis", "benign keratosis"],
    "DF": ["dermatofibroma", "dermatofibromas", "benign fibrous histiocytoma"],
    "VASC": ["hemangioma", "haemangioma", "cherry angioma", "angioma", "angiokeratoma", "pyogenic granuloma", "vascular lesion",
             "lobular capillary hemangioma"],
    "SCC": ["squamous cell carcinoma", "squamous-cell carcinoma", "keratoacanthoma", "bowen's disease", "bowen disease"],
}
# Topic coverage check (keyword-based, approximate): does the corpus speak about each diagnosis in each of these four ways?
TOPICS = {
    "appearance": r"dermoscop|dermatoscop|pigment network|globule|streak|blue-white|veil|vessel|vascular pattern|"
                  r"regression|milia|lacun|white patch|rosette|ulceration|border|asymmetr|colou?r",
    "patients": r"\bage[ds]?\b|elderly|older|young|incidence|prevalen|\bmen\b|\bwomen\b|male|female|sun|ultraviolet|uv\b|"
                r"risk factor|located|location|trunk|face|head|neck|extremit|\blegs?\b|\barms?\b",
    "differential": r"differential|distinguish|differentiat|mimic|simulat|confused with|versus|\bvs\.?\b|resembl",
    "basics": r"benign|malignan|carcinoma|tumou?r|neoplasm|lesion|prognos|histolog|patholog|definition|defined as",
}


def _synonym_pattern(words):
    """Whole words, with an optional plural ending: 'melanoma' also finds 'melanomas', 'carcinoma' finds 'carcinomas' (checked
    2026-11-24: the exact-word rule left such passages untagged)."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es)?\b")


SYNONYM_PATTERNS = {c: _synonym_pattern(words) for c, words in SYNONYMS.items()}


def tag(text, title=""):
    t = f"{title} {text}".lower()
    return sorted(c for c, rx in SYNONYM_PATTERNS.items() if rx.search(t))


def topics(text):
    t = text.lower()
    return sorted(k for k, rx in TOPICS.items() if re.search(rx, t))


def corpus_card(passages, removed, args, snapshot):
    count = Counter((p["library"], p["query_class"]) for p in passages)
    coverage = {c: {t: sum(c in p["tags"] and t in p["topics"] for p in passages) for t in TOPICS} for c in CLASSES}
    return {"snapshot_date": snapshot, "n_passages": len(passages), "n_documents": len({p["doc_id"] for p in passages}),
            "passages_per_library_and_class": {lib: {c: count[(lib, c)] for c in CLASSES} for lib in LIBRARIES},
            "passages_tagged_per_class": {c: sum(c in p["tags"] for p in passages) for c in CLASSES},
            "topic_coverage_per_class": coverage,
            "empty_topic_boxes": [f"{c}/{t}" for c, row in coverage.items() for t, n in row.items() if n == 0],
            "licences": dict(Counter(p["licence"] for p in passages).most_common()),
            "removed": dict(sorted(removed.items())),
            "settings": {k: v for k, v in vars(args).items() if k not in ("out", "cache", "env_file")}}


MANIFEST_COLUMNS = ["passage_id", "library", "doc_id", "url", "licence", "query_class", "tags", "n_words", "md5"]
