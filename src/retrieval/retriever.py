"""Keyword search (BM25), reciprocal rank fusion, the removal of repeated texts, and the retrieval query of each view (only the
question's own words)."""
import heapq
import math
import os
import re
from collections import Counter, defaultdict

from src.retrieval.corpus import OVERLAP_LIMIT, overlap_share, shingles

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


def distinct(hits):
    """The hits without a record whose text repeats a better-ranked one (the same text, or half or more of its 8-word stretches):
    a figure caption can be both a corpus passage and a picture-and-caption record, and the model must not get it twice."""
    out, seen, seen_sh = [], set(), set()
    for p, score in hits:
        text = " ".join(p["text"].lower().split())
        if text in seen or overlap_share(p["text"], seen_sh) >= OVERLAP_LIMIT:
            continue
        out.append((p, score))
        seen.add(text)
        seen_sh |= shingles(p["text"])
    return out



GENERIC_QUERY = "dermoscopic image of a skin lesion"      # the query of an image-only question: it has no patient facts
GENERIC_TEXT_QUERY = "skin lesion"                      # the query of a text or image + text question with no recorded fact


def patient_facts(q):
    """The patient sentence of a question without the generic question and the words about the image: "" for an image-only question."""
    text = q["question"]
    for phrase in ("What is the most likely diagnosis?", "Look at the dermoscopic image of the skin lesion.",
                   "Look at the dermoscopic image of the lesion."):
        text = text.replace(phrase, "")
    return " ".join(text.split())


def recorded_facts(text):
    """The patient sentence without what is "not recorded": "Patient: age not recorded, female. Skin lesion on the trunk." gives
    "Patient: female. Skin lesion on the trunk."; "" when nothing is recorded."""
    text = re.sub(r"\s*Lesion site not recorded\.?", "", text)            # an unknown site
    text = re.sub(r"\bage not recorded,?\s*", "", text)                   # an unknown age
    text = re.sub(r",?\s*\bsex not recorded", "", text)                   # an unknown sex
    text = " ".join(text.split())
    text = re.sub(r"^Patient:\s*\.\s*", "", text)                        # nothing left of the patient sentence: "Patient: ."
    if text == "Patient:":
        return ""
    return text if text.endswith((".", ")")) or not text else text + "."


def retrieval_query(q, view="text_only"):
    """The text the retriever gets: the recorded patient facts of the question. When there are none, a fixed sentence that fits the
    view (the image-only view has no facts at all, and a text view has no image). Never the options, the true class or a tag."""
    facts = recorded_facts(patient_facts(q))
    if facts:
        return facts
    return GENERIC_QUERY if view == "image_only" else GENERIC_TEXT_QUERY


def image_query_vector(q, vectors):
    name = os.path.basename(q["image_path"])
    if name not in vectors:
        raise KeyError(f"no saved image vector for {name}: run python -m src.main visual-index again")
    return vectors[name]
