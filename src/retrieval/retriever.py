"""Keyword search (BM25) and reciprocal rank fusion."""
import heapq
import math
import re
from collections import Counter, defaultdict

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
