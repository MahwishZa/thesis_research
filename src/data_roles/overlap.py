"""Questions of one split that repeat questions of another: exact repeats and near repeats.

The MedMCQA authors removed similar questions between their splits; this check shows how well that worked for the splits
used here (docs/methodology.md, 7.7, rule 2), for example the validation questions against the training questions.
It compares stems only. A near repeat shares most of its five-word shingles with the question it repeats (Jaccard similarity
at or above the threshold). A shingle found in more than ``max_postings`` reference questions ("which of the following is")
is ignored, so that common phrases do not make the search slow; a pair that agrees only in such phrases is not reported.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .questions import Item

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Overlap:
    candidate: str
    reference: str
    similarity: float
    exact: bool


def words(text: str) -> list[str]:
    """Lower-case letters and digits only, so that punctuation and spacing do not hide a repeat."""
    return _WORD.findall(text.lower())


def shingles(text: str, size: int = 5) -> set[str]:
    """The overlapping runs of ``size`` words; a text shorter than that is one shingle."""
    parts = words(text)
    if len(parts) <= size:
        return {" ".join(parts)} if parts else set()
    return {" ".join(parts[i : i + size]) for i in range(len(parts) - size + 1)}


def find_overlaps(
    reference: Sequence[Item], candidates: Sequence[Item], *, size: int = 5, threshold: float = 0.8, max_postings: int = 2000
) -> list[Overlap]:
    """Every candidate question that repeats a reference question, best match first for each candidate."""
    if not 0 < threshold <= 1:
        raise ValueError(f"the threshold must be above 0 and at most 1, got {threshold!r}")
    reference_shingles = [shingles(item.stem, size) for item in reference]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, grams in enumerate(reference_shingles):
        for gram in grams:
            postings[gram].append(index)
    exact = defaultdict(list)
    for index, item in enumerate(reference):
        exact[" ".join(words(item.stem))].append(index)
    found: list[Overlap] = []
    for item in candidates:
        grams = shingles(item.stem, size)
        same = exact.get(" ".join(words(item.stem)), [])
        shared: Counter[int] = Counter()
        for gram in grams:
            hits = postings.get(gram, ())
            if len(hits) <= max_postings:
                shared.update(hits)
        scored = {}
        for index, count in shared.items():
            union = len(grams) + len(reference_shingles[index]) - count
            if union and count / union >= threshold:
                scored[index] = count / union
        for index in same:
            scored[index] = 1.0
        for index, similarity in sorted(scored.items(), key=lambda pair: (-pair[1], reference[pair[0]].id)):
            found.append(Overlap(item.id, reference[index].id, round(similarity, 4), index in same))
    return found
