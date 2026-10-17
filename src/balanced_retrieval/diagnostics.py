"""What the final evidence lists look like (docs/methodology.md section 7.4): corpus shares and cross-corpus duplicates.

These numbers describe the lists; they are not tests of correctness. A candidate is a mapping with ``corpus`` and
``doc_key``; an empty or missing ``doc_key`` means the identity of the document is unknown.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .pool import CORPUS_ORDER, check_corpus_names


def _checked(final: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    items = list(final)
    if not items:
        raise ValueError("the evidence list is empty")
    check_corpus_names([item.get("corpus") for item in items])
    return items


def corpus_share(final: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The share of the list that comes from each corpus; every corpus of ``CORPUS_ORDER`` has a key, and the shares sum to 1."""
    items = _checked(final)
    counts = Counter(item["corpus"] for item in items)
    return {name: counts[name] / len(items) for name in CORPUS_ORDER}


def _known(key: object) -> bool:
    return key is not None and key != ""


def duplicate_rate(final: Sequence[Mapping[str, Any]]) -> float:
    """The share of snippets whose document also appears, in any snippet, in another corpus of the same list.

    Chunks of a shared document each count. The same document twice inside one corpus is not a cross-corpus duplicate, and
    a snippet of unknown identity never matches anything.
    """
    items = _checked(final)
    corpora_of: dict[object, set[str]] = {}
    for item in items:
        key = item.get("doc_key")
        if _known(key):
            corpora_of.setdefault(key, set()).add(item["corpus"])
    shared = sum(1 for item in items if _known(item.get("doc_key")) and len(corpora_of[item["doc_key"]]) > 1)
    return shared / len(items)


def summarise_lists(lists: Sequence[Sequence[Mapping[str, Any]]]) -> dict:
    """Averages over many questions: the mean share of each corpus, the mean duplicate rate, and the unknown identities."""
    usable = [list(final) for final in lists if final]
    summary: dict[str, Any] = {"questions": len(lists), "empty_lists": len(lists) - len(usable)}
    if not usable:
        return summary
    shares = [corpus_share(final) for final in usable]
    summary["mean_list_length"] = round(sum(len(final) for final in usable) / len(usable), 3)
    summary["mean_corpus_share"] = {name: round(sum(s[name] for s in shares) / len(shares), 4) for name in CORPUS_ORDER}
    summary["mean_duplicate_rate"] = round(sum(duplicate_rate(final) for final in usable) / len(usable), 4)
    everything = [item for final in usable for item in final]
    summary["share_without_document_key"] = round(sum(1 for item in everything if not _known(item.get("doc_key"))) / len(everything), 4)
    return summary
