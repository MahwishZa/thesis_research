"""Cutting a document into passages of 128 words that overlap by 32 (the Self-BioRAG corpus the paper reuses).

The paper says "a sliding window with overlap" and the Self-BioRAG README gives the size, "128 words with 32 words
overlap". How words are split and what happens to the last window are not stated; here a word is a run of characters
between white space, the window moves by size minus overlap, and the last window is the one that reaches the end of the
text (no extra window lies wholly inside the previous one). Line breaks and runs of spaces become single spaces.
"""

from __future__ import annotations

from collections.abc import Iterator

DEFAULT_WORDS = 128
DEFAULT_OVERLAP = 32


def check_window(size: int, overlap: int) -> None:
    for name, value in (("size", size), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a whole number")
    if size < 1:
        raise ValueError("size must be at least 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")


def chunk_words(text: str, size: int = DEFAULT_WORDS, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """The passages of one text, in order; a text without words gives none, a short text gives one."""
    check_window(size, overlap)
    words = text.split()
    if not words:
        return []
    step = size - overlap
    passages = []
    start = 0
    while True:
        passages.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            return passages
        start += step


def passages_of(
    corpus: str, doc_key: str, text: str, *, size: int = DEFAULT_WORDS, overlap: int = DEFAULT_OVERLAP
) -> Iterator[dict]:
    """One record per passage: the corpus, the document's identity key ("" when unknown), the chunk number and the text."""
    for number, passage in enumerate(chunk_words(text, size, overlap)):
        yield {"corpus": corpus, "doc_key": doc_key, "chunk_index": number, "text": passage}
