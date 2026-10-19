"""The textbooks corpus: the 18 medical textbooks of MedQA, as published in the ``MedRAG/textbooks`` dataset.

That dataset holds each textbook already cut into pieces of at most 1,000 characters, one JSON-lines file per book under
``chunk/``. The paper's corpus is cut differently (128 words, overlap 32), so ``join_book`` puts the pieces of one book back
into one text, dropping the characters that two neighbouring pieces share if the dataset's cutting overlapped them, and the
book then goes through the same ``chunk`` command as every other corpus. Each book is one document, keyed ``BOOK:<file name>``.
The textbooks are copyrighted books; their reuse terms were not checked.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rationale_query.store import StoreError, write_atomic

REPO = "MedRAG/textbooks"
MIN_SHARED = 20  # a shorter coincidence between the end of one piece and the start of the next is not an overlap
MAX_SHARED = 600


def join_book(pieces: Iterable[str]) -> str:
    """One text from the consecutive pieces of a book; a repeated stretch at a seam is kept once."""
    text = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not text:
            text = piece
            continue
        for size in range(min(len(text), len(piece), MAX_SHARED), MIN_SHARED - 1, -1):
            if text.endswith(piece[:size]):
                text += piece[size:]  # the rest continues the text exactly where the shared stretch ended
                break
        else:
            text = f"{text} {piece}"
    return text


def book_documents(files: dict[str, Iterable[str]]) -> list[dict[str, Any]]:
    """A document per textbook from ``{file name: JSON lines}``; each line has ``content`` (or ``contents``)."""
    documents = []
    for name in sorted(files):
        pieces = []
        for number, line in enumerate(files[name], start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StoreError(f"{name}, line {number} is not valid JSON: {exc}") from exc
            piece = item.get("content")
            pieces.append(piece if isinstance(piece, str) else str(item.get("contents") or ""))
        text = join_book(pieces)
        if text:
            book = name.rsplit("/", 1)[-1].removesuffix(".jsonl")
            documents.append({"doc_key": f"BOOK:{book}", "text": text, "book": book, "pieces": len(pieces)})
    return documents


def download_textbooks(out: Path, *, revision: str | None = None, hub: Any = None) -> dict[str, Any]:
    """Write the textbooks as documents into ``out``; returns what was used. ``hub`` is the ``huggingface_hub`` module."""
    if hub is None:
        import huggingface_hub

        hub = huggingface_hub
    names = sorted(n for n in hub.HfApi().list_repo_files(REPO, repo_type="dataset", revision=revision) if n.startswith("chunk/") and n.endswith(".jsonl"))
    if not names:
        raise StoreError(f"{REPO} holds no chunk/*.jsonl files")
    files = {}
    for name in names:
        path = hub.hf_hub_download(REPO, name, repo_type="dataset", revision=revision)
        with open(path, encoding="utf-8") as handle:
            files[name] = handle.readlines()
    documents = book_documents(files)
    write_atomic(out, "".join(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n" for d in documents))
    info = {"source": REPO, "revision": hub.HfApi().dataset_info(REPO, revision=revision).sha, "documents": len(documents)}
    write_atomic(out.with_name(out.name + ".source.json"), json.dumps(info, indent=2) + "\n")
    return info
