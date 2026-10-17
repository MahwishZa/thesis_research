"""One corpus on disk: its passages, a row-aligned embedding file, and a manifest of how they were made.

A corpus folder ``<root>/<name>/`` holds

    passages.jsonl          one JSON object per line: corpus, doc_key, chunk_index, text
    passages.offsets.npy    the byte position of each line, so one passage is read without loading the file
    embeddings.npy          one MedCPT article vector per passage, in the same row order (written by ``embedding``)
    embeddings.json         how far the embeddings are and with which encoder settings (written by ``embedding``)
    manifest.json           the chunking settings, the counts and the SHA-256 of passages.jsonl

Row ``i`` of the embeddings belongs to line ``i`` of the passages, which is how the authors' retriever decodes a hit.
Nothing here reads a folder other than the one it is given.
"""

from __future__ import annotations

import hashlib
import json
import os
from array import array
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rationale_query.store import StoreError, write_atomic

from .chunking import DEFAULT_OVERLAP, DEFAULT_WORDS, check_window, passages_of
from .pool import CORPUS_ORDER

PASSAGES_FILE = "passages.jsonl"
OFFSETS_FILE = "passages.offsets.npy"
EMBEDDINGS_FILE = "embeddings.npy"
EMBEDDINGS_STATE = "embeddings.json"
MANIFEST_FILE = "manifest.json"


class CorpusError(StoreError):
    """A corpus folder is missing, damaged, or does not match what is asked of it."""


def read_documents(path: str | Path) -> Iterator[tuple[str, str]]:
    """Documents from a JSON-lines file: one object per line with ``text`` and, if known, ``doc_key``.

    ``doc_key`` is the one canonical identity of the document (a PMID when known, else a PMCID, a DOI or a normalised
    title); it is what lets the duplicate rate find the same paper in two corpora. A missing key means unknown.
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{path}, line {number} is not valid JSON: {exc}") from exc
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    raise CorpusError(f"{path}, line {number}: every document needs a text")
                key = item.get("doc_key", "")
                if key is None:
                    key = ""
                if isinstance(key, bool) or not isinstance(key, str | int):
                    raise CorpusError(f"{path}, line {number}: doc_key must be text or a whole number")
                yield str(key), item["text"]
    except OSError as exc:
        raise CorpusError(f"cannot read {path}: {exc}") from exc


def write_corpus(
    root: str | Path,
    corpus: str,
    documents: Iterable[tuple[str, str]],
    *,
    words: int = DEFAULT_WORDS,
    overlap: int = DEFAULT_OVERLAP,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Cut the documents into passages and write the corpus folder; returns the manifest.

    A folder that already holds a corpus is refused unless ``overwrite`` is set, and then the embeddings of the old corpus
    are removed too, because they would no longer belong to the passages. Documents without any word are skipped and counted.
    """
    if corpus not in CORPUS_ORDER:
        raise ValueError(f"unknown corpus {corpus!r}; the corpora are {', '.join(CORPUS_ORDER)}")
    check_window(words, overlap)
    folder = Path(root) / corpus
    if (folder / MANIFEST_FILE).exists() and not overwrite:
        raise CorpusError(f"{folder} already holds a corpus; add --overwrite to replace it (its embeddings go too)")
    folder.mkdir(parents=True, exist_ok=True)

    passages_path = folder / PASSAGES_FILE
    temp = passages_path.with_name(passages_path.name + ".tmp")
    offsets = array("Q")
    digest = hashlib.sha256()
    documents_count = without_key = empty = 0
    position = 0
    with open(temp, "wb") as handle:
        for doc_key, text in documents:
            if not isinstance(text, str):
                raise TypeError("a document's text must be a string")
            documents_count += 1
            without_key += doc_key == ""
            made = 0
            for record in passages_of(corpus, doc_key, text, size=words, overlap=overlap):
                line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                offsets.append(position)
                handle.write(line)
                digest.update(line)
                position += len(line)
                made += 1
            empty += made == 0
    if not offsets:
        temp.unlink(missing_ok=True)
        raise CorpusError("the documents hold no words, so there are no passages to write")

    for stale in (EMBEDDINGS_FILE, EMBEDDINGS_STATE):
        (folder / stale).unlink(missing_ok=True)
    os.replace(temp, passages_path)
    buffer = np.frombuffer(offsets, dtype=np.uint64)
    np.save(folder / OFFSETS_FILE, buffer)
    manifest = {
        "corpus": corpus,
        "chunking": {"unit": "words", "size": words, "overlap": overlap},
        "documents": documents_count,
        "documents_without_key": without_key,
        "documents_without_words": empty,
        "passages": len(offsets),
        "passages_sha256": digest.hexdigest(),
    }
    write_atomic(folder / MANIFEST_FILE, json.dumps(manifest, indent=2) + "\n")
    return manifest


class Corpus:
    """A corpus folder opened for reading. ``passage(row)`` and ``embeddings()`` never load the whole corpus."""

    def __init__(self, root: str | Path, name: str) -> None:
        if name not in CORPUS_ORDER:
            raise ValueError(f"unknown corpus {name!r}; the corpora are {', '.join(CORPUS_ORDER)}")
        self.name = name
        self.directory = Path(root) / name
        try:
            self.manifest: dict[str, Any] = json.loads((self.directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusError(f"{self.directory} holds no usable corpus (no readable {MANIFEST_FILE}): {exc}") from exc
        try:
            self._offsets = np.load(self.directory / OFFSETS_FILE, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise CorpusError(f"cannot read {self.directory / OFFSETS_FILE}: {exc}") from exc
        if len(self._offsets) != self.manifest["passages"]:
            raise CorpusError(f"{self.directory}: {len(self._offsets)} offsets but the manifest says {self.manifest['passages']} passages")

    def __len__(self) -> int:
        return int(self.manifest["passages"])

    @property
    def passages_path(self) -> Path:
        return self.directory / PASSAGES_FILE

    def _check_rows(self, rows: Sequence[int]) -> None:
        for row in rows:
            if isinstance(row, bool) or not isinstance(row, int | np.integer) or not 0 <= row < len(self):
                raise IndexError(f"row {row!r} is not a passage of {self.name} (0 to {len(self) - 1})")

    def passages(self, rows: Sequence[int]) -> list[dict[str, Any]]:
        """The passages at these rows, in the order asked."""
        rows = [int(row) if isinstance(row, np.integer) else row for row in rows]
        self._check_rows(rows)
        found = []
        with open(self.passages_path, "rb") as handle:
            for row in rows:
                handle.seek(int(self._offsets[row]))
                found.append(json.loads(handle.readline().decode("utf-8")))
        return found

    def passage(self, row: int) -> dict[str, Any]:
        return self.passages([row])[0]

    def iter_passages(self, start: int = 0) -> Iterator[dict[str, Any]]:
        """Every passage from row ``start`` on, read in one pass."""
        if not 0 <= start <= len(self):
            raise IndexError(f"start {start} is outside 0 to {len(self)}")
        if start == len(self):
            return
        with open(self.passages_path, "rb") as handle:
            handle.seek(int(self._offsets[start]))
            for line in handle:
                yield json.loads(line.decode("utf-8"))

    def embedding_state(self) -> dict[str, Any] | None:
        path = self.directory / EMBEDDINGS_STATE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusError(f"cannot read {path}: {exc}") from exc

    def embeddings(self) -> np.ndarray:
        """The embedding matrix as a read-only memory map. The corpus must be fully embedded."""
        state = self.embedding_state()
        if state is None or not state.get("complete"):
            done = 0 if state is None else state.get("rows_done", 0)
            raise CorpusError(f"{self.name} is not embedded yet ({done} of {len(self)} rows); run the embed command")
        matrix = np.load(self.directory / EMBEDDINGS_FILE, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != len(self):
            raise CorpusError(f"{self.name}: the embeddings have shape {matrix.shape} for {len(self)} passages")
        return matrix
