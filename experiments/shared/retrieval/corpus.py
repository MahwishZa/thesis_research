"""Read the Alzheimer's corpus chunk file into passages, without touching it.

``alzheimer_corpus/`` is built and owned by its own pipeline. This module only
reads ``data/chunks/chunks.jsonl`` and maps each record onto the ``Evidence``
shape the arms already consume. It never writes, moves or reprocesses anything
under that directory.

The corpus snapshot id is computed here rather than supplied by hand. Frozen
evidence records which corpus build it came from, and a human-typed version
string is exactly the kind of thing that silently stops matching the data it
names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from src.common.evidence import Evidence

#: Path of the chunk file relative to the corpus root.
CHUNKS = Path("data") / "chunks" / "chunks.jsonl"


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be read as expected."""


def parse_publication_date(value: Optional[str]) -> Optional[date]:
    """Parse the corpus's ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` dates.

    Missing parts default to 1, which makes an age comparison possible without
    inventing precision: a passage dated ``2024`` is treated as 2024-01-01 and
    the imprecision is carried in the original string, which is what gets
    frozen.
    """
    if not value:
        return None
    parts = value.strip().split("-")
    try:
        numbers = [int(p) for p in parts if p != ""]
    except ValueError:
        raise CorpusError(f"unparseable publication_date: {value!r}")
    if not numbers:
        return None
    while len(numbers) < 3:
        numbers.append(1)
    try:
        return date(*numbers[:3])
    except ValueError as exc:
        raise CorpusError(f"invalid publication_date {value!r}: {exc}")


@dataclass(frozen=True)
class CorpusPassage:
    """One corpus chunk, ready for embedding and retrieval."""

    chunk_id: str
    document_id: str
    text: str
    #: What gets embedded. The corpus builds this with section headers
    #: attached, which is what the MedCPT article encoder is meant to see.
    retrieval_text: str
    publication_date: Optional[str]
    source_tier: str
    retracted: bool
    section: Optional[str] = None
    claim_classes: tuple[str, ...] = ()

    def to_evidence(self) -> Evidence:
        """Map onto the shared ``Evidence`` type the arms consume.

        ``text`` is used for the answer context, not ``retrieval_text``: the
        section markers exist to help the retriever, and putting them in the
        generator's context would change what both arms read.
        """
        return Evidence(
            evidence_id=self.chunk_id,
            text=self.text,
            source_tier=self.source_tier,
            persistent_id=self.document_id,
            section=self.section,
            publication_date=parse_publication_date(self.publication_date),
            retracted=self.retracted,
            claim_classes=self.claim_classes,
        )


def _as_bool(value) -> bool:
    """The corpus writes ``retracted`` as ``""``, ``"false"`` or a bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _snapshot_id_from_digest(digest: "hashlib._Hash") -> str:
    return f"corpus@{digest.hexdigest()[:16]}"


#: What to do when the same ``chunk_id`` appears more than once.
#: "raise" is the safe default: silently accepting a duplicate would let two
#: different passages be treated as interchangeable, which is exactly the
#: circularity risk ``freezing.py`` exists to prevent. "keep_first" is an
#: explicit opt-in for a known, small, real cause found in the actual
#: corpus (2026-09-21g): a handful of source documents survived Stage 05's
#: near-duplicate detection as two slightly different renderings of the
#: same paper, sharing a document_id, so Stage 06 generated the same
#: chunk_id for the first chunk of a shared section twice. 515 duplicated
#: ids / 900 extra lines out of 4,377,041 (~0.02%) in the corpus this was
#: diagnosed against - a real, bounded, explainable data-quality artifact,
#: not a pipeline failure, and not something to fix by editing the frozen
#: corpus file.
ON_DUPLICATE_POLICIES = ("raise", "keep_first")


def read_passages(corpus_root: str | Path) -> tuple[CorpusPassage, ...]:
    """Read every chunk, in file order.

    File order is preserved because it feeds the index row order, which feeds
    retrieval ties; a set that reorders between builds would not reproduce.
    """
    return read_passages_with_snapshot(corpus_root)[0]


def read_passages_with_snapshot(
    corpus_root: str | Path,
    *,
    on_progress: Optional[Callable[[int], None]] = None,
    progress_every: int = 200_000,
    on_duplicate: str = "raise",
) -> tuple[tuple[CorpusPassage, ...], str, tuple[dict[str, Any], ...]]:
    """``read_passages`` plus ``snapshot_id``, in one pass over the file.

    Returns ``(passages, snapshot_id, duplicates)``. ``duplicates`` is
    always populated when duplicate ids are found and ``on_duplicate`` is
    ``"keep_first"`` - never silent, whichever policy is chosen - listing
    each dropped chunk_id, the 1-based line it was first seen on, and the
    line it was dropped from, so every drop is traceable back to the exact
    corpus lines involved.

    ``build_index.py`` used to call ``read_passages`` then ``snapshot_id``
    separately - two full reads of the corpus file. Measured on a 300k-line/
    439 MB synthetic corpus, that redundant second read cost **nothing**
    (7.75s vs. 7.78s for one pass): the OS page cache made the second read
    essentially free. So this single-pass version is a real, harmless
    simplification - one fewer place the file path can be wrong, one fewer
    thing that could disagree - but **the time for a large real corpus is
    dominated by parsing that many lines in Python, not by I/O**, and this
    change does not make that faster. Each raw line is hashed (as bytes,
    before decoding) into the same running SHA-256 that ``snapshot_id``
    would have produced reading the file in 1 MB blocks - a streaming
    hash's digest depends only on the byte sequence and its order, not how
    it was chunked, so the two are guaranteed identical (locked by
    ``test_combined_reader_matches_the_standalone_snapshot_id``).

    ``on_progress(lines_read)`` is called every ``progress_every`` lines, so
    a long real run can show it is alive rather than sitting silent - which
    is indistinguishable from hung to someone watching it for the first
    time. ``build_index.py`` wires this to a printed line; nothing here
    prints on its own, so the function stays quiet for library/test use.

    ``on_duplicate="raise"`` (default) fails immediately on the first
    duplicate ``chunk_id``, exactly as before - callers who have not
    thought about this get the safe behaviour. ``on_duplicate="keep_first"``
    keeps the first occurrence (file order - the same tie-break rule
    ``DenseIndex`` already uses) and skips later ones, recording every drop
    in the returned ``duplicates`` tuple rather than silently discarding it.
    """
    if on_duplicate not in ON_DUPLICATE_POLICIES:
        raise CorpusError(
            f"on_duplicate must be one of {ON_DUPLICATE_POLICIES}, "
            f"got {on_duplicate!r}"
        )

    path = Path(corpus_root) / CHUNKS
    if not path.exists():
        raise CorpusError(
            f"corpus chunk file not found: {path}. Step 1 (corpus build) "
            "must complete before evidence can be retrieved."
        )

    passages: list[CorpusPassage] = []
    first_seen_line: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    # Opened in binary mode so the exact on-disk bytes are what get hashed
    # (matching the old block-wise ``snapshot_id``); each line is decoded
    # separately for parsing. Streamed rather than read whole - see
    # ``read_passages``'s historical note on the Windows >2GB OSError this
    # avoids.
    with open(path, "rb") as handle:
        for number, raw_line in enumerate(handle, 1):
            if on_progress is not None and number % progress_every == 0:
                on_progress(number)
            digest.update(raw_line)
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{number}: {exc}")
            for required in ("chunk_id", "text"):
                if required not in record:
                    raise CorpusError(f"{path}:{number}: missing {required!r}")
            chunk_id = record["chunk_id"]
            if chunk_id in first_seen_line:
                if on_duplicate == "raise":
                    raise CorpusError(
                        f"duplicate chunk_id {chunk_id!r}; evidence ids must "
                        "be unique or frozen candidate sets cannot be "
                        "replayed"
                    )
                duplicates.append({
                    "chunk_id": chunk_id,
                    "kept_line": first_seen_line[chunk_id],
                    "dropped_line": number,
                    "dropped_text_preview": record["text"][:200],
                })
                continue
            first_seen_line[chunk_id] = number
            claim_classes = record.get("claim_classes") or ()
            passages.append(CorpusPassage(
                chunk_id=chunk_id,
                document_id=record.get("document_id", ""),
                text=record["text"],
                retrieval_text=record.get("retrieval_text") or record["text"],
                publication_date=record.get("publication_date") or None,
                source_tier=record.get("source_tier", "unknown"),
                retracted=_as_bool(record.get("retracted")),
                section=record.get("section") or None,
                claim_classes=tuple(claim_classes),
            ))

    if not passages:
        raise CorpusError(f"{path} contains no passages")

    return (
        tuple(passages),
        _snapshot_id_from_digest(digest),
        tuple(duplicates),
    )


def snapshot_id(corpus_root: str | Path) -> str:
    """Identify the corpus build the evidence came from.

    A content hash of the chunk file, not a hand-written version string, so a
    corpus that changes cannot keep claiming the identifier frozen evidence
    was recorded under.

    Kept as a standalone, single-purpose function (rather than always going
    through ``read_passages_with_snapshot``) for callers - and tests - that
    want the id without paying for a full parse.
    """
    path = Path(corpus_root) / CHUNKS
    if not path.exists():
        raise CorpusError(f"corpus chunk file not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return _snapshot_id_from_digest(digest)


def dated_only(
    passages: Sequence[CorpusPassage],
) -> tuple[CorpusPassage, ...]:
    """Keep only passages carrying a publication date.

    ``_archive/docs_legacy/research_experimental_specification.md`` §15.3 requires candidate
    sets to contain only
    dated passages, applied identically to every arm at construction, so the
    undated branch of the temporal score never fires and ``undated_score``
    stays out of the tunable count. Applying it here - upstream of both arms -
    is what makes it identical by construction rather than by agreement.
    """
    return tuple(p for p in passages if p.publication_date)


def iter_texts(passages: Sequence[CorpusPassage]) -> Iterator[str]:
    for passage in passages:
        yield passage.retrieval_text
