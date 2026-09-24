"""Read the Alzheimer's corpus chunk file into passages, without touching it.

``corpus/`` is built and owned by its own pipeline. This module only
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


class StreamingCorpusReader:
    """Streams ``CorpusPassage`` records one at a time, in file order,
    without ever holding more than one line's worth of the corpus in memory.

    Exists for one reason: at the real corpus's scale (4.3M+ chunks, a
    13 GB file), materializing every passage into a Python list - as
    ``read_passages_with_snapshot`` below does - holds the full corpus text
    resident at once. Measured on the real corpus on 16 GB-RAM hardware,
    that alone (before any model or vectors) pushed free memory to ~0 and
    caused sustained OS paging (2026-09-24 diagnostic run). This class is
    the shared, single-parsing-path fix: anything that can consume passages
    one at a time - like the streaming index build - should iterate this
    instead of materializing a list.

    Iterate this object to consume passages; each full iteration re-reads
    the file from the start (cheap and correct, never stateful across
    iterations). ``snapshot_id`` and ``duplicates`` reflect the most recent
    completed iteration and raise if none has finished yet - reading them
    mid-iteration or before any iteration would silently return a partial
    digest, which is exactly the kind of thing that should fail loudly
    instead.
    """

    def __init__(
        self,
        corpus_root: str | Path,
        *,
        on_progress: Optional[Callable[[int], None]] = None,
        progress_every: int = 200_000,
        on_duplicate: str = "raise",
        dated_only: bool = False,
    ) -> None:
        if on_duplicate not in ON_DUPLICATE_POLICIES:
            raise CorpusError(
                f"on_duplicate must be one of {ON_DUPLICATE_POLICIES}, "
                f"got {on_duplicate!r}"
            )
        self.path = Path(corpus_root) / CHUNKS
        if not self.path.exists():
            raise CorpusError(
                f"corpus chunk file not found: {self.path}. Step 1 (corpus "
                "build) must complete before evidence can be retrieved."
            )
        self._on_progress = on_progress
        self._progress_every = progress_every
        self._on_duplicate = on_duplicate
        self._dated_only = dated_only
        self._snapshot_id: Optional[str] = None
        self._duplicates: Optional[tuple[dict[str, Any], ...]] = None
        self._n_yielded: int = 0

    def __iter__(self) -> Iterator[CorpusPassage]:
        self._snapshot_id = None
        self._duplicates = None
        self._n_yielded = 0
        first_seen_line: dict[str, int] = {}
        duplicates: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        # Opened in binary mode so the exact on-disk bytes are what get
        # hashed (matching the block-wise ``snapshot_id``); each line is
        # decoded separately for parsing. Streamed rather than read whole -
        # see the Windows >2GB OSError this avoids (module docstring).
        with open(self.path, "rb") as handle:
            for number, raw_line in enumerate(handle, 1):
                if self._on_progress is not None and number % self._progress_every == 0:
                    self._on_progress(number)
                digest.update(raw_line)
                line = raw_line.decode("utf-8")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{self.path}:{number}: {exc}")
                for required in ("chunk_id", "text"):
                    if required not in record:
                        raise CorpusError(f"{self.path}:{number}: missing {required!r}")
                chunk_id = record["chunk_id"]
                if chunk_id in first_seen_line:
                    if self._on_duplicate == "raise":
                        raise CorpusError(
                            f"duplicate chunk_id {chunk_id!r}; evidence ids "
                            "must be unique or frozen candidate sets cannot "
                            "be replayed"
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
                passage = CorpusPassage(
                    chunk_id=chunk_id,
                    document_id=record.get("document_id", ""),
                    text=record["text"],
                    retrieval_text=record.get("retrieval_text") or record["text"],
                    publication_date=record.get("publication_date") or None,
                    source_tier=record.get("source_tier", "unknown"),
                    retracted=_as_bool(record.get("retracted")),
                    section=record.get("section") or None,
                    claim_classes=tuple(claim_classes),
                )
                if self._dated_only and not passage.publication_date:
                    continue
                self._n_yielded += 1
                yield passage
        self._snapshot_id = _snapshot_id_from_digest(digest)
        self._duplicates = tuple(duplicates)

    @property
    def snapshot_id(self) -> str:
        if self._snapshot_id is None:
            raise CorpusError(
                "snapshot_id is only available after a full iteration has "
                "completed"
            )
        return self._snapshot_id

    @property
    def duplicates(self) -> tuple[dict[str, Any], ...]:
        if self._duplicates is None:
            raise CorpusError(
                "duplicates is only available after a full iteration has "
                "completed"
            )
        return self._duplicates

    @property
    def n_yielded(self) -> int:
        """Passages yielded by the most recent completed iteration."""
        return self._n_yielded


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

    A thin wrapper around ``StreamingCorpusReader``: this function still
    materializes every passage into a tuple (its whole contract is "give me
    everything"), but the line-by-line parsing, digest and duplicate logic
    live in exactly one place now, so the streaming index build below and
    this function can never silently diverge in behaviour. Use
    ``StreamingCorpusReader`` directly for anything that can consume
    passages one at a time instead of materializing them all.

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
    reader = StreamingCorpusReader(
        corpus_root,
        on_progress=on_progress,
        progress_every=progress_every,
        on_duplicate=on_duplicate,
    )
    passages = tuple(reader)
    if not passages:
        raise CorpusError(f"{reader.path} contains no passages")
    return passages, reader.snapshot_id, reader.duplicates


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
