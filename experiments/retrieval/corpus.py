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
from typing import Iterator, Optional, Sequence

from systems.interfaces.evidence import Evidence

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


def read_passages(corpus_root: str | Path) -> tuple[CorpusPassage, ...]:
    """Read every chunk, in file order.

    File order is preserved because it feeds the index row order, which feeds
    retrieval ties; a set that reorders between builds would not reproduce.
    """
    path = Path(corpus_root) / CHUNKS
    if not path.exists():
        raise CorpusError(
            f"corpus chunk file not found: {path}. Step 1 (corpus build) "
            "must complete before evidence can be retrieved."
        )

    passages: list[CorpusPassage] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{number}: {exc}")
        for required in ("chunk_id", "text"):
            if required not in record:
                raise CorpusError(f"{path}:{number}: missing {required!r}")
        claim_classes = record.get("claim_classes") or ()
        passages.append(CorpusPassage(
            chunk_id=record["chunk_id"],
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

    seen: set[str] = set()
    for passage in passages:
        if passage.chunk_id in seen:
            raise CorpusError(
                f"duplicate chunk_id {passage.chunk_id!r}; evidence ids must "
                "be unique or frozen candidate sets cannot be replayed"
            )
        seen.add(passage.chunk_id)

    return tuple(passages)


def snapshot_id(corpus_root: str | Path) -> str:
    """Identify the corpus build the evidence came from.

    A content hash of the chunk file, not a hand-written version string, so a
    corpus that changes cannot keep claiming the identifier frozen evidence
    was recorded under.
    """
    path = Path(corpus_root) / CHUNKS
    if not path.exists():
        raise CorpusError(f"corpus chunk file not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"alzheimer_corpus@{digest.hexdigest()[:16]}"


def dated_only(
    passages: Sequence[CorpusPassage],
) -> tuple[CorpusPassage, ...]:
    """Keep only passages carrying a publication date.

    ``docs/research_experimental_specification.md`` §15.3 requires candidate
    sets to contain only
    dated passages, applied identically to every arm at construction, so the
    undated branch of the recency score never fires and ``undated_score``
    stays out of the tunable count. Applying it here - upstream of both arms -
    is what makes it identical by construction rather than by agreement.
    """
    return tuple(p for p in passages if p.publication_date)


def iter_texts(passages: Sequence[CorpusPassage]) -> Iterator[str]:
    for passage in passages:
        yield passage.retrieval_text
