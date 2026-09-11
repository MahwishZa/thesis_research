#!/usr/bin/env python3
"""Stage 05 - document-level deduplication, BEFORE chunking.

Deduplicating after chunking would spend expensive tokenization on duplicates
and let a duplicate's chunks outvote the original in retrieval.

Identifier hierarchy: PMID -> PMCID -> DOI -> normalized title+year -> content
SHA-256. Duplicates are RECORDED, never silently deleted: a 2021 guideline and
its 2024 revision are different documents, and version relationships are
evidence the temporal experiment needs.

Input : data/normalized/documents.jsonl   Output: data/deduplicated/documents.jsonl
"""
from __future__ import annotations
import argparse, hashlib, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, METADATA, get_logger, read_jsonl, write_jsonl, write_report, fold

SRC = DATA / "normalized" / "documents.jsonl"
OUT = DATA / "deduplicated" / "documents.jsonl"
_PUNCT = re.compile(r"[^\w\s]+")


def title_key(rec: dict) -> str:
    t = _PUNCT.sub(" ", fold(rec.get("title", "")))
    t = " ".join(t.split())
    return f"{t}|{str(rec.get('publication_date',''))[:4]}"


def content_hash(rec: dict) -> str:
    return hashlib.sha256(
        (fold(rec.get("title", "")) + "\n" + fold(rec.get("abstract", ""))).encode()
    ).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(SRC))
    args = ap.parse_args(argv)
    log = get_logger("05_deduplicate", "deduplication.log")

    src = Path(args.input)
    if not src.exists():
        log.error("no input at %s - run stage 04 first", src)
        return 2

    seen: dict[tuple[str, str], str] = {}
    kept, dupes = [], []
    for rec in read_jsonl(src):
        doc_id = str(rec.get("document_id") or rec.get("pmid") or "")
        keys = []
        for field in ("pmid", "pmcid", "doi"):
            if rec.get(field):
                keys.append((field, str(rec[field]).strip().lower()))
        keys.append(("title_year", title_key(rec)))
        keys.append(("content_sha256", content_hash(rec)))

        hit = next(((k, seen[k]) for k in keys if k in seen), None)
        if hit:
            (k, canonical) = hit
            dupes.append({"canonical_document_id": canonical,
                          "duplicate_document_id": doc_id,
                          "duplicate_reason": k[0], "similarity_method": "exact_key",
                          "similarity_score": 1.0, "decision": "recorded_not_deleted"})
            log.info("duplicate %s of %s via %s", doc_id, canonical, k[0])
            continue
        for k in keys:
            seen[k] = doc_id
        kept.append(rec)

    n = write_jsonl(OUT, kept)
    write_report("deduplication_report.csv", dupes,
                 ["canonical_document_id", "duplicate_document_id", "duplicate_reason",
                  "similarity_method", "similarity_score", "decision"])
    (METADATA / "duplicates.csv").write_text(
        "canonical_document_id,duplicate_document_id,duplicate_reason,similarity_method,"
        "similarity_score,decision\n" +
        "".join(f"{d['canonical_document_id']},{d['duplicate_document_id']},"
                f"{d['duplicate_reason']},{d['similarity_method']},"
                f"{d['similarity_score']},{d['decision']}\n" for d in dupes),
        encoding="utf-8", newline="\n")
    log.info("stage 05 | unique=%d | duplicates recorded=%d", n, len(dupes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
