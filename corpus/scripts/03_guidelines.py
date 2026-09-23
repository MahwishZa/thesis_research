#!/usr/bin/env python3
"""Stage 03 - clinical guidelines and textbooks.

Guidelines and textbooks are acquired through each publisher's official
route, never scraped, and reuse rights are verified PER DOCUMENT - one
organisation publishing openly says nothing about the next document.
Restricted guidance is recorded as metadata only; its text never enters
the distributable corpus.

Populating metadata/guidelines.csv and metadata/textbooks.csv with a row per
document - including a licence verified for that specific document - is a
manual step by design and is not performed here. What this stage does:

  --validate (default): report row counts and licensing status for both
    registries. Network-free.

  --download: for every row that has a source_url but no local_file yet,
    download it (HTTPS + PDF only - see _common.py's module docstring for
    why), verify it, and record local_file back into the registry. Already-
    downloaded rows (local_file already set) are skipped, so this is safe
    to rerun. A row with no source_url (most textbooks, whose access_method
    may not be a public URL) is left for manual acquisition and does not
    count as a failure.

Text extraction into the normalized corpus happens in 04_normalize.py,
which reads whatever this stage has downloaded - the same split PMC
extraction uses: Stage 02/03 retrieve, Stage 04 normalizes everything.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE, METADATA, CorpusPipelineError, GUIDELINE_REGISTRY_FIELDS,
    TEXTBOOK_REGISTRY_FIELDS, download_document, get_logger,
    read_document_registry, redistribution_allowed, write_document_registry,
)

GUIDELINES_CSV = METADATA / "guidelines.csv"
TEXTBOOKS_CSV = METADATA / "textbooks.csv"
GUIDELINE_RAW_DIR = BASE / "data" / "raw" / "guidelines"
TEXTBOOK_RAW_DIR = BASE / "data" / "raw" / "textbooks"


def validate_registry(path: Path, fields: tuple[str, ...], label: str, log) -> None:
    if not path.exists():
        log.warning("%s: registry not found at %s", label, path)
        return
    try:
        rows = read_document_registry(path, fields)
    except CorpusPipelineError as exc:
        log.error("%s: %s", label, exc)
        return

    log.info("stage 03 | %d %s rows in %s", len(rows), label, path.name)

    distributable = restricted = unknown = downloaded = 0
    for row in rows:
        if (row.get("local_file") or "").strip():
            downloaded += 1
        lic = (row.get("license") or "").strip()
        if not lic:
            unknown += 1
            log.warning("%s: no licence recorded -> treated as NOT "
                       "redistributable", row.get("document_id") or row.get("title"))
        elif redistribution_allowed(lic):
            distributable += 1
        else:
            restricted += 1

    log.info("%s licensing: %d distributable | %d restricted | %d unknown "
             "(fail closed) | %d downloaded", label, distributable, restricted,
             unknown, downloaded)
    if not rows:
        log.info("%s registry is empty - populate %s per MANUAL steps",
                 label, path.name)


def download_registry(path: Path, fields: tuple[str, ...], raw_dir: Path,
                      label: str, log, *, corpus_root: Path = BASE
                      ) -> tuple[int, int, int]:
    """Download every not-yet-downloaded row that has a source_url.

    Returns (downloaded, skipped_no_url, failed). Rewrites the registry
    atomically only if at least one row's local_file changed.

    ``corpus_root`` is what ``local_file`` gets recorded relative to -
    parameterised (matching iter_pmc_records/iter_official_documents'
    corpus_root argument) rather than hardcoded to the real BASE, so this
    is exercisable against an isolated test tree, not only the real corpus.
    """
    if not path.exists():
        log.warning("%s: registry not found at %s", label, path)
        return 0, 0, 0
    try:
        rows = read_document_registry(path, fields)
    except CorpusPipelineError as exc:
        log.error("%s: %s", label, exc)
        return 0, 0, 0

    downloaded = skipped = failed = 0
    changed = False
    for row in rows:
        if (row.get("local_file") or "").strip():
            continue  # already downloaded - safe to rerun
        url = (row.get("source_url") or "").strip()
        if not url:
            skipped += 1
            continue
        document_id = (row.get("document_id") or "").strip()
        if not document_id:
            log.error("%s: row with source_url %s has no document_id - "
                      "skipping (every downloadable row needs one)", label, url)
            failed += 1
            continue

        destination = raw_dir / f"{document_id}.pdf"
        try:
            download_document(url, destination, log)
        except CorpusPipelineError as exc:
            failed += 1
            log.error("%s: %s: %s", label, document_id, exc)
            continue

        row["local_file"] = str(destination.relative_to(corpus_root))
        if "redistribution_allowed" in fields:
            row["redistribution_allowed"] = str(
                redistribution_allowed(row.get("license"))).lower()
        downloaded += 1
        changed = True

    if changed:
        write_document_registry(path, fields, rows)

    log.info("%s download | downloaded=%d | no_url=%d | failed=%d",
             label, downloaded, skipped, failed)
    return downloaded, skipped, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true",
                    help="download rows that have a source_url but no local_file yet")
    args = ap.parse_args(argv)
    log = get_logger("03_guidelines", "retrieval.log")

    if args.download:
        _, _, g_failed = download_registry(
            GUIDELINES_CSV, GUIDELINE_REGISTRY_FIELDS, GUIDELINE_RAW_DIR,
            "guidelines", log)
        _, _, t_failed = download_registry(
            TEXTBOOKS_CSV, TEXTBOOK_REGISTRY_FIELDS, TEXTBOOK_RAW_DIR,
            "textbooks", log)
        if g_failed or t_failed:
            log.error("Stage 03 download completed with %d failure(s).",
                     g_failed + t_failed)
            return 1

    validate_registry(GUIDELINES_CSV, GUIDELINE_REGISTRY_FIELDS, "guidelines", log)
    validate_registry(TEXTBOOKS_CSV, TEXTBOOK_REGISTRY_FIELDS, "textbooks", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
