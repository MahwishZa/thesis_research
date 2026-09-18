#!/usr/bin/env python3
"""Stage 04 - normalization.

Cleans encoding, whitespace and XML/HTML artifacts while PRESERVING biomedical
surface forms exactly: Aβ42, Aβ40, p-tau181, p-tau217, ARIA-E, ARIA-H, APOE ε4.
Aggressive terminology folding would destroy the distinctions the thesis studies,
so case-folding is used only for matching (see _common.fold) and never written
back into stored text.

Also applies the Alzheimer's relevance gate and records the full decision trace.
Excluded records keep their metadata and reason - nothing is dropped silently.

Two input modes:

  Real corpus (default): reads the finalized PMC manifest
  (metadata/pmc.csv) and each article's JATS XML, via
  _common.iter_pmc_records(). This is the corpus's actual content source -
  Stage 01 (PubMed) only ever produces PMIDs, never abstracts, so PMC full
  text is what this stage normalizes.

  --input PATH: reads one JSONL file directly instead, in the flat
  {document_id, pmid, title, abstract, mesh_terms, publication_date,
  license, source_tier} shape. This is the offline/fixture path -
  data/raw/pubmed/records.example.jsonl exercises it with no network and no
  real data, and it is what the automated tests use.

PubMed PMIDs that have no PMC full text contribute no content under the
current pipeline (Stage 01 never fetches an abstract). This stage reports
how many such PMIDs exist rather than silently ignoring them; extending
Stage 01 with an efetch/esummary call is how that gap would close.

Guidelines and textbooks (metadata/guidelines.csv, metadata/textbooks.csv)
are included in the real-data path too, for whichever rows Stage 03 has
downloaded (local_file set) under a licence that permits redistribution -
via _common.iter_official_documents(). A curated-but-not-yet-downloaded row,
or a downloaded row under a restricted licence, contributes no text; both
are counted and logged, never silently absorbed or fabricated.

Input : metadata/pmc.csv + data/raw/pmc/**
        + metadata/guidelines.csv + data/raw/guidelines/**
        + metadata/textbooks.csv + data/raw/textbooks/**
        (or --input JSONL for the offline/fixture path)
Output: data/normalized/documents.jsonl
"""
from __future__ import annotations
import argparse, csv, html, re, sys, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE, DATA, METADATA, CorpusPipelineError, GUIDELINE_REGISTRY_FIELDS,
    TEXTBOOK_REGISTRY_FIELDS, get_logger, iter_official_documents,
    iter_pmc_records, read_jsonl, write_jsonl, write_report,
    assess_ad_relevance, redistribution_allowed, PRESERVE_VERBATIM,
)

OUT = DATA / "normalized" / "documents.jsonl"
PMC_MANIFEST = METADATA / "pmc.csv"
PUBMED_MANIFEST = METADATA / "pubmed.csv"
GUIDELINES_MANIFEST = METADATA / "guidelines.csv"
TEXTBOOKS_MANIFEST = METADATA / "textbooks.csv"
_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")
_TAG = re.compile(r"<[^>]{1,200}>")


def normalize_text(s: str) -> str:
    """NFC only - NFKD would decompose 'β' and 'ε' and break the preserved forms."""
    if not s:
        return ""
    s = html.unescape(s)
    s = _TAG.sub(" ", s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS.sub(" ", s)
    s = _NL.sub("\n\n", s)
    return s.strip()


def normalize_record(rec: dict) -> dict:
    """Apply text normalization to a record's title, abstract and sections
    in place, and return it. Shared by both input modes so a fixture-tested
    record and a real PMC record go through identical cleaning."""
    rec["title"] = normalize_text(rec.get("title", ""))
    rec["abstract"] = normalize_text(rec.get("abstract", ""))
    if rec.get("sections"):
        rec["sections"] = [
            {**sec, "text": normalize_text(sec.get("text", ""))}
            for sec in rec["sections"]
        ]
    return rec


def apply_relevance_gate(rec: dict) -> tuple[dict, bool]:
    """Run the AD-relevance gate and stamp its decision onto the record.

    Returns (record, ad_relevant) so the caller can count exclusions without
    re-reading the field back out.
    """
    rel = assess_ad_relevance(rec)
    rec["ad_relevant"] = rel.ad_relevant
    rec["ad_relevance_score"] = rel.score
    rec["ad_rules_fired"] = ";".join(rel.rules_fired)
    rec["ad_exclusion_reason"] = rel.exclusion_reason or ""
    rec["redistribution_allowed"] = redistribution_allowed(rec.get("license"))
    return rec, rel.ad_relevant


def pubmed_only_pmid_count(pmc_pmids: set[str], log) -> int:
    """How many PubMed PMIDs have no PMC full text and so contribute no
    content here. Reported, never silently absorbed - see module docstring."""
    if not PUBMED_MANIFEST.exists():
        return 0
    with PUBMED_MANIFEST.open(encoding="utf-8", newline="") as handle:
        pubmed_pmids = {row["pmid"] for row in csv.DictReader(handle) if row.get("pmid")}
    uncovered = pubmed_pmids - pmc_pmids
    if uncovered:
        log.warning(
            "%d/%d PubMed PMIDs have no PMC full text and contribute no "
            "content under the current pipeline (Stage 01 does not fetch "
            "abstracts). Extend Stage 01 with efetch/esummary if PubMed-only "
            "content is required.",
            len(uncovered), len(pubmed_pmids),
        )
    return len(uncovered)


def iter_official_document_records(log):
    """Yield normalize-ready records from whichever guideline/textbook rows
    Stage 03 has actually downloaded, under a redistributable licence.

    Each registry is optional: a fresh clone or a corpus with no guidelines
    curated yet has neither file, and that is not an error here - Stage 03
    already logs the "registry not found" / "registry is empty" cases.
    """
    if GUIDELINES_MANIFEST.exists():
        yield from iter_official_documents(
            GUIDELINES_MANIFEST, GUIDELINE_REGISTRY_FIELDS, BASE, log,
            default_source_tier="clinical_guideline")
    if TEXTBOOKS_MANIFEST.exists():
        yield from iter_official_documents(
            TEXTBOOKS_MANIFEST, TEXTBOOK_REGISTRY_FIELDS, BASE, log,
            default_source_tier="reference_work")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None,
                    help="JSONL of raw records (offline/fixture path); "
                         "omit to read the real, finalized PMC manifest")
    ap.add_argument("--pmc-manifest", default=str(PMC_MANIFEST))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    log = get_logger("04_normalize", "quality_control.log")

    kept: list[dict] = []
    excluded = 0
    rows = []

    def process(rec: dict) -> None:
        nonlocal excluded
        rec = normalize_record(rec)
        rec, ad_relevant = apply_relevance_gate(rec)
        kept.append(rec)
        if not ad_relevant:
            excluded += 1
        rows.append({
            "document_id": rec.get("document_id", ""),
            "ad_relevant": ad_relevant,
            "score": rec["ad_relevance_score"],
            "rules": rec["ad_rules_fired"],
            "exclusion_reason": rec["ad_exclusion_reason"],
        })

    if args.input:
        # Offline/fixture path: one flat JSONL file, unchanged from before
        # the PMC bridge existed. records.example.jsonl exercises this.
        src = Path(args.input)
        if not src.exists():
            log.error("no input at %s - run stages 01-03 first", src)
            return 2
        for i, rec in enumerate(read_jsonl(src)):
            if args.limit and i >= args.limit:
                break
            process(rec)
    else:
        # Real corpus path: the finalized PMC manifest + its JATS XML.
        manifest_path = Path(args.pmc_manifest)
        try:
            pmc_pmids: set[str] = set()
            for i, rec in enumerate(iter_pmc_records(manifest_path, BASE, log)):
                if args.limit and i >= args.limit:
                    break
                if rec.get("pmid"):
                    pmc_pmids.add(rec["pmid"])
                process(rec)
        except CorpusPipelineError as exc:
            log.error(str(exc))
            return 2
        pubmed_only_pmid_count(pmc_pmids, log)

        for rec in iter_official_document_records(log):
            process(rec)

    n = write_jsonl(OUT, kept)
    log.info("stage 04 | normalized=%d | ad_relevant=%d | excluded=%d",
             n, n - excluded, excluded)
    for f in PRESERVE_VERBATIM[:3]:
        log.info("preserved verbatim: %s", f)
    write_report("normalization_report.csv", rows,
                 ["document_id", "ad_relevant", "score", "rules", "exclusion_reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
