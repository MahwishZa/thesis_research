#!/usr/bin/env python3
"""Stage 04 - normalization.

Cleans encoding, whitespace and XML/HTML artifacts while PRESERVING biomedical
surface forms exactly: Aβ42, Aβ40, p-tau181, p-tau217, ARIA-E, ARIA-H, APOE ε4.
Aggressive terminology folding would destroy the distinctions the thesis studies,
so case-folding is used only for matching (see _common.fold) and never written
back into stored text.

Also applies the Alzheimer's relevance gate and records the full decision trace.
Excluded records keep their metadata and reason - nothing is dropped silently.

Input : data/raw/**            Output: data/normalized/documents.jsonl
"""
from __future__ import annotations
import argparse, html, re, sys, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DATA, get_logger, read_jsonl, write_jsonl, write_report,
                     assess_ad_relevance, redistribution_allowed, PRESERVE_VERBATIM)

OUT = DATA / "normalized" / "documents.jsonl"
_WS = re.compile(r"[ \t ]+")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="JSONL of raw records")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    log = get_logger("04_normalize", "quality_control.log")

    src = Path(args.input) if args.input else (DATA / "raw" / "pubmed" / "records.example.jsonl")
    if not src.exists():
        log.error("no input at %s - run stages 01-03 first", src)
        return 2

    kept, excluded, rows = [], 0, []
    for i, rec in enumerate(read_jsonl(src)):
        if args.limit and i >= args.limit:
            break
        rec["title"] = normalize_text(rec.get("title", ""))
        rec["abstract"] = normalize_text(rec.get("abstract", ""))
        rel = assess_ad_relevance(rec)
        rec["ad_relevant"] = rel.ad_relevant
        rec["ad_relevance_score"] = rel.score
        rec["ad_rules_fired"] = ";".join(rel.rules_fired)
        rec["ad_exclusion_reason"] = rel.exclusion_reason or ""
        rec["redistribution_allowed"] = redistribution_allowed(rec.get("license"))
        kept.append(rec)
        if not rel.ad_relevant:
            excluded += 1
        rows.append({"document_id": rel.document_id, "ad_relevant": rel.ad_relevant,
                     "score": rel.score, "rules": ";".join(rel.rules_fired),
                     "exclusion_reason": rel.exclusion_reason or ""})
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
