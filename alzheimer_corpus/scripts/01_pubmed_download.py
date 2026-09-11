#!/usr/bin/env python3
"""Stage 01 - PubMed retrieval via NCBI E-utilities.

Builds the seven Alzheimer's query families from config/search_queries.yaml and
retrieves records. Always count-only first: --dry-run reports hit counts without
fetching, so the scale of a query is known before any bulk download.

NETWORK: requires egress to eutils.ncbi.nlm.nih.gov. Where organisation policy
blocks it the script reports the blocked host and exits non-zero. It never
routes around the restriction and never fabricates counts.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DATA, load_config, get_logger, utcnow, write_report,
                     assess_ad_relevance)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RAW = DATA / "raw" / "pubmed"


def build_term(cfg: dict, family_id: str) -> str:
    a = cfg["anchor_block"]
    anchor = "(" + " OR ".join(a["mesh"] + [f'{t}[tiab]' for t in a["tiab"]]) + ")"
    fam = cfg["families"][family_id]
    parts = [anchor]
    block = []
    for t in fam.get("concept_block") or []:
        block.append(f"{t}[tiab]")
    block += fam.get("mesh_block") or []
    if block:
        parts.append("(" + " OR ".join(block) + ")")
    parts.append(cfg["filters"]["humans"])
    parts.append(cfg["publication_window"]["filter"])
    return " AND ".join(parts)


def esearch(term: str, retmax: int, api_key: str | None, log) -> dict:
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode())["esearchresult"]
    except urllib.error.URLError as e:
        log.error("NETWORK BLOCKED for eutils.ncbi.nlm.nih.gov: %s", e)
        log.error("Organisation egress policy denial is not retried or bypassed. "
                  "Run this stage where NCBI access is permitted.")
        raise SystemExit(2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", help="family id, e.g. Q01_alzheimer_core; default all")
    ap.add_argument("--limit", type=int, default=100, help="max PMIDs per family")
    ap.add_argument("--dry-run", action="store_true", help="count only, fetch nothing")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--print-queries", action="store_true",
                    help="print the built query strings and exit (no network)")
    args = ap.parse_args(argv)

    log = get_logger("01_pubmed", "retrieval.log")
    cfg = load_config("search_queries.yaml")
    fams = [args.query] if args.query else list(cfg["families"])

    if args.print_queries:
        for f in fams:
            print(f"\n### {f} — {cfg['families'][f]['label']}\n{build_term(cfg, f)}")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    log.info("stage 01 start | families=%d | dry_run=%s | limit=%d",
             len(fams), args.dry_run, args.limit)
    for f in fams:
        term = build_term(cfg, f)
        res = esearch(term, 0 if args.dry_run else args.limit, args.api_key, log)
        count = int(res.get("count", 0))
        ids = res.get("idlist", []) if not args.dry_run else []
        log.info("%s | hits=%d | retrieved=%d", f, count, len(ids))
        rows.append({"query_id": f, "label": cfg["families"][f]["label"],
                     "timestamp_utc": utcnow(), "result_count": count,
                     "retrieved_pmid_count": len(ids), "dry_run": args.dry_run})
        if ids:
            (RAW / f"{f}.pmids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
        time.sleep(0.34 if not args.api_key else 0.1)   # NCBI rate limit
    p = write_report("retrieval_report.csv", rows,
                     ["query_id", "label", "timestamp_utc", "result_count",
                      "retrieved_pmid_count", "dry_run"])
    log.info("stage 01 done | report=%s", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
