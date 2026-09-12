#!/usr/bin/env python3
"""
Stage 01: PubMed PMID retrieval.

This script executes the complete PubMed query expressions defined in
config/search_queries.yaml.

Design principles:
- Each configured query is executed independently.
- The exact query expression in search_queries.yaml is authoritative.
- No universal date, human-study, diagnosis-only, or treatment exclusions
  are appended automatically.
- PMID provenance is preserved by query_id.
- API keys are intentionally NOT supported. PubMed E-utilities can be
  accessed without an API key at the standard unauthenticated rate.
- This stage retrieves PMID lists only. Full metadata/abstract retrieval
  belongs to a later stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import DATA, load_config, get_logger, write_report


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RAW = DATA / "raw" / "pubmed"

# NCBI E-utilities unauthenticated request rate.
# Approximately 3 requests/second maximum.
REQUEST_DELAY_SECONDS = 0.34


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utcnow() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dir() -> None:
    """Create PubMed raw-data directory if necessary."""
    RAW.mkdir(parents=True, exist_ok=True)


def load_queries() -> dict:
    """
    Load and validate config/search_queries.yaml.

    Expected structure:

        version: ...
        database: PubMed
        queries:
          query_id:
            query: "..."
            priority: primary
            execution_group: primary
            ...

    Returns:
        Dictionary containing the complete configuration.
    """
    config_path = PROJECT_ROOT / "config" / "search_queries.yaml"
    cfg = load_config(config_path)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected mapping at top level of {config_path}"
        )

    queries = cfg.get("queries")

    if not isinstance(queries, dict) or not queries:
        raise ValueError(
            f"{config_path} must contain a non-empty top-level 'queries' mapping."
        )

    return cfg


def select_queries(cfg: dict, group: str | None, query_id: str | None) -> dict:
    """Select queries according to --query or --group."""
    queries = cfg["queries"]

    if query_id:
        if query_id not in queries:
            available = ", ".join(sorted(queries))
            raise ValueError(
                f"Unknown query_id '{query_id}'. Available queries: {available}"
            )
        return {query_id: queries[query_id]}

    if group:
        selected = {}

        for qid, spec in queries.items():
            execution_group = spec.get("execution_group")

            if execution_group == group:
                selected[qid] = spec

        if not selected:
            raise ValueError(
                f"No queries found for execution group '{group}'."
            )

        return selected

    return queries


def build_esearch_url(term: str, retmax: int) -> str:
    """
    Build an NCBI ESearch URL.

    No API key is included intentionally.
    """
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
    }

    return f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# PubMed retrieval
# ---------------------------------------------------------------------------

def esearch(term: str, retmax: int, logger) -> tuple[int, list[str]]:
    """
    Execute one PubMed ESearch request.

    Returns:
        (total_result_count, retrieved_pmids)
    """
    url = build_esearch_url(term, retmax)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MS-Thesis-PubMed-Retrieval/1.0"
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "NCBI rate limit reached (HTTP 429). "
                "Increase the delay between requests before retrying."
            ) from exc

        raise RuntimeError(
            f"NCBI HTTP error {exc.code}: {exc.reason}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Unable to connect to NCBI E-utilities: {exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "NCBI E-utilities request timed out."
        ) from exc

    result = payload.get("esearchresult", {})

    try:
        count = int(result.get("count", 0))
    except (TypeError, ValueError):
        count = 0

    pmids = [
        str(pmid)
        for pmid in result.get("idlist", [])
        if pmid
    ]

    logger.info(
        "PubMed ESearch returned %d total records; retrieved %d PMIDs.",
        count,
        len(pmids),
    )

    return count, pmids


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_pmids(query_id: str, pmids: list[str]) -> Path:
    """Save retrieved PMID list for one query."""
    ensure_output_dir()

    output_path = RAW / f"{query_id}.pmids.json"

    payload = {
        "query_id": query_id,
        "database": "PubMed",
        "retrieval_method": "NCBI E-utilities ESearch",
        "retrieval_timestamp": utcnow(),
        "pmid_count": len(pmids),
        "pmids": pmids,
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return output_path


def save_query_expression(query_id: str, term: str) -> Path:
    """Save exact query expression for provenance/audit."""
    ensure_output_dir()

    output_path = RAW / f"{query_id}.query.txt"

    output_path.write_text(
        term.strip() + "\n",
        encoding="utf-8",
    )

    return output_path


def write_retrieval_report(rows: list[dict]) -> None:
    """Write retrieval_report.csv."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "retrieval_report.csv"

    write_report(
        report_path,
        rows,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve PubMed PMIDs using configured ESearch queries."
    )

    parser.add_argument(
        "--query",
        help="Run only one query_id.",
    )

    parser.add_argument(
        "--group",
        choices=[
            "primary",
            "secondary",
            "currency",
            "audit",
        ],
        help="Run all queries belonging to an execution_group.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of PMIDs retrieved per query (default: 100).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Execute count-only ESearch requests without retrieving or "
            "saving PMID lists."
        ),
    )

    parser.add_argument(
        "--print-queries",
        action="store_true",
        help="Print configured query expressions without contacting NCBI.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    if args.query and args.group:
        raise ValueError(
            "--query and --group cannot be used together."
        )

    logger = get_logger("pubmed_download")

    cfg = load_queries()

    selected = select_queries(
        cfg,
        group=args.group,
        query_id=args.query,
    )

    # ---------------------------------------------------------------
    # Print configured queries and exit.
    # ---------------------------------------------------------------

    if args.print_queries:
        for query_id, spec in selected.items():
            print(f"[{query_id}]")
            print(spec["query"])
            print()

        return 0

    # ---------------------------------------------------------------
    # Execute retrieval.
    # ---------------------------------------------------------------

    report_rows = []

    total_queries = len(selected)

    logger.info(
        "Starting PubMed retrieval: %d query/queries.",
        total_queries,
    )

    logger.info(
        "Unauthenticated NCBI mode enabled; request delay = %.2f seconds.",
        REQUEST_DELAY_SECONDS,
    )

    for index, (query_id, spec) in enumerate(selected.items(), start=1):

        term = str(spec.get("query", "")).strip()

        if not term:
            logger.warning(
                "Skipping '%s': query expression is empty.",
                query_id,
            )
            continue

        priority = spec.get("priority", "")
        execution_group = spec.get("execution_group", "")
        disease_scope = spec.get("disease_scope", "")

        logger.info(
            "[%d/%d] Executing query '%s'.",
            index,
            total_queries,
            query_id,
        )

        logger.info(
            "Query expression: %s",
            term,
        )

        retrieval_timestamp = utcnow()

        try:
            if args.dry_run:
                total_count, pmids = esearch(
                    term,
                    retmax=0,
                    logger=logger,
                )
            else:
                total_count, pmids = esearch(
                    term,
                    retmax=args.limit,
                    logger=logger,
                )

        except RuntimeError as exc:
            logger.error(
                "Query '%s' failed: %s",
                query_id,
                exc,
            )

            report_rows.append(
                {
                    "query_id": query_id,
                    "priority": priority,
                    "execution_group": execution_group,
                    "disease_scope": disease_scope,
                    "timestamp_utc": retrieval_timestamp,
                    "result_count": "",
                    "retrieved_pmid_count": "",
                    "dry_run": args.dry_run,
                    "status": "ERROR",
                    "error": str(exc),
                    "query_expression": term,
                }
            )

            # Stop rather than immediately hammering NCBI after a
            # connection/rate-limit error.
            raise

        if not args.dry_run:
            pmid_path = save_pmids(
                query_id,
                pmids,
            )

            query_path = save_query_expression(
                query_id,
                term,
            )

            logger.info(
                "Saved %d PMIDs to %s.",
                len(pmids),
                pmid_path,
            )

            logger.info(
                "Saved query expression to %s.",
                query_path,
            )

        report_rows.append(
            {
                "query_id": query_id,
                "priority": priority,
                "execution_group": execution_group,
                "disease_scope": disease_scope,
                "timestamp_utc": retrieval_timestamp,
                "result_count": total_count,
                "retrieved_pmid_count": len(pmids),
                "dry_run": args.dry_run,
                "status": "OK",
                "error": "",
                "query_expression": term,
            }
        )

        # Respect unauthenticated NCBI request rate.
        if index < total_queries:
            time.sleep(REQUEST_DELAY_SECONDS)

    write_retrieval_report(report_rows)

    logger.info(
        "PubMed retrieval completed successfully."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)