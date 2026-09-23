#!/usr/bin/env python3

"""
Stage 01 - PubMed PMID retrieval.

Reads PubMed search queries from:

    config/search_queries.yaml

Stores PMID lists in:

    data/raw/pubmed/

Writes the retrieval log to:

    logs/retrieval.log

Writes the retrieval report to:

    reports/retrieval_report.csv

Usage examples:

    python scripts/01_pubmed_download.py --print-queries

    python scripts/01_pubmed_download.py ^
        --query ad_core_text ^
        --dry-run

    python scripts/01_pubmed_download.py ^
        --query ad_core_text ^
        --limit 100

    python scripts/01_pubmed_download.py ^
        --query ad_core_text ^
        --confirm-large-job

    python scripts/01_pubmed_download.py ^
        --group all ^
        --all ^
        --confirm-large-job
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "search_queries.yaml"
PUBMED_DIR = PROJECT_ROOT / "data" / "raw" / "pubmed"
LOG_FILE = PROJECT_ROOT / "logs" / "retrieval.log"
REPORT_FILE = PROJECT_ROOT / "reports" / "retrieval_report.csv"


# ---------------------------------------------------------------------------
# PubMed settings
# ---------------------------------------------------------------------------

ESEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

# Keep requests comfortably below the PubMed ESearch retrieval ceiling.
BATCH_SIZE = 9000

# Small pause between requests.
REQUEST_DELAY = 0.34

# Retry temporary network/server failures.
MAX_RETRIES = 5

# Initial wait before retrying.
RETRY_DELAY = 2.0

# Publication-date range used to divide large PubMed result sets.
FIRST_YEAR = 1800
LAST_YEAR = datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage01_pubmed")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = setup_logging()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "search_queries.yaml must contain a mapping."
        )

    return config


def validate_config(config: dict[str, Any]) -> None:
    queries = config.get("queries")
    execution = config.get("execution")

    if not isinstance(queries, dict):
        raise ValueError(
            "'queries' must be a mapping."
        )

    if not isinstance(execution, dict):
        raise ValueError(
            "'execution' must be a mapping."
        )

    if not isinstance(execution.get("mode"), str):
        raise ValueError(
            "execution.mode must be a string."
        )

    groups = (
        "primary_queries",
        "secondary_queries",
        "currency_queries",
        "audit_queries",
    )

    assigned_queries: dict[str, str] = {}

    for group_key in groups:
        query_ids = execution.get(group_key, [])

        if not isinstance(query_ids, list):
            raise ValueError(
                f"execution.{group_key} must be a list."
            )

        group_name = group_key.removesuffix("_queries")

        for query_id in query_ids:
            if not isinstance(query_id, str):
                raise ValueError(
                    f"execution.{group_key} contains "
                    "a non-string query ID."
                )

            if query_id not in queries:
                raise ValueError(
                    f"Query '{query_id}' is not defined "
                    "under queries."
                )

            if query_id in assigned_queries:
                raise ValueError(
                    f"Query '{query_id}' is assigned to both "
                    f"'{assigned_queries[query_id]}' and "
                    f"'{group_name}'."
                )

            assigned_queries[query_id] = group_name


def get_query_text(
    config: dict[str, Any],
    query_id: str,
) -> str:
    queries = config["queries"]

    if query_id not in queries:
        raise ValueError(
            f"Unknown query ID: {query_id}"
        )

    definition = queries[query_id]

    if isinstance(definition, str):
        return definition.strip()

    if isinstance(definition, dict):
        query = definition.get("query")

        if isinstance(query, str) and query.strip():
            return query.strip()

    raise ValueError(
        f"Query '{query_id}' does not contain a valid query."
    )


def get_execution_groups(
    config: dict[str, Any],
) -> dict[str, list[str]]:
    execution = config["execution"]

    return {
        "primary": list(
            execution.get("primary_queries", [])
        ),
        "secondary": list(
            execution.get("secondary_queries", [])
        ),
        "currency": list(
            execution.get("currency_queries", [])
        ),
        "audit": list(
            execution.get("audit_queries", [])
        ),
    }


def select_queries(
    config: dict[str, Any],
    query_id: str | None,
    group: str | None,
    run_all: bool,
) -> list[str]:

    if query_id and group:
        raise ValueError(
            "Use either --query or --group, not both."
        )

    if query_id:
        get_query_text(config, query_id)
        return [query_id]

    if not group:
        raise ValueError(
            "Specify --query QUERY_ID or --group GROUP."
        )

    if not run_all:
        raise ValueError(
            "--group requires --all."
        )

    groups = get_execution_groups(config)

    if group == "all":
        selected: list[str] = []

        for group_name in (
            "primary",
            "secondary",
            "currency",
            "audit",
        ):
            for query_id in groups[group_name]:
                if query_id not in selected:
                    selected.append(query_id)

        return selected

    if group not in groups:
        raise ValueError(
            f"Unknown group '{group}'."
        )

    return groups[group]


# ---------------------------------------------------------------------------
# PubMed request
# ---------------------------------------------------------------------------

def pubmed_request(
    query: str,
    retmax: int = 0,
    retstart: int = 0,
) -> dict[str, Any]:

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)

            response = requests.get(
                ESEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": retmax,
                    "retstart": retstart,
                },
                timeout=120,
            )

            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "PubMed returned an invalid JSON response."
                ) from error

            result = data.get("esearchresult")

            if not isinstance(result, dict):
                raise RuntimeError(
                    "PubMed returned an unexpected response."
                )

            return result

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
        ) as error:

            last_error = error

            status_code = getattr(
                getattr(error, "response", None),
                "status_code",
                None,
            )

            retryable = (
                status_code is None
                or status_code in {429, 500, 502, 503, 504}
            )

            if not retryable or attempt == MAX_RETRIES:
                raise

            wait_seconds = RETRY_DELAY * (2 ** (attempt - 1))

            LOGGER.warning(
                "PubMed request failed "
                "(attempt %d/%d, status=%s). "
                "Retrying in %.1f seconds.",
                attempt,
                MAX_RETRIES,
                status_code,
                wait_seconds,
            )

            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "PubMed request failed unexpectedly."
    )


def pubmed_count(query: str) -> int:
    result = pubmed_request(
        query,
        retmax=0,
    )

    try:
        return int(result.get("count", 0))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "PubMed returned an invalid result count."
        ) from error


def pubmed_ids(
    query: str,
    limit: int = BATCH_SIZE,
) -> list[str]:

    result = pubmed_request(
        query,
        retmax=limit,
        retstart=0,
    )

    ids = result.get("idlist", [])

    if not isinstance(ids, list):
        raise RuntimeError(
            "PubMed returned an invalid PMID list."
        )

    return [str(pmid) for pmid in ids]


# ---------------------------------------------------------------------------
# Date-range retrieval
# ---------------------------------------------------------------------------

def make_date_query(
    base_query: str,
    start_year: int,
    end_year: int,
) -> str:
    return (
        f"({base_query}) AND "
        f'("{start_year}/01/01"[Date - Publication] : '
        f'"{end_year}/12/31"[Date - Publication])'
    )


def retrieve_year_range(
    base_query: str,
    start_year: int,
    end_year: int,
) -> list[str]:

    query = make_date_query(
        base_query,
        start_year,
        end_year,
    )

    count = pubmed_count(query)

    LOGGER.info(
        "Date range %d-%d: %d records",
        start_year,
        end_year,
        count,
    )

    if count == 0:
        return []

    if count <= BATCH_SIZE:
        ids = pubmed_ids(
            query,
            limit=BATCH_SIZE,
        )

        if len(ids) != count:
            raise RuntimeError(
                f"PubMed retrieval mismatch for "
                f"{start_year}-{end_year}: "
                f"expected {count}, "
                f"received {len(ids)}."
            )

        return ids

    # If the range contains too many records, divide it.
    if start_year < end_year:
        middle = (start_year + end_year) // 2

        left_ids = retrieve_year_range(
            base_query,
            start_year,
            middle,
        )

        right_ids = retrieve_year_range(
            base_query,
            middle + 1,
            end_year,
        )

        return left_ids + right_ids

    # A single year contains more than BATCH_SIZE.
    return retrieve_single_year(
        base_query,
        start_year,
        count,
    )


def make_month_query(
    base_query: str,
    year: int,
    month: int,
) -> str:

    # Use the final day of the current month as the upper boundary.
    if month == 2:
        if (
            year % 400 == 0
            or (year % 4 == 0 and year % 100 != 0)
        ):
            last_day = 29
        else:
            last_day = 28
    elif month in {4, 6, 9, 11}:
        last_day = 30
    else:
        last_day = 31

    return (
        f"({base_query}) AND "
        f'("{year}/{month:02d}/01"[Date - Publication] : '
        f'"{year}/{month:02d}/{last_day:02d}"'
        f'[Date - Publication])'
    )


def retrieve_single_year(
    base_query: str,
    year: int,
    year_count: int,
) -> list[str]:

    LOGGER.info(
        "Year %d contains %d records; dividing by month.",
        year,
        year_count,
    )

    all_ids: list[str] = []

    for month in range(1, 13):
        query = make_month_query(
            base_query,
            year,
            month,
        )

        count = pubmed_count(query)

        if count == 0:
            continue

        LOGGER.info(
            "Month %d-%02d: %d records",
            year,
            month,
            count,
        )

        if count > BATCH_SIZE:
            raise RuntimeError(
                f"Month {year}-{month:02d} still contains "
                f"{count} records. "
                "The query is unusually concentrated in one "
                "month and requires further subdivision."
            )

        ids = pubmed_ids(
            query,
            limit=BATCH_SIZE,
        )

        if len(ids) != count:
            raise RuntimeError(
                f"PubMed retrieval mismatch for "
                f"{year}-{month:02d}: "
                f"expected {count}, "
                f"received {len(ids)}."
            )

        all_ids.extend(ids)

    return all_ids


# ---------------------------------------------------------------------------
# Complete query retrieval
# ---------------------------------------------------------------------------

def retrieve_complete_query(
    query_id: str,
    query_text: str,
) -> tuple[int, list[str]]:

    available = pubmed_count(query_text)

    print(
        f"{query_id}: PubMed count = {available:,}"
    )

    LOGGER.info(
        "%s: PubMed count = %d",
        query_id,
        available,
    )

    if available == 0:
        return 0, []

    pmids = retrieve_year_range(
        query_text,
        FIRST_YEAR,
        LAST_YEAR,
    )

    # Remove duplicates while preserving numeric order.
    unique_pmids = sorted(
        set(pmids),
        key=int,
    )

    if len(unique_pmids) != available:
        raise RuntimeError(
            f"Final PubMed count mismatch for {query_id}: "
            f"PubMed initially reported {available:,}, "
            f"but {len(unique_pmids):,} unique PMIDs "
            "were retrieved."
        )

    return available, unique_pmids


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def limited_test(
    query_text: str,
    limit: int,
) -> tuple[int, int]:

    available = pubmed_count(query_text)

    ids = pubmed_ids(
        query_text,
        limit=min(limit, BATCH_SIZE),
    )

    return available, len(ids)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_pmids(
    query_id: str,
    pmids: list[str],
) -> Path:

    PUBMED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PUBMED_DIR /
        f"{query_id}_pmids.txt"
    )

    temporary_file = output_file.with_suffix(
        ".tmp"
    )

    temporary_file.write_text(
        "\n".join(pmids) + "\n",
        encoding="utf-8",
    )

    temporary_file.replace(output_file)

    return output_file


def write_metadata(
    rows: list[dict[str, Any]],
) -> None:

    metadata_file = (
        PROJECT_ROOT /
        "metadata" /
        "pubmed.csv"
    )

    metadata_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = metadata_file.with_suffix(
        ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "pmid",
                "retrieved_at_utc",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_file.replace(metadata_file)


def write_report(
    rows: list[dict[str, Any]],
) -> None:

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = REPORT_FILE.with_suffix(
        ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "status",
                "pubmed_count",
                "retrieved_count",
                "unique_count",
                "retrieved_at_utc",
                "artifact",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_file.replace(REPORT_FILE)


# ---------------------------------------------------------------------------
# Display configured queries
# ---------------------------------------------------------------------------

def print_queries(
    config: dict[str, Any],
) -> None:

    groups = get_execution_groups(config)
    total = 0

    print()

    for group_name in (
        "primary",
        "secondary",
        "currency",
        "audit",
    ):

        query_ids = groups[group_name]

        print(
            f"{group_name}: {len(query_ids)} queries"
        )

        for query_id in query_ids:
            print(
                f"  - {query_id}"
            )

        total += len(query_ids)

    print()

    print(
        f"Total configured execution queries: {total}"
    )

    print()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Stage 01 - download PubMed PMID lists."
        )
    )

    parser.add_argument(
        "--query",
        help="Run one query ID.",
    )

    parser.add_argument(
        "--group",
        choices=[
            "primary",
            "secondary",
            "currency",
            "audit",
            "all",
        ],
        help="Run one execution group.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Confirm execution of the selected group.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Retrieve a small number of PMIDs for testing. "
            "No corpus files are written."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show PubMed result counts without downloading PMIDs.",
    )

    parser.add_argument(
        "--print-queries",
        action="store_true",
        help="Display configured execution queries.",
    )

    parser.add_argument(
        "--confirm-large-job",
        action="store_true",
        help="Confirm a complete PubMed retrieval.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config()
        validate_config(config)

        if args.print_queries:
            print_queries(config)
            return 0

        if args.limit is not None:

            if args.limit <= 0:
                raise ValueError(
                    "--limit must be greater than zero."
                )

            selected = select_queries(
                config,
                args.query,
                args.group,
                args.all,
            )

            if len(selected) != 1:
                raise ValueError(
                    "--limit can only be used with one query."
                )

            query_id = selected[0]

            query_text = get_query_text(
                config,
                query_id,
            )

            available, retrieved = limited_test(
                query_text,
                args.limit,
            )

            print(
                f"{query_id}: "
                f"available={available:,}, "
                f"retrieved_for_test={retrieved}"
            )

            print()

            print(
                "Limited test complete. "
                "No PMID artifact or retrieval report was written."
            )

            return 0

        selected_queries = select_queries(
            config,
            args.query,
            args.group,
            args.all,
        )

        if args.dry_run:

            for query_id in selected_queries:

                query_text = get_query_text(
                    config,
                    query_id,
                )

                count = pubmed_count(
                    query_text
                )

                print(
                    f"{query_id}: "
                    f"PubMed count = {count:,}"
                )

            return 0

        if not args.confirm_large_job:
            raise ValueError(
                "Complete retrieval requires "
                "--confirm-large-job."
            )

        LOGGER.info(
            "Stage 01 started."
        )

        LOGGER.info(
            "Selected queries: %d",
            len(selected_queries),
        )

        report_rows: list[dict[str, Any]] = []
        metadata_rows: list[dict[str, Any]] = []

        retrieved_at = datetime.now(
            timezone.utc
        ).isoformat()

        for query_id in selected_queries:

            query_text = get_query_text(
                config,
                query_id,
            )

            try:

                count, pmids = retrieve_complete_query(
                    query_id,
                    query_text,
                )

                artifact = write_pmids(
                    query_id,
                    pmids,
                )

                for pmid in pmids:

                    metadata_rows.append(
                        {
                            "query_id": query_id,
                            "pmid": pmid,
                            "retrieved_at_utc": retrieved_at,
                        }
                    )

                report_rows.append(
                    {
                        "query_id": query_id,
                        "status": "success",
                        "pubmed_count": count,
                        "retrieved_count": len(pmids),
                        "unique_count": len(set(pmids)),
                        "retrieved_at_utc": retrieved_at,
                        "artifact": str(
                            artifact.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    }
                )

                print(
                    f"{query_id}: "
                    f"retrieved={len(pmids):,}"
                )

            except Exception as error:

                LOGGER.exception(
                    "Query failed: %s",
                    query_id,
                )

                print(
                    f"ERROR: {query_id}: {error}",
                    file=sys.stderr,
                )

                return 1

        # Write metadata only after all selected queries succeed.
        write_metadata(
            metadata_rows
        )

        # The report is produced for the complete corpus run.
        if args.group == "all" and args.all:
            write_report(
                report_rows
            )

        LOGGER.info(
            "Stage 01 completed successfully."
        )

        print()

        print(
            "Stage 01 PubMed retrieval completed successfully."
        )

        if args.group == "all" and args.all:
            print(
                f"Retrieval report: "
                f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
            )

        return 0

    except Exception as error:

        LOGGER.exception(
            "Stage 01 failed."
        )

        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())