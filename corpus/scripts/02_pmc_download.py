#!/usr/bin/env python3

"""
Stage 02 - PMC retrieval via the PMC Article Dataset Cloud Service.

Purpose
-------
Retrieves PMC article versions corresponding to the PubMed PMIDs produced by
Stage 01, using the current PMC Article Dataset Cloud Service.

Supported modes
---------------
Normal run:
    python scripts/02_pmc_download.py

Dry run:
    python scripts/02_pmc_download.py --dry-run

Small test run:
    python scripts/02_pmc_download.py --limit 10

Targeted retry:
    python scripts/02_pmc_download.py --retry-pmcids PMC123 PMC456

Permanent repair mode:
    python scripts/02_pmc_download.py --repair-incomplete

Finalize authoritative manifest:
    python scripts/02_pmc_download.py --finalize

Important
---------
- Normal retrieval does not overwrite an authoritative manifest until the
  complete Stage 02 accounting succeeds.
- --repair-incomplete repairs only incomplete local article-version
  directories. It does not rerun PubMed -> PMC matching.
- --repair-incomplete does not write metadata/pmc.csv.
- --finalize is network-free and validates the local corpus plus the
  accounting evidence recorded by the completed/full Stage 02 run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "pmc"
METADATA = ROOT / "metadata"
LOGS = ROOT / "logs"
PUBMED_RAW = ROOT / "data" / "raw" / "pubmed"

MANIFEST = METADATA / "pmc.csv"
LOG_FILE = LOGS / "retrieval.log"


# ---------------------------------------------------------------------------
# Remote services
# ---------------------------------------------------------------------------

NCBI_ESEARCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com"


# ---------------------------------------------------------------------------
# Runtime parameters
# ---------------------------------------------------------------------------

ESEARCH_BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 2.0
REQUEST_DELAY = 0.35


MANIFEST_FIELDS = [
    "pmid",
    "pmcid",
    "version",
    "doi",
    "title",
    "citation",
    "is_pmc_openaccess",
    "is_manuscript",
    "license_code",
    "is_retracted",
    "json_path",
    "xml_path",
    "status",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger() -> logging.Logger:
    """Create the shared Stage 02 logger."""

    LOGS.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage02_pmc")
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def chunk(values: list[str], size: int) -> Iterable[list[str]]:
    """Yield values in fixed-size chunks."""

    for index in range(0, len(values), size):
        yield values[index : index + size]


def normalize_download_url(url: str) -> str:
    """
    Normalize S3 URLs.

    Supports:
        https://...
        s3://bucket/key
    """

    if url.startswith("s3://"):
        remainder = url[len("s3://") :]
        bucket, _, key = remainder.partition("/")

        if not bucket or not key:
            raise ValueError(f"Invalid S3 URL: {url}")

        return f"https://{bucket}.s3.amazonaws.com/{key}"

    return url


def version_number_from_prefix(prefix: str) -> str:
    """
    Convert:
        PMC12345.1/
    into:
        1
    """

    clean = prefix.rstrip("/")

    match = re.search(r"\.(\d+)$", clean)

    if not match:
        raise ValueError(f"Cannot determine version from prefix: {prefix}")

    return match.group(1)


def article_directory(pmcid: str, version: str) -> Path:
    """Return the local directory for one PMC article version."""

    return RAW / f"{pmcid}.{version}"


def article_paths(
    pmcid: str,
    version: str,
) -> tuple[Path, Path, Path]:
    """
    Return:
        article directory,
        JSON path,
        XML path
    """

    directory = article_directory(pmcid, version)

    json_path = directory / f"{pmcid}.{version}.json"
    xml_path = directory / f"{pmcid}.{version}.xml"

    return directory, json_path, xml_path


# ---------------------------------------------------------------------------
# PubMed input
# ---------------------------------------------------------------------------


def read_pubmed_pmids() -> list[str]:
    """
    Read all Stage 01 PubMed PMID artifacts.

    Stage 01 produces files such as:
        *_pmids.txt

    Duplicates are removed while preserving deterministic ordering.
    """

    if not PUBMED_RAW.exists():
        raise FileNotFoundError(
            f"PubMed raw directory does not exist: {PUBMED_RAW}"
        )

    files = sorted(PUBMED_RAW.glob("*_pmids.txt"))

    if not files:
        raise FileNotFoundError(
            f"No Stage 01 PMID artifacts found in: {PUBMED_RAW}"
        )

    pmids: set[str] = set()

    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()

            if value and value.isdigit():
                pmids.add(value)

    result = sorted(pmids, key=lambda value: int(value))

    if not result:
        raise RuntimeError("Stage 01 PMID artifacts contained no PMIDs.")

    return result


# ---------------------------------------------------------------------------
# NCBI requests
# ---------------------------------------------------------------------------


def ncbi_request(
    params: dict[str, str],
    logger: logging.Logger,
) -> str:
    """
    Perform an NCBI E-utilities request with retries.

    ConnectionError is explicitly handled because NCBI/remote endpoints can
    terminate connections with errors such as:
        Remote end closed connection without response
    """

    query = "&".join(
        f"{key}={value}"
        for key, value in params.items()
    )

    url = f"{NCBI_ESEARCH}?{query}"

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0",
                },
            )

            with urlopen(request, timeout=120) as response:
                return response.read().decode("utf-8")

        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_error = exc

            logger.warning(
                "NCBI request failed on attempt %d/%d: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"NCBI request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ---------------------------------------------------------------------------
# PubMed -> PMC mapping
# ---------------------------------------------------------------------------


def find_pmcids(
    pmids: list[str],
    logger: logging.Logger,
) -> list[str]:
    """
    Map PubMed PMIDs to PMC IDs using NCBI PMC ESearch.

    Batches are intentionally small enough to remain below the ESearch
    10,000-record limit.
    """

    pmcids: set[str] = set()

    total_batches = (
        (len(pmids) + ESEARCH_BATCH_SIZE - 1)
        // ESEARCH_BATCH_SIZE
    )

    for batch_index, batch_pmids in enumerate(
        chunk(pmids, ESEARCH_BATCH_SIZE),
        start=1,
    ):
        logger.info(
            "Mapping PubMed batch %d/%d (%d PMIDs) to PMC records.",
            batch_index,
            total_batches,
            len(batch_pmids),
        )

        params = {
            "db": "pmc",
            "term": (
                "("
                + " OR ".join(
                    f"{pmid}[PMID]" for pmid in batch_pmids
                )
                + ")"
            ),
            "retmode": "json",
            "retmax": str(ESEARCH_BATCH_SIZE),
        }

        response_text = ncbi_request(params, logger)

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "NCBI returned invalid JSON while mapping PMID -> PMC."
            ) from exc

        ids = (
            payload
            .get("esearchresult", {})
            .get("idlist", [])
        )

        for value in ids:
            if str(value).isdigit():
                pmcids.add(f"PMC{value}")

        time.sleep(REQUEST_DELAY)

    return sorted(
        pmcids,
        key=lambda value: int(value[3:]),
    )


# ---------------------------------------------------------------------------
# PMC Article Dataset Cloud Service
# ---------------------------------------------------------------------------


def get_article_versions(
    pmcid_number: str,
    logger: logging.Logger,
) -> list[str]:
    """
    List currently available article-version prefixes for one PMCID.

    Example:
        ["PMC12345.1/", "PMC12345.2/"]
    """

    url = (
        f"{PMC_S3}/"
        f"?list-type=2"
        f"&prefix=PMC{pmcid_number}."
        f"&delimiter=/"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0",
                },
            )

            with urlopen(request, timeout=120) as response:
                xml_text = response.read().decode("utf-8")

            prefixes = re.findall(
                r"<CommonPrefixes><Prefix>(PMC\d+\.\d+/)</Prefix>",
                xml_text,
            )

            return sorted(
                set(prefixes),
                key=lambda value: int(
                    version_number_from_prefix(value)
                ),
            )

        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_error = exc

            logger.warning(
                "PMC version listing failed for PMC%s "
                "on attempt %d/%d: %s",
                pmcid_number,
                attempt,
                MAX_RETRIES,
                exc,
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"PMC version listing failed for PMC{pmcid_number} "
        f"after {MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# File download and checksum
# ---------------------------------------------------------------------------


def download_url(
    url: str,
    destination: Path,
) -> bytes:
    """
    Download a file atomically.

    The final destination is replaced only after the complete response has
    been received successfully.

    This prevents partial files from being mistaken for valid artifacts.
    """

    url = normalize_download_url(url)

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0",
                },
            )

            with urlopen(request, timeout=120) as response:
                data = response.read()

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary.write_bytes(data)

            temporary.replace(destination)

            return data

        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_error = exc

            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY * (2 ** (attempt - 1))
                )

    raise RuntimeError(
        f"Download failed after {MAX_RETRIES} attempts: "
        f"{url}: {last_error}"
    )


def md5_matches(
    data: bytes,
    expected_md5: str,
) -> bool:
    """Check MD5 of bytes against expected checksum."""

    actual = hashlib.md5(data).hexdigest()

    return actual.lower() == expected_md5.lower()


def md5_file_matches(
    path: Path,
    expected_md5: str,
) -> bool:
    """Check MD5 of an existing file."""

    digest = hashlib.md5()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest().lower() == expected_md5.lower()


# ---------------------------------------------------------------------------
# PMC metadata
# ---------------------------------------------------------------------------


def load_json_metadata(
    pmcid: str,
    version: str,
    prefix: str,
    logger: logging.Logger,
) -> dict:
    """
    Download and parse the article-version JSON metadata.
    """

    _, json_path, _ = article_paths(
        pmcid,
        version,
    )

    url = (
        f"{PMC_S3}/"
        f"{prefix.rstrip('/')}/"
        f"{pmcid}.{version}.json"
    )

    data = download_url(
        url,
        json_path,
    )

    try:
        metadata = json.loads(
            data.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        try:
            if json_path.exists():
                json_path.unlink()
        except OSError:
            pass

        raise RuntimeError(
            f"Invalid PMC metadata JSON for "
            f"{pmcid}.{version}"
        ) from exc

    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"PMC metadata is not a JSON object for "
            f"{pmcid}.{version}"
        )

    return metadata


def load_existing_json_metadata(
    json_path: Path,
) -> dict | None:
    """Load existing local metadata, returning None if invalid."""

    if not json_path.exists():
        return None

    try:
        payload = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def expected_xml_md5(
    metadata: dict,
) -> str | None:
    """
    Extract the XML MD5 checksum from PMC metadata.

    The current PMC Article Dataset metadata exposes XML URLs containing
    the checksum as an md5 query parameter.
    """

    xml_url = metadata.get("xml_url")

    if not isinstance(xml_url, str):
        return None

    match = re.search(
        r"(?:[?&])md5=([0-9a-fA-F]{32})",
        xml_url,
    )

    if match:
        return match.group(1).lower()

    return None


# ---------------------------------------------------------------------------
# XML retrieval
# ---------------------------------------------------------------------------


def retrieve_xml(
    pmcid: str,
    version: str,
    prefix: str,
    metadata: dict,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """
    Retrieve and verify the article XML.

    Returns:
        (success, status)
    """

    _, _, destination = article_paths(
        pmcid,
        version,
    )

    xml_url = metadata.get("xml_url")

    if not isinstance(xml_url, str) or not xml_url:
        return False, "missing_xml_url"

    expected_md5 = expected_xml_md5(metadata)

    if not expected_md5:
        return False, "missing_xml_md5"

    # Existing XML can be reused if it is already correct.
    if destination.exists():
        if md5_file_matches(
            destination,
            expected_md5,
        ):
            return True, "already_verified"

        logger.warning(
            "%s.%s existing XML failed MD5 verification; "
            "redownloading.",
            pmcid,
            version,
        )

        try:
            destination.unlink()
        except OSError:
            pass

    # Do not create the directory before the download.
    # download_url creates it only after a successful download.
    data = download_url(
        xml_url,
        destination,
    )

    if not md5_matches(
        data,
        expected_md5,
    ):
        try:
            destination.unlink()
        except OSError:
            pass

        return False, "xml_md5_mismatch"

    return True, "downloaded_verified"


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------


def verify_existing_article_version(
    pmcid: str,
    version: str,
) -> tuple[bool, dict | None, str]:
    """
    Fully verify one local article-version.

    This is intentionally stricter than the repair-candidate scan.
    Finalization uses this function as the integrity gate.
    """

    _, json_path, xml_path = article_paths(
        pmcid,
        version,
    )

    if not json_path.exists():
        return False, None, "missing_json"

    if not xml_path.exists():
        return False, None, "missing_xml"

    metadata = load_existing_json_metadata(
        json_path
    )

    if metadata is None:
        return False, None, "invalid_json"

    if not bool(
        metadata.get("is_pmc_openaccess", False)
    ):
        return False, metadata, "not_pmc_openaccess"

    expected_md5 = expected_xml_md5(metadata)

    if not expected_md5:
        return False, metadata, "missing_xml_md5"

    if not md5_file_matches(
        xml_path,
        expected_md5,
    ):
        return False, metadata, "xml_md5_mismatch"

    return True, metadata, "already_verified"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def manifest_row_from_metadata(
    pmid: str,
    pmcid: str,
    version: str,
    metadata: dict,
    status: str,
) -> dict[str, str]:
    """Build one authoritative manifest row."""

    _, json_path, xml_path = article_paths(
        pmcid,
        version,
    )

    return {
        "pmid": str(metadata.get("pmid") or pmid),
        "pmcid": pmcid,
        "version": str(version),
        "doi": str(metadata.get("doi", "")),
        "title": str(metadata.get("title", "")),
        "citation": str(metadata.get("citation", "")),
        "is_pmc_openaccess": str(
            bool(
                metadata.get(
                    "is_pmc_openaccess",
                    False,
                )
            )
        ),
        "is_manuscript": str(
            bool(
                metadata.get(
                    "is_manuscript",
                    False,
                )
            )
        ),
        "license_code": str(
            metadata.get(
                "license_code",
                "",
            )
        ),
        "is_retracted": str(
            bool(
                metadata.get(
                    "is_retracted",
                    False,
                )
            )
        ),
        "json_path": str(
            json_path.relative_to(ROOT)
        ),
        "xml_path": str(
            xml_path.relative_to(ROOT)
        ),
        "status": status,
    }


def unavailable_manifest_row(
    pmcid: str,
) -> dict[str, str]:
    """Build a manifest row for an explicitly unavailable PMCID."""

    return {
        "pmid": "",
        "pmcid": pmcid,
        "version": "",
        "doi": "",
        "title": "",
        "citation": "",
        "is_pmc_openaccess": "",
        "is_manuscript": "",
        "license_code": "",
        "is_retracted": "",
        "json_path": "",
        "xml_path": "",
        "status": "unavailable_current_dataset",
    }


def write_manifest(
    rows: list[dict[str, str]],
) -> None:
    """Atomically write metadata/pmc.csv."""

    METADATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = MANIFEST.with_suffix(
        ".csv.part"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    temporary.replace(MANIFEST)


# ---------------------------------------------------------------------------
# Normal retrieval
# ---------------------------------------------------------------------------


def run(
    logger: logging.Logger,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """
    Execute the normal Stage 02 retrieval.

    The authoritative manifest is written only if the complete run has
    successful retrieval accounting.
    """

    logger.info("Stage 02 started.")

    pmids = read_pubmed_pmids()

    logger.info(
        "Stage 01 PubMed PMIDs loaded: %d",
        len(pmids),
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero.")

        pmids = pmids[:limit]

        logger.info(
            "Test limit active: processing first %d PMIDs.",
            len(pmids),
        )

    if dry_run:
        logger.info(
            "Dry run: PubMed -> PMC matching will be performed, "
            "but no article files will be downloaded."
        )

    pmcids = find_pmcids(
        pmids,
        logger,
    )

    logger.info(
        "PMC Open Access records matched: %d",
        len(pmcids),
    )

    if dry_run:
        logger.info(
            "Dry run complete. No authoritative manifest written."
        )
        return 0

    failures = 0
    unavailable = 0
    successful_versions = 0
    manifest_rows: list[dict[str, str]] = []

    total = len(pmcids)

    for index, pmcid in enumerate(
        pmcids,
        start=1,
    ):
        logger.info(
            "Processing PMC record %d/%d: %s",
            index,
            total,
            pmcid,
        )

        pmcid_number = pmcid[3:]

        try:
            versions = get_article_versions(
                pmcid_number,
                logger,
            )
        except Exception as exc:
            failures += 1

            logger.error(
                "%s failed: %s",
                pmcid,
                exc,
            )

            continue

        if not versions:
            unavailable += 1

            logger.warning(
                "%s: no article version found; "
                "recording unavailable_current_dataset",
                pmcid,
            )

            manifest_rows.append(
                unavailable_manifest_row(pmcid)
            )

            continue

        for prefix in versions:
            version = version_number_from_prefix(
                prefix
            )

            _, json_path, _ = article_paths(
                pmcid,
                version,
            )

            metadata = load_existing_json_metadata(
                json_path
            )

            json_status = "already_verified"

            try:
                if metadata is None:
                    metadata = load_json_metadata(
                        pmcid,
                        version,
                        prefix,
                        logger,
                    )
                    json_status = "downloaded"

                if not bool(
                    metadata.get(
                        "is_pmc_openaccess",
                        False,
                    )
                ):
                    logger.warning(
                        "%s.%s: article version is not "
                        "marked PMC Open Access",
                        pmcid,
                        version,
                    )

                    continue

                xml_ok, xml_status = retrieve_xml(
                    pmcid,
                    version,
                    prefix,
                    metadata,
                    logger,
                )

                if not xml_ok:
                    failures += 1

                    logger.error(
                        "%s.%s: XML retrieval failed: %s",
                        pmcid,
                        version,
                        xml_status,
                    )

                    continue

                valid, verified_metadata, verify_status = (
                    verify_existing_article_version(
                        pmcid,
                        version,
                    )
                )

                if not valid:
                    failures += 1

                    logger.error(
                        "%s.%s failed final local verification: %s",
                        pmcid,
                        version,
                        verify_status,
                    )

                    continue

                final_status = (
                    "already_verified"
                    if (
                        json_status == "already_verified"
                        and xml_status == "already_verified"
                    )
                    else "downloaded_verified"
                )

                manifest_rows.append(
                    manifest_row_from_metadata(
                        pmid="",
                        pmcid=pmcid,
                        version=version,
                        metadata=verified_metadata
                        or metadata,
                        status=final_status,
                    )
                )

                successful_versions += 1

            except Exception as exc:
                failures += 1

                logger.error(
                    "%s.%s failed: %s",
                    pmcid,
                    version,
                    exc,
                )

        time.sleep(REQUEST_DELAY)

    if failures:
        logger.error(
            "Stage 02 failed: %d retrieval failures occurred.",
            failures,
        )

        logger.error(
            "No completed PMC manifest was written."
        )

        return 1

    logger.info(
        "Stage 02 retrieval completed successfully."
    )

    logger.info(
        "Successful article versions: %d",
        successful_versions,
    )

    logger.info(
        "Unavailable current-dataset PMC records: %d",
        unavailable,
    )

    # For a limited/test run, do not write an authoritative manifest.
    if limit is not None:
        logger.info(
            "Limited test run completed. "
            "No authoritative PMC manifest written."
        )

        return 0

    write_manifest(
        manifest_rows
    )

    logger.info(
        "Authoritative PMC manifest written: %s",
        MANIFEST,
    )

    return 0


# ---------------------------------------------------------------------------
# Targeted retry
# ---------------------------------------------------------------------------


def retry_pmcids(
    pmcids: list[str],
    logger: logging.Logger,
) -> int:
    """
    Retry explicitly supplied PMCIDs.

    This mode does not rerun PubMed -> PMC matching and does not write the
    authoritative manifest.
    """

    if not pmcids:
        raise ValueError(
            "--retry-pmcids requires at least one PMCID."
        )

    normalized: list[str] = []

    for value in pmcids:
        value = value.strip().upper()

        if not re.fullmatch(
            r"PMC\d+",
            value,
        ):
            raise ValueError(
                f"Invalid PMCID: {value}"
            )

        normalized.append(value)

    normalized = sorted(
        set(normalized),
        key=lambda value: int(value[3:]),
    )

    logger.info(
        "Targeted retry started for %d PMCIDs.",
        len(normalized),
    )

    failures = 0
    repaired = 0
    unavailable = 0

    for index, pmcid in enumerate(
        normalized,
        start=1,
    ):
        logger.info(
            "Targeted retry %d/%d: %s",
            index,
            len(normalized),
            pmcid,
        )

        try:
            versions = get_article_versions(
                pmcid[3:],
                logger,
            )
        except Exception as exc:
            failures += 1

            logger.error(
                "%s targeted retry failed: %s",
                pmcid,
                exc,
            )

            continue

        if not versions:
            unavailable += 1

            logger.warning(
                "%s remains unavailable in the current "
                "PMC dataset.",
                pmcid,
            )

            continue

        for prefix in versions:
            version = version_number_from_prefix(
                prefix
            )

            try:
                valid, metadata, status = (
                    verify_existing_article_version(
                        pmcid,
                        version,
                    )
                )

                if valid:
                    logger.info(
                        "%s.%s already verified.",
                        pmcid,
                        version,
                    )

                    repaired += 1
                    continue

                metadata = load_json_metadata(
                    pmcid,
                    version,
                    prefix,
                    logger,
                )

                if not bool(
                    metadata.get(
                        "is_pmc_openaccess",
                        False,
                    )
                ):
                    logger.warning(
                        "%s.%s is not marked PMC Open Access.",
                        pmcid,
                        version,
                    )

                    continue

                xml_ok, xml_status = retrieve_xml(
                    pmcid,
                    version,
                    prefix,
                    metadata,
                    logger,
                )

                if not xml_ok:
                    failures += 1

                    logger.error(
                        "%s.%s targeted retry XML failure: %s",
                        pmcid,
                        version,
                        xml_status,
                    )

                    continue

                valid, _, verify_status = (
                    verify_existing_article_version(
                        pmcid,
                        version,
                    )
                )

                if not valid:
                    failures += 1

                    logger.error(
                        "%s.%s targeted retry verification failed: %s",
                        pmcid,
                        version,
                        verify_status,
                    )

                    continue

                repaired += 1

                logger.info(
                    "%s.%s targeted retry recovered successfully.",
                    pmcid,
                    version,
                )

            except Exception as exc:
                failures += 1

                logger.error(
                    "%s.%s targeted retry failed: %s",
                    pmcid,
                    version,
                    exc,
                )

        time.sleep(REQUEST_DELAY)

    logger.info(
        "Targeted retry summary: recovered_or_verified=%d, "
        "unavailable=%d, failures=%d",
        repaired,
        unavailable,
        failures,
    )

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Repair incomplete local artifacts
# ---------------------------------------------------------------------------


def local_version_needs_repair(
    pmcid: str,
    version: str,
) -> bool:
    """
    Identify structurally incomplete local article-version directories.

    This intentionally performs a lightweight scan:
        - missing JSON
        - missing XML
        - leftover .part file

    Full XML MD5 verification is intentionally deferred to --finalize.
    """

    directory, json_path, xml_path = article_paths(
        pmcid,
        version,
    )

    if not directory.exists():
        return True

    if not json_path.exists():
        return True

    if not xml_path.exists():
        return True

    if any(
        directory.glob("*.part")
    ):
        return True

    return False


def discover_local_article_versions() -> dict[
    str,
    list[tuple[str, Path]],
]:
    """
    Discover local article-version directories.

    Returns:
        {
            "PMC123": [
                ("1", Path(...)),
                ("2", Path(...)),
            ]
        }
    """

    result: dict[
        str,
        list[tuple[str, Path]],
    ] = {}

    if not RAW.exists():
        return result

    pattern = re.compile(
        r"^(PMC\d+)\.(\d+)$"
    )

    for directory in RAW.iterdir():
        if not directory.is_dir():
            continue

        match = pattern.fullmatch(
            directory.name
        )

        if not match:
            continue

        pmcid = match.group(1)
        version = match.group(2)

        result.setdefault(
            pmcid,
            [],
        ).append(
            (version, directory)
        )

    for pmcid in result:
        result[pmcid].sort(
            key=lambda item: int(item[0])
        )

    return result


def discover_incomplete_local_versions() -> list[
    tuple[str, str, Path]
]:
    """
    Find structurally incomplete local article-version directories.

    No network access occurs here.
    """

    discovered = discover_local_article_versions()

    candidates: list[
        tuple[str, str, Path]
    ] = []

    for pmcid, versions in discovered.items():
        for version, directory in versions:
            if local_version_needs_repair(
                pmcid,
                version,
            ):
                candidates.append(
                    (
                        pmcid,
                        version,
                        directory,
                    )
                )

    candidates.sort(
        key=lambda item: (
            int(item[0][3:]),
            int(item[1]),
        )
    )

    return candidates


def repair_article_version(
    pmcid: str,
    version: str,
    prefix: str,
    logger: logging.Logger,
) -> str:
    """
    Repair one local article-version.

    Returns a machine-readable status.
    """

    directory, json_path, xml_path = article_paths(
        pmcid,
        version,
    )

    metadata = load_existing_json_metadata(
        json_path
    )

    # Existing JSON is reusable only when it is valid and represents
    # a PMC Open Access article version.
    if (
        metadata is None
        or not bool(
            metadata.get(
                "is_pmc_openaccess",
                False,
            )
        )
        or not expected_xml_md5(metadata)
    ):
        metadata = load_json_metadata(
            pmcid,
            version,
            prefix,
            logger,
        )

    if not bool(
        metadata.get(
            "is_pmc_openaccess",
            False,
        )
    ):
        logger.warning(
            "%s.%s: current remote metadata is not marked "
            "PMC Open Access.",
            pmcid,
            version,
        )

        return "not_pmc_openaccess"

    xml_ok, xml_status = retrieve_xml(
        pmcid,
        version,
        prefix,
        metadata,
        logger,
    )

    if not xml_ok:
        return xml_status

    valid, _, verify_status = (
        verify_existing_article_version(
            pmcid,
            version,
        )
    )

    if not valid:
        return verify_status

    if (
        json_path.exists()
        and xml_path.exists()
        and directory.exists()
    ):
        return "repaired"

    return "verification_failed"


def repair_incomplete(
    logger: logging.Logger,
) -> int:
    """
    Permanently repair incomplete local article-version directories.

    Important:
    - Does not rerun Stage 01.
    - Does not rerun PubMed -> PMC matching.
    - Does not write metadata/pmc.csv.
    - Queries the current PMC Cloud dataset only for PMCID(s) that have
      incomplete local versions.
    """

    logger.info(
        "Stage 02 permanent incomplete-artifact repair started."
    )

    candidates = discover_incomplete_local_versions()

    logger.info(
        "Incomplete local article-version candidates: %d",
        len(candidates),
    )

    if not candidates:
        logger.info(
            "No structurally incomplete local article-version "
            "directories were found."
        )

        logger.info(
            "Repair mode completed successfully."
        )

        return 0

    # Group candidates by PMCID so the current PMC version listing is
    # performed once per PMCID rather than once per version.
    grouped: dict[
        str,
        list[tuple[str, Path]],
    ] = {}

    for pmcid, version, directory in candidates:
        grouped.setdefault(
            pmcid,
            [],
        ).append(
            (version, directory)
        )

    logger.info(
        "PMCIDs requiring remote repair lookup: %d",
        len(grouped),
    )

    repaired = 0
    already_current = 0
    stale = 0
    not_open_access = 0
    failures = 0

    total_candidates = len(candidates)
    processed_candidates = 0

    for pmcid in sorted(
        grouped,
        key=lambda value: int(value[3:]),
    ):
        logger.info(
            "Repair lookup for %s.",
            pmcid,
        )

        try:
            remote_prefixes = get_article_versions(
                pmcid[3:],
                logger,
            )
        except Exception as exc:
            failures += len(
                grouped[pmcid]
            )

            logger.error(
                "%s repair lookup failed: %s",
                pmcid,
                exc,
            )

            continue

        remote_by_version = {
            version_number_from_prefix(prefix): prefix
            for prefix in remote_prefixes
        }

        for version, directory in sorted(
            grouped[pmcid],
            key=lambda item: int(item[0]),
        ):
            processed_candidates += 1

            prefix = remote_by_version.get(
                version
            )

            if prefix is None:
                stale += 1

                logger.warning(
                    "%s.%s is a stale local incomplete artifact: "
                    "the version is no longer present in the current "
                    "PMC Article Dataset.",
                    pmcid,
                    version,
                )

                continue

            logger.info(
                "Repairing incomplete artifact "
                "%d/%d: %s.%s",
                processed_candidates,
                total_candidates,
                pmcid,
                version,
            )

            try:
                status = repair_article_version(
                    pmcid,
                    version,
                    prefix,
                    logger,
                )

                if status == "repaired":
                    repaired += 1

                    logger.info(
                        "%s.%s repaired and fully verified.",
                        pmcid,
                        version,
                    )

                elif status == "already_verified":
                    already_current += 1

                elif status == "not_pmc_openaccess":
                    not_open_access += 1

                    logger.warning(
                        "%s.%s could not be repaired because the "
                        "current remote metadata is not PMC Open Access.",
                        pmcid,
                        version,
                    )

                else:
                    failures += 1

                    logger.error(
                        "%s.%s repair failed: %s",
                        pmcid,
                        version,
                        status,
                    )

            except Exception as exc:
                failures += 1

                logger.error(
                    "%s.%s repair failed: %s",
                    pmcid,
                    version,
                    exc,
                )

        time.sleep(REQUEST_DELAY)

    logger.info(
        "Permanent repair summary: candidates=%d, repaired=%d, "
        "already_current=%d, stale=%d, not_open_access=%d, failures=%d",
        total_candidates,
        repaired,
        already_current,
        stale,
        not_open_access,
        failures,
    )

    if stale:
        logger.warning(
            "%d incomplete local artifact(s) were not repaired "
            "because their versions are no longer present in the "
            "current PMC dataset.",
            stale,
        )

    if failures:
        logger.error(
            "Permanent repair completed with failures."
        )

        logger.error(
            "Do not finalize the Stage 02 corpus yet."
        )

        return 1

    logger.info(
        "Permanent repair completed without retrieval failures."
    )

    logger.info(
        "Next step is full Stage 02 finalization after reviewing "
        "the repair summary."
    )

    return 0


# ---------------------------------------------------------------------------
# Full-run log parsing for finalization
# ---------------------------------------------------------------------------


def parse_stage02_run_blocks() -> list[list[str]]:
    """
    Split the shared retrieval log into Stage 02 run blocks.

    Only blocks beginning with:
        Stage 02 started.

    are considered normal full-run candidates.
    """

    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"Stage 02 retrieval log does not exist: {LOG_FILE}"
        )

    lines = LOG_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    blocks: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        if "Stage 02 started." in line:
            if current:
                blocks.append(current)

            current = [line]
            continue

        if current is not None:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks


def extract_full_run_evidence(
    logger: logging.Logger,
) -> tuple[
    int,
    set[str],
    set[str],
    set[str],
    set[str],
]:
    """
    Recover accounting evidence from the largest normal Stage 02 run.

    Returns:
        matched_count,
        processed_pmcids,
        unavailable_pmcids,
        non_oa_versions,
        failed_pmcids
    """

    blocks = parse_stage02_run_blocks()

    candidates: list[
        tuple[int, list[str]]
    ] = []

    matched_pattern = re.compile(
        r"PMC Open Access records matched:\s*(\d+)"
    )

    for block in blocks:
        matched_count = None

        for line in block:
            match = matched_pattern.search(line)

            if match:
                matched_count = int(
                    match.group(1)
                )

                break

        if matched_count is not None:
            candidates.append(
                (
                    matched_count,
                    block,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No Stage 02 run with a recorded "
            "'PMC Open Access records matched' count was found."
        )

    # Select the largest full corpus run. This avoids accidentally selecting
    # a limited test run.
    matched_count, selected_block = max(
        candidates,
        key=lambda item: item[0],
    )

    processing_pattern = re.compile(
        r"Processing PMC record \d+/\d+:\s*(PMC\d+)"
    )

    # Match the core statement only. The same fact - PMC holds no article
    # version for this PMCID - has been written in three different wordings
    # by three versions of this script:
    #
    #   "PMC123: no article version found"
    #   "PMC123: no article version found during targeted retry"
    #   "PMC123: no article version found; recording unavailable_current_dataset"
    #
    # The corpus was retrieved by the earliest of those, so requiring the
    # newest wording made the historical evidence unreadable and left every
    # such PMCID permanently unaccountable. Log lines already written cannot
    # be reworded, so the reader accepts the statement and ignores the
    # trailing prose. This does not weaken the evidence standard: all three
    # wordings assert exactly the same finding.
    unavailable_pattern = re.compile(
        r"(PMC\d+): no article version found"
    )

    non_oa_pattern = re.compile(
        r"(PMC\d+\.\d+): article version is not "
        r"marked PMC Open Access"
    )

    failed_pattern = re.compile(
        r"(PMC\d+) failed:"
    )

    processed: set[str] = set()
    unavailable: set[str] = set()
    non_oa_versions: set[str] = set()
    failed_pmcids: set[str] = set()

    for line in selected_block:
        processing_match = processing_pattern.search(line)

        if processing_match:
            processed.add(
                processing_match.group(1)
            )

        unavailable_match = unavailable_pattern.search(line)

        if unavailable_match:
            unavailable.add(
                unavailable_match.group(1)
            )

        non_oa_match = non_oa_pattern.search(line)

        if non_oa_match:
            non_oa_versions.add(
                non_oa_match.group(1)
            )

        failed_match = failed_pattern.search(line)

        if failed_match:
            failed_pmcids.add(
                failed_match.group(1)
            )

    logger.info(
        "Selected Stage 02 run for finalization: "
        "%d matched records.",
        matched_count,
    )

    logger.info(
        "Recovered %d processed PMCIDs from the selected run.",
        len(processed),
    )

    logger.info(
        "Selected run explicitly recorded %d unavailable PMCIDs.",
        len(unavailable),
    )

    logger.info(
        "Selected run explicitly recorded %d non-OA versions.",
        len(non_oa_versions),
    )

    logger.info(
        "Selected run recorded %d failed PMCIDs.",
        len(failed_pmcids),
    )

    return (
        matched_count,
        processed,
        unavailable,
        non_oa_versions,
        failed_pmcids,
    )


# ---------------------------------------------------------------------------
# Repair-run log parsing for finalization
# ---------------------------------------------------------------------------


def parse_latest_repair_block() -> list[str]:
    """Return the most recent permanent repair block from the shared log."""

    if not LOG_FILE.exists():
        return []

    lines = LOG_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    start_indices = [
        index
        for index, line in enumerate(lines)
        if "Stage 02 permanent incomplete-artifact repair started." in line
    ]

    if not start_indices:
        return []

    start = start_indices[-1]
    end = len(lines)

    # A later top-level Stage 02 operation marks the end of the repair block.
    stage_marker = re.compile(r"\| (?:INFO|WARNING|ERROR) \| Stage 02 ")

    for index in range(start + 1, len(lines)):
        if stage_marker.search(lines[index]):
            end = index
            break

    return lines[start:end]


def extract_repair_evidence(
    logger: logging.Logger,
) -> tuple[set[str], set[str]]:
    """Recover non-OA and stale version IDs from the latest repair run."""

    block = parse_latest_repair_block()

    non_oa_pattern = re.compile(
        r"(PMC\d+\.\d+): current remote metadata is not marked "
        r"PMC Open Access\."
    )

    stale_pattern = re.compile(
        r"(PMC\d+\.\d+) is a stale local incomplete artifact:"
    )

    non_oa_versions: set[str] = set()
    stale_versions: set[str] = set()

    for line in block:
        non_oa_match = non_oa_pattern.search(line)
        if non_oa_match:
            non_oa_versions.add(non_oa_match.group(1))

        stale_match = stale_pattern.search(line)
        if stale_match:
            stale_versions.add(stale_match.group(1))

    logger.info(
        "Latest repair run explicitly recorded %d current non-OA versions "
        "and %d stale versions.",
        len(non_oa_versions),
        len(stale_versions),
    )

    return non_oa_versions, stale_versions


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize(
    logger: logging.Logger,
) -> int:
    """
    Finalize the authoritative Stage 02 manifest.

    This is network-free.

    Finalization verifies:
    1. The selected full Stage 02 run accounted for all matched PMCIDs.
    2. Every local OA article-version artifact is valid.
    3. Empty/incomplete local directories are NOT silently treated as
       unavailable.
    4. Explicitly logged current-dataset-unavailable records are accounted
       for as unavailable.
    5. Recovered records with valid local files are counted as recovered.
    6. The latest permanent repair evidence is used to exclude current non-OA
       versions and stale versions removed from the current dataset.
    """

    logger.info(
        "Stage 02 finalization started."
    )

    (
        matched_count,
        processed_pmcids,
        unavailable_logged,
        non_oa_versions,
        failed_pmcids,
    ) = extract_full_run_evidence(
        logger
    )

    repair_non_oa_versions, repair_stale_versions = extract_repair_evidence(
        logger
    )

    # The permanent repair pass is authoritative for the incomplete local
    # artifacts it inspected. A current version whose remote metadata is no
    # longer marked PMC Open Access is excluded from the corpus, while a
    # version removed from the current dataset is retained only as historical
    # local debris and excluded from the authoritative manifest.
    non_oa_versions.update(repair_non_oa_versions)
    ignored_stale_versions = repair_stale_versions

    if len(processed_pmcids) != matched_count:
        logger.error(
            "Finalization refused: selected run processed %d unique "
            "PMCIDs but reported %d matched records.",
            len(processed_pmcids),
            matched_count,
        )

        return 1

    discovered = discover_local_article_versions()

    manifest_rows: list[
        dict[str, str]
    ] = []

    accounted_pmcids: set[str] = set()

    invalid_versions: list[
        tuple[str, str, str]
    ] = []

    stale_or_ignored_versions = 0

    for pmcid in sorted(
        processed_pmcids,
        key=lambda value: int(value[3:]),
    ):
        local_versions = discovered.get(
            pmcid,
            [],
        )

        valid_rows_for_pmcid: list[
            dict[str, str]
        ] = []

        invalid_for_pmcid = False

        # If the original run explicitly recorded this PMCID as having no
        # current article version, local incomplete directories should not
        # turn that accounting decision into a false retrieval failure.
        #
        # However, if a valid local version now exists, it represents a
        # successful recovery and should take precedence.
        for version, directory in local_versions:
            version_key = f"{pmcid}.{version}"

            # Versions explicitly identified by the full run or the latest
            # permanent repair pass as non-OA are not authoritative corpus
            # artifacts.
            if version_key in non_oa_versions:
                stale_or_ignored_versions += 1

                logger.info(
                    "%s.%s ignored during finalization because current "
                    "PMC metadata is not marked PMC Open Access.",
                    pmcid,
                    version,
                )

                continue

            # A local incomplete version that the repair pass confirmed is no
            # longer present in the current PMC dataset is stale historical
            # debris, not a retrieval failure. Do not attempt network access
            # during finalization.
            if version_key in ignored_stale_versions:
                stale_or_ignored_versions += 1

                logger.info(
                    "%s.%s ignored during finalization because the repair "
                    "pass confirmed that this version is no longer present "
                    "in the current PMC dataset.",
                    pmcid,
                    version,
                )

                continue

            valid, metadata, status = (
                verify_existing_article_version(
                    pmcid,
                    version,
                )
            )

            if valid:
                valid_rows_for_pmcid.append(
                    manifest_row_from_metadata(
                        pmid="",
                        pmcid=pmcid,
                        version=version,
                        metadata=metadata or {},
                        status="already_verified",
                    )
                )

                continue

            # An incomplete local directory for a PMCID explicitly recorded
            # as unavailable can be stale debris from an earlier attempt.
            # Do not classify it as a current retrieval failure if there is
            # no valid recovered version.
            if (
                pmcid in unavailable_logged
                and status in {
                    "missing_json",
                    "missing_xml",
                    "invalid_json",
                }
            ):
                stale_or_ignored_versions += 1

                logger.warning(
                    "%s.%s ignored during finalization because the "
                    "selected Stage 02 run explicitly recorded this PMCID "
                    "as unavailable and this local directory is incomplete.",
                    pmcid,
                    version,
                )

                continue

            invalid_for_pmcid = True

            invalid_versions.append(
                (
                    pmcid,
                    version,
                    status,
                )
            )

            logger.error(
                "%s.%s failed local verification: %s",
                pmcid,
                version,
                status,
            )

        if valid_rows_for_pmcid:
            manifest_rows.extend(
                valid_rows_for_pmcid
            )

            accounted_pmcids.add(
                pmcid
            )

            continue

        if (
            pmcid in unavailable_logged
            and not invalid_for_pmcid
        ):
            manifest_rows.append(
                unavailable_manifest_row(
                    pmcid
                )
            )

            accounted_pmcids.add(
                pmcid
            )

            continue

        # If no valid version exists and the original run did not explicitly
        # record the PMCID as unavailable, this is an accounting failure.
        if not local_versions:
            logger.error(
                "%s has no local article-version directory and was "
                "not explicitly recorded as unavailable by the selected "
                "Stage 02 run.",
                pmcid,
            )

        elif not invalid_for_pmcid:
            logger.error(
                "%s has no valid local article version and no valid "
                "unavailable accounting record.",
                pmcid,
            )

        invalid_for_pmcid = True

    if invalid_versions:
        logger.error(
            "Finalization refused: %d local article-version "
            "artifact(s) failed verification.",
            len(invalid_versions),
        )

        logger.error(
            "Run --repair-incomplete before attempting finalization again."
        )

        return 1

    if len(accounted_pmcids) != matched_count:
        missing = sorted(
            processed_pmcids - accounted_pmcids,
            key=lambda value: int(value[3:]),
        )

        logger.error(
            "Finalization refused: accounted PMCIDs=%d, "
            "matched PMCIDs=%d.",
            len(accounted_pmcids),
            matched_count,
        )

        logger.error(
            "Unaccounted PMCIDs: %d",
            len(missing),
        )

        if missing:
            logger.error(
                "First unaccounted PMCIDs: %s",
                ", ".join(missing[:20]),
            )

        return 1

    # Ensure no unexpected duplicate PMCID/version rows exist.
    manifest_keys = [
        (
            row["pmcid"],
            row["version"],
        )
        for row in manifest_rows
        if row["version"]
    ]

    if len(manifest_keys) != len(
        set(manifest_keys)
    ):
        logger.error(
            "Finalization refused: duplicate PMCID/version "
            "entries detected in the proposed manifest."
        )

        return 1

    write_manifest(
        manifest_rows
    )

    logger.info(
        "Stage 02 finalization completed successfully."
    )

    logger.info(
        "Authoritative PMC manifest written: %s",
        MANIFEST,
    )

    logger.info(
        "Final accounting: matched_pmcids=%d, "
        "accounted_pmcids=%d, manifest_rows=%d, "
        "ignored_stale_or_non_oa_versions=%d",
        matched_count,
        len(accounted_pmcids),
        len(manifest_rows),
        stale_or_ignored_versions,
    )

    if failed_pmcids:
        logger.info(
            "Original selected run contained %d failed PMCID(s); "
            "successful local verification confirms any subsequently "
            "recovered artifacts.",
            len(failed_pmcids),
        )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 02 command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Stage 02 - retrieve and verify PMC article versions."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Perform PubMed -> PMC matching without downloading "
            "article files or writing the authoritative manifest."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Test-only limit on the number of Stage 01 PMIDs processed. "
            "Never writes the authoritative manifest."
        ),
    )

    parser.add_argument(
        "--retry-pmcids",
        nargs="+",
        metavar="PMCID",
        help=(
            "Retry only the explicitly supplied PMCIDs. "
            "Does not rerun PubMed -> PMC matching and does not "
            "write metadata/pmc.csv."
        ),
    )

    parser.add_argument(
        "--repair-incomplete",
        action="store_true",
        help=(
            "Permanently repair incomplete local PMC article-version "
            "directories using the current PMC Article Dataset. "
            "Does not rerun PubMed -> PMC matching and does not "
            "write metadata/pmc.csv."
        ),
    )

    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Perform network-free final verification and write the "
            "authoritative metadata/pmc.csv manifest."
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    selected_modes = sum(
        bool(value)
        for value in [
            args.retry_pmcids,
            args.repair_incomplete,
            args.finalize,
        ]
    )

    if selected_modes > 1:
        parser.error(
            "--retry-pmcids, --repair-incomplete, and --finalize "
            "are mutually exclusive."
        )

    if args.dry_run and (
        args.retry_pmcids
        or args.repair_incomplete
        or args.finalize
    ):
        parser.error(
            "--dry-run cannot be combined with "
            "--retry-pmcids, --repair-incomplete, or --finalize."
        )

    if args.limit is not None and args.limit <= 0:
        parser.error(
            "--limit must be greater than zero."
        )

    logger = get_logger()

    try:
        if args.finalize:
            return finalize(logger)

        if args.repair_incomplete:
            return repair_incomplete(logger)

        if args.retry_pmcids:
            return retry_pmcids(
                args.retry_pmcids,
                logger,
            )

        return run(
            logger,
            limit=args.limit,
            dry_run=args.dry_run,
        )

    except KeyboardInterrupt:
        logger.error(
            "Stage 02 interrupted."
        )

        return 1

    except Exception as exc:
        logger.exception(
            "Stage 02 failed unexpectedly: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )