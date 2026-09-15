#!/usr/bin/env python3

"""Stage 02 - PMC Open Access full-text retrieval.

Uses the PMID artifacts produced by Stage 01.

For each PMID, Stage 02:
1. Finds matching PMC Open Access records.
2. Retrieves the article-version JSON metadata from the PMC Cloud Service.
3. Accepts only article versions marked as PMC Open Access.
4. Uses the article JSON license_code as the authoritative license field.
5. Downloads the JATS XML full text.
6. Verifies downloaded XML against PMC's MD5 value.
7. Records successful article versions in metadata/pmc.csv.

The PMC Article Dataset is accessed through NCBI E-Utilities and the
public PMC AWS Cloud Service. No AWS account is required.

The script is resumable. Existing verified XML files are not downloaded
again.

A failed run does not create a completed PMC manifest.
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
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw" / "pmc"
METADATA = ROOT / "metadata"
LOGS = ROOT / "logs"

PUBMED_RAW = DATA / "raw" / "pubmed"
MANIFEST = METADATA / "pmc.csv"
LOG_FILE = LOGS / "retrieval.log"

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com"

ESEARCH_BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 2.0
REQUEST_DELAY = 0.35


def get_logger() -> logging.Logger:
    """Create the shared retrieval logger."""

    LOGS.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("02_pmc")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def read_pubmed_pmids() -> list[str]:
    """Read and deduplicate PMID artifacts produced by Stage 01."""

    pmids: set[str] = set()

    for path in sorted(PUBMED_RAW.glob("*_pmids.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                pmid = line.strip()

                if pmid.isdigit():
                    pmids.add(pmid)

    return sorted(pmids, key=int)


def chunk(items: list[str], size: int):
    """Yield fixed-size batches."""

    for start in range(0, len(items), size):
        yield items[start:start + size]


def normalize_download_url(url: str) -> str:
    """Convert PMC s3:// URLs to HTTPS URLs usable by urllib."""

    if not url.startswith("s3://"):
        return url

    parsed = urlsplit(url)

    https_url = (
        f"https://{parsed.netloc}.s3.amazonaws.com"
        f"{parsed.path}"
    )

    if parsed.query:
        https_url += f"?{parsed.query}"

    return https_url


def ncbi_request(params: dict[str, str]) -> bytes:
    """Send an NCBI E-Utilities request with retries."""

    query = "&".join(
        f"{quote(str(key))}={quote(str(value))}"
        for key, value in params.items()
    )

    url = f"{NCBI_ESEARCH}?{query}"

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0"
                },
            )

            with urlopen(request, timeout=60) as response:
                return response.read()

        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            time.sleep(
                RETRY_DELAY * (2 ** (attempt - 1))
            )

    raise RuntimeError(
        f"NCBI request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def find_pmcids(pmids: list[str]) -> list[str]:
    """Find PMC Open Access records corresponding to a PMID batch."""

    if not pmids:
        return []

    terms = " OR ".join(
        f"{pmid}[PMID]"
        for pmid in pmids
    )

    term = (
        f"({terms}) AND "
        "open_access[Filter]"
    )

    payload = ncbi_request(
        {
            "db": "pmc",
            "term": term,
            "retmode": "json",
            "retmax": str(len(pmids)),
        }
    )

    data = json.loads(
        payload.decode("utf-8")
    )

    return data.get(
        "esearchresult",
        {}
    ).get(
        "idlist",
        []
    )


def get_article_versions(pmcid: str) -> list[str]:
    """Return article-version prefixes available in the PMC Cloud dataset."""

    url = (
        f"{PMC_S3}/?list-type=2"
        f"&prefix=PMC{pmcid}."
        f"&delimiter=/"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0"
                },
            )

            with urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8")

            return re.findall(
                r"<CommonPrefixes><Prefix>"
                r"(PMC\d+\.\d+/)"
                r"</Prefix>",
                text,
            )

        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            time.sleep(
                RETRY_DELAY * (2 ** (attempt - 1))
            )

    raise RuntimeError(
        f"PMC Cloud listing failed for {pmcid}: "
        f"{last_error}"
    )


def download_url(
    url: str,
    destination: Path,
) -> bytes:
    """Download a URL, supporting both HTTPS and PMC s3:// URLs."""

    url = normalize_download_url(url)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0"
                },
            )

            with urlopen(request, timeout=120) as response:
                data = response.read()

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = destination.with_suffix(
                destination.suffix + ".part"
            )

            temporary.write_bytes(data)
            temporary.replace(destination)

            return data

        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            time.sleep(
                RETRY_DELAY * (2 ** (attempt - 1))
            )

    raise RuntimeError(
        f"PMC download failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def md5_matches(
    data: bytes,
    expected_md5: str,
) -> bool:
    """Return whether data matches the expected MD5 digest."""

    actual = hashlib.md5(data).hexdigest()

    return actual.lower() == expected_md5.lower()


def load_json_metadata(
    version_prefix: str,
    destination: Path,
) -> dict:
    """Download and parse article-version JSON metadata."""

    version_prefix = version_prefix.rstrip("/")

    url = (
        f"{PMC_S3}/{version_prefix}/"
        f"{version_prefix}.json"
    )

    data = download_url(
        url,
        destination,
    )

    return json.loads(
        data.decode("utf-8")
    )


def retrieve_xml(
    metadata: dict,
    destination: Path,
) -> tuple[bool, str]:
    """Download and MD5-verify the article XML."""

    xml_url = metadata.get("xml_url")

    if not xml_url:
        return False, "no_xml_url"

    match = re.search(
        r"[?&]md5=([0-9a-fA-F]{32})",
        xml_url,
    )

    if not match:
        return False, "missing_xml_md5"

    expected_md5 = match.group(1)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        existing = destination.read_bytes()

        if md5_matches(
            existing,
            expected_md5,
        ):
            return True, "already_verified"

        destination.unlink()

    data = download_url(
        xml_url,
        destination,
    )

    if not md5_matches(
        data,
        expected_md5,
    ):
        destination.unlink(
            missing_ok=True
        )

        return False, "xml_md5_mismatch"

    return True, "downloaded_verified"


def write_manifest(rows: list[dict]) -> None:
    """Write the completed PMC manifest atomically."""

    METADATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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

    temporary = MANIFEST.with_suffix(
        ".csv.part"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(MANIFEST)


def run(
    pmids: list[str],
    limit: int | None,
    dry_run: bool,
    logger: logging.Logger,
) -> int:
    """Run the PMC retrieval stage."""

    if limit is not None:
        pmids = pmids[:limit]

    if not pmids:
        logger.error(
            "No PubMed PMID artifacts were found."
        )
        return 2

    logger.info(
        "Stage 02 started."
    )

    logger.info(
        "PubMed PMIDs available for PMC matching: %d",
        len(pmids),
    )

    all_pmcids: set[str] = set()

    batches = list(
        chunk(
            pmids,
            ESEARCH_BATCH_SIZE,
        )
    )

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        logger.info(
            "PMC matching batch %d/%d: %d PMIDs",
            batch_number,
            len(batches),
            len(batch),
        )

        pmcids = find_pmcids(batch)

        all_pmcids.update(pmcids)

        time.sleep(
            REQUEST_DELAY
        )

    logger.info(
        "PMC Open Access records matched: %d",
        len(all_pmcids),
    )

    if dry_run:
        logger.info(
            "Dry run complete. "
            "No PMC files were downloaded."
        )
        return 0

    RAW.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []
    failures = 0
    retrieved = 0

    sorted_pmcids = sorted(
        all_pmcids,
        key=int,
    )

    for index, pmcid_number in enumerate(
        sorted_pmcids,
        start=1,
    ):
        pmcid = f"PMC{pmcid_number}"

        logger.info(
            "Processing PMC record %d/%d: %s",
            index,
            len(sorted_pmcids),
            pmcid,
        )

        try:
            versions = get_article_versions(
                pmcid_number
            )

            if not versions:
                logger.warning(
                    "%s: no article version found",
                    pmcid,
                )
                failures += 1
                continue

            for version_prefix in versions:
                version = (
                    version_prefix
                    .rstrip("/")
                    .split(".")[-1]
                )

                article_dir = (
                    RAW / f"{pmcid}.{version}"
                )

                json_path = (
                    article_dir
                    / f"{pmcid}.{version}.json"
                )

                xml_path = (
                    article_dir
                    / f"{pmcid}.{version}.xml"
                )

                metadata = load_json_metadata(
                    version_prefix,
                    json_path,
                )

                is_oa = bool(
                    metadata.get(
                        "is_pmc_openaccess",
                        False,
                    )
                )

                if not is_oa:
                    json_path.unlink(
                        missing_ok=True
                    )
                    continue

                xml_ok, xml_status = retrieve_xml(
                    metadata,
                    xml_path,
                )

                if not xml_ok:
                    failures += 1
                    logger.error(
                        "%s.%s: XML retrieval failed: %s",
                        pmcid,
                        version,
                        xml_status,
                    )
                else:
                    retrieved += 1

                rows.append(
                    {
                        "pmid": metadata.get(
                            "pmid",
                            "",
                        ),
                        "pmcid": metadata.get(
                            "pmcid",
                            pmcid,
                        ),
                        "version": metadata.get(
                            "version",
                            version,
                        ),
                        "doi": metadata.get(
                            "doi",
                            "",
                        ),
                        "title": metadata.get(
                            "title",
                            "",
                        ),
                        "citation": metadata.get(
                            "citation",
                            "",
                        ),
                        "is_pmc_openaccess": str(
                            is_oa
                        ).lower(),
                        "is_manuscript": str(
                            metadata.get(
                                "is_manuscript",
                                False,
                            )
                        ).lower(),
                        "license_code": (
                            metadata.get(
                                "license_code",
                                "",
                            )
                            or ""
                        ),
                        "is_retracted": str(
                            metadata.get(
                                "is_retracted",
                                False,
                            )
                        ).lower(),
                        "json_path": str(
                            json_path.relative_to(
                                ROOT
                            )
                        ),
                        "xml_path": (
                            str(
                                xml_path.relative_to(
                                    ROOT
                                )
                            )
                            if xml_ok
                            else ""
                        ),
                        "status": xml_status,
                    }
                )

                time.sleep(
                    REQUEST_DELAY
                )

        except Exception as exc:
            failures += 1

            logger.error(
                "%s failed: %s",
                pmcid,
                exc,
            )

    if failures:
        logger.error(
            "Stage 02 failed: %d retrieval failures occurred.",
            failures,
        )

        logger.error(
            "No completed PMC manifest was written."
        )

        return 1

    write_manifest(
        rows
    )

    logger.info(
        "Stage 02 completed successfully: "
        "%d article versions retrieved.",
        retrieved,
    )

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N PubMed PMIDs "
            "for testing."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Match PMIDs to PMC Open Access records "
            "without downloading files."
        ),
    )

    args = parser.parse_args(
        argv
    )

    logger = get_logger()

    try:
        pmids = read_pubmed_pmids()

        return run(
            pmids=pmids,
            limit=args.limit,
            dry_run=args.dry_run,
            logger=logger,
        )

    except KeyboardInterrupt:
        logger.error(
            "Stage 02 interrupted."
        )
        return 130

    except Exception as exc:
        logger.error(
            "Stage 02 failed: %s",
            exc,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())