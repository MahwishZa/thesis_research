#!/usr/bin/env python3
"""Fetch PubMed records via NCBI E-utilities (ESearch + EFetch)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
QUERIES_PATH = BASE_DIR / "search_queries.txt"
CSV_PATH = BASE_DIR / "pubmed_results.csv"
JSON_PATH = BASE_DIR / "pubmed_results.json"
LOG_PATH = BASE_DIR / "search_log.csv"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH = f"{EUTILS}/esearch.fcgi"
EFETCH = f"{EUTILS}/efetch.fcgi"

TOOL = "thesis_research_pubmed_fetch"
EMAIL = os.environ.get("NCBI_EMAIL", "thesis.research@example.com")
API_KEY = os.environ.get("NCBI_API_KEY", "").strip()

# NCBI: 3 req/s without key, 10 req/s with key. Stay slightly under.
MIN_INTERVAL = 0.12 if API_KEY else 0.4
ESEARCH_RETMAX = 10000
EFETCH_BATCH = 200
MAX_RETRIES = 6

CSV_FIELDS = [
    "pmid",
    "title",
    "abstract",
    "publication_date",
    "journal",
    "authors",
    "doi",
    "pmcid",
    "mesh_terms",
    "publication_types",
    "search_queries",
]

LOG_FIELDS = [
    "query_id",
    "query",
    "search_timestamp",
    "result_count",
    "retrieved_pmid_count",
    "retrieved_pmids",
    "errors_warnings",
]

SKIP_QUERY_IDS = {"Q12.1", "Q12.2", "Q12.3"}

DATE_FILTER_RE = re.compile(
    r'\("(\d{4}/\d{2}/\d{2})"\[Date - Publication\]\s*:\s*"(\d{4}/\d{2}/\d{2})"\[Date - Publication\]\)',
    re.I,
)
QUERY_HEADER_RE = re.compile(r"^#\s+(Q\d+\.\d+)\b(.*)$")


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = self.min_interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


LIMITER = RateLimiter(MIN_INTERVAL)


def parse_date_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y/%m/%d").date()


def format_date_ymd(value: date) -> str:
    return value.strftime("%Y/%m/%d")


def date_filter_term(start: date, end: date) -> str:
    return (
        f'("{format_date_ymd(start)}"[Date - Publication] : '
        f'"{format_date_ymd(end)}"[Date - Publication])'
    )


def midpoint_date(start: date, end: date) -> date:
    days = (end - start).days
    return start + timedelta(days=days // 2)


def load_queries_file(path: Path) -> tuple[date, date, list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    matches = DATE_FILTER_RE.findall(text)
    if not matches:
        raise ValueError(f"No publication date range found in {path}")
    start = parse_date_ymd(matches[0][0])
    end = parse_date_ymd(matches[0][1])

    queries: list[dict[str, str]] = []
    current_id: str | None = None
    current_title = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, buf
        if current_id is None:
            buf = []
            return
        lines = [ln.rstrip() for ln in buf if ln.strip() and not ln.strip().startswith("#")]
        term = "\n".join(lines).strip()
        term = re.sub(r"\s+", " ", term)
        if term:
            queries.append({"id": current_id, "title": current_title.strip(), "term": term})
        current_id = None
        current_title = ""
        buf = []

    for raw in text.splitlines():
        header = QUERY_HEADER_RE.match(raw)
        if header:
            flush()
            current_id = header.group(1)
            current_title = header.group(2).strip(" -")
            buf = []
            continue
        if raw.startswith("# ==="):
            flush()
            continue
        if current_id is not None:
            buf.append(raw)
    flush()
    return start, end, queries


def ensure_date_filter(term: str, start: date, end: date) -> str:
    if DATE_FILTER_RE.search(term):
        return term
    return f"({term}) AND {date_filter_term(start, end)}"


def eutils_post(url: str, params: dict[str, Any]) -> bytes:
    payload = {k: v for k, v in params.items() if v is not None and v != ""}
    data = urllib.parse.urlencode(payload, doseq=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("User-Agent", f"{TOOL}/1.0 ({EMAIL})")
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        LIMITER.wait()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            body = b""
            try:
                body = exc.read()
            except Exception:
                pass
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"HTTP {exc.code} from NCBI: {body[:500]!r}"
                ) from exc
            sleep_s = min(60.0, 2 ** attempt)
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(sleep_s, float(retry_after))
                    except ValueError:
                        pass
            time.sleep(sleep_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Network error calling NCBI: {exc}") from exc
            time.sleep(min(60.0, 2 ** attempt))
    raise RuntimeError(f"E-utilities request failed: {last_error}")


def common_params() -> dict[str, str]:
    params = {"tool": TOOL, "email": EMAIL}
    if API_KEY:
        params["api_key"] = API_KEY
    return params


def esearch_count_and_ids(term: str, retmax: int) -> tuple[int, list[str], str]:
    """Return (count, pmid list up to retmax, warning)."""
    warning = ""
    base = {
        **common_params(),
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "usehistory": "n",
        "sort": "pub_date",
    }
    raw = eutils_post(ESEARCH, {**base, "retmax": 0})
    payload = json.loads(raw.decode("utf-8"))
    result = payload.get("esearchresult") or {}
    error_list = result.get("errorlist") or {}
    if error_list:
        warning = json.dumps(error_list, ensure_ascii=True)
    if "ERROR" in result:
        raise RuntimeError(f"ESearch error: {result.get('ERROR')}")
    count = int(result.get("count") or 0)
    if count == 0 or retmax == 0:
        return count, [], warning
    ids: list[str] = []
    remaining = min(count, retmax, ESEARCH_RETMAX)
    retstart = 0
    while retstart < remaining:
        batch = min(ESEARCH_RETMAX, remaining - retstart)
        raw = eutils_post(
            ESEARCH,
            {**base, "retmax": batch, "retstart": retstart},
        )
        payload = json.loads(raw.decode("utf-8"))
        result = payload.get("esearchresult") or {}
        if "ERROR" in result:
            raise RuntimeError(f"ESearch error: {result.get('ERROR')}")
        batch_ids = [str(x) for x in (result.get("idlist") or [])]
        ids.extend(batch_ids)
        if not batch_ids:
            break
        retstart += len(batch_ids)
    return count, ids, warning


def collect_pmids_for_term(
    term: str,
    window_start: date,
    window_end: date,
    warnings: list[str],
) -> tuple[int, list[str]]:
    """Get all PMIDs for term, splitting the date window if PubMed's 10k cap is hit."""
    scoped = f"({term}) AND {date_filter_term(window_start, window_end)}"
    count, _empty, warn = esearch_count_and_ids(scoped, 0)
    if warn:
        warnings.append(warn)
    if count <= ESEARCH_RETMAX:
        _count, ids, warn2 = esearch_count_and_ids(scoped, count)
        if warn2:
            warnings.append(warn2)
        return count, ids
    if window_start >= window_end:
        msg = (
            f"PubMed 10,000-ID cap hit for a single-day window "
            f"{format_date_ymd(window_start)}; keeping first {len(ids)} of {count}"
        )
        warnings.append(msg)
        return count, ids
    mid = midpoint_date(window_start, window_end)
    if mid < window_start:
        mid = window_start
    if mid >= window_end:
        mid = window_end - timedelta(days=1)
        if mid < window_start:
            warnings.append(
                f"Could not split date window {format_date_ymd(window_start)}-"
                f"{format_date_ymd(window_end)}; keeping first {len(ids)} of {count}"
            )
            return count, ids
    _c1, left = collect_pmids_for_term(term, window_start, mid, warnings)
    _c2, right = collect_pmids_for_term(term, mid + timedelta(days=1), window_end, warnings)
    merged: list[str] = []
    seen: set[str] = set()
    for pmid in left + right:
        if pmid not in seen:
            seen.add(pmid)
            merged.append(pmid)
    return count, merged


def elem_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def first_text(parent: ET.Element, paths: list[str]) -> str:
    for path in paths:
        found = parent.find(path)
        if found is not None:
            value = elem_text(found)
            if value:
                return value
    return ""


def parse_pub_date(article: ET.Element) -> str:
    pubdate = article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pubdate is not None:
        medline = elem_text(pubdate.find("MedlineDate"))
        year = elem_text(pubdate.find("Year"))
        month = elem_text(pubdate.find("Month"))
        day = elem_text(pubdate.find("Day"))
        if year:
            parts = [year]
            if month:
                parts.append(month)
            if day:
                parts.append(day)
            return "-".join(parts)
        if medline:
            return medline
    article_date = article.find("./MedlineCitation/Article/ArticleDate")
    if article_date is not None:
        year = elem_text(article_date.find("Year"))
        month = elem_text(article_date.find("Month"))
        day = elem_text(article_date.find("Day"))
        if year:
            return "-".join(p for p in (year, month, day) if p)
    history = article.find('./PubmedData/History/PubMedPubDate[@PubStatus="pubmed"]')
    if history is not None:
        year = elem_text(history.find("Year"))
        month = elem_text(history.find("Month"))
        day = elem_text(history.find("Day"))
        if year:
            return "-".join(p for p in (year, month, day) if p)
    return ""


def parse_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall("./MedlineCitation/Article/AuthorList/Author"):
        collective = elem_text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last = elem_text(author.find("LastName"))
        fore = elem_text(author.find("ForeName")) or elem_text(author.find("Initials"))
        if last and fore:
            authors.append(f"{last}, {fore}")
        elif last:
            authors.append(last)
    return authors


def parse_abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText"):
        label = (node.get("Label") or "").strip()
        body = elem_text(node)
        if not body:
            continue
        parts.append(f"{label}: {body}" if label else body)
    return "\n".join(parts)


def parse_ids(article: ET.Element) -> tuple[str, str, str]:
    pmid = elem_text(article.find("./MedlineCitation/PMID"))
    doi = ""
    pmcid = ""
    for eloc in article.findall("./MedlineCitation/Article/ELocationID"):
        if (eloc.get("EIdType") or "").lower() == "doi":
            doi = elem_text(eloc) or doi
    for aid in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        id_type = (aid.get("IdType") or "").lower()
        value = elem_text(aid)
        if id_type == "doi" and value:
            doi = value
        elif id_type in {"pmc", "pmcid"} and value:
            pmcid = value
        elif id_type == "pubmed" and not pmid:
            pmid = value
    return pmid, doi, pmcid


def parse_mesh(article: ET.Element) -> list[str]:
    terms: list[str] = []
    for heading in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        name = elem_text(descriptor)
        if not name:
            continue
        maj = descriptor.get("MajorTopicYN") == "Y" if descriptor is not None else False
        quals: list[str] = []
        for qualifier in heading.findall("QualifierName"):
            qname = elem_text(qualifier)
            if qname:
                qmaj = qualifier.get("MajorTopicYN") == "Y"
                quals.append(qname + ("*" if qmaj else ""))
        label = name + ("*" if maj else "")
        if quals:
            label = f"{label} ({', '.join(quals)})"
        terms.append(label)
    return terms


def parse_publication_types(article: ET.Element) -> list[str]:
    types: list[str] = []
    for node in article.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType"):
        value = elem_text(node)
        if value:
            types.append(value)
    if types:
        return types
    for node in article.findall("./MedlineCitation/PublicationTypeList/PublicationType"):
        value = elem_text(node)
        if value:
            types.append(value)
    return types


def parse_pubmed_xml(xml_bytes: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    records: list[dict[str, Any]] = []
    for article in root.findall("PubmedArticle"):
        pmid, doi, pmcid = parse_ids(article)
        if not pmid:
            continue
        records.append(
            {
                "pmid": pmid,
                "title": first_text(
                    article,
                    ["./MedlineCitation/Article/ArticleTitle"],
                ),
                "abstract": parse_abstract(article),
                "publication_date": parse_pub_date(article),
                "journal": first_text(
                    article,
                    [
                        "./MedlineCitation/Article/Journal/Title",
                        "./MedlineCitation/Article/Journal/ISOAbbreviation",
                    ],
                ),
                "authors": parse_authors(article),
                "doi": doi,
                "pmcid": pmcid,
                "mesh_terms": parse_mesh(article),
                "publication_types": parse_publication_types(article),
            }
        )
    return records


def efetch_records(pmids: list[str], errors: list[str]) -> list[dict[str, Any]]:
    fetched: list[dict[str, Any]] = []
    for i in range(0, len(pmids), EFETCH_BATCH):
        batch = pmids[i : i + EFETCH_BATCH]
        try:
            raw = eutils_post(
                EFETCH,
                {
                    **common_params(),
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                    "retmode": "xml",
                },
            )
            fetched.extend(parse_pubmed_xml(raw))
        except Exception as exc:
            errors.append(f"EFetch failed for PMIDs {batch[0]}-{batch[-1]}: {exc}")
    return fetched


def join_list(values: list[str]) -> str:
    return "; ".join(values)


def record_to_csv_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "pmid": record["pmid"],
        "title": record.get("title") or "",
        "abstract": record.get("abstract") or "",
        "publication_date": record.get("publication_date") or "",
        "journal": record.get("journal") or "",
        "authors": join_list(record.get("authors") or []),
        "doi": record.get("doi") or "",
        "pmcid": record.get("pmcid") or "",
        "mesh_terms": join_list(record.get("mesh_terms") or []),
        "publication_types": join_list(record.get("publication_types") or []),
        "search_queries": join_list(record.get("search_queries") or []),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_csv_row(record))


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def csv_pmids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["pmid"] for row in csv.DictReader(handle) if row.get("pmid")]


def verify_outputs(csv_path: Path, json_path: Path) -> tuple[int, int, list[str]]:
    problems: list[str] = []
    csv_ids = csv_pmids(csv_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    json_ids = [str(item.get("pmid") or "") for item in data]
    if len(csv_ids) != len(set(csv_ids)):
        problems.append("CSV contains duplicate PMIDs")
    if len(json_ids) != len(set(json_ids)):
        problems.append("JSON contains duplicate PMIDs")
    if csv_ids != json_ids:
        problems.append("CSV and JSON PMID lists differ (order or membership)")
    if set(csv_ids) != set(json_ids):
        only_csv = set(csv_ids) - set(json_ids)
        only_json = set(json_ids) - set(csv_ids)
        problems.append(
            f"PMID set mismatch: only_csv={len(only_csv)} only_json={len(only_json)}"
        )
    return len(csv_ids), len(set(csv_ids)), problems


def merge_record(
    store: dict[str, dict[str, Any]],
    record: dict[str, Any],
    query_id: str,
) -> None:
    pmid = record["pmid"]
    if pmid not in store:
        store[pmid] = {**record, "search_queries": [query_id]}
        return
    queries = store[pmid]["search_queries"]
    if query_id not in queries:
        queries.append(query_id)
    for field in (
        "title",
        "abstract",
        "publication_date",
        "journal",
        "doi",
        "pmcid",
    ):
        if not store[pmid].get(field) and record.get(field):
            store[pmid][field] = record[field]
    for field in ("authors", "mesh_terms", "publication_types"):
        if not store[pmid].get(field) and record.get(field):
            store[pmid][field] = record[field]


def run_queries(
    queries: list[dict[str, str]],
    date_start: date,
    date_end: date,
    test_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int, list[str]]:
    store: dict[str, dict[str, Any]] = {}
    log_rows: list[dict[str, str]] = []
    global_errors: list[str] = []
    retrieved_total = 0

    for query in queries:
        qid = query["id"]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        warnings: list[str] = []
        if qid in SKIP_QUERY_IDS:
            log_rows.append(
                {
                    "query_id": qid,
                    "query": query["term"],
                    "search_timestamp": ts,
                    "result_count": "",
                    "retrieved_pmid_count": "0",
                    "retrieved_pmids": "",
                    "errors_warnings": "skipped (filter/comment/date-only block, not a standalone search)",
                }
            )
            continue

        term = ensure_date_filter(query["term"], date_start, date_end)
        try:
            count, pmids, split_warnings = collect_pmids_wrapped(
                query["term"], date_start, date_end
            )
            warnings.extend(split_warnings)
            if test_limit is not None:
                pmids = pmids[:test_limit]
                warnings.append(f"test mode: fetched at most {test_limit} PMIDs")
            fetch_errors: list[str] = []
            records = efetch_records(pmids, fetch_errors)
            warnings.extend(fetch_errors)
            fetched_ids = [r["pmid"] for r in records]
            missing = [p for p in pmids if p not in set(fetched_ids)]
            if missing:
                warnings.append(f"EFetch missing {len(missing)} PMIDs (kept as ID-only)")
                for pmid in missing:
                    records.append(
                        {
                            "pmid": pmid,
                            "title": "",
                            "abstract": "",
                            "publication_date": "",
                            "journal": "",
                            "authors": [],
                            "doi": "",
                            "pmcid": "",
                            "mesh_terms": [],
                            "publication_types": [],
                        }
                    )
            for record in records:
                merge_record(store, record, qid)
            retrieved_total += len(pmids)
            log_rows.append(
                {
                    "query_id": qid,
                    "query": term,
                    "search_timestamp": ts,
                    "result_count": str(count),
                    "retrieved_pmid_count": str(len(pmids)),
                    "retrieved_pmids": ";".join(pmids),
                    "errors_warnings": " | ".join(warnings),
                }
            )
            print(
                f"{qid}: count={count} retrieved={len(pmids)} unique_so_far={len(store)}",
                flush=True,
            )
        except Exception as exc:
            msg = str(exc)
            global_errors.append(f"{qid}: {msg}")
            log_rows.append(
                {
                    "query_id": qid,
                    "query": term,
                    "search_timestamp": ts,
                    "result_count": "",
                    "retrieved_pmid_count": "0",
                    "retrieved_pmids": "",
                    "errors_warnings": msg,
                }
            )
            print(f"{qid}: ERROR {msg}", flush=True)

    records = [store[pmid] for pmid in sorted(store, key=lambda x: int(x))]
    for record in records:
        record["search_queries"] = sorted(
            record["search_queries"],
            key=lambda q: tuple(int(p) if p.isdigit() else p for p in q.replace("Q", "").split(".")),
        )
    return records, log_rows, retrieved_total, global_errors


def collect_pmids_wrapped(
    raw_term: str,
    date_start: date,
    date_end: date,
) -> tuple[int, list[str], list[str]]:
    """Search using the file term, applying the file date window via splitting."""
    warnings: list[str] = []
    # Strip an existing 5-year filter so date-splitting can narrow the window.
    term = DATE_FILTER_RE.sub("", raw_term)
    term = re.sub(r"\s+AND\s+$", "", term.strip())
    term = re.sub(r"^\s+AND\s+", "", term)
    term = re.sub(r"\s+", " ", term).strip()
    if term.endswith("AND"):
        term = term[: -len("AND")].strip()
    count, pmids = collect_pmids_for_term(term, date_start, date_end, warnings)
    return count, pmids, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Small pipeline test (Q0.2, 5 records)")
    args = parser.parse_args()

    date_start, date_end, queries = load_queries_file(QUERIES_PATH)
    print(f"Date window from file: {date_start.isoformat()} to {date_end.isoformat()}")
    print(f"Parsed {len(queries)} query blocks")

    if args.test:
        queries = [q for q in queries if q["id"] == "Q0.2"]
        if not queries:
            print("Q0.2 not found in search_queries.txt", file=sys.stderr)
            return 1
        test_limit = 5
        print("TEST: Q0.2, max 5 PMIDs")
    else:
        test_limit = None

    records, log_rows, retrieved_total, errors = run_queries(
        queries, date_start, date_end, test_limit
    )
    write_csv(CSV_PATH, records)
    write_json(JSON_PATH, records)
    write_log(LOG_PATH, log_rows)

    n_csv, n_unique, problems = verify_outputs(CSV_PATH, JSON_PATH)
    print(f"Records retrieved (PMID hits across queries, before dedup): {retrieved_total}")
    print(f"Unique records saved: {n_unique}")
    print(f"CSV rows: {n_csv}")
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  {err}")
    if problems:
        print("Verification issues:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("CSV and JSON contain the same deduplicated PMIDs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
