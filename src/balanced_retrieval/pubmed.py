"""The PubMed corpus: title and abstract of every record that matches the scope, from NCBI's E-utilities.

``parse_pubmed`` turns the XML of ``efetch`` into one document per usable record: its title and abstract (a structured abstract
keeps its section labels), keyed ``PMID:<number>``, with the DOI, the PMC identifier and the date kept beside the text. Records
without an abstract, records that were retracted and withdrawn citations are left out. ``fetch_pubmed`` downloads a scope as of a
date, so that a corpus has a name and can be made again.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from rationale_query.store import StoreError

from .documents import download_in_batches
from .ncbi import Client, search_ids
from .scope import pubmed_scope

BATCH = 200


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _date(citation: ET.Element) -> str:
    """The best publication date of a record as ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``, or an empty string."""
    for path in ("Article/ArticleDate", "Article/Journal/JournalIssue/PubDate"):
        node = citation.find(path)
        if node is None:
            continue
        year = _text(node.find("Year"))
        if year.isdigit():
            month, day = _text(node.find("Month")), _text(node.find("Day"))
            months = {m: f"{i:02d}" for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
            month = month.zfill(2) if month.isdigit() else months.get(month[:3].title(), "")
            if month and day.isdigit():
                return f"{year}-{month}-{day.zfill(2)}"
            return f"{year}-{month}" if month else year
    return ""


def parse_pubmed(xml: bytes | str, left_out: Counter[str] | None = None) -> list[dict[str, Any]]:
    """The documents of one ``efetch`` answer (a ``PubmedArticleSet``); an empty list when nothing is usable.

    ``left_out``, if given, counts the records that were not kept, by reason.
    """
    left_out = Counter() if left_out is None else left_out
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise StoreError(f"the PubMed answer is not valid XML: {exc}") from exc
    documents = []
    for article in root.iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        pmid = _text(citation.find("PMID"))
        if not pmid.isdigit():
            continue
        types = {_text(t) for t in citation.iterfind("Article/PublicationTypeList/PublicationType")}
        if "Retracted Publication" in types:
            left_out["retracted"] += 1
            continue
        parts = []
        for node in citation.iterfind("Article/Abstract/AbstractText"):
            label, body = node.get("Label", ""), _text(node)
            if body:
                parts.append(f"{label}: {body}" if label else body)
        if not parts:
            left_out["no abstract"] += 1
            continue
        title = _text(citation.find("Article/ArticleTitle"))
        ids = {node.get("IdType"): _text(node) for node in article.iterfind("PubmedData/ArticleIdList/ArticleId")}
        documents.append(
            {
                "doc_key": f"PMID:{pmid}",
                "text": " ".join([title, *parts]).strip(),
                "pmid": pmid,
                "doi": ids.get("doi", ""),
                "pmcid": ids.get("pmc", ""),
                "date": _date(citation),
            }
        )
    return documents


def fetch_pubmed(
    client: Client,
    out: Path,
    *,
    as_of: date,
    query: str | None = None,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    found_so_far: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Download the records of the scope published up to ``as_of`` into the JSON-lines file ``out`` (resumable)."""
    term = query or pubmed_scope()
    settings = {"source": "pubmed", "query": term, "as_of": as_of.isoformat()}

    def fetch_batch(ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        left_out: Counter[str] = Counter()
        found = parse_pubmed(client.call("efetch.fcgi", db="pubmed", id=",".join(ids), retmode="xml", rettype="abstract"), left_out)
        left_out["not returned, or not a journal article"] += len(ids) - len(found) - sum(left_out.values())
        return found, {reason: number for reason, number in left_out.items() if number}

    return download_in_batches(
        out,
        settings,
        lambda: search_ids(client, "pubmed", term, as_of, found_so_far),
        fetch_batch,
        batch_size=BATCH,
        limit=limit,
        progress=progress,
    )
