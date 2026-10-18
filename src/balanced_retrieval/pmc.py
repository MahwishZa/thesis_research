"""The PMC corpus: the open-access full text of the PubMed records that carry a PMC identifier.

The identifiers come from a PubMed download (``fetch_pubmed``), so the PMC corpus covers the same records as the PubMed corpus
and each document keeps the PubMed key ``PMID:<number>`` when it has one. ``parse_pmc`` reads the JATS XML that ``efetch`` returns:
title, abstract and the paragraphs and headings of the body; tables, figures, formulas and the reference list are left out.
An article whose full text is not in the open-access subset arrives without a body and is dropped. The licence group of each
article is read from its own licence statement and kept beside the text.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rationale_query.store import StoreError

from .documents import download_in_batches
from .ncbi import Client

BATCH = 25
_SKIP = {"table-wrap", "fig", "fig-group", "disp-formula", "inline-formula", "supplementary-material", "ref-list", "xref", "label", "graphic", "media"}
_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_HREF = "{http://www.w3.org/1999/xlink}href"
_CC = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)", re.IGNORECASE)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _plain(element: ET.Element) -> str:
    """The text of an element without the parts that are not prose (tables, figures, formulas, reference markers)."""
    pieces = [element.text or ""]
    for child in element:
        if child.tag not in _SKIP:
            pieces.append(_plain(child))
        pieces.append(child.tail or "")
    return "".join(pieces)


def _prose(element: ET.Element, lines: list[str]) -> None:
    for child in element:
        if child.tag in ("p", "title"):
            line = _clean(_plain(child))
            if line:
                lines.append(line)
        elif child.tag in ("sec", "list", "list-item", "boxed-text", "disp-quote", "def-list", "def-item", "def", "statement"):
            _prose(child, lines)


def licence_group(article: ET.Element) -> str:
    """cc0, cc-by, cc-by-nc, ... from the article's licence statement, else ``other`` (a licence not recognised) or ``none``."""
    node = article.find("front/article-meta/permissions/license")
    if node is None:
        return "none"
    candidates = [node.get(_HREF, ""), node.get("{http://www.w3.org/1999/xlink}href", "")]
    candidates += [link.get(_HREF, "") for link in node.iter("ext-link")] + [_clean("".join(node.itertext()))]
    for candidate in candidates:
        found = _CC.search(candidate)
        if found:
            kind, name = found.group(1).lower(), found.group(2).lower()
            return "cc0" if kind == "publicdomain" else f"cc-{name}"
    return "other"


def parse_pmc(xml: bytes | str, left_out: Counter[str] | None = None) -> list[dict[str, Any]]:
    """The documents of one ``efetch`` answer (a ``pmc-articleset``): articles in English that have a body.

    ``left_out``, if given, counts the articles that were not kept, by reason.
    """
    left_out = Counter() if left_out is None else left_out
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise StoreError(f"the PMC answer is not valid XML: {exc}") from exc
    documents = []
    for article in root.iter("article"):
        language = article.get(_LANG, "en")
        body = article.find("body")
        if not language.lower().startswith("en"):
            left_out["not English"] += 1
            continue
        if body is None:
            left_out["no full text (not open access, or a scan)"] += 1
            continue
        ids = {node.get("pub-id-type"): _clean("".join(node.itertext())) for node in article.iterfind("front/article-meta/article-id")}
        pmcid = ids.get("pmc") or ids.get("pmcid") or ""
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        pmid = ids.get("pmid", "")
        lines: list[str] = []
        title = article.find("front/article-meta/title-group/article-title")
        if title is not None:
            lines.append(_clean(_plain(title)))
        for abstract in article.iterfind("front/article-meta/abstract"):
            if abstract.get("abstract-type") in (None, "", "summary"):  # the plain abstract, not a graphical or teaser one
                _prose(abstract, lines)
        _prose(body, lines)
        text = "\n".join(line for line in lines if line)
        if not text or not (pmcid or pmid):
            left_out["no text or no identifier"] += 1
            continue
        documents.append(
            {
                "doc_key": f"PMID:{pmid}" if pmid.isdigit() else f"PMCID:{pmcid}",
                "text": text,
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": ids.get("doi", ""),
                "licence": licence_group(article),
            }
        )
    return documents


def pmc_ids_of(pubmed_file: Path) -> list[str]:
    """The PMC identifiers (digits only) of a PubMed download, in file order and without repeats."""
    found: dict[str, None] = {}
    with open(pubmed_file, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pmcid = str(json.loads(line).get("pmcid", "")).upper().removeprefix("PMC")
                if pmcid.isdigit():
                    found[pmcid] = None
    return list(found)


def fetch_pmc(
    client: Client,
    out: Path,
    *,
    pubmed_file: Path,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download the full text of the PMC identifiers of ``pubmed_file`` into ``out`` (resumable)."""
    settings = {"source": "pmc", "from": pubmed_file.name, "pubmed_state": _pubmed_signature(pubmed_file)}

    def fetch_batch(ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        left_out: Counter[str] = Counter()
        found = parse_pmc(client.call("efetch.fcgi", db="pmc", id=",".join(ids), retmode="xml"), left_out)
        left_out["not returned"] += len(ids) - len(found) - sum(left_out.values())
        return found, {reason: number for reason, number in left_out.items() if number}

    return download_in_batches(out, settings, lambda: pmc_ids_of(pubmed_file), fetch_batch, batch_size=BATCH, limit=limit, progress=progress)


def _pubmed_signature(pubmed_file: Path) -> dict[str, Any]:
    """What a PubMed download was made with, so a PMC download is tied to it (its state file, when it has one)."""
    state = pubmed_file.with_name(pubmed_file.name + ".state.json")
    try:
        return json.loads(state.read_text(encoding="utf-8"))["settings"]
    except (OSError, ValueError, KeyError):
        return {}
