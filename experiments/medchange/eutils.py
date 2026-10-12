"""PubMed access: a client for NCBI's E-utilities (search, summaries, XML records) and the bounds on the date a record
first became public. Used by ``pubmed_asof``, ``freeze_candidates`` and ``ad_cochrane_supply``.

NCBI usage: <= 3 requests/s without a key, <= 10 with one.
"""

from __future__ import annotations

import datetime as dt
import http.client
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Optional, Sequence

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

_MON = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def parse_sortpubdate(s: str) -> Optional[str]:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def date_bounds(text: str) -> Optional[tuple[str, str]]:
    """Earliest and latest ISO date a PubMed date string can mean.

    "2012 Oct 17" -> that day; "2012 Oct" -> the whole month; "2012" -> the whole
    year; "2012 Oct-Dec" -> the whole span; "2012 Dec-Jan" (wraps the year) and
    seasons ("2012 Winter", "2012 Fall-Winter") -> widened to whole years. Anything
    unparsable -> None. Never raises: one odd string must not stop a long run, and
    widening only ever makes a record look LATER, so it can be dropped but never
    wrongly admitted.
    """
    try:
        m = re.match(r"\s*(\d{4})(?:\s+([A-Za-z]+)(?:-([A-Za-z]+))?(?:\s+(\d{1,2})(?:-(\d{1,2}))?)?)?", text or "")
        if not m:
            return None
        y = int(m.group(1))
        if not m.group(2):
            return f"{y:04d}-01-01", f"{y:04d}-12-31"
        m1 = _MON.get(m.group(2)[:3].title()) if len(m.group(2)) == 3 else None
        m2 = (_MON.get(m.group(3)[:3].title()) if m.group(3) and len(m.group(3)) == 3
              else (m1 if not m.group(3) else None))
        if m1 is None or m2 is None:           # season words, unknown tokens
            return f"{y:04d}-01-01", f"{y + (1 if m.group(3) else 0):04d}-12-31"
        if m2 < m1:                             # "Dec-Jan": spans into the next year
            return f"{y:04d}-{m1:02d}-01", f"{y + 1:04d}-12-31"
        if m.group(4) and not m.group(3):
            lo = dt.date(y, m1, int(m.group(4))).isoformat()
            hi = dt.date(y, m1, int(m.group(5) or m.group(4))).isoformat()
            return lo, hi
        last = (dt.date(y + (m2 == 12), (m2 % 12) + 1, 1) - dt.timedelta(days=1)).day
        return f"{y:04d}-{m1:02d}-01", f"{y:04d}-{m2:02d}-{last:02d}"
    except (ValueError, TypeError):
        return None


def availability(pubdate: str, epubdate: str, sortpubdate: Optional[str]) -> Optional[dict]:
    """When a record first became public, as bounds.

    PubMed's [dp] filter matches the print date OR the electronic date, so a
    record can pass a "published before X" search while its print date is after
    X. It was still available if its electronic date is before X: availability
    is the EARLIEST of the dates we know. ``upper`` is the latest it can have
    first appeared (used to prove a record precedes the cutoff, so uncertain
    boundary records are dropped, never admitted); ``lower`` the earliest
    (used to prove a record is newer than the previous review version).
    """
    cands = [b for b in (date_bounds(pubdate), date_bounds(epubdate)) if b]
    if not cands and sortpubdate:
        cands = [(sortpubdate, sortpubdate)]
    if not cands:
        return None
    return {"lower": min(c[0] for c in cands), "upper": min(c[1] for c in cands)}


class EUtils:
    def __init__(self, api_key: Optional[str] = None, *, opener: Optional[Callable] = None,
                 sleep: Callable = time.sleep):
        self.api_key = api_key
        self.delay = 0.11 if api_key else 0.34
        self._open = opener or (lambda url: urllib.request.urlopen(url, timeout=60).read())
        self._sleep = sleep

    def get(self, endpoint: str, params: dict) -> dict:
        """One JSON request, retried up to five times with a growing pause."""
        params = dict(params, retmode="json", tool="thesis_research")
        if self.api_key:
            params["api_key"] = self.api_key
        url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            self._sleep(self.delay)
            try:
                return json.loads(self._open(url))
            except (OSError, json.JSONDecodeError, http.client.HTTPException) as exc:     # OSError: URLError, timeouts, a reset connection
                if attempt == 4:
                    raise RuntimeError(f"E-utilities failed after 5 attempts: {exc}") from exc
                self._sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def efetch(self, pmids: Sequence[str], **extra: str) -> bytes:
        """The XML records of ``pmids`` (``extra``: further parameters, e.g. ``rettype="abstract"``), retried up to five
        times; a download that is not well-formed XML counts as a failed attempt."""
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **extra, "tool": "thesis_research"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = EUTILS + "efetch.fcgi?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            self._sleep(self.delay)
            try:
                data = self._open(url)
                ET.fromstring(data)
                return data
            except (ET.ParseError, OSError, http.client.HTTPException) as exc:
                if attempt == 4:
                    raise RuntimeError(f"efetch failed after 5 attempts: {exc}") from exc
                self._sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def search(self, term: str, maxdate: str, retmax: int) -> tuple[int, list[str]]:
        r = self.get("esearch.fcgi", {"db": "pubmed", "term": term, "sort": "relevance",
                                      "datetype": "pdat", "mindate": "1800/01/01",
                                      "maxdate": maxdate, "retmax": retmax})["esearchresult"]
        return int(r.get("count", 0)), list(r.get("idlist", []))

    def summaries(self, pmids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(pmids), 200):
            batch = pmids[i:i + 200]
            res = self.get("esummary.fcgi", {"db": "pubmed", "id": ",".join(batch)})["result"]
            for p in batch:
                d = res.get(p, {})
                sort = parse_sortpubdate(d.get("sortpubdate", ""))
                av = availability(d.get("pubdate", ""), d.get("epubdate", ""), sort) or {}
                out.append({"pmid": p, "pubdate": d.get("pubdate", ""),
                            "epubdate": d.get("epubdate", ""), "sortpubdate": sort,
                            "lower": av.get("lower"), "upper": av.get("upper"),
                            "pubtypes": d.get("pubtype", []), "journal": d.get("source", "")})
        return out
