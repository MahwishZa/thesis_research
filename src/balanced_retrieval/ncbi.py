"""A small, polite client of NCBI's E-utilities, and the date-window search that gets past their 10,000-record limit.

``Client.call`` sends one request (at most three a second, ten with an API key in ``NCBI_API_KEY``) and tries again after a
pause when the service answers with an error that passes. ``search_ids`` returns every identifier that a search finds, however
many: ESearch will not page beyond the first 10,000 hits, so a window of publication dates that holds more is cut in half until
every window holds fewer. Nothing here needs more than the standard library.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, timedelta

from rationale_query.store import StoreError

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MAX_HITS = 9_999  # ESearch returns no hit beyond the 10,000th
EARLIEST = date(1800, 1, 1)


class NcbiError(StoreError):
    """The service refused a request, kept failing, or answered something that is not what was asked for."""


def _fetch(url: str, data: bytes) -> bytes:
    request = urllib.request.Request(url, data=data, headers={"User-Agent": "thesis-research (balanced_retrieval)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


class Client:
    """``opener(url, data)`` returns the bytes of the answer; the default sends a POST request over the internet."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        opener: Callable[[str, bytes], bytes] = _fetch,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        retries: int = 6,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        self.email = email if email is not None else os.environ.get("NCBI_EMAIL")
        self.opener, self.sleep, self.clock, self.retries = opener, sleep, clock, retries
        self.interval = 0.11 if self.api_key else 0.34
        self._last = -1e9

    def call(self, endpoint: str, **params: object) -> bytes:
        params = {**params, "tool": "thesis_research"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        data = urllib.parse.urlencode(params).encode("ascii")
        for attempt in range(self.retries + 1):
            wait = self._last + self.interval - self.clock()
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()
            try:
                return self.opener(BASE + endpoint, data)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise NcbiError(f"{endpoint}: the service answered {exc.code} {exc.reason}") from exc
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as exc:  # a dropped connection too
                if attempt == self.retries:
                    raise NcbiError(f"{endpoint}: cannot reach the service ({exc})") from exc
            self.sleep(min(2.0 ** (attempt + 1), 60.0))
        raise NcbiError(f"{endpoint}: no answer")  # unreachable; keeps the type checker content

    def search(self, db: str, term: str, mindate: date, maxdate: date, *, retmax: int = 0) -> tuple[int, list[str]]:
        """(number of hits, up to ``retmax`` identifiers) of ``term`` among records published in the window, both ends included."""
        raw = self.call(
            "esearch.fcgi",
            db=db,
            term=term,
            retmode="json",
            retmax=retmax,
            sort="pub_date",  # newest first inside a window; ESearch's default order is by relevance
            datetype="pdat",
            mindate=mindate.strftime("%Y/%m/%d"),
            maxdate=maxdate.strftime("%Y/%m/%d"),
        )
        try:
            result = json.loads(raw)["esearchresult"]
            if "ERROR" in result:
                raise NcbiError(f"the search was refused: {result['ERROR']}")
            if result.get("errorlist"):
                raise NcbiError(f"the search was refused: {result['errorlist']}")
            return int(result["count"]), [str(i) for i in result.get("idlist", [])]
        except (ValueError, KeyError, TypeError) as exc:
            raise NcbiError(f"esearch answered something unexpected: {raw[:200]!r}") from exc


def count_hits(client: Client, db: str, term: str, until: date) -> int:
    return client.search(db, term, EARLIEST, until)[0]


def search_ids(client: Client, db: str, term: str, until: date, found_so_far: Callable[[int], None] | None = None) -> list[str]:
    """Every identifier that ``term`` finds among records published up to ``until``, newest publication window first."""
    found: list[str] = []
    seen: set[str] = set()

    def take(start: date, end: date) -> None:
        total, _ = client.search(db, term, start, end)
        if total == 0:
            return
        if total > MAX_HITS:
            if start == end:
                raise NcbiError(f"{total} hits on the single day {start}; the search cannot be cut further")
            middle = start + timedelta(days=(end - start).days // 2)
            take(middle + timedelta(days=1), end)  # the newest first, so that a trial run holds recent records
            take(start, middle)
            return
        _, ids = client.search(db, term, start, end, retmax=total)
        if len(ids) != total:
            raise NcbiError(f"{start} to {end}: {total} hits but {len(ids)} identifiers were returned")
        for i in ids:
            if i not in seen:
                seen.add(i)
                found.append(i)
        if found_so_far:
            found_so_far(len(found))

    take(EARLIEST, until)
    return found
