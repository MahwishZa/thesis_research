"""G0: is the evidence that changed each verdict retrievable from PubMed, as of
the newest review's date? Network required - run on the local machine
(PubMed E-utilities are blocked in the cloud session that wrote this).

For every benchmark item it searches PubMed with the QUESTION ONLY (the system's
real input), restricted to records published BEFORE the newest review version
(no future leakage) and excluding the Cochrane Database (the review itself and
its versions). It stores the top-ranked PMIDs with dates, publication types and
journals, then counts trials / systematic reviews published inside the update
window (previous version, newest version] - the evidence an update could rest
on. This is a retrievability proxy, not the final candidate set (that comes
after MedCPT reranking).

    python -m experiments.medchange.pubmed_asof [--api-key KEY] [--limit N]

Resumable: each item's result is cached as one JSON file; reruns skip them.
NCBI usage: <= 3 requests/s without a key, <= 10 with one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_STOP = set("""a an the of for in on to and or with without is are was were be been
does do did can could should would may might will people person patients adults
children effective effectively effect effects help helps improve improves improving
reduce reduces treatment treating treat prevent preventing safe safety use using used
there any than more less better compared versus vs what which who how whether it its
this that these those from by as at into about""".split())
TRIAL_TYPES = ("Randomized Controlled Trial", "Clinical Trial", "Controlled Clinical Trial",
               "Clinical Trial, Phase III", "Clinical Trial, Phase II", "Clinical Trial, Phase IV")
REVIEW_TYPES = ("Systematic Review", "Meta-Analysis")


def query_terms(question: str, max_terms: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question.lower())
    out = []
    for w in words:
        if w not in _STOP and len(w) > 2 and w not in out:
            out.append(w)
    return out[:max_terms]


def build_term(terms: list[str], *, mode: str) -> str:
    if not terms:
        raise ValueError("no content words in question")
    joiner = " AND " if mode == "and" else " OR "
    return "(" + joiner.join(terms) + ') NOT "Cochrane Database Syst Rev"[Journal]'


def day_before(iso: str) -> str:
    return (dt.date.fromisoformat(iso) - dt.timedelta(days=1)).strftime("%Y/%m/%d")


def parse_sortpubdate(s: str) -> Optional[str]:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def window_counts(records: list[dict], after: str, until: str, top_k: int) -> dict:
    """Counts among the top_k records published in (after, until]."""
    win = [r for r in records[:top_k] if r.get("date") and after < r["date"] <= until]
    trial = sum(any(t in r["pubtypes"] for t in TRIAL_TYPES) for r in win)
    review = sum(any(t in r["pubtypes"] for t in REVIEW_TYPES) for r in win)
    return {"in_window": len(win), "trials_in_window": trial, "reviews_in_window": review}


class EUtils:
    def __init__(self, api_key: Optional[str] = None, *, opener: Callable = None,
                 sleep: Callable = time.sleep):
        self.api_key = api_key
        self.delay = 0.11 if api_key else 0.34
        self._open = opener or (lambda url: urllib.request.urlopen(url, timeout=60).read())
        self._sleep = sleep

    def _get(self, endpoint: str, params: dict) -> dict:
        params = dict(params, retmode="json", tool="thesis_research")
        if self.api_key:
            params["api_key"] = self.api_key
        url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            self._sleep(self.delay)
            try:
                return json.loads(self._open(url))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == 4:
                    raise RuntimeError(f"E-utilities failed after 5 attempts: {exc}") from exc
                self._sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def search(self, term: str, maxdate: str, retmax: int) -> tuple[int, list[str]]:
        r = self._get("esearch.fcgi", {"db": "pubmed", "term": term, "sort": "relevance",
                                       "datetype": "pdat", "mindate": "1800/01/01",
                                       "maxdate": maxdate, "retmax": retmax})["esearchresult"]
        return int(r.get("count", 0)), list(r.get("idlist", []))

    def summaries(self, pmids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(pmids), 200):
            batch = pmids[i:i + 200]
            res = self._get("esummary.fcgi", {"db": "pubmed", "id": ",".join(batch)})["result"]
            for p in batch:
                d = res.get(p, {})
                out.append({"pmid": p, "date": parse_sortpubdate(d.get("sortpubdate", "")),
                            "pubtypes": d.get("pubtype", []), "journal": d.get("source", "")})
        return out


def probe_item(item: dict, eu: EUtils, *, retmax: int, min_hits: int) -> dict:
    terms = query_terms(item["question"])
    maxdate = day_before(item["newest"]["date"])
    mode = "and"
    count, ids = eu.search(build_term(terms, mode="and"), maxdate, retmax)
    if count < min_hits:
        mode = "or"
        count, ids = eu.search(build_term(terms, mode="or"), maxdate, retmax)
    recs = eu.summaries(ids) if ids else []
    if any(r["date"] and r["date"] > item["newest"]["date"] for r in recs):
        raise RuntimeError(f"{item['item_id']}: a record postdates the cutoff - leakage")
    return {"item_id": item["item_id"], "query_terms": terms, "mode": mode,
            "maxdate": maxdate, "count": count, "records": recs,
            "top50": window_counts(recs, item["previous"]["date"], item["newest"]["date"], 50),
            "top200": window_counts(recs, item["previous"]["date"], item["newest"]["date"], 200)}


def summarize(results: list[dict], items: dict) -> dict:
    out = {}
    for kind in ("changed", "unchanged"):
        rs = [r for r in results if items[r["item_id"]]["kind"] == kind]
        if not rs:
            continue
        n = len(rs)
        out[kind] = {
            "n": n,
            "zero_hits": sum(r["count"] == 0 for r in rs),
            "share_with_trial_or_review_in_window_top50": round(sum(
                (r["top50"]["trials_in_window"] + r["top50"]["reviews_in_window"]) > 0
                for r in rs) / n, 4),
            "share_with_trial_or_review_in_window_top200": round(sum(
                (r["top200"]["trials_in_window"] + r["top200"]["reviews_in_window"]) > 0
                for r in rs) / n, 4),
            "median_in_window_top200": sorted(r["top200"]["in_window"] for r in rs)[n // 2],
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--benchmark", default=str(HERE / "data" / "benchmark.jsonl"))
    ap.add_argument("--cache-dir", default=str(HERE / "data" / "pubmed_g0"))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--retmax", type=int, default=200)
    ap.add_argument("--min-hits", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    ap.add_argument("--split", default="dev", choices=("dev", "confirm", "all"),
                    help="G0 is a dev-split gate by default")
    args = ap.parse_args(argv)

    items = [json.loads(l) for l in open(args.benchmark, encoding="utf-8")]
    items = [i for i in items if not i["likely_label_noise"]
             and (args.split == "all" or i["split"] == args.split)]
    if args.limit:
        items = items[:args.limit]
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    eu = EUtils(args.api_key)
    t0 = time.time()
    for k, it in enumerate(items, 1):
        f = cache / f"{it['item_id']}.json"
        if f.exists():
            continue
        res = probe_item(it, eu, retmax=args.retmax, min_hits=args.min_hits)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(res), encoding="utf-8")
        tmp.replace(f)
        if k % 10 == 0 or k == len(items):
            print(f"  {k}/{len(items)} items ({time.time() - t0:.0f}s)", flush=True)
    results = [json.loads((cache / f"{i['item_id']}.json").read_text(encoding="utf-8"))
               for i in items if (cache / f"{i['item_id']}.json").exists()]
    summary = summarize(results, {i["item_id"]: i for i in items})
    print(json.dumps(summary, indent=2))
    ch = summary.get("changed", {})
    if ch:
        ok = ch["share_with_trial_or_review_in_window_top200"] >= 0.5
        print(f"G0 (pre-stated): >= 50% of changed items have a trial or review in the "
              f"update window among the as-of candidates -> {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
