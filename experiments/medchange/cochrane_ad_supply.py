"""How many new Cochrane dementia and Alzheimer's reviews exist after the MedRevQA snapshot? (counts only; needs the network; no model)

    python -m experiments.medchange.cochrane_ad_supply --medchange-dir ..\\MedChange [--api-key KEY]

MedRevQA ends at a date (its newest review); this tool asks PubMed for Cochrane reviews with Alzheimer Disease or Dementia as a major topic published
after it, fetches those records, and counts how many could become a question by a fixed rule: an abstract with an
"Authors' conclusions" section, a title of the form "X for Y" (intervention: "Is X effective for Y?"; a title with "diagnos" or "accuracy":
"Can X be used for Y?"), and no protocol or withdrawn notice. It selects nothing. The counts and PubMed identifiers go to
``results/cochrane_ad_supply.json`` (no titles, no abstract text); the titles are only printed, for you to read, and are not saved."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from .benchmark import parse_cochrane_date, read_csv
from .pubmed_asof import EUTILS, EUtils

HERE = Path(__file__).resolve().parent
JOURNAL = '"Cochrane Database Syst Rev"[Journal]'
TOPIC = '(Alzheimer Disease[majr] OR Dementia[majr])'
TITLE_FORM = re.compile(r"^(?P<x>.+?)\s+for\s+(?P<y>.+?)\.?$", re.IGNORECASE)
GO, PILOT = 30, 20


def snapshot_end(medchange_dir: Optional[Path], benchmark: Path) -> str:
    """The date of the newest review of MedRevQA (or, without the files, of the benchmark)."""
    if medchange_dir and (medchange_dir / "Datasets" / "MedRevQA.csv").is_file():
        dates = []
        for r in read_csv(medchange_dir / "Datasets" / "MedRevQA.csv"):
            try:
                dates.append(parse_cochrane_date(r["DOI_Date"])[0])
            except Exception:
                continue
        return max(dates)
    rows = [json.loads(l) for l in benchmark.read_text(encoding="utf-8").splitlines() if l.strip()]
    return max(r["newest"]["date"] for r in rows)


def search(eu: EUtils, term: str, since: str, until: str, retmax: int = 1000) -> list[str]:
    p = {"db": "pubmed", "term": term, "datetype": "pdat", "mindate": since.replace("-", "/"), "maxdate": until.replace("-", "/"), "retmax": retmax}
    r = eu._get("esearch.fcgi", p)["esearchresult"]
    if int(r.get("count", 0)) > retmax:
        raise RuntimeError(f"{r['count']} records for {term!r}: more than retmax")
    return list(r.get("idlist", []))


def fetch_xml(eu: EUtils, pmids: Sequence[str]) -> bytes:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": "thesis_research"}
    if eu.api_key:
        params["api_key"] = eu.api_key
    url = EUTILS + "efetch.fcgi?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        eu._sleep(eu.delay)
        try:
            data = eu._open(url)
            ET.fromstring(data)
            return data
        except (urllib.error.URLError, TimeoutError, ET.ParseError, OSError) as exc:
            if attempt == 4:
                raise RuntimeError(f"efetch failed after 5 attempts: {exc}") from exc
            eu._sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse(xml: bytes) -> list[dict]:
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        sections = [((a.get("Label") or "").strip().lower().replace("’", "'"), _text(a)) for a in art.findall(".//Abstract/AbstractText")]
        out.append({"pmid": _text(art.find("./MedlineCitation/PMID")), "title": _text(art.find(".//ArticleTitle")),
                    "has_conclusions": any(l.startswith("authors' conclusions") or l.startswith("conclusions") for l, t in sections if t),
                    "pubtypes": [_text(p) for p in art.findall(".//PublicationType")],
                    "majors": [_text(d) for d in art.findall(".//MeshHeading/DescriptorName") if d.get("MajorTopicYN") == "Y"]})
    return out


def classify(rec: dict) -> dict:
    """What the fixed rule makes of a record: its question template, or the reason it is out."""
    title = rec["title"].strip()
    if re.search(r"\b(protocol|withdrawn)\b", title, re.IGNORECASE) or any("withdrawn" in p.lower() for p in rec["pubtypes"]):
        return {"status": "protocol_or_withdrawn"}
    if not rec["has_conclusions"]:
        return {"status": "no_authors_conclusions"}
    m = TITLE_FORM.match(re.sub(r"^\[|\]$", "", title))
    if not m:
        return {"status": "title_not_x_for_y"}
    template = "test" if re.search(r"diagnos|accuracy", title, re.IGNORECASE) else "effect"
    return {"status": "usable", "template": template, "names_alzheimer": "alzheimer" in title.lower()}


def collect(eu: EUtils, since: str, until: str) -> dict:
    ids = search(eu, f"{JOURNAL} AND hasabstract[text] AND {TOPIC}", since, until)
    only_ad = set(search(eu, f"{JOURNAL} AND hasabstract[text] AND Alzheimer Disease[majr]", since, until))
    recs = []
    for i in range(0, len(ids), 100):
        recs += parse(fetch_xml(eu, ids[i:i + 100]))
    dates = {s["pmid"]: s["sortpubdate"] for s in eu.summaries(ids)} if ids else {}
    rows = []
    for r in recs:
        d = dates.get(r["pmid"])
        c = classify(r)
        if d is None or d <= since:
            c = {"status": "not_after_snapshot"}
        rows.append(dict(c, pmid=r["pmid"], date=d, alzheimer_major=r["pmid"] in only_ad, title=r["title"]))
    status = Counter(r["status"] for r in rows)
    usable = [r for r in rows if r["status"] == "usable"]
    return {"window": {"after": since, "until": until}, "records_dementia_or_alzheimer_major": len(rows), "alzheimer_major": len(only_ad),
            "status": dict(status), "usable": len(usable), "usable_by_template": dict(Counter(r["template"] for r in usable)),
            "usable_alzheimer_major": sum(r["alzheimer_major"] for r in usable),
            "usable_title_names_alzheimer": sum(r["names_alzheimer"] for r in usable),
            "usable_by_year": dict(sorted(Counter(r["date"][:4] for r in usable).items())),
            "decision": "GO" if len(usable) >= GO else "PILOT" if len(usable) >= PILOT else "TOO FEW",
            "usable_pmids": sorted(r["pmid"] for r in usable), "_titles": {r["pmid"]: r["title"] for r in usable}}


def main(argv=None, *, eu: Optional[EUtils] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--medchange-dir", default=None)
    ap.add_argument("--since", default=None, help="override the end of the MedRevQA snapshot (YYYY-MM-DD)")
    ap.add_argument("--until", default=dt.date.today().isoformat())
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args(argv)
    since = a.since or snapshot_end(Path(a.medchange_dir) if a.medchange_dir else None, Path(a.data_dir) / "benchmark.jsonl")
    rep = collect(eu or EUtils(a.api_key), since, a.until)
    titles = rep.pop("_titles")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cochrane_ad_supply.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "usable_pmids"}, indent=2))
    print("\nUsable reviews (titles are printed for you to read and are not saved):")
    for p in rep["usable_pmids"]:
        print(f"  {p}  {titles[p]}")
    print(f"\nDecision rule: {GO} or more = GO; {PILOT} to {GO - 1} = a small pilot; fewer = too few.  Result: {rep['decision']}")
    return 0 if rep["decision"] != "TOO FEW" else 3


if __name__ == "__main__":
    sys.exit(main())
