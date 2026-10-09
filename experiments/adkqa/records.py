"""Source records of the Alzheimer's-specific question set: fetching and structuring (needs the network only in ``fetch``).

The source list is frozen in ``experiments/adkqa/results/stage0_counts_r2.json`` (``docs/protocol.md`` §8); this module never
searches. Abstract text is publisher text: it is written to ``experiments/adkqa/data/`` (not tracked) and only counts, hashes and
offsets reach the tracked results."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence

from experiments.medchange.abstracts import _SENTENCE
from experiments.medchange.pubmed_asof import EUTILS, EUtils

from . import spec, stage0

SOURCE_FILE = Path(__file__).resolve().parent / "results" / "stage0_counts_r2.json"


def area_assignment(counts: dict) -> dict[str, str]:
    """pmid -> covered area; a record in two covered areas serves the scarcer one (protocol §8, amendment)."""
    gate = stage0.evaluate_gate_v2(counts)
    covered = gate["areas_covered"]
    out: dict[str, str] = {}
    for area in sorted(covered, key=lambda a: (counts["per_area"][a]["distinct"], a)):
        for pmid in counts["per_area"][area]["ids"]:
            out.setdefault(pmid, area)
    return out


def _text(el: Optional[ET.Element]) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_efetch(xml: bytes) -> list[dict]:
    """One dict per PubmedArticle: title, abstract sections [(label, text)], publication types, major MeSH descriptors."""
    root = ET.fromstring(xml)
    out = []
    for art in root.iter("PubmedArticle"):
        pmid = _text(art.find("./MedlineCitation/PMID"))
        sections = [((a.get("Label") or "").strip(), _text(a)) for a in art.findall(".//Abstract/AbstractText")]
        out.append({"pmid": pmid, "title": _text(art.find(".//ArticleTitle")),
                    "sections": [(l, t) for l, t in sections if t],
                    "pubtypes": [_text(p) for p in art.findall(".//PublicationType")],
                    "major_mesh": [_text(d) for d in art.findall(".//MeshHeading/DescriptorName")
                                   if d.get("MajorTopicYN") == "Y"]})
    return out


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


def fetch_records(eu: EUtils, pmids: Sequence[str], batch: int = 100) -> list[dict]:
    pmids = list(pmids)
    dates = {s["pmid"]: s["sortpubdate"] for s in eu.summaries(pmids)}
    out = []
    for i in range(0, len(pmids), batch):
        for rec in parse_efetch(fetch_xml(eu, pmids[i:i + batch])):
            rec["date"] = dates.get(rec["pmid"])
            out.append(rec)
    return out


def structure(rec: dict) -> dict:
    """Abstract as plain text, the conclusion (offsets and hash) and the objectives, by the rules of protocol §8."""
    texts = [t for _, t in rec["sections"]]
    plain = " ".join(texts)
    low = lambda s: s.lower().replace("\u2019", "'").rstrip(":").strip()
    is_conclusion = lambda l: low(l) in spec.CONCLUSION_SECTIONS or low(l).startswith("conclusion")
    conclusion = next((t for l, t in rec["sections"] if is_conclusion(l)), None)
    if conclusion is None and len(rec["sections"]) == 1 and not rec["sections"][0][0]:
        last = [s for s in _SENTENCE.split(plain.strip()) if s][-1:]
        if last and spec.UNSTRUCTURED_CONCLUSION.match(last[0]):
            conclusion = last[0]
    objectives = next((t for l, t in rec["sections"] if low(l) in spec.OBJECTIVE_SECTIONS), None)
    if objectives is None and plain:
        objectives = ([s for s in _SENTENCE.split(plain.strip()) if s] or [""])[0]
    span = None
    if conclusion:
        start = plain.rfind(conclusion)
        span = [start, start + len(conclusion)]
    return {"abstract": plain, "conclusion": conclusion, "conclusion_span": span,
            "conclusion_sha256": hashlib.sha256(conclusion.encode()).hexdigest() if conclusion else None,
            "objectives": objectives or ""}


def prepare(records: Sequence[dict], areas: dict[str, str], since: str = "2023-04-01") -> list[dict]:
    """One row per frozen source record, with its area, cluster, split and eligibility. Every source record appears (an
    ineligible one counts against the survival rate)."""
    by = {r["pmid"]: r for r in records}
    rows = []
    for pmid, area in sorted(areas.items()):
        rec = by.get(pmid)
        row = {"pmid": pmid, "area": area, "status": "eligible"}
        if rec is None:
            row.update(status="not_fetched", cluster=f"pmid:{pmid}")
        else:
            row.update(title=rec["title"], date=rec["date"], pubtypes=rec["pubtypes"], major_mesh=rec["major_mesh"],
                       cluster=spec.cluster_key(rec["major_mesh"], pmid), **structure(rec))
            if not rec["sections"]:
                row["status"] = "no_abstract"
            elif not row["conclusion"]:
                row["status"] = "no_conclusion"
            elif not rec["date"] or rec["date"] < since:
                row["status"] = "outside_window"
        row["split"] = spec.split_of(row["cluster"])
        rows.append(row)
    return rows


def pools(rows: Sequence[dict]) -> dict:
    out = {"records": len(rows), "by_status": {}, "by_split": {}, "by_area_split": {}, "clusters": len({r["cluster"] for r in rows})}
    for r in rows:
        out["by_status"][r["status"]] = out["by_status"].get(r["status"], 0) + 1
        out["by_split"][r["split"]] = out["by_split"].get(r["split"], 0) + 1
        k = f"{r['area']}/{r['split']}"
        out["by_area_split"][k] = out["by_area_split"].get(k, 0) + 1
    out["dev_drafted"] = min(spec.DRAFT_N, out["by_split"].get("dev", 0))
    return out


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
