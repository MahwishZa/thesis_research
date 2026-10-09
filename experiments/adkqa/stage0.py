"""Stage 0 of the Alzheimer's-specific evaluation: are there enough eligible source records? (needs network; no model)

The question set AD-KQA (``docs/protocol.md`` §8) is built from PubMed records on Alzheimer's disease of the types systematic
review, meta-analysis or guideline, published after the generator's stated knowledge cutoff. This script only COUNTS such
records, per knowledge area (through MeSH qualifiers), and evaluates the go/no-go criterion of the protocol. It reads no
abstract, builds no question and runs no model.

    python -m experiments.adkqa.stage0 [--api-key KEY] [--since 2023-04-01] [--until 2026-10-09] [--probe-references 20]

Writes ``experiments/adkqa/results/stage0_counts.json``: counts and PubMed identifiers only, no source text.
``--probe-references N`` asks Europe PMC for the reference lists of N sampled records, to see whether a recall measure
against the studies a source cites is possible (an optional, secondary retrieval measure).
NCBI usage: at most 3 requests per second without a key, 10 with one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

from experiments.medchange.pubmed_asof import EUtils

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "stage0_counts.json"

#: Knowledge areas and the MeSH qualifiers of "Alzheimer Disease" used to sample them (``docs/protocol.md`` §8). MEDLINE
#: indexing is partly automated since 2022, so the qualifiers sample areas; they are not a ground truth.
CORE_AREAS = {
    "treatment": ("drug therapy", "therapy"),
    "prevention": ("prevention & control",),
    "diagnosis": ("diagnosis", "diagnostic imaging", "blood", "cerebrospinal fluid"),
    "causes_risk": ("etiology", "genetics"),
    "progression": ("physiopathology", "mortality"),
    "symptoms": ("psychology", "complications"),
    "care_management": ("nursing", "rehabilitation"),
}
#: Strata added only if the pilot passes; counted for information, outside the go/no-go criterion.
OPTIONAL_AREAS = {
    "background_history": ("history",),
    "terminology": ("classification",),
}
TYPES = ("Systematic Review", "Meta-Analysis", "Practice Guideline", "Guideline")
CONDITION = '"Alzheimer Disease"[majr]'
FILTERS = 'hasabstract[text] AND medline[sb] AND english[lang] NOT "retracted publication"[pt]'

#: The go/no-go criterion (``docs/protocol.md`` §8): enough records in total and enough covered areas.
MIN_TOTAL = 700
MIN_AREA = 100
MIN_AREAS = 6
RETMAX = 10000
SAMPLE_SIZE = 20


def type_clause(types: Sequence[str] = TYPES) -> str:
    return "(" + " OR ".join(f'"{t}"[pt]' for t in types) + ")"


def build_term(qualifier: Optional[str] = None, types: Sequence[str] = TYPES) -> str:
    """The PubMed term of the eligible records, optionally restricted to one MeSH qualifier of Alzheimer Disease."""
    condition = f'"Alzheimer Disease/{qualifier}"[majr]' if qualifier else CONDITION
    return f"{condition} AND {type_clause(types)} AND {FILTERS}"


def search(eu: EUtils, term: str, since: str, until: str) -> tuple[list[str], dict]:
    """All identifiers of ``term`` published in [since, until] (ISO dates), with what PubMed says about the query (how it
    was translated and any warning), so that a syntax problem shows in the output instead of as a silent zero.
    Refuses a truncated list."""
    r = eu._get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": RETMAX, "datetype": "pdat",
                                 "mindate": since.replace("-", "/"), "maxdate": until.replace("-", "/")})["esearchresult"]
    ids, count = list(r.get("idlist", [])), int(r.get("count", 0))
    if len(ids) < count:
        raise RuntimeError(f"{count} records but only {len(ids)} returned (limit {RETMAX}); narrow the window")
    notes = {k: r[k] for k in ("querytranslation", "warninglist", "errorlist") if r.get(k)}
    return ids, notes


def search_ids(eu: EUtils, term: str, since: str, until: str) -> list[str]:
    return search(eu, term, since, until)[0]


def periods(since: str, until: str) -> list[tuple[str, str]]:
    """Calendar-year slices of the window, for spotting indexing lag in the newest months."""
    a, b = dt.date.fromisoformat(since), dt.date.fromisoformat(until)
    out, start = [], a
    while start <= b:
        end = min(dt.date(start.year, 12, 31), b)
        out.append((start.isoformat(), end.isoformat()))
        start = end + dt.timedelta(days=1)
    return out


def evaluate_gate(total: int, distinct: dict[str, int]) -> dict:
    """Go when there are at least MIN_TOTAL eligible records and at least MIN_AREAS core areas with at least MIN_AREA
    distinct records each. Areas below the threshold are reported as not covered (they are not a reason to stop alone)."""
    covered = sorted(a for a in CORE_AREAS if distinct.get(a, 0) >= MIN_AREA)
    below = sorted(a for a in CORE_AREAS if distinct.get(a, 0) < MIN_AREA)
    total_ok, areas_ok = total >= MIN_TOTAL, len(covered) >= MIN_AREAS
    return {"criterion": {"min_total": MIN_TOTAL, "min_per_area": MIN_AREA, "min_areas": MIN_AREAS},
            "total": total, "total_ok": total_ok, "areas_covered": covered, "areas_below": below,
            "areas_ok": areas_ok, "go": total_ok and areas_ok}


def sample_pmids(ids: Sequence[str], n: int = SAMPLE_SIZE) -> list[str]:
    """A fixed pseudo-random sample for manual spot checks (hash order; the same records on every run)."""
    return sorted(ids, key=lambda p: hashlib.sha256(f"adk-stage0|{p}".encode()).hexdigest())[:n]


def probe_references(pmids: Sequence[str], opener: Optional[Callable[[str], bytes]] = None) -> dict:
    """Do Europe PMC reference lists exist for the sampled records, and how many references carry a PubMed identifier?
    Field names follow the Europe PMC REST documentation as I understand it and were not tested against the live service,
    so the raw failure is reported rather than hidden."""
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    rows = []
    for p in pmids:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{p}/references?format=json&pageSize=1000"
        try:
            body = json.loads(opener(url))
            refs = (body.get("referenceList") or {}).get("reference") or []
            rows.append({"pmid": p, "references": len(refs),
                         "with_pubmed_id": sum(1 for r in refs if r.get("id") and r.get("source") == "MED")})
        except Exception as exc:  # noqa: BLE001 - a probe must report, not crash
            rows.append({"pmid": p, "error": f"{type(exc).__name__}: {exc}"})
    usable = [r for r in rows if r.get("with_pubmed_id", 0) >= 10]
    return {"sampled": len(rows), "with_at_least_10_pubmed_references": len(usable),
            "share": round(len(usable) / len(rows), 3) if rows else None,
            "threshold_for_a_recall_measure": 0.6, "rows": rows}


def collect(eu: EUtils, since: str, until: str) -> dict:
    base_term = build_term()
    main_ids, main_notes = search(eu, base_term, since, until)
    notes = {"main": main_notes} if main_notes else {}
    by_type = {t: len(search_ids(eu, build_term(types=(t,)), since, until)) for t in TYPES}
    by_period = {f"{a}..{b}": len(search_ids(eu, base_term, a, b)) for a, b in periods(since, until)}
    per_area = {}
    for area, qualifiers in {**CORE_AREAS, **OPTIONAL_AREAS}.items():
        sets, counts = set(), {}
        for q in qualifiers:
            ids, n = search(eu, build_term(q), since, until)
            counts[q] = len(ids)
            sets |= set(ids)
            if n.get("warninglist") or n.get("errorlist"):
                notes[f"{area}/{q}"] = n
        per_area[area] = {"core": area in CORE_AREAS, "qualifier_counts": counts, "distinct": len(sets)}
    return {"stage": 0, "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window": {"since": since, "until": until}, "base_term": base_term, "total": len(main_ids),
            "by_type": by_type, "by_period": by_period, "per_area": per_area,
            "gate": evaluate_gate(len(main_ids), {a: v["distinct"] for a, v in per_area.items()}),
            "pubmed_notes": notes, "sample_pmids": sample_pmids(main_ids), "references_probe": None}


def render(result: dict) -> str:
    g = result["gate"]
    lines = [f"Eligible records {result['window']['since']}..{result['window']['until']}: {result['total']}",
             "by type: " + ", ".join(f"{k} {v}" for k, v in result["by_type"].items()),
             "by period: " + ", ".join(f"{k} {v}" for k, v in result["by_period"].items()), "",
             f"{'area':<20}{'distinct':>9}  qualifiers"]
    for area, v in result["per_area"].items():
        lines.append(f"{area + ('' if v['core'] else ' (optional)'):<20}{v['distinct']:>9}  "
                     + ", ".join(f"{q} {n}" for q, n in v["qualifier_counts"].items()))
    lines += ["", f"criterion: total >= {g['criterion']['min_total']} and >= {g['criterion']['min_areas']} core areas with "
              f">= {g['criterion']['min_per_area']} distinct records",
              f"total ok: {g['total_ok']}; areas covered: {', '.join(g['areas_covered']) or 'none'}; "
              f"below: {', '.join(g['areas_below']) or 'none'}", "GO" if g["go"] else "NO-GO (do not build; report these counts)"]
    if result.get("pubmed_notes"):
        lines += ["", "PubMed reported notes on some queries (check that no term was ignored):"]
        for key, note in result["pubmed_notes"].items():
            lines.append(f"  {key}: " + json.dumps(note)[:400])
    probe = result.get("references_probe")
    if probe:
        lines += ["", f"reference lists: {probe['with_at_least_10_pubmed_references']} of {probe['sampled']} sampled records have "
                  f">= 10 references with a PubMed identifier (share {probe['share']}; a recall measure needs >= "
                  f"{probe['threshold_for_a_recall_measure']})"]
    return "\n".join(lines)


def main(argv=None, *, eu: Optional[EUtils] = None, ref_opener: Optional[Callable[[str], bytes]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--since", default="2023-04-01", help="first publication date (after the generator's cutoff)")
    ap.add_argument("--until", default=dt.date.today().isoformat())
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--probe-references", type=int, default=0, metavar="N")
    args = ap.parse_args(argv)
    eu = eu or EUtils(args.api_key)
    result = collect(eu, args.since, args.until)
    if args.probe_references:
        result["references_probe"] = probe_references(result["sample_pmids"][:args.probe_references], ref_opener)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(render(result))
    print(f"\nwritten: {out}")
    return 0 if result["gate"]["go"] else 3


if __name__ == "__main__":
    sys.exit(main())
