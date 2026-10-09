"""Stage 0 of the Alzheimer's-specific evaluation: are there enough eligible source records? (needs network; no model)

The question set AD-KQA (``docs/protocol.md`` §8) is built from PubMed records on Alzheimer's disease of the types systematic
review, meta-analysis or guideline, published after the generator's stated knowledge cutoff. This script only COUNTS such
records, per knowledge area (through MeSH qualifiers), and evaluates the go/no-go criterion of the protocol. It reads no
abstract, builds no question and runs no model.

    python -m experiments.adkqa.stage0 [--api-key KEY] [--since 2023-04-01] [--until 2026-10-09] [--probe-references 20]

Writes ``experiments/adkqa/results/stage0_counts_r2.json``: counts and PubMed identifiers only, no source text.
``--probe-references N`` asks Europe PMC for the reference lists of N sampled records, to see whether a recall measure
against the studies a source cites is possible (an optional, secondary retrieval measure).
NCBI usage: at most 3 requests per second without a key, 10 with one.

Revision 2 (after the first run, ``stage0_counts.json``, which gave NO-GO): the qualifier "prevention and control" was written
with an ampersand and PubMed did not find it; PubMed's warnings are now shown only when they say something. The criterion and
its gate are unchanged. Added, for information only and not part of the gate: counts by MeSH main headings, the records
with Alzheimer in the title but not as major MeSH topic, the pre-cutoff window, the identifier lists, and an exclusive
assignment (each record serves one area, scarcest area first), which is the supply that matters when one question is made
per source record.
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
DEFAULT_OUT = HERE / "results" / "stage0_counts_r2.json"
REVISION = 2

#: Knowledge areas and the MeSH qualifiers of "Alzheimer Disease" used to sample them (``docs/protocol.md`` §8). MEDLINE
#: indexing is partly automated since 2022, so the qualifiers sample areas; they are not a ground truth.
CORE_AREAS = {
    "treatment": ("drug therapy", "therapy"),
    "prevention": ("prevention and control",),
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
#: Information only (not part of the gate): MeSH main headings that indexers assign to records of each area, because MeSH
#: qualifiers of "Alzheimer Disease" alone miss areas that are indexed under other headings. A misspelt heading is
#: reported by PubMed's warning, not silently counted as zero.
AREA_HEADINGS = {
    "treatment": ("Cholinesterase Inhibitors", "Antibodies, Monoclonal", "Memantine", "Donepezil", "Neuroprotective Agents",
                  "Dietary Supplements"),
    "prevention": ("Primary Prevention", "Secondary Prevention", "Risk Reduction Behavior", "Protective Factors", "Exercise",
                   "Life Style", "Diet, Mediterranean"),
    "diagnosis": ("Biomarkers", "Positron-Emission Tomography", "Magnetic Resonance Imaging", "tau Proteins",
                  "Amyloid beta-Peptides", "Neuropsychological Tests"),
    "causes_risk": ("Risk Factors", "Genetic Predisposition to Disease", "Apolipoprotein E4",
                    "Polymorphism, Single Nucleotide", "Environmental Exposure"),
    "progression": ("Disease Progression", "Prognosis", "Survival Analysis"),
    "symptoms": ("Signs and Symptoms", "Behavioral Symptoms", "Depression", "Anxiety", "Psychomotor Agitation",
                 "Sleep Wake Disorders", "Psychotic Disorders", "Apathy"),
    "care_management": ("Caregivers", "Quality of Life", "Palliative Care", "Nursing Care", "Rehabilitation"),
    "background_history": ("History, 20th Century", "History, 21st Century"),
    "terminology": ("Terminology as Topic",),
}
TYPES = ("Systematic Review", "Meta-Analysis", "Practice Guideline", "Guideline")
CONDITION = '"Alzheimer Disease"[majr]'
FILTERS = 'hasabstract[text] AND medline[sb] AND english[lang] NOT "retracted publication"[pt]'

#: The go/no-go criterion (``docs/protocol.md`` §8): enough records in total and enough covered areas.
MIN_TOTAL = 700
MIN_AREA = 100
MIN_AREAS = 6
#: Amended criterion (``docs/protocol.md`` §8, amendment of 2026-10-09 after Stage 0 run 2): the areas that reach MIN_AREA
#: are the covered areas; at least MIN_AREAS_V2 are required, and the distinct records of the covered areas must be enough for
#: 200 test + 60 development questions at the 40% draft survival of the trial-run gate (260 / 0.40 = 650).
MIN_AREAS_V2 = 4
MIN_COVERED_SUPPLY = 650
RETMAX = 10000
SAMPLE_SIZE = 20


def type_clause(types: Sequence[str] = TYPES) -> str:
    return "(" + " OR ".join(f'"{t}"[pt]' for t in types) + ")"


def build_term(qualifier: Optional[str] = None, types: Sequence[str] = TYPES, *, headings: Sequence[str] = (),
               condition: Optional[str] = None) -> str:
    """The PubMed term of the eligible records: optionally restricted to one MeSH qualifier of Alzheimer Disease, or to
    records also indexed under any of ``headings`` (information only), or with another ``condition`` clause."""
    cond = condition or (f'"Alzheimer Disease/{qualifier}"[majr]' if qualifier else CONDITION)
    term = f"{cond} AND {type_clause(types)} AND {FILTERS}"
    if headings:
        term += " AND (" + " OR ".join(f'"{h}"[mh]' for h in headings) + ")"
    return term


def meaningful(notes: dict) -> dict:
    """Keep what PubMed says only when it says something: a phrase it did not find or ignored, or an error. Its routine
    message about the result limit is dropped."""
    warn = notes.get("warninglist") or {}
    bad = {k: v for k, v in warn.items() if k in ("phrasesignored", "quotedphrasesnotfound") and v}
    out = {}
    if bad:
        out["warninglist"] = bad
    if notes.get("errorlist"):
        out["errorlist"] = notes["errorlist"]
    return out


def search(eu: EUtils, term: str, since: str, until: str) -> tuple[list[str], dict]:
    """All identifiers of ``term`` published in [since, until] (ISO dates), with what PubMed says about the query (how it
    was translated, any phrase it did not find), so that a syntax problem shows in the output instead of as a silent zero.
    Refuses a truncated list."""
    r = eu._get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": RETMAX, "datetype": "pdat",
                                 "mindate": since.replace("-", "/"), "maxdate": until.replace("-", "/")})["esearchresult"]
    ids, count = list(r.get("idlist", [])), int(r.get("count", 0))
    if len(ids) < count:
        raise RuntimeError(f"{count} records but only {len(ids)} returned (limit {RETMAX}); narrow the window")
    notes = meaningful(r)
    if notes and r.get("querytranslation"):
        notes["querytranslation"] = r["querytranslation"]
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


def evaluate_gate_v2(result: dict) -> dict:
    """The amended criterion (governing since 2026-10-09; the original is ``evaluate_gate``), computed from a Stage 0 result (also from a committed file, offline)."""
    core = {a: v for a, v in result["per_area"].items() if v["core"]}
    covered = sorted(a for a, v in core.items() if v["distinct"] >= MIN_AREA)
    below = sorted(a for a, v in core.items() if v["distinct"] < MIN_AREA)
    supply = len(set().union(*[set(core[a]["ids"]) for a in covered])) if covered else 0
    total_ok, areas_ok, supply_ok = (result["total"] >= MIN_TOTAL, len(covered) >= MIN_AREAS_V2, supply >= MIN_COVERED_SUPPLY)
    return {"criterion": {"min_total": MIN_TOTAL, "min_per_area": MIN_AREA, "min_areas": MIN_AREAS_V2,
                          "min_covered_supply": MIN_COVERED_SUPPLY},
            "total": result["total"], "total_ok": total_ok, "areas_covered": covered, "areas_below": below, "areas_ok": areas_ok,
            "covered_supply": supply, "supply_ok": supply_ok, "go": total_ok and areas_ok and supply_ok}


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


def exclusive_counts(sets: dict[str, set]) -> dict[str, int]:
    """Supply when each record may serve one area only (one question per source record): areas are served scarcest first."""
    remaining = set().union(*sets.values()) if sets else set()
    out = {}
    for area in sorted(sets, key=lambda a: (len(sets[a]), a)):
        take = sets[area] & remaining
        out[area] = len(take)
        remaining -= take
    return {a: out[a] for a in sets}


def collect(eu: EUtils, since: str, until: str, pre_since: str = "2021-04-01") -> dict:
    result = _collect(eu, since, until, pre_since)
    result["gate_amended"] = evaluate_gate_v2(result)
    return result


def _collect(eu: EUtils, since: str, until: str, pre_since: str) -> dict:
    base_term = build_term()
    main_ids, main_notes = search(eu, base_term, since, until)
    notes = {"main": main_notes} if main_notes else {}
    by_type = {t: len(search_ids(eu, build_term(types=(t,)), since, until)) for t in TYPES}
    by_period = {f"{a}..{b}": len(search_ids(eu, base_term, a, b)) for a, b in periods(since, until)}
    q_sets, h_sets, per_area = {}, {}, {}
    for area in {**CORE_AREAS, **OPTIONAL_AREAS}:
        qualifiers = {**CORE_AREAS, **OPTIONAL_AREAS}[area]
        ids_q, counts = set(), {}
        for q in qualifiers:
            ids, n = search(eu, build_term(q), since, until)
            counts[q] = len(ids)
            ids_q |= set(ids)
            if n:
                notes[f"{area}/{q}"] = n
        ids_h, n = search(eu, build_term(headings=AREA_HEADINGS[area]), since, until)
        if n:
            notes[f"{area}/headings"] = n
        q_sets[area], h_sets[area] = ids_q, set(ids_h)
        per_area[area] = {"core": area in CORE_AREAS, "qualifier_counts": counts, "distinct": len(ids_q),
                          "headings_distinct": len(h_sets[area]), "either_distinct": len(ids_q | h_sets[area]),
                          "ids": sorted(ids_q)}
    either = {a: q_sets[a] | h_sets[a] for a in q_sets}
    pool = set(main_ids)
    title_only, n_t = search(eu, build_term(condition="(Alzheimer*[ti] NOT \"Alzheimer Disease\"[majr])"), since, until)
    pre_until = (dt.date.fromisoformat(since) - dt.timedelta(days=1)).isoformat()
    pre, n_p = search(eu, base_term, pre_since, pre_until)
    for key, n in (("title_only", n_t), ("pre_cutoff_window", n_p)):
        if n:
            notes[key] = n
    return {"stage": 0, "revision": REVISION, "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window": {"since": since, "until": until}, "base_term": base_term, "total": len(main_ids),
            "by_type": by_type, "by_period": by_period, "per_area": per_area,
            "gate": evaluate_gate(len(main_ids), {a: v["distinct"] for a, v in per_area.items()}),
            "gate_amended": None,
            "information_only": {
                "alzheimer_in_title_not_major_topic": len(title_only),
                "pre_cutoff_window": {"since": pre_since, "until": pre_until, "total": len(pre)},
                "records_in_no_qualifier_area": len(pool - set().union(*q_sets.values())),
                "exclusive_supply_qualifier_frame": exclusive_counts(q_sets),
                "exclusive_supply_qualifier_or_heading_frame": exclusive_counts(either)},
            "pool_ids": sorted(main_ids), "pubmed_notes": notes, "sample_pmids": sample_pmids(main_ids),
            "references_probe": None}


def render(result: dict) -> str:
    g = result["gate"]
    info = result.get("information_only") or {}
    lines = [f"Eligible records {result['window']['since']}..{result['window']['until']}: {result['total']}",
             "by type: " + ", ".join(f"{k} {v}" for k, v in result["by_type"].items()),
             "by period: " + ", ".join(f"{k} {v}" for k, v in result["by_period"].items()), "",
             f"{'area':<26}{'qualifiers':>11}{'headings':>10}{'either':>8}  qualifier counts"]
    for area, v in result["per_area"].items():
        lines.append(f"{area + ('' if v['core'] else ' (optional)'):<26}{v['distinct']:>11}{v.get('headings_distinct', 0):>10}"
                     f"{v.get('either_distinct', 0):>8}  " + ", ".join(f"{q} {n}" for q, n in v["qualifier_counts"].items()))
    lines += ["", f"original criterion (qualifier frame): total >= {g['criterion']['min_total']} and >= "
              f"{g['criterion']['min_areas']} core areas with >= {g['criterion']['min_per_area']} distinct records",
              f"total ok: {g['total_ok']}; areas covered: {', '.join(g['areas_covered']) or 'none'}; "
              f"below: {', '.join(g['areas_below']) or 'none'}", "GO" if g["go"] else "NO-GO"]
    a = result.get("gate_amended") or evaluate_gate_v2(result)
    lines += ["", f"amended criterion (2026-10-09): total >= {a['criterion']['min_total']}, >= {a['criterion']['min_areas']} core "
              f"areas with >= {a['criterion']['min_per_area']} distinct records, and >= {a['criterion']['min_covered_supply']} "
              f"distinct records in the covered areas",
              f"total ok: {a['total_ok']}; covered: {', '.join(a['areas_covered']) or 'none'}; not covered: "
              f"{', '.join(a['areas_below']) or 'none'}; covered supply {a['covered_supply']}",
              "GO" if a["go"] else "NO-GO (do not build; report these counts)"]
    if info:
        lines += ["", "Information only (not part of the gate):",
                  f"  Alzheimer in the title but not a major MeSH topic: {info['alzheimer_in_title_not_major_topic']}",
                  f"  pool in the window before the cutoff {info['pre_cutoff_window']['since']}..{info['pre_cutoff_window']['until']}: "
                  f"{info['pre_cutoff_window']['total']}",
                  f"  records in no qualifier area: {info['records_in_no_qualifier_area']}",
                  "  exclusive supply (one question per record, scarcest area first):"]
        for key, label in (("exclusive_supply_qualifier_frame", "qualifier frame"),
                           ("exclusive_supply_qualifier_or_heading_frame", "qualifier or heading frame")):
            lines.append(f"    {label}: " + ", ".join(f"{a} {n}" for a, n in info[key].items()))
    if result.get("pubmed_notes"):
        lines += ["", "PubMed notes (a phrase it did not find or ignored; check that no term was dropped):"]
        for key, note in result["pubmed_notes"].items():
            lines.append(f"  {key}: " + json.dumps({k: v for k, v in note.items() if k != 'querytranslation'})[:300])
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
    ap.add_argument("--recheck", default=None, metavar="FILE", help="evaluate the amended criterion on a committed counts "
                    "file; no network, nothing is written")
    ap.add_argument("--probe-references", type=int, default=0, metavar="N")
    args = ap.parse_args(argv)
    if args.recheck:
        saved = json.loads(Path(args.recheck).read_text(encoding="utf-8"))
        saved["gate_amended"] = evaluate_gate_v2(saved)
        print(render(saved))
        return 0 if saved["gate_amended"]["go"] else 3
    eu = eu or EUtils(args.api_key)
    result = collect(eu, args.since, args.until)
    if args.probe_references:
        result["references_probe"] = probe_references(result["sample_pmids"][:args.probe_references], ref_opener)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(render(result))
    print(f"\nwritten: {out}")
    return 0 if result["gate_amended"]["go"] else 3


if __name__ == "__main__":
    sys.exit(main())
