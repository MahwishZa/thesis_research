#!/usr/bin/env python3
"""M4 - Contested-pair existence.

Proposal 4.4: a claim class is contested when two passages of that class carry
OPPOSING conclusions, both dated within a contest window [OPEN], and both above a
minimum source tier [UNDEFINED].

This script tests only the STRUCTURAL preconditions that are objectively
checkable from frozen metadata. It does NOT judge whether two documents oppose
each other - that is a manual research judgement - and it does NOT manufacture a
pair or ingest rebuttals.
"""
import csv, json, datetime, subprocess, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "pmc" / "metadata"
INDEX = ROOT / "pmc" / "index"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
def classes(v):
    return {p.strip() for p in (v or "").replace("|",";").replace("/",";").split(";") if p.strip()}
def months(a, b):
    """Whole months between two YYYY[-MM[-DD]] strings, coarsest common precision."""
    def ym(s):
        s = (s or "").strip()
        if len(s) >= 7: return int(s[:4]), int(s[5:7])
        if len(s) >= 4: return int(s[:4]), None
        return None, None
    ya, ma = ym(a); yb, mb = ym(b)
    if ya is None or yb is None: return None, "missing date"
    if ma is None or mb is None:
        return abs(yb - ya) * 12, "year-precision only; +/-12 months uncertainty"
    return abs((yb - ya) * 12 + (mb - ma)), "exact"

def main():
    res = {
        "measurement": "M4_contested_pair_existence",
        "executed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                                 capture_output=True,text=True).stdout.strip(),
        "inputs": {n: {"sha256_16": sha(META/n)} for n in
                   ("currency_pack.csv","cpg_registry.csv")},
        "anchor": None, "candidate_counterparties": [],
        "preconditions": {}, "verdict": None,
    }

    cur = list(csv.DictReader(open(META/"currency_pack.csv", encoding="utf-8")))
    anchor = next((r for r in cur if "contested" in classes(r.get("claim_class"))), None)
    if anchor is None:
        res["verdict"] = "NO_ANCHOR"
        (ROOT/"docs/stage1_measurements/m4_contested_pair.json").write_text(
            json.dumps(res, indent=2, sort_keys=True)+"\n", newline="\n")
        print("no contested anchor found"); return

    a_cls = classes(anchor.get("claim_class"))
    res["anchor"] = {"pmcid": anchor["pmcid"], "title": anchor["title"][:90],
                     "date": anchor["canonical_date"],
                     "precision": anchor["date_precision"],
                     "claim_classes": sorted(a_cls)}

    # counterparties: any curated doc sharing a claim class, excluding the anchor
    for r in cur:
        if r["pmcid"] == anchor["pmcid"]: continue
        shared = a_cls & classes(r.get("claim_class"))
        if not shared: continue
        gap, note = months(anchor["canonical_date"], r["canonical_date"])
        res["candidate_counterparties"].append({
            "pmcid": r["pmcid"], "title": r["title"][:90],
            "date": r["canonical_date"], "precision": r["date_precision"],
            "shared_claim_classes": sorted(shared),
            "months_from_anchor": gap, "gap_note": note,
        })
    res["candidate_counterparties"].sort(key=lambda d: (d["months_from_anchor"] is None,
                                                        d["months_from_anchor"]))

    gaps = [c["months_from_anchor"] for c in res["candidate_counterparties"]
            if c["months_from_anchor"] is not None]
    res["preconditions"] = {
        "same_claim_class_counterparty_exists": bool(res["candidate_counterparties"]),
        "opposing_conclusions_verified": {
            "status": "NOT_MACHINE_CHECKABLE",
            "note": "requires manual research judgement; not asserted here"},
        "within_contest_window": {
            "status": "UNDECIDABLE",
            "reason": "contest window length is [OPEN] in proposal 4.4",
            "closest_counterparty_months": min(gaps) if gaps else None,
            "furthest_counterparty_months": max(gaps) if gaps else None},
        "above_minimum_source_tier": {
            "status": "UNDECIDABLE",
            "reason": "source-tier values are undefined in the proposal; "
                      "authority ordering is a tested variable (ablation A12)"},
        "retrievable": {
            "status": "NOT_RUN",
            "reason": "no MedCPT index present (pmc/index absent)"}
                       if not INDEX.exists() or not any(INDEX.iterdir()) else
                      {"status": "RUNNABLE"},
    }
    res["verdict"] = "STRUCTURAL_CANDIDATE_EXISTS_DECISION_BLOCKED"
    out = ROOT/"docs/stage1_measurements/m4_contested_pair.json"
    out.write_text(json.dumps(res, indent=2, sort_keys=True)+"\n", newline="\n")
    print(json.dumps({"anchor": res["anchor"],
                      "counterparties": res["candidate_counterparties"],
                      "verdict": res["verdict"]}, indent=2))
    print("written:", out.relative_to(ROOT))

if __name__ == "__main__":
    main()
