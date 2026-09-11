#!/usr/bin/env python3
"""M2 - Retrieval reachability.

Architecture requirement R-M: SCAF (Algorithm 1) takes the reranked candidate set
S_rr as INPUT. It is a filter over already-retrieved evidence, so it cannot admit
what retrieval never returns. This measures whether decisive evidence is reachable.

TWO prerequisites, and the second is NOT an environment problem:

  1. A built MedCPT index (pmc/index/) and the frozen retrieval configuration.
  2. A DEFINED decisive-evidence set: (query, expected-evidence) pairs.

Prerequisite 2 does not exist in the repository and cannot be invented here -
choosing which evidence is "decisive", and the queries used to reach it, is a
research decision that determines the measured result. This script therefore
refuses to run without an explicit, human-approved probe file rather than
fabricating one.

Probe file format (docs/stage1_measurements/decisive_evidence_probes.json):
  {"probes":[{"probe_id":"...","query":"...","expected":{"document_ids":[...],
              "claim_class":"...","temporal_category":"pre|post",
              "source_type":"..."}}]}
"""
import json, sys, datetime, subprocess, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "pmc" / "index"
PROBES = ROOT / "docs" / "stage1_measurements" / "decisive_evidence_probes.json"
K_VALUES = (5, 10, 20, 50)     # reported; retrieval config itself is frozen

def emit(res):
    out = ROOT / "docs" / "stage1_measurements" / "m2_retrieval_reachability.json"
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", newline="\n")
    print(json.dumps({k: res[k] for k in ("status", "blockers") if k in res}, indent=2))
    print("written:", out.relative_to(ROOT))

def main():
    res = {"measurement": "M2_retrieval_reachability",
           "executed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "commit": subprocess.run(["git","rev-parse","HEAD"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip(),
           "k_values": list(K_VALUES), "blockers": []}

    index_ready = INDEX.exists() and any(INDEX.iterdir())
    if not index_ready:
        res["blockers"].append({
            "blocker": "NO_INDEX", "kind": "ENVIRONMENT",
            "detail": "pmc/index/ absent or empty (gitignored, machine-specific)",
            "remedy": "python3 pmc/embed_chunks.py --device cuda && python3 pmc/verify_index.py"})
    if not PROBES.exists():
        res["blockers"].append({
            "blocker": "NO_DECISIVE_EVIDENCE_SET", "kind": "REQUIRES_HUMAN_DECISION",
            "detail": "no (query, expected-evidence) probe set exists. Which evidence "
                      "counts as decisive, and the queries used to reach it, determine "
                      "the measured reachability - this is a research decision, not an "
                      "engineering default.",
            "remedy": "student/supervisor define and approve "
                      + str(PROBES.relative_to(ROOT))})

    if res["blockers"]:
        res["status"] = "NOT_RUN"
        emit(res); return 1

    probes = json.loads(PROBES.read_text())["probes"]
    sys.path.insert(0, str(ROOT / "pmc"))
    import retrieve as R   # frozen retrieval configuration; not reconfigured here

    rows, ranks = [], []
    fail = {"claim_class": collections.Counter(), "source_type": collections.Counter(),
            "temporal_category": collections.Counter()}
    maxk = max(K_VALUES)
    for p in probes:
        cands = R.search(p["query"], top_k=maxk)
        want_docs = set(p["expected"].get("document_ids", []))
        doc_rank = passage_rank = None
        for i, c in enumerate(cands, 1):
            if passage_rank is None and c.get("chunk_id") in set(p["expected"].get("chunk_ids", [])):
                passage_rank = i
            if doc_rank is None and c.get("document_id") in want_docs:
                doc_rank = i
        rows.append({"probe_id": p["probe_id"], "document_rank": doc_rank,
                     "passage_rank": passage_rank,
                     "hit_at": {f"@{k}": bool(doc_rank and doc_rank <= k) for k in K_VALUES}})
        if doc_rank is None:
            e = p["expected"]
            fail["claim_class"][e.get("claim_class","")] += 1
            fail["source_type"][e.get("source_type","")] += 1
            fail["temporal_category"][e.get("temporal_category","")] += 1
        else:
            ranks.append(doc_rank)

    n = len(rows)
    res.update({
        "status": "RUN", "probes": n,
        "reachability_at_k": {f"@{k}": round(sum(1 for r in rows if r["hit_at"][f"@{k}"])/n, 4)
                              for k in K_VALUES},
        "zero_retrieval_rate": round(sum(1 for r in rows if r["document_rank"] is None)/n, 4),
        "rank_distribution": {"ranks_when_found": sorted(ranks),
                              "median": (sorted(ranks)[len(ranks)//2] if ranks else None)},
        "failures_by_claim_class": dict(fail["claim_class"]),
        "failures_by_source_type": dict(fail["source_type"]),
        "failures_by_temporal_category": dict(fail["temporal_category"]),
        "per_probe": rows})
    emit(res); return 0

if __name__ == "__main__":
    sys.exit(main())
