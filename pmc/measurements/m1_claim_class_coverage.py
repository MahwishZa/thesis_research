#!/usr/bin/env python3
"""M1 - Claim-class coverage.

Architecture requirement R-L: gamma and the contested test operate per claim class.
An unpopulated class is a dead branch. This measures what coverage actually exists.

Runs at two levels:
  metadata-level  - always available (curated layers + date distribution)
  passage-level   - requires pmc/chunks/chunks.jsonl; skipped with an explicit
                    status when absent. NEVER silently estimated.
"""
import csv, json, sys, collections, hashlib, subprocess, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "pmc" / "metadata"
CHUNKS = ROOT / "pmc" / "chunks" / "chunks.jsonl"
SPLIT_ANCHOR = "2024-06"          # proposal 6.4: June 2024 criteria revision
TAXONOMY_MIN, TAXONOMY_MAX = 40, 80   # proposal 5.1: 40-80 classes

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None

def split_classes(v):
    if not v: return []
    out = []
    for part in v.replace("|", ";").replace("/", ";").split(";"):
        part = part.strip()
        if part: out.append(part)
    return out

def main():
    res = {
        "measurement": "M1_claim_class_coverage",
        "executed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                                 capture_output=True,text=True).stdout.strip(),
        "split_anchor": SPLIT_ANCHOR,
        "inputs": {},
        "taxonomy": {},
        "curated_layers": {},
        "corpus_dates": {},
        "passage_level": {},
    }

    # ---- inputs + hashes (reproducibility) -------------------------------
    for name in ("cpg_registry.csv","currency_pack.csv","canonical_dates.csv",
                 "corpus_policy.csv","cpg_external_targets.csv"):
        p = META / name
        res["inputs"][name] = {"present": p.exists(), "sha256_16": sha(p),
                               "rows": sum(1 for _ in open(p))-1 if p.exists() else None}

    # ---- taxonomy status --------------------------------------------------
    classes = collections.Counter()
    doc_classes = {}
    for fname, col in (("cpg_registry.csv","claim_classes"),
                       ("currency_pack.csv","claim_class")):
        p = META / fname
        if not p.exists(): continue
        layer = collections.Counter(); n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n += 1
            ks = split_classes(r.get(col))
            doc_classes[r.get("pmcid") or r.get("cpg_id")] = ks
            for k in ks:
                classes[k] += 1; layer[k] += 1
        res["curated_layers"][fname] = {"documents": n, "by_class": dict(layer.most_common())}

    # bulk corpus: does ANY claim_class column exist outside the curated layers?
    pol = META / "corpus_policy.csv"
    bulk_tagged = False
    if pol.exists():
        hdr = next(csv.reader(open(pol, encoding="utf-8")))
        bulk_tagged = "claim_class" in hdr or "claim_classes" in hdr
    res["taxonomy"] = {
        "distinct_classes_observed": len(classes),
        "required_min": TAXONOMY_MIN, "required_max": TAXONOMY_MAX,
        "meets_required_size": TAXONOMY_MIN <= len(classes) <= TAXONOMY_MAX,
        "documents_tagged": len(doc_classes),
        "bulk_corpus_has_claim_class_column": bulk_tagged,
        "by_class": dict(classes.most_common()),
        "status": ("BULK_TAGGING_ABSENT" if not bulk_tagged else "PRESENT"),
    }

    # ---- corpus date distribution (real, full corpus) ---------------------
    cd = META / "canonical_dates.csv"
    if cd.exists():
        years = collections.Counter(); splits = collections.Counter()
        prec = collections.Counter(); src = collections.Counter()
        total = 0
        for r in csv.DictReader(open(cd, encoding="utf-8")):
            total += 1
            d = (r.get("canonical_date") or "")[:4]
            if d.isdigit(): years[d] += 1
            splits[r.get("split_june_2024") or "missing"] += 1
            prec[r.get("canonical_date_precision") or "missing"] += 1
            src[r.get("date_source") or "missing"] += 1
        res["corpus_dates"] = {
            "records": total,
            "by_year": dict(sorted(years.items())),
            "split_june_2024": dict(splits),
            "precision": dict(prec),
            "date_source": dict(src),
        }

    # ---- passage level (requires chunks) ----------------------------------
    if not CHUNKS.exists():
        res["passage_level"] = {
            "status": "NOT_RUN",
            "reason": "pmc/chunks/chunks.jsonl absent (gitignored derived artifact)",
            "remedy": "run on the machine holding the corpus, after pmc/build_chunks.py",
        }
    else:
        per = collections.defaultdict(lambda: {"passages":0,"documents":set(),
                                               "pre":0,"post":0,"unknown":0,
                                               "source_category":collections.Counter()})
        n = 0
        with open(CHUNKS, encoding="utf-8") as fh:
            for line in fh:
                n += 1
                c = json.loads(line)
                date = (c.get("canonical_date") or "")
                side = ("unknown" if len(date) < 7 else
                        ("pre" if date[:7] < SPLIT_ANCHOR else "post"))
                for k in split_classes(c.get("claim_class")) or ["<untagged>"]:
                    e = per[k]
                    e["passages"] += 1
                    e["documents"].add(c.get("document_id"))
                    e[side] += 1
                    e["source_category"][c.get("source_category","")] += 1
        res["passage_level"] = {
            "status": "RUN", "total_passages": n,
            "by_class": {k: {"passages":v["passages"],"documents":len(v["documents"]),
                             "pre_june_2024":v["pre"],"post_june_2024":v["post"],
                             "unknown_date":v["unknown"],
                             "source_category":dict(v["source_category"])}
                         for k,v in sorted(per.items(),
                                           key=lambda kv:-kv[1]["passages"])},
        }

    out = ROOT / "docs" / "stage1_measurements" / "m1_claim_class_coverage.json"
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", newline="\n")
    print(json.dumps({"taxonomy": res["taxonomy"],
                      "passage_level_status": res["passage_level"].get("status")},
                     indent=2))
    print("written:", out.relative_to(ROOT))

if __name__ == "__main__":
    sys.exit(main())
