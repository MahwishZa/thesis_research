#!/usr/bin/env python3
"""Stage 07 - claim classification (multi-label).

Two inspectable stages, never collapsed into one opaque score:

  Stage 1  keyword / rule matching  -> claim_class, confidence, method, matched_terms
  Stage 2  similarity to class prototypes -> claim_class, similarity_score, method

Classification is NOT a deletion filter. It drives retrieval ranking,
down-weighting and manual review. The currency gate down-weights rather than
deletes precisely because tagging precision is unverified, so a mis-tagged
passage must remain retrievable.

Also emits the stratified sample for the 300-passage human validation: a simple
random sample would under-represent small claim classes, which is exactly where
tagging is weakest.

Input : data/chunks/chunks.jsonl   Output: in-place claim_classes + reports
"""
from __future__ import annotations
import argparse, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DATA, BASE, load_config, get_logger, read_jsonl,
                     write_jsonl, write_report, fold)

SRC = DATA / "chunks" / "chunks.jsonl"


def load_classes(cfg: dict) -> list[dict]:
    out = []
    for gid, g in cfg["groups"].items():
        for c in g["classes"]:
            out.append({"id": c["id"], "group": gid, "name": c["name"],
                        "keywords": c.get("keywords") or [],
                        "definition": c.get("definition", "")})
    return out


def keyword_match(text: str, classes: list[dict]) -> list[dict]:
    """Stage 1. Falls back to the class name's own tokens where keywords are
    unpopulated, and reports low confidence so the gap is visible, not hidden."""
    f = fold(text)
    hits = []
    for c in classes:
        terms = c["keywords"] or [t for t in c["name"].split("-") if len(t) > 3]
        matched = [t for t in terms if re.search(r"(?<!\w)" + re.escape(fold(t)) + r"(?!\w)", f)]
        if matched:
            hits.append({"claim_class": c["id"], "confidence": round(
                            min(1.0, len(matched) / max(1, len(terms))) * (1.0 if c["keywords"] else 0.4), 3),
                         "method": "keyword" if c["keywords"] else "keyword-from-classname",
                         "matched_terms": matched})
    return hits


def stratified_sample(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over (source, claim class) cells so small classes appear."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for c in chunks:
        key = (c.get("source_tier", "") or "unknown",
               (c.get("claim_classes") or ["<untagged>"])[0])
        cells[key].append(c)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells)
    out = []
    while len(out) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(SRC))
    ap.add_argument("--sample", type=int, default=300, help="validation sample size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    log = get_logger("07_claim_classification", "quality_control.log")

    cfg = load_config("claim_taxonomy.yaml")
    classes = load_classes(cfg)
    unpopulated = sum(1 for c in classes if not c["keywords"])
    log.info("taxonomy | %d classes in %d groups | %d have no keywords yet",
             len(classes), len(cfg["groups"]), unpopulated)
    if unpopulated:
        log.warning("%d/%d classes have empty keyword lists - stage 1 falls back to "
                    "class-name tokens at reduced confidence. Populate keywords before "
                    "treating these labels as reliable.", unpopulated, len(classes))

    src = Path(args.input)
    if not src.exists():
        log.error("no input at %s - run stage 06 first", src)
        return 2

    chunks = list(read_jsonl(src))
    dist = Counter()
    for ch in chunks:
        hits = keyword_match(ch.get("text", ""), classes)
        ch["claim_classes"] = [h["claim_class"] for h in hits]
        ch["claim_confidence"] = max([h["confidence"] for h in hits], default=0.0)
        ch["claim_method"] = hits[0]["method"] if hits else "none"
        for h in hits:
            dist[h["claim_class"]] += 1
        if not hits:
            dist["<untagged>"] += 1
    write_jsonl(src, chunks)

    write_report("claim_class_distribution.csv",
                 [{"claim_class": k, "chunks": v} for k, v in dist.most_common()],
                 ["claim_class", "chunks"])
    sample = stratified_sample(chunks, args.sample, args.seed)
    ann = BASE / "reports" / "annotation_report.csv"
    write_report("annotation_report.csv",
                 [{"chunk_id": c["chunk_id"], "document_id": c["document_id"],
                   "source_tier": c.get("source_tier", ""),
                   "automatic_labels": ";".join(c.get("claim_classes") or []),
                   "automatic_confidence": c.get("claim_confidence", 0.0),
                   "human_labels": "", "annotator_notes": ""} for c in sample],
                 ["chunk_id", "document_id", "source_tier", "automatic_labels",
                  "automatic_confidence", "human_labels", "annotator_notes"])
    log.info("stage 07 | chunks=%d | tagged classes=%d | validation sample=%d (stratified, seed=%d)",
             len(chunks), len([k for k in dist if k != "<untagged>"]), len(sample), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
