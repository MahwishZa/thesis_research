"""Freeze one candidate pool per benchmark item (needs network + MedCPT).

For each item: take the as-of PubMed hits cached by ``pubmed_asof`` (every record
proven to precede the newest review's date, Cochrane records excluded), fetch
their abstracts, rank by MedCPT dense similarity to the QUESTION, rerank the top
``--dense-k`` with the MedCPT cross-encoder, and freeze the top ``--pool-size``
with their dates. Every arm later sees exactly this pool; arms differ only in
admission. No label, verdict or generated text is read here, so building pools for
the confirmatory split does not touch any outcome.

    python -m experiments.medchange.freeze_candidates --split dev --device cpu

Resumable (one JSON line per finished item; reruns skip them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .pubmed_asof import EUtils, EUTILS

HERE = Path(__file__).resolve().parent
#: Record types that are not primary evidence and are dropped before ranking.
EXCLUDED_TYPES = ("Retraction of Publication", "Retracted Publication", "Published Erratum",
                  "Comment", "Editorial", "Letter", "News", "Patient Education Handout")
MIN_ABSTRACT_CHARS = 200


def parse_efetch_xml(xml_text: str) -> dict[str, dict]:
    """PubmedArticleSet XML -> {pmid: {"title", "abstract"}}; records without an
    abstract are returned with an empty abstract (and dropped later)."""
    out: dict[str, dict] = {}
    root = ET.fromstring(xml_text)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext("./MedlineCitation/PMID")
        if not pmid:
            continue
        title = "".join(art.find("./MedlineCitation/Article/ArticleTitle").itertext()) \
            if art.find("./MedlineCitation/Article/ArticleTitle") is not None else ""
        parts = []
        for node in art.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            label = node.get("Label")
            if text:
                parts.append(f"{label}: {text}" if label else text)
        out[pmid] = {"title": title.strip(), "abstract": " ".join(parts)}
    return out


def fetch_abstracts(eu: EUtils, pmids: Sequence[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(pmids), 100):
        batch = list(pmids[i:i + 100])
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract",
                  "tool": "thesis_research"}
        if eu.api_key:
            params["api_key"] = eu.api_key
        url = EUTILS + "efetch.fcgi?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            eu._sleep(eu.delay)
            try:
                out.update(parse_efetch_xml(eu._open(url).decode("utf-8")))
                break
            except (ET.ParseError, TimeoutError, OSError) as exc:
                if attempt == 4:
                    raise RuntimeError(f"efetch failed after 5 attempts: {exc}") from exc
                eu._sleep(2 ** attempt)
    return out


def pool_hash(candidates: Sequence[dict]) -> str:
    """Order-sensitive hash of the frozen pool (ids, order, dates)."""
    payload = [[c["pmid"], c["rank"], c["lower"], c["upper"]] for c in candidates]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def eligible(record: dict, abstracts: dict[str, dict]) -> bool:
    a = abstracts.get(record["pmid"])
    if not a or len(a["abstract"]) < MIN_ABSTRACT_CHARS:
        return False
    return not any(t in record.get("pubtypes", []) for t in EXCLUDED_TYPES)


def freeze_item(item: dict, probe: dict, abstracts: dict[str, dict], *, query_encoder,
                article_encoder, reranker, dense_k: int, pool_size: int) -> dict:
    cutoff = item["newest"]["date"]
    recs = [r for r in probe["records"]
            if r["upper"] and r["upper"] <= cutoff and eligible(r, abstracts)]
    if not recs:
        return {"item_id": item["item_id"], "cutoff": cutoff, "n_eligible": 0,
                "candidates": [], "pool_hash": pool_hash([])}
    texts = [f'{abstracts[r["pmid"]]["title"]}. {abstracts[r["pmid"]]["abstract"]}' for r in recs]
    q = np.asarray(query_encoder.encode([item["question"]]))[0]
    d = np.asarray(article_encoder.encode(texts)) @ q
    order = np.argsort(-d, kind="stable")[:dense_k]
    ce = np.asarray(reranker.score(item["question"], [texts[i] for i in order]))
    reranked = sorted(zip(order, ce), key=lambda t: (-float(t[1]), recs[t[0]]["pmid"]))[:pool_size]
    cands = []
    for rank, (i, s) in enumerate(reranked, 1):
        r = recs[i]
        cands.append({"pmid": r["pmid"], "rank": rank, "rerank_score": float(s),
                      "dense_score": float(d[i]), "lower": r["lower"], "upper": r["upper"],
                      "pubtypes": r["pubtypes"], "journal": r["journal"],
                      "is_review": any(t in r["pubtypes"] for t in ("Systematic Review", "Meta-Analysis")),
                      "title": abstracts[r["pmid"]]["title"], "abstract": abstracts[r["pmid"]]["abstract"]})
    return {"item_id": item["item_id"], "cutoff": cutoff, "n_eligible": len(recs),
            "candidates": cands, "pool_hash": pool_hash(cands)}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:      # truncated last line from a kill
                    break
                raise
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--benchmark", default=str(HERE / "data" / "benchmark.jsonl"))
    ap.add_argument("--probe-dir", default=str(HERE / "data" / "pubmed_g0"))
    ap.add_argument("--abstract-cache", default=str(HERE / "data" / "abstracts.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm", "all"))
    ap.add_argument("--dense-k", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    out = Path(args.out or HERE / "data" / f"frozen_{args.split}.jsonl")
    items = [json.loads(l) for l in open(args.benchmark, encoding="utf-8")]
    items = [i for i in items if not i["likely_label_noise"]
             and (args.split == "all" or i["split"] == args.split)]
    if args.limit:
        items = items[:args.limit]
    probes = {}
    for it in items:
        f = Path(args.probe_dir) / f"{it['item_id']}.json"
        if not f.exists():
            print(f"missing PubMed probe for {it['item_id']}: run pubmed_asof "
                  f"--split {args.split} first", file=sys.stderr)
            return 2
        probes[it["item_id"]] = json.loads(f.read_text(encoding="utf-8"))
    done = {r["item_id"] for r in _load_jsonl(out)}
    todo = [i for i in items if i["item_id"] not in done]
    print(f"{len(items)} items, {len(done)} already frozen, {len(todo)} to do")
    if not todo:
        return 0

    cache_path = Path(args.abstract_cache)
    abstracts = {r["pmid"]: r for r in _load_jsonl(cache_path)}
    eu = EUtils(args.api_key)
    need = sorted({r["pmid"] for it in todo for r in probes[it["item_id"]]["records"]}
                  - set(abstracts))
    if need:
        print(f"fetching {len(need)} abstracts ...", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8", newline="\n") as h:
            for i in range(0, len(need), 500):
                got = fetch_abstracts(eu, need[i:i + 500])
                for pmid in need[i:i + 500]:
                    rec = {"pmid": pmid, **got.get(pmid, {"title": "", "abstract": ""})}
                    abstracts[pmid] = rec
                    h.write(json.dumps(rec) + "\n")
                h.flush()
                print(f"  abstracts {min(i + 500, len(need))}/{len(need)}", flush=True)

    from experiments.shared.retrieval.encoders import (
        MedCPTReranker, medcpt_article_encoder, medcpt_query_encoder)
    qe = medcpt_query_encoder(device=args.device)
    ae = medcpt_article_encoder(device=args.device)
    rr = MedCPTReranker(device=args.device)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, it in enumerate(todo, 1):
            rec = freeze_item(it, probes[it["item_id"]], abstracts, query_encoder=qe,
                              article_encoder=ae, reranker=rr, dense_k=args.dense_k,
                              pool_size=args.pool_size)
            rec["encoders"] = {"query": qe.name, "article": ae.name, "reranker": rr.name}
            h.write(json.dumps(rec) + "\n")
            h.flush()
            if k % 5 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} frozen ({time.time() - t0:.0f}s)", flush=True)
    sizes = [len(r["candidates"]) for r in _load_jsonl(out)]
    print(f"frozen pools: {len(sizes)}; empty: {sum(s == 0 for s in sizes)}; "
          f"median size {sorted(sizes)[len(sizes) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
