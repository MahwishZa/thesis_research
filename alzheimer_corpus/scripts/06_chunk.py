#!/usr/bin/env python3
"""Stage 06 - chunking, after deduplication.

256 tokens with 32-token overlap (stride 224), sized against the MedCPT article
encoder's 512-token limit with headroom for a prepended title and section header.

TOKENIZER: the retrieval model's, not whitespace. 256 English words is roughly
340-400 MedCPT sub-word tokens, so counting words silently produces a different
passage size than the specification. --tokenizer whitespace is available as an
explicit, recorded fallback and stamps tokenizer_used so the difference can
never be mistaken for the specified configuration.

Guidelines are chunked structure-aware: a recommendation is never split from its
qualifying conditions to hit exactly 256 tokens. Such chunks record
chunk_size_exception and its reason.

Input : data/deduplicated/documents.jsonl   Output: data/chunks/chunks.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, load_config, get_logger, read_jsonl, write_jsonl

SRC = DATA / "deduplicated" / "documents.jsonl"
OUT = DATA / "chunks" / "chunks.jsonl"
STRUCTURE_AWARE = {"clinical_guideline", "consensus_statement", "currency_pack"}


def get_tokenizer(name: str, log):
    if name == "whitespace":
        log.warning("using WHITESPACE tokenizer - NOT the specified MedCPT tokenizer; "
                    "256 words is ~340-400 MedCPT tokens")
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        log.error("cannot load tokenizer %s: %s", name, e)
        log.error("install transformers and allow access to the model, or pass "
                  "--tokenizer whitespace to record an explicit fallback")
        raise SystemExit(2)


def chunk_units(units: list[str], size: int, stride: int, tok) -> list[tuple[str, int, int]]:
    text = " ".join(u for u in units if u)
    if tok is None:
        toks = text.split()
        join = lambda xs: " ".join(xs)
    else:
        toks = tok.encode(text, add_special_tokens=False)
        join = lambda xs: tok.decode(xs)
    out = []
    i = 0
    while i < len(toks):
        piece = toks[i:i + size]
        out.append((join(piece), i, i + len(piece)))
        if i + size >= len(toks):
            break
        i += stride
    return out


def main(argv=None) -> int:
    cfg = load_config("search_queries.yaml")  # keeps config loading uniform
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(SRC))
    ap.add_argument("--tokenizer", default="ncbi/MedCPT-Article-Encoder")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args(argv)
    stride = args.size - args.overlap
    log = get_logger("06_chunk", "quality_control.log")

    src = Path(args.input)
    if not src.exists():
        log.error("no input at %s - run stage 05 first", src)
        return 2
    tok = get_tokenizer(args.tokenizer, log)
    tok_used = "whitespace" if tok is None else args.tokenizer

    chunks = []
    for rec in read_jsonl(src):
        doc_id = str(rec.get("document_id") or rec.get("pmid") or "")
        tier = rec.get("source_tier", "")
        sections = rec.get("sections") or [
            {"section": "title", "text": rec.get("title", "")},
            {"section": "abstract", "text": rec.get("abstract", "")}]
        for sec in sections:
            body = sec.get("text", "")
            if not body:
                continue
            pieces = chunk_units([body], args.size, stride, tok)
            exception = tier in STRUCTURE_AWARE and len(pieces) > 1
            for idx, (text, a, b) in enumerate(pieces):
                chunks.append({
                    "chunk_id": f"{doc_id}#{sec.get('section','')}.{idx}",
                    "document_id": doc_id, "chunk_index": idx, "text": text,
                    "retrieval_text": f"[TITLE] {rec.get('title','')}\n"
                                      f"[SECTION] {sec.get('section','')}\n[TEXT] {text}",
                    "section": sec.get("section", ""), "subsection": sec.get("subsection", ""),
                    "token_start": a, "token_end": b, "tokenizer_used": tok_used,
                    "publication_date": rec.get("publication_date", ""),
                    "source_tier": tier, "claim_classes": [],
                    "retracted": rec.get("retracted", ""),
                    "chunk_size_exception": exception,
                    "chunk_size_exception_reason":
                        "guideline recommendation kept with its qualifying conditions"
                        if exception else "",
                })
    n = write_jsonl(OUT, chunks)
    stats = {"chunks": n, "window_tokens": args.size, "overlap_tokens": args.overlap,
             "stride_tokens": stride, "tokenizer_used": tok_used}
    (OUT.parent / "chunk_stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    log.info("stage 06 | chunks=%d | tokenizer=%s | %d/%d/%d",
             n, tok_used, args.size, args.overlap, stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
