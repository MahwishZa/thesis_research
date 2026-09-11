#!/usr/bin/env python3
"""M3 - MedCPT token-length / truncation audit.

Proposal 5.1 specifies 256 TOKENS + 32-token overlap, sized against the article
encoder's 512-token limit. pmc/build_chunks.py implements 256 WHITESPACE WORDS.
This measures the real distribution with the real tokenizer. It changes nothing.

Requires: pmc/chunks/chunks.jsonl, transformers, and ncbi/MedCPT-Article-Encoder
(tokenizer only - the model weights are NOT needed).
"""
import json, sys, datetime, subprocess, statistics, collections, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "pmc" / "chunks" / "chunks.jsonl"
MODEL = "ncbi/MedCPT-Article-Encoder"
LIMIT = 512
PCTL = (50, 75, 90, 95, 99)

def blocked(reason, remedy):
    res = {"measurement":"M3_token_truncation","status":"NOT_RUN",
           "reason":reason,"remedy":remedy,
           "executed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                                   capture_output=True,text=True).stdout.strip()}
    out = ROOT/"docs/stage1_measurements/m3_token_truncation.json"
    out.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n",newline="\n")
    print("NOT RUN:", reason); print("remedy:", remedy)
    return 1

def main():
    if not CHUNKS.exists():
        return blocked("pmc/chunks/chunks.jsonl absent (gitignored derived artifact)",
                       "run pmc/build_chunks.py on the machine holding the corpus")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return blocked("transformers not installed",
                       "pip install transformers, then re-run")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        return blocked(f"cannot load tokenizer {MODEL}: {type(e).__name__}: {e}",
                       "ensure network access to HuggingFace or a local cache of "
                       "the tokenizer (model weights are not required)")

    # compose_embed_text is the single shared rule for what the encoder sees;
    # reuse it rather than re-deriving, so this measures the real input.
    sys.path.insert(0, str(ROOT/"pmc"))
    try:
        from build_chunks import compose_embed_text
    except Exception:
        compose_embed_text = None

    lens, trunc_by_src, trunc_by_doc = [], collections.Counter(), collections.Counter()
    tot_by_src, tot_by_doc = collections.Counter(), collections.Counter()
    over = 0; n = 0; lost = []
    with open(CHUNKS, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line); n += 1
            text = compose_embed_text(c) if compose_embed_text else c.get("text","")
            ids = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
            L = len(ids); lens.append(L)
            src = c.get("source_category",""); dt = c.get("document_type","")
            tot_by_src[src] += 1; tot_by_doc[dt] += 1
            if L > LIMIT:
                over += 1; trunc_by_src[src] += 1; trunc_by_doc[dt] += 1
                lost.append(L - LIMIT)

    lens.sort()
    res = {
        "measurement":"M3_token_truncation","status":"RUN",
        "executed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                                capture_output=True,text=True).stdout.strip(),
        "tokenizer":MODEL,"encoder_limit":LIMIT,
        "chunks_sha256_16":hashlib.sha256(CHUNKS.read_bytes()).hexdigest()[:16],
        "used_compose_embed_text": compose_embed_text is not None,
        "chunks":n,
        "token_length":{"min":lens[0],"max":lens[-1],
                        "mean":round(statistics.fmean(lens),2),
                        "median":statistics.median(lens),
                        **{f"p{p}":lens[min(len(lens)-1,int(len(lens)*p/100))] for p in PCTL}},
        "exceeding_limit":{"count":over,"pct":round(100*over/n,3) if n else 0.0},
        "truncated_tokens":{"total":sum(lost),
                            "mean_per_truncated":round(statistics.fmean(lost),2) if lost else 0,
                            "max":max(lost) if lost else 0},
        "truncation_by_source_category":{k:{"truncated":trunc_by_src[k],"total":v,
                                            "pct":round(100*trunc_by_src[k]/v,3)}
                                         for k,v in tot_by_src.items()},
        "truncation_by_document_type":{k:{"truncated":trunc_by_doc[k],"total":v,
                                          "pct":round(100*trunc_by_doc[k]/v,3)}
                                       for k,v in tot_by_doc.items()},
    }
    out = ROOT/"docs/stage1_measurements/m3_token_truncation.json"
    out.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n",newline="\n")
    print(json.dumps({k:res[k] for k in ("chunks","token_length","exceeding_limit")},indent=2))
    print("written:", out.relative_to(ROOT))
    return 0

if __name__ == "__main__":
    sys.exit(main())
