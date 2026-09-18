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

PERFORMANCE. The original implementation called the tokenizer's encode()
once per (document, section) and decode() once per output window - on the
real corpus (114k+ documents, ~7 sections each) that is roughly 800k encode
calls and 1-2M decode calls, each paying Python-level call overhead
individually rather than through Hugging Face's batched, parallelised path.
That is the entire explanation for a multi-hour run: it is real work done
inefficiently, not a hang. This version batches encode/decode across many
documents at once (--batch-size, default 200) and streams output as each
batch completes rather than holding the whole corpus in memory. The
tokenizer boundaries it computes - window size, overlap, stride, and every
(start, end) token position - are identical to the original; batching
changes how many Python calls compute them, not what they compute.

Input : data/deduplicated/documents.jsonl   Output: data/chunks/chunks.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, get_logger, read_jsonl, write_jsonl

SRC = DATA / "deduplicated" / "documents.jsonl"
OUT = DATA / "chunks" / "chunks.jsonl"
PROGRESS_FILE = DATA / "chunks" / ".chunk_progress.json"
STRUCTURE_AWARE = {"clinical_guideline", "consensus_statement", "currency_pack"}

#: How many documents' sections are batched into one encode()/decode() call.
#: Bounds peak memory to O(batch) chunks rather than O(whole corpus); large
#: enough that batching overhead is negligible next to per-call overhead,
#: small enough that one very large document cannot make a batch pathological
#: on its own (each document's sections are still processed section-by-section
#: within a batch, so one huge document only inflates its own share of it).
DEFAULT_BATCH_SIZE = 200

#: How often (in documents) a progress line is logged. Frequent enough to be
#: useful on a multi-hour run, infrequent enough not to flood the terminal.
PROGRESS_EVERY = 2000


def get_tokenizer(name: str, log):
    if name == "whitespace":
        log.warning("using WHITESPACE tokenizer - NOT the specified MedCPT tokenizer; "
                    "256 words is ~340-400 MedCPT tokens")
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        log.error("cannot load tokenizer %s: %s", name, e)
        log.error("install transformers and allow access to the model, or pass "
                  "--tokenizer whitespace to record an explicit fallback")
        raise SystemExit(2)
    # is_fast is the single most decisive diagnostic for "why is this slow":
    # the fast (Rust) tokenizer is what makes batching pay off. Logged once,
    # not per call, so the answer is on record without adding overhead.
    log.info("tokenizer %s loaded | fast=%s | class=%s",
             name, getattr(tok, "is_fast", "unknown"), type(tok).__name__)
    return tok


def window_boundaries(n_tokens: int, size: int, stride: int) -> list[tuple[int, int]]:
    """(start, end) token index pairs for one section's sliding windows.

    Pure index arithmetic, no tokenizer involved - this is the exact
    boundary logic the original chunk_units() computed inline, pulled out
    so it can be tested and reasoned about without a tokenizer at all, and
    so batching the actual encode/decode calls around it cannot silently
    change where a window starts or ends.

    The last window is always included even if shorter than ``size`` (a
    document that isn't an exact multiple of stride tokens long must not
    lose its tail), and no window is ever skipped or duplicated.
    """
    if n_tokens <= 0:
        return []
    out: list[tuple[int, int]] = []
    i = 0
    while True:
        end = min(i + size, n_tokens)
        out.append((i, end))
        if i + size >= n_tokens:
            break
        i += stride
    return out


def encode_batch(texts: list[str], tok) -> list[list]:
    """Tokenize many texts in one call.

    For a real Hugging Face tokenizer this is what actually engages fast
    (Rust-backed) batched, multi-threaded tokenization instead of paying
    Python call overhead once per text; ``padding=False`` is explicit
    because padding would insert pad-token ids into input_ids and corrupt
    the true per-text lengths window_boundaries depends on. For the
    whitespace stand-in (tok is None) each text is still split
    independently - whitespace splitting has no batched form to exploit and
    was never the bottleneck.
    """
    if tok is None:
        return [t.split() for t in texts]
    if not texts:
        return []
    encoded = tok(texts, add_special_tokens=False, padding=False, truncation=False)
    return encoded["input_ids"]


def decode_batch(windows: list[list], tok) -> list[str]:
    """Decode many token-id windows in one call - batch_decode(), not one
    decode() per window. Same reasoning as encode_batch()."""
    if tok is None:
        return [" ".join(w) for w in windows]
    if not windows:
        return []
    return tok.batch_decode(windows, skip_special_tokens=False)


def iter_document_sections(rec: dict):
    """Yield (section_dict, body_text) for one document's sections, applying
    the same title/abstract fallback the original implementation used when
    a document carries no explicit ``sections`` list."""
    sections = rec.get("sections") or [
        {"section": "title", "text": rec.get("title", "")},
        {"section": "abstract", "text": rec.get("abstract", "")},
    ]
    for sec in sections:
        body = sec.get("text", "")
        if body:
            yield sec, body


def chunk_batch(records: list[dict], size: int, stride: int, tok,
                tok_used: str) -> list[dict]:
    """Chunk a batch of documents with two tokenizer calls total (one encode,
    one decode) regardless of how many documents or sections are in it.

    Returns fully-formed chunk records in the same shape and field order of
    intent as the original per-section implementation - only the number of
    tokenizer calls differs, not the chunking result.
    """
    # Flatten every (document, section) pair in the batch into one list of
    # texts, remembering which document/section each belongs to.
    flat_texts: list[str] = []
    flat_meta: list[tuple[int, dict, dict]] = []  # (doc_index, rec, sec)
    for doc_index, rec in enumerate(records):
        for sec, body in iter_document_sections(rec):
            flat_texts.append(body)
            flat_meta.append((doc_index, rec, sec))

    if not flat_texts:
        return []

    token_lists = encode_batch(flat_texts, tok)

    # Now flatten every window across every section into one list, so
    # decoding is a single batched call too. section_window_count is carried
    # alongside each window rather than recomputed later - a per-window scan
    # over window_meta would make reassembly quadratic in windows-per-batch.
    flat_windows: list = []
    window_meta: list[tuple[int, dict, dict, int, int, int, int]] = []
    for (doc_index, rec, sec), toks in zip(flat_meta, token_lists):
        boundaries = window_boundaries(len(toks), size, stride)
        section_window_count = len(boundaries)
        for idx, (start, end) in enumerate(boundaries):
            flat_windows.append(toks[start:end])
            window_meta.append(
                (doc_index, rec, sec, idx, start, end, section_window_count))

    decoded = decode_batch(flat_windows, tok)

    chunks: list[dict] = []
    for (doc_index, rec, sec, idx, start, end, section_window_count), text \
            in zip(window_meta, decoded):
        doc_id = str(rec.get("document_id") or rec.get("pmid") or "")
        tier = rec.get("source_tier", "")
        exception = tier in STRUCTURE_AWARE and section_window_count > 1
        chunks.append({
            "chunk_id": f"{doc_id}#{sec.get('section', '')}.{idx}",
            "document_id": doc_id, "chunk_index": idx, "text": text,
            "retrieval_text": f"[TITLE] {rec.get('title', '')}\n"
                              f"[SECTION] {sec.get('section', '')}\n[TEXT] {text}",
            "section": sec.get("section", ""), "subsection": sec.get("subsection", ""),
            "token_start": start, "token_end": end, "tokenizer_used": tok_used,
            "publication_date": rec.get("publication_date", ""),
            "source_tier": tier, "claim_classes": [],
            "retracted": rec.get("retracted", ""),
            "ad_relevant": rec.get("ad_relevant", ""),
            "ad_relevance_score": rec.get("ad_relevance_score", ""),
            "chunk_size_exception": exception,
            "chunk_size_exception_reason":
                "guideline recommendation kept with its qualifying conditions"
                if exception else "",
        })
    return chunks


def _run_params(args) -> dict:
    """The parameters that must match between an interrupted run and its
    --resume - resuming under a different tokenizer, size or overlap would
    silently produce a chunks.jsonl with inconsistent windowing partway
    through, with nothing in the file itself to reveal where."""
    return {"tokenizer": args.tokenizer, "size": args.size, "overlap": args.overlap}


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {"documents_done": 0, "params": None}
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return {"documents_done": data.get("documents_done", 0),
               "params": data.get("params")}
    except (json.JSONDecodeError, OSError):
        return {"documents_done": 0, "params": None}


def _save_progress(documents_done: int, params: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps({"documents_done": documents_done, "params": params}),
        encoding="utf-8")


def iter_chunks(records_iter, size: int, stride: int, tok, tok_used: str,
                batch_size: int, log, *, start_index: int = 0,
                run_params: dict | None = None):
    """Stream chunks for the whole corpus, batching batch_size documents at
    a time and logging progress roughly every PROGRESS_EVERY documents.

    A generator, not a list: write_jsonl() consumes it one record at a time,
    so peak memory is bounded by one batch's worth of chunks, never the
    whole corpus's.
    """
    start_time = time.monotonic()
    processed = start_index
    total_chunks = 0
    batch: list[dict] = []
    # Batches, not a document-count modulo: a modulo only lands exactly on
    # PROGRESS_EVERY when batch_size divides it, which --batch-size can
    # easily violate.
    batches_between_logs = max(1, PROGRESS_EVERY // max(1, batch_size))
    batches_since_log = 0

    def flush():
        nonlocal total_chunks
        if not batch:
            return 0
        produced = chunk_batch(batch, size, stride, tok, tok_used)
        total_chunks += len(produced)
        return produced

    for rec in records_iter:
        batch.append(rec)
        if len(batch) >= batch_size:
            yield from flush()
            processed += len(batch)
            batch = []
            _save_progress(processed, run_params)
            batches_since_log += 1
            if batches_since_log >= batches_between_logs:
                batches_since_log = 0
                elapsed = time.monotonic() - start_time
                rate = (processed - start_index) / elapsed if elapsed > 0 else 0
                log.info(
                    "stage 06 progress | documents=%d | chunks=%d | "
                    "%.1f docs/sec | elapsed=%.0fs",
                    processed, total_chunks, rate, elapsed,
                )
    if batch:
        yield from flush()
        processed += len(batch)
        _save_progress(processed, run_params)

    elapsed = time.monotonic() - start_time
    log.info("stage 06 | documents=%d | chunks=%d | elapsed=%.0fs",
             processed, total_chunks, elapsed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(SRC))
    ap.add_argument("--tokenizer", default="ncbi/MedCPT-Article-Encoder")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="documents tokenized/decoded per batched call")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: skip documents already "
                         "written (per .chunk_progress.json) and append rather "
                         "than overwrite")
    args = ap.parse_args(argv)
    stride = args.size - args.overlap
    log = get_logger("06_chunk", "quality_control.log")

    src = Path(args.input)
    if not src.exists():
        log.error("no input at %s - run stage 05 first", src)
        return 2
    tok = get_tokenizer(args.tokenizer, log)
    tok_used = "whitespace" if tok is None else args.tokenizer

    params = _run_params(args)
    records = read_jsonl(src)
    start_index = 0
    write_mode = "w"
    if args.resume:
        progress = _load_progress()
        prior_index, prior_params = progress["documents_done"], progress["params"]
        if prior_index and OUT.exists():
            if prior_params != params:
                log.error(
                    "--resume refused: the interrupted run used %s, this "
                    "invocation asked for %s. Resuming under different "
                    "chunking parameters would silently mix incompatible "
                    "windowing in one output file. Delete %s and %s to "
                    "start over, or match the original parameters.",
                    prior_params, params, OUT, PROGRESS_FILE,
                )
                return 2
            log.info("resuming: skipping %d already-chunked documents", prior_index)
            for _ in range(prior_index):
                next(records, None)
            start_index = prior_index
            write_mode = "a"
        else:
            log.info("--resume given but no prior progress found - starting fresh")
    else:
        _save_progress(0, params)  # a fresh run invalidates any old checkpoint

    n = write_jsonl(
        OUT,
        iter_chunks(records, args.size, stride, tok, tok_used,
                   args.batch_size, log, start_index=start_index,
                   run_params=params),
        mode=write_mode,
    )

    # chunk_stats.json reports the FULL file's current chunk count, not just
    # this run's contribution, so it stays accurate after a --resume run too.
    total_chunks = sum(1 for _ in read_jsonl(OUT))
    stats = {"chunks": total_chunks, "window_tokens": args.size,
             "overlap_tokens": args.overlap, "stride_tokens": stride,
             "tokenizer_used": tok_used}
    (OUT.parent / "chunk_stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    log.info("stage 06 | chunks_this_run=%d | chunks_total=%d | tokenizer=%s | %d/%d/%d",
             n, total_chunks, tok_used, args.size, args.overlap, stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
