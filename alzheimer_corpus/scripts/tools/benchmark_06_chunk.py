#!/usr/bin/env python3
"""Benchmark Stage 06 on a subset before committing to a full run.

Run this on your own machine (where the real MedCPT tokenizer and
transformers/torch are available - this sandbox has neither). It runs
06_chunk.py's actual chunking code - the same iter_chunks()/chunk_batch()
functions the real run uses - against the first N documents of your real
input file, measures elapsed time and documents/sec, and extrapolates to
the full corpus.

Not a pipeline stage (no number, not read by any other stage) - a
standalone diagnostic tool, kept out of scripts/ so that directory holds
only the numbered pipeline (01-07 + _common.py).

Usage:
    python alzheimer_corpus/scripts/tools/benchmark_06_chunk.py \
        --input alzheimer_corpus/data/deduplicated/documents.jsonl \
        --sample 500 --batch-size 200

Only chunking is timed; nothing is written to the real output path.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from _common import DATA, get_logger, read_jsonl  # noqa: E402
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "chunk06", SCRIPTS_DIR / "06_chunk.py"
)
chunk06 = importlib.util.module_from_spec(_spec)
sys.modules["chunk06"] = chunk06
_spec.loader.exec_module(chunk06)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DATA / "deduplicated" / "documents.jsonl"))
    ap.add_argument("--tokenizer", default="ncbi/MedCPT-Article-Encoder")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--sample", type=int, default=500,
                     help="number of documents to benchmark against")
    args = ap.parse_args(argv)

    log = get_logger("benchmark_06_chunk", "quality_control.log")

    src = Path(args.input)
    if not src.exists():
        log.error("no input at %s - run stages 04/05 first", src)
        return 2

    total_docs = sum(1 for _ in read_jsonl(src))
    if total_docs == 0:
        log.error("%s is empty", src)
        return 2

    sample_n = min(args.sample, total_docs)
    records = itertools.islice(read_jsonl(src), sample_n)

    tok = chunk06.get_tokenizer(args.tokenizer, log)
    tok_used = "whitespace" if tok is None else args.tokenizer
    stride = args.size - args.overlap

    start = time.monotonic()
    n_chunks = 0
    for _ in chunk06.iter_chunks(
        records, args.size, stride, tok, tok_used, args.batch_size, log
    ):
        n_chunks += 1
    elapsed = time.monotonic() - start

    docs_per_sec = sample_n / elapsed if elapsed > 0 else float("inf")
    est_full_seconds = total_docs / docs_per_sec if docs_per_sec else float("inf")

    print(f"\nBenchmark result ({tok_used}, batch_size={args.batch_size}):")
    print(f"  sampled documents : {sample_n} / {total_docs} total in {src.name}")
    print(f"  elapsed           : {elapsed:.1f}s")
    print(f"  throughput        : {docs_per_sec:.2f} docs/sec")
    print(f"  chunks produced   : {n_chunks}")
    print(f"  estimated full run: {est_full_seconds / 60:.1f} min "
          f"({est_full_seconds / 3600:.2f} hr) for all {total_docs} documents")
    print(
        "\nNote: this extrapolation assumes the sampled documents are "
        "representative of section-length distribution across the corpus. "
        "If the sample is drawn from the start of the file and documents "
        "are not shuffled, consider re-running with a larger --sample or a "
        "shuffled input for a more reliable estimate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
