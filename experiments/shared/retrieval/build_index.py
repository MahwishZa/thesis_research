"""Build the dense index over the corpus. Run on the machine holding it.

    python -m experiments.shared.retrieval.build_index \
        --corpus corpus \
        --out results/index

Reads ``corpus/`` and writes nothing into it. Refuses to overwrite an
existing index, because the frozen evidence records which index it came from
and silently rebuilding one under the same path breaks that link.

Downloads the MedCPT article encoder (~0.44 GB) on first run. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import dated_only, read_passages_with_snapshot
from .encoders import medcpt_article_encoder
from .index import build_index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus",
                        help="corpus root (read-only)")
    parser.add_argument("--out", default="results/index",
                        help="directory to write the index into")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda; omit for CPU")
    parser.add_argument("--include-undated", action="store_true",
                        help="keep undated passages. Off by default: frozen "
                             "scope requires dated-only candidate sets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be indexed and exit without "
                             "loading a model or writing anything")
    parser.add_argument("--on-duplicate", choices=("raise", "keep_first"),
                        default="raise",
                        help="what to do if the corpus has two chunks with "
                             "the same chunk_id. 'raise' (default) stops "
                             "immediately, safe for a corpus you haven't "
                             "diagnosed. 'keep_first' keeps the first "
                             "occurrence and writes a report of every "
                             "dropped duplicate to <out>/duplicate_chunks.json "
                             "- only pass this once you've confirmed what "
                             "the duplicates actually are.")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and (out / "manifest.json").exists():
        print(f"refusing to overwrite an existing index at {out}. "
              "Frozen evidence records the index it came from; write a new "
              "directory instead.", file=sys.stderr)
        return 2

    # One pass over the corpus file, not two (read_passages() then a
    # separate snapshot_id() re-read) - a real simplification, though
    # measurement showed the second read cost near-nothing (OS page cache).
    # For a multi-million-line real corpus, parsing time in Python is what
    # dominates, not I/O - so this prints progress every 200k lines rather
    # than sitting silent, which is what actually looks like "hanging."
    print(f"reading corpus (this can take a few minutes for a large "
          f"corpus - progress prints every 200,000 lines)...")
    t0 = time.time()

    def _progress(n: int) -> None:
        print(f"  ...{n:,} lines read ({time.time() - t0:.0f}s elapsed)")

    passages, snapshot, duplicates = read_passages_with_snapshot(
        args.corpus, on_progress=_progress, on_duplicate=args.on_duplicate
    )
    kept = passages if args.include_undated else dated_only(passages)

    print(f"corpus snapshot : {snapshot}")
    print(f"passages read   : {len(passages)}")
    print(f"passages indexed: {len(kept)}"
          f"{'' if args.include_undated else ' (dated only)'}")

    if duplicates:
        print(f"duplicate chunk_ids dropped: {len(duplicates)} "
              f"(kept the first occurrence of each)")
        if args.dry_run:
            print("  dry run: report not written (nothing is written in "
                  "--dry-run)")
        else:
            out.mkdir(parents=True, exist_ok=True)
            report_path = out / "duplicate_chunks.json"
            report_path.write_text(
                json.dumps(list(duplicates), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"  full report written to {report_path}")

    if not kept:
        print("nothing to index", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry run: no model loaded, nothing written")
        return 0

    print(f"encoding {len(kept):,} passages "
          f"(batch size {args.batch_size}, device={args.device or 'cpu'})...")
    if not args.device:
        print("  no --device given: running on CPU. If this machine has a "
              "GPU, pass --device cuda - MedCPT's article encoder is small "
              "(~0.44 GB) and fits even a 4 GB card, and GPU encoding is "
              "typically far faster than CPU for a corpus this size.")
    encode_t0 = time.time()

    def _encode_progress(done: int, total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        print(f"  ...batch {done:,}/{total:,} "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    encoder = medcpt_article_encoder(batch_size=args.batch_size,
                                     device=args.device)
    index = build_index(kept, encoder, corpus_snapshot=snapshot,
                        metadata={"dated_only": not args.include_undated},
                        on_progress=_encode_progress)
    print(f"encoding finished in {time.time() - encode_t0:.0f}s")
    index.save(out)
    print(f"wrote index to {out} ({len(index.passage_ids)} × {index.dim})")
    print(json.dumps({"corpus_snapshot": snapshot,
                      "encoder": index.encoder_name,
                      "n_passages": len(index.passage_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
