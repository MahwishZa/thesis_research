"""Build the dense index over the corpus. Run on the machine holding it.

    python -m experiments.shared.retrieval.build_index \
        --corpus corpus \
        --out experiments/results/index

Reads ``corpus/`` and writes nothing into it. Refuses to overwrite an
existing index, because the frozen evidence records which index it came from
and silently rebuilding one under the same path breaks that link.

Downloads the MedCPT article encoder (~0.44 GB) on first run. Nothing else.

**Memory-bounded and checkpointed.** At the real corpus's scale (4.3M+
chunks), holding every passage's text and the whole vector matrix in
process memory at once does not fit on typical laptop hardware - a
2026-09-24 diagnostic run showed even the text-only read alone exhausting
free RAM on a 16 GB machine. This script streams the corpus in two bounded
passes (``streaming_index_build.py``) instead, and checkpoints progress
every ``--checkpoint-every-batches`` batches so an interrupted run - kill,
crash, closed laptop lid - can continue with ``--resume`` instead of
restarting a multi-hour job from zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .encoders import medcpt_article_encoder
from .streaming_index_build import StreamingBuildError, encode_index_streaming, plan_index_build


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus",
                        help="corpus root (read-only)")
    parser.add_argument("--out", default="experiments/results/index",
                        help="directory to write the index into")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda; omit for CPU")
    parser.add_argument("--include-undated", action="store_true",
                        help="keep undated passages. Off by default: frozen "
                             "scope requires dated-only candidate sets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be indexed and exit without "
                             "loading a model or encoding anything. Writes a "
                             "small planning cache under <out>/_build/ "
                             "(passage ids + counts, not the corpus text) so "
                             "a later real build skips re-reading the corpus.")
    parser.add_argument("--on-duplicate", choices=("raise", "keep_first"),
                        default="raise",
                        help="what to do if the corpus has two chunks with "
                             "the same chunk_id. 'raise' (default) stops "
                             "immediately, safe for a corpus you haven't "
                             "diagnosed. 'keep_first' keeps the first "
                             "occurrence and records every dropped duplicate "
                             "in the build plan - only pass this once you've "
                             "confirmed what the duplicates actually are.")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted build found at --out "
                             "instead of refusing to proceed. Validates the "
                             "corpus file, encoder, and duplicate/dated-only "
                             "policy all still match the checkpoint before "
                             "continuing.")
    parser.add_argument("--checkpoint-every-batches", type=int, default=200,
                        help="how often (in encoded batches) to flush "
                             "progress to disk. Smaller = less work lost on "
                             "an interruption, at the cost of slightly more "
                             "disk I/O; the default matches the existing "
                             "progress-print cadence.")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and (out / "manifest.json").exists():
        print(f"refusing to overwrite an existing index at {out}. "
              "Frozen evidence records the index it came from; write a new "
              "directory instead.", file=sys.stderr)
        return 2

    dated_only = not args.include_undated
    t0 = time.time()

    def _progress(n: int) -> None:
        print(f"  ...{n:,} lines read ({time.time() - t0:.0f}s elapsed)")

    if args.dry_run:
        print("planning (this streams the corpus once, without holding it "
              "in memory - progress prints every 200,000 lines)...")
        try:
            plan = plan_index_build(
                args.corpus, out, dated_only=dated_only,
                on_duplicate=args.on_duplicate, on_progress=_progress,
            )
        except StreamingBuildError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"corpus snapshot : {plan.snapshot_id}")
        print(f"passages indexed: {plan.total}"
              f"{'' if dated_only else ' (including undated)'}")
        if plan.duplicates:
            print(f"duplicate chunk_ids dropped: {len(plan.duplicates)} "
                  f"(kept the first occurrence of each; recorded in "
                  f"{out / '_build' / 'plan_meta.json'})")
        print("dry run: no model loaded, no vectors written. A small "
              f"planning cache was written to {out / '_build'} so a real "
              "build (or a --dry-run repeat) can skip re-reading the "
              "corpus.")
        return 0 if plan.total else 1

    print(f"encoding (device={args.device or 'cpu'}, "
          f"batch size {args.batch_size})...")
    if not args.device:
        print("  no --device given: running on CPU. If this machine has a "
              "GPU, pass --device cuda - MedCPT's article encoder is small "
              "(~0.44 GB) and fits even a 4 GB card, and GPU encoding is "
              "typically far faster than CPU for a corpus this size.")

    def _encode_progress(done: int, total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        print(f"  ...{done:,}/{total:,} passages encoded "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    encoder = medcpt_article_encoder(batch_size=args.batch_size,
                                     device=args.device)
    try:
        result = encode_index_streaming(
            args.corpus, out, encoder, dated_only=dated_only,
            on_duplicate=args.on_duplicate, batch_size=args.batch_size,
            checkpoint_every_batches=args.checkpoint_every_batches,
            resume=args.resume, on_progress=_progress,
            on_batch_progress=_encode_progress,
        )
    except StreamingBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"wrote index to {result.out_dir} "
          f"({result.n_passages} x {result.dim})")
    if result.duplicates_count:
        print(f"duplicate chunk_ids dropped during this build: "
              f"{result.duplicates_count}")
    print(json.dumps({"corpus_snapshot": result.snapshot_id,
                      "encoder": result.encoder_name,
                      "n_passages": result.n_passages}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
