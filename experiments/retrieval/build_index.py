"""Build the dense index over the corpus. Run on the machine holding it.

    python -m experiments.retrieval.build_index \
        --corpus alzheimer_corpus \
        --out experiments/outputs/index

Reads ``alzheimer_corpus/`` and writes nothing into it. Refuses to overwrite an
existing index, because the frozen evidence records which index it came from
and silently rebuilding one under the same path breaks that link.

Downloads the MedCPT article encoder (~0.44 GB) on first run. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import dated_only, read_passages, snapshot_id
from .encoders import medcpt_article_encoder
from .index import build_index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="alzheimer_corpus",
                        help="corpus root (read-only)")
    parser.add_argument("--out", default="experiments/outputs/index",
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
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and (out / "manifest.json").exists():
        print(f"refusing to overwrite an existing index at {out}. "
              "Frozen evidence records the index it came from; write a new "
              "directory instead.", file=sys.stderr)
        return 2

    passages = read_passages(args.corpus)
    snapshot = snapshot_id(args.corpus)
    kept = passages if args.include_undated else dated_only(passages)

    print(f"corpus snapshot : {snapshot}")
    print(f"passages read   : {len(passages)}")
    print(f"passages indexed: {len(kept)}"
          f"{'' if args.include_undated else ' (dated only)'}")

    if not kept:
        print("nothing to index", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry run: no model loaded, nothing written")
        return 0

    encoder = medcpt_article_encoder(batch_size=args.batch_size,
                                     device=args.device)
    index = build_index(kept, encoder, corpus_snapshot=snapshot,
                        metadata={"dated_only": not args.include_undated})
    index.save(out)
    print(f"wrote index to {out} ({len(index.passage_ids)} × {index.dim})")
    print(json.dumps({"corpus_snapshot": snapshot,
                      "encoder": index.encoder_name,
                      "n_passages": len(index.passage_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
