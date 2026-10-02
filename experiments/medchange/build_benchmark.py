"""CLI: build the as-of MedChange benchmark (no network, no model).

    python -m experiments.medchange.build_benchmark --medchange-dir <clone of jvladika/MedChange>

Writes ``experiments/medchange/data/benchmark.jsonl`` (gitignored - the MedChange
release states no licence, so its text is not redistributed here) and
``experiments/medchange/manifest.json`` (counts, input hashes, seed, split ids:
safe to commit, enough to rebuild identically).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .benchmark import (
    BenchmarkError, assign_splits, build_items, file_sha256, load_groups, read_csv,
    rebuild_medchangeqa, verify_against_release,
)

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--medchange-dir", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"))
    ap.add_argument("--n-unchanged", type=int, default=250)
    ap.add_argument("--dev-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args(argv)

    ds = Path(args.medchange_dir) / "Datasets"
    paths = {n: ds / n for n in ("MedRevQA.csv", "AllStudyGroups.csv", "MedChangeQA.csv")}
    for p in paths.values():
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2
    try:
        medrev = {int(r[""]): r for r in read_csv(paths["MedRevQA.csv"])}
        groups = load_groups(read_csv(paths["AllStudyGroups.csv"]))
        rebuilt = rebuild_medchangeqa(medrev, groups)
        verify_against_release(rebuilt, medrev, read_csv(paths["MedChangeQA.csv"]))
        items = build_items(medrev, groups, rebuilt, n_unchanged=args.n_unchanged,
                            seed=args.seed)
        assign_splits(items, dev_fraction=args.dev_fraction, seed=args.seed)
    except BenchmarkError as exc:
        print(f"benchmark build refused: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

    c = collections.Counter
    usable = [i for i in items if not i.likely_label_noise]
    summary = {
        "source": "MedChange (Vladika et al., EMNLP 2025 Findings), github.com/jvladika/MedChange",
        "input_sha256": {n: file_sha256(p) for n, p in paths.items()},
        "seed": args.seed, "dev_fraction": args.dev_fraction,
        "n_unchanged_requested": args.n_unchanged,
        "medchangeqa_rebuilt_and_verified": len(rebuilt),
        "counts": {
            "items": len(items),
            "by_kind_split": {f"{k}/{s}": n for (k, s), n in
                              sorted(c((i.kind, i.split) for i in items).items())},
            "likely_label_noise_excluded": sum(i.likely_label_noise for i in items),
            "usable_changed": sum(i.kind == "changed" for i in usable),
            "decisive_flips": sum(i.decisive_flip for i in usable),
            "change_types": dict(c(i.change_type for i in usable if i.kind == "changed")),
            "ad_related": dict(c(i.kind for i in usable if i.ad_related)),
            "newest_date_precision": dict(c(i.newest.date_precision for i in items)),
        },
        "splits": {s: sorted(i.item_id for i in items if i.split == s)
                   for s in ("dev", "confirm")},
    }
    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2))
    print(json.dumps(summary["counts"], indent=2))
    print(f"wrote {out / 'benchmark.jsonl'} and {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
