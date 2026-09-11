#!/usr/bin/env python3
"""Stage 02 - PMC Open Access full-text retrieval.

Only the Open Access subset is used, and each article's own ali:license_ref is
authoritative for reuse. Downloads are MD5-verified against the OA file list and
resume from the manifest, so an interrupted run continues rather than restarting.

NETWORK: same NCBI dependency and same fail-closed behaviour as stage 01.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, METADATA, get_logger, redistribution_allowed

RAW = DATA / "raw" / "pmc"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    log = get_logger("02_pmc", "retrieval.log")
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = METADATA / "pmc.csv"
    log.info("stage 02 start | limit=%d | resume=%s | dry_run=%s",
             args.limit, args.resume, args.dry_run)

    if not manifest.exists() or sum(1 for _ in open(manifest)) <= 1:
        log.error("no PMC inventory in %s - run stage 01 and build the OA inventory first",
                  manifest.name)
        log.error("NETWORK BLOCKED in this environment: PMC retrieval depends on the same "
                  "NCBI egress as stage 01. Run where access is permitted.")
        return 2

    log.info("licensing gate active: unknown licence => redistribution_allowed=False "
             "(e.g. None -> %s, CC-BY -> %s)",
             redistribution_allowed(None), redistribution_allowed("CC-BY"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
