#!/usr/bin/env python3
"""Stage 03 - clinical guidelines and the currency pack.

Guidelines are acquired through each publisher's official route, never scraped,
and reuse rights are verified PER DOCUMENT - one organisation publishing openly
says nothing about the next document. Restricted guidance is recorded as
metadata only; its text never enters the distributable corpus.

This stage validates and reports on metadata/guidelines.csv. Acquisition of
restricted documents is a manual step by design.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import METADATA, get_logger, redistribution_allowed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true", default=True)
    args = ap.parse_args(argv)
    log = get_logger("03_guidelines", "retrieval.log")

    p = METADATA / "guidelines.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []
    log.info("stage 03 | %d guideline rows in %s", len(rows), p.name)

    distributable = restricted = unknown = 0
    for r in rows:
        lic = (r.get("license") or "").strip()
        if not lic:
            unknown += 1
            log.warning("%s: no licence recorded -> treated as NOT redistributable",
                        r.get("document_id"))
        elif redistribution_allowed(lic):
            distributable += 1
        else:
            restricted += 1
    log.info("licensing: %d distributable | %d restricted | %d unknown (fail closed)",
             distributable, restricted, unknown)
    if not rows:
        log.info("registry is empty - populate metadata/guidelines.csv per MANUAL steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
