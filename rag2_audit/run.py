"""Run the reproduction audit.

    python -m rag2_audit.run                  # full audit, human-readable
    python -m rag2_audit.run --json out.json  # machine-readable
    python -m rag2_audit.run --only FLT       # one component family
    python -m rag2_audit.run --strict         # exit non-zero on FAIL

Exit codes: 0 = no FAIL, 1 = at least one FAIL (always, so CI catches a
regression in the filter's direction or the paper-specified constants).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rag2_audit import checks_filter, checks_pipeline, checks_structure  # noqa: E402,F401
from rag2_audit.registry import CHECKS, Result, Status, run_all  # noqa: E402

ORDER = [Status.FAIL, Status.PARTIAL, Status.UNKNOWN, Status.APPROXIMATION, Status.MANUAL, Status.PASS]
GLYPH = {
    Status.PASS: "PASS ", Status.FAIL: "FAIL ", Status.PARTIAL: "PART ",
    Status.UNKNOWN: "UNKN ", Status.APPROXIMATION: "APPX ", Status.MANUAL: "MANU ",
}


def verdict(results: List[Result]) -> str:
    """Overall verdict, decided by the worst outcomes present.

    Deliberately conservative: a single FAIL on a method-defining component is
    enough to withhold verification, and unresolved UNKNOWNs cap the verdict
    below VERIFIED however many checks pass.
    """
    counts = Counter(r.status for r in results)
    method_critical = {"filter scoring", "perplexity", "labeling", "retrieval", "reranking", "architecture"}
    critical_fail = any(
        r.status is Status.FAIL and r.component in method_critical for r in results
    )
    if critical_fail:
        return "NOT VERIFIED"
    if counts[Status.FAIL]:
        return "PARTIALLY VERIFIED"
    if counts[Status.UNKNOWN] or counts[Status.APPROXIMATION] or counts[Status.PARTIAL]:
        return "MOSTLY VERIFIED"
    return "VERIFIED"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="filter by check-id prefix or component name")
    parser.add_argument("--json", default="", help="write the full results to this path")
    parser.add_argument("--quiet", action="store_true", help="summary table only")
    parser.add_argument("--strict", action="store_true", help="also exit non-zero on PARTIAL/UNKNOWN")
    args = parser.parse_args(argv)

    results = run_all(args.only or None)
    if not results:
        print(f"no checks matched {args.only!r}; {len(CHECKS)} registered")
        return 2

    print("=" * 78)
    print("RAG2 REPRODUCTION AUDIT")
    print("=" * 78)
    by_status = {s: [r for r in results if r.status is s] for s in ORDER}
    for status in ORDER:
        for result in by_status[status]:
            print(f"[{GLYPH[status]}] {result.check_id:7s} {result.component:16s} {result.summary}")

    if not args.quiet:
        detailed = [r for r in results if r.status is not Status.PASS]
        if detailed:
            print()
            print("=" * 78)
            print("FINDINGS THAT NEED A DECISION")
            print("=" * 78)
            for result in sorted(detailed, key=lambda r: ORDER.index(r.status)):
                print(f"\n--- {result.check_id} [{result.status.value}] {result.component}: {result.summary}")
                for label, text in (
                    ("paper", result.paper_says), ("code", result.code_does),
                    ("impact", result.why_it_matters), ("fix", result.how_to_fix),
                ):
                    if text:
                        print(f"    {label:7s}: {_wrap(text)}")

    counts = Counter(r.status.value for r in results)
    print()
    print("=" * 78)
    print("  ".join(f"{name}={counts.get(name, 0)}" for name in
                    ("PASS", "FAIL", "PARTIAL", "UNKNOWN", "APPROXIMATION", "MANUAL")))
    print(f"VERDICT: {verdict(results)}")
    print("=" * 78)

    if args.json:
        payload = {
            "verdict": verdict(results),
            "counts": dict(counts),
            "results": [r.to_dict() for r in results],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")

    failed = counts.get("FAIL", 0)
    if args.strict:
        return 1 if failed or counts.get("PARTIAL", 0) or counts.get("UNKNOWN", 0) else 0
    return 1 if failed else 0


def _wrap(text: str, width: int = 66, indent: int = 13) -> str:
    import textwrap

    lines = textwrap.wrap(" ".join(text.split()), width=width)
    pad = " " * indent
    return ("\n" + pad).join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
