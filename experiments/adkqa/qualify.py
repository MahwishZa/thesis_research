"""Qualification of the independent verifier (``docs/protocol.md`` §8): re-score a label-audit file of the development split.

    python -m experiments.adkqa.qualify --audit qual\\data\\label_audit_dev.jsonl --model-name Phi-3.5-mini-instruct.Q4_K_M.gguf

The audit file is written by ``experiments.medchange.label_audit`` (unchanged). Its stored reply is read with
``spec.parse_verdict`` instead of the audit's strict "LABEL:" pattern; the replies are not regenerated. The thresholds are those
declared: agreement at least 75%, kappa at least 0.60, unparsed at most 2%. Only counts are written."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from experiments.medchange.label_audit import cohens_kappa

from . import spec

HERE = Path(__file__).resolve().parent
MIN_AGREEMENT, MIN_KAPPA, MAX_UNPARSED = 0.75, 0.60, 0.02


def rescore(rows: Sequence[dict]) -> dict:
    newest = [r for r in rows if r.get("which") == "newest"]
    read = [(r["gold"], spec.parse_verdict(r.get("raw", ""))) for r in newest]
    pairs = [(g, p) for g, p in read if p]
    unparsed = len(newest) - len(pairs)
    agreement = sum(g == p for g, p in pairs) / len(pairs) if pairs else None
    kappa = cohens_kappa(pairs)
    share = unparsed / len(newest) if newest else None
    per_class = {c: {"n": sum(g == c for g, _ in pairs),
                     "agreement": round(sum(g == p for g, p in pairs if g == c) / max(1, sum(g == c for g, _ in pairs)), 4)}
                 for c in spec.VERDICTS}
    checks = {"agreement_at_least_75pct": agreement is not None and agreement >= MIN_AGREEMENT,
              "kappa_at_least_0_60": kappa is not None and kappa >= MIN_KAPPA,
              "unparsed_at_most_2pct": share is not None and share <= MAX_UNPARSED}
    return {"n_items": len(newest), "n_unparsed": unparsed, "unparsed_share": None if share is None else round(share, 4),
            "agreement": None if agreement is None else round(agreement, 4), "kappa": kappa, "per_gold_class": per_class,
            "checks": checks, "qualified": all(checks.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--model-name", default="")
    ap.add_argument("--out", default=str(HERE / "results" / "adkqa_verifier_qualification.json"))
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in Path(a.audit).read_text(encoding="utf-8").splitlines() if line.strip()]
    rep = rescore(rows)
    rep["verifier"] = a.model_name
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=2))
    print("QUALIFIED" if rep["qualified"] else "NOT QUALIFIED")
    return 0 if rep["qualified"] else 3


if __name__ == "__main__":
    sys.exit(main())
