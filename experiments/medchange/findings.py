"""Plain-language reports written by the pipeline: ``DEV_REPORT.md`` (before the go/no-go) and
``FINDINGS.md`` (after the confirmatory run). They only restate what the pre-declared rules say.

The wording of every conclusion is fixed in advance in §10 of the stage-2 protocol (``docs/experiment_plan.md`` at commit 92e3aaf); this module
chooses between the pre-written sentences according to the computed results and never adds
interpretation of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if Path(path).is_file() else None


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def dev_report(results: Path, gate1: Optional[dict]) -> tuple[str, Optional[str]]:
    """(markdown, gate2) from the files a dev run leaves in ``results``; gate2 is PASS, FAIL or None."""
    model = _load(results / "synthesis_model.json")
    audit = _load(results / "label_audit_dev.json")
    cons = _load(results / "consistency_auto_dev.json")
    gate2 = model["gate2"]["gate2"] if model else None
    lines = ["# Dev report (read this before saying \"go\")", "",
             "Everything below uses the dev split only. Nothing from the confirmatory split has been looked at.", ""]
    if gate1:
        lines += [f"## Gate 1 (stance pilot, machine checks): **{gate1['gate1']}**", ""]
        lines += [f"* {'PASS' if v else 'FAIL'}: {k}" for k, v in gate1["checks"].items()] + [""]
    if model:
        cv = model["cv"]
        lines += [f"## Gate 2 (does stance predict the gold verdict on dev?): **{gate2}**", "",
                  f"Selected hybrid: **{model['selected']}**. Stance-direction AUC {model['stance_direction_auc']:.3f} "
                  f"(0.5 = coin flip, needs >= 0.60). Constant-guess accuracy {_pct(cv['prior_accuracy'])}.", "",
                  "| Variant | cross-validated accuracy |", "|---|---|"]
        lines += [f"| {k} | {_pct(v['accuracy'])} |" for k, v in cv["variants"].items()]
        lines += [""] + [f"* {'PASS' if ok else 'FAIL'}: {name}" for name, ok in model["gate2"]["checks"].items()] + [""]
    if audit:
        lines += ["## Label audit (is the gold label reproducible by another model?)", "",
                  f"Agreement {_pct(audit['agreement'])}, kappa {audit['kappa']}, {audit['n_stable']} label-stable items "
                  f"of {audit['n_items']}.", ""]
    if cons:
        lines += ["## Consistency of stated verdicts", "",
                  f"Parse rate {_pct(cons['parse_rate'])} ({cons['parse_gate']}); independent judge agrees on "
                  f"{_pct(cons['agreement'])} of the sample (diagnostic).", ""]
    if gate2 == "PASS":
        lines += ["## What happens next", "", "Gate 2 passed. The confirmatory run will include the synthesis layer "
                  "(stance on the confirmatory pools) and test RQ1 and RQ2. Passing gate 2 does not show the layer "
                  "works: it passes about one time in three even if the layer does nothing."]
    elif gate2 == "FAIL":
        lines += ["## What happens next", "", "Gate 2 failed. The confirmatory run will test RQ1 only (retrieval versus no "
                  "evidence); RQ2 is reported as not supported on dev and not tested on the confirmatory split."]
    else:
        lines += ["## What happens next", "", "Gate 2 has not been computed yet."]
    return "\n".join(lines) + "\n", gate2


def findings(results: Path) -> str:
    rep = _load(results / "stage2_analysis_confirm.json")
    if not rep:
        return "# Findings\n\nThe confirmatory analysis has not been run.\n"
    audit = _load(results / "label_audit_confirm.json")
    cons = _load(results / "consistency_auto_confirm.json")
    r = rep["reading"]
    pf = rep["primary_family"]
    lines = ["# Findings (confirmatory split, run once)", "",
             "Each conclusion is chosen by the rules fixed in `docs/experiment_plan.md` before the data were opened.", ""]
    rq1 = pf.get("RQ1 B1 vs B0")
    if rq1:
        lines += [f"## RQ1: does as-of retrieval beat no evidence?  **{r['RQ1']}**", "",
                  f"B1 minus B0 = {100 * rq1['diff_a_minus_b']:+.1f} pp (95% CI {100 * rq1['ci95'][0]:+.1f} to "
                  f"{100 * rq1['ci95'][1]:+.1f}; Holm p = {rq1['holm_p']:.4f}).", ""]
        if r.get("RQ1_note"):
            lines += [f"Reading: {r['RQ1_note']}.", ""]
    lines += [f"## RQ2: does the synthesis layer beat ordinary retrieval-augmented answering?  **{r['RQ2_tier']}**", ""]
    rq2 = pf.get(f"RQ2 {rep['primary_arm']} vs B1R")
    if rq2:
        lines += [f"{rep['primary_arm']} minus B1R = {100 * rq2['diff_a_minus_b']:+.1f} pp (95% CI "
                  f"{100 * rq2['ci95'][0]:+.1f} to {100 * rq2['ci95'][1]:+.1f}; Holm p = {rq2['holm_p']:.4f}).", ""]
    lines += [f"* {'met' if v else ('not met' if v is False else 'not assessed')}: {k}" for k, v in r["criteria"].items()]
    lines += [""]
    tier = r["RQ2_tier"]
    lines += ["## What this means", ""]
    if tier == "genuine positive":
        lines.append("All pre-declared criteria were met: the synthesis layer improved on both the refitted and the raw "
                     "answer, not only by abstaining more, and the direction held on label-stable items.")
    elif tier == "fragile positive":
        lines.append("RQ2 was confirmed against B1R, but at least one stricter criterion was not met or could not be "
                     "assessed; this is reported as a fragile positive, not as an improvement.")
    elif tier == "not run":
        lines.append("The synthesis layer was not tested on the confirmatory split because gate 2 failed on dev; "
                     "the thesis reports RQ1 and the negative dev evidence for RQ2.")
    else:
        lines.append("RQ2 was not confirmed. With this many questions only effects of about 5 points or more can be "
                     "confirmed, so smaller gains cannot be excluded; the estimate and its interval are the result.")
    if audit:
        lines += ["", f"Label reproducibility: another model agrees with {_pct(audit['agreement'])} of the gold labels "
                  f"(kappa {audit['kappa']}); this limits what any accuracy number can mean."]
    if cons:
        lines += [f"Stated-verdict consistency (independent judge): {_pct(cons['agreement'])}; parse rate "
                  f"{_pct(cons['parse_rate'])}."]
    lines += ["", "Full tables: `stage2_analysis_confirm.md`."]
    return "\n".join(lines) + "\n"


def write_dev_report(results: Path, gate1: Optional[dict]) -> Optional[str]:
    text, gate2 = dev_report(results, gate1)
    (results / "DEV_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    return gate2


def write_findings(results: Path) -> None:
    (results / "FINDINGS.md").write_text(findings(results), encoding="utf-8", newline="\n")
