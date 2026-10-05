"""One command per phase of the realigned study (docs/experimentation.md §8).

    python -m experiments.medchange.rag2_pipeline dev --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
    python -m experiments.medchange.rag2_pipeline confirm --go --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
    python -m experiments.medchange.rag2_pipeline ad --go --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
    python -m experiments.medchange.rag2_pipeline status

``dev``: integrity checks; rationales; candidate lists; filter; answers R2, R2C, R2V and R2V-ND (with
``--ablations`` also R2-RQ, R2-BR and R2-NF); the optional directness judge (``--judge-path``); analysis; the
dev report with the pre-declared dev check; the design record ``results/rag2_design.json``; publishing the
shareable outputs; with ``--commit``, commit and push to main.

``confirm`` (once): refuses to start without ``--go``, without the dev report, or unless the design record on
origin/main equals the current design (settings, prompts, model file). Then the same steps on the confirmatory
split for R2, R2C, R2V and R2V-ND (``--no-temporal-ablation`` leaves R2V-ND out, decided before the run),
analysis, findings, publishing and, with ``--commit``, commit and push.

``ad`` (once, same guards as ``confirm``): the Alzheimer's/dementia secondary test set built by ``ad_benchmark``:
as-of PubMed records and abstracts (network), B0 and B1 answers, then R2, R2C and R2V, analysis, findings.

Every step is resumable: after an interruption, rerun the same command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import rag2 as R
from .generate_answers import file_sha256, load_jsonl
from .pipeline import EXPECTED_ITEMS, _checks, commit_and_push, execute, frozen_is_pushed, py

HERE = Path(__file__).resolve().parent
DESIGN = "/".join(("experiments", "medchange", "results", "rag2_design.json"))   # written by the dev phase
SHARE = ("rag2_rationales", "rag2_filter", "rag2_answers", "rag2_directness")
DEV_ARMS = ("R2", "R2C", "R2V", "R2V-ND")
ABLATION_ARMS = ("R2-NF", "R2-RQ", "R2-BR")
AD_ARMS = ("R2", "R2C", "R2V")
FINDINGS = {"confirm": "RAG2_FINDINGS.md", "ad": "RAG2_FINDINGS_AD.md"}
MAX_UNPARSED = 0.05
MIN_VALID = 0.95
BASELINE_FLOOR_PP = 5.0


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def preflight(data: Path, split: str) -> list[tuple[str, bool, str]]:
    """Inputs the realigned run needs: benchmark, as-of records and abstracts, and the B0/B1 answers."""
    rows = []
    bench = [r for r in load_jsonl(data / "benchmark.jsonl") if not r["likely_label_noise"]]
    items = [r for r in bench if r["split"] == split]
    rows.append((f"{split} has {EXPECTED_ITEMS[split]} usable items", len(items) == EXPECTED_ITEMS[split],
                 f"found {len(items)}"))
    no_probe = [r["item_id"] for r in items if not (data / "pubmed_g0" / f"{r['item_id']}.json").is_file()]
    rows.append(("every item has its as-of PubMed records", not no_probe, f"{len(no_probe)} missing"))
    rows.append(("the abstract cache exists", (data / "abstracts.jsonl").is_file(), ""))
    answers = {(r["item_id"], r["arm"]) for r in load_jsonl(data / f"answers_{split}.jsonl")}
    lacking = [r["item_id"] for r in items for arm in ("B0", "B1") if (r["item_id"], arm) not in answers]
    rows.append(("B0 and B1 answers exist for every item (reused)", not lacking, f"{len(lacking)} missing"))
    return rows


def design_differences(results: Path, model_path: str) -> list[str]:
    path = results / "rag2_design.json"
    if not path.is_file():
        return ["no design record (run the dev phase)"]
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = R.design_record(file_sha256(model_path), recorded.get("encoders"))
    return [k for k in current if recorded.get(k) != current[k]]


def write_design(results: Path, model_path: str) -> tuple[bool, str]:
    results.mkdir(parents=True, exist_ok=True)
    (results / "rag2_design.json").write_text(
        json.dumps(R.design_record(file_sha256(model_path)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True, "wrote the design record"


def dev_check(rep: dict) -> dict:
    """The pre-declared dev check (docs/experimentation.md §8)."""
    gen, ver = rep["generation"], rep["verification"]
    parse = {arm: (row["unparsed"] / row["all"]["n"]) <= MAX_UNPARSED
             for arm, row in gen.items() if row.get("all")}
    valid = {arm: row["valid_rate"] >= MIN_VALID for arm, row in ver.items() if arm != "R2C"}
    a_ok = all(parse.values()) and all(valid.values())
    acc = {arm: row["all"]["accuracy"] for arm, row in gen.items() if row.get("all")}
    b_ok = "R2" in acc and "B1" in acc and (acc["R2"] - acc["B1"]) * 100 >= -BASELINE_FLOOR_PP
    c_ok = "R2V" in acc and "R2" in acc and acc["R2V"] - acc["R2"] >= 0
    status = "READY" if a_ok and b_ok and c_ok else ("DEFECT" if not (a_ok and b_ok) else "REVISE ONCE")
    return {"a_parse_and_valid": a_ok, "parse_by_arm": parse, "valid_by_arm": valid,
            "b_baseline_works": b_ok, "c_direction": c_ok, "accuracy": acc, "status": status}


def _analysis(results: Path, split: str) -> Optional[dict]:
    path = results / f"rag2_analysis_{split}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_dev_report(results: Path) -> tuple[bool, str]:
    rep = _analysis(results, "dev")
    if rep is None:
        return False, "no dev analysis"
    check = dev_check(rep)
    acc = check["accuracy"]
    p = rep["primary"] or {}
    lines = ["# Realigned study: dev report (exploratory)", "",
             "Dev split only (226 items); nothing here is a confirmatory result. Full tables: "
             "`rag2_analysis_dev.md`.", "",
             "* Accuracy: " + ", ".join(f"{a} {100 * v:.1f}%" for a, v in acc.items()) + ".",
             f"* R2V − R2: {100 * p.get('diff_a_minus_b', float('nan')):+.1f} pp "
             f"(95% CI {100 * p.get('ci95', [float('nan')] * 2)[0]:+.1f} to {100 * p.get('ci95', [float('nan')] * 2)[1]:+.1f})."
             if p else "* R2V − R2: not available.", "",
             "## Pre-declared dev check", "",
             f"* (a) every arm parses ≥ 95% and the verifier output is valid ≥ 95%: "
             f"**{'PASS' if check['a_parse_and_valid'] else 'FAIL'}**",
             f"* (b) the adapted RAG² baseline works (R2 ≥ B1 − {BASELINE_FLOOR_PP:.0f} pp): "
             f"**{'PASS' if check['b_baseline_works'] else 'FAIL'}**",
             f"* (c) direction (R2V − R2 ≥ 0): **{'PASS' if check['c_direction'] else 'FAIL'}**", "",
             f"Status: **{check['status']}**. "]
    lines.append({"READY": "Freeze the design (it is written and committed with --commit) and, after reading this "
                           "report, run the confirmatory phase once with --go.",
                  "DEFECT": "A defect: fix its cause, record it in docs/experimentation.md §9 and rerun the dev phase.",
                  "REVISE ONCE": "The plan allows one recorded revision of the verification prompt on dev; then the "
                                 "design is frozen whatever dev shows."}[check["status"]])
    (results / "RAG2_DEV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, f"dev check: {check['status']}"


def write_findings(results: Path, split: str = "confirm") -> tuple[bool, str]:
    rep = _analysis(results, split)
    if rep is None:
        return False, f"no {split} analysis"
    p = rep["primary"]
    title = {"confirm": "confirmatory split", "ad": "Alzheimer's/dementia secondary test set"}[split]
    lines = [f"# Findings of the realigned study ({title}, run once)", "",
             "Each conclusion follows the rules fixed in `docs/experimentation.md` before the run.", "",
             "## Requirement: R2V at least 1.0 pp above the adapted RAG² baseline (R2)", ""]
    if p:
        lines += [f"R2V − R2 = {100 * p['diff_a_minus_b']:+.1f} pp (95% CI {100 * p['ci95'][0]:+.1f} to "
                  f"{100 * p['ci95'][1]:+.1f}; exact McNemar p = {p['mcnemar_p']:.4f}).", "",
                  f"Reading: **{rep['requirement']}**."]
    else:
        lines.append("Not run.")
    lines += ["", "## Secondary comparisons (Holm)", ""]
    for name, r in rep["secondary"].items():
        lines.append(f"* {name}: {100 * r['diff_a_minus_b']:+.1f} pp (95% CI {100 * r['ci95'][0]:+.1f} to "
                     f"{100 * r['ci95'][1]:+.1f}; Holm p = {r['holm_p']:.4f}) — "
                     f"{'confirmed' if r['confirmed'] else 'not confirmed'}")
    lines += ["", f"Full tables: `rag2_analysis_{split}.md`."]
    (results / FINDINGS[split]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, f"wrote {FINDINGS[split]}"


def publish(data: Path, results: Path, split: str) -> tuple[bool, str]:
    """Copy the outputs without publisher text to results/; the candidate lists without titles/abstracts."""
    results.mkdir(parents=True, exist_ok=True)
    copied = 0
    stems = SHARE + (("answers",) if split == "ad" else ())      # B0/B1 of the new split
    for stem in stems:
        for name in (f"{stem}_{split}.jsonl", f"{stem}_{split}.config.json"):
            if (data / name).is_file():
                shutil.copyfile(data / name, results / name)
                copied += 1
    lists = data / f"rag2_lists_{split}.jsonl"
    if lists.is_file():
        with open(results / f"rag2_lists_{split}.ids.jsonl", "w", encoding="utf-8", newline="\n") as h:
            for rec in load_jsonl(lists):
                h.write(json.dumps(R.strip_text(rec)) + "\n")
        copied += 1
        if (data / f"rag2_lists_{split}.config.json").is_file():
            shutil.copyfile(data / f"rag2_lists_{split}.config.json", results / f"rag2_lists_{split}.config.json")
            copied += 1
    return True, f"copied {copied} files to results/"


# --------------------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------------------

def run_steps(a, split: str, arms: Sequence[str], variants: Sequence[str]) -> list:
    common = ["--split", split]
    steps = [(f"rationales ({split})", py("rag2_run", "rationale", *common, "--model-path", a.model_path,
                                          "--n-threads", a.n_threads)),
             (f"candidate lists ({split})", py("rag2_run", "lists", *common)),
             (f"filter ({split})", py("rag2_run", "filter", *common, "--model-path", a.model_path,
                                      "--n-threads", a.n_threads, "--variants", *variants)),
             (f"answers ({split}): {' '.join(arms)}", py("rag2_run", "answers", *common, "--model-path", a.model_path,
                                                       "--n-threads", a.n_threads, "--arms", *arms))]
    if a.judge_path:
        steps.append((f"directness judge ({split})", py("rag2_run", "judge", *common, "--model-path", a.judge_path,
                                                         "--n-threads", a.n_threads)))
    return steps


def dev_plan(a, data: Path, results: Path) -> list:
    arms = DEV_ARMS + (ABLATION_ARMS if a.ablations else ())
    variants = ("R2", "R2-RQ", "R2-BR") if a.ablations else ("R2",)
    steps = [("preflight (dev)", lambda: _checks(preflight(data, "dev")))]
    steps += run_steps(a, "dev", arms, variants)
    steps += [("analysis (dev)", py("analyze_rag2", "--split", "dev", "--out-dir", results,
                                    "--label-audit", results / "label_audit_dev.jsonl")),
              ("publish outputs (dev)", lambda: publish(data, results, "dev")),
              ("dev report", lambda: write_dev_report(results)),
              ("design record", lambda: write_design(results, a.model_path))]
    if a.commit:
        steps.append(("commit and push", lambda: commit_and_push("Realigned study: dev run, dev report, design record")))
    return steps


def confirm_plan(a, data: Path, results: Path) -> list:
    arms = ("R2", "R2C", "R2V") + (() if a.no_temporal_ablation else ("R2V-ND",))
    steps = [("preflight (confirm)", lambda: _checks(preflight(data, "confirm")))]
    steps += run_steps(a, "confirm", arms, ("R2",))
    steps += [("analysis (confirm)", py("analyze_rag2", "--split", "confirm", "--out-dir", results,
                                        "--label-audit", results / "label_audit_confirm.jsonl")),
              ("publish outputs (confirm)", lambda: publish(data, results, "confirm")),
              ("findings", lambda: write_findings(results))]
    if a.commit:
        steps.append(("commit and push", lambda: commit_and_push("Realigned study: confirmatory run and findings")))
    return steps


def ad_plan(a, data: Path, results: Path) -> list:
    count = lambda: _checks([preflight(data, "ad")[0]])
    steps = [("Alzheimer's items present (run ad_benchmark first)", count),
             ("as-of PubMed records (ad)", py("pubmed_asof", "--split", "ad")),
             ("candidate pools and abstracts (ad)", py("freeze_candidates", "--split", "ad", "--device", "cpu")),
             ("B0 and B1 answers (ad)", py("generate_answers", "--split", "ad", "--arms", "B0", "B1",
                                           "--model-path", a.model_path, "--n-threads", a.n_threads)),
             ("preflight (ad)", lambda: _checks(preflight(data, "ad")))]
    steps += run_steps(a, "ad", AD_ARMS, ("R2",))
    steps += [("analysis (ad)", py("analyze_rag2", "--split", "ad", "--out-dir", results)),
              ("publish outputs (ad)", lambda: publish(data, results, "ad")),
              ("findings (ad)", lambda: write_findings(results, "ad"))]
    if a.commit:
        steps.append(("commit and push", lambda: commit_and_push("Realigned study: Alzheimer's/dementia test set")))
    return steps


def status(data: Path, results: Path) -> int:
    dev = _analysis(results, "dev")
    pushed = frozen_is_pushed(rel=DESIGN)[0] if (results / "rag2_design.json").is_file() else False
    print(json.dumps({"dev_answers": len(load_jsonl(data / "rag2_answers_dev.jsonl")),
                      "dev_check": dev_check(dev)["status"] if dev else None,
                      "design_record_pushed": pushed,
                      "confirm_answers": len(load_jsonl(data / "rag2_answers_confirm.jsonl")),
                      "findings": (results / "RAG2_FINDINGS.md").is_file(),
                      "ad_answers": len(load_jsonl(data / "rag2_answers_ad.jsonl")),
                      "ad_findings": (results / "RAG2_FINDINGS_AD.md").is_file()}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("phase", choices=("dev", "confirm", "ad", "status"))
    ap.add_argument("--model-path", help="the generator GGUF (Meta-Llama-3-8B-Instruct Q4_K_M)")
    ap.add_argument("--judge-path", default=None, help="optional second-family GGUF for the directness judge")
    ap.add_argument("--n-threads", default="6")
    ap.add_argument("--ablations", action="store_true", help="dev only: also R2-RQ, R2-BR and R2-NF")
    ap.add_argument("--no-temporal-ablation", action="store_true", help="confirm only: leave R2V-ND out")
    ap.add_argument("--go", action="store_true", help="confirm only: your explicit go after reading the dev report")
    ap.add_argument("--commit", action="store_true", help="commit and push the results to main after the phase")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    data, results = Path(a.data_dir), Path(a.results_dir)
    if a.phase == "status":
        return status(data, results)
    if not a.model_path or (not a.dry_run and not Path(a.model_path).is_file()):
        print("--model-path must point to the generator GGUF file", file=sys.stderr)
        return 2
    if a.judge_path and not a.dry_run and not Path(a.judge_path).is_file():
        print(f"judge model file not found: {a.judge_path}", file=sys.stderr)
        return 2
    if a.phase == "dev":
        return execute(dev_plan(a, data, results), a.dry_run)
    if not a.go:
        print(f"the {a.phase} run needs your explicit go: read results/RAG2_DEV_REPORT.md, then add --go",
              file=sys.stderr)
        return 2
    if not a.dry_run:
        if not (results / "RAG2_DEV_REPORT.md").is_file():
            print("refusing to start: no dev report (run the dev phase first)", file=sys.stderr)
            return 2
        ok, why = frozen_is_pushed(rel=DESIGN)
        if not ok:
            print(f"refusing to start: {why}", file=sys.stderr)
            return 2
        differs = design_differences(results, a.model_path)
        if differs:
            print("refusing to start: the current design differs from the frozen record in "
                  + ", ".join(differs), file=sys.stderr)
            return 2
    plan = confirm_plan if a.phase == "confirm" else ad_plan
    return execute(plan(a, data, results), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
