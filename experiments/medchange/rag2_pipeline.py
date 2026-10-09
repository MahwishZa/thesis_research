"""One command per phase of the realigned study (docs/protocol.md §6, docs/reproducibility.md §4).

    python -m experiments.medchange.rag2_pipeline dev --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
    python -m experiments.medchange.rag2_pipeline confirm --go --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
    python -m experiments.medchange.rag2_pipeline ad --go --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
    python -m experiments.medchange.rag2_pipeline status

``dev``: integrity checks; rationales; candidate lists; filter; answers R2, R2C, R2V and R2V-ND (with
``--ablations`` also R2-RQ, R2-BR and R2-NF); the optional directness judge (``--judge-path``); analysis; the
dev report with the pre-declared dev check; the design record ``results/rag2_design.json``; publishing the
shareable outputs; with ``--commit``, commit (the push is yours: ``git push origin main``).

``confirm`` (once): refuses to start without ``--go``, without the dev report, or unless the design record on
origin/main equals the current design (settings, prompts, model file). Then the same steps on the confirmatory
split for R2, R2C, R2V and R2V-ND (``--no-temporal-ablation`` leaves R2V-ND out, decided before the run),
analysis, findings, publishing and, with ``--commit``, commit (never push).

``fresh`` (once; the pre-registered test of ``protocol.md`` §10 on the questions built by ``fresh_benchmark``): refuses to start unless
the pre-registration is in force, ``manifest_fresh.json`` is on origin/main and the items match it, and the guards of ``confirm`` hold; as-of
records and abstracts (network), rationales, candidate lists, filter, answers R2, R2C and R2V, then the blinded analysis ``analyze_fresh``.

``ad`` (once, same guards as ``confirm``): the Alzheimer's/dementia secondary test set built by ``ad_benchmark``:
as-of PubMed records and abstracts (network), B0 and B1 answers, then R2, R2C and R2V, analysis, findings.

Every step is resumable: after an interruption, rerun the same command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence

from . import rag2 as R
from .generate_answers import file_sha256, load_jsonl
from .runner import EXPECTED_ITEMS, commit_results, execute, frozen_is_pushed, print_checks, py

HERE = Path(__file__).resolve().parent
DESIGN = "/".join(("experiments", "medchange", "results", "rag2_design.json"))   # written by the dev phase
SHARE = ("rag2_rationales", "rag2_filter", "rag2_answers", "rag2_directness")
DEV_ARMS = ("R2", "R2C", "R2V", "R2V-ND")
ABLATION_ARMS = ("R2-NF", "R2-RQ", "R2-BR")
AD_ARMS = ("R2", "R2C", "R2V")
FRESH_ARMS = ("R2", "R2C", "R2V")
FINDINGS = {"confirm": "RAG2_FINDINGS.md", "ad": "RAG2_FINDINGS_AD.md"}
MANIFEST_FRESH = "/".join(("experiments", "medchange", "manifest_fresh.json"))
IN_FORCE = re.compile(r"\*\*Status\.\*\* This section is IN FORCE since \d{4}-\d{2}-\d{2}")
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
    if split != "fresh":                      # the fresh test does not run B0 and B1
        answers = {(r["item_id"], r["arm"]) for r in load_jsonl(data / f"answers_{split}.jsonl")}
        lacking = [r["item_id"] for r in items for arm in ("B0", "B1") if (r["item_id"], arm) not in answers]
        rows.append(("B0 and B1 answers exist for every item (reused)", not lacking, f"{len(lacking)} missing"))
    return rows


def freeze_problems(data: Path, root: Path = HERE.parents[1]) -> list[str]:
    """Why the pre-registered test may not start yet (protocol §10.8); empty when the freeze is complete."""
    problems = []
    protocol = root / "docs" / "protocol.md"
    if not (protocol.is_file() and IN_FORCE.search(protocol.read_text(encoding="utf-8"))):
        problems.append("protocol.md §10 is not marked IN FORCE (the freeze of §10.8 is not complete)")
    ok, why = frozen_is_pushed(rel=MANIFEST_FRESH)
    if not ok:
        problems.append(why)
    else:
        manifest = json.loads((root / MANIFEST_FRESH).read_text(encoding="utf-8"))
        ids = [r["item_id"] for r in load_jsonl(data / "benchmark.jsonl") if r["split"] == "fresh"]
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != manifest["item_ids_sha256"]:
            problems.append("the fresh items in benchmark.jsonl differ from manifest_fresh.json (run fresh_benchmark)")
    return problems


def _write(path: Path, text: str) -> None:
    """UTF-8 with LF line endings on every platform (a Windows checkout must stay byte-identical)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


PACKAGES = ("numpy", "torch", "transformers", "llama-cpp-python", "matplotlib")


def _git(*args: str, repo: Optional[Path] = None) -> Optional[str]:
    try:
        done = subprocess.run(["git", *args], cwd=repo or HERE.parents[1], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def environment_record(phase: str, model_path: str, judge_path: Optional[str] = None) -> dict:
    """What a rerun needs to know about the machine and the code: versions, the git commit the code was at and
    whether tracked code differed from it (``results/`` excluded, because the run itself writes there).
    Informational only: nothing compares it, so a different machine never blocks a run."""
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)experiments/medchange/results")
    return {"phase": phase, "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "git_commit": _git("rev-parse", "HEAD"),
            "tracked_code_modified": bool(changed) if changed is not None else None,
            "generator_file": Path(model_path).name, "judge_file": Path(judge_path).name if judge_path else None}


def write_environment(results: Path, phase: str, model_path: str, judge_path: Optional[str] = None) -> tuple[bool, str]:
    results.mkdir(parents=True, exist_ok=True)
    rec = environment_record(phase, model_path, judge_path)
    _write(results / f"rag2_environment_{phase}.json", json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return True, f"recorded the environment (commit {str(rec['git_commit'])[:8]})"


def design_differences(results: Path, model_path: str) -> list[str]:
    path = results / "rag2_design.json"
    if not path.is_file():
        return ["no design record (run the dev phase)"]
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = R.design_record(file_sha256(model_path), recorded.get("encoders"))
    return [k for k in current if recorded.get(k) != current[k]]


def write_design(results: Path, model_path: str) -> tuple[bool, str]:
    results.mkdir(parents=True, exist_ok=True)
    _write(results / "rag2_design.json",
           json.dumps(R.design_record(file_sha256(model_path)), indent=2, sort_keys=True) + "\n")
    return True, "wrote the design record"


def dev_check(rep: dict) -> dict:
    """The pre-declared dev check (docs/protocol.md §6)."""
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
                  "DEFECT": "A defect: fix its cause, record it in docs/protocol.md §7 and rerun the dev phase.",
                  "REVISE ONCE": "The plan allows one recorded revision of the verification prompt on dev; then the "
                                 "design is frozen whatever dev shows."}[check["status"]])
    _write(results / "RAG2_DEV_REPORT.md", "\n".join(lines) + "\n")
    return True, f"dev check: {check['status']}"


def write_findings(results: Path, split: str = "confirm") -> tuple[bool, str]:
    rep = _analysis(results, split)
    if rep is None:
        return False, f"no {split} analysis"
    p = rep["primary"]
    title = {"confirm": "confirmatory split", "ad": "Alzheimer's/dementia secondary test set"}[split]
    lines = [f"# Findings of the realigned study ({title}, run once)", "",
             "Each conclusion follows the rules fixed in `docs/protocol.md` and `docs/evaluation.md` before the run.", "",
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
    _write(results / FINDINGS[split], "\n".join(lines) + "\n")
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
    steps = [("preflight (dev)", lambda: print_checks(preflight(data, "dev")))]
    steps += run_steps(a, "dev", arms, variants)
    steps += [("analysis (dev)", py("analyze_rag2", "--split", "dev", "--out-dir", results,
                                    "--label-audit", results / "label_audit_dev.jsonl")),
              ("publish outputs (dev)", lambda: publish(data, results, "dev")),
              ("dev report", lambda: write_dev_report(results)),
              ("design record", lambda: write_design(results, a.model_path)),
              ("environment record", lambda: write_environment(results, "dev", a.model_path, a.judge_path))]
    if a.commit:
        steps.append(("commit", lambda: commit_results("Realigned study: dev run, dev report, design record")))
    return steps


def confirm_plan(a, data: Path, results: Path) -> list:
    arms = ("R2", "R2C", "R2V") + (() if a.no_temporal_ablation else ("R2V-ND",))
    steps = [("preflight (confirm)", lambda: print_checks(preflight(data, "confirm")))]
    steps += run_steps(a, "confirm", arms, ("R2",))
    steps += [("analysis (confirm)", py("analyze_rag2", "--split", "confirm", "--out-dir", results,
                                        "--label-audit", results / "label_audit_confirm.jsonl")),
              ("publish outputs (confirm)", lambda: publish(data, results, "confirm")),
              ("findings", lambda: write_findings(results)),
              ("environment record", lambda: write_environment(results, "confirm", a.model_path, a.judge_path))]
    if a.commit:
        steps.append(("commit", lambda: commit_results("Realigned study: confirmatory run and findings")))
    return steps


def ad_plan(a, data: Path, results: Path) -> list:
    count = lambda: print_checks([preflight(data, "ad")[0]])
    steps = [("Alzheimer's items present (run ad_benchmark first)", count),
             ("as-of PubMed records (ad)", py("pubmed_asof", "--split", "ad")),
             ("candidate pools and abstracts (ad)", py("freeze_candidates", "--split", "ad", "--device", "cpu")),
             ("B0 and B1 answers (ad)", py("generate_answers", "--split", "ad", "--arms", "B0", "B1",
                                           "--model-path", a.model_path, "--n-threads", a.n_threads)),
             ("preflight (ad)", lambda: print_checks(preflight(data, "ad")))]
    steps += run_steps(a, "ad", AD_ARMS, ("R2",))
    steps += [("analysis (ad)", py("analyze_rag2", "--split", "ad", "--out-dir", results)),
              ("publish outputs (ad)", lambda: publish(data, results, "ad")),
              ("findings (ad)", lambda: write_findings(results, "ad")),
              ("environment record", lambda: write_environment(results, "ad", a.model_path, a.judge_path))]
    if a.commit:
        steps.append(("commit", lambda: commit_results("Realigned study: Alzheimer's/dementia test set")))
    return steps


def fresh_plan(a, data: Path, results: Path) -> list:
    lean = argparse.Namespace(**{**vars(a), "judge_path": None})              # no directness judge in the fresh test
    steps = [("fresh items present (run fresh_benchmark first)", lambda: print_checks([preflight(data, "fresh")[0]])),
             ("as-of PubMed records (fresh)", py("pubmed_asof", "--split", "fresh")),
             ("abstracts of the as-of records (fresh)", py("freeze_candidates", "--split", "fresh", "--abstracts-only")),
             ("preflight (fresh)", lambda: print_checks(preflight(data, "fresh")))]
    steps += run_steps(lean, "fresh", FRESH_ARMS, ("R2",))
    steps += [("blinded analysis (fresh)", py("analyze_fresh", "--data-dir", data, "--results-dir", results)),
              ("publish outputs (fresh)", lambda: publish(data, results, "fresh")),
              ("environment record", lambda: write_environment(results, "fresh", a.model_path, None))]
    if a.commit:
        steps.append(("commit", lambda: commit_results("Pre-registered test on fresh questions: answers and blinded analysis")))
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
                      "ad_findings": (results / "RAG2_FINDINGS_AD.md").is_file(),
                      "fresh_answers": len(load_jsonl(data / "rag2_answers_fresh.jsonl")),
                      "fresh_findings": (results / "RAG2_FINDINGS_FRESH.md").is_file()}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("phase", choices=("dev", "confirm", "ad", "fresh", "status"))
    ap.add_argument("--model-path", help="the generator GGUF (Meta-Llama-3-8B-Instruct Q4_K_M)")
    ap.add_argument("--judge-path", default=None, help="optional second-family GGUF for the directness judge")
    ap.add_argument("--n-threads", default="6")
    ap.add_argument("--ablations", action="store_true", help="dev only: also R2-RQ, R2-BR and R2-NF")
    ap.add_argument("--no-temporal-ablation", action="store_true", help="confirm only: leave R2V-ND out")
    ap.add_argument("--go", action="store_true", help="confirm only: your explicit go after reading the dev report")
    ap.add_argument("--commit", action="store_true", help="commit the results after the phase (it never pushes: you push by hand)")
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
    if a.phase == "fresh" and not a.dry_run:
        problems = freeze_problems(data)
        if problems:
            print("refusing to start: " + "; ".join(problems), file=sys.stderr)
            return 2
    plan = {"confirm": confirm_plan, "ad": ad_plan, "fresh": fresh_plan}[a.phase]
    return execute(plan(a, data, results), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
