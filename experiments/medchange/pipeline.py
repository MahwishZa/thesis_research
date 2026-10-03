"""One-command, resumable stage-2 pipeline with automatic gates (no human labelling).

    python -m experiments.medchange.pipeline dev --model-path models\\<llama>.gguf ^
        --judge-path models\\<qwen>.gguf --medchange-dir ..\\MedChange --commit
    python -m experiments.medchange.pipeline confirm --go --model-path models\\<llama>.gguf ^
        --judge-path models\\<qwen>.gguf --medchange-dir ..\\MedChange --commit
    python -m experiments.medchange.pipeline status

``dev`` checks gate 1 on the pilot, runs the dev audits (label audit, consistency check), the stance step
with both wordings, the fit and gate 2, writes ``results/DEV_REPORT.md`` and stops: the only decision
left is yours (``confirm --go``). ``confirm`` refuses to start unless the frozen model file is pushed to
``origin/main``, checks that the generator is the same file as on dev, builds the confirmatory pools,
generates B0/B1, runs the stance step and the frozen-model prediction only if gate 2 passed (otherwise
RQ1 only), runs the after-freeze audits, analyses and writes ``results/FINDINGS.md``. Every step is
resumable: rerun the same command after an interruption. ``--dry-run`` prints the plan.
Without ``--judge-path`` the two audit steps are skipped (the gates do not depend on them).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import findings, stance_check
from .generate_answers import MAX_NEW_TOKENS, RESULT_RELEVANT, file_sha256, load_jsonl, run_config
from .stance import TOP_K

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_ITEMS = {"dev": 226, "confirm": 528}
FROZEN = "/".join(("experiments", "medchange", "results", "synthesis_model.json"))   # written by synthesis fit
SHARE = ("stance_pilot.jsonl", "stance_pilot.config.json", "stance_dev.jsonl", "stance_dev.config.json",
         "synthesis_dev.jsonl", "stance_confirm.jsonl", "stance_confirm.config.json", "synthesis_confirm.jsonl",
         "synthesis_confirm.config.json", "answers_confirm.jsonl", "answers_confirm.config.json",
         "label_audit_dev.jsonl", "label_audit_confirm.jsonl", "consistency_auto_dev.jsonl",
         "consistency_auto_confirm.jsonl")


def py(module: str, *args) -> list[str]:
    return [sys.executable, "-m", f"experiments.medchange.{module}", *map(str, args)]


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def preflight(data: Path, split: str) -> list[tuple[str, bool, str]]:
    """Integrity checks on the benchmark, the frozen pools and (when present) the answers."""
    out: list[tuple[str, bool, str]] = []
    bench = [r for r in load_jsonl(data / "benchmark.jsonl") if not r["likely_label_noise"]]
    by_split = {s: {r["item_id"] for r in bench if r["split"] == s} for s in ("dev", "confirm")}
    out.append(("dev and confirm share no item", not (by_split["dev"] & by_split["confirm"]), ""))
    n = len(by_split[split])
    out.append((f"{split} has {EXPECTED_ITEMS[split]} usable items", n == EXPECTED_ITEMS[split], f"found {n}"))
    items = {r["item_id"]: r for r in bench if r["split"] == split}
    pools = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{split}.jsonl")}
    missing = [i for i in items if i not in pools]
    out.append(("every item has a frozen pool", not missing, f"{len(missing)} missing"))
    short = [i for i in items if i in pools and len(pools[i]["candidates"]) < TOP_K]
    out.append((f"every pool has at least {TOP_K} candidates", not short, f"{len(short)} short"))
    leaks = [i for i, it in items.items() if i in pools and
             {c["pmid"] for c in pools[i]["candidates"]} & {it["newest"].get("pmid"), it["previous"].get("pmid")} - {None}]
    out.append(("no pool contains its own review", not leaks, f"{len(leaks)} leaks"))
    answers = load_jsonl(data / f"answers_{split}.jsonl")
    if answers:
        rate = sum(a["verdict"] is not None for a in answers) / len(answers)
        out.append(("answers parse rate >= 95%", rate >= 0.95, f"{100 * rate:.1f}%"))
    return out


def gate1_from_pilot(data: Path) -> Optional[dict]:
    records = load_jsonl(data / "stance_pilot.jsonl")
    return stance_check.gate1(stance_check.pilot_report(records)) if records else None


def frozen_is_pushed(repo: Path = ROOT, rel: str = FROZEN) -> tuple[bool, str]:
    """True when the working copy of the frozen model equals the one on origin/main (last known state)."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
    local = git("hash-object", rel)
    remote = git("rev-parse", f"origin/main:{rel}")
    if local.returncode != 0:
        return False, f"{rel} does not exist"
    if remote.returncode != 0:
        return False, f"{rel} is not on origin/main: commit and push it first (dev --commit does)"
    if local.stdout.strip() != remote.stdout.strip():
        return False, f"{rel} differs from origin/main: commit and push the final version first"
    return True, "frozen model is on origin/main"


def config_differences(expected: dict, recorded: dict, fields: Sequence[str] = RESULT_RELEVANT) -> list[str]:
    return [k for k in fields if recorded.get(k) != expected.get(k)]


def generation_precheck(model_path: str, dev_config: Path, n_ctx: int = 4096) -> list[str]:
    """Fields on which a confirmatory generation would differ from the recorded dev generation."""
    if not dev_config.is_file():
        return ["no recorded dev generator configuration"]
    expected = run_config(model_path=model_path, model_sha256=file_sha256(model_path), n_ctx=n_ctx,
                          max_new_tokens=MAX_NEW_TOKENS, n_threads=None, n_gpu_layers=0, llama_version="")
    return config_differences(expected, json.loads(dev_config.read_text(encoding="utf-8")))


def stance_model_precheck(model_path: str, dev_stance_config: Path) -> list[str]:
    if not dev_stance_config.is_file():
        return ["no recorded dev stance configuration"]
    recorded = json.loads(dev_stance_config.read_text(encoding="utf-8"))
    return [] if recorded.get("model_sha256") == file_sha256(model_path) else ["model_sha256"]


# --------------------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------------------

Step = tuple[str, object]          # (name, argv list | callable returning (ok, message))


def publish(data: Path, results: Path) -> tuple[bool, str]:
    """Copy the no-source-text outputs that are worth keeping into the tracked results folder."""
    copied = []
    for name in SHARE:
        if (data / name).is_file():
            shutil.copyfile(data / name, results / name)
            copied.append(name)
    return True, f"copied {len(copied)} files to results/"


def commit_and_push(message: str, repo: Path = ROOT) -> tuple[bool, str]:
    subprocess.run(["git", "add", "experiments/medchange/results"], cwd=repo)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0:
        return True, "nothing new to commit"
    for cmd in (["git", "commit", "-q", "-m", message], ["git", "push", "origin", "main"]):
        if subprocess.run(cmd, cwd=repo).returncode != 0:
            return False, f"{' '.join(cmd[:2])} failed; fix it and rerun the same command"
    return True, "committed and pushed to main"


def dev_plan(a, data: Path, results: Path) -> list[Step]:
    steps: list[Step] = [("preflight (dev)", lambda: _checks(preflight(data, "dev")))]
    if a.judge_path and a.medchange_dir:
        steps += [("label audit (dev)", py("label_audit", "--split", "dev", "--medchange-dir", a.medchange_dir,
                                            "--model-path", a.judge_path)),
                  ("consistency check (dev)", py("consistency_auto", "--split", "dev", "--model-path", a.judge_path))]
    steps += [("stance, both wordings (dev)", py("stance", "--split", "dev", "--wording", "both",
                                                 "--model-path", a.model_path, "--n-threads", a.n_threads)),
              ("fit, cross-validation, gate 2, freeze", py("synthesis", "fit")),
              ("publish outputs", lambda: publish(data, results)),
              ("dev report", lambda: (True, f"gate 2: {findings.write_dev_report(results, gate1_from_pilot(data))}"))]
    if a.commit:
        steps.append(("commit and push", lambda: commit_and_push("Dev run: stance, frozen model, audits, dev report")))
    return steps


def confirm_plan(a, data: Path, results: Path, gate2: str) -> list[Step]:
    steps: list[Step] = [
        ("confirmatory candidate probe", py("pubmed_asof", "--split", "confirm")),
        ("confirmatory frozen pools", py("freeze_candidates", "--split", "confirm", "--device", "cpu")),
        ("preflight (confirm)", lambda: _checks(preflight(data, "confirm"))),
        ("B0 and B1 answers (confirm)", py("generate_answers", "--split", "confirm", "--arms", "B0", "B1",
                                           "--model-path", a.model_path, "--n-threads", a.n_threads)),
        ("preflight (confirm, with answers)", lambda: _checks(preflight(data, "confirm")))]
    if gate2 == "PASS":
        steps += [("stance, both wordings (confirm)", py("stance", "--split", "confirm", "--wording", "both",
                                                         "--model-path", a.model_path, "--n-threads", a.n_threads)),
                  ("frozen-model prediction", py("synthesis", "predict", "--split", "confirm"))]
    if a.judge_path and a.medchange_dir:
        steps += [("label audit (confirm, after freeze)", py("label_audit", "--split", "confirm", "--medchange-dir",
                                                              a.medchange_dir, "--model-path", a.judge_path)),
                  ("consistency check (confirm)", py("consistency_auto", "--split", "confirm", "--model-path",
                                                      a.judge_path))]
    analysis = ["analyze_stage2", "--split", "confirm", "--out-dir", results,
                "--label-audit", results / "label_audit_confirm.json"]
    steps += [("analysis", py(*analysis, *([] if gate2 == "PASS" else ["--rq1-only"]))),
              ("publish outputs", lambda: publish(data, results)),
              ("findings", lambda: (findings.write_findings(results) or True, "wrote FINDINGS.md"))]
    if a.commit:
        steps.append(("commit and push", lambda: commit_and_push("Confirmatory run: answers, stance, analysis, findings")))
    return steps


def _checks(rows: Sequence[tuple[str, bool, str]]) -> tuple[bool, str]:
    for name, ok, detail in rows:
        print(f"    {'ok  ' if ok else 'FAIL'} {name} {detail}")
    return all(ok for _, ok, _ in rows), "integrity checks"


def execute(steps: Sequence[Step], dry_run: bool) -> int:
    for k, (name, action) in enumerate(steps, 1):
        print(f"[{k}/{len(steps)}] {name}", flush=True)
        if dry_run:
            if isinstance(action, list):
                print("    " + " ".join(map(str, action)))
            continue
        if callable(action):
            ok, message = action()
            print(f"    {message}")
        else:
            ok = subprocess.run([str(x) for x in action], cwd=ROOT).returncode == 0
        if not ok:
            print(f"STOPPED at step {k} ({name}). Nothing later was run; fix the cause and rerun the same command.",
                  file=sys.stderr)
            return 2
    return 0


def status(data: Path, results: Path) -> int:
    g1 = gate1_from_pilot(data)
    model = results / "synthesis_model.json"
    gate2 = json.loads(model.read_text(encoding="utf-8"))["gate2"]["gate2"] if model.is_file() else None
    pushed = frozen_is_pushed()[0] if model.is_file() else False
    print(json.dumps({"gate1": g1["gate1"] if g1 else None, "gate2": gate2, "frozen_model_pushed": pushed,
                      "dev_report": (results / "DEV_REPORT.md").is_file(),
                      "confirmatory_answers": (data / "answers_confirm.jsonl").is_file(),
                      "findings": (results / "FINDINGS.md").is_file()}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("phase", choices=("dev", "confirm", "status"))
    ap.add_argument("--model-path", help="the generator GGUF (Llama-3-8B-Instruct Q4_K_M)")
    ap.add_argument("--judge-path", default=None, help="a second-family GGUF (e.g. Qwen2.5-7B-Instruct) for the audits")
    ap.add_argument("--medchange-dir", default=None, help="clone of jvladika/MedChange (label audit)")
    ap.add_argument("--n-threads", default="6")
    ap.add_argument("--go", action="store_true", help="confirm only: your explicit go after reading DEV_REPORT.md")
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
    if a.phase == "dev":
        g1 = gate1_from_pilot(data)
        if not a.dry_run and (not g1 or g1["gate1"] != "PASS"):
            print(f"gate 1 (stance pilot, machine checks) is not PASS: {g1}", file=sys.stderr)
            return 2
        return execute(dev_plan(a, data, results), a.dry_run)
    # confirm
    if not a.go:
        print("the confirmatory run needs your explicit go: read results/DEV_REPORT.md, then add --go",
              file=sys.stderr)
        return 2
    model = results / "synthesis_model.json"
    if a.dry_run and not model.is_file():
        gate2 = "PASS"
    else:
        ok, why = frozen_is_pushed()
        if not ok:
            print(f"refusing to start: {why}", file=sys.stderr)
            return 2
        gate2 = json.loads(model.read_text(encoding="utf-8"))["gate2"]["gate2"]
        bad = generation_precheck(a.model_path, results / "answers_dev.config.json")
        if gate2 == "PASS":
            bad += stance_model_precheck(a.model_path, results / "stance_dev.config.json")
        if bad:
            print("refusing to start: the confirmatory generator would differ from the dev run in: "
                  + ", ".join(bad), file=sys.stderr)
            return 2
    return execute(confirm_plan(a, data, results, gate2), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
