"""Automatic consistency check of the stated verdict (replaces the human G1 sheet).

The primary outcome is the parsed ``VERDICT:`` line, so this is a measurement-validity diagnostic, not
a gate on results. (1) Deterministic: every answer must open with exactly one parsable VERDICT line
(parse rate >= 95%, the only pass/fail criterion). (2) Diagnostic: a DIFFERENT model family reads only
the explanation of a seeded, arm-balanced sample of answers (the verdict line removed) and says which
verdict it supports; we report how often that equals the stated verdict, per arm. A 7B judge makes its
own mistakes, so the agreement is reported, not thresholded.

    python -m experiments.medchange.consistency_auto --split dev --model-path models\\Qwen2.5-7B-Instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

from .generate_answers import check_config, config_path, file_sha256, llama_generator, llama_version, load_jsonl
from .label_audit import LABELS, parse_label

HERE = Path(__file__).resolve().parent
MIN_PARSE = 0.95
SYSTEM = "You are a careful reader. You judge what a text says and do not use outside knowledge."
TEMPLATE = (
    "Below is an explanation that an assistant wrote when answering a medical question. Decide which verdict "
    "the explanation itself supports.\n"
    "SUPPORTED = the explanation says the hypothesis in the question is supported (the intervention or claim "
    "works or is true).\nREFUTED = it says the hypothesis is not supported (no benefit, harmful or false).\n"
    "NOT ENOUGH INFORMATION = it says the evidence is too limited to decide.\n\n"
    "Answer with exactly one line: LABEL: <label>\n\nQUESTION: {question}\nEXPLANATION: {explanation}")
_VERDICT_LINE = re.compile(r"^\s*VERDICT\s*:[^\n]*\n?", re.IGNORECASE)


def explanation(text: str) -> str:
    return _VERDICT_LINE.sub("", text or "", count=1).strip()


def sample_answers(answers: Sequence[dict], n: int, seed: int) -> list[dict]:
    """About ``n`` answers, equal numbers per arm, seeded."""
    arms = sorted({a["arm"] for a in answers})
    per = max(1, n // max(len(arms), 1))
    out = []
    for arm in arms:
        pool = sorted((a for a in answers if a["arm"] == arm and a.get("verdict")), key=lambda r: r["item_id"])
        random.Random(f"{seed}-{arm}").shuffle(pool)
        out += pool[:per]
    return out


def run(picked: Sequence[dict], questions: dict, generate: Callable[[str, str], str], out: Path) -> int:
    done = {(r["item_id"], r["arm"]) for r in load_jsonl(out)}
    todo = [a for a in picked if (a["item_id"], a["arm"]) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as handle:
        for a in todo:
            prompt = TEMPLATE.format(question=questions[a["item_id"]], explanation=explanation(a["text"]))
            judged = parse_label(generate(SYSTEM, prompt))
            handle.write(json.dumps({"item_id": a["item_id"], "arm": a["arm"], "stated": a["verdict"],
                                     "judged": judged}) + "\n")
            handle.flush()
    return len(todo)


def summarize(rows: Sequence[dict], answers: Sequence[dict]) -> dict:
    parse_rate = sum(a["verdict"] is not None for a in answers) / len(answers) if answers else 0.0
    valid = [r for r in rows if r["judged"]]
    by_arm = {}
    for r in valid:
        by_arm.setdefault(r["arm"], []).append(r["stated"] == r["judged"])
    return {"parse_rate": round(parse_rate, 4), "parse_gate": "PASS" if parse_rate >= MIN_PARSE else "FAIL",
            "n_judged": len(valid), "n_unparsed_by_judge": len(rows) - len(valid),
            "agreement": round(sum(r["stated"] == r["judged"] for r in valid) / len(valid), 4) if valid else None,
            "agreement_by_arm": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_arm.items())},
            "disagreements": {f"{s}->{j}": sum(1 for r in valid if r["stated"] == s and r["judged"] == j and s != j)
                              for s in LABELS for j in LABELS if s != j}}


def to_markdown(split: str, rep: dict, judge: str) -> str:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = [f"# Automatic consistency check, {split} split", "",
             f"* Parse rate of all answers: **{pct(rep['parse_rate'])}** ({rep['parse_gate']}; needs >= 95%).",
             f"* Independent judge ({judge}) agrees with the stated verdict on **{pct(rep['agreement'])}** of "
             f"{rep['n_judged']} sampled answers (diagnostic, not a gate).", "", "| Arm | agreement |", "|---|---|"]
    lines += [f"| {k} | {pct(v)} |" for k, v in rep["agreement_by_arm"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--answers", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=None)
    args = ap.parse_args(argv)
    d = Path(args.data_dir)
    answers = load_jsonl(Path(args.answers) if args.answers else d / f"answers_{args.split}.jsonl")
    if not answers:
        print("no answers yet", file=sys.stderr)
        return 2
    if not Path(args.model_path).is_file():
        print(f"model file not found: {args.model_path}", file=sys.stderr)
        return 2
    questions = {r["item_id"]: r["question"] for r in load_jsonl(d / "benchmark.jsonl")}
    out = d / f"consistency_auto_{args.split}.jsonl"
    sha = file_sha256(args.model_path)
    differs = check_config(out, {"judge_sha256": sha, "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                                 "n": args.n, "seed": args.seed, "llama_cpp_python": llama_version()},
                           ("judge_sha256", "template_sha256", "n", "seed"))
    if differs:
        print(f"refusing to extend {out.name}: configuration differs in {', '.join(differs)} "
              f"(see {config_path(out).name})", file=sys.stderr)
        return 2
    picked = sample_answers(answers, args.n, args.seed)
    gen = llama_generator(args.model_path, args.n_ctx, args.n_threads, 0, 12)
    run(picked, questions, gen, out)
    keys = {(a["item_id"], a["arm"]) for a in picked}
    rep = summarize([r for r in load_jsonl(out) if (r["item_id"], r["arm"]) in keys], answers)
    rep.update(split=args.split, judge=Path(args.model_path).name, judge_sha256=sha)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"consistency_auto_{args.split}.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = to_markdown(args.split, rep, Path(args.model_path).name)
    (od / f"consistency_auto_{args.split}.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
