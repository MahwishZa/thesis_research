"""Audit of the label generator's answer extraction.

``correct`` is False whenever no option letter can be extracted, and
``extract_answer_letter`` takes the LAST standalone A/B/C/D token - including
the article "A" ("A 45-year-old ...") - so a rationale cut off by
``max_new_tokens`` before its final "Answer: X" line is scored wrong or, worse,
gets a stray letter. The label-noise control showed irrelevant passages flip
answers as often as retrieved ones; if long with-evidence rationales are
truncated more often, that alone would produce it.

This re-generates a small seeded sample of the finished run's questions, with
and without the retrieved passage and with the same generator settings, and
records for each generation: stopped by length?, explicit "Answer: X" line?,
which letter was scored. Writes raw text to a JSONL for inspection; writes no
labels.

    python -m experiments.baseline.filter_training.rationale_audit \\
        --progress experiments/baseline/filter_training/labels/medqa_filter_labels.progress_completed \\
        --output experiments/baseline/filter_training/labels/rationale_audit.jsonl --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

_EXPLICIT = re.compile(r"answer\s*:\s*\(?\s*([ABCD])\b", re.IGNORECASE)


def classify_generation(text: str, finish_reason: Optional[str],
                        scored_letter: Optional[str]) -> dict:
    explicit = _EXPLICIT.findall(text)
    return {
        "truncated": finish_reason == "length",
        "has_explicit_answer": bool(explicit),
        "explicit_letter": explicit[-1].upper() if explicit else None,
        "scored_letter": scored_letter,
        # the scored letter came from the fallback (no 'Answer: X' line) or
        # disagrees with the explicit line
        "fallback_or_mismatch": (not explicit)
        or (scored_letter != explicit[-1].upper()),
    }


def summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("no rows")
    out = {}
    for cond in ("without", "with_evidence", "control"):
        if not rows or cond not in rows[0]:
            continue
        sub = [r[cond] for r in rows]
        n = len(sub)
        out[cond] = {
            "n": n,
            "truncated_rate": sum(g["truncated"] for g in sub) / n,
            "explicit_answer_rate": sum(g["has_explicit_answer"] for g in sub) / n,
            "fallback_or_mismatch_rate": sum(g["fallback_or_mismatch"] for g in sub) / n,
            "no_letter_rate": sum(g["scored_letter"] is None for g in sub) / n,
            # greedy decoding should reproduce the original run exactly
            "matches_original_rate": sum(
                g.get("matches_original") is not False for g in sub) / n,
        }
    return out


def strict_flip_summary(rows: Sequence[dict], cond: str = "with_evidence") -> dict:
    """Flips between ``without`` and ``cond`` using ONLY explicit
    "Answer: X" lines (a generation without one is 'unscorable'), and how many
    of the lenient-extraction flips involved a guessed (fallback) generation."""
    flips_lenient = [r for r in rows
                     if r["without"]["correct"] != r[cond]["correct"]]
    involve_guess = [r for r in flips_lenient
                     if r["without"]["fallback_or_mismatch"]
                     or r[cond]["fallback_or_mismatch"]]
    scorable = [r for r in rows
                if r["without"]["has_explicit_answer"]
                and r[cond]["has_explicit_answer"]]
    to_ok = to_bad = 0
    for r in scorable:
        w = r["without"]["explicit_letter"] == r["gold_letter"]
        c = r[cond]["explicit_letter"] == r["gold_letter"]
        to_ok += (not w) and c
        to_bad += w and (not c)
    return {
        "condition": cond,
        "pairs": len(rows),
        "lenient_flips": len(flips_lenient),
        "lenient_flips_involving_a_guessed_generation": len(involve_guess),
        "strict_scorable_pairs": len(scorable),
        "strict_flip_to_correct": to_ok,
        "strict_flip_to_wrong": to_bad,
        "strict_flip_rate": (to_ok + to_bad) / len(scorable) if scorable else None,
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    break
                raise
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gguf-model-path", default=None)
    ap.add_argument("--tokenizer-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze-only", action="store_true",
                    help="no model: just summarize an existing --output JSONL")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the run's value (a deliberate, recorded test "
                         "of whether truncation causes the noise)")
    ap.add_argument("--with-control", action="store_true",
                    help="also generate with an irrelevant (derangement) passage")
    args = ap.parse_args(argv)

    if args.analyze_only:
        rows = _load_jsonl(Path(args.output))
        print(json.dumps(summarize(rows), indent=2))
        for cond in ("with_evidence", "control"):
            if rows and cond in rows[0]:
                print(json.dumps(strict_flip_summary(rows, cond), indent=2))
        return 0

    progress = Path(args.progress)
    state = json.loads((progress / "state.json").read_text(encoding="utf-8"))
    outcomes = _load_jsonl(progress / "outcomes.jsonl")
    n = min(args.n, len(outcomes))
    out = Path(args.output)
    if out.exists() and not args.resume:
        print(f"{out} exists; pass --resume to continue it", file=sys.stderr)
        return 2
    done = {r["pair_id"] for r in _load_jsonl(out)}
    model_path = args.gguf_model_path or state.get("model_path")
    if not model_path:
        print("no model path in state.json; pass --gguf-model-path", file=sys.stderr)
        return 2

    from .medqa_data import extract_answer_letter, load_medqa
    from .rationale_gguf import GGUFRationaleScorer
    items = {q.item_id: q for q in load_medqa(n=state["n_questions"], seed=state["seed"])}
    scorer = GGUFRationaleScorer(
        model_path, tokenizer_model_id=args.tokenizer_model,
        tokenizer_revision=state["model_revision"],
        max_new_tokens=args.max_new_tokens or state["max_new_tokens"])

    # record the raw completion of every score() call without changing it
    captured: dict = {}
    original = scorer._llm.create_completion

    def recording(*a, **k):
        res = original(*a, **k)
        captured["choice"] = res["choices"][0]
        return res
    scorer._llm.create_completion = recording

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(len(outcomes)), n))
    from .label_noise_control import derangement
    perm = derangement(n, args.seed) if args.with_control else None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as handle:
        for k, idx in enumerate(chosen):
            o = outcomes[idx]
            if o["pair_id"] in done:
                continue
            item = items[o["pair_id"]]
            choices = [item.options[x] for x in ("A", "B", "C", "D")]
            ans = item.options[item.answer_letter]
            row = {"pair_id": o["pair_id"], "gold_letter": item.answer_letter,
                   "max_new_tokens": scorer.max_new_tokens}
            conds = [("without", None), ("with_evidence", o["evidence"])]
            if args.with_control:
                conds.append(("control", outcomes[chosen[perm[k]]]["evidence"]))
            for cond, ev in conds:
                res = scorer.score(item.question, choices, ans, evidence=ev)
                ch = captured["choice"]
                text = ch["text"]
                g = classify_generation(text, ch.get("finish_reason"),
                                        extract_answer_letter(text))
                g.update(correct=res.correct, text=text,
                         matches_original=(res.correct == o[cond]['correct']) if cond in o else None)
                row[cond] = g
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"  {k + 1}/{n} done", flush=True)

    rows = _load_jsonl(out)
    print(json.dumps(summarize(rows), indent=2))
    for cond in ("with_evidence", "control"):
        if rows and cond in rows[0]:
            print(json.dumps(strict_flip_summary(rows, cond), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
