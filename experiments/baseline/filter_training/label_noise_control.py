"""Control experiment: do the RAG² correctness flips measure *evidence content*?

For a subset of the already-labelled questions, re-score the with-evidence
generation using an IRRELEVANT passage (another question's retrieved passage,
assigned by a fixed derangement so no question keeps its own) and compare:

    retrieved passage vs irrelevant passage, against the same no-evidence run.

If an irrelevant passage flips the labeller's answer about as often as the
retrieved one, the flips are prompt sensitivity (greedy decoding on a 4-bit
model changes its chain of thought whenever the prompt changes), not
helpfulness - and the label function cannot teach a filter what "helpful"
means at this scale. This is a validity check of the labelling step, not a
change to the recipe; it writes no labels.

Reads the finished run's settings (model path/revision, max_new_tokens, seed)
from its ``state.json`` so the control uses exactly the same generator.
Checkpointed (JSONL) and resumable like build_labels.

    python -m experiments.baseline.filter_training.label_noise_control \\
        --progress experiments/baseline/filter_training/labels/medqa_filter_labels.progress_completed \\
        --output experiments/baseline/filter_training/labels/noise_control.jsonl \\
        --n 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence


def derangement(n: int, seed: int) -> list[int]:
    """A permutation of range(n) with no fixed point (n >= 2)."""
    if n < 2:
        raise ValueError("need at least 2 items for a derangement")
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(i != p for i, p in enumerate(perm)):
            return perm


def summarize(rows: Sequence[dict]) -> dict:
    """``rows``: dicts with bool keys without_correct, retrieved_correct,
    control_correct."""
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to summarize")

    def flips(key):
        to_ok = sum(r[key] and not r["without_correct"] for r in rows)
        to_bad = sum(r["without_correct"] and not r[key] for r in rows)
        return to_ok, to_bad

    r_ok, r_bad = flips("retrieved_correct")
    c_ok, c_bad = flips("control_correct")
    return {
        "n": n,
        "accuracy_without": sum(r["without_correct"] for r in rows) / n,
        "accuracy_retrieved": sum(r["retrieved_correct"] for r in rows) / n,
        "accuracy_control": sum(r["control_correct"] for r in rows) / n,
        "retrieved_flip_to_correct": r_ok, "retrieved_flip_to_wrong": r_bad,
        "control_flip_to_correct": c_ok, "control_flip_to_wrong": c_bad,
        "retrieved_flip_rate": (r_ok + r_bad) / n,
        "control_flip_rate": (c_ok + c_bad) / n,
    }


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:   # truncated last line from a kill
                    break
                raise
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--output", required=True, help="JSONL checkpoint")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--gguf-model-path", default=None,
                    help="default: the path recorded in the run's state.json")
    ap.add_argument("--tokenizer-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    progress = Path(args.progress)
    state = json.loads((progress / "state.json").read_text(encoding="utf-8"))
    outcomes_raw = _load_jsonl(progress / "outcomes.jsonl")
    if len(outcomes_raw) < 2:
        print("need at least 2 finished outcomes", file=sys.stderr)
        return 2
    n = min(args.n, len(outcomes_raw))

    out = Path(args.output)
    if out.exists() and not args.resume:
        print(f"{out} exists; pass --resume to continue it", file=sys.stderr)
        return 2
    done = {r["pair_id"] for r in _load_jsonl(out)}

    model_path = args.gguf_model_path or state.get("model_path")
    if not model_path:
        print("no model path in state.json; pass --gguf-model-path", file=sys.stderr)
        return 2

    from .medqa_data import load_medqa
    from .rationale_gguf import GGUFRationaleScorer
    items = {q.item_id: q for q in load_medqa(n=state["n_questions"], seed=state["seed"])}

    # subset + passage assignment are a pure function of (seed, n): resumable
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(len(outcomes_raw)), n))
    perm = derangement(n, args.seed)
    scorer = GGUFRationaleScorer(
        model_path, tokenizer_model_id=args.tokenizer_model,
        tokenizer_revision=state["model_revision"],
        max_new_tokens=state["max_new_tokens"])

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as handle:
        for k, idx in enumerate(chosen):
            o = outcomes_raw[idx]
            if o["pair_id"] in done:
                continue
            item = items[o["pair_id"]]
            donor = outcomes_raw[chosen[perm[k]]]
            choices = [item.options[x] for x in ("A", "B", "C", "D")]
            res = scorer.score(item.question, choices,
                               item.options[item.answer_letter],
                               evidence=donor["evidence"])
            handle.write(json.dumps({
                "pair_id": o["pair_id"], "donor_pair_id": donor["pair_id"],
                "without_correct": o["without"]["correct"],
                "retrieved_correct": o["with_evidence"]["correct"],
                "control_correct": res.correct,
            }) + "\n")
            handle.flush()
            print(f"  {k + 1}/{n} done", flush=True)

    s = summarize(_load_jsonl(out))
    print(json.dumps(s, indent=2))
    print("If control_flip_rate is close to retrieved_flip_rate, the flips "
          "mostly reflect prompt sensitivity, not evidence helpfulness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
