"""CLI: generate RAG² filter training labels from MedQA + a textbook corpus.

Retrieves one passage per MedQA question from a general-medical textbook
corpus (never the thesis's Alzheimer's corpus - specification SS10.6),
scores each (question, passage) pair with a ``RationaleScorer`` twice
(without and with the passage), applies the paper's label decision tree
(``labeling.py``), and writes the training file
``experiments/baseline/filter_training/train.py`` reads.

Two scorer backends (``--scorer``), same decision tree, same downstream
file format either way:

  ``hf`` (default): ``rationale.Llama3RationaleScorer`` - Llama-3-8B-
  Instruct via HF transformers + bitsandbytes. Needs ~5.5-6 GB VRAM
  (4-bit) or ~16 GB RAM (CPU, full precision) - a free-tier GPU (Colab/
  Kaggle) or a workstation with a real GPU, not most laptops.

  ``gguf``: ``rationale_gguf.GGUFRationaleScorer`` - the same model,
  4-bit GGUF-quantised via llama.cpp, running on CPU RAM (~6-7 GB) with
  no VRAM requirement. See ``rationale_gguf.py``'s module docstring for
  exactly what is and is not identical to the ``hf`` path, and
  docs/reproducibility.md for local setup and how to obtain a GGUF file.

**Checkpointed.** A real-scale run is a long, unattended job either way;
progress is saved incrementally to ``<output>.progress/`` so an
interrupted run (killed process, closed laptop lid) resumes with
``--resume`` instead of restarting. A resume validates that the question
count, seed, textbook-index size, and scorer identity all still match the
checkpoint before continuing, refusing rather than silently mixing runs.

**Calibrate before committing.** ``--calibrate N`` scores only the first
``N`` pairs, reports measured throughput and an extrapolated estimate for
the full ``--n-questions`` run, and writes nothing to ``--output`` or the
checkpoint - repeatable, and never mistakeable for a real (partial) run.

    python -m experiments.baseline.filter_training.build_labels \\
        --scorer gguf --gguf-model-path /path/to/model.gguf \\
        --n-questions 500 \\
        --output experiments/baseline/filter_training/labels/medqa_filter_labels.json \\
        --model-revision <pinned-commit-sha> \\
        --calibrate 10

``--n-questions`` should come from a calibration measurement (a budget,
not a target) - this script does not choose it for you.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .labeling import (
    PairOutcome, label_dataset, label_distribution, outcome_from_dict,
    outcome_to_dict, write_training_file,
)
from .medqa_data import load_medqa, load_textbook_passages

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


class BuildLabelsError(RuntimeError):
    """Raised when label generation cannot proceed safely."""


def build_textbook_index(n_passages: int, seed: int, device: Optional[str]):
    from experiments.shared.retrieval.encoders import medcpt_article_encoder
    from experiments.shared.retrieval.index import build_index

    passages = load_textbook_passages(n=n_passages, seed=seed)
    print(f"loaded {len(passages)} textbook passages")
    encoder = medcpt_article_encoder(device=device)

    t0 = time.time()

    def _progress(done: int, total: int, elapsed: float) -> None:
        print(f"  ...encoding batch {done}/{total} ({elapsed:.0f}s elapsed)")

    vectors_index = build_index(
        passages, encoder,
        corpus_snapshot=f"medrag_textbooks@n={n_passages},seed={seed}",
        on_progress=_progress,
    )
    print(f"built textbook index: {len(vectors_index.passage_ids)} x "
          f"{vectors_index.dim} in {time.time() - t0:.0f}s")
    return vectors_index, passages


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _progress_paths(output: Path) -> tuple[Path, Path]:
    progress_dir = output.parent / f"{output.stem}.progress"
    return progress_dir / "outcomes.jsonl", progress_dir / "state.json"


def _fingerprint(*, scorer_name: str, n_questions: int, seed: int,
                  n_textbook_passages: int) -> dict[str, Any]:
    return {
        "scorer_name": scorer_name, "n_questions": n_questions,
        "seed": seed, "n_textbook_passages": n_textbook_passages,
    }


def load_checkpoint(
    output: Path, *, expected: dict[str, Any],
) -> tuple[list[PairOutcome], set[str]]:
    """Read a prior run's checkpoint, or start empty if there is none.

    Refuses (``BuildLabelsError``) if a checkpoint exists but its
    fingerprint does not match ``expected`` - a resume must not silently
    mix outcomes generated under different settings into one label set.
    """
    outcomes_path, state_path = _progress_paths(output)
    if not state_path.exists():
        return [], set()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mismatches = [
        f"{key}: checkpoint has {state.get(key)!r}, this run asked for {value!r}"
        for key, value in expected.items()
        if state.get(key) != value
    ]
    if mismatches:
        raise BuildLabelsError(
            "cannot resume: this run's settings do not match the "
            f"checkpoint at {state_path}: " + "; ".join(mismatches) +
            f". Delete {outcomes_path.parent} to start fresh if intentional."
        )
    outcomes: list[PairOutcome] = []
    if outcomes_path.exists():
        for line in outcomes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                outcomes.append(outcome_from_dict(json.loads(line)))
    return outcomes, {o.pair_id for o in outcomes}


def run_labeling(
    questions: Sequence[Any],
    index: Any,
    passages: Sequence[Any],
    query_encoder: Any,
    scorer: Any,
    *,
    output: Path,
    fingerprint: dict[str, Any],
    resume: bool = False,
    checkpoint_every: int = 20,
    calibrate_n: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> list[PairOutcome]:
    """Score every question against its retrieved passage, with or without
    checkpointing depending on ``calibrate_n``.

    ``calibrate_n`` set: scores only the first ``calibrate_n`` questions,
    writes no checkpoint at all (a calibration run must be freely
    repeatable and must never look like partial progress on the real
    run). ``calibrate_n`` unset: the real run - checkpoints every
    ``checkpoint_every`` pairs, and with ``resume=True`` continues from
    whatever a prior interrupted run already completed.
    """
    outcomes_path, state_path = _progress_paths(output)
    already_done: set[str] = set()
    outcomes: list[PairOutcome] = []

    if calibrate_n is None:
        if resume:
            outcomes, already_done = load_checkpoint(output, expected=fingerprint)
        elif state_path.exists():
            raise BuildLabelsError(
                f"a checkpoint already exists at {state_path} but --resume "
                "was not given. Pass --resume to continue it, or delete "
                f"{outcomes_path.parent} to start fresh."
            )
        else:
            outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(state_path, fingerprint)

    targets = questions if calibrate_n is None else questions[:calibrate_n]
    t0 = time.time()
    handle = None if calibrate_n is not None else open(outcomes_path, "a", encoding="utf-8")
    try:
        for i, item in enumerate(targets, 1):
            if item.item_id in already_done:
                continue
            query_vector = query_encoder.encode([item.rendered_question()])[0]
            hits = index.search(query_vector, top_k=1)
            if not hits:
                continue
            row, _score = hits[0]
            passage = passages[row]

            choices = [item.options[letter] for letter in ("A", "B", "C", "D")]
            answer_text = item.options[item.answer_letter]

            without = scorer.score(item.question, choices, answer_text, evidence=None)
            with_evidence = scorer.score(
                item.question, choices, answer_text, evidence=passage.text
            )

            outcome = PairOutcome(
                pair_id=item.item_id, question=item.rendered_question(),
                evidence=passage.text, without=without, with_evidence=with_evidence,
            )
            outcomes.append(outcome)

            if handle is not None:
                handle.write(json.dumps(outcome_to_dict(outcome)))
                handle.write("\n")
                if len(outcomes) % checkpoint_every == 0:
                    handle.flush()

            if on_progress is not None and (i % 10 == 0 or i == len(targets)):
                on_progress(i, len(targets), time.time() - t0)
    finally:
        if handle is not None:
            handle.close()

    return outcomes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-questions", type=int, required=True)
    ap.add_argument("--n-textbook-passages", type=int, default=50_000,
                    help="how many textbook passages to build the "
                         "retrieval index over")
    ap.add_argument("--output", required=True,
                    help="path to write the training-label JSON")
    ap.add_argument("--scorer", choices=("hf", "gguf"), default="hf",
                    help="hf: rationale.Llama3RationaleScorer (needs a "
                         "capable GPU or ~16GB RAM). gguf: "
                         "rationale_gguf.GGUFRationaleScorer (runs on "
                         "CPU RAM, ~6-7GB). See module docstring.")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--model-revision", required=True,
                    help="pinned commit sha, not a branch name")
    ap.add_argument("--quantization", default="auto",
                    choices=("auto", "nf4", "int8", "none"),
                    help="--scorer hf only. 'auto': nf4 if CUDA is "
                         "available, else none.")
    ap.add_argument("--device", default=None, help="--scorer hf only. e.g. cuda")
    ap.add_argument("--gguf-model-path", default=None,
                    help="--scorer gguf only, required. Path to the local "
                         ".gguf file - see docs/reproducibility.md.")
    ap.add_argument("--gguf-n-ctx", type=int, default=4096)
    ap.add_argument("--gguf-n-threads", type=int, default=None)
    ap.add_argument("--gguf-n-gpu-layers", type=int, default=0,
                    help="0 (default): CPU-only, no VRAM needed. Raise "
                         "only as an optional speed boost if you have "
                         "VRAM to spare.")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="rationale length cap. Lower it (e.g. 96-128) "
                         "for a faster run at some cost to rationale "
                         "quality.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run found next to "
                         "--output instead of refusing to proceed.")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="how often (in completed pairs) to flush the "
                         "checkpoint to disk.")
    ap.add_argument("--calibrate", type=int, default=None, metavar="N",
                    help="score only the first N questions, report "
                         "measured throughput and an extrapolated "
                         "estimate for the full --n-questions run, and "
                         "write nothing - no output file, no checkpoint. "
                         "Run this before a real-scale run.")
    args = ap.parse_args(argv)

    out_path = Path(args.output)
    if args.calibrate is None and out_path.exists():
        print(f"refusing to overwrite existing labels at {out_path}",
              file=sys.stderr)
        return 2
    if args.scorer == "gguf" and not args.gguf_model_path:
        print("--scorer gguf requires --gguf-model-path", file=sys.stderr)
        return 2

    if args.scorer == "hf":
        import torch
        cuda_available = torch.cuda.is_available()
        quantization = args.quantization
        if quantization == "auto":
            quantization = "nf4" if cuda_available else "none"
        quantization = None if quantization == "none" else quantization
        if not cuda_available:
            est_seconds_per_pair = 2 * args.max_new_tokens / 1.5
            est_total_hours = args.n_questions * est_seconds_per_pair / 3600
            print(
                f"no CUDA device available - running on CPU. At a rough "
                f"~1.5 tokens/sec, {args.n_questions} questions x 2 "
                f"generations x {args.max_new_tokens} tokens is "
                f"approximately {est_total_hours:.1f} hours. Consider "
                "--scorer gguf instead - see docs/reproducibility.md.",
                file=sys.stderr,
            )
        from .rationale import Llama3RationaleScorer
        scorer = Llama3RationaleScorer(
            args.model_name, args.model_revision, device=args.device,
            quantization=quantization, max_new_tokens=args.max_new_tokens,
        )
        scorer_device = args.device or ("cuda" if cuda_available else "cpu")
    else:
        from .rationale_gguf import GGUFRationaleScorer
        scorer = GGUFRationaleScorer(
            args.gguf_model_path,
            tokenizer_model_id=args.model_name,
            tokenizer_revision=args.model_revision,
            n_ctx=args.gguf_n_ctx, n_threads=args.gguf_n_threads,
            n_gpu_layers=args.gguf_n_gpu_layers,
            max_new_tokens=args.max_new_tokens,
        )
        scorer_device = "cpu" if args.gguf_n_gpu_layers == 0 else "cpu+gpu-partial"

    index, passages = build_textbook_index(
        args.n_textbook_passages, args.seed, args.device
    )
    from experiments.shared.retrieval.encoders import medcpt_query_encoder
    query_encoder = medcpt_query_encoder(device=args.device)

    questions = load_medqa(n=args.n_questions, seed=args.seed)
    print(f"loaded {len(questions)} MedQA questions")

    fingerprint = _fingerprint(
        scorer_name=scorer.name if hasattr(scorer, "name") else args.scorer,
        n_questions=args.n_questions, seed=args.seed,
        n_textbook_passages=args.n_textbook_passages,
    )

    def _report(i: int, total: int, elapsed: float) -> None:
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (total - i) / rate if rate > 0 else float("nan")
        print(f"  ...{i}/{total} pairs scored "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    if args.calibrate is not None:
        print(f"CALIBRATION RUN: scoring the first {args.calibrate} of "
              f"{len(questions)} questions (writes nothing) ...")
        t0 = time.time()
        outcomes = run_labeling(
            questions, index, passages, query_encoder, scorer,
            output=out_path, fingerprint=fingerprint,
            calibrate_n=args.calibrate, on_progress=_report,
        )
        elapsed = time.time() - t0
        n = len(outcomes)
        if n == 0:
            print("calibration scored 0 pairs - nothing to extrapolate from",
                  file=sys.stderr)
            return 1
        per_pair = elapsed / n
        full_estimate_hours = per_pair * args.n_questions / 3600
        print(f"\nCALIBRATION RESULT (device={scorer_device})")
        print(f"  {n} pairs scored in {elapsed:.1f}s "
              f"({per_pair:.2f}s/pair, 2 generations/pair)")
        print(f"  extrapolated estimate for the full --n-questions "
              f"{args.n_questions}: {full_estimate_hours:.1f} hours")
        print("  nothing was written - no output file, no checkpoint. "
              "Re-run --calibrate any time; it never conflicts with a "
              "real run.")
        return 0

    outcomes = run_labeling(
        questions, index, passages, query_encoder, scorer,
        output=out_path, fingerprint=fingerprint, resume=args.resume,
        checkpoint_every=args.checkpoint_every, on_progress=_report,
    )

    print(f"scored {len(outcomes)} pairs")
    if not outcomes:
        print("no pairs were scored - nothing to label", file=sys.stderr)
        return 1

    examples = label_dataset(outcomes, dataset_name="medqa")
    dist = label_distribution(examples)
    print("label distribution:")
    print(json.dumps(dist, indent=2, sort_keys=True))

    dominant_fraction = max(
        v for k, v in dist.items() if not k.startswith("rule:")
    ) / len(examples)
    if dominant_fraction > 0.95:
        print(
            f"WARNING: one label is {dominant_fraction:.0%} of the set - a "
            "filter trained on this will mostly learn the prior, not the "
            "task. Consider more/different questions before training on "
            "this file.",
            file=sys.stderr,
        )

    write_training_file(examples, out_path)
    print(f"wrote {len(examples)} labelled examples to {out_path}")

    outcomes_path, state_path = _progress_paths(out_path)
    if outcomes_path.parent.exists():
        import shutil
        completed_dir = outcomes_path.parent.parent / f"{outcomes_path.parent.name}_completed"
        if completed_dir.exists():
            shutil.rmtree(completed_dir)
        outcomes_path.parent.rename(completed_dir)
        print(f"checkpoint archived to {completed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
