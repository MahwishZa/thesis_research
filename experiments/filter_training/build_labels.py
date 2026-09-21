"""CLI: generate RAG² filter training labels from MedQA + a textbook corpus.

Run on a GPU (Colab). Retrieves one passage per MedQA question from a
general-medical textbook corpus (never the thesis's Alzheimer's corpus -
specification SS10.6), scores each (question, passage) pair with
``Llama3RationaleScorer`` twice (without and with the passage), applies the
paper's label decision tree (``labeling.py``), and writes the training file
``experiments/filter_training/train.py`` reads.

    python -m experiments.filter_training.build_labels \\
        --n-questions 2000 \\
        --output experiments/filter_training/labels/medqa_filter_labels.json \\
        --model-revision <pinned-commit-sha>

``--n-questions`` should come from the generation-speed timing measurement
(specification SS9.3/SS10.4: "a budget, not a target"), not be guessed -
this script does not choose it for you.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from .labeling import PairOutcome, label_dataset, label_distribution, write_training_file
from .medqa_data import load_medqa, load_textbook_passages

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def build_textbook_index(n_passages: int, seed: int, device: Optional[str]):
    from experiments.retrieval.encoders import medcpt_article_encoder
    from experiments.retrieval.index import build_index

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-questions", type=int, required=True)
    ap.add_argument("--n-textbook-passages", type=int, default=50_000,
                    help="how many textbook passages to build the "
                         "retrieval index over")
    ap.add_argument("--output", required=True,
                    help="path to write the training-label JSON")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--model-revision", required=True,
                    help="pinned commit sha, not a branch name")
    ap.add_argument("--quantization", default="nf4",
                    choices=("nf4", "int8", "none"))
    ap.add_argument("--device", default=None, help="e.g. cuda; omit to auto-pick")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out_path = Path(args.output)
    if out_path.exists():
        print(f"refusing to overwrite existing labels at {out_path}",
              file=sys.stderr)
        return 2

    index, passages = build_textbook_index(
        args.n_textbook_passages, args.seed, args.device
    )

    from experiments.retrieval.encoders import medcpt_query_encoder
    query_encoder = medcpt_query_encoder(device=args.device)

    questions = load_medqa(n=args.n_questions, seed=args.seed)
    print(f"loaded {len(questions)} MedQA questions")

    from .rationale import Llama3RationaleScorer
    scorer = Llama3RationaleScorer(
        args.model_name, args.model_revision, device=args.device,
        quantization=None if args.quantization == "none" else args.quantization,
    )

    outcomes = []
    t0 = time.time()
    for i, item in enumerate(questions, 1):
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

        outcomes.append(PairOutcome(
            pair_id=item.item_id,
            question=item.rendered_question(),
            evidence=passage.text,
            without=without,
            with_evidence=with_evidence,
        ))

        if i % 50 == 0 or i == len(questions):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(questions) - i) / rate if rate > 0 else float("nan")
            print(f"  ...{i}/{len(questions)} pairs scored "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

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
            "task (specification SS10.5). Consider more/different "
            "questions before training on this file.",
            file=sys.stderr,
        )

    write_training_file(examples, out_path)
    print(f"wrote {len(examples)} labelled examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
