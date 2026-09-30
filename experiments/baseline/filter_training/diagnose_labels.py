"""Read-only diagnostics on a finished label-generation run.

Answers two questions before more training time is spent:
  1. Which decision-tree rule produced each label? (correctness flips vs the
     perplexity-percentile tie-break.) Tie-break HELPFUL labels are "top tau
     of perplexity reduction" BY CONSTRUCTION, not a judgement about content.
  2. Is there ANY signal in the prompt text a cheap model can find? 5-fold
     cross-validated TF-IDF + logistic regression vs the majority baseline
     (needs scikit-learn; skipped with a message if absent). Not the T5 filter
     - a lower bound on learnability only.

    python -m experiments.baseline.filter_training.diagnose_labels \\
        --progress experiments/baseline/filter_training/labels/medqa_filter_labels.progress_completed
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .labeling import FILTER_TRAINING_PROMPT, HELPFUL, label_dataset, outcome_from_dict


def _load_outcomes(progress_dir: Path):
    path = progress_dir / "outcomes.jsonl"
    if not path.exists():
        cands = sorted(progress_dir.glob("*.jsonl"))
        if not cands:
            raise SystemExit(f"no outcomes .jsonl in {progress_dir}")
        path = cands[0]
    return [outcome_from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--progress", required=True,
                    help="the *.progress_completed directory from build_labels")
    args = ap.parse_args(argv)

    outcomes = _load_outcomes(Path(args.progress))
    examples = label_dataset(outcomes, dataset_name="medqa")
    print(f"{len(examples)} labelled pairs")
    by_rule = collections.Counter((e.rule, e.answer) for e in examples)
    for (rule, label), n in sorted(by_rule.items()):
        print(f"  {rule:40s} {label:14s} {n:4d}")
    n_help = sum(e.answer == HELPFUL for e in examples)
    n_tie_help = sum(e.answer == HELPFUL and e.rule.startswith("perplexity")
                     for e in examples)
    print(f"HELPFUL total {n_help}; from the perplexity tie-break {n_tie_help} "
          f"({100 * n_tie_help / max(1, n_help):.0f}%)")
    with_correct = sum(o.with_evidence.correct for o in outcomes)
    without_correct = sum(o.without.correct for o in outcomes)
    print(f"rationale answer correct: without evidence {without_correct}/"
          f"{len(outcomes)}, with evidence {with_correct}/{len(outcomes)}")

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("scikit-learn not installed - skipping the learnability probe "
              "(pip install scikit-learn to enable)")
        return 0
    def probe(name, idx):
        texts = [FILTER_TRAINING_PROMPT.format(evidence=outcomes[i].evidence,
                                               question=outcomes[i].question)
                 for i in idx]
        y = np.array([examples[i].answer == HELPFUL for i in idx])
        if min(y.sum(), (~y).sum()) < 5:
            print(f"{name}: too few examples in one class, skipped")
            return
        pipe = make_pipeline(
            TfidfVectorizer(min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight="balanced"))
        pred = cross_val_predict(
            pipe, texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
        print(f"TF-IDF+LR 5-fold [{name}, n={len(idx)}]: "
              f"accuracy {np.mean(pred == y):.3f}  "
              f"balanced {balanced_accuracy_score(y, pred):.3f}  "
              f"(majority baseline {max(y.mean(), 1 - y.mean()):.3f}, "
              "chance balanced 0.500)")

    probe("all labels", list(range(len(examples))))
    # The flip labels come from an observed change in answer correctness - no
    # percentile rule - so they are the cleaner subset.
    probe("correctness-flip labels only",
          [i for i, e in enumerate(examples) if e.rule.startswith("correctness")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
