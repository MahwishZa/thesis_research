"""Read-only diagnostics on a finished label-generation run.

Answers two questions before more training time is spent:
  1. Which decision-tree rule produced each label? (correctness flips vs the
     perplexity-percentile tie-break.) Tie-break HELPFUL labels are "top tau
     of perplexity reduction" BY CONSTRUCTION, not a judgement about content.
  2. Is there ANY signal in the prompt text a cheap model can find? 5-fold
     cross-validated TF-IDF + logistic regression vs the majority baseline
     (needs scikit-learn; skipped with a message if absent). Not the T5 filter
     - a lower bound on learnability only.

    python -m _archive.rag2_filter_reproduction.filter_training.diagnose_labels \\
        --progress _archive/rag2_filter_reproduction/filter_training/labels/medqa_filter_labels.progress_completed
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .labeling import FILTER_TRAINING_PROMPT, HELPFUL, label_dataset, outcome_from_dict


def _ranks(x):
    """Average ranks (ties share the mean rank). numpy only."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(scores, positive):
    """P(score of a random positive > score of a random negative); 0.5 =
    no information. Mann-Whitney form, ties counted half."""
    import numpy as np
    y = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(scores)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a, b):
    import numpy as np
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def relevance_probe(outcomes, examples, device=None) -> None:
    """Does MedCPT query-passage relevance (the retrieval score itself)
    predict the labels? If even that is at chance, the labels carry little
    content signal a filter could learn."""
    import numpy as np
    from experiments.medchange.encoders import (
        medcpt_article_encoder, medcpt_query_encoder)
    q = medcpt_query_encoder(device=device).encode([o.question for o in outcomes])
    a = medcpt_article_encoder(device=device).encode([o.evidence for o in outcomes])
    rel = np.einsum("ij,ij->i", np.asarray(q), np.asarray(a))
    helpful = np.array([e.answer == HELPFUL for e in examples])
    is_flip = np.array([e.rule.startswith("correctness") for e in examples])
    red = np.array([o.perplexity_reduction for o in outcomes])
    print(f"relevance mean {rel.mean():.3f} sd {rel.std():.3f}")
    print(f"AUC(relevance -> HELPFUL): all {auc(rel, helpful):.3f} | "
          f"flip labels {auc(rel[is_flip], helpful[is_flip]):.3f} | "
          f"tie-break labels {auc(rel[~is_flip], helpful[~is_flip]):.3f}   "
          "(0.5 = no information)")
    print(f"Spearman(relevance, perplexity reduction): {spearman(rel, red):+.3f}")


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
    ap.add_argument("--relevance-probe", action="store_true",
                    help="also score pairs with the MedCPT encoders (needs the "
                         "cached models; a few minutes on CPU)")
    ap.add_argument("--device", default=None)
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

    if args.relevance_probe:
        relevance_probe(outcomes, examples, device=args.device)

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
