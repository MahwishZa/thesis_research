"""Validation metrics for the RAG² filter, scored the way it is deployed.

``FlanT5RAG2Filter`` decides with a two-way argmax over the [HELPFUL] /
[NOT_HELPFUL] logits at the first decoder position. Training is therefore
evaluated with exactly that rule (teacher-forced logits at position 0), NOT
with ``generate()``: T5's ``generate()`` output begins with the decoder-start
id (0), so comparing its column 0 to the gold label id is always False and
reports 0.0 accuracy for a model that has learned the task (reproduced on a
tiny T5: old metric 0.0 vs deployed-rule accuracy 0.79).

Also reported: the majority-class baseline and balanced accuracy. The label
set is imbalanced (~71% NOT_HELPFUL), so plain accuracy > 0.5 is satisfied
by a filter that rejects everything.

Pure Python - no torch/numpy - so it is unit-testable without the ML stack.
"""

from __future__ import annotations

from typing import Sequence


def classification_metrics(
    predicted_helpful: Sequence[bool], gold_helpful: Sequence[bool],
) -> dict[str, float]:
    if len(predicted_helpful) != len(gold_helpful):
        raise ValueError("predictions and gold labels differ in length")
    n = len(gold_helpful)
    if n == 0:
        raise ValueError("cannot score an empty validation set")
    pairs = list(zip(predicted_helpful, gold_helpful))
    n_pos = sum(1 for _, g in pairs if g)
    n_neg = n - n_pos
    tp = sum(1 for p, g in pairs if p and g)
    tn = sum(1 for p, g in pairs if not p and not g)
    recall_helpful = tp / n_pos if n_pos else float("nan")
    recall_not = tn / n_neg if n_neg else float("nan")
    recalls = [r for r in (recall_helpful, recall_not) if r == r]
    return {
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "majority_baseline": max(n_pos, n_neg) / n,
        "recall_helpful": recall_helpful,
        "recall_not_helpful": recall_not,
        "predicted_helpful_fraction": sum(1 for p, _ in pairs if p) / n,
    }


def two_way_prediction(logit_helpful: float, logit_not_helpful: float) -> bool:
    """The deployed rule (``FlanT5RAG2Filter``): ``>=`` -> helpful."""
    return logit_helpful >= logit_not_helpful
