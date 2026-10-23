"""How well the filter reproduces held-out labels: accuracy as in the authors' evaluation, the majority rate, and the AUC.

The authors' ``run_classifier.py`` writes the overall accuracy and the accuracy per class (``final_eval_results.json`` and
``final_eval_results_perClass.json``) from the prediction of the larger of the two label-token logits. This module gives the
same numbers from predictions and gold labels, plus the accuracy of always answering the larger class, which the filter has
to beat, and the AUC of a score for the helpful class (the probability of the helpful token, or any monotone function of it).
"""

from __future__ import annotations

from collections.abc import Mapping

from .inference import decide_helpful
from .labels import TOKEN_HELPFUL, TOKEN_NOT_HELPFUL

CLASSES = (TOKEN_HELPFUL, TOKEN_NOT_HELPFUL)


def _check(predictions: Mapping[str, str], gold: Mapping[str, str]) -> list[str]:
    if set(predictions) != set(gold):
        missing, extra = sorted(set(gold) - set(predictions)), sorted(set(predictions) - set(gold))
        raise ValueError(f"predictions and gold labels cover different ids; missing {missing[:5]}, unexpected {extra[:5]}")
    if not gold:
        raise ValueError("there is nothing to evaluate")
    for name, labels in (("prediction", predictions), ("gold label", gold)):
        for key, value in labels.items():
            if value not in CLASSES:
                raise ValueError(f"{name} of {key} must be {TOKEN_HELPFUL} or {TOKEN_NOT_HELPFUL}, got {value!r}")
    return sorted(gold)


def accuracy_report(predictions: Mapping[str, str], gold: Mapping[str, str]) -> dict[str, object]:
    """Accuracy overall and per class, the confusion counts, and the accuracy of always naming the larger gold class."""
    ids = _check(predictions, gold)
    confusion = {actual: {guess: 0 for guess in CLASSES} for actual in CLASSES}
    for key in ids:
        confusion[gold[key]][predictions[key]] += 1
    correct = sum(confusion[name][name] for name in CLASSES)
    per_class = {}
    for name in CLASSES:
        total = sum(confusion[name].values())
        per_class[name] = round(confusion[name][name] / total, 4) if total else None
    larger = max(sum(row.values()) for row in confusion.values())
    return {
        "n": len(ids),
        "accuracy": round(correct / len(ids), 4),
        "accuracy_per_class": per_class,
        "majority_rate": round(larger / len(ids), 4),
        "confusion": confusion,
    }


def auc(scores: Mapping[str, float], gold: Mapping[str, str]) -> float:
    """The chance that a helpful pair gets a higher score than a not helpful pair (ties count half). Needs both classes."""
    if set(scores) != set(gold):
        raise ValueError("scores and gold labels cover different ids")
    helpful = [scores[key] for key in gold if gold[key] == TOKEN_HELPFUL]
    other = [scores[key] for key in gold if gold[key] == TOKEN_NOT_HELPFUL]
    if len(helpful) + len(other) != len(gold):
        raise ValueError(f"every gold label must be {TOKEN_HELPFUL} or {TOKEN_NOT_HELPFUL}")
    if not helpful or not other:
        raise ValueError("the AUC needs at least one pair of each class")
    ranked = sorted([(value, 1) for value in helpful] + [(value, 0) for value in other])
    # Rank sum of the helpful class, with the average rank for tied scores (the Mann-Whitney statistic).
    rank_sum, position = 0.0, 0
    while position < len(ranked):
        end = position
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[position][0]:
            end += 1
        average = (position + end) / 2 + 1
        rank_sum += average * sum(label for _, label in ranked[position : end + 1])
        position = end + 1
    return (rank_sum - len(helpful) * (len(helpful) + 1) / 2) / (len(helpful) * len(other))


def predictions_from_logits(logits: Mapping[str, tuple[float, float]]) -> dict[str, str]:
    """``{id: (logit of [HELPFUL], logit of [NOT_HELPFUL])}`` to labels, a tie counting as helpful as in the authors' code."""
    return {
        key: TOKEN_HELPFUL if decide_helpful(logit_helpful=helpful, logit_not_helpful=other) else TOKEN_NOT_HELPFUL
        for key, (helpful, other) in logits.items()
    }


def helpful_scores(logits: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    """The difference of the two logits, a monotone function of the probability of the helpful token (for the AUC)."""
    return {key: float(helpful - other) for key, (helpful, other) in logits.items()}

