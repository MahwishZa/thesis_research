"""Classification metrics with confidence intervals. An unreadable answer (None) always counts as wrong."""

from __future__ import annotations

import math
import random
from collections import Counter

UNREADABLE = "(unreadable)"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_class(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict[str, dict]:
    """Precision, recall, F1 and support for each class."""
    out = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        predicted = sum(p == label for p in y_pred)
        support = sum(t == label for t in y_true)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": predicted}
    return out


def _balanced_and_macro(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> tuple[float, float]:
    """Mean recall and mean F1 over the classes present in y_true."""
    present = [label for label in labels if label in set(y_true)]
    stats = per_class(y_true, y_pred, present)
    if not present:
        return (0.0, 0.0)
    return (sum(stats[c]["recall"] for c in present) / len(present), sum(stats[c]["f1"] for c in present) / len(present))


def summarize(y_true: list[str], y_pred: list[str | None], labels: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """All metrics for one run. Balanced accuracy and macro-F1 get percentile bootstrap intervals over images."""
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    balanced, macro = _balanced_and_macro(y_true, y_pred, labels)
    rng = random.Random(seed)
    boot_b, boot_m = [], []
    for _ in range(n_boot if n else 0):
        idx = [rng.randrange(n) for _ in range(n)]
        b, m = _balanced_and_macro([y_true[i] for i in idx], [y_pred[i] for i in idx], labels)
        boot_b.append(b)
        boot_m.append(m)

    def interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return (0.0, 0.0)
        values = sorted(values)
        return (values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])

    columns = labels + [UNREADABLE]
    confusion = {t: {c: 0 for c in columns} for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p if p is not None else UNREADABLE] += 1
    counts = Counter(y_true)
    majority_label, majority_n = counts.most_common(1)[0] if counts else ("", 0)
    return {
        "n": n,
        "correct": correct,
        "unreadable": sum(p is None for p in y_pred),
        "accuracy": correct / n if n else 0.0,
        "accuracy_ci95": wilson(correct, n),
        "balanced_accuracy": balanced,
        "balanced_accuracy_ci95": interval(boot_b),
        "macro_f1": macro,
        "macro_f1_ci95": interval(boot_m),
        "per_class": per_class(y_true, y_pred, labels),
        "confusion": confusion,
        "baselines": {
            "chance_accuracy": 1 / len(labels) if labels else 0.0,
            "majority_class": majority_label,
            "majority_class_accuracy": majority_n / n if n else 0.0,
        },
        "bootstrap": {"resamples": n_boot, "seed": seed},
    }
