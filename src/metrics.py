"""Accuracy with its 95% interval. An unreadable answer (None) always counts as wrong.

Accuracy is the only metric reported: further metrics are added only once the supervisor approves them. They can be
computed later from the saved answers (predictions.jsonl) without asking the model again.
"""

from __future__ import annotations

import math

UNREADABLE = "(unreadable)"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(y_true: list[str], y_pred: list[str | None]) -> dict:
    """Accuracy of one run: correct answers out of all images, with its 95% Wilson interval."""
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return {"n": n, "correct": correct, "unreadable": sum(p is None for p in y_pred),
            "accuracy": correct / n if n else 0.0, "accuracy_ci95": wilson(correct, n)}
