"""Shared statistics: percentiles, bootstrap intervals over lesions, Holm correction, McNemar's exact test, cosine."""
import math
import random
from collections import defaultdict


def percentile(sorted_values, q):
    """The q-th percentile with linear interpolation (the same as numpy's default)."""
    k = (len(sorted_values) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def bootstrap_ci(y, p, groups, fn, n=1000, seed=42):
    """95% percentile interval of fn. Whole lesions are drawn with replacement, so photos of one lesion stay together."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    keys, rng, values = list(members), random.Random(seed), []
    for _ in range(n):
        idx = [i for g in rng.choices(keys, k=len(keys)) for i in members[g]]
        values.append(fn([y[i] for i in idx], [p[i] for i in idx]))
    values.sort()
    return [round(percentile(values, 2.5), 4), round(percentile(values, 97.5), 4)]


def holm(p_values):
    """Holm's step-down correction: the i-th smallest of m p-values is multiplied by (m - i + 1), the results are made
    non-decreasing in that order and capped at 1. Returns the adjusted p-values in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def mcnemar_exact(y, pa, pb):
    """McNemar's exact test: a two-sided binomial test (p = 0.5) on the questions that exactly one system got right."""
    b = sum(a == t and c != t for t, a, c in zip(y, pa, pb))
    c = sum(a != t and x == t for t, a, x in zip(y, pa, pb))
    n = b + c
    p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": round(p_value, 6)}


def pct(x, digits=1):
    """A fraction as a percentage for the screen (0.2542 -> "25.4%"); the files keep fractions."""
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0
