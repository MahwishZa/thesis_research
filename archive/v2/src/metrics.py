"""The metrics: every score is computed from the saved answers, so it can be recomputed at any time without asking a model again.
Pure functions, no files. The definitions and their sources are in docs/method.md.

An answer gives a diagnosis (the first of its differential), the differential itself (three different classes, the most likely
first) and whether the model judged the evidence sufficient ("yes" or "no"). An invalid answer (None: not the required form) has
none of these; it counts as wrong in every score and is counted apart. Answers of the first phase give only a diagnosis.

    balanced accuracy          the mean over the classes of the share of each class's questions answered with it: the macro-
                               averaged recall, which is the balanced accuracy of Brodersen et al. (2010, defined there for two
                               classes) extended to many; every class counts the same (macro averaging, Sokolova and Lapalme,
                               2009). The main score.
    macro-F1                   the mean over the classes of F1 = 2 TP / (2 TP + FP + FN) (Sokolova and Lapalme, 2009). It also
                               counts the answers that name a class wrongly (false positives), which balanced accuracy ignores.
    per class                  precision TP / (TP + FP) and recall TP / (TP + FN); a class never answered has no precision and
                               counts as 0 in the macro mean; a class with no question is left out of every mean.
    macro-averaged top-3       as balanced accuracy, with an answer right when the true class is anywhere in its differential
    accuracy                   (top-k accuracy, as reported by Liu et al., 2020, averaged over the classes). The key stays
                               ``top3_balanced_accuracy``.
    coverage, selective risk   selective classification (Geifman and El-Yaniv, 2017): the model answers a question by marking its
                               evidence "sufficient" and abstains by marking it "not sufficient". Coverage is the share of all
                               questions answered validly with "sufficient"; selective risk is the share of wrong diagnoses among
                               those questions (called confident misdiagnosis in the thesis).
    invalid-answer rate        the share of the questions whose answer is not of the required form.
    precision@k, nDCG@k        open-book runs only: how many of the first k retrieved passages are relevant, and the normalised
                               discounted cumulative gain of the ranking (Jarvelin and Kekalainen, 2002; nDCG@10 is the main
                               measure of BEIR, Thakur et al., 2021). Relevance is silver: a passage is relevant when its disease
                               topics include the question's true class. See ``precision_at_k`` and ``ndcg_at_k``.

The 95% intervals: photos of one lesion, like patients of one article, are not independent, so when the answers carry their
group, every interval comes from a cluster bootstrap (Field and Welsh, 2007) that resamples whole groups within each true class, so
that each class keeps its number of groups (a stratified cluster bootstrap: the stratification is a design choice of this thesis;
2,000 resamples, a fixed seed).
When every answer is its own group, a share of counts gets the Wilson score interval and the averages over classes the bootstrap.
ROC curves and AUC are absent: they need a score per class for every question, and the models give a choice, not scores.
"""

from __future__ import annotations

import math
import random

INVALID = "(invalid)"  # the confusion matrix's column of the answers that are not of the required form
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def confusion_matrix(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> list[list[int]]:
    """Rows: the true class (in the order of ``labels``). Columns: the answer, in the same order, then one column for invalid."""
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("the class list contains a name twice")
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for true, answer in zip(y_true, y_pred, strict=True):  # strict: lists of different lengths are an error, not a shorter score
        if true not in index:
            raise ValueError(f"true label {true!r} is not one of the classes")
        if answer is not None and answer not in index:
            raise ValueError(f"answer {answer!r} is not one of the classes (an invalid answer must be None)")
        matrix[index[true]][len(labels) if answer is None else index[answer]] += 1
    return matrix


def _mean(values: list[float | None], present: list[int]) -> float | None:
    """The mean over the classes that occur; a value that is undefined (a class never answered) counts as 0."""
    if not present:
        return None
    return math.fsum(0.0 if values[i] is None else values[i] for i in present) / len(present)  # exactly rounded on every Python


def _scores(matrix: list[list[int]]) -> dict:
    """The scores computed from a confusion matrix (rows: true class; last column: invalid)."""
    k = len(matrix)
    support = [sum(row) for row in matrix]
    answered = [sum(matrix[i][j] for i in range(k)) for j in range(k + 1)]  # how often each answer was given; the last is invalid
    tp = [matrix[i][i] for i in range(k)]
    precision = [_ratio(tp[i], answered[i]) for i in range(k)]
    recall = [_ratio(tp[i], support[i]) for i in range(k)]
    f1 = [_ratio(2 * tp[i], 2 * tp[i] + (answered[i] - tp[i]) + (support[i] - tp[i])) for i in range(k)]
    present = [i for i in range(k) if support[i]]
    return {"n": sum(support), "support": support, "answered": answered, "tp": tp, "precision": precision, "recall": recall,
            "f1": f1, "present": present, "balanced_accuracy": _mean(recall, present),
            "macro": {"precision": _mean(precision, present), "recall": _mean(recall, present), "f1": _mean(f1, present)}}


def balanced_accuracy(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> float | None:
    """The mean over the classes that occur of the share of each class's questions answered with it."""
    return _scores(confusion_matrix(y_true, y_pred, labels))["balanced_accuracy"]


def top_k_balanced_accuracy(y_true: list[str], differentials: list[list[str] | None], labels: list[str]) -> float | None:
    """As balanced accuracy, with an answer right when the true class is anywhere in its differential (None: invalid, wrong)."""
    hits = [true if differential and true in differential else None for true, differential in zip(y_true, differentials, strict=True)]
    return balanced_accuracy(y_true, hits, labels)


def _selective(y_true, y_pred, flags) -> dict:
    """Selective prediction from the valid answers' diagnoses and flags ("yes": the model answers, "no": it abstains)."""
    valid = [(t == p, f) for t, p, f in zip(y_true, y_pred, flags, strict=True) if p is not None and f in ("yes", "no")]
    covered = [ok for ok, f in valid if f == "yes"]
    return {"valid": len(valid), "covered": len(covered), "coverage": _ratio(len(covered), len(y_true)),
            "covered_wrong": sum(not ok for ok in covered), "selective_risk": _ratio(sum(not ok for ok in covered), len(covered))}


def precision_at_k(relevant: list[int], k: int) -> float:
    """The share of the first k retrieved passages that are relevant. ``relevant`` holds 1 (relevant) or 0 for each retrieved passage,
    best rank first; a ranking shorter than k counts the missing places as not relevant."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return sum(relevant[:k]) / k


def ndcg_at_k(relevant: list[int], k: int, n_relevant: int) -> float | None:
    """Normalised discounted cumulative gain at k with binary relevance (Jarvelin and Kekalainen, 2002): DCG = sum of rel / log2(rank + 1)
    over the first k ranks, divided by the DCG of the best possible ranking, which puts min(k, n_relevant) relevant passages first.
    ``n_relevant`` is how many relevant passages the whole index holds for the question; None when it is 0 (nothing to find)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if n_relevant < 0:
        raise ValueError("n_relevant cannot be negative")
    if n_relevant == 0:
        return None
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevant[:k], start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, n_relevant) + 1))
    return dcg / ideal


def _percentile(sorted_values: list[float], q: float) -> float:
    """The q-quantile (0 to 1) of sorted values with linear interpolation between neighbours."""
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _units(y_true, labels, groups) -> dict:
    """For each true class that occurs, its groups, each as the list of the positions of its answers."""
    by_class = {}
    for label in labels:
        units = {}
        for i, true in enumerate(y_true):
            if true == label:
                units.setdefault(groups[i], []).append(i)
        if units:
            by_class[label] = list(units.values())
    return by_class


def _resamples(by_class: dict, resamples: int, seed: int):
    """Seeded resamples: whole groups drawn with replacement inside each class, so every resample keeps each class's number of
    groups. Yields the positions of each resample's answers (random() is stable across Python versions)."""
    rng = random.Random(seed)
    for _ in range(resamples):
        yield [i for units in by_class.values() for _ in units for i in units[int(rng.random() * len(units))]]


def interval(series: list[float | None]) -> tuple[float, float] | None:
    """The 95% percentile interval of bootstrap values; None when a value is undefined in some resample."""
    if not series or None in series:
        return None
    series = sorted(series)
    return (_percentile(series, 0.025), _percentile(series, 0.975))


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value: the chance of a split at least this uneven of the questions exactly one run gets right,
    if both runs were equally good (a binomial test with p = 1/2)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(p_values: list[float]) -> list[float]:
    """Holm's step-down correction for several tests: the smallest p-value is multiplied by the number of tests, the next by one
    less, and so on; no adjusted value may be smaller than the one before it or larger than 1. Returned in the order given."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _bootstrap(y_true, y_pred, labels, groups, differentials, flags) -> dict:
    """The bootstrap values of every score that gets an interval. Whole groups are resampled with replacement inside each true
    class, so every resample keeps the number of groups of each class, and the answers of one group are drawn together."""
    by_class = _units(y_true, labels, groups)
    values = {name: [] for name in ("balanced_accuracy", "macro_f1", "top3_balanced_accuracy", "coverage",
                                    "selective_risk")}
    recall = {label: [] for label in by_class}
    if not by_class:
        return values | {"recall": recall}
    for sample in _resamples(by_class, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED):
        true = [y_true[i] for i in sample]
        s = _scores(confusion_matrix(true, [y_pred[i] for i in sample], labels))
        values["balanced_accuracy"].append(s["balanced_accuracy"])
        values["macro_f1"].append(s["macro"]["f1"])
        if differentials is not None:
            values["top3_balanced_accuracy"].append(top_k_balanced_accuracy(true, [differentials[i] for i in sample], labels))
        if flags is not None:
            d = _selective(true, [y_pred[i] for i in sample], [flags[i] for i in sample])
            for name in ("coverage", "selective_risk"):
                values[name].append(d[name])
        for label in by_class:
            recall[label].append(s["recall"][labels.index(label)])
    return values | {"recall": recall}


def summarize(y_true: list[str], y_pred: list[str | None], labels: list[str], groups: list | None = None,
              group_name: str = "lesion", differentials: list | None = None, flags: list | None = None) -> dict:
    """Every score of one run (see the module's description).

    ``labels`` is the full class list (the answer options); a class that no question has as its true class appears in
    ``classes_absent`` and is left out of the averages. ``groups`` names each answer's group, ``group_name`` what a group is.
    ``differentials`` (each answer's three classes, None when invalid) and ``flags`` ("yes", "no", or None when invalid) are
    given for answers of the current form; without them the scores that need them are None.
    """
    if groups is not None and len(groups) != len(y_true):
        raise ValueError("there must be one group per answer")
    groups = list(groups) if groups is not None else list(range(len(y_true)))
    independent = len(set(groups)) == len(groups)  # every answer its own group: Wilson intervals for the shares
    matrix = confusion_matrix(y_true, y_pred, labels)
    s = _scores(matrix)
    n = s["n"]
    boot = _bootstrap(y_true, y_pred, labels, groups, differentials, flags)
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {"support": s["support"][i], "answered": s["answered"][i], "tp": s["tp"][i],
                            "precision": s["precision"][i], "recall": s["recall"][i], "f1": s["f1"][i],
                            "recall_ci95": None if not s["support"][i] else
                            wilson(s["tp"][i], s["support"][i]) if independent else interval(boot["recall"][label])}
    out = {"n": n, "correct": sum(s["tp"]), "invalid": s["answered"][-1], "invalid_rate": _ratio(s["answered"][-1], n),
           "groups": len(set(groups)), "group_name": group_name,
           "interval_method": "bootstrap, and Wilson for shares of counts" if independent else f"bootstrap over {group_name}s",
           "balanced_accuracy": s["balanced_accuracy"], "balanced_accuracy_ci95": interval(boot["balanced_accuracy"]),
           "macro_f1": s["macro"]["f1"], "macro_f1_ci95": interval(boot["macro_f1"]), "macro_precision": s["macro"]["precision"],
           "classes_absent": [label for i, label in enumerate(labels) if not s["support"][i]], "per_class": per_class,
           "confusion_matrix": {"rows": list(labels), "columns": [*labels, INVALID], "counts": matrix},
           "top3_balanced_accuracy": None, "top3_balanced_accuracy_ci95": None, "selective": None}
    if differentials is not None:
        out["top3_balanced_accuracy"] = top_k_balanced_accuracy(y_true, differentials, labels)
        out["top3_balanced_accuracy_ci95"] = interval(boot["top3_balanced_accuracy"])
    if flags is not None:
        d = _selective(y_true, y_pred, flags)
        for name, k, total in (("coverage", "covered", None), ("selective_risk", "covered_wrong", "covered")):
            of = n if total is None else d[total]
            d[f"{name}_ci95"] = None if not of else wilson(d[k], of) if independent else interval(boot[name])
        out["selective"] = d
    return out
