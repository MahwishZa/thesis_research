"""The metrics: every score is computed from the saved answers, so it can be recomputed at any time without asking a
model again. Pure functions, no files."""

from __future__ import annotations

import math
import random



# ==============================================================================================================================
# Metrics
#
# Metrics for a model that answers each file (an image or a text case) with one class name, or with nothing readable.
#
# Everything is computed from the saved answers (predictions.jsonl), so the metrics can be recomputed at any time without asking
# the model again. An unreadable answer (None) is a wrong answer. In the confusion matrix it gets its own column (UNREADABLE),
# so it lowers recall and every score derived from it, and it is never silently dropped.
#
# For class k (TP = true class k and answered k, FP = other class answered k, FN = true class k and answered otherwise):
#     precision = TP / (TP + FP)         undefined (None) when class k was never answered
#     recall    = TP / (TP + FN)         undefined (None) when no image has true class k
#     F1        = 2 TP / (2 TP + FP + FN)
# Averages over classes: macro (every class counts the same) and weighted (by the number of true files). Only classes that have at
# least one true file enter an average; a class that was never answered gets precision 0 in the macro precision. Balanced accuracy
# is the macro recall, and the main score: every class counts the same, whatever its number of test files (the measure of the ISIC
# 2018 lesion-diagnosis challenge). With one answer per file, micro-averaged precision, recall and F1 all equal accuracy, so they
# are not reported. MCC (Matthews correlation coefficient) and Cohen's kappa are the two usual chance-corrected scores for more
# than two classes. Precision and F1 depend on how common each class is in the test set; recall and balanced accuracy do not.
#
# The 95% intervals: photos of one lesion, like patients of one article, are not independent, so when the files carry their group
# (``groups``), every interval comes from a bootstrap that resamples whole groups within each true class. Without group
# information each file is its own group; accuracy and the recall of each class then get the Wilson score interval.
#
# A class with very few files in a split (actinic keratosis in the text dataset: 2 test cases) weighs as much as any other class
# in balanced accuracy, yet its recall can take only a few values. The report then also gives the scores without its files,
# beside the main score (DATASETS names such classes).
#
# Hallucination: an answer is right (it names the true class), a hallucination (it names a class the ground truth does not
# support: a wrong diagnosis, stated as the answer), or no valid answer (a refusal, a sentence, two classes, a cut-off: it names no
# diagnosis, so it is no hallucination, and is counted apart). Right + hallucination + no valid answer = every question. The
# hallucination rate is the share of the questions answered with a wrong diagnosis; the class-balanced rate is the mean of the
# shares of the classes (the same weighting as balanced accuracy, and the same as the plain rate when every class has the same
# number of questions). The models are forced to choose a class, so a model cannot avoid hallucinating by saying it is unsure.
#
# Abstention (only a run whose prompt offers the answer "uncertain"; the main runs force a choice): the share of the questions
# answered "uncertain" (the abstention rate), and the accuracy among the questions where a diagnosis was named (selective
# accuracy, which is one minus the hallucination rate among them). An abstention is no hallucination and no right answer: a model
# that always abstains has no hallucination and no use, so the three are read together. No question of these datasets is
# unanswerable (each has its true class), so an abstention is never "correct" in itself: whether it was wise is measured by
# comparing the run with the forced-choice run of the same model (compare: hallucinations avoided, right answers lost).
#
# Clinical risk: of the cancers (MALIGNANT), the share answered as a benign class (missed malignancy, the most harmful error a
# diagnosis can make); of the benign lesions (BENIGN), the share answered as a cancer (over-call). Actinic keratosis is
# precancerous and is in neither group as a true class. Both shares are of ALL the files of the group: a cancer answered as
# actinic keratosis, or not readable, is no miss but stays in the total (so a model cannot lower its rate by not answering).
#
# Specificity of a class: of the files that are not that class, the share not answered as it (an unreadable answer is not an
# answer of the class). With eight classes most files are negatives of any one class, so specificity is high by construction and is
# read beside recall, never alone. Malignancy sensitivity: of the cancers, the share answered as any cancer (the right cancer or
# another one); it is not 1 minus the missed malignancy, because a cancer answered as actinic keratosis or not readably is neither
# detected nor missed. Benign specificity: of the benign lesions, the share answered as any benign class.
#
# ROC curves and AUC are deliberately absent: they need a score per class for every image, and a model that answers with one class
# name gives none. Malignancy sensitivity with benign specificity is the one point of such a curve these answers give.
# ==============================================================================================================================


UNREADABLE = "(unreadable)"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
ABSTAIN = "uncertain"  # the answer of a run that may abstain (config [prompt] abstain = true); never a class name
MALIGNANT = ("melanoma", "basal cell carcinoma", "squamous cell carcinoma")  # cancers
BENIGN = ("nevus", "pigmented benign keratosis", "dermatofibroma", "vascular lesion")  # actinic keratosis: precancerous, in neither


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
    """Rows: the true class (in the order of ``labels``). Columns: the answer, in the same order, then one column for unreadable."""
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("the class list contains a name twice")
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for true, answer in zip(y_true, y_pred, strict=True):  # strict: lists of different lengths are an error, not a shorter score
        if true not in index:
            raise ValueError(f"true label {true!r} is not one of the classes")
        if answer is not None and answer not in index:
            raise ValueError(f"answer {answer!r} is not one of the classes (an unreadable answer must be None)")
        matrix[index[true]][len(labels) if answer is None else index[answer]] += 1
    return matrix


def _scores(matrix: list[list[int]]) -> dict:
    """Every score that is computed from a confusion matrix (rows: true class; last column: unreadable)."""
    k = len(matrix)
    n = sum(map(sum, matrix))
    support = [sum(row) for row in matrix]
    answered = [sum(matrix[i][j] for i in range(k)) for j in range(k + 1)]  # how often each answer was given; the last is unreadable
    tp = [matrix[i][i] for i in range(k)]
    precision = [_ratio(tp[i], answered[i]) for i in range(k)]
    recall = [_ratio(tp[i], support[i]) for i in range(k)]
    f1 = [_ratio(2 * tp[i], 2 * tp[i] + (answered[i] - tp[i]) + (support[i] - tp[i])) for i in range(k)]
    specificity = [_ratio((n - support[i]) - (answered[i] - tp[i]), n - support[i]) for i in range(k)]  # true negatives / negatives
    present = [i for i in range(k) if support[i]]  # only these enter an average
    out = {"n": n, "support": support, "answered": answered, "tp": tp, "precision": precision, "recall": recall, "f1": f1,
           "specificity": specificity, "present": present}

    def mean(values, weights=None):
        if not present:
            return None
        values = [0.0 if values[i] is None else values[i] for i in present]  # a class never answered: precision 0
        weights = [1] * len(present) if weights is None else [weights[i] for i in present]
        return math.fsum(v * w for v, w in zip(values, weights)) / sum(weights)  # an exactly rounded sum: the same on every Python version

    out["macro"] = {"precision": mean(precision), "recall": mean(recall), "f1": mean(f1),
                    "specificity": None if None in [specificity[i] for i in present] else mean(specificity)}
    out["weighted"] = {"precision": mean(precision, support), "recall": mean(recall, support), "f1": mean(f1, support)}
    correct = sum(tp)
    if n:
        expected = sum(answered[j] * (support[j] if j < k else 0) for j in range(k + 1))  # unreadable is never a true class
        spread_answers = n * n - sum(a * a for a in answered)
        spread_truth = n * n - sum(s * s for s in support)
        out["mcc"] = _ratio(correct * n - expected, math.sqrt(spread_answers * spread_truth))
        out["kappa"] = _ratio(correct / n - expected / (n * n), 1 - expected / (n * n))
    else:
        out["mcc"] = out["kappa"] = None
    return out


def _hallucination(matrix: list[list[int]]) -> dict:
    """The share of the questions answered with a wrong diagnosis (plain, and the mean over the classes that occur), and the
    questions with no valid answer. ``matrix`` has the true class in its rows and, in its last column, the unreadable answers."""
    k = len(matrix)
    support = [sum(row) for row in matrix]
    wrong = [support[i] - matrix[i][i] - matrix[i][k] for i in range(k)]
    present = [i for i in range(k) if support[i]]
    n = sum(support)
    return {"rate": _ratio(sum(wrong), n), "wrong": sum(wrong), "n": n,
            "balanced_rate": _ratio(math.fsum(wrong[i] / support[i] for i in present), len(present)),
            "no_valid_answer": sum(row[k] for row in matrix), "no_valid_answer_rate": _ratio(sum(row[k] for row in matrix), n)}


def _risk(y_true, y_pred) -> dict:
    """Missed malignancy (cancers answered as a benign class), over-calls (benign lesions answered as a cancer), malignancy
    sensitivity (cancers answered as any cancer) and benign specificity (benign lesions answered as any benign class), as counts
    and shares; a share is None when no answer has a true class in its group."""
    cancers = [answer for true, answer in zip(y_true, y_pred) if true in MALIGNANT]
    benign = [answer for true, answer in zip(y_true, y_pred) if true in BENIGN]
    missed, overcalled = sum(a in BENIGN for a in cancers), sum(a in MALIGNANT for a in benign)
    detected, cleared = sum(a in MALIGNANT for a in cancers), sum(a in BENIGN for a in benign)
    return {"missed_malignancy": _ratio(missed, len(cancers)), "missed": missed, "cancers": len(cancers),
            "overcall": _ratio(overcalled, len(benign)), "overcalled": overcalled, "benign": len(benign),
            "malignancy_sensitivity": _ratio(detected, len(cancers)), "detected": detected,
            "benign_specificity": _ratio(cleared, len(benign)), "cleared": cleared}


def _percentile(sorted_values: list[float], q: float) -> float:
    """The q-quantile (0 to 1) of sorted values with linear interpolation between neighbours."""
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _units(y_true, labels, groups) -> dict:
    """For each true class that occurs, its groups, each as the list of the positions of its files."""
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
    groups. Yields the positions of each resample's files (random() is stable across Python versions)."""
    rng = random.Random(seed)
    for _ in range(resamples):
        yield [i for units in by_class.values() for _ in units for i in units[int(rng.random() * len(units))]]


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value: the chance of a split at least this uneven of the questions exactly one run gets right,
    if both runs were equally good (a binomial test with p = 1/2)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _bootstrap(y_true, y_pred, labels, groups, resamples: int, seed: int) -> dict:
    """95% percentile intervals of every score that gets one.

    Whole groups are resampled with replacement inside each true class (``groups`` names each file's group: the lesion an image
    shows, or the article a case comes from), so every resample keeps the number of groups of each class, and the files of one
    group are drawn together. The generator is seeded: the same answers always give the same interval.
    """
    by_class = _units(y_true, labels, groups)
    names = ["accuracy", "balanced_accuracy", "macro_f1", "missed_malignancy", "overcall", "hallucination", "hallucination_balanced",
             "abstention", "selective_accuracy", "macro_precision", "macro_specificity", "malignancy_sensitivity", "benign_specificity"]
    if not by_class:
        return {name: None for name in names} | {"recall": {}, "specificity": {}}
    values = {name: [] for name in names}
    recall = {label: [] for label in by_class}
    specificity = {label: [] for label in labels}
    for sample in _resamples(by_class, resamples, seed):
        true, pred = [y_true[i] for i in sample], [y_pred[i] for i in sample]
        matrix = confusion_matrix(true, [None if a == ABSTAIN else a for a in pred], labels)  # an abstention is not a class
        scores, hallucination = _scores(matrix), _hallucination(matrix)
        values["abstention"].append(sum(a == ABSTAIN for a in pred) / len(pred))
        values["selective_accuracy"].append(_ratio(sum(scores["tp"]), sum(scores["tp"]) + hallucination["wrong"]))
        values["hallucination"].append(hallucination["rate"])
        values["hallucination_balanced"].append(hallucination["balanced_rate"])
        values["accuracy"].append(sum(scores["tp"]) / scores["n"])
        values["balanced_accuracy"].append(scores["macro"]["recall"])
        values["macro_f1"].append(scores["macro"]["f1"])
        values["macro_precision"].append(scores["macro"]["precision"])
        values["macro_specificity"].append(scores["macro"]["specificity"])
        risk = _risk(true, pred)
        for name in ("missed_malignancy", "overcall", "malignancy_sensitivity", "benign_specificity"):
            values[name].append(risk[name])
        for label in by_class:
            recall[label].append(scores["recall"][labels.index(label)])
        for i, label in enumerate(labels):
            specificity[label].append(scores["specificity"][i])

    def interval(series):  # every resample holds every class, so a score is defined in all resamples or in none
        if None in series:
            return None
        series = sorted(series)
        return (_percentile(series, 0.025), _percentile(series, 0.975))

    return {name: interval(series) for name, series in values.items()} | {
        "recall": {k: interval(v) for k, v in recall.items()}, "specificity": {k: interval(v) for k, v in specificity.items()}}


def summarize(y_true: list[str], y_pred: list[str | None], labels: list[str], groups: list[str] | None = None,
              group_name: str = "lesion") -> dict:
    """All metrics of one run.

    ``labels`` is the full class list (the answer options); a class that no file has as its true class appears in
    ``classes_absent`` and is left out of the averages. ``groups`` names the group of each file, ``group_name`` what a group is
    (the files of one group are resampled together for the intervals). With no answers at all, every score is None (and accuracy
    0.0).
    """
    abstained = sum(a == ABSTAIN for a in y_pred)  # "uncertain": no class, so the confusion matrix counts it with the unreadable
    matrix = confusion_matrix(y_true, [None if a == ABSTAIN else a for a in y_pred], labels)
    s = _scores(matrix)
    n = s["n"]
    correct = sum(s["tp"])
    if groups is not None and len(groups) != len(y_true):
        raise ValueError("there must be one group per answer")
    groups = list(groups) if groups is not None else list(range(len(y_true)))
    independent = len(set(groups)) == len(groups)  # every file its own group: Wilson intervals for accuracy and recall
    interval = _bootstrap(y_true, y_pred, labels, groups, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    hallucination = _hallucination(matrix)
    hallucination["rate_ci95"] = wilson(hallucination["wrong"], n) if independent and n else interval["hallucination"]
    hallucination["balanced_rate_ci95"] = interval["hallucination_balanced"]
    hallucination["no_valid_answer"] -= abstained  # an abstention is neither a hallucination nor a failure to answer
    hallucination["no_valid_answer_rate"] = _ratio(hallucination["no_valid_answer"], n)
    committed = correct + hallucination["wrong"]  # the questions where a diagnosis was named
    abstention = {"count": abstained, "rate": _ratio(abstained, n),
                  "rate_ci95": None if not n else wilson(abstained, n) if independent else interval["abstention"],
                  "committed": committed, "selective_accuracy": _ratio(correct, committed),
                  "selective_accuracy_ci95": None if not committed else wilson(correct, committed) if independent
                  else interval["selective_accuracy"]}
    risk = _risk(y_true, y_pred)
    for share, count, of in (("missed_malignancy", "missed", "cancers"), ("overcall", "overcalled", "benign"),
                             ("malignancy_sensitivity", "detected", "cancers"), ("benign_specificity", "cleared", "benign")):
        risk[f"{share}_ci95"] = None if not risk[of] else wilson(risk[count], risk[of]) if independent else interval[share]
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {"support": s["support"][i], "answered": s["answered"][i], "tp": s["tp"][i],
                            "fp": s["answered"][i] - s["tp"][i], "fn": s["support"][i] - s["tp"][i],
                            "precision": s["precision"][i], "recall": s["recall"][i], "f1": s["f1"][i],
                            "recall_ci95": None if not s["support"][i] else
                            wilson(s["tp"][i], s["support"][i]) if independent else interval["recall"][label],
                            "tn": (n - s["support"][i]) - (s["answered"][i] - s["tp"][i]), "specificity": s["specificity"][i],
                            "specificity_ci95": None if s["specificity"][i] is None else
                            wilson((n - s["support"][i]) - (s["answered"][i] - s["tp"][i]), n - s["support"][i]) if independent
                            else interval["specificity"][label]}
    return {"n": n, "correct": correct, "unreadable": s["answered"][-1] - abstained, "groups": len(set(groups)), "group_name": group_name,
            "interval_method": "Wilson for accuracy and recall, bootstrap for the rest" if independent else f"bootstrap over {group_name}s",
            "balanced_accuracy": s["macro"]["recall"], "balanced_accuracy_ci95": interval["balanced_accuracy"],
            "accuracy": correct / n if n else 0.0, "accuracy_ci95": wilson(correct, n) if independent else interval["accuracy"],
            "classes_absent": [label for i, label in enumerate(labels) if not s["support"][i]],
            "macro": {**s["macro"], "f1_ci95": interval["macro_f1"], "precision_ci95": interval["macro_precision"],
                      "specificity_ci95": interval["macro_specificity"]},
            "weighted": s["weighted"],
            "mcc": s["mcc"], "kappa": s["kappa"], "risk": risk, "hallucination": hallucination,
            "abstention": abstention,
            "per_class": per_class,
            "confusion_matrix": {"rows": list(labels), "columns": [*labels, UNREADABLE], "counts": matrix}}
