"""Diagnosis metrics: accuracy, macro-F1, per-class scores, invalid answers and refusals, tokens and time."""
import re
from collections import Counter

from src.evaluation.metrics import (
    INVALID,
    bootstrap_ci,
    gold_class,
    mcnemar_exact,
    mean,
    predicted_class,
)
from src.utils.common import CLASSES

# A refusal: the host flagged it (finish_reason "refusal...") or the reply says so in one of these fixed phrases (set on 2026-11-22
# from a real reply: "I'm sorry, I can't help with identifying or diagnosing images of skin lesions."). Only an invalid answer
# (no usable letter) is checked, so a valid answer that apologises is never counted.
REFUSAL = re.compile(r"\bI(?:['’]m| am) sorry\b|\bI(?: can['’]?t| cannot| can not| won['’]t| am unable to|['’]m unable to"
                     r"| am not able to|['’]m not able to) (?:help|assist|provide|identify|diagnose|answer|comply|interpret"
                     r"|analy[sz]e|evaluate)", re.I)


def invalid_causes(ids, classes, pred):
    """Splits the invalid answers into refusals (REFUSAL) and other invalid answers (a reply without a usable letter)."""
    out = {"refusals": [], "other": []}
    for i, c in zip(ids, classes):
        r = pred.get(i)
        if r is None or c != INVALID:
            continue
        refused = str(r.get("finish_reason") or "").startswith("refusal") or REFUSAL.search(str(r.get("raw_output") or ""))
        out["refusals" if refused else "other"].append(i)
    return {"refusals": len(out["refusals"]), "other_invalid": len(out["other"]), "refusal_ids": out["refusals"],
            "other_invalid_ids": out["other"]}


def per_class(y, p):
    """Precision, recall (= sensitivity), specificity and F1 of each class; an invalid answer is wrong for every class."""
    pairs, n, rows = Counter(zip(y, p)), len(y), {}
    for c in CLASSES:
        tp = pairs[(c, c)]
        fn = sum(v for (t, q), v in pairs.items() if t == c and q != c)
        fp = sum(v for (t, q), v in pairs.items() if t != c and q == c)
        tn = n - tp - fn - fp
        prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        rows[c] = {"precision": prec, "recall": rec, "specificity": tn / (tn + fp) if tn + fp else 0.0,
                   "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn}
    return rows


def accuracy(y, p):
    return sum(t == q for t, q in zip(y, p)) / len(y)


def macro_f1(y, p):
    """The mean F1 over the eight classes (every class weighs the same)."""
    return sum(r["f1"] for r in per_class(y, p).values()) / len(CLASSES)


def diagnosis_metrics(qa, pred, pred_b=None):
    unknown = sorted(set(pred) - set(qa))
    if unknown:
        raise ValueError(f"{len(unknown)} predictions are not in the question file, e.g. {unknown[:3]}")
    ids = list(qa)
    y = [gold_class(qa[i]) for i in ids]
    p = [predicted_class(qa[i], pred.get(i)) for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    rows = per_class(y, p)
    columns = [*CLASSES, INVALID]
    out = {"n_questions": len(ids),
           "missing_predictions": sum(i not in pred for i in ids),
           "invalid_answers": sum(i in pred and c == INVALID for i, c in zip(ids, p)),
           "invalid_answers_by_cause": invalid_causes(ids, p, pred),
           "accuracy": round(accuracy(y, p), 4), "accuracy_95ci": bootstrap_ci(y, p, groups, accuracy),
           "macro_f1": round(macro_f1(y, p), 4), "macro_f1_95ci": bootstrap_ci(y, p, groups, macro_f1),
           "per_class": {c: {k: (v if k == "support" else round(v, 3)) for k, v in r.items()} for c, r in rows.items()},
           "confusion_matrix": {"rows_true": CLASSES, "columns_predicted": columns,
                                "matrix": [[sum(t == r and q == c for t, q in zip(y, p)) for c in columns] for r in CLASSES]}}
    out["chosen_letters"] = {k: v for k, v in sorted(Counter(
        (str((pred.get(i) or {}).get("pred") or "").strip().upper() or "none") if c != INVALID else "invalid"
        for i, c in zip(ids, p)).items())}
    if pred_b is not None:
        out["mcnemar_vs_system_b"] = mcnemar_exact(y, p, [predicted_class(qa[i], pred_b.get(i)) for i in ids])
    return out


def usage_summary(qa, pred):
    """Totals and means of tokens and seconds, for the predictions that report them (None when none does)."""
    out = {}
    for key in ("tokens_in", "tokens_out", "seconds"):
        values = [float(pred[i][key]) for i in qa if i in pred and pred[i].get(key) is not None]
        if values:
            out[key] = {"total": round(sum(values), 2), "mean": round(mean(values), 2), "n": len(values)}
    return out or None
