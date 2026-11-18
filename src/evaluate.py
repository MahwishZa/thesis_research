"""Scores of finished runs, from their saved answers only: no model is asked.

    python -m src.evaluate report results/<dataset>/<approach>/<model>/<run-folder>
    python -m src.evaluate compare results/<dataset>/<approach>/<model>/<run-folder> results/<dataset>/<approach>/<model>/<run-folder>
    python -m src.evaluate summary

A run folder of the first phase is results/phase1/<dataset>/<model>/<run-folder>.

``report`` rewrites metrics.json and report.md from a run's saved answers.
``compare`` sets two finished runs on the same questions side by side (two models, or two variants of a method in an ablation):
McNemar's exact test on the questions one run gets right and the other wrong, the difference in balanced accuracy and in
accuracy, and the change in the hallucination rate (the difference, and the reduction relative to the first run), with paired
95% intervals (both runs resampled together, by the same groups as the reports).
``summary`` rewrites the comparison tables of docs/results.md from every finished test run of the first phase (results/phase1/):
the scores of every model on every dataset, the clinical risk, the recall per class, and every paired comparison with McNemar's exact test and its Holm correction.
The text of the document around the tables is written by hand and is left as it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .common import read_jsonl, write_json
from .datasets import DATASETS, MISSING_FACT, SAME_QUESTIONS, TITLES, Dataset, dataset_of
from .metrics import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, UNREADABLE, _hallucination, 
                      _percentile, _resamples, _scores, _units, confusion_matrix, mcnemar, summarize)
from .prompts import parse_last_line, can_still_become_readable

RESULTS = Path("results")  # results/<dataset>/<approach>/<model>/<run>/; the first phase in results/phase1/


# ---------------------------------------------------------------------------------------------------------------------------------
# The report of one run (python -m src.evaluate report): metrics.json and report.md, from the saved answers only


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(pair) -> str:
    return "–" if pair is None else f"{_pct(pair[0])} to {_pct(pair[1])}"


def _share(x: float | None) -> str:
    """A share as a percentage; a dash when the score is undefined."""
    return "–" if x is None else _pct(x)


def _coefficient(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _scoring(title: str, m: dict, kind: Dataset) -> list[str]:
    """The metrics of one scoring as report lines: the summary table, the per-class table and the confusion matrix."""
    counted = [c for c, v in m["per_class"].items() if v["support"]]
    item, items, groups = kind.item, f"{kind.item}s", f"{kind.group}s"
    grouped = m["interval_method"] == f"bootstrap over {groups}"
    over = f" over {groups}" if grouped else ""
    simple = f"bootstrap{over}" if grouped else "Wilson"
    lines = [f"## {title}", "", "| Metric | Value | 95% interval |", "|---|---|---|",
             f"| **Balanced accuracy (main score: the mean recall of the classes)** | **{_share(m['balanced_accuracy'])}** | "
             f"{_ci(m['balanced_accuracy_ci95'])} (bootstrap{over}) |",
             f"| Accuracy | {_pct(m['accuracy'])} ({m['correct']}/{m['n']}) | {_ci(m['accuracy_ci95'])} ({simple}) |",
             *_hallucination_rows(m, over if grouped else None),
             f"| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | "
             f"{m['unreadable']} | |",
             *_risk_rows(m, over if grouped else None),
             f"| Macro-F1 (classes with {items}: {len(counted)}) | {_share(m['macro']['f1'])} | {_ci(m['macro']['f1_ci95'])} "
             f"(bootstrap{over}) |",
             f"| Macro-precision | {_share(m['macro']['precision'])} | {_ci(m['macro']['precision_ci95'])} (bootstrap{over}) |",
             f"| Macro-specificity | {_share(m['macro']['specificity'])} | {_ci(m['macro']['specificity_ci95'])} (bootstrap{over}) |",
             f"| Weighted F1 (by number of true {items}) | {_share(m['weighted']['f1'])} | |",
             f"| Matthews correlation (MCC, -1 to 1) | {_coefficient(m['mcc'])} | |",
             f"| Cohen's kappa (-1 to 1) | {_coefficient(m['kappa'])} | |", "",
             "### Per class", "",
             f"| Class | True {items} | Answered as this class | Precision | Recall (95% interval: {simple}) | F1 | Specificity (95% interval: {simple}) |",
             "|---|---|---|---|---|---|---|"]
    for label, c in m["per_class"].items():
        recall = _share(c["recall"]) + (f" ({_ci(c['recall_ci95'])})" if c["recall_ci95"] else "")
        specificity = _share(c["specificity"]) + (f" ({_ci(c['specificity_ci95'])})" if c["specificity_ci95"] else "")
        lines.append(f"| {label} | {c['support']} | {c['answered']} | {_share(c['precision'])} | {recall} | {_share(c['f1'])} | {specificity} |")
    matrix = m["confusion_matrix"]
    numbers = range(1, len(matrix["rows"]) + 1)
    lines += ["", "### Confusion matrix", "",
              "Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.", "",
              "| True class | " + " | ".join(str(i) for i in numbers) + " | unreadable |",
              "|---|" + "---|" * (len(matrix["rows"]) + 1)]
    lines += [f"| {i}. {label} | " + " | ".join(str(v) for v in row) + " |"
              for i, label, row in zip(numbers, matrix["rows"], matrix["counts"])]
    absent = m["classes_absent"]
    supports = {v["support"] for v in m["per_class"].values() if v["support"]}
    lines += ["", "Notes on these scores:",
              f"- Balanced accuracy is the main score: every class counts the same, whatever its number of test {items} (the "
              "measure of the ISIC 2018 lesion-diagnosis challenge).",
              *([f"- Every class has the same number of test {items} ({supports.pop()}), so plain accuracy equals balanced accuracy."]
                if len(supports) == 1 and len(counted) > 1 else []),
              "- Precision and F1 depend on how common each class is in the test set, and the test set's mix of classes is not a "
              "clinic's: a rare disease is far more common here, so its precision is higher than it would be in a clinic, and it "
              "is not a clinical figure. Recall and balanced accuracy do not depend on it.",
              (f"- {kind.why_grouped}: the intervals resample whole {groups} ({m['groups']} {groups} for {m['n']} {items})."
               if grouped else f"- No two {items} share one {kind.group} (or the run recorded none), so every {item} counts as independent."),
              *([f"- No {item} has the true class {', '.join(absent)}: its recall is undefined and it is left out of the averages."]
                if absent else []),
              f"- A dash means undefined: precision when a class was never answered, recall when no {item} has that class. A class "
              "that was never answered counts as precision 0 in the macro-precision.",
              f"- With one answer per {item}, micro-averaged precision, recall and F1 equal accuracy, so they are not listed.",
              "- Hallucination: an answer that names a wrong diagnosis. Right answers, hallucinations and answers with no valid "
              "answer (a refusal, a sentence, two classes, a cut-off) add up to every question; a model that refuses is not "
              "hallucinating, but it is not right either. The models must choose a class, so none can avoid hallucinating by "
              f"declaring itself unsure. With the same number of {items} in every class, the two rates are the same.",
              "- Missed malignancy: melanoma, basal cell carcinoma and squamous cell carcinoma answered as nevus, pigmented benign "
              "keratosis, dermatofibroma or vascular lesion, as a share of all cancers. Over-call: the reverse, as a share of all benign "
              "lesions. An unreadable answer, or actinic keratosis (precancerous) given for a cancer, is no miss but stays in the "
              "total; actinic keratosis as a true class is in neither group.",
              "- Malignancy sensitivity: cancers answered as any cancer (the right one or another), as a share of all cancers. It is "
              "not 100% minus the missed malignancy: a cancer answered as actinic keratosis, or not readably, is neither detected nor "
              "missed. Benign specificity: benign lesions answered as any benign class, as a share of all benign lesions.",
              f"- Specificity of a class: of the {items} that are not that class, the share not answered as it. With {len(m['per_class'])} "
              "classes most are negatives of any one class, so specificity is high by construction and is read beside recall.",
              "- No ROC curve or AUC: the models answer with one class name, not a score per class, so there is nothing to rank. "
              "Malignancy sensitivity with benign specificity is the one point of such a curve these answers give.", ""]
    return lines


def _hallucination_rows(m: dict, over: str | None) -> list[str]:
    """The two hallucination rows of a summary table: the class-balanced rate (the main one) and the plain share of the questions."""
    h, method = m["hallucination"], f"bootstrap{over}" if over is not None else "Wilson"
    if h["rate"] is None:
        return []
    return [f"| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **{_share(h['balanced_rate'])}** | "
            f"{_ci(h['balanced_rate_ci95'])} (bootstrap{over or ''}) |",
            f"| Hallucination rate, all questions | {_pct(h['rate'])} ({h['wrong']}/{h['n']}) | {_ci(h['rate_ci95'])} ({method}) |"]


def _risk_rows(m: dict, over: str | None) -> list[str]:
    """The clinical-risk rows of a summary table (none when the run has neither cancers nor benign lesions)."""
    r, method = m["risk"], f"bootstrap{over}" if over is not None else "Wilson"
    rows = []
    for title, share, count, of in (("Malignancy sensitivity: cancers answered as a cancer", "malignancy_sensitivity", "detected", "cancers"),
                                    ("Missed malignancy: cancers answered as benign", "missed_malignancy", "missed", "cancers"),
                                    ("Benign specificity: benign lesions answered as benign", "benign_specificity", "cleared", "benign"),
                                    ("Over-calls: benign lesions answered as cancer", "overcall", "overcalled", "benign")):
        if r[of]:
            rows.append(f"| {title} | {_pct(r[share])} ({r[count]}/{r[of]}) | {_ci(r[f'{share}_ci95'])} ({method}) |")
    return rows


def _brief(m: dict, kind: Dataset) -> list[str]:
    """The main scores of a second scoring, as a short table: balanced accuracy, accuracy, macro-F1 and the clinical risk."""
    items, groups = f"{kind.item}s", f"{kind.group}s"
    grouped = m["interval_method"] == f"bootstrap over {groups}"
    over = f" over {groups}" if grouped else ""
    counted = [c for c, v in m["per_class"].items() if v["support"]]
    return ["| Metric | Value | 95% interval |", "|---|---|---|",
            f"| Balanced accuracy ({len(counted)} classes) | {_share(m['balanced_accuracy'])} | "
            f"{_ci(m['balanced_accuracy_ci95'])} (bootstrap{over}) |",
            f"| Accuracy | {_pct(m['accuracy'])} ({m['correct']}/{m['n']}) | {_ci(m['accuracy_ci95'])} "
            f"({f'bootstrap{over}' if grouped else 'Wilson'}) |",
            f"| Macro-F1 (classes with {items}: {len(counted)}) | {_share(m['macro']['f1'])} | {_ci(m['macro']['f1_ci95'])} "
            f"(bootstrap{over}) |", *_hallucination_rows(m, over if grouped else None), *_risk_rows(m, over if grouped else None)]


def _without(set_apart: dict, kind: Dataset, n_labels: int) -> list[str]:
    """The report lines of the scores without the classes too small to weigh like the others, and those classes' files."""
    m, left_out, files = set_apart["metrics"], set_apart["left_out"], set_apart["files"]
    items = f"{kind.item}s"
    lines = [f"## Without {' and '.join(left_out)}", ""]
    for label in left_out:
        n = sum(f["label"] == label for f in files)
        values = sorted({round(100 * k / n) for k in range(n + 1)})
        lines.append(f"The class {label} has only {n} {kind.item if n == 1 else items} in this split, so its recall can only be "
                     f"{', '.join(f'{v}%' for v in values[:-1])} or {values[-1]}%, yet it weighs as much as any other class in the "
                     f"balanced accuracy above (one class in {n_labels}).")
    lines += ["", f"The scores below leave out the {items} of {' and '.join(left_out)}, to show how much "
              f"{'they move' if len(left_out) > 1 else 'it moves'} the result. They are shown beside the main score, never "
              f"instead of it. The answer options stay the same {n_labels} classes, so answering {' or '.join(left_out)} for another "
              f"{kind.item} still counts as wrong.", "", *_brief(m, kind), "",
              f"The {items} left out, and the answers:", "", f"| {kind.item.capitalize()} | True class | Answer |", "|---|---|---|"]
    lines += [f"| `{f['path']}` | {f['label']} | {f['prediction'] or UNREADABLE} |" for f in files]
    return lines + [""]


def write_report(run_dir: str | Path) -> Path:
    """Writes metrics.json and report.md from a run's saved answers."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    preds = read_jsonl(run_dir / "predictions.jsonl")
    labels = card["labels"]
    kind = DATASETS[dataset_of(card)]

    def scores(rows):
        return summarize([p["label"] for p in rows], [p["prediction"] for p in rows], labels,
                         [p.get(kind.group_column) or p["path"] for p in rows], kind.group)

    metrics = {"main": scores(preds)}
    left_out = [label for label in kind.set_apart if any(p["label"] == label for p in preds)]
    if left_out:  # a class too small to weigh like the others: the scores without it, beside the main score
        metrics["without_small_classes"] = {
            "left_out": left_out, "metrics": scores([p for p in preds if p["label"] not in left_out]),
            "files": [{"path": p["path"], "label": p["label"], "prediction": p["prediction"]}
                      for p in sorted(preds, key=lambda p: p["path"]) if p["label"] in left_out]}
    if kind.facts_column:  # a missing fact points to the photo's collection, and so to the class: the questions with all facts
        complete_facts = [p for p in preds if MISSING_FACT not in p["facts"]]
        metrics["all_facts_recorded"] = {"questions": len(complete_facts), "metrics": scores(complete_facts)}
    # Sensitivity analysis: an unreadable answer that ends with a class name alone on its last line (after some reasoning), read as
    # that class. An answer cut off at the output limit is never read this way: its last line is not what the model meant to say.
    last_line = {p["path"]: parse_last_line(p["raw_text"], labels) for p in preds
                 if p["prediction"] is None and p["finish_reason"] != "MAX_TOKENS"}
    last_line = {path: answer for path, answer in last_line.items() if answer}
    if last_line:
        reread = [{**p, "prediction": last_line.get(p["path"], p["prediction"])} for p in preds]
        metrics["last_line_reading"] = {"answers_reread": len(last_line),
                                        "right_among_them": sum(p["label"] == p["prediction"] for p in reread if p["path"] in last_line),
                                        "metrics": scores(reread)}
    write_json(run_dir / "metrics.json", metrics)

    n_files = card[f"n_{kind.item}s"]
    complete = card["status"] == "complete" and len(preds) == n_files
    lines = [f"# {card['name']}: {card['split']} split, run {card['run_id']}", ""]
    if card["split"] == "val":
        lines += ["**Validation run: for tuning the method, not a reported result.**", ""]
    if not complete:
        lines += [f"**Incomplete: {len(preds)} of {n_files} {kind.item}s answered.**", ""]
    model = card.get("model") or {}
    table = card["dataset_table"]
    lines += [f"- Model requested: `{model.get('model_id')}`" + (f" (revision `{model['model_revision']}`)" if model.get("model_revision") else "")
              + "; versions returned by the API: "
              f"{', '.join(f'`{v}`' for v in card['model_versions_returned']) or 'not recorded yet'}",
              f"- Generation settings: `{json.dumps(model.get('generation_config'))}`",
              f"- Started {card['started_utc']}, finished {card['finished_utc']} (UTC)",
              f"- Prompt: `{card['prompt_file']}` (SHA-256 `{card['prompt_sha256'][:12]}`); {kind.item}s: {n_files} "
              f"(hash `{card[f'{kind.item}s_sha256'][:12]}`), "
              + ("each put into the prompt word for word" if kind.in_prompt else "sent exactly as stored"
                 + (", each with its patient's facts in the prompt" if kind.facts_column else ""))
              + f"; dataset table `{table['file']}` (SHA-256 `{table['sha256'][:12]}`); Git commit "
              f"`{(card['git']['commit'] or 'unknown')[:10]}`", ""]
    lines += _scoring("Results", metrics["main"], kind)
    if left_out:
        lines += _without(metrics["without_small_classes"], kind, len(labels))
    if kind.facts_column:
        part = metrics["all_facts_recorded"]
        lines += ["## Questions with all facts recorded", "",
                  f"A missing fact (\"{MISSING_FACT}\") points to the photo's collection, and so mostly to melanoma or nevus "
                  f"(data/image_metadata/datasheet.md, limitation 1). These scores count only the {part['questions']} of {len(preds)} "
                  "questions with every fact recorded, beside the main score, never instead of it.", "", *_brief(part["metrics"], kind), ""]

    if last_line:
        part = metrics["last_line_reading"]
        lines += ["## Sensitivity analysis: the last line of the answer", "",
                  f"The main score reads an answer only when it is exactly one class name, as the prompt asks. {part['answers_reread']} "
                  f"{'answer wrote its' if part['answers_reread'] == 1 else 'answers wrote their'} reasoning first and then a class name "
                  f"alone on the last line; {part['right_among_them']} of them {'names' if part['right_among_them'] == 1 else 'name'} the "
                  f"right class. Here {'it is' if part['answers_reread'] == 1 else 'they are'} read as that class. This is a check on "
                  "the main score, never a replacement for it.", "", *_brief(part["metrics"], kind), ""]

    sessions = card.get("sessions", [])
    commits = sorted({s["commit"] or "unknown" for s in sessions})
    revisions = sorted({s["model_revision"] for s in sessions if s.get("model_revision")})
    finish = Counter(p["finish_reason"] for p in preds)
    other = {k: v for k, v in finish.items() if k not in ("STOP", "MAX_TOKENS")}
    cut = [p for p in preds if p["finish_reason"] == "MAX_TOKENS"]
    harmful = sum(can_still_become_readable(p["raw_text"], labels) for p in cut)  # the limit may have decided these
    lines += ["## Notes", "",
              f"- How answers ended (finish reason): {', '.join(f'{k}: {v}' for k, v in sorted(finish.items(), key=str))}",
              *([f"- **{harmful} answers were cut off at the output limit (finish reason MAX_TOKENS) and count as wrong: "
                 "the output limit in the model list (max_tokens) is too low for this model (hidden reasoning can count toward "
                 "it). Results are not valid until it is raised, under a new model entry.**"] if harmful else []),
              *([f"- {len(cut) - harmful} answers reached the output limit but already differed from every class name before it "
                 "(an essay, or a second class name), so the limit did not change their score; they count as wrong."]
                if len(cut) > harmful else []),
              *([f"- **{sum(other.values())} answers did not end normally ("
                 + ", ".join(f"{k}: {v}" for k, v in sorted(other.items(), key=str))
                 + "); an answer without readable text counts as wrong. Check them in predictions.jsonl.**"] if other else []),
              f"- Sessions (starts and restarts of this run): {len(sessions)}; Git commits: "
              f"{', '.join(c[:10] for c in commits) or 'not recorded'}"
              + ("; **the run spanned more than one commit, so its answers do not all come from the same code**" if len(commits) > 1 else ""),
              *([f"- Model revision at the start of each session: {', '.join(f'`{r}`' for r in revisions)}"
                 + ("; **the model changed while the run was stopped, so its answers do not all come from the same model**"
                    if len(revisions) > 1 else "")] if revisions else []),
              f"- Interruptions before the run finished: {len(card.get('interruptions', []))}"
              + "".join(f"; {i['utc']} after {i['answered']} answers ({i['kind']})" for i in card.get("interruptions", []))]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------------------------------------------------------------
# Two runs side by side (python -m src.evaluate compare): one model or method against another, on the same questions


def _finished(run_dir: Path) -> tuple[dict, list[dict]]:
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    preds = read_jsonl(run_dir / "predictions.jsonl")
    kind = DATASETS[dataset_of(card)]
    if card["status"] != "complete" or len(preds) != card[f"n_{kind.item}s"]:
        raise ValueError(f"{run_dir.as_posix()} is not a finished run: only finished runs are compared")
    return card, preds


def compare(run_a: str | Path, run_b: str | Path) -> dict:
    """How run B differs from run A on the same questions: McNemar's exact test; the differences (B minus A) in balanced accuracy,
    accuracy and class-balanced hallucination rate, and the hallucination reduction (A minus B, as a share of A), with paired 95%
    intervals; and the hallucinations B removed and introduced."""
    (card_a, preds_a), (card_b, preds_b) = _finished(Path(run_a)), _finished(Path(run_b))
    dataset, dataset_b = dataset_of(card_a), dataset_of(card_b)
    kind = DATASETS[dataset]
    # The one pair of different datasets that can be compared: the same photos without and with their patients' facts (the
    # image + metadata questions are the image dataset's photos). Then the questions differ by the facts, so the photos and their
    # classes are compared instead of the questions' fingerprint.
    # The case datasets ask some of the text dataset's cases with their photos: a text run is compared on those cases only.
    same_photos = {dataset, dataset_b} == SAME_QUESTIONS[0]
    same_cases = dataset != dataset_b and {dataset, dataset_b} <= SAME_QUESTIONS[1]
    if dataset != dataset_b and not (same_photos or same_cases):
        raise ValueError("the two runs differ in their dataset, so they cannot be compared question by question")
    checks = [("split", card_a["split"], card_b["split"]), ("answer options", card_a["labels"], card_b["labels"])]
    if not (same_photos or same_cases):
        checks.append(("questions", card_a[f"{kind.item}s_sha256"], card_b[f"{kind.item}s_sha256"]))
    for what, a, b in checks:
        if a != b:
            raise ValueError(f"the two runs differ in their {what}, so they cannot be compared question by question")
    if same_cases:
        shared = {p["path"] for p in preds_a} & {p["path"] for p in preds_b}
        smaller = min(({p["path"] for p in preds_a}, {p["path"] for p in preds_b}), key=len)
        if shared != smaller:
            raise ValueError("the cases of one run are not all among the other run's cases")
        preds_a, preds_b = [p for p in preds_a if p["path"] in shared], [p for p in preds_b if p["path"] in shared]
    if (same_photos or same_cases) and {(p["path"], p["label"]) for p in preds_a} != {(p["path"], p["label"]) for p in preds_b}:
        raise ValueError(f"the two runs did not answer the same {'photos' if same_photos else 'cases'} with the same classes")
    by_path = {p["path"]: p for p in preds_b}
    if set(by_path) != {p["path"] for p in preds_a} or len(by_path) != len(preds_b):
        raise ValueError("the two runs did not answer the same questions")
    pairs = [(p, by_path[p["path"]]) for p in sorted(preds_a, key=lambda p: p["path"])]
    labels = card_a["labels"]
    y_true = [a["label"] for a, _ in pairs]
    pred_a, pred_b = [a["prediction"] for a, _ in pairs], [b["prediction"] for _, b in pairs]
    groups = [a.get(kind.group_column) or a["path"] for a, _ in pairs]
    right_a = [t == p for t, p in zip(y_true, pred_a)]
    right_b = [t == p for t, p in zip(y_true, pred_b)]
    only_a = sum(a and not b for a, b in zip(right_a, right_b))
    only_b = sum(b and not a for a, b in zip(right_a, right_b))
    hallucinated_a = [p is not None and p != t for t, p in zip(y_true, pred_a)]  # a wrong diagnosis named
    hallucinated_b = [p is not None and p != t for t, p in zip(y_true, pred_b)]

    def scores(sample):
        """B minus A in balanced accuracy, accuracy and balanced hallucination rate, and the reduction of the rate (None when A's
        rate is 0)."""
        true = [y_true[i] for i in sample]
        out = []
        for pred in (pred_a, pred_b):
            matrix = confusion_matrix(true, [pred[i] for i in sample], labels)
            s = _scores(matrix)
            out.append((s["macro"]["recall"], sum(s["tp"]) / s["n"], _hallucination(matrix)["balanced_rate"]))
        (bal_a, acc_a, hall_a), (bal_b, acc_b, hall_b) = out
        return bal_b - bal_a, acc_b - acc_a, hall_b - hall_a, (hall_a - hall_b) / hall_a if hall_a else None

    differences = [scores(sample) for sample in _resamples(_units(y_true, labels, groups), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)]

    def interval(k):
        series = [d[k] for d in differences]
        if None in series:  # A had no hallucination in some resample: a relative reduction is undefined there
            return None
        series.sort()
        return (_percentile(series, 0.025), _percentile(series, 0.975))

    balanced, accuracy, hallucination, reduction = scores(range(len(pairs)))
    return {"run_a": Path(run_a).as_posix(), "run_b": Path(run_b).as_posix(),
            "dataset": dataset if dataset == dataset_b else f"{dataset} and {dataset_b}", "split": card_a["split"],
            "n": len(pairs), "both_right": sum(a and b for a, b in zip(right_a, right_b)), "only_a": only_a, "only_b": only_b,
            "both_wrong": sum(not a and not b for a, b in zip(right_a, right_b)), "mcnemar_p": mcnemar(only_a, only_b),
            "balanced_accuracy_difference": balanced, "balanced_accuracy_difference_ci95": interval(0),
            "accuracy_difference": accuracy, "accuracy_difference_ci95": interval(1),
            "hallucination_difference": hallucination, "hallucination_difference_ci95": interval(2),
            "hallucination_reduction": reduction, "hallucination_reduction_ci95": interval(3),
            "hallucinations_removed": sum(a and not h and r for a, h, r in zip(hallucinated_a, hallucinated_b, right_b)),
            "hallucinations_introduced": sum(h and r for h, r in zip(hallucinated_b, right_a)), "groups": f"{kind.group}s"}


def comparison_text(c: dict) -> str:
    """A comparison as a short Markdown report."""
    def points(x):
        return f"{100 * x:+.1f} points"

    return "\n".join([
        f"# {c['run_b']} against {c['run_a']}", "",
        f"{c['dataset']} dataset, {c['split']} split, {c['n']} questions answered by both runs.", "",
        "| | Run A right | Run A wrong |", "|---|---|---|",
        f"| **Run B right** | {c['both_right']} | {c['only_b']} |", f"| **Run B wrong** | {c['only_a']} | {c['both_wrong']} |", "",
        f"- McNemar's exact test: p = {c['mcnemar_p']:.4f} ({c['only_b'] + c['only_a']} questions only one run got right). "
        + ("Below 0.05: the two runs differ by more than chance." if c["mcnemar_p"] < 0.05
           else "Not below 0.05: the difference may be chance."),
        f"- Balanced accuracy, B minus A: {points(c['balanced_accuracy_difference'])} (95% interval "
        f"{points(c['balanced_accuracy_difference_ci95'][0])} to {points(c['balanced_accuracy_difference_ci95'][1])}, paired "
        f"bootstrap over {c['groups']})",
        f"- Accuracy, B minus A: {points(c['accuracy_difference'])} (95% interval {points(c['accuracy_difference_ci95'][0])} to "
        f"{points(c['accuracy_difference_ci95'][1])})",
        f"- Hallucination rate (a wrong diagnosis named; mean over the classes), B minus A: {points(c['hallucination_difference'])} "
        f"(95% interval {points(c['hallucination_difference_ci95'][0])} to {points(c['hallucination_difference_ci95'][1])})",
        "- Hallucination reduction, A minus B as a share of A: " + (
            "undefined (run A has no hallucination)" if c["hallucination_reduction"] is None else
            f"{100 * c['hallucination_reduction']:+.1f}%" + (
                f" (95% interval {100 * c['hallucination_reduction_ci95'][0]:+.1f}% to "
                f"{100 * c['hallucination_reduction_ci95'][1]:+.1f}%)" if c["hallucination_reduction_ci95"] else
                " (no interval: run A has no hallucination in some resamples)")),
        f"- Hallucinations removed (A named a wrong diagnosis, B is right): {c['hallucinations_removed']}; introduced (A was right, "
        f"B named a wrong diagnosis): {c['hallucinations_introduced']}", ""])


# ---------------------------------------------------------------------------------------------------------------------------------
# All finished test runs side by side (python -m src.evaluate summary): the comparison tables of docs/results.md


SUMMARY = Path("docs") / "results.md"
SUMMARY_START, SUMMARY_END = "<!-- tables: start -->", "<!-- tables: end -->"  # between these, the document is written by "summary"
MODEL_NAMES = {"claude-sonnet-5-5": "Claude Sonnet 5.5", "gpt-4o": "GPT-4o", "gemini-3.8-flash": "Gemini 3.8 Flash",
               "llama-4-maverick": "Llama 4 Maverick"}  # the order of the tables; a model not named here follows, under its folder name


def holm(p_values: list[float]) -> list[float]:
    """Holm's step-down correction for several tests: the smallest p-value is multiplied by the number of tests, the next by one
    less, and so on; no adjusted value may be smaller than the one before it or larger than 1. Returned in the order given."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, highest = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        highest = max(highest, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = highest
    return adjusted


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _one(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _range(pair) -> str:
    return "" if pair is None else f" ({100 * pair[0]:.1f} to {100 * pair[1]:.1f})"


def _from_results(folder: Path) -> str:
    """A run folder's path from the repository's results folder on (results/...), wherever the repository is."""
    parts = Path(folder).parts
    return "/".join(parts[len(parts) - 1 - parts[::-1].index(RESULTS.name):])


def summary_tables(results: Path = RESULTS, link_prefix: str = "../") -> list[str]:
    """The comparison tables of every finished test run under ``results``, as Markdown lines: the scores per dataset,
    the clinical risk, the recall per class, the photos without and with their facts, and model against model, with McNemar's
    exact test and its Holm correction over all the tests of the page. Only saved answers are read."""
    runs = {}  # (dataset, model) -> (folder, main metrics)
    for card_path in sorted(Path(results).rglob("run_card.json")):
        folder = card_path.parent
        if not folder.name.startswith("test_"):
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["status"] != "complete":
            continue
        dataset, model = dataset_of(card), card["name"]
        if (dataset, model) in runs:
            raise ValueError(f"two finished test runs of {model} on the {dataset} dataset: {runs[dataset, model][0].name} and {folder.name}")
        runs[dataset, model] = (folder, json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["main"])
    if not runs:
        raise ValueError(f"no finished test run under {Path(results).as_posix()}")
    datasets = [d for d in DATASETS if any(key[0] == d for key in runs)]
    found = {key[1] for key in runs}
    models = [m for m in MODEL_NAMES if m in found] + sorted(found - set(MODEL_NAMES))
    name = lambda model: MODEL_NAMES.get(model, model)  # noqa: E731
    in_dataset = lambda dataset: [m for m in models if (dataset, m) in runs]  # noqa: E731

    tests = []  # every significance test of the page, so that the Holm correction covers them all
    for model in models:
        if ("image", model) in runs and ("image_metadata", model) in runs:
            tests.append(("facts", model, None, compare(runs["image", model][0], runs["image_metadata", model][0])))
    for dataset in datasets:
        present = in_dataset(dataset)
        for i, first in enumerate(present):
            for second in present[i + 1:]:
                tests.append((dataset, first, second, compare(runs[dataset, first][0], runs[dataset, second][0])))
    corrected = holm([c["mcnemar_p"] for *_, c in tests])

    lines = ["### Scores", "",
             "Percentages. Macro recall (balanced accuracy) is the main score, with its 95% interval. Each model name links to "
             "the run's report: every score with its interval, the per-class table and the confusion matrix.", ""]
    for dataset in datasets:
        lines += [f"**{TITLES[dataset]}**", "",
                  "| Model | Accuracy | Macro Recall (95% CI) | Macro Precision | Macro F1 | Macro Specificity | Hallucination Rate | No Valid Answer |",
                  "|---|---|---|---|---|---|---|---|"]
        for model in in_dataset(dataset):
            folder, m = runs[dataset, model]
            lines.append(f"| [{name(model)}]({link_prefix}{_from_results(folder)}/report.md) | {_one(m['accuracy'])} | "
                         f"{_one(m['balanced_accuracy'])}{_range(m['balanced_accuracy_ci95'])} | {_one(m['macro']['precision'])} | "
                         f"{_one(m['macro']['f1'])} | {_one(m['macro']['specificity'])} | {_one(m['hallucination']['balanced_rate'])} | "
                         f"{m['hallucination']['no_valid_answer']} of {m['n']} |")
        lines.append("")
    lines += ["### Clinical risk", "",
              "Percentages with 95% intervals. Sensitivity and missed malignancy are shares of the cancers; benign specificity and "
              "over-call are shares of the benign lesions.", ""]
    for dataset in datasets:
        lines += [f"**{TITLES[dataset]}**", "",
                  "| Model | Malignancy Sensitivity | Missed Malignancy | Benign Specificity | Over-call |", "|---|---|---|---|---|"]
        for model in in_dataset(dataset):
            r = runs[dataset, model][1]["risk"]
            lines.append(f"| {name(model)} | " + " | ".join(f"{_one(r[k])}{_range(r[f'{k}_ci95'])}" for k in (
                "malignancy_sensitivity", "missed_malignancy", "benign_specificity", "overcall")) + " |")
        lines.append("")
    lines += ["### Recall per class", "",
              "Percentages: of the questions of a class, the share answered with it. Precision, F1 and specificity per class are in each report.", ""]
    for dataset in datasets:
        present = in_dataset(dataset)
        per = {model: runs[dataset, model][1]["per_class"] for model in present}
        lines += [f"**{TITLES[dataset]}**", "", "| Class | Test questions | " + " | ".join(name(m) for m in present) + " |",
                  "|---|---|" + "---|" * len(present)]
        for label, c in per[present[0]].items():
            lines.append(f"| {label} | {c['support']} | " + " | ".join(_one(per[m][label]["recall"]) for m in present) + " |")
        lines.append("")

    def difference(c):
        low, high = c["balanced_accuracy_difference_ci95"]
        return f"{100 * c['balanced_accuracy_difference']:+.1f} ({100 * low:+.1f} to {100 * high:+.1f})"

    facts = [(model, c, q) for (kind, model, _, c), q in zip(tests, corrected) if kind == "facts"]
    if facts:
        lines += ["### The same photos without and with the patient's facts", "",
                  "Balanced accuracy in percent; the difference is image + metadata minus image in percentage points, with a paired 95% interval. "
                  "p: McNemar's exact test; Holm p: the same, corrected for all the tests of this page.", "",
                  "| Model | Image | Image + metadata | Difference (95% CI) | p | Holm p | Wrong answers corrected / introduced |", "|---|---|---|---|---|---|---|"]
        for model, c, q in facts:
            lines.append(f"| {name(model)} | {_one(runs['image', model][1]['balanced_accuracy'])} | {_one(runs['image_metadata', model][1]['balanced_accuracy'])} | "
                         f"{difference(c)} | {_p(c['mcnemar_p'])} | {_p(q)} | {c['hallucinations_removed']} / {c['hallucinations_introduced']} |")
        lines.append("")
    lines += ["### Model against model", "",
              "The difference in balanced accuracy, second model minus first, in percentage points with a paired 95% interval, on the same "
              "questions. p: McNemar's exact test on right against wrong answers; Holm p: corrected for all the tests of this page "
              f"({len(tests)}). A difference counts as significant when its Holm p is below 0.05.", ""]
    for dataset in datasets:
        lines += [f"**{TITLES[dataset]}**", "", "| Second minus first | Difference (95% CI) | p | Holm p |", "|---|---|---|---|"]
        for (kind, first, second, c), q in zip(tests, corrected):
            if kind == dataset:
                lines.append(f"| {name(second)} − {name(first)} | {difference(c)} | {_p(c['mcnemar_p'])} | {_p(q)} |")
        lines.append("")
    return lines[:-1]


def write_summary(path: Path = SUMMARY, results: Path = RESULTS / "phase1") -> Path:
    """Rewrites the tables of the results document between its two markers and leaves the written text around them as it is."""
    text = Path(path).read_text(encoding="utf-8")
    if text.count(SUMMARY_START) != 1 or text.count(SUMMARY_END) != 1 or text.index(SUMMARY_START) > text.index(SUMMARY_END):
        raise ValueError(f"{Path(path).as_posix()} must hold the lines {SUMMARY_START} and {SUMMARY_END}, once each and in this order")
    before, after = text[:text.index(SUMMARY_START)], text[text.index(SUMMARY_END) + len(SUMMARY_END):]
    tables = "\n".join(summary_tables(results, "../" * len(Path(path).parent.parts)))
    Path(path).write_text(f"{before}{SUMMARY_START}\n\n{tables}\n\n{SUMMARY_END}{after}", encoding="utf-8", newline="\n")
    return Path(path)


# ---------------------------------------------------------------------------------------------------------------------------------
# The command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="rewrite metrics.json and report.md from a run's saved answers")
    rep.add_argument("run_dir")
    cmp = sub.add_parser("compare", help="compare two finished runs on the same questions (McNemar's test, paired intervals)")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    sub.add_parser("summary", help="rewrite the comparison tables of docs/results.md from the finished test runs of the first phase")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "report":
            print(write_report(args.run_dir))
        elif args.command == "summary":
            print(write_summary())
        else:
            print(comparison_text(compare(args.run_a, args.run_b)))
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
