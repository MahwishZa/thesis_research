"""Scores of finished runs, from their saved answers only: no model is asked.

    python -m src.evaluate report results/<dataset>/<approach>/<model>/<run-folder>
    python -m src.evaluate compare results/<dataset>/<approach>/<model>/<run-folder> results/<dataset>/<approach>/<model>/<run-folder>
    python -m src.evaluate summary

A run folder of the first phase is results/phase1/<dataset>/<model>/<run-folder>.

``report`` writes a run's scores from its saved answers: metrics.json and report.md for a run of the current answer format; for a
run of the first phase, whose answers are single class names, scores.json and scores.md beside the archived files, which are never
changed. The metrics and their sources are in docs/method.md.
``compare`` sets two finished runs on the same questions side by side (two models, or two variants of a method in an ablation):
the difference in balanced accuracy and in macro-F1 with paired 95% intervals (both runs resampled together, by the same groups
as the reports; the main comparison), and McNemar's exact test on the questions one run gets right and the other wrong (a
supporting test).
``summary`` rewrites the tables of docs/results.md from every finished test run of the first phase (results/phase1/). The text of
the document around the tables is written by hand and is left as it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .common import read_jsonl, write_json
from .datasets import DATASETS, MISSING_FACT, SAME_QUESTIONS, TITLES, Dataset, dataset_of
from .metrics import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, INVALID, _resamples, _scores, _units, confusion_matrix, holm, interval,
                      mcnemar, summarize, within_groups)
from .prompts import ANSWER_FORMAT, can_still_become_readable, parse_last_line

RESULTS = Path("results")  # results/<dataset>/<approach>/<model>/<run>/; the first phase in results/phase1/
SUMMARY = Path("docs") / "results.md"
SUMMARY_START, SUMMARY_END = "<!-- tables: start -->", "<!-- tables: end -->"  # between these, the document is written by "summary"
MODEL_NAMES = {"claude-sonnet-5-5": "Claude Sonnet 5.5", "gpt-4o": "GPT-4o", "gemini-3.8-flash": "Gemini 3.8 Flash",
               "llama-4-maverick": "Llama 4 Maverick"}  # the order of the tables; a model not named here follows, under its folder name
# The photos come from two ISIC collections that hold different classes (data/image/datasheet.md): melanoma and nevus from one,
# the other six from the other, so a model could tell the two groups apart by the collection alone. Each run on these photos is
# also scored inside each group.
COLLECTIONS = {dataset: [("melanoma", "nevus"), ("actinic keratosis", "basal cell carcinoma", "dermatofibroma",
                                                 "pigmented benign keratosis", "squamous cell carcinoma", "vascular lesion")]
               for dataset in ("image", "image_metadata")}


def current_format(card: dict) -> bool:
    """True for a run of the current answer format; False for a run of the first phase (single class names)."""
    return card.get("answer_format") == ANSWER_FORMAT


def output_files(card: dict) -> tuple[str, str]:
    """Where a run's scores go: metrics.json and report.md, or, beside a first-phase run's archived files, scores.json and
    scores.md."""
    return ("metrics.json", "report.md") if current_format(card) else ("scores.json", "scores.md")


# ---------------------------------------------------------------------------------------------------------------------------------
# The scores of one run


def score_run(run_dir: str | Path) -> tuple[dict, list[dict], dict]:
    """A run's card, saved answers and every score (computed, nothing written)."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    preds = read_jsonl(run_dir / "predictions.jsonl")
    labels, kind, current = card["labels"], DATASETS[dataset_of(card)], current_format(card)

    def scores(rows):
        return summarize([p["label"] for p in rows], [p["prediction"] for p in rows], labels,
                         [p.get(kind.group_column) or p["path"] for p in rows], kind.group,
                         [p.get("differential") for p in rows] if current else None,
                         [p.get("evidence_sufficient") for p in rows] if current else None)

    metrics = {"answer_format": card.get("answer_format", "class name (first phase)"), "main": scores(preds)}
    left_out = [label for label in kind.set_apart if any(p["label"] == label for p in preds)]
    if left_out:  # a class too small to weigh like the others: the scores without it, beside the main score
        metrics["without_small_classes"] = {
            "left_out": left_out, "metrics": scores([p for p in preds if p["label"] not in left_out]),
            "files": [{"path": p["path"], "label": p["label"], "prediction": p["prediction"]}
                      for p in sorted(preds, key=lambda p: p["path"]) if p["label"] in left_out]}
    if kind.facts_column:  # a missing fact points to the photo's collection, and so to the class: the questions with all facts
        complete_facts = [p for p in preds if MISSING_FACT not in p["facts"]]
        metrics["all_facts_recorded"] = {"questions": len(complete_facts), "metrics": scores(complete_facts)}
    if dataset_of(card) in COLLECTIONS:
        metrics["within_collections"] = within_groups([p["label"] for p in preds], [p["prediction"] for p in preds],
                                                      COLLECTIONS[dataset_of(card)])
    if not current:
        # Sensitivity analysis of the first phase: an unreadable answer that ends with a class name alone on its last line (after
        # some reasoning), read as that class. An answer cut off at the output limit is never read this way.
        last_line = {p["path"]: parse_last_line(p["raw_text"], labels) for p in preds
                     if p["prediction"] is None and p["finish_reason"] != "MAX_TOKENS"}
        last_line = {path: answer for path, answer in last_line.items() if answer}
        if last_line:
            reread = [{**p, "prediction": last_line.get(p["path"], p["prediction"])} for p in preds]
            metrics["last_line_reading"] = {"answers_reread": len(last_line),
                                            "right_among_them": sum(p["label"] == p["prediction"] for p in reread if p["path"] in last_line),
                                            "metrics": scores(reread)}
    return card, preds, metrics


# ---------------------------------------------------------------------------------------------------------------------------------
# The report of one run (python -m src.evaluate report)


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _ci(pair) -> str:
    return "–" if pair is None else f"{_pct(pair[0])} to {_pct(pair[1])}"


def _count(k: int, n: int) -> str:
    return f"{k}/{n}"


def _main_rows(m: dict) -> list[str]:
    """The summary table of one scoring: balanced accuracy, macro-F1, top-3 balanced accuracy and the invalid answers."""
    method = f"bootstrap over {m['group_name']}s" if m["interval_method"].startswith("bootstrap over") else "bootstrap"
    return ["| Score | Value | 95% interval |", "|---|---|---|",
            f"| **Balanced accuracy (main score)** | **{_pct(m['balanced_accuracy'])}** | {_ci(m['balanced_accuracy_ci95'])} ({method}) |",
            f"| Macro-F1 | {_pct(m['macro_f1'])} | {_ci(m['macro_f1_ci95'])} ({method}) |",
            *([f"| Top-3 balanced accuracy (the true class in the differential) | {_pct(m['top3_balanced_accuracy'])} | "
               f"{_ci(m['top3_balanced_accuracy_ci95'])} ({method}) |"] if m["top3_balanced_accuracy"] is not None else []),
            f"| Invalid answers (counted as wrong) | {_pct(m['invalid_rate'])} ({_count(m['invalid'], m['n'])}) | |"]


def _detection_rows(m: dict) -> list[str]:
    d = m["detection"]
    if d is None:
        return []
    return ["### Misdiagnosis detection and selective prediction", "",
            f"Over the {d['valid']} valid answers. The positive case is a wrong diagnosis; the model marks it by answering "
            "\"evidence_sufficient\": \"no\".", "", "| Score | Value | 95% interval |", "|---|---|---|",
            f"| Sensitivity: wrong diagnoses marked \"not sufficient\" | {_pct(d['sensitivity'])} ({_count(d['flagged_wrong'], d['wrong'])}) | {_ci(d['sensitivity_ci95'])} |",
            f"| Specificity: right diagnoses not marked | {_pct(d['specificity'])} ({_count(d['unflagged_right'], d['right'])}) | {_ci(d['specificity_ci95'])} |",
            f"| Coverage: questions answered with \"sufficient\" | {_pct(d['coverage'])} ({_count(d['covered'], m['n'])}) | {_ci(d['coverage_ci95'])} |",
            f"| Selective risk: wrong among them | {_pct(d['selective_risk'])} ({_count(d['covered_wrong'], d['covered'])}) | {_ci(d['selective_risk_ci95'])} |", ""]


def _class_rows(m: dict, kind: Dataset) -> list[str]:
    counted = [c for c, v in m["per_class"].items() if v["support"]]
    lines = ["### Per class", "", f"| Class | {kind.item.capitalize()}s | Recall (95% interval) | Precision | F1 |", "|---|---|---|---|---|"]
    for label, c in m["per_class"].items():
        if c["support"] or c["answered"]:
            lines.append(f"| {label} | {c['support']} | {_pct(c['recall'])} ({_ci(c['recall_ci95'])}) | {_pct(c['precision'])} | "
                         f"{_pct(c['f1'])} |")
    rows, columns, counts = m["confusion_matrix"]["rows"], m["confusion_matrix"]["columns"], m["confusion_matrix"]["counts"]
    lines += ["", "### Confusion matrix", "", "Rows: the true class; columns: the diagnosis given.", "",
              "| True \\ Answer | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    lines += [f"| {r} | " + " | ".join(str(v) for v in row) + " |" for r, row in zip(rows, counts) if r in counted]
    return lines + [""]


def _scoring(title: str, m: dict, kind: Dataset) -> list[str]:
    absent = m["classes_absent"]
    return [f"## {title}", "", *_main_rows(m), "", *_detection_rows(m), *_class_rows(m, kind),
            f"- {kind.why_grouped}: the intervals resample whole {kind.group}s ({m['groups']} {kind.group}s for {m['n']} {kind.item}s)."
            if m["interval_method"].startswith("bootstrap over") else
            f"- No two {kind.item}s share one {kind.group}, so every {kind.item} counts as independent.",
            *([f"- No {kind.item} has the true class {', '.join(absent)}: it is left out of the averages."] if absent else []),
            "- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.",
            "- The definitions and their sources are in docs/method.md.", ""]


def _brief(m: dict) -> list[str]:
    return _main_rows(m)


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
              f"{kind.item} still counts as wrong.", "", *_brief(m), "",
              f"The {items} left out, and the answers:", "", f"| {kind.item.capitalize()} | True class | Answer |", "|---|---|---|"]
    lines += [f"| `{f['path']}` | {f['label']} | {f['prediction'] or INVALID} |" for f in files]
    return lines + [""]


def write_report(run_dir: str | Path) -> Path:
    """Writes a run's scores and report (see the module's description); returns the report's path."""
    run_dir = Path(run_dir)
    card, preds, metrics = score_run(run_dir)
    labels, kind = card["labels"], DATASETS[dataset_of(card)]
    scores_file, report_file = output_files(card)
    write_json(run_dir / scores_file, metrics)

    n_files = card[f"n_{kind.item}s"]
    complete = card["status"] == "complete" and len(preds) == n_files
    lines = [f"# {card['name']}: {card['split']} split, run {card['run_id']}", ""]
    if not current_format(card):
        lines += ["**A run of the first phase: its answers are single class names, so the scores that need a differential or an "
                  "evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**", ""]
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
              f"- Dataset `{dataset_of(card)}`, approach `{card['config'].get('approach', 'closed_book')}`, answer format "
              f"`{metrics['answer_format']}`",
              f"- Started {card['started_utc']}, finished {card['finished_utc']} (UTC)",
              f"- Prompt: `{card['prompt_file']}` (SHA-256 `{card['prompt_sha256'][:12]}`); {kind.item}s: {n_files} "
              f"(hash `{card[f'{kind.item}s_sha256'][:12]}`), "
              + ("each put into the prompt word for word" if kind.in_prompt else "sent exactly as stored"
                 + (", each with its patient's facts in the prompt" if kind.facts_column else ""))
              + f"; dataset table `{table['file']}` (SHA-256 `{table['sha256'][:12]}`); Git commit "
              f"`{(card['git']['commit'] or 'unknown')[:10]}`", ""]
    lines += _scoring("Results", metrics["main"], kind)
    if "without_small_classes" in metrics:
        lines += _without(metrics["without_small_classes"], kind, len(labels))
    if "all_facts_recorded" in metrics:
        part = metrics["all_facts_recorded"]
        lines += ["## Questions with all facts recorded", "",
                  f"A missing fact (\"{MISSING_FACT}\") points to the photo's collection, and so mostly to melanoma or nevus "
                  f"(data/image_metadata/datasheet.md, limitation 1). These scores count only the {part['questions']} of {len(preds)} "
                  "questions with every fact recorded, beside the main score, never instead of it.", "", *_brief(part["metrics"]), ""]
    if "within_collections" in metrics:
        lines += ["## Inside each photo collection", "",
                  "Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone "
                  "could separate the two groups. Inside each group the collection gives nothing away: a score near chance there "
                  "means the model told the groups apart, not the diseases.", "",
                  "| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |", "|---|---|---|---|---|"]
        lines += [f"| {', '.join(g['classes'])} | {g['questions']} | {_pct(g['balanced_accuracy'])} | {_pct(g['chance'])} | "
                  f"{_pct(g['inside_share'])} |" for g in metrics["within_collections"]]
        lines.append("")
    if "last_line_reading" in metrics:
        part = metrics["last_line_reading"]
        lines += ["## Sensitivity analysis: the last line of the answer", "",
                  f"The main score reads an answer only when it is exactly one class name, as the prompt of the first phase asked. "
                  f"{part['answers_reread']} answers wrote their reasoning first and then a class name alone on the last line; "
                  f"{part['right_among_them']} of them name the right class. Here they are read as that class: a check on the main "
                  "score, never a replacement for it.", "", *_brief(part["metrics"]), ""]

    sessions = card.get("sessions", [])
    commits = sorted({s["commit"] or "unknown" for s in sessions})
    revisions = sorted({s["model_revision"] for s in sessions if s.get("model_revision")})
    finish = Counter(p["finish_reason"] for p in preds)
    other = {k: v for k, v in finish.items() if k not in ("STOP", "MAX_TOKENS")}
    cut = [p for p in preds if p["finish_reason"] == "MAX_TOKENS"]
    if current_format(card):  # a JSON answer cut off is incomplete, so invalid: the limit decided it
        harmful = sum(p["prediction"] is None for p in cut)
    else:
        harmful = sum(can_still_become_readable(p["raw_text"], labels) for p in cut)
    invalid = Counter(p.get("invalid") for p in preds if p["prediction"] is None and p.get("invalid"))
    lines += ["## Notes", "",
              f"- How answers ended (finish reason): {', '.join(f'{k}: {v}' for k, v in sorted(finish.items(), key=str))}",
              *(["- Why answers were invalid: " + "; ".join(f"{k}: {v}" for k, v in invalid.most_common())] if invalid else []),
              *([f"- **{harmful} answers were cut off at the output limit (finish reason MAX_TOKENS) and count as wrong: "
                 "the output limit in the model list (max_tokens) is too low for this model (hidden reasoning can count toward "
                 "it). Results are not valid until it is raised, under a new model entry.**"] if harmful else []),
              *([f"- {len(cut) - harmful} answers reached the output limit but "
                 + ("were complete and valid all the same, so the limit did not change their score."
                    if current_format(card) else
                    "already differed from every class name, so the limit did not change their score; they count as wrong.")]
                if len(cut) > harmful else []),
              *([f"- **{sum(other.values())} answers did not end normally ("
                 + ", ".join(f"{k}: {v}" for k, v in sorted(other.items(), key=str))
                 + "); an answer without valid text counts as wrong. Check them in predictions.jsonl.**"] if other else []),
              f"- Sessions (starts and restarts of this run): {len(sessions)}; Git commits: "
              f"{', '.join(c[:10] for c in commits) or 'not recorded'}"
              + ("; **the run spanned more than one commit, so its answers do not all come from the same code**" if len(commits) > 1 else ""),
              *([f"- Model revision at the start of each session: {', '.join(f'`{r}`' for r in revisions)}"
                 + ("; **the model changed while the run was stopped, so its answers do not all come from the same model**"
                    if len(revisions) > 1 else "")] if revisions else []),
              f"- Interruptions before the run finished: {len(card.get('interruptions', []))}"
              + "".join(f"; {i['utc']} after {i['answered']} answers ({i['kind']})" for i in card.get("interruptions", []))]
    path = run_dir / report_file
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
    """How run B differs from run A on the same questions: the differences (B minus A) in balanced accuracy and macro-F1, with
    paired 95% intervals (the main comparison), and McNemar's exact test on right against wrong answers (a supporting test)."""
    (card_a, preds_a), (card_b, preds_b) = _finished(Path(run_a)), _finished(Path(run_b))
    dataset, dataset_b = dataset_of(card_a), dataset_of(card_b)
    kind = DATASETS[dataset]
    # The one pair of different datasets with the same photos: without and with their patients' facts. Then the questions differ
    # by the facts, so the photos and their classes are compared instead of the questions' fingerprint. The case datasets ask some
    # of the text dataset's cases with their photos: a text run is compared on those cases only.
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

    def differences(sample):
        """B minus A in balanced accuracy and macro-F1 on the questions of a sample."""
        true = [y_true[i] for i in sample]
        a = _scores(confusion_matrix(true, [pred_a[i] for i in sample], labels))
        b = _scores(confusion_matrix(true, [pred_b[i] for i in sample], labels))
        return b["balanced_accuracy"] - a["balanced_accuracy"], b["macro"]["f1"] - a["macro"]["f1"]

    resampled = [differences(s) for s in _resamples(_units(y_true, labels, groups), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)]
    balanced, f1 = differences(range(len(pairs)))
    return {"run_a": Path(run_a).as_posix(), "run_b": Path(run_b).as_posix(),
            "dataset": dataset if dataset == dataset_b else f"{dataset} and {dataset_b}", "split": card_a["split"],
            "n": len(pairs), "groups": f"{kind.group}s",
            "balanced_accuracy_difference": balanced, "balanced_accuracy_difference_ci95": interval([d[0] for d in resampled]),
            "macro_f1_difference": f1, "macro_f1_difference_ci95": interval([d[1] for d in resampled]),
            "both_right": sum(a and b for a, b in zip(right_a, right_b)), "only_a": only_a, "only_b": only_b,
            "both_wrong": sum(not a and not b for a, b in zip(right_a, right_b)), "mcnemar_p": mcnemar(only_a, only_b)}


def comparison_text(c: dict) -> str:
    """A comparison as a short Markdown report."""
    def points(x):
        return f"{100 * x:+.1f} points"

    def beyond(ci):
        return ci is not None and (ci[0] > 0 or ci[1] < 0)

    low, high = c["balanced_accuracy_difference_ci95"]
    return "\n".join([
        f"# {c['run_b']} against {c['run_a']}", "",
        f"{c['dataset']} dataset, {c['split']} split, {c['n']} questions answered by both runs.", "",
        f"- **Balanced accuracy, B minus A: {points(c['balanced_accuracy_difference'])}** (95% interval {points(low)} to "
        f"{points(high)}, paired bootstrap over {c['groups']}). "
        + ("The interval does not hold 0: the runs differ by more than chance." if beyond(c["balanced_accuracy_difference_ci95"])
           else "The interval holds 0: the difference may be chance."),
        f"- Macro-F1, B minus A: {points(c['macro_f1_difference'])} (95% interval {points(c['macro_f1_difference_ci95'][0])} to "
        f"{points(c['macro_f1_difference_ci95'][1])})", "",
        "| | Run A right | Run A wrong |", "|---|---|---|",
        f"| **Run B right** | {c['both_right']} | {c['only_b']} |", f"| **Run B wrong** | {c['only_a']} | {c['both_wrong']} |", "",
        f"- McNemar's exact test (supporting; it counts right against wrong answers, so on unequal classes it can disagree with "
        f"the balanced accuracy): p = {c['mcnemar_p']:.4f} ({c['only_b'] + c['only_a']} questions only one run got right).", ""])


# ---------------------------------------------------------------------------------------------------------------------------------
# All finished test runs side by side (python -m src.evaluate summary): the tables of docs/results.md


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _one(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _range(pair) -> str:
    return "" if pair is None else f" ({100 * pair[0]:.1f} to {100 * pair[1]:.1f})"


def _from_results(folder: Path) -> str:
    """A run folder's path from the repository's results folder on (results/...), wherever the repository is."""
    parts = Path(folder).parts
    return "/".join(parts[len(parts) - 1 - parts[::-1].index(RESULTS.name):])


def finished_test_runs(results: Path) -> dict:
    """(dataset, model) -> (folder, scores) of every finished test run under ``results``; two runs of one pair are an error."""
    runs = {}
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
        runs[dataset, model] = (folder, score_run(folder)[2])
    if not runs:
        raise ValueError(f"no finished test run under {Path(results).as_posix()}")
    return runs


def summary_tables(results: Path = RESULTS / "phase1", link_prefix: str = "../") -> list[str]:
    """The tables of every finished test run under ``results``, as Markdown lines: the scores per dataset, the recall per class,
    the scores inside each photo collection, the photos without and with their facts, and model against model (the paired
    difference in balanced accuracy, and McNemar's exact test with Holm's correction over all the tests of the page)."""
    runs = finished_test_runs(results)
    datasets = [d for d in DATASETS if any(key[0] == d for key in runs)]
    found = {key[1] for key in runs}
    models = [m for m in MODEL_NAMES if m in found] + sorted(found - set(MODEL_NAMES))
    name = lambda model: MODEL_NAMES.get(model, model)  # noqa: E731
    in_dataset = lambda dataset: [m for m in models if (dataset, m) in runs]  # noqa: E731

    tests = []  # every comparison of the page, so that the Holm correction covers all its McNemar tests
    for model in models:
        if ("image", model) in runs and ("image_metadata", model) in runs:
            tests.append(("facts", model, None, compare(runs["image", model][0], runs["image_metadata", model][0])))
    for dataset in datasets:
        present = in_dataset(dataset)
        for i, first in enumerate(present):
            for second in present[i + 1:]:
                tests.append((dataset, first, second, compare(runs[dataset, first][0], runs[dataset, second][0])))
    corrected = holm([c["mcnemar_p"] for *_, c in tests])
    any_top3 = any(m["main"]["top3_balanced_accuracy"] is not None for _, m in runs.values())

    lines = ["### Scores", "",
             "Percentages with 95% intervals. Balanced accuracy is the main score. Each model name links to the run's scores: every "
             "score with its interval, the per-class table and the confusion matrix.", ""]
    for dataset in datasets:
        lines += [f"**{TITLES[dataset]}**", "",
                  "| Model | Balanced Accuracy (95% CI) | Macro-F1 (95% CI) |" + (" Top-3 Balanced Accuracy |" if any_top3 else "")
                  + " Invalid Answers |", "|---|---|---|" + ("---|" if any_top3 else "") + "---|"]
        for model in in_dataset(dataset):
            folder, metrics = runs[dataset, model]
            m = metrics["main"]
            scores_md = "report.md" if metrics["answer_format"] == ANSWER_FORMAT else "scores.md"
            lines.append(f"| [{name(model)}]({link_prefix}{_from_results(folder)}/{scores_md}) | "
                         f"{_one(m['balanced_accuracy'])}{_range(m['balanced_accuracy_ci95'])} | {_one(m['macro_f1'])}{_range(m['macro_f1_ci95'])} |"
                         + (f" {_one(m['top3_balanced_accuracy'])} |" if any_top3 else "") + f" {m['invalid']} of {m['n']} |")
        lines.append("")
    lines += ["### Recall per class", "",
              "Percentages: of the questions of a class, the share answered with it. Precision and F1 per class are in each run's scores.", ""]
    for dataset in datasets:
        present = in_dataset(dataset)
        per = {model: runs[dataset, model][1]["main"]["per_class"] for model in present}
        lines += [f"**{TITLES[dataset]}**", "", "| Class | Test questions | " + " | ".join(name(m) for m in present) + " |",
                  "|---|---|" + "---|" * len(present)]
        for label, c in per[present[0]].items():
            lines.append(f"| {label} | {c['support']} | " + " | ".join(_one(per[m][label]["recall"]) for m in present) + " |")
        lines.append("")
    collection_sets = [d for d in datasets if d in COLLECTIONS]
    if collection_sets:
        lines += ["### Inside each photo collection", "",
                  "Balanced accuracy inside each group of classes that comes from one ISIC collection, in percent, and in brackets "
                  "the share of the group's questions answered with a class of the same group. Chance inside the groups: 50.0 for "
                  "melanoma and nevus, 16.7 for the six others.", ""]
        for dataset in collection_sets:
            lines += [f"**{TITLES[dataset]}**", "", "| Model | Melanoma and nevus | The six other classes |", "|---|---|---|"]
            for model in in_dataset(dataset):
                groups = runs[dataset, model][1]["within_collections"]
                lines.append(f"| {name(model)} | " + " | ".join(f"{_one(g['balanced_accuracy'])} ({_one(g['inside_share'])})" for g in groups) + " |")
            lines.append("")

    def difference(c):
        low, high = c["balanced_accuracy_difference_ci95"]
        return f"{100 * c['balanced_accuracy_difference']:+.1f} ({100 * low:+.1f} to {100 * high:+.1f})"

    facts = [(model, c, q) for (kind, model, _, c), q in zip(tests, corrected) if kind == "facts"]
    if facts:
        lines += ["### The same photos without and with the patient's facts", "",
                  "Balanced accuracy in percent; the difference is image + metadata minus image in percentage points, with a paired "
                  "95% interval (the main comparison). p: McNemar's exact test; Holm p: the same, corrected for all the tests of this page.", "",
                  "| Model | Image | Image + metadata | Difference (95% CI) | p | Holm p |", "|---|---|---|---|---|---|"]
        for model, c, q in facts:
            lines.append(f"| {name(model)} | {_one(runs['image', model][1]['main']['balanced_accuracy'])} | "
                         f"{_one(runs['image_metadata', model][1]['main']['balanced_accuracy'])} | {difference(c)} | {_p(c['mcnemar_p'])} | {_p(q)} |")
        lines.append("")
    lines += ["### Model against model", "",
              "The difference in balanced accuracy, second model minus first, in percentage points with a paired 95% interval, on the same "
              "questions: the main comparison. p: McNemar's exact test on right against wrong answers; Holm p: corrected for all the "
              f"tests of this page ({len(tests)}).", ""]
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
    rep = sub.add_parser("report", help="write a run's scores and report from its saved answers")
    rep.add_argument("run_dir")
    cmp = sub.add_parser("compare", help="compare two finished runs on the same questions (paired intervals, McNemar's test)")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    sub.add_parser("summary", help="rewrite the tables of docs/results.md from the finished test runs of the first phase")
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
