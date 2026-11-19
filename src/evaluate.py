"""Scores of finished runs, from their saved answers only: no model is asked.

    python -m src.evaluate report results/<dataset>/<approach>/<model>/<run-folder>
    python -m src.evaluate compare results/<dataset>/<approach>/<model>/<run-folder> results/<dataset>/<approach>/<model>/<run-folder>
    python -m src.evaluate reread results/<dataset>/<approach>/<model>/<run-folder>

``report`` writes a run's scores from its saved answers: metrics.json and report.md. It reads the answer format of the code
(``json-v2``); a run of the first phase of Version 1, whose answers are single class names, is in archive/v1/ with its own scores and
is not read here. The metrics and their sources are in docs/method.md.
``compare`` sets two finished runs on the same questions side by side (two models, or two variants of a method in an ablation):
the difference in balanced accuracy and in macro-F1 with paired 95% intervals (both runs resampled together, by the same groups
as the reports; the main comparison), and McNemar's exact test on the questions one run gets right and the other wrong (a
supporting test); ``compare --reread`` does the same on the re-read answers (see ``reread``).
``reread`` reads the raw text of every saved answer of a closed-book run again with the answer reader of the code, and writes
predictions_reread.jsonl, metrics_reread.json and report_reread.md beside the originals, which it never touches. The reader changed on
2026-11-19, after the first test runs: a claim may leave out its empty ``evidence`` list. Answers read under the old rule and the new
one are both kept, and both scores are reported (docs/method.md, section 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .common import read_jsonl, write_json
from .datasets import DATASETS, SAME_QUESTIONS, Dataset, dataset_of, question_of
from .metrics import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, INVALID, _resamples, _scores, _units, confusion_matrix, interval, mcnemar,
                      summarize)
from .prompts import ANSWER_FORMAT, parse_json_answer

RESULTS = Path("results")  # results/<dataset>/<approach>/<model>/<run>/
REREAD_ANSWERS = "predictions_reread.jsonl"  # beside predictions.jsonl: the same answers read again by the current reader


def require_current_format(card: dict) -> None:
    """Stops for a run of another answer format: the first phase of Version 1 (single class names) is archived, with its own scores."""
    if card.get("answer_format") != ANSWER_FORMAT:
        raise ValueError(f"this run's answers are in the format {card.get('answer_format', 'class name (first phase)')!r}, not {ANSWER_FORMAT!r}: "
                         "runs of the first phase are in archive/v1/ with their own scores and are not read by this code")


ANSWER_FIELDS = ("path", "label", "prediction", "differential", "evidence_sufficient", "reason", "claims", "evidence", "invalid", "raw_text",
                 "finish_reason", "model_version")  # what every saved answer must hold (src/run.py writes them)


def verify_answers(card: dict, preds: list[dict], complete: bool = False) -> list[str]:
    """What is wrong with a run's saved answers, as a list of messages (empty: nothing). Checked before anything is scored or saved as
    finished, so a damaged or inconsistent answer file cannot give a plausible-looking score: every field is present; every question is
    answered at most once (a second answer would count twice); every true class and every diagnosis is a class of the run; a valid
    answer is consistent (the diagnosis is the first of its differential, the flag is yes or no, no evidence without retrieval) and an
    invalid one holds its reason and nothing else; with ``complete``, every question of the run is answered."""
    kind = DATASETS[dataset_of(card)]
    labels, problems = set(card["labels"]), []
    fields = (*ANSWER_FIELDS, kind.group_column)
    for number, p in enumerate(preds, start=1):
        absent = [f for f in fields if f not in p]
        if absent:
            problems.append(f"answer {number}: no field {', '.join(absent)}")
            continue
        where = f"answer {number} ({p['path']})"
        if p["label"] not in labels:
            problems.append(f"{where}: the true class {p['label']!r} is not a class of the run")
        if p["prediction"] is None:
            if not p["invalid"] or any(p[k] is not None for k in ("differential", "evidence_sufficient", "reason", "claims", "evidence")):
                problems.append(f"{where}: an invalid answer must hold its reason and no differential, flag, reason or evidence")
            continue
        if p["invalid"] is not None:
            problems.append(f"{where}: a valid answer must not hold an invalid reason")
        if not isinstance(p["differential"], list) or not p["differential"] or p["differential"][0] != p["prediction"]:
            problems.append(f"{where}: the diagnosis is not the first of the differential")
        elif not set(p["differential"]) <= labels or len(set(p["differential"])) != len(p["differential"]):
            problems.append(f"{where}: the differential is not different classes of the run")
        if p["evidence_sufficient"] not in ("yes", "no"):
            problems.append(f"{where}: the flag {p['evidence_sufficient']!r} is not yes or no")
        if card.get("config", {}).get("approach", "closed_book") == "closed_book" and p["evidence"] != []:
            problems.append(f"{where}: a closed-book answer must hold no evidence")
        if not isinstance(p["claims"], list) or not p["claims"]:
            problems.append(f"{where}: a valid answer must hold at least one claim")
        elif card.get("config", {}).get("approach", "closed_book") == "closed_book" and any(c.get("evidence") != [] for c in p["claims"]):
            problems.append(f"{where}: a closed-book claim must rely on no passage")
    paths = Counter(p["path"] for p in preds if "path" in p)
    problems += [f"{path} is answered {n} times" for path, n in sorted(paths.items()) if n > 1]
    if complete and len(paths) != card[f"n_{kind.item}s"]:
        problems.append(f"{len(paths)} questions answered, {card[f'n_{kind.item}s']} expected")
    return problems


def _verified(card: dict, preds: list[dict], complete: bool = False) -> None:
    problems = verify_answers(card, preds, complete)
    if problems:
        shown = "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else "")
        raise ValueError(f"the saved answers of this run are not consistent, so nothing is scored: {shown}")


# ---------------------------------------------------------------------------------------------------------------------------------
# The scores of one run


def score_run(run_dir: str | Path, reread: bool = False) -> tuple[dict, list[dict], dict]:
    """A run's card, saved answers (the re-read ones with ``reread``) and every score (computed, nothing written)."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    preds = read_jsonl(run_dir / (REREAD_ANSWERS if reread else "predictions.jsonl"))
    require_current_format(card)
    _verified(card, preds, complete=card["status"] == "complete")
    labels, kind = card["labels"], DATASETS[dataset_of(card)]

    def scores(rows):
        return summarize([p["label"] for p in rows], [p["prediction"] for p in rows], labels,
                         [p.get(kind.group_column) or p["path"] for p in rows], kind.group,
                         [p.get("differential") for p in rows], [p.get("evidence_sufficient") for p in rows])

    metrics = {"answer_format": card["answer_format"], "main": scores(preds)}
    left_out = [label for label in kind.set_apart if any(p["label"] == label for p in preds)]
    if left_out:  # a class too small to weigh like the others: the scores without it, beside the main score
        metrics["without_small_classes"] = {
            "left_out": left_out, "metrics": scores([p for p in preds if p["label"] not in left_out]),
            "files": [{"path": p["path"], "label": p["label"], "prediction": p["prediction"]}
                      for p in sorted(preds, key=lambda p: p["path"]) if p["label"] in left_out]}
    if kind.missing_fact:  # a missing fact points to the photo's collection, and so to the class: the questions with all facts
        complete_facts = [p for p in preds if kind.missing_fact not in p["facts"]]
        metrics["all_facts_recorded"] = {"questions": len(complete_facts), "metrics": scores(complete_facts)}
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
            *([f"| Macro-averaged top-3 accuracy (the true class in the differential) | {_pct(m['top3_balanced_accuracy'])} | "
               f"{_ci(m['top3_balanced_accuracy_ci95'])} ({method}) |"] if m["top3_balanced_accuracy"] is not None else []),
            f"| Invalid answers (counted as wrong) | {_pct(m['invalid_rate'])} ({_count(m['invalid'], m['n'])}) | |"]


def _selective_rows(m: dict) -> list[str]:
    d = m["selective"]
    if d is None:
        return []
    return ["### Selective prediction", "",
            f"Over the {d['valid']} valid answers. The model answers a question by marking its evidence \"sufficient\" and abstains by "
            "marking it \"not sufficient\" (selective classification, Geifman & El-Yaniv 2017).", "",
            "| Score | Value | 95% interval |", "|---|---|---|",
            f"| Coverage: questions answered with \"sufficient\" | {_pct(d['coverage'])} ({_count(d['covered'], m['n'])}) | {_ci(d['coverage_ci95'])} |",
            f"| Selective risk (confident misdiagnosis): wrong among them | {_pct(d['selective_risk'])} ({_count(d['covered_wrong'], d['covered'])}) | {_ci(d['selective_risk_ci95'])} |", ""]


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
    return [f"## {title}", "", *_main_rows(m), "", *_selective_rows(m), *_class_rows(m, kind),
            f"- {kind.why_grouped}: the intervals resample whole {kind.group}s ({m['groups']} {kind.group}s for {m['n']} {kind.item}s)."
            if m["interval_method"].startswith("bootstrap over") else
            f"- No two {kind.item}s share one {kind.group}, so every {kind.item} counts as independent.",
            *([f"- No {kind.item} has the true class {', '.join(absent)}: it is left out of the averages."] if absent else []),
            "- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.",
            "- The definitions and their sources are in docs/method.md.", ""]


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
              f"{kind.item} still counts as wrong.", "", *_main_rows(m), "",
              f"The {items} left out, and the answers:", "", f"| {kind.item.capitalize()} | True class | Answer |", "|---|---|---|"]
    lines += [f"| `{f['path']}` | {f['label']} | {f['prediction'] or INVALID} |" for f in files]
    return lines + [""]


def write_report(run_dir: str | Path, reread: bool = False, changed: int | None = None) -> Path:
    """Writes a run's scores and report (see the module's description); returns the report's path. With ``reread`` the scores of the
    re-read answers go to metrics_reread.json and report_reread.md, and ``changed`` is how many answers the reader read differently."""
    run_dir = Path(run_dir)
    card, preds, metrics = score_run(run_dir, reread)
    labels, kind = card["labels"], DATASETS[dataset_of(card)]
    write_json(run_dir / ("metrics_reread.json" if reread else "metrics.json"), metrics)

    n_files = card[f"n_{kind.item}s"]
    complete = card["status"] == "complete" and len(preds) == n_files
    lines = [f"# {card['name']}: {card['split']} split, run {card['run_id']}", ""]
    if reread:
        lines += [f"**Answers re-read with the current answer reader** (a claim may leave out its empty evidence list; rule of 2026-11-19, "
                  f"docs/method.md section 2). {changed} of {len(preds)} answers were read differently from the original run. "
                  "The original `report.md`, `metrics.json` and `predictions.jsonl` are unchanged and remain the strict result.", ""]
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
                  f"A missing fact (\"{kind.missing_fact}\") can itself point to the class: which facts are missing is not the same for every "
                  f"class (see the data sheet of the dataset). These scores count only the {part['questions']} of {len(preds)} "
                  "questions with every fact recorded, beside the main score, never instead of it.", "", *_main_rows(part["metrics"]), ""]

    sessions = card.get("sessions", [])
    commits = sorted({s["commit"] or "unknown" for s in sessions})
    revisions = sorted({s["model_revision"] for s in sessions if s.get("model_revision")})
    finish = Counter(p["finish_reason"] for p in preds)
    other = {k: v for k, v in finish.items() if k not in ("STOP", "MAX_TOKENS")}
    cut = [p for p in preds if p["finish_reason"] == "MAX_TOKENS"]
    harmful = sum(p["prediction"] is None for p in cut)  # a JSON answer cut off is incomplete, so invalid: the limit decided it
    invalid = Counter(p.get("invalid") for p in preds if p["prediction"] is None and p.get("invalid"))
    lines += ["## Notes", "",
              f"- How answers ended (finish reason): {', '.join(f'{k}: {v}' for k, v in sorted(finish.items(), key=str))}",
              *(["- Why answers were invalid: " + "; ".join(f"{k}: {v}" for k, v in invalid.most_common())] if invalid else []),
              *([f"- **{harmful} answers were cut off at the output limit (finish reason MAX_TOKENS) and count as wrong: "
                 "the output limit in the model list (max_tokens) is too low for this model (hidden reasoning can count toward "
                 "it). Results are not valid until it is raised, under a new model entry.**"] if harmful else []),
              *([f"- {len(cut) - harmful} answers reached the output limit but were complete and valid all the same, so the limit did "
                 "not change their score."] if len(cut) > harmful else []),
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
    path = run_dir / ("report_reread.md" if reread else "report.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def reread(run_dir: str | Path) -> Path:
    """Reads the raw text of every saved answer of a closed-book run again with the current answer reader and writes
    predictions_reread.jsonl, metrics_reread.json and report_reread.md beside the originals (nothing is overwritten). Only the fields
    the reader makes are replaced; the rest of each answer (raw text, usage, model version) is kept. Returns the report's path."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    require_current_format(card)
    if card["config"].get("approach", "closed_book") != "closed_book":
        raise ValueError("only closed-book answers can be read again from their text: an open-book answer is read against the "
                         "passages the run gave, which are not kept with the answer")
    if card["status"] != "complete":
        raise ValueError(f"{run_dir.as_posix()} is not a finished run")
    preds, again, changed = read_jsonl(run_dir / "predictions.jsonl"), [], 0
    for p in preds:
        parsed, problem = parse_json_answer(p["raw_text"], card["labels"])
        new = {**p, "prediction": parsed["diagnosis"] if parsed else None, "differential": parsed["differential"] if parsed else None,
               "evidence_sufficient": parsed["evidence_sufficient"] if parsed else None, "reason": parsed["reason"] if parsed else None,
               "claims": parsed["claims"] if parsed else None, "evidence": parsed["evidence"] if parsed else None, "invalid": problem}
        changed += (new["prediction"], new["invalid"]) != (p["prediction"], p["invalid"])
        again.append(new)
    path = run_dir / REREAD_ANSWERS
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in again), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return write_report(run_dir, reread=True, changed=changed)


# ---------------------------------------------------------------------------------------------------------------------------------
# Two runs side by side (python -m src.evaluate compare): one model or method against another, on the same questions


def _finished(run_dir: Path, reread: bool = False) -> tuple[dict, list[dict]]:
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    if reread and not (run_dir / REREAD_ANSWERS).is_file():
        raise ValueError(f"{run_dir.as_posix()} has no {REREAD_ANSWERS}: run python -m src.evaluate reread on it first")
    preds = read_jsonl(run_dir / (REREAD_ANSWERS if reread else "predictions.jsonl"))
    kind = DATASETS[dataset_of(card)]
    if card["status"] != "complete" or len(preds) != card[f"n_{kind.item}s"]:
        raise ValueError(f"{run_dir.as_posix()} is not a finished run: only finished runs are compared")
    require_current_format(card)
    _verified(card, preds, complete=True)
    return card, preds


def compare(run_a: str | Path, run_b: str | Path, reread: bool = False) -> dict:
    """How run B differs from run A on the same questions: the differences (B minus A) in balanced accuracy and macro-F1, with
    paired 95% intervals (the main comparison), and McNemar's exact test on right against wrong answers (a supporting test)."""
    (card_a, preds_a), (card_b, preds_b) = _finished(Path(run_a), reread), _finished(Path(run_b), reread)
    dataset, dataset_b = dataset_of(card_a), dataset_of(card_b)
    kind, kind_b = DATASETS[dataset], DATASETS[dataset_b]
    # Two different datasets whose questions can be paired (SAME_QUESTIONS: a photo without and with its patient's facts; a clinic photo,
    # its record, and both) are compared question by question: the questions differ in what is shown, so the photos (or cases) and
    # their classes are compared instead of the questions' fingerprint. A question is known by its path without the file ending (a
    # photo and its record differ in the ending only). A case dataset asks some of the text dataset's cases: a text run is compared on
    # those cases only.
    paired = dataset != dataset_b and any({dataset, dataset_b} <= group for group in SAME_QUESTIONS)
    if dataset != dataset_b and not paired:
        raise ValueError("the two runs differ in their dataset, so they cannot be compared question by question")
    part = paired and (kind.subset or kind_b.subset)
    checks = [("split", card_a["split"], card_b["split"]), ("answer options", card_a["labels"], card_b["labels"])]
    if not paired:
        checks.append(("questions", card_a[f"{kind.item}s_sha256"], card_b[f"{kind.item}s_sha256"]))
    for what, a, b in checks:
        if a != b:
            raise ValueError(f"the two runs differ in their {what}, so they cannot be compared question by question")
    if part:
        shared = {question_of(p["path"]) for p in preds_a} & {question_of(p["path"]) for p in preds_b}
        smaller = min(({question_of(p["path"]) for p in preds_a}, {question_of(p["path"]) for p in preds_b}), key=len)
        if shared != smaller:
            raise ValueError("the cases of one run are not all among the other run's cases")
        preds_a, preds_b = [p for p in preds_a if question_of(p["path"]) in shared], [p for p in preds_b if question_of(p["path"]) in shared]
    if paired and {(question_of(p["path"]), p["label"]) for p in preds_a} != {(question_of(p["path"]), p["label"]) for p in preds_b}:
        raise ValueError(f"the two runs did not answer the same {'cases' if part else 'photos'} with the same classes")
    by_question = {question_of(p["path"]): p for p in preds_b}
    if set(by_question) != {question_of(p["path"]) for p in preds_a} or len(by_question) != len(preds_b):
        raise ValueError("the two runs did not answer the same questions")
    pairs = [(p, by_question[question_of(p["path"])]) for p in sorted(preds_a, key=lambda p: p["path"])]
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
# The command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="write a run's scores and report from its saved answers")
    rep.add_argument("run_dir")
    cmp = sub.add_parser("compare", help="compare two finished runs on the same questions (paired intervals, McNemar's test)")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.add_argument("--reread", action="store_true", help="compare the re-read answers of both runs (python -m src.evaluate reread first)")
    again = sub.add_parser("reread", help="read the raw text of a closed-book run's answers again with the current answer reader")
    again.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "report":
            print(write_report(args.run_dir))
        elif args.command == "reread":
            print(reread(args.run_dir))
        else:
            print(comparison_text(compare(args.run_a, args.run_b, args.reread)))
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
