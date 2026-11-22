#!/usr/bin/env python3
"""
Shows where the research pipeline stands: for every step, whether it has been run, whether its output is complete and up to date,
and the command to run next. It only reads; it changes nothing and calls no API.

  done      the output is there and agrees with its input
  not run   the output is missing (the step has not been run here)
  problem   the output is there but incomplete, out of date or inconsistent: rerun the step

Usage: python src/check_pipeline.py            (exit code 1 when any step has a problem)
"""
import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile

import build_views
from balance_test import TEST_PER_CLASS
from common import (ALL_CASES, CLASSES, KNOWLEDGE_CORPUS, PROCESSED_CASES, PROCESSED_IMAGES, RAW, SPLITS, VIEWS,
                    images_not_in_table, question_file, read_csv, view_table)

DONE, NOT_RUN, PROBLEM = "done", "not run", "problem"
PROCESSED_FILES = ("all_cases.csv", "raw_files_audit.csv", "near_duplicates.csv", "classes.json", "normalization.json",
                   "class_weights.json", "train_oversampled.csv", "train_undersampled.csv")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def step_raw(data):
    audit = os.path.join(data, PROCESSED_CASES, "raw_files_audit.csv")
    folders = [os.path.join(data, RAW, d) for d in ("Train", "Test")]
    if not all(os.path.isdir(f) for f in folders):
        return NOT_RUN, "the Kaggle download is not in data/raw/ (Train/, Test/)"
    n = sum(f.lower().endswith(".jpg") for d in folders for _, _, fs in os.walk(d) for f in fs)
    expected = len(read_csv(audit)) if os.path.isfile(audit) else None
    if expected is not None and n != expected:
        return PROBLEM, f"{n} JPG files, but the audit lists {expected}"
    return DONE, f"{n} JPG files"


def step_clean(data):
    missing = [f for f in PROCESSED_FILES if not os.path.isfile(os.path.join(data, PROCESSED_CASES, f))]
    if len(missing) == len(PROCESSED_FILES):
        return NOT_RUN, "no table in data/processed_cases/"
    if missing:
        return PROBLEM, f"missing: {', '.join(missing)}"
    cases = read_csv(os.path.join(data, ALL_CASES))
    test = {c: sum(r["split"] == "test" and r["label"] == c for r in cases) for c in CLASSES}
    if set(test.values()) != {TEST_PER_CLASS}:
        return PROBLEM, (f"the test split is not balanced to {TEST_PER_CLASS} per class ("
                         + ", ".join(f"{c} {n}" for c, n in test.items()) + "): rerun preprocess.py, then build_views.py")
    images = os.path.join(data, PROCESSED_IMAGES)
    if not os.path.isdir(images):
        return NOT_RUN, f"{len(cases)} cases; the cleaned images are not in data/{PROCESSED_IMAGES}/ (laptop only)"
    absent = [c["isic_id"] for c in cases if not os.path.isfile(os.path.join(data, c["image_path"]))]
    extra = images_not_in_table(data, [c["isic_id"] for c in cases])
    notes = []
    if absent:
        notes.append(f"{len(absent)} images missing, rerun preprocess.py (e.g. {', '.join(absent[:3])})")
    if extra:
        notes.append(f"{len(extra)} images not in the case table, left from an earlier run; delete them from "
                     f"data/{PROCESSED_IMAGES}/: {', '.join(extra[:10])}" + (" ..." if len(extra) > 10 else ""))
    if notes:
        return PROBLEM, "; ".join(notes)
    return DONE, f"{len(cases)} cases, {TEST_PER_CLASS} test images per class, every image present"


def step_views(data):
    names = [view_table(v) for v in VIEWS] + [question_file(v, s) for v in VIEWS for s in SPLITS]
    if not os.path.isfile(os.path.join(data, ALL_CASES)):
        return NOT_RUN, "needs step 1"
    missing = [n for n in names if not os.path.isfile(os.path.join(data, n))]
    if len(missing) == len(names):
        return NOT_RUN, "no view built"
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, PROCESSED_CASES))
        shutil.copy(os.path.join(data, ALL_CASES), os.path.join(d, ALL_CASES))
        build_views.build(d)
        stale = [n for n in names if n in missing or not filecmp.cmp(os.path.join(d, n), os.path.join(data, n), shallow=False)]
    if stale:
        return PROBLEM, f"{len(stale)} view files differ from a fresh build (e.g. {stale[0]})"
    return DONE, f"{len(names)} files, the same as a fresh build"


def step_feature_check(results):
    path = os.path.join(results, "feature_check", "feature_check_results.json")
    return (DONE, "results present") if os.path.isfile(path) else (NOT_RUN, "no results")


def step_corpus(data):
    folder = os.path.join(data, KNOWLEDGE_CORPUS)
    files = {f: os.path.join(folder, f) for f in ("passages.jsonl", "manifest.csv", "corpus_card.json")}
    present = [f for f, p in files.items() if os.path.isfile(p)]
    if not present:
        return NOT_RUN, "the corpus has not been built"
    if "manifest.csv" in present and "passages.jsonl" not in present:
        return NOT_RUN, f"manifest only ({len(read_csv(files['manifest.csv']))} passages); passages.jsonl is on the laptop"
    if len(present) < 3:
        return PROBLEM, f"only {', '.join(present)}"
    n_passages, n_manifest = count_lines(files["passages.jsonl"]), len(read_csv(files["manifest.csv"]))
    if n_passages != n_manifest:
        return PROBLEM, f"{n_passages} passages but {n_manifest} manifest rows: rebuild"
    with open(files["corpus_card.json"], encoding="utf-8") as f:
        empty = json.load(f).get("empty_topic_boxes", [])
    return DONE, f"{n_passages} passages" + (f"; topics with no passage: {', '.join(empty)}" if empty else "")


def step_references(data):
    path = os.path.join(data, KNOWLEDGE_CORPUS, "reference_texts.json")
    if not os.path.isfile(path):
        return NOT_RUN, "no reference texts"
    with open(path, encoding="utf-8") as f:
        refs = json.load(f)
    old = [c for c, r in refs.items() if "texts" not in r]
    if old:
        return PROBLEM, "the file has the old form (one text per disease): rerun build_reference_texts.py"
    missing = [c for c in CLASSES if c not in refs]
    empty = [c for c, r in refs.items() if not r["texts"]]
    if missing or empty:
        return PROBLEM, "; ".join(([f"missing {missing}"] if missing else []) + ([f"no text: {empty}"] if empty else []))
    counts = ", ".join(f"{c} {len(refs[c]['texts'])}" for c in CLASSES)
    return DONE, f"{sum(len(r['texts']) for r in refs.values())} reference texts ({counts})"


def step_runs(data, results, models):
    """One line per model, view and split that has a run folder; the test split is listed even when missing."""
    rows = []
    for model in models:
        for view in VIEWS:
            for split in ("val", "test"):
                final = os.path.join(results, "llm_only", model, f"{view}_{split}")
                name = f"llm_only {model} {view} {split}"
                if os.path.isdir(final + ".partial"):
                    with open(os.path.join(final + ".partial", "run.json"), encoding="utf-8") as f:
                        info = json.load(f)
                    rows.append((name, PROBLEM, f"unfinished: {info['answered']}/{info['of']} ({info['status']}); run the same "
                                                "command again to continue"))
                elif os.path.isdir(final):
                    rows.append((name, *check_run(data, final, view, split)))
                elif split == "test":
                    rows.append((name, NOT_RUN, ""))
    return rows


def check_run(data, folder, view, split):
    with open(os.path.join(folder, "run.json"), encoding="utf-8") as f:
        info = json.load(f)
    qfile = os.path.join(data, question_file(view, split))
    if info["settings"].get("questions_sha256") != sha256(qfile):
        return PROBLEM, "the question file changed after this run: the run does not match the current questions"
    report = os.path.join(folder, "report.json")
    if not os.path.isfile(report):
        return PROBLEM, "complete, but not evaluated yet (no report.json)"
    return DONE, "complete and evaluated"


def check(data="data", results="results"):
    from run_llm_only import MODELS
    steps = [("1 raw Kaggle download", step_raw(data)), ("1 cleaning (preprocess.py)", step_clean(data)),
             ("2 views (build_views.py)", step_views(data)), ("- feature check (feature_check.py)", step_feature_check(results)),
             ("3 knowledge corpus (build_knowledge_corpus.py)", step_corpus(data)),
             ("3b reference texts (build_reference_texts.py)", step_references(data))]
    return [(name, *r) for name, r in steps] + step_runs(data, results, sorted(MODELS))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Where the research pipeline stands (reads only).")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    rows = check(a.data, a.results)
    width = max(len(r[0]) for r in rows)
    mark = {DONE: "[x]", NOT_RUN: "[ ]", PROBLEM: "[!]"}
    for name, status, note in rows:
        print(f"{mark[status]} {name.ljust(width)}  {status:8s} {note}")
    problems = sum(r[1] == PROBLEM for r in rows)
    print(f"\n{sum(r[1] == DONE for r in rows)} done, {sum(r[1] == NOT_RUN for r in rows)} not run, {problems} with a problem")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
