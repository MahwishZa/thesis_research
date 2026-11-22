#!/usr/bin/env python3
"""
Step 2 of the data pipeline: turns the cleaned cases (data/processed/master_clean.csv) into the three views of the same cases.

  data/views/image_only/   the image only
  data/views/text_only/    the patient's recorded facts only (age, sex, body site in one fixed sentence)
  data/views/multimodal/   the image and the same sentence

Each view folder holds
  cases.csv                one row per case: id, split, fold, lesion group, label, and only the inputs of that view
  qa_train.jsonl, qa_val.jsonl, qa_test.jsonl
                           one 8-option multiple-choice question per case (MedQA style) for the LLM and RAG systems

The questions use one fixed template and real data only; nothing is generated. The 8 options are shuffled for each case with a
fixed seed (the same order for a case in all three views), so a letter never stands for one class. The question never names a
class, and a text-only question never mentions an image.

Usage: python src/build_views.py --data data
"""
import argparse
import os
import random

from common import CLASSES, LETTERS, MASTER, OPTION_TEXT, SEED, SPLITS, VIEWS, get_logger, question_file, read_csv, \
    view_table, write_csv, write_jsonl

OPTION_SEED = SEED
CORE = ["isic_id", "split", "cv_fold", "group_id", "label", "label_idx", "label_name"]
VIEW_INPUTS = {"image_only": ["image_path"],
               "text_only": ["clinical_text", "age", "sex", "site"],
               "multimodal": ["image_path", "clinical_text", "age", "sex", "site"]}
Q_IMAGE = "Look at the dermoscopic image of the skin lesion. What is the most likely diagnosis?"
Q_TEXT = "{vignette} What is the most likely diagnosis?"
Q_MM = "{vignette} Look at the dermoscopic image of the lesion. What is the most likely diagnosis?"


def options_for(case_id, seed=OPTION_SEED):
    """The options of one case: the eight classes in an order fixed by the seed and the case id (the same on every computer).
    Returns {letter: option text} and {class: letter}."""
    order = random.Random(f"{seed}:{case_id}").sample(CLASSES, len(CLASSES))
    return {LETTERS[i]: OPTION_TEXT[c] for i, c in enumerate(order)}, {c: LETTERS[i] for i, c in enumerate(order)}


def question(case, view):
    """The question record of one case in one view."""
    options, letter = options_for(case["isic_id"])
    q = {"id": case["isic_id"], "split": case["split"], "options": options, "answer": letter[case["label"]],
         "answer_text": OPTION_TEXT[case["label"]], "gold_label": case["label"], "group_id": case["group_id"]}
    if view == "image_only":
        return {**q, "question": Q_IMAGE, "image_path": case["image_path"]}
    if view == "text_only":
        return {**q, "question": Q_TEXT.format(vignette=case["clinical_text"])}
    return {**q, "question": Q_MM.format(vignette=case["clinical_text"]), "image_path": case["image_path"]}


def build(data):
    cases = read_csv(os.path.join(data, MASTER))
    unknown = sorted({c["label"] for c in cases} - set(CLASSES))
    if unknown:
        raise ValueError(f"{MASTER} holds classes that are not among the 8: {unknown}")
    counts = {}
    for view in VIEWS:
        write_csv(os.path.join(data, view_table(view)), cases, CORE + VIEW_INPUTS[view])
        for split in SPLITS:
            items = [question(c, view) for c in cases if c["split"] == split]
            write_jsonl(os.path.join(data, question_file(view, split)), items)
            counts[(view, split)] = len(items)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Builds the three views and their question files from the cleaned cases.")
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    log = get_logger("build_views")
    for (view, split), n in build(a.data).items():
        log.info(f"{view:11s} {split:5s} {n:5d} questions")


if __name__ == "__main__":
    main()
