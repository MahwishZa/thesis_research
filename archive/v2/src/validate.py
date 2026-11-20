"""Checks of the whole pipeline that cost nothing: no model is asked, nothing is downloaded, and (unless --files is given) none of
the photos and texts kept on the researcher's computer is needed. Run it before every experiment:

    python -m src.validate [--files]

It checks, from the committed tables, prompts and configs:
- each dataset's table: consistent, every class of a split also in training, and no patient or article in two splits (a leak);
- the views that must answer the same questions (SAME_QUESTIONS): the same questions, splits, classes and groups; the three clinic
  views hold the same photo and the same record to the byte; the text, text + photos and photos views of a published case hold the
  same case and the same pictures (the modalities are aligned, so their answers can be paired);
- the prompts: each loads, each ends with the same answer-format block, and that block's example is an answer the reader accepts;
- every model of the model list on every dataset and approach (settings only; no key is read);
- the knowledge index: no validation or test case of the published cases, and each case's file as the dataset has it.
With --files every file of a dataset that is on this computer is also hashed against its table (the check a run makes).
Each line says OK, FAIL or SKIP; the exit status is 1 when any line failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from .datasets import (DATA, DATASETS, SAME_QUESTIONS, SPLITS, classes, dataset_paths, load_dataset, question_of, read_table)
from .prompts import ANSWER_KEYS, load_template
from .run import APPROACHES, MODELS, load_config, prompt_file

ANSWER_BLOCK = "Answer with one JSON object and nothing else"  # where every prompt's answer format begins
INDEX = DATA / "knowledge" / "index.csv"


def table_problems(dataset: str, items: list) -> list[str]:
    """What is wrong with one dataset's questions: a missing group, a group in two splits, a split without questions."""
    problems = []
    if not all(item.group for item in items):
        problems.append("a question has no group (patient or article)")
    splits_of = defaultdict(set)
    for item in items:
        splits_of[item.group].add(item.split)
    leaked = sorted(group for group, splits in splits_of.items() if group and len(splits) > 1)
    if leaked:
        problems.append(f"{DATASETS[dataset].group}s in more than one split: {', '.join(leaked[:3])}" + (" ..." if len(leaked) > 3 else ""))
    empty = [s for s in SPLITS if not any(item.split == s for item in items)]
    if empty:
        problems.append(f"no question in the split(s): {', '.join(empty)}")
    try:
        classes(items)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def pair_problems(first: str, a: list, second: str, b: list) -> list[str]:
    """Whether two views of the same questions are aligned: the same questions (a view that is a subset of the other asks some of its
    questions; a question is known by its path, which holds its split and class, so a question in another split or class is another
    question), each with the same group, and the same answer options."""
    problems = []
    by_a, by_b = {question_of(i.path): i for i in a}, {question_of(i.path): i for i in b}
    if len(by_a) != len(a) or len(by_b) != len(b):
        return [f"{first} or {second} lists a question twice"]
    kind_a, kind_b = DATASETS[first], DATASETS[second]
    if kind_a.subset == kind_b.subset:
        if set(by_a) != set(by_b):
            problems.append(f"the questions differ: {len(set(by_a) - set(by_b))} only in {first}, {len(set(by_b) - set(by_a))} only in {second}")
    else:
        small, large, names = (by_a, by_b, (first, second)) if kind_a.subset else (by_b, by_a, (second, first))
        outside = sorted(set(small) - set(large))
        if outside:
            problems.append(f"{len(outside)} questions of {names[0]} are not questions of {names[1]} (e.g. {outside[0]})")
    for question in sorted(set(by_a) & set(by_b)):
        x, y = by_a[question], by_b[question]
        if x.group != y.group:
            problems.append(f"{question}: group is {x.group!r} in {first} and {y.group!r} in {second}")
    if classes(a) != classes(b):
        problems.append("the answer options differ")
    return problems[:5]


def same_content_problems(first: str, a: list, second: str, b: list, what: str = "file") -> list[str]:
    """Questions of two views that must show the same file (same path and hash)."""
    by_b = {question_of(i.path): i for i in b}
    return [f"{question_of(i.path)}: not the same {what} in {first} and {second}" for i in a
            if question_of(i.path) in by_b and (i.path, i.sha256) != (by_b[question_of(i.path)].path, by_b[question_of(i.path)].sha256)][:5]


def record_problems(record: list, both: list) -> list[str]:
    """The record view's file and the record text the photo + record view puts into its prompt are the same bytes."""
    by_record = {question_of(i.path): i for i in record}
    return [f"{question_of(i.path)}: the record in the photo + record view is not the record file of the record view" for i in both
            if question_of(i.path) in by_record and hashlib.sha256(i.facts.encode("utf-8")).hexdigest() != by_record[question_of(i.path)].sha256][:5]


def picture_problems(first: str, a: list, second: str, b: list) -> list[str]:
    """Two views of a published case that send pictures send the same pictures, in the same order."""
    by_b = {question_of(i.path): i for i in b}
    return [f"{question_of(i.path)}: other pictures in {first} than in {second}" for i in a
            if question_of(i.path) in by_b and i.pictures != by_b[question_of(i.path)].pictures][:5]


def case_problems(text: list, photos: list) -> list[str]:
    """A question of a view that does not show the text still names the same case file as the text view (same hash)."""
    by_text = {question_of(i.path): i for i in text}
    return [f"{question_of(i.path)}: not the same case in the text view" for i in photos
            if question_of(i.path) in by_text and i.sha256 != by_text[question_of(i.path)].sha256][:5]


def prompt_problems() -> list[str]:
    """Each prompt loads, ends with the same answer-format block, and that block's example has the keys the reader requires."""
    problems, blocks = [], {}
    for approach in APPROACHES:
        for dataset in DATASETS:
            file = prompt_file(dataset, approach)
            try:
                text = load_template(file, dataset)
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
                continue
            if ANSWER_BLOCK not in text:
                problems.append(f"{file.as_posix()} has no answer-format block")
                continue
            blocks.setdefault(approach, {})[dataset] = block = text[text.index(ANSWER_BLOCK):].strip()
            example = next((line for line in block.splitlines() if line.startswith('{"differential"')), None)
            try:
                keys = tuple(json.loads(example)) if example else None
            except json.JSONDecodeError:
                keys = None
            if keys != ANSWER_KEYS:
                problems.append(f"{file.as_posix()}: the example answer does not have the keys {list(ANSWER_KEYS)} in order")
    for approach, found in blocks.items():
        reference = next(iter(found.values()))
        problems += [f"{prompt_file(d, approach).as_posix()}: the answer-format block differs from the other prompts'"
                     for d, block in found.items() if block != reference]
    return problems


def config_problems() -> list[str]:
    """Every model of the list, on every dataset and approach: the settings are complete (nothing is sent, no key is read)."""
    with MODELS.open("rb") as f:
        names = [entry["name"] for entry in tomllib.load(f)["models"]]
    problems = []
    for model in names:
        for dataset in DATASETS:
            for approach in APPROACHES:
                try:
                    load_config(model, dataset, approach, MODELS)
                except (OSError, ValueError) as exc:
                    problems.append(f"{model} on {dataset}/{approach}: {exc}")
    return problems or ([] if names else [f"{MODELS.as_posix()} lists no model"])


def knowledge_problems(index: Path = INDEX, cases: str = "published_text") -> list[str]:
    """No validation or test case of the published cases in the knowledge index, and each indexed case as the dataset has it."""
    with Path(index).open(encoding="utf-8", newline="") as f:
        items = [row for row in csv.DictReader(f) if row["kind"] == "case"]
    table, _ = dataset_paths(cases)
    with table.open(encoding="utf-8", newline="") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}
    prefix = "data/" + DATASETS[cases].files + "/"
    problems = []
    for item in items:
        row = rows.get(item["path"].removeprefix(prefix))
        if row is None:
            problems.append(f"{item['item_id']}: not a case of the published cases")
        elif row["split"] != "train":
            problems.append(f"{item['item_id']}: a {row['split']} case is in the knowledge index (a leak)")
        elif row["sha256"] != item["sha256"]:
            problems.append(f"{item['item_id']}: the index holds another version of the case than the dataset")
    return problems[:5]


def validate(files: bool = False, say=print) -> bool:
    """Prints one line per check and returns True when none failed."""
    failed = False

    def line(status: str, what: str, problems: list[str] | None = None, detail: str = "") -> None:
        nonlocal failed
        failed = failed or bool(problems)
        say(f"[{'FAIL' if problems else status:>4}] {what}" + (f": {detail}" if detail and not problems else ""))
        for problem in problems or []:
            say(f"       - {problem}")

    loaded = {}
    for dataset in DATASETS:
        table, root = dataset_paths(dataset)
        try:
            items = read_table(table, dataset)
        except (OSError, ValueError) as exc:
            line("FAIL", f"Table of {dataset}", [str(exc)])
            continue
        loaded[dataset] = items
        line("OK", f"Table of {dataset}", table_problems(dataset, items),
             f"{len(items)} {DATASETS[dataset].item}s, {len(classes(items))} classes, no {DATASETS[dataset].group} in two splits")
        if files:
            if not root.is_dir():
                line("SKIP", f"Files of {dataset}", None, f"{root.as_posix()} is not on this computer")
                continue
            try:
                load_dataset(table, root, dataset)
                line("OK", f"Files of {dataset}", None, "every file is the one its table lists, byte for byte")
            except (OSError, ValueError) as exc:
                line("FAIL", f"Files of {dataset}", [str(exc)])
    for group in SAME_QUESTIONS:
        names = [d for d in DATASETS if d in group and d in loaded]
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                some = DATASETS[first].subset != DATASETS[second].subset
                line("OK", f"Alignment {first} / {second}", pair_problems(first, loaded[first], second, loaded[second]),
                     "the cases of the photo view are some of the text's; same groups and answer options" if some
                     else "same questions, groups and answer options")
    if {"clinic_photo", "clinic_photo_record"} <= set(loaded):
        line("OK", "Alignment clinic_photo / clinic_photo_record: the photo", same_content_problems(
            "clinic_photo", loaded["clinic_photo"], "clinic_photo_record", loaded["clinic_photo_record"], "photo"), "the same photo, byte for byte")
    if {"clinic_record", "clinic_photo_record"} <= set(loaded):
        line("OK", "Alignment clinic_record / clinic_photo_record: the record", record_problems(
            loaded["clinic_record"], loaded["clinic_photo_record"]), "the same record text, byte for byte")
    if {"published_text_photos", "published_photos"} <= set(loaded):
        line("OK", "Alignment published_text_photos / published_photos: the pictures", picture_problems(
            "published_text_photos", loaded["published_text_photos"], "published_photos", loaded["published_photos"]),
             "the same pictures in the same order")
    for dataset in ("published_text_photos", "published_photos"):
        if {dataset, "published_text"} <= set(loaded):
            line("OK", f"Alignment {dataset} / published_text: the case", case_problems(loaded["published_text"], loaded[dataset]),
                 "the same case file")
    line("OK", "Prompts", prompt_problems(), f"{len(DATASETS) * len(APPROACHES)} prompts, one answer format, read by the answer reader")
    line("OK", "Model list", config_problems(), "every model on every dataset and approach (no key read, nothing sent)")
    if INDEX.is_file() and "published_text" in loaded:
        line("OK", "Knowledge index", knowledge_problems(), "no validation or test case in it")
    else:
        line("SKIP", "Knowledge index", None, f"{INDEX.as_posix()} is not built")
    say("All checks passed." if not failed else "Fix the lines marked FAIL before any experiment.")
    return not failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.validate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--files", action="store_true", help="also hash every file of a dataset that is on this computer")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(0 if validate(files=args.files) else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
