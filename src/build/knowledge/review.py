"""A reading of the appearance passages by the researcher, before any of them enters the retrieval index.

    python -m src.build.knowledge.review draw      # on the computer that holds the passages' texts
    python -m src.build.knowledge.review score     # after the file is filled in

The appearance passages (data/knowledge/appearance.csv) were chosen by fixed rules and tagged by word counts; no person has read them.
``draw`` writes 30 of them, chosen by a fixed seed and spread over the classes, with their texts, into
data/knowledge/texts/appearance_review.csv (not in Git: the texts are under their publishers' licences). The researcher reads each
passage and fills in two columns with yes or no:
- ``about_appearance``: does the passage describe how a skin lesion looks (colour, border, surface, shape, dermoscopic patterns)?
- ``about_its_disease``: is it about the disease named in ``topics`` (not only another condition the review mentions)?
``note`` is free text. No clinical training is needed: this is a reading of what the passage says, not a diagnosis.
``score`` reads the filled file and writes data/knowledge/appearance_review.md (numbers only, no text): the passages that pass both
questions, with a 95% interval, and whether they reach the bar fixed here before the reading: at least 90% pass both. The bar
only says whether the passages are suitable to add to the index; it never changes the index. Adding them is a separate decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ...common import read_rows
from ...metrics import wilson
from .library import KNOWLEDGE

TABLE = "appearance.csv"
TEXTS = Path("texts") / "appearance.jsonl"  # in data/knowledge/, not in Git
REVIEW_FILE = Path("texts") / "appearance_review.csv"  # not in Git: it holds the passages' texts
REPORT = "appearance_review.md"  # committed: numbers only
SIZE = 30
SEED = 20261119  # fixed before the draw
BAR = 0.9  # fixed before the reading: the share of the passages that must pass both questions
COLUMNS = ["passage_id", "topics", "source", "title", "section", "text", "about_appearance", "about_its_disease", "note"]
ANSWERS = ("yes", "no")


def _yes(row: dict, column: str) -> bool:
    return row[column].strip().lower() == "yes"


def draw(data: Path = KNOWLEDGE, size: int = SIZE, seed: int = SEED) -> dict:
    """Writes the file to read. Passages are taken round-robin over the classes (by each passage's first topic), each class in a seeded
    random order, so that small classes are read too. An existing file is never overwritten: it may hold the researcher's readings."""
    data = Path(data)
    path = data / REVIEW_FILE
    if path.exists():
        raise ValueError(f"{path.as_posix()} exists and may hold your readings: score it, or move it away to draw again")
    table = read_rows(data / TABLE, "run python -m src.build.knowledge.enrich appearance first")
    if not (data / TEXTS).is_file():
        raise FileNotFoundError(f"{(data / TEXTS).as_posix()} not found: the passages' texts are only on the computer that built them")
    with (data / TEXTS).open(encoding="utf-8") as f:
        texts = {p["passage_id"]: p["text"] for p in map(json.loads, (line for line in f if line.strip()))}
    missing = [r["passage_id"] for r in table if r["passage_id"] not in texts]
    if missing:
        raise ValueError(f"{len(missing)} passages have no text in {(data / TEXTS).as_posix()} (e.g. {missing[0]}): build them again")
    by_class = defaultdict(list)
    for row in sorted(table, key=lambda r: r["passage_id"]):
        by_class[row["topics"].split(";")[0].strip()].append(row)
    rng = random.Random(seed)
    for rows in by_class.values():
        rng.shuffle(rows)
    chosen = []
    while len(chosen) < min(size, len(table)):
        for label in sorted(by_class):
            if by_class[label] and len(chosen) < size:
                chosen.append(by_class[label].pop())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({**{c: r[c] for c in ("passage_id", "topics", "source", "title", "section")}, "text": texts[r["passage_id"]],
                          "about_appearance": "", "about_its_disease": "", "note": ""} for r in chosen)
    return {"file": path.as_posix(), "passages": len(chosen)}


def score(data: Path = KNOWLEDGE) -> dict:
    """Reads the filled file and writes the report. Stops, naming the first gap, if any answer is not yes or no."""
    data = Path(data)
    rows = read_rows(data / REVIEW_FILE, "run draw first, then read the passages")
    if not rows:
        raise ValueError(f"{(data / REVIEW_FILE).as_posix()} holds no passage")
    for number, row in enumerate(rows, start=2):  # line 1 is the header
        for column in ("about_appearance", "about_its_disease"):
            if row[column].strip().lower() not in ANSWERS:
                raise ValueError(f"line {number} ({row['passage_id']}): {column} must be yes or no, here {row[column]!r}")
    n = len(rows)
    counts = {c: sum(_yes(r, c) for r in rows) for c in ("about_appearance", "about_its_disease")}
    both = sum(_yes(r, "about_appearance") and _yes(r, "about_its_disease") for r in rows)
    low, high = wilson(both, n)
    per_class = defaultdict(lambda: [0, 0])
    for r in rows:
        label = r["topics"].split(";")[0].strip()
        per_class[label][0] += 1
        per_class[label][1] += _yes(r, "about_appearance") and _yes(r, "about_its_disease")
    meets = both / n >= BAR
    lines = ["# Reading of the appearance passages", "",
             f"Written by `python -m src.build.knowledge.review score`. {n} of the {len(read_rows(data / TABLE, 'run enrich first'))} appearance passages "
             f"were drawn with seed {SEED}, spread over the classes, and read by the researcher (no clinical training is needed: the two questions are "
             "about what the text says). The bar was fixed before the reading: at least "
             f"{BAR:.0%} of the passages must pass both questions.", "",
             "| Question | Yes | Share |", "|---|---|---|",
             f"| Does it describe how a lesion looks? | {counts['about_appearance']} of {n} | {counts['about_appearance'] / n:.0%} |",
             f"| Is it about the disease named in its topics? | {counts['about_its_disease']} of {n} | {counts['about_its_disease'] / n:.0%} |",
             f"| **Both** | **{both} of {n}** | **{both / n:.0%}** (95% interval {low:.0%} to {high:.0%}) |", "",
             "| Class (first topic) | Read | Pass both |", "|---|---|---|",
             *[f"| {label} | {k[0]} | {k[1]} |" for label, k in sorted(per_class.items())], "",
             f"**{'The bar is met' if meets else 'The bar is not met'}**: {both / n:.0%} pass both questions"
             f" ({'at least' if meets else 'less than'} {BAR:.0%}). "
             + ("The passages are suitable to add to the index; adding them is a separate decision of the researcher."
                if meets else "The passages are not added to the index."), ""]
    report = data / REPORT
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"n": n, "both": both, "meets": meets, "report": report.as_posix()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.build.knowledge.review", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("draw", help="write the passages to read, with their texts")
    sub.add_parser("score", help="read the filled file and write the report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.step == "draw":
            done = draw()
            print(f"{done['passages']} passages written to {done['file']} (not in Git). Open it in a spreadsheet, read each passage and "
                  "fill about_appearance and about_its_disease with yes or no; then run: python -m src.build.knowledge.review score")
        else:
            done = score()
            print(f"{done['both']} of {done['n']} pass both questions: the bar is {'met' if done['meets'] else 'not met'}. Written: {done['report']}. "
                  "Commit it; nothing else changes.")
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
