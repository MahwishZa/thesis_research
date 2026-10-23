"""The commands of ``python -m data_roles``: look at a split, write its questions out, find repeats between splits.

    check    read one split and print what is in it (counts, answer letters, repeated stems, subjects)
    export   write the questions (for ``python -m rationale_query``), the answer key and a manifest of hashes
    overlap  list the questions of one split that repeat questions of another

A split comes from a local file (JSON, JSON lines, or Parquet if pyarrow is installed) or, with ``--hub``, from the Hugging
Face hub through the ``datasets`` package. Exit status: 0 done, 1 a problem with the data or the arguments, 3 the answers of
the split are withheld.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from rationale_query.store import write_atomic

from .overlap import find_overlaps
from .questions import DATASETS, READERS, AnswersWithheld, Item, QuestionDataError, hub_rows, manifest, read_rows, render_question, summarise


@dataclass
class Streams:
    out: TextIO
    err: TextIO


def _add_source(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """The options that name one split. With a prefix such as ``reference-`` they read ``--reference-file`` and so on."""
    dest = prefix.replace("-", "_")
    parser.add_argument(f"--{prefix}file", dest=f"{dest}file", help="a local JSON, JSON-lines or Parquet file")
    parser.add_argument(f"--{prefix}hub", dest=f"{dest}hub", help="a Hugging Face dataset, for example awinml/medqa")
    parser.add_argument(f"--{prefix}config", dest=f"{dest}config", help="the dataset configuration, if it has one")
    parser.add_argument(f"--{prefix}revision", dest=f"{dest}revision", help="a dataset revision to pin")
    parser.add_argument(f"--{prefix}split", dest=f"{dest}split", required=True, help="train, validation or test")


def _load(args: argparse.Namespace, prefix: str = "") -> list[Item]:
    file, hub = getattr(args, f"{prefix}file"), getattr(args, f"{prefix}hub")
    split = getattr(args, f"{prefix}split")
    if bool(file) == bool(hub):
        raise QuestionDataError("give exactly one of a file and a hub dataset")
    rows = read_rows(file) if file else hub_rows(hub, split, config=getattr(args, f"{prefix}config"), revision=getattr(args, f"{prefix}revision"))
    return READERS[args.dataset](rows, split)


def _refuse_overwrite(*paths: str | None) -> None:
    for path in paths:
        if path and Path(path).exists():
            raise QuestionDataError(f"{path} exists; this command does not overwrite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data_roles", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("check", "describe one split"), ("export", "write questions, answers and a manifest")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--dataset", required=True, choices=DATASETS)
        _add_source(sub)
        if name == "export":
            sub.add_argument("--questions-out", required=True, help="JSON lines of id and question, for rationale_query")
            sub.add_argument("--answers-out", required=True, help="JSON lines of id and the answer letter")
            sub.add_argument("--manifest-out", required=True, help="a JSON file with a hash for every question")
    sub = commands.add_parser("overlap", help="questions of the candidate split that repeat questions of the reference split")
    sub.add_argument("--dataset", required=True, choices=DATASETS)
    _add_source(sub, "reference-")
    _add_source(sub, "candidate-")
    sub.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity of five-word shingles that counts as a near repeat")
    return parser


def main(argv: Sequence[str] | None = None, streams: Streams | None = None) -> int:
    streams = streams or Streams(sys.stdout, sys.stderr)
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    try:
        if args.command == "check":
            items = _load(args)
            streams.out.write(json.dumps(summarise(items), indent=2, sort_keys=True) + "\n")
        elif args.command == "export":
            items = _load(args)
            _refuse_overwrite(args.questions_out, args.answers_out, args.manifest_out)
            write_atomic(args.questions_out, "".join(json.dumps({"id": i.id, "question": render_question(i)}, ensure_ascii=False) + "\n" for i in items))
            write_atomic(args.answers_out, "".join(json.dumps({"id": i.id, "answer": i.answer}) + "\n" for i in items))
            write_atomic(args.manifest_out, json.dumps(manifest(items), indent=2, sort_keys=True) + "\n")
            streams.out.write(f"wrote {len(items)} questions\n")
        else:
            reference = _load(args, "reference_")
            candidates = _load(args, "candidate_")
            found = find_overlaps(reference, candidates, threshold=args.threshold)
            exact = sum(1 for pair in found if pair.exact)
            repeated = len({pair.candidate for pair in found})
            streams.out.write(f"{repeated} of {len(candidates)} candidate questions repeat a reference question ({exact} pairs are exact)\n")
            for pair in found[:20]:
                streams.out.write(f"  {pair.candidate} ~ {pair.reference}  similarity {pair.similarity}\n")
    except AnswersWithheld as problem:
        streams.err.write(f"{problem}\n")
        return 3
    except QuestionDataError as problem:
        streams.err.write(f"error: {problem}\n")
        return 1
    return 0
