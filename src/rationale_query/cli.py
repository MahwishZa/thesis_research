"""The commands of ``python -m rationale_query``.

    generate         write a rationale for every question (stop and start again as often as needed)
    show-prompt      print what the model is given for one question, without running a model
    export-queries   write the rationales as the list of queries that the retriever reads

Exit status: 0 done, 1 a problem with the arguments or the data, 2 the run stopped or some questions failed (run the same
command again), 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .generator import CHAT_FORMAT, GenerationError, GenerationSettings, LlamaGenerator, render_user_message
from .prompt import INSTRUCTION, LEAD_IN, SEPARATOR, build_rationale_prompt
from .query import export_queries, queries_for
from .run import RunAborted, generate_rationales
from .store import RationaleRecord, RationaleStore, StoreError, question_hash, read_questions

#: The settings that decide what a rationale is. A folder made with other values is refused.
RELEVANT = ("model_sha256", "chat_format", "n_ctx", "sampling", "prompt_sha256")


@dataclass
class Streams:
    out: TextIO
    err: TextIO


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rationale_query", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a rationale for every question (resumable)")
    gen.add_argument("--questions", type=Path, required=True, help="JSON lines of id and question (the options are part of the question)")
    gen.add_argument("--out", type=Path, required=True, help="the output folder (new, or the same as an earlier run)")
    gen.add_argument("--model", type=Path, required=True, help="path of the GGUF model file (Meerkat-7B)")
    gen.add_argument("--model-sha256", help="stop unless the file has this SHA-256")
    gen.add_argument("--n-ctx", type=int, default=GenerationSettings.n_ctx, help="memory window in tokens; the length Meerkat-7B was trained with")
    gen.add_argument("--n-threads", type=int, default=None)
    gen.add_argument("--n-gpu-layers", type=int, default=0)
    gen.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    gen.add_argument("--quiet", action="store_true", help="no line per question")
    gen.add_argument("--verbose", action="store_true", help="show llama.cpp's own messages")

    show = sub.add_parser("show-prompt", help="print what the model is given for one question")
    show.add_argument("--question", required=True, help="the question, with its options")

    export = sub.add_parser("export-queries", help="write the rationales as a JSON list of queries for the retriever")
    export.add_argument("--questions", type=Path, required=True)
    export.add_argument("--rationales", type=Path, required=True, help="the output folder of the generate command")
    export.add_argument("--out", type=Path, required=True, help="the query file to write (never overwritten)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    generator_factory: Callable[[argparse.Namespace, Streams], Any] | None = None,
    streams: Streams | None = None,
) -> int:
    streams = streams or Streams(sys.stdout, sys.stderr)
    for stream in (streams.out, streams.err):  # a console that cannot show a character prints ? instead of failing
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    try:
        if args.command == "show-prompt":
            return _show_prompt(args, streams)
        if args.command == "generate":
            return _generate(args, streams, generator_factory or _default_generator)
        return _export(args, streams)
    except RunAborted as exc:
        streams.err.write(f"stopped: {exc}\n")
        return 2
    except KeyboardInterrupt:
        streams.err.write("\ninterrupted; run the same command again to continue\n")
        return 130
    except (StoreError, GenerationError, OSError, ValueError, ImportError) as exc:
        streams.err.write(f"error: {exc}\n")
        return 1


def _show_prompt(args: argparse.Namespace, streams: Streams) -> int:
    prompt = build_rationale_prompt(args.question)
    streams.out.write("The paper's prompt (section 3.3), with the question in place of [initial_query]:\n\n")
    streams.out.write(prompt + "\n\n")
    streams.out.write(f"What the model reads ({CHAT_FORMAT} chat format, one user message, no system message; the tokenizer adds the start token):\n\n")
    streams.out.write(render_user_message(prompt) + "\n")
    return 0


def _default_generator(args: argparse.Namespace, streams: Streams) -> LlamaGenerator:
    settings = GenerationSettings(n_ctx=args.n_ctx, n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers)
    streams.err.write(f"checking and loading {args.model.name} (the SHA-256 of a large file takes about half a minute)\n")
    return LlamaGenerator(args.model, settings, expected_sha256=args.model_sha256, verbose=args.verbose)


def _generate(args: argparse.Namespace, streams: Streams, factory: Callable[[argparse.Namespace, Streams], Any]) -> int:
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[: args.limit]
    store = RationaleStore(args.out)
    generator = factory(args, streams)
    try:
        from hashlib import sha256

        config = {**generator.config, "prompt_sha256": sha256((INSTRUCTION + SEPARATOR + LEAD_IN).encode("utf-8")).hexdigest()}
        saved = store.load() if store.records_path.exists() else {}
        pending = sum(1 for q in questions if not (q.id in saved and saved[q.id].status == "ok" and saved[q.id].question_sha256 == question_hash(q.text)))
        begun = time.perf_counter()
        done = 0

        def progress(position: int, total: int, record: RationaleRecord) -> None:
            nonlocal done
            done += 1
            if args.quiet:
                return
            left = (time.perf_counter() - begun) / done * max(0, pending - done)
            what = record.error if record.status == "error" else f"{record.completion_tokens} tokens, {record.finish_reason}"
            streams.err.write(f"[{position}/{total}] {record.id}: {what}, {record.seconds} s; about {duration(left)} left\n")

        summary = generate_rationales(questions, generator, store, config, RELEVANT, progress=progress)
    finally:
        close = getattr(generator, "close", None)
        if close is not None:
            close()
    streams.out.write(
        f"{summary.total} questions: {summary.already_done} already done, {summary.written} written, {len(summary.errors)} failed, "
        f"{summary.cut_off} stopped by the memory window, {duration(summary.seconds)}\n"
    )
    for question_id, message in summary.errors:
        streams.out.write(f"  failed {question_id}: {message}\n")
    if summary.cut_off:
        streams.out.write("a rationale that filled the memory window may not end with its final answer; it is kept as written\n")
    if summary.errors:
        streams.out.write("run the same command again to try the failed questions once more\n")
        return 2
    return 0


def _export(args: argparse.Namespace, streams: Streams) -> int:
    questions = read_questions(args.questions)
    store = RationaleStore(args.rationales)
    if not store.records_path.exists():
        raise StoreError(f"{args.rationales} holds no rationales; run the generate command first")
    records = store.load()
    queries = queries_for(questions, records)
    export_queries(queries, args.out)
    cut = sum(1 for q in questions if records[q.id].finish_reason == "length")
    streams.out.write(f"wrote {len(queries)} queries to {args.out}\n")
    if cut:
        streams.out.write(f"{cut} of them filled the memory window and may lack their final answer line\n")
    return 0
