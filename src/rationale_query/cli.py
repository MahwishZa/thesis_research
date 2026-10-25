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
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO

from .generator import (
    CHAT_FORMATS,
    DEFAULT_CHAT_FORMAT,
    GenerationError,
    GenerationSettings,
    TransformersGenerator,
    render_model_input,
)
from .prompt import INSTRUCTION, LEAD_IN, SEPARATOR, build_rationale_prompt
from .query import export_queries, queries_for
from .run import RunAborted, generate_rationales
from .store import (
    RationaleRecord,
    RationaleStore,
    StoreError,
    question_hash,
    read_questions,
)

#: The settings that decide what a rationale is. A folder made with other values is refused.
RELEVANT = ("model_sha256", "dtype", "chat_format", "n_ctx", "decoding", "prompt_sha256")


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
    gen.add_argument("--model", type=Path, required=True, help="the folder of the Meerkat-7B weights (as downloaded from Hugging Face)")
    gen.add_argument("--model-sha256", help="stop unless the model folder has this hash (the generate command prints it)")
    gen.add_argument("--n-ctx", type=int, default=GenerationSettings.n_ctx, help="memory window in tokens; the length Meerkat-7B was trained with")
    gen.add_argument("--chat-format", choices=CHAT_FORMATS, default=DEFAULT_CHAT_FORMAT, help="none: the paper's text alone (default); meerkat: Meerkat's own template, for a labelled check")
    gen.add_argument("--n-threads", type=int, default=None, help="CPU threads (default: PyTorch's choice)")
    gen.add_argument("--max-gpu-gib", type=float, default=None, help="memory the graphics card may use, for example 3.2 for a 4 GiB card")
    gen.add_argument("--max-cpu-gib", type=float, default=None, help="memory the CPU side may use, for example 10; the rest of the model goes to the disk")
    gen.add_argument("--offload-dir", type=Path, default=None, help="a folder on the disk for the parts of the model that fit nowhere else. On the author's Windows laptop (RTX 2050, 15 GB) this crashed inside PyTorch; --max-gpu-gib 3.0 --max-cpu-gib 11 without it worked")
    gen.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    gen.add_argument("--quiet", action="store_true", help="no line per question")

    show = sub.add_parser("show-prompt", help="print what the model is given for one question")
    show.add_argument("--question", required=True, help="the question, with its options")
    show.add_argument("--chat-format", choices=CHAT_FORMATS, default=DEFAULT_CHAT_FORMAT)

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
    streams.out.write(f"What the model reads (chat format: {args.chat_format}; no system message; the tokenizer adds the start token once):\n\n")
    streams.out.write(render_model_input(prompt, args.chat_format) + "\n")
    return 0


def _default_generator(args: argparse.Namespace, streams: Streams) -> TransformersGenerator:
    settings = GenerationSettings(
        n_ctx=args.n_ctx, n_threads=args.n_threads, max_gpu_gib=args.max_gpu_gib, max_cpu_gib=args.max_cpu_gib,
        offload_dir=str(args.offload_dir) if args.offload_dir else None, chat_format=args.chat_format,
    )
    streams.err.write(f"reading {args.model.name} to identify it (a few minutes for 29 GB), then loading it in BF16; parts that do not fit on the graphics card run on the CPU, which is slow\n")
    return TransformersGenerator(args.model, settings, expected_sha256=args.model_sha256)


def _generate(args: argparse.Namespace, streams: Streams, factory: Callable[[argparse.Namespace, Streams], Any]) -> int:
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[: args.limit]
    store = RationaleStore(args.out)
    generator = factory(args, streams)
    try:
        config = {**generator.config, "prompt_sha256": sha256((INSTRUCTION + SEPARATOR + LEAD_IN).encode("utf-8")).hexdigest()}
        saved = store.load() if store.records_path.exists() else {}
        pending = sum(1 for q in questions if not (q.id in saved and saved[q.id].status == "ok" and saved[q.id].question_sha256 == question_hash(q.text)))
        begun = time.perf_counter()
        done = 0

        def progress(position: int, total: int, record: RationaleRecord) -> None:
            nonlocal done
            done += 1
            began[0] = time.perf_counter()
            if args.quiet:
                return
            left = (time.perf_counter() - begun) / done * max(0, pending - done)
            what = record.error if record.status == "error" else f"{record.completion_tokens} tokens, {record.finish_reason}"
            streams.err.write(f"[{position}/{total}] {record.id}: {what}, {record.seconds} s; about {duration(left)} left\n")

        began = [time.perf_counter()]  # when the current question began

        if not args.quiet and hasattr(generator, "progress"):

            def tokens_written(count: int) -> None:
                streams.err.write(f"    ... {count} tokens written, {duration(time.perf_counter() - began[0])} since this question began\n")

            generator.progress = tokens_written
        summary = generate_rationales(questions, generator, store, config, RELEVANT, progress=progress)
    finally:
        close = getattr(generator, "close", None)
        if close is not None:
            close()
    streams.out.write(f"model: {generator.config.get('model_folder')}, hash {generator.config.get('model_sha256')}\n")
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
