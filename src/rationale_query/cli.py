"""``python -m rationale_query``: write rationales, encode them as queries, export them, and describe them.

    show-prompt     print the exact text that is sent to the model for one question (no model needed)
    generate        write one rationale per question with a local GGUF model (resumable)
    encode          turn the rationales into MedCPT query vectors
    export-queries  write the rationales as the JSON list that the authors' retriever reads
    report          describe one or more output folders and compare them

Exit status: 0 done; 1 the setup is wrong (message on stderr); 2 finished or stopped with failed questions (a mistyped
option also gives 2, with a usage message); 130 interrupted (run the same command again to continue).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .diagnostics import agreement, format_table, provenance, summarise
from .generator import (
    CHAT_FORMATS,
    RESULT_RELEVANT,
    GenerationError,
    GenerationSettings,
    LlamaCppGenerator,
    llama3_prompt_text,
    text_sha256,
)
from .prompt import build_rationale_prompt
from .query import dense_query
from .run import RunAborted, generate_rationales, plan
from .store import (
    CONFIG_FILE,
    RATIONALES_FILE,
    Question,
    RationaleRecord,
    RationaleStore,
    StoreError,
    load_rationales,
    read_questions,
    write_atomic,
)

VECTORS_FILE = "query_vectors.npy"
VECTORS_META = "query_vectors.json"

_DEFAULTS = GenerationSettings()  # the one place the default settings are written; the options below read them


@dataclass
class Streams:
    out: TextIO
    err: TextIO


def duration(seconds: float) -> str:
    """Seconds as a short phrase: '45 s', '3 min 05 s', '2 h 07 min'."""
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def tolerate_unencodable(*streams: TextIO) -> None:
    """Never fail on a character that a console cannot show (a Windows console often cannot): print a ? instead."""
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _questions(args: argparse.Namespace) -> list[Question]:
    """The questions of ``--questions``, cut to ``--limit`` when that is given."""
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[: args.limit]
    return questions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rationale_query", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-prompt", help="print the text sent to the model for one question")
    source = show.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="the question text, with its options inline")
    source.add_argument("--questions", type=Path, help="a questions file (JSON lines with id and question)")
    show.add_argument("--id", help="with --questions: the question to show (default: the first)")

    gen = sub.add_parser("generate", help="write one rationale per question (resumable)")
    gen.add_argument("--questions", type=Path, required=True, help="JSON lines, one {id, question} per line")
    gen.add_argument("--out", type=Path, required=True, help="output folder (new, or the same as an earlier run)")
    gen.add_argument("--model", type=Path, required=True, help="path of the GGUF model file")
    gen.add_argument("--model-name", help="a label for the model (default: the file name without extension)")
    gen.add_argument("--model-sha256", help="stop unless the file has this SHA-256")
    gen.add_argument("--chat-format", choices=CHAT_FORMATS, default=_DEFAULTS.chat_format, help="llama-3 or gguf")
    gen.add_argument("--max-new-tokens", type=int, default=_DEFAULTS.max_new_tokens)
    gen.add_argument("--n-ctx", type=int, default=_DEFAULTS.n_ctx)
    gen.add_argument("--n-batch", type=int, default=_DEFAULTS.n_batch)
    gen.add_argument("--n-ubatch", type=int, default=_DEFAULTS.n_ubatch)
    gen.add_argument("--n-threads", type=int, default=_DEFAULTS.n_threads, help="default: the library's choice, which is recorded")
    gen.add_argument("--n-gpu-layers", type=int, default=_DEFAULTS.n_gpu_layers)
    gen.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    gen.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    gen.add_argument("--allow-template-mismatch", action="store_true")
    gen.add_argument("--quiet", action="store_true", help="no progress lines")
    gen.add_argument("--verbose", action="store_true", help="show llama.cpp's own messages (why a model does not load)")

    enc = sub.add_parser("encode", help="turn the rationales into MedCPT query vectors")
    enc.add_argument("--rationales", type=Path, required=True, help="the output folder of generate")
    enc.add_argument("--questions", type=Path, required=True, help="the same questions file")
    enc.add_argument("--encoder", default="ncbi/MedCPT-Query-Encoder")
    enc.add_argument("--revision", default=None)
    enc.add_argument("--device", default="auto", help="auto, cpu or cuda")
    enc.add_argument("--batch-size", type=int, default=1, help="1 matches the authors' call; more is faster")
    enc.add_argument("--local-files-only", action="store_true", help="do not contact the Hugging Face hub")
    enc.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    enc.add_argument("--overwrite", action="store_true")

    exp = sub.add_parser("export-queries", help="write the JSON list of queries the authors' retriever reads")
    exp.add_argument("--rationales", type=Path, required=True)
    exp.add_argument("--questions", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")

    rep = sub.add_parser("report", help="describe output folders and compare them")
    rep.add_argument("folders", type=Path, nargs="+")
    rep.add_argument("--json", type=Path, default=None, help="also write the numbers to this file")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    generator_factory: Callable[[argparse.Namespace, Streams], Any] | None = None,
    encoder_factory: Callable[[argparse.Namespace], Any] | None = None,
    streams: Streams | None = None,
) -> int:
    streams = streams or Streams(sys.stdout, sys.stderr)
    tolerate_unencodable(streams.out, streams.err)
    args = _parser().parse_args(argv)
    try:
        if args.command == "show-prompt":
            return _show_prompt(args, streams)
        if args.command == "generate":
            return _generate(args, streams, generator_factory or _default_generator)
        if args.command == "encode":
            return _encode(args, streams, encoder_factory or _default_encoder)
        if args.command == "export-queries":
            return _export(args, streams)
        return _report(args, streams)
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
    if args.question is not None:
        text = args.question
    else:
        questions = read_questions(args.questions)
        if args.id is None:
            text = questions[0].text
        else:
            chosen = [q for q in questions if q.id == args.id]
            if not chosen:
                raise StoreError(f"no question with id {args.id} in {args.questions}")
            text = chosen[0].text
    prompt = build_rationale_prompt(text)
    streams.out.write(f"--- sent to the model as one user turn ---\n{prompt}\n")
    streams.out.write(f"--- the same turn in Llama-3 format ---\n{llama3_prompt_text(prompt)}\n")
    return 0


def _default_generator(args: argparse.Namespace, streams: Streams) -> LlamaCppGenerator:
    settings = GenerationSettings(
        model_name=args.model_name or args.model.stem,
        chat_format=args.chat_format,
        max_new_tokens=args.max_new_tokens,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        allow_template_mismatch=args.allow_template_mismatch,
    )

    def hashing(done: int, total: int) -> None:
        streams.err.write(f"  hashing the model file: {done / 2**30:.1f} of {total / 2**30:.1f} GiB\n")

    streams.err.write(f"checking and loading {args.model.name} (the SHA-256 of a large file takes about half a minute)\n")
    try:
        return LlamaCppGenerator(
            args.model, settings, expected_sha256=args.model_sha256, progress=hashing, verbose=args.verbose
        )
    except ValueError as exc:  # the library raises it when it cannot load a file, and keeps its own reason to itself
        hint = "" if args.verbose else " (add --verbose to see llama.cpp's own messages)"
        raise ValueError(f"{exc}{hint}") from exc


def _generate(args: argparse.Namespace, streams: Streams, factory: Callable[[argparse.Namespace, Streams], Any]) -> int:
    questions = _questions(args)
    store = RationaleStore(args.out)
    pending = len(plan(questions, store.load(), args.out).pending)  # also stops here, before the model is loaded, if a question changed
    if not pending:
        streams.out.write(f"nothing to do: all {len(questions)} questions already have a rationale in {args.out}\n")
        return 0

    begun = 0.0
    written = 0

    def progress(position: int, total: int, record: RationaleRecord) -> None:
        nonlocal written
        written += 1
        if args.quiet:
            return
        what = f"{record.completion_tokens} tokens" if record.status == "ok" else record.error
        left = (time.perf_counter() - begun) / written * (pending - written)
        streams.err.write(f"[{position}/{total}] {record.id} {what}, {record.seconds} s; about {duration(left)} left\n")

    generator = factory(args, streams)
    try:
        begun = time.perf_counter()
        summary = generate_rationales(questions, generator, store, RESULT_RELEVANT, progress=progress)
    finally:
        close = getattr(generator, "close", None)
        if close is not None:
            close()
    if summary.removed_cut_off_line:
        streams.out.write("note: the last line of the records file was cut short by an earlier interruption and was removed\n")
    streams.out.write(
        f"{summary.total} questions: {summary.already_done} already done, {summary.written} written, "
        f"{len(summary.errors)} failed, {duration(summary.seconds)}\n"
    )
    for question_id, message in summary.errors:
        streams.out.write(f"  failed {question_id}: {message}\n")
    if summary.errors:
        streams.out.write("run the same command again to try the failed questions once more\n")
        return 2
    return 0


def _default_encoder(args: argparse.Namespace) -> Any:
    try:
        from .encoder import QueryEncoder

        return QueryEncoder.from_pretrained(
            args.encoder,
            revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
    except ImportError as exc:
        raise ImportError(f"encoding needs numpy, torch and transformers ({exc})") from exc


def _encode(args: argparse.Namespace, streams: Streams, factory: Callable[[argparse.Namespace], Any]) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(f"encoding needs numpy, torch and transformers ({exc})") from exc

    vectors_path = args.rationales / VECTORS_FILE
    meta_path = args.rationales / VECTORS_META
    if (vectors_path.exists() or meta_path.exists()) and not args.overwrite:
        raise StoreError(f"{args.rationales} already holds query vectors; add --overwrite to replace them")
    questions = _questions(args)
    rationales = load_rationales(args.rationales, questions)
    texts = [dense_query(rationales[q.id]) for q in questions]
    encoder = factory(args)
    counts = encoder.token_counts(texts)
    vectors = encoder.encode(texts)
    if vectors.shape[0] != len(texts) or vectors.dtype != np.float32:
        raise ValueError(f"the encoder returned {vectors.shape} {vectors.dtype} for {len(texts)} texts")
    meta = dict(
        encoder.info,
        shape=list(vectors.shape),
        ids=[q.id for q in questions],
        token_counts=counts,
        truncated=[c > encoder.max_length for c in counts],
        rationales_sha256=text_sha256(json.dumps(texts, ensure_ascii=False)),
    )
    buffer = io.BytesIO()
    np.save(buffer, vectors)
    write_atomic(vectors_path, buffer.getvalue())
    write_atomic(meta_path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    streams.out.write(
        f"{vectors.shape[0]} query vectors of {vectors.shape[1]} numbers written to {vectors_path}; "
        f"{sum(meta['truncated'])} longer than {encoder.max_length} tokens were cut by the encoder\n"
    )
    return 0


def _export(args: argparse.Namespace, streams: Streams) -> int:
    questions = _questions(args)
    rationales = load_rationales(args.rationales, questions)
    queries = [dense_query(rationales[q.id]) for q in questions]
    write_atomic(args.out, json.dumps(queries, ensure_ascii=True) + "\n")  # plain ASCII, so any locale reads it
    streams.out.write(f"{len(queries)} queries written to {args.out}, in the order of the questions file\n")
    return 0


def _report(args: argparse.Namespace, streams: Streams) -> int:
    labels: dict[str, Path] = {}
    for folder in args.folders:
        label = folder.name or str(folder)
        while label in labels:
            label += "'"
        labels[label] = folder
    summaries: dict[str, dict] = {}
    configs: dict[str, dict] = {}
    records_by_label: dict[str, list[RationaleRecord]] = {}
    for label, folder in labels.items():
        if not (folder / RATIONALES_FILE).exists():
            raise StoreError(f"no {RATIONALES_FILE} in {folder}")
        records = list(RationaleStore(folder).load().values())
        counts = None
        meta_path = folder / VECTORS_META
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            counts = dict(zip(meta["ids"], meta["token_counts"], strict=True))
        summaries[label] = summarise(records, counts)
        records_by_label[label] = records
        if (folder / CONFIG_FILE).exists():
            configs[label] = RationaleStore(folder).read_config()
    streams.out.write(format_table(summaries) + "\n")
    for label, config in configs.items():
        streams.out.write(provenance(label, config) + "\n")
    pairs: dict[str, dict] = {}
    names = list(records_by_label)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairs[f"{left} vs {right}"] = agreement(records_by_label[left], records_by_label[right])
    for pair, result in pairs.items():
        streams.out.write(f"same final option, {pair}: {result['same_option']} of {result['compared']}\n")
    if args.json is not None:
        write_atomic(args.json, json.dumps({"summaries": summaries, "agreement": pairs, "configs": configs}, indent=2) + "\n")
    return 0
