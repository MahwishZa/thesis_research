"""The commands of ``python -m answer_evaluation``: answer the questions with every system, and report the accuracies.

    run     answer every question with the chosen systems, resumable; the filter judges the re-ranked snippets (needs torch)
    report  accuracy per system, the best k on validation data, and the paired comparison of two systems

The retrieval folders are named by their k: ``--retrieved 4=folder4 --retrieved 8=folder8``. A test split needs ``--ledger`` and
``--system``; the ledger refuses a second scoring of the same system on the same test set. Exit status: 0 done, 1 a problem
with the arguments or the data, 2 stopped or some answers failed (run the same command again), 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rationale_query.batch import RunAborted
from rationale_query.cli import Streams, duration, tolerate_unencodable
from rationale_query.generator import RESULT_RELEVANT, GenerationError, GenerationSettings, LlamaCppGenerator, text_sha256
from rationale_query.store import StoreError, load_rationales, read_questions

from .evidence import FILTER_ARMS, NO_FILTER, WITH_FILTER, EvidenceProvider, read_evidence, retrieval_top_k
from .ledger import AlreadyScored, Ledger, LedgerError
from .paired_tests import StatsError
from .prompt import EVIDENCE_HEADING, INSTRUCTION
from .reader import FINAL_ANSWER
from .records import RELEVANT, AnswerRecord, AnswerStore
from .results import accuracy_table, best_k, compare_systems
from .runner import NO_RETRIEVAL, Arm, EvalQuestion, run_arms

DEFAULT_MAX_NEW_TOKENS = 768
RUN_RELEVANT = RELEVANT + ("filter", "retrieval")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="answer_evaluation", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="answer the questions with every chosen system (resumable)")
    run.add_argument("--questions", type=Path, required=True, help="JSON lines of id and question (from data_roles export)")
    run.add_argument("--answers", type=Path, required=True, help="JSON lines of id and answer letter (from data_roles export)")
    run.add_argument("--rationales", type=Path, required=True, help="the output folder of rationale_query generate")
    run.add_argument("--retrieved", action="append", default=[], metavar="K=FOLDER", help="a balanced_retrieval retrieve folder made with top_k = K; repeat for each k")
    run.add_argument("--arms", nargs="+", choices=(NO_RETRIEVAL,) + FILTER_ARMS, default=[NO_RETRIEVAL, NO_FILTER, WITH_FILTER])
    run.add_argument("--filter-model", type=Path, default=None, help="a trained filter folder; needed for the arm rag2")
    run.add_argument("--filter-batch-size", type=int, default=8)
    run.add_argument("--filter-device", default=None)
    run.add_argument("--out", type=Path, required=True, help="the output folder (new, or the same as an earlier run)")
    run.add_argument("--dataset", required=True, help="a name for the questions, such as medqa; it is bound to the folder")
    run.add_argument("--split", required=True, help="validation or test; it is bound to the folder")
    run.add_argument("--ledger", type=Path, default=None, help="the test-once ledger; required for the test split")
    run.add_argument("--system", default=None, help="the name of the system as the ledger records it; required for the test split")
    run.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    run.add_argument("--model", type=Path, required=True, help="path of the GGUF model file")
    run.add_argument("--model-sha256", help="stop unless the file has this SHA-256")
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--n-ctx", type=int, default=8192)
    run.add_argument("--n-threads", type=int, default=None)
    run.add_argument("--n-gpu-layers", type=int, default=0)
    run.add_argument("--quiet", action="store_true", help="no line per answer")
    run.add_argument("--verbose", action="store_true", help="show llama.cpp's own messages")
    report = sub.add_parser("report", help="accuracy per system and, if asked, the paired comparison of two systems")
    report.add_argument("--out", type=Path, required=True, help="the output folder of a run")
    report.add_argument("--baseline", nargs=2, metavar=("ARM", "K"), help="the system to compare against")
    report.add_argument("--proposed", nargs=2, metavar=("ARM", "K"), help="the system that is tested against the baseline")
    report.add_argument("--choose-k", metavar="ARM", help="the k with the best accuracy for this arm (validation data only)")
    report.add_argument("--seed", default="bootstrap")
    report.add_argument("--iterations", type=int, default=10000)
    return parser


def _retrieval_folders(args: argparse.Namespace) -> dict[int, Path]:
    folders: dict[int, Path] = {}
    for item in args.retrieved:
        key, separator, folder = item.partition("=")
        if not separator or not key.isdigit() or not folder or int(key) < 1:
            raise ValueError(f"--retrieved wants K=FOLDER with K a whole number from 1, got {item!r}")
        if int(key) in folders:
            raise ValueError(f"k = {key} is given twice")
        folders[int(key)] = Path(folder)
    for k, folder in folders.items():
        if retrieval_top_k(folder) != k:
            raise ValueError(f"{folder} was made with top_k = {retrieval_top_k(folder)}, not {k}")
    return folders


def _arms(args: argparse.Namespace, ks: Sequence[int]) -> list[Arm]:
    arms: list[Arm] = []
    for name in args.arms:
        if name == NO_RETRIEVAL:
            arms.append(Arm(name, 0))
        else:
            if not ks:
                raise ValueError(f"the arm {name} needs at least one --retrieved K=FOLDER")
            arms.extend(Arm(name, k) for k in ks)
    return arms


def _default_generator(args: argparse.Namespace, streams: Streams) -> LlamaCppGenerator:
    settings = GenerationSettings(
        model_name=args.model.stem, max_new_tokens=args.max_new_tokens, n_ctx=args.n_ctx, n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers
    )
    streams.err.write(f"checking and loading {args.model.name} (the SHA-256 of a large file takes about half a minute)\n")
    return LlamaCppGenerator(args.model, settings, expected_sha256=args.model_sha256, verbose=args.verbose)


def _default_filter(args: argparse.Namespace, streams: Streams) -> Any:
    from rationale_filter.filter_model import FilterModel

    streams.err.write(f"loading the filter {args.filter_model}\n")
    return FilterModel.load(args.filter_model, device=args.filter_device)


def main(
    argv: Sequence[str] | None = None,
    *,
    generator_factory: Callable[[argparse.Namespace, Streams], Any] | None = None,
    filter_factory: Callable[[argparse.Namespace, Streams], Any] | None = None,
    streams: Streams | None = None,
) -> int:
    streams = streams or Streams(sys.stdout, sys.stderr)
    tolerate_unencodable(streams.out, streams.err)
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args, streams, generator_factory or _default_generator, filter_factory or _default_filter)
        return _report(args, streams)
    except RunAborted as exc:
        streams.err.write(f"stopped: {exc}\n")
        return 2
    except KeyboardInterrupt:
        streams.err.write("\ninterrupted; run the same command again to continue\n")
        return 130
    except (StoreError, GenerationError, AlreadyScored, LedgerError, StatsError, OSError, ValueError, ImportError) as exc:
        streams.err.write(f"error: {exc}\n")
        return 1


def _run(args: argparse.Namespace, streams: Streams, generator_factory: Callable[..., Any], filter_factory: Callable[..., Any]) -> int:
    folders = _retrieval_folders(args)
    arms = _arms(args, sorted(folders))
    if WITH_FILTER in args.arms and args.filter_model is None:
        raise ValueError("the arm rag2 needs --filter-model")
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[: args.limit]
    from rationale_filter.labelling import read_answers  # the answer key reader lives with the labelling that first needed it

    gold = read_answers(args.answers)
    missing = [q.id for q in questions if q.id not in gold]
    if missing:
        raise StoreError(f"{missing[0]} has no answer key")
    rationales = load_rationales(args.rationales, questions)
    items = [EvalQuestion(q.id, q.text, gold[q.id], rationales[q.id]) for q in questions]
    lists = {k: read_evidence(folder, questions, rationales) for k, folder in folders.items()}
    ledger = Ledger(args.ledger) if args.ledger is not None else None

    generator = generator_factory(args, streams)
    judge = None
    try:
        provider = EvidenceProvider(lists)
        filter_config: dict[str, Any] | None = None
        if WITH_FILTER in args.arms:
            from rationale_filter.training import base_files_sha256

            judge = filter_factory(args, streams)
            provider = EvidenceProvider(lists, lambda texts: judge.judge_many(texts, batch_size=args.filter_batch_size))
            filter_config = base_files_sha256(args.filter_model)
        config = {
            "dataset": args.dataset,
            "split": args.split,
            "generator": {name: generator.config.get(name) for name in RESULT_RELEVANT},
            "answer_prompt_sha256": text_sha256(INSTRUCTION + EVIDENCE_HEADING),
            "reader_pattern": FINAL_ANSWER.pattern,
            "filter": filter_config,
            "retrieval": {str(k): str(folder) for k, folder in sorted(folders.items())},
        }
        store = AnswerStore(args.out)
        begun = time.perf_counter()
        written = 0
        pending = len(items) * len(arms) - sum(1 for r in store.load().values() if r.status != "error") if store.config_path.exists() else len(items) * len(arms)

        def progress(position: int, total: int, record: AnswerRecord) -> None:
            nonlocal written
            written += 1
            if args.quiet:
                return
            left = (time.perf_counter() - begun) / written * max(0, pending - written)
            what = record.status if record.status != "ok" else ("correct" if record.correct else "wrong")
            streams.err.write(f"[{position}/{total}] {record.id} {what}; about {duration(left)} left\n")

        summary = run_arms(items, arms, provider, generator, store, config, RUN_RELEVANT, ledger=ledger, system=args.system, progress=progress)
    finally:
        for model in (judge, generator):
            close = getattr(model, "close", None)
            if close is not None:
                close()
    streams.out.write(f"{summary.total} answers: {summary.already_done} already done, {summary.written} written, {len(summary.errors)} failed, {duration(summary.seconds)}\n")
    if judge is not None or provider.considered:
        streams.out.write("snippets kept by the filter: " + json.dumps(provider.summary(), sort_keys=True) + "\n")
    for answer, message in summary.errors:
        streams.out.write(f"  failed {answer}: {message}\n")
    if summary.errors:
        streams.out.write("run the same command again to try the failed answers once more\n")
        return 2
    return 0


def _report(args: argparse.Namespace, streams: Streams) -> int:
    store = AnswerStore(args.out)
    if not store.config_path.exists():
        raise StoreError(f"{args.out} holds no answers; run the run command first")
    records = list(store.load().values())
    report: dict[str, Any] = {"config": {name: store.read_config().get(name) for name in ("dataset", "split")}, "accuracy": accuracy_table(records)}
    if args.choose_k:
        ks = sorted({r.k for r in records if r.arm == args.choose_k and r.status == "ok"})
        report["best_k"] = best_k(records, args.choose_k, ks)
    if bool(args.baseline) != bool(args.proposed):
        raise ValueError("give both --baseline and --proposed, or neither")
    if args.baseline:
        report["comparison"] = {
            "baseline": args.baseline,
            "proposed": args.proposed,
            **compare_systems(
                records, (args.baseline[0], int(args.baseline[1])), (args.proposed[0], int(args.proposed[1])), seed=args.seed, iterations=args.iterations
            ),
        }
    streams.out.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0
