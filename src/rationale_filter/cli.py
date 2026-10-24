"""The commands of ``python -m rationale_filter``: make the pair records, then the labels and the training file.

    label         for each question, label its first N re-ranked snippets (resumable); prints the seconds per pair, so a run
                  with ``--limit 20`` is the timing pilot
    labels        τ, the labels and a summary from the records; with ``--training-file`` also the training file for the filter
    prepare-model add the two label tokens to a base Flan-T5 and save it into a new folder
    memory-test   one training step on the longest inputs: does a run fit, and how long does a step take
    train         fine-tune the prepared model on a training file (resumable), validating after every epoch
    evaluate      accuracy, accuracy per class, majority rate and AUC of a trained filter on a labelled file

Inputs are the files of the earlier steps: the questions and the answer key (``python -m data_roles export``), the rationales
(``python -m rationale_query generate``) and the evidence lists (``python -m balanced_retrieval retrieve``).
Exit status: 0 done; 1 the setup is wrong (message on stderr); 2 finished or stopped with failed pairs (a mistyped option also
gives 2, with a usage message); 130 interrupted (run the same command again to continue).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from answer_evaluation.prompt import EVIDENCE_HEADING, INSTRUCTION
from answer_evaluation.reader import FINAL_ANSWER
from rationale_query.batch import RunAborted
from rationale_query.cli import Streams, duration, tolerate_unencodable
from rationale_query.generator import RESULT_RELEVANT, GenerationError, GenerationSettings, LlamaCppGenerator, text_sha256
from rationale_query.store import StoreError, read_questions

from .label_records import RELEVANT, PairRecord, PairStore, label_records, summarise
from .filter_model import DOC_STRIDE, MAX_ANSWER_LENGTH, MAX_SEQ_LENGTH, FilterModel, FilterModelError, prepare_model
from .labelling import PairTask, build_tasks, label_pairs, read_answers
from .scorer import LlamaLogprobScorer
from .training import TrainingError, TrainSettings, evaluate_examples, memory_test, read_examples, train, write_report
from .training_data import label_balance, training_records, write_training_file

#: The cap on the answers written with a snippet (decision D15). The rationale of Part 1 is cut at its own cap (512 by default).
DEFAULT_MAX_NEW_TOKENS = 768


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rationale_filter", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("label", "write one record per question and snippet (resumable)"), ("labels", "make the labels from the records")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--questions", type=Path, required=True, help="JSON lines of id and question (from data_roles export)")
        command.add_argument("--answers", type=Path, required=True, help="JSON lines of id and answer letter (from data_roles export)")
        command.add_argument("--rationales", type=Path, required=True, help="the output folder of rationale_query generate")
        command.add_argument("--retrieved", type=Path, required=True, help="the output folder of balanced_retrieval retrieve")
        command.add_argument("--out", type=Path, required=True, help="the folder of the pair records (new, or the same as an earlier run)")
        command.add_argument("--snippets", type=int, default=4, help="snippets labelled per question: the first N re-ranked (decision D19)")
        command.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    label = sub.choices["label"]
    label.add_argument("--dataset", required=True, help="a name for the questions, such as medqa; it is bound to the folder")
    label.add_argument("--split", required=True, help="train or validation; it is bound to the folder")
    label.add_argument("--model", type=Path, required=True, help="path of the GGUF model file")
    label.add_argument("--model-sha256", help="stop unless the file has this SHA-256")
    label.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="cap on the answer written with a snippet")
    label.add_argument("--n-ctx", type=int, default=8192, help="context of the model that writes the answers")
    label.add_argument("--scorer-n-ctx", type=int, default=4096, help="context of the model that scores the rationale")
    label.add_argument("--n-threads", type=int, default=None)
    label.add_argument("--n-gpu-layers", type=int, default=0)
    label.add_argument("--quiet", action="store_true", help="no line per pair")
    label.add_argument("--verbose", action="store_true", help="show llama.cpp's own messages")
    labels = sub.choices["labels"]
    labels.add_argument("--training-file", type=Path, default=None, help="also write the filter's training file here (never overwrites)")
    labels.add_argument("--dataset-name", default=None, help="the dataset_name field of the training file (default: the name bound to the folder)")

    prepare = sub.add_parser("prepare-model", help="add the two label tokens to a base Flan-T5 (needs torch and transformers)")
    prepare.add_argument("--base", required=True, help="a hub name such as google/flan-t5-base, or a local folder")
    prepare.add_argument("--out", type=Path, required=True, help="a new, empty folder")
    prepare.add_argument("--local-files-only", action="store_true", help="never go to the hub")

    for name, text in (("train", "fine-tune the prepared model (resumable)"), ("memory-test", "one training step on the longest inputs")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--model", type=Path, required=True, help="the folder made by prepare-model")
        command.add_argument("--train-file", type=Path, required=True, help="the training file (from the labels command)")
        command.add_argument("--epochs", type=int, default=40, help="the authors' value")
        command.add_argument("--batch-size", type=int, default=16, help="per-device batch size; the authors' value")
        command.add_argument("--learning-rate", type=float, default=3e-5, help="the authors' value")
        command.add_argument("--weight-decay", type=float, default=0.0)
        command.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
        command.add_argument("--doc-stride", type=int, default=DOC_STRIDE)
        command.add_argument("--grad-accum", type=int, default=1, help="batches per optimizer step; 1 is the authors' setting")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default=None, help="cuda, cpu, ...; default: cuda when there is one")
    trainer = sub.choices["train"]
    trainer.add_argument("--validation-file", type=Path, default=None, help="a labelled file; the filter is scored on it after every epoch")
    trainer.add_argument("--out", type=Path, required=True, help="the output folder (new, or the same as an earlier run)")
    trainer.add_argument("--eval-batch-size", type=int, default=8)
    trainer.add_argument("--save-every", type=int, default=0, help="also keep the model of every N-th epoch (0: only last, best and final)")
    trainer.add_argument("--quiet", action="store_true", help="no line per epoch")

    evaluator = sub.add_parser("evaluate", help="score a trained filter on a labelled file")
    evaluator.add_argument("--model", type=Path, required=True, help="a trained folder, for example the best or final folder of a run")
    evaluator.add_argument("--file", type=Path, required=True, help="a labelled file, in the format of the training file")
    evaluator.add_argument("--report-out", type=Path, default=None, help="write the report and the predictions here (never overwrites)")
    evaluator.add_argument("--batch-size", type=int, default=8)
    evaluator.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    evaluator.add_argument("--doc-stride", type=int, default=DOC_STRIDE)
    evaluator.add_argument("--device", default=None)
    return parser


def _tasks(args: argparse.Namespace) -> list[PairTask]:
    if getattr(args, "split", None) == "test":
        raise ValueError("labels are made from training and validation questions only, never from a test split")
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[: args.limit]
    return build_tasks(questions, read_answers(args.answers), args.rationales, args.retrieved, args.snippets)


def _config(args: argparse.Namespace, generator_config: dict[str, Any], scorer_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "split": args.split,
        "generator": {name: generator_config.get(name) for name in RESULT_RELEVANT},
        "scorer": {name: scorer_config.get(name) for name in ("model_sha256", "prompt_format", "scored")},
        "answer_prompt_sha256": text_sha256(INSTRUCTION + EVIDENCE_HEADING),
        "reader_pattern": FINAL_ANSWER.pattern,
        "snippets_per_question": args.snippets,
    }


def _default_generator(args: argparse.Namespace, streams: Streams) -> LlamaCppGenerator:
    settings = GenerationSettings(
        model_name=args.model.stem, max_new_tokens=args.max_new_tokens, n_ctx=args.n_ctx, n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers
    )
    streams.err.write(f"checking and loading {args.model.name} to write the answers (the SHA-256 of a large file takes about half a minute)\n")
    return LlamaCppGenerator(args.model, settings, expected_sha256=args.model_sha256, verbose=args.verbose)


def _default_scorer(args: argparse.Namespace, streams: Streams) -> LlamaLogprobScorer:
    streams.err.write(f"loading {args.model.name} again to score the rationale (the weights are shared through the page cache)\n")
    return LlamaLogprobScorer(
        args.model, n_ctx=args.scorer_n_ctx, n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers, expected_sha256=args.model_sha256, verbose=args.verbose
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    generator_factory: Callable[[argparse.Namespace, Streams], Any] | None = None,
    scorer_factory: Callable[[argparse.Namespace, Streams], Any] | None = None,
    streams: Streams | None = None,
) -> int:
    streams = streams or Streams(sys.stdout, sys.stderr)
    tolerate_unencodable(streams.out, streams.err)
    args = _parser().parse_args(argv)
    try:
        if args.command == "label":
            return _label(args, streams, generator_factory or _default_generator, scorer_factory or _default_scorer)
        if args.command == "labels":
            return _labels(args, streams)
        if args.command == "prepare-model":
            return _prepare(args, streams)
        if args.command == "train":
            return _train(args, streams)
        if args.command == "memory-test":
            return _memory_test(args, streams)
        return _evaluate(args, streams)
    except RunAborted as exc:
        streams.err.write(f"stopped: {exc}\n")
        return 2
    except KeyboardInterrupt:
        streams.err.write("\ninterrupted; run the same command again to continue\n")
        return 130
    except (StoreError, GenerationError, FilterModelError, TrainingError, OSError, ValueError, ImportError) as exc:
        streams.err.write(f"error: {exc}\n")
        return 1


def _label(args: argparse.Namespace, streams: Streams, generator_factory: Callable[..., Any], scorer_factory: Callable[..., Any]) -> int:
    tasks = _tasks(args)
    store = PairStore(args.out)
    generator = generator_factory(args, streams)
    scorer = None
    try:
        scorer = scorer_factory(args, streams)
        config = _config(args, generator.config, scorer.config)
        pending = len(tasks) - sum(1 for r in store.load().values() if r.status != "error")
        begun = time.perf_counter()
        written = 0

        def progress(position: int, total: int, record: PairRecord) -> None:
            nonlocal written
            written += 1
            if args.quiet:
                return
            what = record.status if record.status != "skipped" else f"skipped ({record.reason})"
            left = (time.perf_counter() - begun) / written * max(0, pending - written)
            streams.err.write(f"[{position}/{total}] {record.id} {what}, {record.seconds} s; about {duration(left)} left\n")

        summary = label_pairs(tasks, generator, scorer, store, config, RELEVANT, progress=progress)
    finally:
        for model in (scorer, generator):
            close = getattr(model, "close", None)
            if close is not None:
                close()
    per_pair = f", {summary.seconds / summary.written:.1f} s per pair" if summary.written else ""
    streams.out.write(
        f"{summary.total} pairs: {summary.already_done} already done, {summary.written} written, {len(summary.errors)} failed, "
        f"{duration(summary.seconds)}{per_pair}\n"
    )
    for pair, message in summary.errors:
        streams.out.write(f"  failed {pair}: {message}\n")
    if summary.errors:
        streams.out.write("run the same command again to try the failed pairs once more\n")
        return 2
    return 0


def _labels(args: argparse.Namespace, streams: Streams) -> int:
    store = PairStore(args.out)
    if not store.config_path.exists():
        raise StoreError(f"{args.out} holds no pair records; run the label command first")
    config = store.read_config()
    tasks = _tasks(args)
    records = store.load()
    needed = {task.id for task in tasks}
    present = [records[key] for key in sorted(records) if key in needed]
    unfinished = sorted(needed - {r.id for r in present if r.status != "error"})
    if unfinished:
        streams.err.write(f"note: {len(unfinished)} pairs have no finished record yet (first: {unfinished[0]}); they are left out\n")
    tau, labelled = label_records(present)
    report = {"dataset": config.get("dataset"), "split": config.get("split"), "tau": tau, **summarise(present, labelled)}
    if args.training_file is not None:
        name = args.dataset_name or config.get("dataset")
        rows = training_records(labelled, tasks, name)
        write_training_file(args.training_file, rows)
        report["training_file"] = {"path": str(args.training_file), "records": len(rows), **label_balance(rows)}
    streams.out.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def _settings(args: argparse.Namespace) -> TrainSettings:
    return TrainSettings(
        model_dir=str(args.model), train_file=str(args.train_file), out_dir=str(getattr(args, "out", None) or "-"),
        validation_file=str(args.validation_file) if getattr(args, "validation_file", None) else None, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride, max_answer_length=MAX_ANSWER_LENGTH, gradient_accumulation_steps=args.grad_accum, seed=args.seed,
        eval_batch_size=getattr(args, "eval_batch_size", 8), save_every=getattr(args, "save_every", 0),
    )


def _prepare(args: argparse.Namespace, streams: Streams) -> int:
    info = prepare_model(args.base, args.out, local_files_only=args.local_files_only)
    streams.out.write(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return 0


def _train(args: argparse.Namespace, streams: Streams) -> int:
    settings = _settings(args)

    def progress(event: dict[str, Any]) -> None:
        if event["event"] != "epoch" or args.quiet:
            return
        valid = event.get("validation")
        text = f", validation accuracy {valid['accuracy']} (majority {valid['majority_rate']})" if valid else ""
        streams.err.write(f"[epoch {event['epoch']}/{settings.epochs}] loss {event['mean_loss']}{text}, {duration(event['seconds'])}\n")

    summary = train(settings, device=args.device, progress=progress)
    streams.out.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _memory_test(args: argparse.Namespace, streams: Streams) -> int:
    result = memory_test(_settings(args), device=args.device)
    streams.out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("fits") else 3


def _evaluate(args: argparse.Namespace, streams: Streams) -> int:
    examples = read_examples(args.file)
    filter_model = FilterModel.load(args.model, device=args.device, max_seq_length=args.max_seq_length, doc_stride=args.doc_stride)
    result = evaluate_examples(filter_model, examples, batch_size=args.batch_size)
    result["report"]["windowed_inputs"] = filter_model.windowed_inputs
    result["report"]["windowed_share"] = filter_model.windowed_share()
    if args.report_out is not None:
        write_report(args.report_out, result)
    streams.out.write(json.dumps(result["report"], indent=2, sort_keys=True) + "\n")
    return 0
