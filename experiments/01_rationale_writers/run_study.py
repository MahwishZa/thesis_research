"""Experiment 01: let each writer in settings.json write the rationales of one questions file, then compare them.

    python experiments/01_rationale_writers/run_study.py --questions PATH [--run-name NAME] [--writers NAME ...]
                                                         [--models-dir DIR] [--limit N] [--encode]

The command can be repeated after an interruption or a failure: finished questions are not written again, and a
writer that is complete is passed over at once. The work itself is done by the command line of src/rationale_query;
this file only reads the settings and calls it. It stops at the first step that does not end well.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SETTINGS = HERE / "settings.json"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # a writer name and a run name become folder names


class StudyError(ValueError):
    """The settings or the options cannot be used."""


def _package():
    """The rationale_query modules, importable from the repository even when it is not installed."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from rationale_query import cli, generator, store

    return cli, generator, store


def load_settings(path: Path = SETTINGS) -> dict:
    """The settings, checked: at least one writer, each with a folder-safe unique name, a chat format and a hash or null."""
    _, generator, _ = _package()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StudyError(f"cannot read {path}: {exc}") from exc
    writers = data.get("writers") if isinstance(data, dict) else None
    if not isinstance(writers, list) or not writers:
        raise StudyError(f"{path} must list at least one writer")
    seen: set[str] = set()
    for writer in writers:
        for key in ("name", "model_file", "chat_format"):
            if not isinstance(writer.get(key), str) or not writer[key]:
                raise StudyError(f"every writer needs a text '{key}': {writer}")
        if not NAME.fullmatch(writer["name"]):
            raise StudyError(f"the writer name {writer['name']!r} must be letters, digits, dots, dashes and underscores")
        if writer["name"] in seen:
            raise StudyError(f"the writer name {writer['name']} appears twice")
        seen.add(writer["name"])
        if writer["chat_format"] not in generator.CHAT_FORMATS:
            raise StudyError(f"{writer['name']}: chat_format must be one of {', '.join(generator.CHAT_FORMATS)}")
        sha = writer.get("sha256")
        if sha is not None and not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise StudyError(f"{writer['name']}: sha256 must be 64 lower-case hexadecimal digits or null")
    return data


def choose_writers(settings: dict, wanted: Sequence[str] | None, models_dir: Path) -> tuple[list[dict], list[dict]]:
    """The writers to run and the writers passed over because their model file is not in ``models_dir``.

    Named writers must exist and have their file. Without names, every writer whose file is present runs.
    """
    by_name = {w["name"]: w for w in settings["writers"]}
    if wanted:
        unknown = [name for name in wanted if name not in by_name]
        if unknown:
            raise StudyError(f"unknown writer {', '.join(unknown)}; the settings list: {', '.join(by_name)}")
        chosen = [by_name[name] for name in dict.fromkeys(wanted)]
        missing = [w["model_file"] for w in chosen if not (models_dir / w["model_file"]).is_file()]
        if missing:
            raise StudyError(f"model file not found in {models_dir}: {', '.join(missing)}")
        return chosen, []
    present = [w for w in settings["writers"] if (models_dir / w["model_file"]).is_file()]
    if not present:
        files = ", ".join(w["model_file"] for w in settings["writers"])
        raise StudyError(f"none of the model files is in {models_dir} (looked for: {files})")
    return present, [w for w in settings["writers"] if w not in present]


def generate_command(
    writer: dict, *, questions: Path, out: Path, models_dir: Path, limit: int | None, allow_template_mismatch: bool = False
) -> list[str]:
    argv = [
        "generate",
        "--questions",
        str(questions),
        "--out",
        str(out),
        "--model",
        str(models_dir / writer["model_file"]),
        "--model-name",
        writer["name"],
        "--chat-format",
        writer["chat_format"],
    ]
    if writer.get("sha256"):
        argv += ["--model-sha256", writer["sha256"]]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if allow_template_mismatch:
        argv.append("--allow-template-mismatch")
    return argv


def main(argv: Sequence[str] | None = None, *, cli_main: Callable[..., int] | None = None) -> int:
    cli, generator, store = _package()
    cli_main = cli_main or cli.main
    cli.tolerate_unencodable(sys.stdout, sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True, help="JSON lines, one {id, question} per line")
    parser.add_argument("--run-name", help="name of this run (default: the name of the questions file)")
    parser.add_argument("--writers", nargs="+", help="only these writers (default: every writer whose file is present)")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models", help="folder with the .gguf files")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (a quick trial)")
    parser.add_argument("--encode", action="store_true", help="also turn the rationales into MedCPT query vectors")
    parser.add_argument(
        "--allow-template-mismatch",
        action="store_true",
        help="go on although the chat template in a model file differs from the Llama-3 format (it is recorded)",
    )
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--outputs-dir", type=Path, default=HERE / "outputs", help="the rationales (not committed)")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results", help="the small report files (committed)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
        run_name = args.run_name if args.run_name is not None else re.sub(r"[^A-Za-z0-9._-]+", "-", args.questions.stem).strip("-.")
        if not NAME.fullmatch(run_name):
            raise StudyError("give a --run-name of letters, digits, dots, dashes and underscores")
        writers, skipped = choose_writers(settings, args.writers, args.models_dir)
    except StudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for writer in skipped:
        print(f"passed over {writer['name']}: {writer['model_file']} is not in {args.models_dir}")
    folders = [args.outputs_dir / run_name / writer["name"] for writer in writers]
    steps: list[tuple[str, list[str]]] = []
    for writer, folder in zip(writers, folders, strict=True):
        steps.append(
            (
                f"{writer['name']}: write the rationales",
                generate_command(
                    writer,
                    questions=args.questions,
                    out=folder,
                    models_dir=args.models_dir,
                    limit=args.limit,
                    allow_template_mismatch=args.allow_template_mismatch,
                ),
            )
        )
        if args.encode:
            encode = ["encode", "--rationales", str(folder), "--questions", str(args.questions), "--overwrite"]
            if args.limit is not None:
                encode += ["--limit", str(args.limit)]
            steps.append((f"{writer['name']}: encode the rationales", encode))
    for title, command in steps:
        print(f"=== {title}")
        status = cli_main(command)
        if status != 0:
            hint = "run the same command again to continue" if status in (2, 130) else "the message above says what to change"
            print(f"stopped at '{title}' (exit status {status}); {hint}", file=sys.stderr)
            return status

    report = io.StringIO()
    json_path = args.results_dir / run_name / "report.json"
    status = cli_main(
        ["report", *map(str, folders), "--json", str(json_path)], streams=cli.Streams(report, sys.stderr)
    )
    if status != 0:
        return status
    store.write_atomic(args.results_dir / run_name / "report.txt", report.getvalue())
    used = store.read_questions(args.questions)[: args.limit]  # a limit of None keeps every question
    run = {
        "questions_file": args.questions.name,
        "questions_sha256": generator.file_sha256(args.questions),
        "questions_used": len(used),
        "writers": [{key: writer.get(key) for key in ("name", "model_file", "sha256", "chat_format")} for writer in writers],
        "encoded": args.encode,
    }
    store.write_atomic(args.results_dir / run_name / "run.json", json.dumps(run, indent=2) + "\n")
    print("=== comparison of the writers\n" + report.getvalue())
    print(f"rationales: {args.outputs_dir / run_name}\nreport: {json_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
