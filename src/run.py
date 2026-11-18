"""Runs: one model answers one split of one dataset, every answer saved as it arrives.

    python -m src.run check --config configs/<dataset>/<model>.toml [--model <id>]
    python -m src.run run --config configs/<dataset>/<model>.toml --split val [--limit 5]
    python -m src.run run --config configs/<dataset>/<model>.toml --split test

``check`` tests everything a run needs without sending anything from the dataset; run it before every run. The model is asked
once, with a blank synthetic image (for a text dataset, with a short question and no image), so an unavailable model, a wrong
key or a used-up quota shows before a run starts.
``run`` sends every file of one split, with the prompt, to the model named in the config and saves each answer as it arrives: an
image goes next to the prompt, exactly as stored; a text case goes into the prompt word for word, where the prompt has
``{case}``; an image + metadata question is its image with its patient's facts in the prompt, where the prompt has ``{facts}``;
a case question sends its photos, with its case text where the prompt has ``{case}``. A validation run (``--split val``) is for
tuning the method; it stays on this computer (Git ignores it) and is not a reported result, and ``--limit N`` sends only its first
N files. A test run (``--split test``) is the result: it runs once per config (the base paper's practice), needs a clean Git state
so its run card names the exact code, and needs the model pinned to one version. Answers are kept in a folder ending in
``.partial`` until the run is complete; the same command continues a stopped run. A finished run gets its report
(src/evaluate.py). Results go to results/<dataset>/<model>/<run>/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .common import read_jsonl, repair_jsonl, write_json
from .datasets import (DATASETS, SPLITS, classes, dataset_paths, items_hash, load_dataset, pictures_root, read_case, read_image, 
                       read_picture)
from .evaluate import RESULTS, write_report
from .metrics import ABSTAIN
from .models import FatalError, QuotaExhausted, TransientError, load, load_env
from .prompts import CASE, FACTS, build_prompt, fill, load_template, parse_answer


RUN_SPLITS = ("val", "test")  # the splits a model answers; the training split is only ever looked up, never answered
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
MAX_ATTEMPTS = 8  # waits of 10, 20, 40, 80, 120, 120, 120 seconds between them: about 8.5 minutes in all
PROBE_ATTEMPTS = 3
PROBE_PROMPT = "Reply with the single word: ok."


# ---------------------------------------------------------------------------------------------------------------------------------
# Settings: one config file per dataset and model (configs/<dataset>/<model>.toml)


def _check_name(value, what: str) -> None:
    """A name that becomes a folder: letters, digits, dot, underscore and hyphen only, so it cannot point elsewhere."""
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"{what} must be 1 to 80 characters: letters, digits, '.', '_' or '-', starting with a letter or "
                         f"digit (got {value!r})")


def load_config(path: str | Path) -> dict:
    """A model's settings file, checked before anything is sent."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    for section, keys in {"model": ("provider", "id"), "prompt": ("file",), "run": ("requests_per_minute", "order_seed")}.items():
        for key in keys:
            if key not in config.get(section, {}):
                raise ValueError(f"{path}: missing [{section}] {key}")
    _check_name(config.get("name"), f"{path}: name")  # it becomes the results folder
    if config.get("dataset") not in DATASETS:
        raise ValueError(f"{path}: dataset must be one of {list(DATASETS)} (got {config.get('dataset')!r})")
    if not isinstance(config["prompt"].get("abstain", False), bool):
        raise ValueError(f"{path}: [prompt] abstain must be true or false (got {config['prompt']['abstain']!r})")
    rate, seed = config["run"]["requests_per_minute"], config["run"]["order_seed"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"{path}: [run] requests_per_minute must be a number above 0 (got {rate!r})")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path}: [run] order_seed must be a whole number (got {seed!r})")
    return config


# ---------------------------------------------------------------------------------------------------------------------------------
# The setup check (python -m src.run check): each line says OK or FAIL, and a FAIL says what to do


def _blank_image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "PNG")
    return buffer.getvalue()


def _probe(model, say, images: int = 1) -> tuple[dict | None, str | None]:
    """The answer to one request with that many blank images (0: none), or (None, reason) when it failed. Retries transient
    failures a few times; a used-up quota and a refusal are not retried."""
    pictures = [(_blank_image(), "image/png")] * images
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return model.predict(pictures, PROBE_PROMPT), None
        except TransientError as exc:
            if attempt == PROBE_ATTEMPTS:
                return None, f"temporary problem, not a setting: {exc}"
            say(f"       attempt {attempt}/{PROBE_ATTEMPTS} failed ({exc}); waiting {5 * attempt} s")
            time.sleep(5 * attempt)
        except (QuotaExhausted, FatalError) as exc:
            return None, str(exc)
    raise AssertionError("unreachable")


def check_setup(config: dict, config_path: str, say=print) -> bool:
    """Prints one line per check and returns True when none failed. Nothing from the dataset is sent."""
    failed = False
    dataset = config["dataset"]
    kind = DATASETS[dataset]

    def line(status: str, what: str, detail: str = "") -> None:
        nonlocal failed
        failed = failed or status == "FAIL"
        say(f"[{status:>4}] {what}" + (f": {detail}" if detail else ""))

    line("OK", "Config", config_path)
    try:
        load_template(config["prompt"]["file"], dataset, config["prompt"].get("abstain", False))
        line("OK", "Prompt file", config["prompt"]["file"])
    except (OSError, ValueError) as exc:
        line("FAIL", "Prompt file", str(exc))
    try:  # every file is hashed and compared with the committed table, as a run does
        items = load_dataset(*dataset_paths(dataset), dataset)
        labels = classes(items)
        counts = Counter(item.split for item in items)
        line("OK", "Prepared dataset", f"{len(items)} {kind.item}s ({', '.join(f'{s} {counts[s]}' for s in SPLITS)}), "
                                       f"{len(labels)} classes")
    except (OSError, ValueError) as exc:
        line("FAIL", "Prepared dataset", str(exc))

    load_env()
    settings = config["model"]
    try:
        model = load(settings["provider"], settings)
        line("OK", "Model adapter and access", f"{settings['provider']}, {getattr(model, 'loaded_note', 'key found (not shown)')}")
    except ValueError as exc:
        line("FAIL", "Model adapter and access", str(exc))
        model = None
    if model is not None:
        blank = 2 if kind.pictures_column else 0 if kind.in_prompt else 1  # a case dataset sends several pictures: two show it works
        say(f"       asking {settings['id']} one question with {['no image', 'a blank image', 'two blank images'][blank]} (nothing from "
            f"the dataset is sent; {getattr(model, 'probe_note', 'it counts as one request against the daily quota')}) ...")
        answer, problem = _probe(model, say, images=blank)
        if answer is None:
            line("FAIL", f"Model {settings['id']}", problem)
        else:
            detail = (f"answered {answer['text']!r}, finish reason {answer['finish_reason']}, version {answer['model_version']}"
                      + (f", usage {answer['usage']}" if answer["usage"] else ""))
            if answer["finish_reason"] != "STOP" or not answer["text"]:
                line("FAIL", f"Model {settings['id']}", detail + " (the answer was cut off or empty: raise the output limit, "
                                                                "max_tokens or max_output_tokens)")
            else:
                line("OK", f"Model {settings['id']}", detail)
    say("All checks passed: the run can start." if not failed else
        "Fix the lines marked FAIL, then run the check again. A temporary problem needs a few minutes; a used-up daily quota "
        "needs the reset, a paid key or another model (check --model <id> tries one).")
    return not failed


# ---------------------------------------------------------------------------------------------------------------------------------
# Runs (python -m src.run run): one folder per run, every answer saved as it arrives


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git() -> dict:
    """The commit the code is at, and whether anything but the results folder differs from it (a finished run's own folder, not yet
    committed, must not stop the next test run)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--", ".", f":(exclude){RESULTS.as_posix()}"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "uncommitted_changes": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def _versions(packages: tuple[str, ...] = ("pillow",)) -> dict:
    """The Python version and the versions of the packages the model adapter says it uses (``packages``)."""
    out = {"python": platform.python_version()}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def _ask(model, images: list[tuple[bytes, str]], text: str, throttle, say=print) -> tuple[dict, int]:
    """One answer and the number of attempts it took. Waits longer after each transient failure; raises when all fail."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        throttle()
        try:
            return model.predict(images, text), attempt
        except TransientError as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = min(120, 10 * 2 ** (attempt - 1))
            say(f"  attempt {attempt}/{MAX_ATTEMPTS} failed ({exc}); waiting {wait} s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def run(config_path: str, split: str, run_id: str | None = None, limit: int | None = None, results: Path = RESULTS) -> Path:
    """A validation or test run (one folder each). Returns the finished run's folder, or the ``.partial`` folder of a stopped
    run (the same command continues it)."""
    config = load_config(config_path)
    if split not in RUN_SPLITS:
        raise ValueError(f"split must be one of {list(RUN_SPLITS)} (got {split!r})")
    if run_id is not None:
        _check_name(run_id, "--run-id")
    if split == "test" and limit:
        raise ValueError("--limit is for validation runs only; a test run always uses the whole test split")
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be at least 1 (got {limit})")
    git = _git()
    if split == "test" and git["uncommitted_changes"]:
        raise ValueError("a test run needs a clean Git state, so that its run card names the exact code it came from: commit your "
                         "changes (git status shows them), then run the same command again")
    if split == "test" and git["uncommitted_changes"] is None:
        print("WARNING: Git is not available here, so the run card cannot name the code this run came from.")
    dataset = config["dataset"]
    table, root = dataset_paths(dataset)
    everything = load_dataset(table, root, dataset)  # stops if any file differs from the committed table
    labels = classes(everything)
    items = sorted((item for item in everything if item.split == split), key=lambda item: item.path)
    if not items:
        raise ValueError(f"the {split} split of the prepared dataset is empty")
    # A seeded random order, so that any drift of a hosted model during a run is not lined up with the classes.
    random.Random(config["run"]["order_seed"]).shuffle(items)
    items = items[: limit or None]

    abstain = config["prompt"].get("abstain", False)
    prompt_text = build_prompt(load_template(config["prompt"]["file"], dataset, abstain), labels, abstain)
    config_bytes = Path(config_path).read_bytes()
    fingerprint = hashlib.sha256(config_bytes + prompt_text.encode("utf-8") + items_hash(items).encode()).hexdigest()
    load_env()
    model_dir = results / config["dataset"] / config["name"]
    job = {"config": config, "config_path": config_path, "split": split, "items": items, "labels": labels,
           "prompt_text": prompt_text, "fingerprint": fingerprint, "model_dir": model_dir, "git": git,
           "table": table, "root": root, "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(), "kind": DATASETS[dataset]}
    if split == "val":
        return _run_once(job, run_id or f"val_{fingerprint[:8]}")  # the same settings give the same folder: a stop resumes

    name = run_id or f"test_{fingerprint[:8]}"
    _refuse_other_test_runs(model_dir, name, fingerprint)
    if (model_dir / name).exists():
        print(f"{model_dir / name} is finished (results are not overwritten)")
        return model_dir / name
    return _run_once(job, name)


def _refuse_other_test_runs(model_dir: Path, name: str, fingerprint: str) -> None:
    """The test split is run once per config: any other test run of this config stops the command."""
    for card_path in sorted(model_dir.glob("*/run_card.json")):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["split"] != "test":
            continue
        folder, finished = card_path.parent, not card_path.parent.name.endswith(".partial")
        other = folder.name.removesuffix(".partial")
        if other == name and card["fingerprint"] == fingerprint:
            continue  # this command's own run
        if card["fingerprint"] == fingerprint:
            raise SystemExit(f"the test run of this config already exists under the name {other} ({folder}). Continue it "
                             f"with --run-id {other}, or leave out --run-id if that is the default name")
        if finished:
            raise SystemExit(f"a finished test run already exists for this config, made with other settings, prompt or files: "
                             f"{folder} (results are not overwritten). A changed setting needs a new config file")
        raise SystemExit(f"an unfinished test run made with other settings, prompt or files exists: {folder}. Delete that "
                         "folder to start over, or undo the change")


def _run_once(job: dict, run_id: str) -> Path:
    """One run over the job's files: answers are saved as they arrive in ``<run_id>.partial``, renamed when complete."""
    config, items, labels, prompt_text, fingerprint, kind = (job["config"], job["items"], job["labels"], job["prompt_text"],
                                                             job["fingerprint"], job["kind"])
    settings = dict(config["model"])
    model = load(settings["provider"], settings)  # a missing key or model stops the run before any file is written
    if job["split"] == "test" and getattr(model, "pinned", True) is False:
        raise SystemExit("a test run needs the model pinned to one version: use a fixed model id (a dated snapshot, not a name "
                         "that moves such as one ending in -latest) in the config, and commit the config, so that every answer "
                         "comes from the same model")
    described = model.describe()

    model_dir = job["model_dir"]
    final, out = model_dir / run_id, model_dir / f"{run_id}.partial"  # unfinished work lives in the .partial folder (git-ignored)
    card_path, pred_path, log_path = out / "run_card.json", out / "predictions.jsonl", out / "run.log"
    if final.exists():
        raise SystemExit(f"{final} is finished; choose another --run-id to run it again")
    created = not card_path.exists()
    if created:
        out.mkdir(parents=True, exist_ok=True)
        card = {"name": config["name"], "split": job["split"], "run_id": run_id, "status": "running", "fingerprint": fingerprint,
                "seed": settings.get("seed"), "config_file": Path(job["config_path"]).as_posix(), "config": config,
                "prompt_file": config["prompt"]["file"], "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "prompt": prompt_text, "labels": labels, "abstain": config["prompt"].get("abstain", False), f"n_{kind.item}s": len(items), f"{kind.item}s_sha256": items_hash(items),
                "group_column": kind.group_column,
                "dataset_table": {"file": job["table"].as_posix(), "sha256": job["table_sha256"]}, "git": job["git"],
                "versions": _versions(getattr(model, "packages", ("pillow",))),
                "started_utc": _now(), "finished_utc": None, "model": None, "model_versions_returned": []}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["fingerprint"] != fingerprint:
            raise SystemExit(f"{out} was started with a different config, prompt or set of files; delete that folder to start "
                             "over, or choose another --run-id")
    card.setdefault("sessions", []).append({"utc": _now(), "commit": job["git"]["commit"],
                                            "uncommitted_changes": job["git"]["uncommitted_changes"],
                                            "model_revision": described.get("model_revision")})  # each session's, as a run may last days
    card["model"] = described
    write_json(card_path, card)

    def say(message: str, show: bool = True) -> None:
        """Prints a progress message and appends it, with the time, to the run's log (when the run's folder exists)."""
        if show:
            print(message)
        if out.exists():
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(f"{_now()} {message.replace(chr(10), ' ')}\n")

    repair_jsonl(pred_path)
    done = {p["path"] for p in read_jsonl(pred_path)}
    todo = [item for item in items if item.path not in done]
    interval = 60.0 / config["run"]["requests_per_minute"]
    last = [0.0]

    def throttle() -> None:
        time.sleep(max(0.0, interval - (time.monotonic() - last[0])))
        last[0] = time.monotonic()

    answered = len(done)
    git = job["git"]
    say(f"session start: Git commit {str(git['commit'])[:10]}{', UNCOMMITTED CHANGES' if git['uncommitted_changes'] else ''}; "
        f"model {described.get('model_id')}")
    say(f"{out}: {answered} of {len(items)} answered, {len(todo)} to go")
    try:
        with pred_path.open("a", encoding="utf-8", newline="\n") as f:
            for i, item in enumerate(todo, start=len(done) + 1):
                images = []
                if kind.pictures_column:  # a case question: its pictures, and with the case text only when the prompt holds it
                    shown = [read_picture(path, pictures_root(config["dataset"])) for path, _ in item.pictures]
                    images = [(content, mime) for content, mime, _ in shown]
                    sent = {"pictures": [record for _, _, record in shown]}
                    if kind.in_prompt:
                        case, sent["case"] = read_case(item, job["root"])
                        text = fill(prompt_text, CASE, case)
                    else:
                        text = prompt_text
                elif kind.in_prompt:  # a text case: inside the prompt, with no image
                    case, sent = read_case(item, job["root"])
                    text = fill(prompt_text, CASE, case)
                else:
                    image, mime, sent = read_image(item, job["root"])
                    images = [(image, mime)]
                    text = fill(prompt_text, FACTS, item.facts) if kind.facts_column else prompt_text
                answer, attempt = _ask(model, images, text, throttle, say)
                parsed = parse_answer(answer["text"], [*labels, ABSTAIN] if card.get("abstain") else labels)
                line = {"path": item.path, "label": item.label, kind.group_column: item.group, "prediction": parsed,
                        "raw_text": answer["text"], "finish_reason": answer["finish_reason"], "model_version": answer["model_version"],
                        "response_id": answer["response_id"], "usage": answer["usage"], "attempts": attempt,
                        f"{kind.item}_sent": sent, **({"facts": item.facts} if kind.facts_column else {}),
                        "answered_utc": _now()}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                answered += 1
                say(f"[{i}/{len(items)}] {item.label} -> {parsed if parsed else 'UNREADABLE: ' + repr(answer['text'])}")
    except (TransientError, QuotaExhausted, FatalError) as exc:
        nothing_to_continue = answered == 0 and created
        if nothing_to_continue:
            shutil.rmtree(out)  # an empty folder is only a trap: nothing in it can be continued or committed
        else:  # the run card keeps every stop, so a finished run shows whether and why it was interrupted
            card.setdefault("interruptions", []).append({"utc": _now(), "answered": answered, "kind": type(exc).__name__,
                                                         "message": str(exc)})
            write_json(card_path, card)
        if isinstance(exc, FatalError):
            message = f"STOPPED, {answered} of {len(items)} answered. {exc}"
            say(message, show=False)
            raise SystemExit(message) from exc
        quota = isinstance(exc, QuotaExhausted)
        say(f"NOT FINISHED: {answered} of {len(items)} answered. {'' if quota else 'Last problem: '}{exc}\n"
            + ("Nothing was saved. " if nothing_to_continue else f"Your answers are saved in {out} (Git ignores it). ")
            + ("Run the same command again once the quota allows it" if quota else "Run the same command again to continue")
            + "; nothing needs to be committed until it finishes.")
        return out

    card["status"], card["finished_utc"] = "complete", _now()
    card["model_versions_returned"] = sorted({p["model_version"] for p in read_jsonl(pred_path) if p["model_version"]})
    write_json(card_path, card)
    write_report(out)
    say(f"FINISHED: {len(items)} of {len(items)} answered. Report: {final / 'report.md'}\n"
        + ("This folder is complete and can be committed." if job["split"] == "test"
           else "A validation run stays on this computer: Git ignores it and it is not a reported result."))
    try:
        out.rename(final)
    except OSError as exc:  # Windows refuses while a program (Explorer, an editor) has the folder open
        raise SystemExit(f"The run is complete and saved in {out}, but the folder could not be renamed ({exc}). Close any "
                         "program showing it and run the same command again; nothing is sent again.") from exc
    return final


# ---------------------------------------------------------------------------------------------------------------------------------
# The command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.run", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="test everything a run needs, without sending anything from the dataset")
    c.add_argument("--config", required=True)
    c.add_argument("--model", help="test this model id instead of the config's (the config file is not changed)")
    r = sub.add_parser("run", help="run one model on the validation or the test split")
    r.add_argument("--config", required=True)
    r.add_argument("--split", choices=RUN_SPLITS, required=True)
    r.add_argument("--run-id", help="folder name; by default one that follows from the settings, so a stopped run resumes")
    r.add_argument("--limit", type=int, help="only the first N files of the validation split, in the seeded order")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            config = load_config(args.config)
            if args.model:
                config["model"] = {**config["model"], "id": args.model}
            sys.exit(0 if check_setup(config, args.config) else 1)
        else:
            run(args.config, args.split, args.run_id, args.limit)
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
