"""Command line for phase 1: profile the dataset, check the setup, run one model on one split, and write the report.

    python -m src.evaluate profile --root data --train <train-folder> --test <test-folder>
    python -m src.evaluate isic          (checks the dataset's labels against the ISIC Archive; needs internet)
    python -m src.evaluate check --config configs/<model>.toml [--model <id>]
    python -m src.evaluate run --config configs/<model>.toml --split dev --limit 5
    python -m src.evaluate run --config configs/<model>.toml --split test
    python -m src.evaluate report results/<model>/<run-folder>

``check`` tests everything a run needs, without sending any dataset image. Run it before every run.
``dev`` is a small, seeded sample of training images for developing the prompt; it is never a reported result, stays on this
computer (Git ignores its folder) and is not documented.
``test`` is the dataset's own test split, run once per config (the base paper's practice: one run at temperature 0); its
report is the result. A test run needs a clean Git state (everything committed), so that its run card names the exact code it
came from, and a downloaded model must be pinned to one commit. A finished test run is never overwritten.
A run saves every answer as it arrives, and a log of its progress (run.log), in a folder ending in ``.partial`` (ignored by
Git). An interrupted run continues where it stopped when the same command is run again; a finished run's folder loses the
``.partial`` ending.
The report scores the same answers twice: against the dataset's folder labels and against the ISIC-corrected labels
listed in the file named by ``[data] corrections`` (written by the isic command). The model never sees a label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from . import dataset, isic, metrics, models, preflight, prompt

RESULTS = Path("results")
MAX_ATTEMPTS = 8  # waits of 10, 20, 40, 80, 120, 120, 120 seconds between them: about 8.5 minutes in all


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
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


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


def _check_name(value, what: str) -> None:
    """A name that becomes a folder: letters, digits, dot, underscore and hyphen only, so it cannot point elsewhere."""
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"{what} must be 1 to 80 characters: letters, digits, '.', '_' or '-', starting with a letter or "
                         f"digit (got {value!r})")


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    for section, keys in {"data": ("root", "train", "test", "corrections", "manifest"), "model": ("provider", "id"), "prompt": ("file",),
                          "run": ("requests_per_minute", "dev_per_class", "dev_seed", "order_seed", "max_image_side")}.items():
        for key in keys:
            if key not in config.get(section, {}):
                raise ValueError(f"{path}: missing [{section}] {key}")
    _check_name(config.get("name"), f"{path}: name")  # it becomes the results folder
    run = config["run"]
    rate = run["requests_per_minute"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"{path}: [run] requests_per_minute must be a number above 0 (got {rate!r})")
    for key in ("dev_per_class", "max_image_side"):
        if isinstance(run[key], bool) or not isinstance(run[key], int) or run[key] < 1:
            raise ValueError(f"{path}: [run] {key} must be a whole number of at least 1 (got {run[key]!r})")
    for key in ("dev_seed", "order_seed"):
        if isinstance(run[key], bool) or not isinstance(run[key], int):
            raise ValueError(f"{path}: [run] {key} must be a whole number (got {run[key]!r})")
    if not config["data"]["train"] or not config["data"]["test"]:
        raise ValueError(f"{path}: set [data] train and test to the dataset's split folder names first")
    return config


def _split_last_line(data: bytes) -> tuple[str, bytes]:
    """The complete lines of a JSON-lines file (decoded strictly) and the bytes after its last end of line."""
    cut = data.rfind(b"\n") + 1
    return data[:cut].decode("utf-8"), data[cut:]


def _complete_row(tail: bytes) -> dict | None:
    """The row in bytes that lack their end of line, or None when they are a cut-off line (even one cut inside a character)."""
    try:
        row = json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


def _read_jsonl(path: Path) -> list[dict]:
    """The rows of a JSON-lines file. An incomplete last line (the program was stopped while writing it) is ignored; a damaged
    line anywhere else is an error, never silently skipped."""
    if not path.exists():
        return []
    head, tail = _split_last_line(path.read_bytes())
    rows = []
    for number, line in enumerate(head.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not valid JSON ({exc.msg}); the file is damaged") from exc
    if tail.strip() and (last := _complete_row(tail)) is not None:
        rows.append(last)
    return rows


def _repair_jsonl(path: Path) -> None:
    """Before answers are appended: remove an incomplete last line, or add the missing end of line after a complete one."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, tail = _split_last_line(data)
    if _complete_row(tail) is not None:
        fixed = data + b"\n"
    else:
        fixed = head.encode("utf-8")
        print(f"{path}: removed an incomplete last line (the program was stopped while writing it); that image is answered again")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(fixed)
    os.replace(tmp, path)


def _write_json(path: Path, data: dict) -> None:
    """Written to a temporary file first, so a stop in the middle never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _ask(model, image: bytes, mime: str, text: str, throttle, say=print) -> tuple[dict, int]:
    """One answer and the number of attempts it took. Waits longer after each transient failure; raises when all fail."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        throttle()
        try:
            return model.predict(image, mime, text), attempt
        except models.TransientError as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = min(120, 10 * 2 ** (attempt - 1))
            say(f"  attempt {attempt}/{MAX_ATTEMPTS} failed ({exc}); waiting {wait} s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def run(config_path: str, split: str, run_id: str | None = None, limit: int | None = None, results: Path = RESULTS) -> Path:
    """A development run or the test run (one folder each). Returns the finished run's folder, or the ``.partial`` folder of a
    stopped run (the same command continues it)."""
    config = load_config(config_path)
    if run_id is not None:
        _check_name(run_id, "--run-id")
    if split == "test" and limit:
        raise ValueError("--limit is for development runs only; a test run always uses the whole test split")
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be at least 1 (got {limit})")
    git = _git()
    if split == "test" and git["uncommitted_changes"]:
        raise ValueError("a test run needs a clean Git state, so that its run card names the exact code it came from: commit your "
                         "changes (git status shows them), then run the same command again")
    if split == "test" and git["uncommitted_changes"] is None:
        print("WARNING: Git is not available here, so the run card cannot name the code this run came from.")
    records, problems = dataset.scan(config["data"]["root"], {"train": config["data"]["train"], "test": config["data"]["test"]})
    broken = [path for path, why in problems if why.startswith("unreadable image")]
    if broken:
        raise ValueError(f"{len(broken)} image files cannot be read and would silently drop out of the evaluation: {broken[:5]}")
    labels = dataset.labels_of(records, "train")
    unknown = sorted({r.label for r in records if r.split == "test"} - set(labels))
    if unknown:
        raise ValueError(f"test classes missing from training: {unknown}")
    dataset.verify_against_manifest(records, config["data"]["manifest"])
    if split == "dev":
        items = dataset.dev_subset(records, config["run"]["dev_per_class"], config["run"]["dev_seed"])
    elif split == "test":
        items = [r for r in records if r.split == "test"]
    else:
        raise ValueError("split must be 'dev' or 'test'")
    # A seeded random order, so that any drift of a hosted model during a run is not lined up with the classes.
    random.Random(config["run"]["order_seed"]).shuffle(items)
    if split == "dev":
        items, copies = dataset.without_test_copies(items, records)
        if copies:
            print(f"{len(copies)} development images are byte-identical to test images and were left out: "
                  f"{[r.path for r in copies][:5]}")
    items = items[: limit or None]
    isic.read_corrections(config["data"]["corrections"], labels, {r.path: r.label for r in items})  # refuse a bad file now

    template = prompt.load_template(config["prompt"]["file"])
    prompt_text = prompt.build(template, labels)
    config_bytes = Path(config_path).read_bytes()
    fingerprint = hashlib.sha256(config_bytes + prompt_text.encode("utf-8") + dataset.items_hash(items).encode()).hexdigest()
    models.load_env()
    model_dir = results / config["name"]
    job = {"config": config, "config_path": config_path, "split": split, "items": items, "labels": labels,
           "prompt_text": prompt_text, "fingerprint": fingerprint, "problems": problems, "model_dir": model_dir, "git": git}
    if split == "dev":
        return _run_once(job, run_id or f"dev_{fingerprint[:8]}")  # the same settings give the same folder: a stop resumes

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
            raise SystemExit(f"a finished test run already exists for this config, made with other settings, prompt or images: "
                             f"{folder} (results are not overwritten). A changed setting needs a new config file")
        raise SystemExit(f"an unfinished test run made with other settings, prompt or images exists: {folder}. Delete that "
                         "folder to start over, or undo the change")


def _run_once(job: dict, run_id: str) -> Path:
    """One run over the job's images: answers are saved as they arrive in ``<run_id>.partial``, renamed when complete."""
    config, items, labels, prompt_text, fingerprint = (job["config"], job["items"], job["labels"], job["prompt_text"],
                                                       job["fingerprint"])
    settings = dict(config["model"])
    model = models.load(settings["provider"], settings)  # a missing key or model stops the run before any file is written
    if job["split"] == "test" and getattr(model, "pinned", True) is False:
        raise SystemExit("a test run needs the model pinned to one version: set revision in the config to the commit hash that "
                         "the check prints, and commit the config, so that every answer comes from the same model files")
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
                "prompt": prompt_text, "labels": labels, "n_images": len(items), "images_sha256": dataset.items_hash(items),
                "dataset_files_not_images": len(job["problems"]), "git": job["git"],
                "versions": _versions(getattr(model, "packages", ("pillow",))),
                "started_utc": _now(), "finished_utc": None, "model": None, "model_versions_returned": []}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["fingerprint"] != fingerprint:
            raise SystemExit(f"{out} was started with a different config, prompt or image set; delete that folder to start "
                             "over, or choose another --run-id")
        before = (card.get("model") or {}).get("revision_loaded")
        if before is not None and before != described.get("revision_loaded"):
            raise SystemExit(f"{out} was started with model files at commit {before}, but commit {described.get('revision_loaded')} "
                             "is loaded now; answers from two versions of a model must not be mixed. Delete that folder to start "
                             "over (a revision pinned to a commit hash in the config prevents this)")
    card.setdefault("sessions", []).append({"utc": _now(), "commit": job["git"]["commit"],
                                            "uncommitted_changes": job["git"]["uncommitted_changes"]})
    card["model"] = described
    _write_json(card_path, card)

    def say(message: str, show: bool = True) -> None:
        """Prints a progress message and appends it, with the time, to the run's log (when the run's folder exists)."""
        if show:
            print(message)
        if out.exists():
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(f"{_now()} {message.replace(chr(10), ' ')}\n")

    _repair_jsonl(pred_path)
    done = {p["path"] for p in _read_jsonl(pred_path)}
    todo = [r for r in items if r.path not in done]
    interval = 60.0 / config["run"]["requests_per_minute"]
    last = [0.0]

    def throttle() -> None:
        time.sleep(max(0.0, interval - (time.monotonic() - last[0])))
        last[0] = time.monotonic()

    root = Path(config["data"]["root"])
    answered = len(done)
    git = job["git"]
    say(f"session start: Git commit {str(git['commit'])[:10]}{', UNCOMMITTED CHANGES' if git['uncommitted_changes'] else ''}; "
        f"model {described.get('model_id')}, revision loaded {described.get('revision_loaded')}")
    say(f"{out}: {answered} of {len(items)} answered, {len(todo)} to go")
    try:
        with pred_path.open("a", encoding="utf-8", newline="\n") as f:
            for i, record in enumerate(todo, start=len(done) + 1):
                image, mime, sent = dataset.prepare_image(root / record.path, config["run"]["max_image_side"])
                answer, attempt = _ask(model, image, mime, prompt_text, throttle, say)
                parsed = prompt.parse(answer["text"], labels)
                line = {"path": record.path, "label": record.label, "prediction": parsed, "raw_text": answer["text"],
                        "finish_reason": answer["finish_reason"], "model_version": answer["model_version"],
                        "response_id": answer["response_id"], "usage": answer["usage"], "attempts": attempt, "image_sent": sent,
                        "answered_utc": _now()}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                answered += 1
                say(f"[{i}/{len(items)}] {record.label} -> {parsed if parsed else 'UNREADABLE: ' + repr(answer['text'])}")
    except (models.TransientError, models.QuotaExhausted, models.FatalError) as exc:
        nothing_to_continue = answered == 0 and created
        if nothing_to_continue:
            shutil.rmtree(out)  # an empty folder is only a trap: nothing in it can be continued or committed
        else:  # the run card keeps every stop, so a finished run shows whether and why it was interrupted
            card.setdefault("interruptions", []).append({"utc": _now(), "answered": answered, "kind": type(exc).__name__,
                                                         "message": str(exc)})
            _write_json(card_path, card)
        if isinstance(exc, models.FatalError):
            message = f"STOPPED, {answered} of {len(items)} answered. {exc}"
            say(message, show=False)
            raise SystemExit(message) from exc
        quota = isinstance(exc, models.QuotaExhausted)
        say(f"NOT FINISHED: {answered} of {len(items)} answered. {'' if quota else 'Last problem: '}{exc}\n"
            + ("Nothing was saved. " if nothing_to_continue else f"Your answers are saved in {out} (Git ignores it). ")
            + ("Run the same command again once the quota allows it" if quota else "Run the same command again to continue")
            + "; nothing needs to be committed until it finishes.")
        return out

    card["status"], card["finished_utc"] = "complete", _now()
    card["model_versions_returned"] = sorted({p["model_version"] for p in _read_jsonl(pred_path) if p["model_version"]})
    _write_json(card_path, card)
    report(out)
    say(f"FINISHED: {len(items)} of {len(items)} answered. Report: {final / 'report.md'}\n"
        + ("This folder is complete and can be committed." if job["split"] == "test"
           else "A development run stays on this computer: Git ignores it and it is not documented."))
    try:
        out.rename(final)
    except OSError as exc:  # Windows refuses while a program (Explorer, an editor) has the folder open
        raise SystemExit(f"The run is complete and saved in {out}, but the folder could not be renamed ({exc}). Close any "
                         "program showing it and run the same command again; no image is sent again.") from exc
    return final


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(pair) -> str:
    return f"{_pct(pair[0])} to {_pct(pair[1])}"


def _scoring(title: str, m: dict) -> list[str]:
    """The accuracy of one scoring, as report lines."""
    return [f"## {title}", "", "| Metric | Value | 95% interval |", "|---|---|---|",
            f"| Accuracy | {_pct(m['accuracy'])} ({m['correct']}/{m['n']}) | {_ci(m['accuracy_ci95'])} (Wilson) |",
            f"| Unreadable answers (counted as wrong) | {m['unreadable']} | |", ""]


def report(run_dir: str | Path) -> Path:
    """Writes metrics.json and report.md from a run's predictions, scored against the folder and the corrected labels."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    labels = card["labels"]
    corrections_file = Path(card["config"]["data"]["corrections"])
    corrections_sha256 = hashlib.sha256(corrections_file.read_bytes()).hexdigest()
    fixed = isic.read_corrections(corrections_file, labels, {p["path"]: p["label"] for p in preds})
    answers = [p["prediction"] for p in preds]
    folder = metrics.summarize([p["label"] for p in preds], answers)
    corrected = metrics.summarize([fixed.get(p["path"], p["label"]) for p in preds], answers) if fixed else None
    _write_json(run_dir / "metrics.json", {
        "folder_labels": folder,
        "corrected_labels": corrected,
        "label_corrections": {"file": corrections_file.as_posix(), "sha256": corrections_sha256,
                              "applied": [{"path": p["path"], "folder_label": p["label"], "corrected_label": fixed[p["path"]]}
                                          for p in preds if p["path"] in fixed]}})

    complete = card["status"] == "complete" and len(preds) == card["n_images"]
    lines = [f"# {card['name']}: {card['split']} split, run {card['run_id']}", ""]
    if card["split"] == "dev":
        lines += ["**Development run on training images: for checking the prompt and answer reader, not a reported result.**", ""]
    if not complete:
        lines += [f"**Incomplete: {len(preds)} of {card['n_images']} images answered.**", ""]
    model = card.get("model") or {}
    lines += [f"- Model requested: `{model.get('model_id')}`; versions returned by the API: "
              f"{', '.join(f'`{v}`' for v in card['model_versions_returned']) or 'not recorded yet'}",
              f"- Generation settings: `{json.dumps(model.get('generation_config'))}`",
              f"- Started {card['started_utc']}, finished {card['finished_utc']} (UTC)",
              f"- Prompt: `{card['prompt_file']}` (SHA-256 `{card['prompt_sha256'][:12]}`); images: {card['n_images']} "
              f"(hash `{card['images_sha256'][:12]}`); Git commit `{(card['git']['commit'] or 'unknown')[:10]}`",
              f"- Label corrections: `{corrections_file.as_posix()}` (SHA-256 `{corrections_sha256[:12]}`): "
              + (f"{len(fixed)} of these images have an ISIC-corrected label, so the same answers are scored twice."
                 if fixed else "none of these images has a corrected label, so the answers are scored once."), ""]
    if corrected:  # the headline figure (decision of 2026-11-04): ISIC histopathology outranks the folder name
        lines += _scoring("Main result: scored against the ISIC-corrected labels", corrected)
        lines += _scoring("For comparison: scored against the dataset's folder labels", folder)
    else:
        lines += _scoring("Scored against the dataset's folder labels (no label among these images is corrected)", folder)
    if corrected:
        lines += ["Images whose label is corrected (evidence: the ISIC record in the corrections file):", "",
                  "| Image | Folder label | Corrected label | Answer |", "|---|---|---|---|"]
        lines += [f"| {p['path']} | {p['label']} | {fixed[p['path']]} | {p['prediction'] or metrics.UNREADABLE} |"
                  for p in sorted(preds, key=lambda p: p["path"]) if p["path"] in fixed]
        lines.append("")
    sessions = card.get("sessions", [])
    commits = sorted({s["commit"] or "unknown" for s in sessions})
    finish = {}
    for p in preds:
        finish[p["finish_reason"]] = finish.get(p["finish_reason"], 0) + 1
    other = {k: v for k, v in finish.items() if k not in ("STOP", "MAX_TOKENS")}
    cut = [p for p in preds if p["finish_reason"] == "MAX_TOKENS"]
    harmful = sum(prompt.can_still_become_readable(p["raw_text"], labels) for p in cut)  # the limit may have decided these
    lines += ["## Notes", "",
              f"- How answers ended (finish reason): {', '.join(f'{k}: {v}' for k, v in sorted(finish.items(), key=str))}",
              *([f"- **{harmful} answers were cut off at the output limit (finish reason MAX_TOKENS) and count as wrong: "
                 "the output limit in the config (max_output_tokens or max_new_tokens) is too low for this model (hidden reasoning "
                 "can count toward it). Results are not valid until it is raised in a new config.**"] if harmful else []),
              *([f"- {len(cut) - harmful} answers reached the output limit but already differed from every class name before it "
                 "(an essay, or a second class name), so the limit did not change their score; they count as wrong."]
                if len(cut) > harmful else []),
              *([f"- **{sum(other.values())} answers did not end normally ("
                 + ", ".join(f"{k}: {v}" for k, v in sorted(other.items(), key=str))
                 + "); an answer without readable text counts as wrong. Check them in predictions.jsonl.**"] if other else []),
              f"- Sessions (starts and restarts of this run): {len(sessions)}; Git commits: "
              f"{', '.join(c[:10] for c in commits) or 'not recorded'}"
              + ("; **the run spanned more than one commit, so its answers do not all come from the same code**" if len(commits) > 1 else ""),
              f"- Interruptions before the run finished: {len(card.get('interruptions', []))}"
              + "".join(f"; {i['utc']} after {i['answered']} answers ({i['kind']})" for i in card.get("interruptions", [])),
              f"- Images scaled down before sending (longer side above {card['config']['run']['max_image_side']} px): "
              f"{sum(bool(p.get('image_sent', {}).get('resized')) for p in preds)}"]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="inventory the dataset (no model is run)")
    p.add_argument("--root", default="data")
    p.add_argument("--train", required=True, help="training-split folder inside --root")
    p.add_argument("--test", required=True, help="testing-split folder inside --root")
    p.add_argument("--out", default="results/dataset_profile", help="manifest.csv and profile.md (committed)")
    p.add_argument("--sheets", default="data/contact_sheets", help="thumbnail sheets (stay out of Git)")
    i = sub.add_parser("isic", help="check the dataset's labels against the diagnoses recorded in the ISIC Archive")
    i.add_argument("--manifest", default="results/dataset_profile/manifest.csv")
    i.add_argument("--out", default="results/dataset_profile")
    i.add_argument("--cache", default="data/isic_cache", help="raw ISIC records (stay out of Git)")
    i.add_argument("--all", action="store_true", help="check every image, not only conflicting and testing images")
    c = sub.add_parser("check", help="test everything a run needs, without sending any dataset image")
    c.add_argument("--config", required=True)
    c.add_argument("--model", help="test this model id instead of the config's (the config file is not changed)")
    r = sub.add_parser("run", help="run one model on the dev sample or the test split")
    r.add_argument("--config", required=True)
    r.add_argument("--split", choices=("dev", "test"), required=True)
    r.add_argument("--run-id", help="folder name (for test: the name the run folders start with); by default one that "
                   "follows from the settings, so a stopped run resumes")
    r.add_argument("--limit", type=int, help="first N images of the dev sample (dev only)")
    rep = sub.add_parser("report", help="rewrite metrics.json and report.md from saved predictions")
    rep.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "profile":
            print(dataset.profile(args.root, {"train": args.train, "test": args.test}, args.out, args.sheets))
        elif args.command == "isic":
            print(isic.check(args.manifest, args.out, args.cache, everything=args.all))
        elif args.command == "check":
            config = load_config(args.config)
            if args.model:
                config["model"] = {**config["model"], "id": args.model}
            sys.exit(0 if preflight.check(config, args.config) else 1)
        elif args.command == "run":
            run(args.config, args.split, args.run_id, args.limit)
        else:
            print(report(args.run_dir))
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
