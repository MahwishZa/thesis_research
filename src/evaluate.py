"""Command line for phase 1: profile the dataset, run one model on one split, and write the report.

    python -m src.evaluate profile --root data --train <train folder> --test <test folder>
    python -m src.evaluate run --config configs/gemini-2.5-flash.toml --split dev --limit 5
    python -m src.evaluate run --config configs/gemini-2.5-flash.toml --split test
    python -m src.evaluate report results/gemini-2.5-flash/<run folder>
    python -m src.evaluate isic          (checks the dataset's labels against the ISIC Archive; needs internet)

``dev`` is a small, seeded sample of training images for developing the prompt; it is never a reported result.
``test`` is the dataset's own test split. It is run once per config, and a finished test run is never overwritten.
A run saves every answer as it arrives, so an interrupted run continues where it stopped when started again.
The report scores the same answers twice: against the dataset's folder labels and against the ISIC-corrected labels
listed in the file named by ``[data] corrections`` (written by the isic command). The model never sees a label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from . import dataset, isic, metrics, models, prompt

RESULTS = Path("results")
MAX_ATTEMPTS = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for package in ("google-genai", "pillow"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    for section, keys in {"data": ("root", "train", "test", "corrections"), "model": ("provider", "id"), "prompt": ("file",),
                          "run": ("requests_per_minute", "dev_per_class", "dev_seed", "order_seed", "max_image_side")}.items():
        for key in keys:
            if key not in config.get(section, {}):
                raise ValueError(f"{path}: missing [{section}] {key}")
    if not config["data"]["train"] or not config["data"]["test"]:
        raise ValueError(f"{path}: set [data] train and test to the dataset's split folder names first")
    return config


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def run(config_path: str, split: str, run_id: str | None = None, limit: int | None = None, results: Path = RESULTS) -> Path:
    config = load_config(config_path)
    if split == "test" and limit:
        raise ValueError("--limit is for development runs only; a test run always uses the whole test split")
    records, problems = dataset.scan(config["data"]["root"], {"train": config["data"]["train"], "test": config["data"]["test"]})
    broken = [path for path, why in problems if why.startswith("unreadable image")]
    if broken:
        raise ValueError(f"{len(broken)} image files cannot be read and would silently drop out of the evaluation: {broken[:5]}")
    labels = dataset.labels_of(records, "train")
    unknown = sorted({r.label for r in records if r.split == "test"} - set(labels))
    if unknown:
        raise ValueError(f"test classes missing from training: {unknown}")
    if split == "dev":
        items = dataset.dev_subset(records, config["run"]["dev_per_class"], config["run"]["dev_seed"])
    elif split == "test":
        items = [r for r in records if r.split == "test"]
    else:
        raise ValueError("split must be 'dev' or 'test'")
    # A seeded random order, so that any drift of a hosted model during a run is not lined up with the classes.
    random.Random(config["run"]["order_seed"]).shuffle(items)
    items = items[: limit or None]
    isic.read_corrections(config["data"]["corrections"], labels, {r.path: r.label for r in items})  # refuse a bad file now

    template = prompt.load_template(config["prompt"]["file"])
    prompt_text = prompt.build(template, labels)
    config_bytes = Path(config_path).read_bytes()
    fingerprint = hashlib.sha256(config_bytes + prompt_text.encode("utf-8") + dataset.items_hash(items).encode()).hexdigest()

    model_dir = results / config["name"]
    out = model_dir / (run_id or f"{datetime.now(timezone.utc):%Y%m%d}_{split}")
    card_path, pred_path = out / "run_card.json", out / "predictions.jsonl"
    if split == "test":
        for other in model_dir.glob("*/run_card.json"):
            card = json.loads(other.read_text(encoding="utf-8"))
            if card["split"] != "test":
                continue
            if card["status"] == "complete":
                raise SystemExit(f"a finished test run already exists for this config: {other.parent} (results are not overwritten)")
            if other.parent != out:
                raise SystemExit(f"an unfinished test run exists: {other.parent}; continue it with --run-id {other.parent.name}")
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["status"] == "complete":
            raise SystemExit(f"{out} is finished; choose another --run-id")
        if card["fingerprint"] != fingerprint:
            raise SystemExit(f"{out} was started with a different config, prompt or image set; choose another --run-id")
    else:
        out.mkdir(parents=True, exist_ok=True)
        card = {"name": config["name"], "split": split, "run_id": out.name, "status": "running", "fingerprint": fingerprint,
                "config_file": Path(config_path).as_posix(), "config": config,
                "prompt_file": config["prompt"]["file"], "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "prompt": prompt_text, "labels": labels, "n_images": len(items), "images_sha256": dataset.items_hash(items),
                "dataset_files_not_images": len(problems), "git": _git(), "versions": _versions(),
                "started_utc": _now(), "finished_utc": None, "model": None, "model_versions_returned": []}
        _write_json(card_path, card)

    model = models.load(config["model"]["provider"], config["model"])
    card["model"] = model.describe()
    _write_json(card_path, card)
    done = {p["path"] for p in _read_jsonl(pred_path)}
    todo = [r for r in items if r.path not in done]
    interval = 60.0 / config["run"]["requests_per_minute"]
    last = 0.0
    root = Path(config["data"]["root"])
    print(f"{out}: {len(done)} of {len(items)} answered, {len(todo)} to go")
    with pred_path.open("a", encoding="utf-8", newline="\n") as f:
        for i, record in enumerate(todo, start=len(done) + 1):
            image, mime, sent = dataset.prepare_image(root / record.path, config["run"]["max_image_side"])
            for attempt in range(1, MAX_ATTEMPTS + 1):
                time.sleep(max(0.0, interval - (time.monotonic() - last)))
                last = time.monotonic()
                try:
                    answer = model.predict(image, mime, prompt_text)
                    break
                except models.TransientError as exc:
                    if attempt == MAX_ATTEMPTS:
                        print(f"stopped after {MAX_ATTEMPTS} failed attempts ({exc}); run the same command again later to continue")
                        return out
                    time.sleep(min(120, 10 * 2 ** (attempt - 1)))
            parsed = prompt.parse(answer["text"], labels)
            line = {"path": record.path, "label": record.label, "prediction": parsed, "raw_text": answer["text"],
                    "finish_reason": answer["finish_reason"], "model_version": answer["model_version"],
                    "response_id": answer["response_id"], "usage": answer["usage"], "attempts": attempt, "image_sent": sent,
                    "answered_utc": _now()}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(items)}] {record.label} -> {parsed if parsed else 'UNREADABLE: ' + repr(answer['text'])}")

    card["status"], card["finished_utc"] = "complete", _now()
    card["model_versions_returned"] = sorted({p["model_version"] for p in _read_jsonl(pred_path) if p["model_version"]})
    _write_json(card_path, card)
    report(out)
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(pair) -> str:
    return f"{_pct(pair[0])} to {_pct(pair[1])}"


def _scoring(title: str, m: dict, labels: list[str]) -> list[str]:
    """The results, per-class figures and confusion matrix of one scoring, as report lines."""
    largest = m["baselines"]["majority_classes"]
    lines = [f"## {title}", "", "| Metric | Value | 95% interval |", "|---|---|---|",
             f"| Accuracy | {_pct(m['accuracy'])} ({m['correct']}/{m['n']}) | {_ci(m['accuracy_ci95'])} (Wilson) |",
             f"| Balanced accuracy | {_pct(m['balanced_accuracy'])} | {_ci(m['balanced_accuracy_ci95'])} (bootstrap) |",
             f"| Macro-F1 | {_pct(m['macro_f1'])} | {_ci(m['macro_f1_ci95'])} (bootstrap) |",
             f"| Unreadable answers (counted as wrong) | {m['unreadable']} | |", "",
             f"Reference points: guessing at random {_pct(m['baselines']['chance_accuracy'])}; always answering "
             + (f"the largest class ({largest[0]})" if len(largest) == 1 else f"any one of the {len(largest)} largest classes "
                f"({', '.join(largest)})") + f" {_pct(m['baselines']['majority_class_accuracy'])}.", "",
             "| Class | Images | Recall | Precision | F1 | Times predicted |", "|---|---|---|---|---|---|"]
    for label in labels:
        c = m["per_class"][label]
        has_images, predicted = c["support"] > 0, c["predicted"] > 0
        lines.append(f"| {label} | {c['support']} | {_pct(c['recall']) if has_images else 'n/a'} | "
                     f"{_pct(c['precision']) if predicted else 'n/a'} | {_pct(c['f1']) if has_images else 'n/a'} | {c['predicted']} |")
    columns = labels + [metrics.UNREADABLE]
    lines += ["", "Confusion matrix (rows: true class; columns: answer):", "",
              "| True class | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(m["confusion"][label][c]) for c in columns) + " |")
    present = [x for x in labels if m["per_class"][x]["support"]]
    absent = [x for x in labels if not m["per_class"][x]["support"]]
    small = [f"{x} ({m['per_class'][x]['support']})" for x in present if m["per_class"][x]["support"] < 10]
    lines += ["", f"- Balanced accuracy and macro-F1 average over the {len(present)} classes that have images in this scoring"
              + (f"; answer options with no image: {', '.join(absent)} (n/a: not defined)" if absent else "") + ".",
              f"- Classes with fewer than 10 images (their per-class figures are unreliable): {', '.join(small) or 'none'}", ""]
    return lines


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
    folder = metrics.summarize([p["label"] for p in preds], answers, labels)
    corrected = metrics.summarize([fixed.get(p["path"], p["label"]) for p in preds], answers, labels) if fixed else None
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
    lines += _scoring("Scored against the dataset's folder labels", folder, labels)
    if corrected:
        lines += _scoring("Scored against the ISIC-corrected labels", corrected, labels)
        lines += ["Images whose label is corrected (evidence: the ISIC record in the corrections file):", "",
                  "| Image | Folder label | Corrected label | Answer |", "|---|---|---|---|"]
        lines += [f"| {p['path']} | {p['label']} | {fixed[p['path']]} | {p['prediction'] or metrics.UNREADABLE} |"
                  for p in sorted(preds, key=lambda p: p["path"]) if p["path"] in fixed]
        lines.append("")
    finish = {}
    for p in preds:
        finish[p["finish_reason"]] = finish.get(p["finish_reason"], 0) + 1
    lines += ["## Notes", "",
              f"- How answers ended (finish reason): {', '.join(f'{k}: {v}' for k, v in sorted(finish.items(), key=str))}",
              f"- Images scaled down before sending (longer side above {card['config']['run']['max_image_side']} px): "
              f"{sum(bool(p.get('image_sent', {}).get('resized')) for p in preds)}"]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="inventory the dataset (no model is run)")
    p.add_argument("--root", default="data")
    p.add_argument("--train", required=True, help="training-split folder inside --root")
    p.add_argument("--test", required=True, help="testing-split folder inside --root")
    p.add_argument("--out", default="results/dataset_profile", help="manifest.csv and profile.md (committed)")
    p.add_argument("--sheets", default="data/contact_sheets", help="thumbnail sheets (stay out of Git)")
    r = sub.add_parser("run", help="run one model on the dev sample or the test split")
    r.add_argument("--config", required=True)
    r.add_argument("--split", choices=("dev", "test"), required=True)
    r.add_argument("--run-id")
    r.add_argument("--limit", type=int, help="first N images of the dev sample (dev only)")
    rep = sub.add_parser("report", help="rewrite metrics.json and report.md from saved predictions")
    rep.add_argument("run_dir")
    i = sub.add_parser("isic", help="check the dataset's labels against the diagnoses recorded in the ISIC Archive")
    i.add_argument("--manifest", default="results/dataset_profile/manifest.csv")
    i.add_argument("--out", default="results/dataset_profile")
    i.add_argument("--cache", default="data/isic_cache", help="raw ISIC records (stay out of Git)")
    i.add_argument("--all", action="store_true", help="check every image, not only conflicting and testing images")
    args = parser.parse_args(argv)
    if args.command == "profile":
        print(dataset.profile(args.root, {"train": args.train, "test": args.test}, args.out, args.sheets))
    elif args.command == "run":
        print(run(args.config, args.split, args.run_id, args.limit))
    elif args.command == "report":
        print(report(args.run_dir))
    else:
        print(isic.check(args.manifest, args.out, args.cache, everything=args.all))


if __name__ == "__main__":
    main(sys.argv[1:])
