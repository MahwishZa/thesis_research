"""The preflight check: everything a run needs, tested without sending a single dataset image.

    python -m src.evaluate check --config configs/<model>.toml [--list-models] [--model <id>]

Run it before every development or test run. Each line says OK or FAIL; a FAIL says what to do. The model is called once with
a blank synthetic image, so it finds an unavailable model, a wrong key, a used-up daily quota or an overloaded service before
a run is started, and nothing from the dataset is sent. That one call counts as one request against the provider's quota.
"""

from __future__ import annotations

import csv
import io
import textwrap
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from . import dataset, isic, models, prompt

PROBE_PROMPT = "Reply with the single word: ok."
PROBE_ATTEMPTS = 3


def _blank_image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "PNG")
    return buffer.getvalue()


def _probe(model, say) -> tuple[dict | None, str | None]:
    """The answer to one blank-image request, or (None, reason) when it failed. Retries transient failures a few times;
    a used-up daily quota and a refusal are not retried."""
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return model.predict(_blank_image(), "image/png", PROBE_PROMPT), None
        except models.TransientError as exc:
            if attempt == PROBE_ATTEMPTS:
                return None, f"temporary problem, not a setting: {exc}"
            say(f"       attempt {attempt}/{PROBE_ATTEMPTS} failed ({exc}); waiting {5 * attempt} s")
            time.sleep(5 * attempt)
        except (models.QuotaExhausted, models.FatalError) as exc:
            return None, str(exc)
    raise AssertionError("unreachable")


def _show_models(model, say) -> None:
    """The models the key can use, when the provider can list them (a help only: a failure of it is reported, not raised)."""
    names = getattr(model, "list_models", None)
    if names is None:
        say("       (this provider cannot list its models)")
        return
    try:
        say("       models your key can use (a listed model can still be refused; the check tells):")
        say(textwrap.fill(", ".join(names()), width=110, initial_indent="         ", subsequent_indent="         "))
    except Exception as exc:
        say(f"       (the model list could not be read: {exc})")


def check(config: dict, config_path: str, say=print, *, list_models: bool = False) -> bool:
    """Prints one line per check and returns True when none failed."""
    failed = False

    def line(status: str, what: str, detail: str = "") -> None:
        nonlocal failed
        failed = failed or status == "FAIL"
        say(f"[{status:>4}] {what}" + (f": {detail}" if detail else ""))

    line("OK", "Config", config_path)
    try:
        prompt.load_template(config["prompt"]["file"])
        line("OK", "Prompt file", config["prompt"]["file"])
    except (OSError, ValueError) as exc:
        line("FAIL", "Prompt file", str(exc))

    # Dataset folders, compared with the committed profile by file count (a run compares every file's SHA-256).
    root, classes, counts = Path(config["data"]["root"]), {}, {}
    for split in ("train", "test"):
        folder = root / config["data"][split]
        if not folder.is_dir():
            line("FAIL", f"Dataset {split} folder", f"{folder} not found; extract the dataset ZIP so that this folder exists")
            continue
        classes[split] = sorted(p.name for p in folder.iterdir() if p.is_dir())
        counts[split] = sum(1 for p in folder.rglob("*") if p.suffix.lower() in dataset.IMAGE_EXTENSIONS)
    if len(counts) == 2:
        extra = sorted(set(classes["test"]) - set(classes["train"]))
        line("FAIL" if extra else "OK", "Classes", f"test classes missing from training: {extra}" if extra
             else f"{len(classes['train'])} classes in both splits")
        try:
            with open(config["data"]["manifest"], encoding="utf-8") as f:
                profiled = Counter(r["split"] for r in csv.DictReader(f))
            same = profiled == Counter(counts)
            line("OK" if same else "FAIL", "Images vs committed profile",
                 f"{counts['train']} training and {counts['test']} testing images" if same else
                 f"folders hold {dict(counts)} but the profile lists {dict(profiled)}; extract the ZIP again or profile again")
        except OSError as exc:
            line("FAIL", "Committed profile", str(exc))
    if "train" in classes:
        try:
            isic.read_corrections(config["data"]["corrections"], classes["train"], {})  # refuses labels that are not classes
            with open(config["data"]["corrections"], encoding="utf-8") as f:
                listed = sum(1 for _ in csv.DictReader(f))
            line("OK", "Label corrections", f"{config['data']['corrections']} ({listed} testing images listed with a corrected label)")
        except (OSError, ValueError) as exc:
            line("FAIL", "Label corrections", str(exc))

    models.load_env()
    provider, settings = config["model"]["provider"], config["model"]
    try:
        model = models.load(provider, settings)
        line("OK", "Model adapter and API key", f"{provider}, {getattr(model, 'loaded_note', 'key found (not shown)')}")
    except ValueError as exc:
        line("FAIL", "Model adapter and API key", str(exc))
        return False

    if list_models:
        _show_models(model, say)
    say(f"       asking {settings['id']} one question with a blank image (nothing from the dataset is sent; "
        f"{getattr(model, 'probe_note', 'it counts as one request against the daily quota')}) ...")
    answer, problem = _probe(model, say)
    if answer is None:
        line("FAIL", f"Model {settings['id']}", problem)
        if "HTTP 404" in problem and not list_models:
            _show_models(model, say)
    else:
        detail = (f"answered {answer['text']!r}, finish reason {answer['finish_reason']}, version {answer['model_version']}"
                  + (f", usage {answer['usage']}" if answer["usage"] else ""))
        if answer["finish_reason"] != "STOP" or not answer["text"]:
            line("FAIL", f"Model {settings['id']}", detail + " (the answer was cut off or empty: raise max_output_tokens)")
        else:
            line("OK", f"Model {settings['id']}", detail)

    say("All checks passed: the run can start." if not failed else
        "Fix the lines marked FAIL, then run the check again. A temporary problem needs a few minutes; a used-up daily quota "
        "needs the reset, a paid key or another model (check --model <id> tries one).")
    return not failed
