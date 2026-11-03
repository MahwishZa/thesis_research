"""The preflight check: everything a run needs, tested without sending a single dataset image.

    python -m src.evaluate check --config configs/<model>.toml [--thinking-levels]

Run it before every development or test run. Each line says OK, WARN or FAIL; a FAIL says what to do. The model is called
with a blank synthetic image, so it finds an unavailable model, a refused setting, a wrong key or an overloaded service
before a run is started, and nothing from the dataset is sent.
"""

from __future__ import annotations

import csv
import io
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
    """The answer to one blank-image request, or (None, reason) when it failed. Retries transient failures a few times."""
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return model.predict(_blank_image(), "image/png", PROBE_PROMPT), None
        except models.TransientError as exc:
            if attempt == PROBE_ATTEMPTS:
                return None, f"temporary problem, not a setting: {exc}"
            say(f"       attempt {attempt}/{PROBE_ATTEMPTS} failed ({exc}); waiting {5 * attempt} s")
            time.sleep(5 * attempt)
        except models.FatalError as exc:
            return None, str(exc)
    raise AssertionError("unreachable")


def check(config: dict, config_path: str, try_variants: bool = False, say=print) -> bool:
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
            fixes = isic.read_corrections(config["data"]["corrections"], classes["train"], {})
            line("OK", "Label corrections", f"{config['data']['corrections']} ({len(fixes)} testing images corrected)")
        except (OSError, ValueError) as exc:
            line("FAIL", "Label corrections", str(exc))

    models.load_env()
    provider, settings = config["model"]["provider"], config["model"]
    try:
        model = models.load(provider, settings)
        line("OK", "Model adapter and API key", f"{provider}, key found (not shown)")
    except ValueError as exc:
        line("FAIL", "Model adapter and API key", str(exc))
        return False

    say(f"       asking {settings['id']} one question with a blank image (nothing from the dataset is sent) ...")
    answer, problem = _probe(model, say)
    if answer is None:
        line("FAIL", f"Model {settings['id']}", problem)
        names = getattr(model, "list_models", None)
        if names and "HTTP 404" in problem:
            try:
                say("       models your key can use: " + ", ".join(names()))
            except Exception as exc:  # the listing is only a help; any failure of it is reported, not raised
                say(f"       (the model list could not be read: {exc})")
    else:
        thinking = (answer["usage"] or {}).get("thinking_tokens") or 0
        detail = (f"answered {answer['text']!r}, finish reason {answer['finish_reason']}, thinking tokens {thinking}, "
                  f"version {answer['model_version']}")
        if answer["finish_reason"] != "STOP" or not answer["text"]:
            line("FAIL", f"Model {settings['id']}", detail + " (the answer was cut off or empty: raise max_output_tokens)")
        elif thinking and settings.get("thinking_budget") == 0:
            line("WARN", f"Model {settings['id']}", detail + " (it thinks although thinking_budget = 0)")
        else:
            line("OK", f"Model {settings['id']}", detail)

    if try_variants:
        variants = models.settings_variants(provider, settings)
        say("       thinking settings tried with the same blank image:")
        for name, variant in variants or []:
            answer, problem = _probe(models.load(provider, variant), say)
            if answer is None:
                say(f"       - {name}: REFUSED or failed: {problem.splitlines()[0]}")
            else:
                thinking = (answer["usage"] or {}).get("thinking_tokens") or 0
                say(f"       - {name}: accepted, thinking tokens {thinking}, finish reason {answer['finish_reason']}")
        if not variants:
            say(f"       (no alternatives are known for provider {provider})")

    say("All checks passed: the run can start." if not failed else
        "Fix the lines marked FAIL (for a temporary problem: wait a few minutes), then run the check again.")
    return not failed
