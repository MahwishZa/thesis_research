"""The preflight check: everything a run needs, tested without sending a single dataset image.

    python -m src.evaluate check --config configs/<model>.toml [--thinking-levels] [--list-models] [--model <id>]

Run it before every development or test run. Each line says OK, WARN or FAIL; a FAIL says what to do. The model is called
with a blank synthetic image, so it finds an unavailable model, a refused setting, a wrong key or an overloaded service
before a run is started, and nothing from the dataset is sent.
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


def _probe(model, say) -> tuple[dict | None, str | None, str]:
    """(answer, problem, kind) for one blank-image request; kind is ok, temporary, quota or refused.
    Transient failures are retried a few times; a daily quota and a refusal are not."""
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return model.predict(_blank_image(), "image/png", PROBE_PROMPT), None, "ok"
        except models.TransientError as exc:
            if attempt == PROBE_ATTEMPTS:
                return None, f"temporary problem, not a setting: {exc}", "temporary"
            say(f"       attempt {attempt}/{PROBE_ATTEMPTS} failed ({exc}); waiting {5 * attempt} s")
            time.sleep(5 * attempt)
        except models.QuotaExhausted as exc:
            return None, str(exc), "quota"
        except models.FatalError as exc:
            return None, str(exc), "refused"
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


def check(config: dict, config_path: str, try_variants: bool = False, say=print, *, list_models: bool = False) -> bool:
    """Prints one line per check and returns True when none failed."""
    failed = warned = False

    def line(status: str, what: str, detail: str = "") -> None:
        nonlocal failed, warned
        failed, warned = failed or status == "FAIL", warned or status == "WARN"
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

    if list_models:
        _show_models(model, say)
    say(f"       asking {settings['id']} one question with a blank image (nothing from the dataset is sent; "
        "it counts as one request against the daily quota) ...")
    answer, problem, kind = _probe(model, say)
    if answer is None:
        line("FAIL", f"Model {settings['id']}", problem)
        if "HTTP 404" in problem and not list_models:
            _show_models(model, say)
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

    if try_variants and kind == "quota":
        say("       (the thinking settings were not tested: the daily quota is used up)")
    elif try_variants:
        variants = models.settings_variants(provider, settings)
        say("       thinking settings tried with the same blank image (one request each; the configured one was tested above):")
        for name, variant in variants or []:
            if variant == settings:
                continue
            answer, problem, kind = _probe(models.load(provider, variant), say)
            if answer is not None:
                thinking = (answer["usage"] or {}).get("thinking_tokens") or 0
                say(f"       - {name}: accepted, thinking tokens {thinking}, finish reason {answer['finish_reason']}")
            elif kind == "refused":
                say(f"       - {name}: REFUSED: {problem.splitlines()[0]}")
            else:
                say(f"       - {name}: not tested: {problem.splitlines()[0]}")
                if kind == "quota":
                    line("WARN", "Thinking settings", "the remaining ones were not tested (daily quota used up)")
                    break
        if not variants:
            say(f"       (no alternatives are known for provider {provider})")

    if failed:
        say("Fix the lines marked FAIL, then run the check again. A temporary problem needs a few minutes; a used-up daily "
            "quota needs the reset, a paid key or another model (check --model <id> tries one).")
    elif warned:
        say("Checks passed, but read the WARN lines before starting the run.")
    else:
        say("All checks passed: the run can start.")
    return not failed
