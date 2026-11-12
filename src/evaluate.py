"""Evaluation of LLMs on the prepared dataset: every model gets the same images, prompt, answer reader and metrics.

    python -m src.evaluate check --config configs/<dataset>/<model>.toml [--model <id>]
    python -m src.evaluate run --config configs/<dataset>/<model>.toml --split val [--limit 5]
    python -m src.evaluate run --config configs/<dataset>/<model>.toml --split test
    python -m src.evaluate report results/<dataset>/<model>/<run-folder>

``check`` tests everything a run needs without sending any dataset image; run it before every run. The model is asked once,
with a blank synthetic image, so an unavailable model, a wrong key or a used-up quota shows before a run starts.
``run`` sends every image of one split, with the prompt, to the model named in the config and saves each answer as it arrives.
A validation run (``--split val``) is for tuning the method; it stays on this computer (Git ignores it) and is not a reported
result, and ``--limit N`` sends only its first N images. A test run (``--split test``) is the result: it runs once per config
(the base paper's practice), needs a clean Git state so its run card names the exact code, and needs the model pinned to one
version. Answers are kept in a folder ending in ``.partial`` until the run is complete; the same command continues a stopped run.
``report`` rewrites metrics.json and report.md from a run's saved answers (no model is asked).

Each config names its dataset (``dataset = "image"``); the code here is shared by every dataset, so all are scored alike. A
prepared dataset is read from data/<dataset>/dataset/<split>/<class>/<file> (the files, exactly as every model receives them;
split: train, val or test) and data/<dataset>/dataset.csv (one row per file: split, label, path inside the dataset folder,
sha256, and the lesion group it belongs to, plus any further columns the preparation records). The table is committed to Git and
is the single source of truth; the files stay on this computer. Every file on disk is checked against the table before a run, so
a changed, missing or extra file stops the run and every model is evaluated on byte-identical files. Results go to
results/<dataset>/<model>/<run>/. Only the image dataset exists so far; the text and multimodal datasets are added in their
phases.

The file has three parts, in this order: the metrics, the model adapters (one class per API) and the pipeline with its
command line.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


# ==============================================================================================================================
# Metrics
#
# Metrics for a model that answers each image with one class name, or with nothing readable.
#
# Everything is computed from the saved answers (predictions.jsonl), so the metrics can be recomputed at any time without asking
# the model again. An unreadable answer (None) is a wrong answer. In the confusion matrix it gets its own column (UNREADABLE),
# so it lowers recall and every score derived from it, and it is never silently dropped.
#
# For class k (TP = true class k and answered k, FP = other class answered k, FN = true class k and answered otherwise):
#     precision = TP / (TP + FP)         undefined (None) when class k was never answered
#     recall    = TP / (TP + FN)         undefined (None) when no image has true class k
#     F1        = 2 TP / (2 TP + FP + FN)
# Averages over classes: macro (every class counts the same) and weighted (by the number of true images). Only classes that have at
# least one true image enter an average; a class that was never answered gets precision 0 in the macro precision. Balanced accuracy
# is the macro recall, and the main score: every class counts the same, whatever its number of test images (the measure of the ISIC
# 2018 lesion-diagnosis challenge). With one answer per image, micro-averaged precision, recall and F1 all equal accuracy, so they
# are not reported. MCC (Matthews correlation coefficient) and Cohen's kappa are the two usual chance-corrected scores for more
# than two classes. Precision and F1 depend on how common each class is in the test set; recall and balanced accuracy do not.
#
# The 95% intervals: photos of one lesion are not independent, so when the images carry their lesion (``lesions``), every interval
# comes from a bootstrap that resamples whole lesions within each true class. Without lesion information each image is its own
# lesion; accuracy and the recall of each class then get the Wilson score interval.
#
# ROC curves and AUC are deliberately absent: they need a score per class for every image, and a model that answers with one class
# name gives none.
# ==============================================================================================================================


UNREADABLE = "(unreadable)"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def confusion_matrix(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> list[list[int]]:
    """Rows: the true class (in the order of ``labels``). Columns: the answer, in the same order, then one column for unreadable."""
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("the class list contains a name twice")
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for true, answer in zip(y_true, y_pred, strict=True):  # strict: lists of different lengths are an error, not a shorter score
        if true not in index:
            raise ValueError(f"true label {true!r} is not one of the classes")
        if answer is not None and answer not in index:
            raise ValueError(f"answer {answer!r} is not one of the classes (an unreadable answer must be None)")
        matrix[index[true]][len(labels) if answer is None else index[answer]] += 1
    return matrix


def _scores(matrix: list[list[int]]) -> dict:
    """Every score that is computed from a confusion matrix (rows: true class; last column: unreadable)."""
    k = len(matrix)
    n = sum(map(sum, matrix))
    support = [sum(row) for row in matrix]
    answered = [sum(matrix[i][j] for i in range(k)) for j in range(k + 1)]  # how often each answer was given; the last is unreadable
    tp = [matrix[i][i] for i in range(k)]
    precision = [_ratio(tp[i], answered[i]) for i in range(k)]
    recall = [_ratio(tp[i], support[i]) for i in range(k)]
    f1 = [_ratio(2 * tp[i], 2 * tp[i] + (answered[i] - tp[i]) + (support[i] - tp[i])) for i in range(k)]
    present = [i for i in range(k) if support[i]]  # only these enter an average
    out = {"n": n, "support": support, "answered": answered, "tp": tp, "precision": precision, "recall": recall, "f1": f1,
           "present": present}

    def mean(values, weights=None):
        if not present:
            return None
        values = [0.0 if values[i] is None else values[i] for i in present]  # a class never answered: precision 0
        weights = [1] * len(present) if weights is None else [weights[i] for i in present]
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    out["macro"] = {"precision": mean(precision), "recall": mean(recall), "f1": mean(f1)}
    out["weighted"] = {"precision": mean(precision, support), "recall": mean(recall, support), "f1": mean(f1, support)}
    correct = sum(tp)
    if n:
        expected = sum(answered[j] * (support[j] if j < k else 0) for j in range(k + 1))  # unreadable is never a true class
        spread_answers = n * n - sum(a * a for a in answered)
        spread_truth = n * n - sum(s * s for s in support)
        out["mcc"] = _ratio(correct * n - expected, math.sqrt(spread_answers * spread_truth))
        out["kappa"] = _ratio(correct / n - expected / (n * n), 1 - expected / (n * n))
    else:
        out["mcc"] = out["kappa"] = None
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    """The q-quantile (0 to 1) of sorted values with linear interpolation between neighbours."""
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _bootstrap(y_true, y_pred, labels, lesions, resamples: int, seed: int) -> dict:
    """95% percentile intervals of every score that gets one.

    Whole lesions are resampled with replacement inside each true class (``lesions`` names each image's lesion; each image is its
    own lesion when it does not), so every resample keeps the number of lesions of each class, and photos of one lesion are drawn
    together. The generator is seeded: the same answers always give the same interval.
    """
    by_class = {}
    for label in labels:
        units = {}
        for i, true in enumerate(y_true):
            if true == label:
                units.setdefault(lesions[i], []).append(i)
        if units:
            by_class[label] = list(units.values())
    names = ["accuracy", "balanced_accuracy", "macro_f1"]
    if not by_class:
        return {name: None for name in names} | {"recall": {}}
    rng = random.Random(seed)
    values = {name: [] for name in names}
    recall = {label: [] for label in by_class}
    for _ in range(resamples):
        sample = [i for units in by_class.values() for _ in units for i in units[int(rng.random() * len(units))]]  # random() is
        true, pred = [y_true[i] for i in sample], [y_pred[i] for i in sample]  # stable across Python versions
        scores = _scores(confusion_matrix(true, pred, labels))
        values["accuracy"].append(sum(scores["tp"]) / scores["n"])
        values["balanced_accuracy"].append(scores["macro"]["recall"])
        values["macro_f1"].append(scores["macro"]["f1"])
        for label in by_class:
            recall[label].append(scores["recall"][labels.index(label)])

    def interval(series):  # every resample holds every class, so no value is ever undefined
        series = sorted(series)
        return (_percentile(series, 0.025), _percentile(series, 0.975))

    return {name: interval(series) for name, series in values.items()} | {"recall": {k: interval(v) for k, v in recall.items()}}


def summarize(y_true: list[str], y_pred: list[str | None], labels: list[str], lesions: list[str] | None = None) -> dict:
    """All metrics of one run.

    ``labels`` is the full class list; a class that has no true image appears in ``classes_without_test_images`` and is left out of
    the averages. ``lesions`` names the lesion of each image (photos of one lesion are resampled together for the intervals). With no answers at all, every
    score is None (and accuracy 0.0).
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    s = _scores(matrix)
    n = s["n"]
    correct = sum(s["tp"])
    if lesions is not None and len(lesions) != len(y_true):
        raise ValueError("there must be one lesion per image")
    lesions = list(lesions) if lesions is not None else list(range(len(y_true)))
    independent = len(set(lesions)) == len(lesions)  # every image its own lesion: Wilson intervals for accuracy and recall
    interval = _bootstrap(y_true, y_pred, labels, lesions, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {"support": s["support"][i], "answered": s["answered"][i], "tp": s["tp"][i],
                            "fp": s["answered"][i] - s["tp"][i], "fn": s["support"][i] - s["tp"][i],
                            "precision": s["precision"][i], "recall": s["recall"][i], "f1": s["f1"][i],
                            "recall_ci95": None if not s["support"][i] else
                            wilson(s["tp"][i], s["support"][i]) if independent else interval["recall"][label]}
    return {"n": n, "correct": correct, "unreadable": s["answered"][-1], "lesions": len(set(lesions)),
            "interval_method": "Wilson for accuracy and recall, bootstrap for the rest" if independent else "bootstrap over lesions",
            "balanced_accuracy": s["macro"]["recall"], "balanced_accuracy_ci95": interval["balanced_accuracy"],
            "accuracy": correct / n if n else 0.0, "accuracy_ci95": wilson(correct, n) if independent else interval["accuracy"],
            "classes_without_test_images": [label for i, label in enumerate(labels) if not s["support"][i]],
            "macro": {**s["macro"], "f1_ci95": interval["macro_f1"]},
            "weighted": s["weighted"],
            "mcc": s["mcc"], "kappa": s["kappa"],
            "per_class": per_class,
            "confusion_matrix": {"rows": list(labels), "columns": [*labels, UNREADABLE], "counts": matrix}}


# ==============================================================================================================================
# Model adapters
#
# The LLM adapters: one class per API, each answering one image plus one prompt in the same form.
#
#     anthropic          Claude through the Anthropic API
#     gemini             Gemini through the Google Gemini API
#     openai_compatible  GPT-4o at OpenAI, and any host that speaks the same chat API (for example Llama at OpenRouter or
#                        Hugging Face)
#
# Every adapter has ``describe()`` (what is asked: provider, model id, settings) and ``predict(image, mime_type, prompt)``. The
# answer is a dict with ``text`` (or None), ``finish_reason`` ("STOP" for a normal end, "MAX_TOKENS" when the answer was cut off at
# the output limit; any other value passes through), ``model_version`` (as the API reports it), ``response_id`` and ``usage``
# (token counts). A failure raises TransientError (worth retrying), QuotaExhausted (credit or a per-day quota is used up) or
# FatalError (retrying cannot fix it; the message says what to do). An adapter also names the ``packages`` whose versions the run
# card records and says whether its model id is ``pinned`` to one version; ``loaded_note`` and ``probe_note`` tell the setup check
# how to describe it.
#
# Every image is sent to the provider's servers, so the dataset's terms must allow it. The SDKs' own retries are off, because a
# run retries failures itself, waits longer each time and counts them. API keys are read from .env (git-ignored), never shown.
# ==============================================================================================================================


TIMEOUT_SECONDS = 120  # the longest wait for one answer; a stalled connection must never wait forever
IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")  # what the Anthropic and OpenAI-style APIs accept


class TransientError(Exception):
    """A failure worth retrying later (rate limit, server overload, network). The image is left unanswered, not scored."""


class QuotaExhausted(Exception):
    """Credit or a per-day quota is used up, so retrying within minutes cannot help. Answers already saved are kept."""


class FatalError(Exception):
    """A failure that retrying cannot fix (wrong key, unavailable model, refused setting). The message says what to do."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def load_env(path: str | Path = ".env") -> None:
    """Puts the NAME=value lines of a local .env file (git-ignored; it holds the API keys) into the environment.

    A variable that is already set keeps its value, so a key set in the terminal wins. Blank lines, # comments, lines
    without "=" and empty values are skipped. A missing file is not an error.
    """
    path = Path(path)
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} must be saved as plain UTF-8 text; the editor saved it in another encoding") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and value:
            os.environ.setdefault(name, value)


def load(provider: str, settings: dict):
    """The adapter named by ``provider`` in a config's [model] section, set up with that section's ``settings``."""
    adapters = {"anthropic": AnthropicModel, "gemini": GeminiModel, "openai_compatible": OpenAICompatibleModel}
    if provider not in adapters:
        raise ValueError(f"unknown model provider: {provider!r} (adapters: {', '.join(adapters)})")
    return adapters[provider](settings)


def _require(settings: dict, keys: tuple[str, ...], adapter: str) -> None:
    missing = [key for key in keys if key not in settings]
    if missing:
        raise ValueError(f"[model] is missing {', '.join(missing)} (the {adapter} adapter needs {', '.join(keys)})")
    if not isinstance(settings["id"], str) or not settings["id"]:
        raise ValueError("[model] id must be non-empty text")


def _whole(settings: dict, key: str) -> int:
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"[model] {key} must be a whole number of at least 1 (got {value!r})")
    return value


# ---------------------------------------------------------------------------------------------------------------------------------
# Claude (Anthropic)
# The key is ANTHROPIC_API_KEY or, if that is not set, CLAUDE_API_KEY. The current Claude models reject temperature, top_p and
# top_k, so none is sent and the provider's default sampling is used; the run card says so. No refusal fallback is requested:
# every answer comes from the model named in the config, and a refusal is recorded as finish reason REFUSAL.

ANTHROPIC_KEYS = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
ANTHROPIC_THINKING = ("between_tools", "adaptive")  # between_tools = no reasoning for a request without tools; "disabled" is refused
ANTHROPIC_EFFORT = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_FINISH = {"end_turn": "STOP", "stop_sequence": "STOP", "max_tokens": "MAX_TOKENS", "refusal": "REFUSAL"}


class AnthropicModel:
    packages = ("anthropic", "pillow")
    probe_note = "it is one request to Anthropic and costs a fraction of a cent of your API credit"

    def __init__(self, settings: dict):
        _require(settings, ("id", "max_tokens", "thinking", "effort"), "anthropic")
        self.model_id, self.max_tokens = settings["id"], _whole(settings, "max_tokens")
        self.thinking, self.effort = settings["thinking"], settings["effort"]
        if self.thinking not in ANTHROPIC_THINKING:
            raise ValueError(f"[model] thinking must be one of {list(ANTHROPIC_THINKING)} (got {self.thinking!r})")
        if self.effort not in ANTHROPIC_EFFORT:
            raise ValueError(f"[model] effort must be one of {list(ANTHROPIC_EFFORT)} (got {self.effort!r})")
        self.pinned = not self.model_id.endswith("-latest")  # a Claude model id without a date is a fixed version; "-latest" moves
        try:
            import anthropic
        except ImportError as exc:
            raise ValueError("The anthropic package is not installed: run pip install -r requirements.txt") from exc
        key = next((os.environ[name].strip() for name in ANTHROPIC_KEYS if os.environ.get(name, "").strip()), None)
        if key is None:
            raise ValueError("no Anthropic API key found: put the line CLAUDE_API_KEY=<your key> (or ANTHROPIC_API_KEY=<your key>) "
                             "in the .env file next to README.md")
        self.sdk = anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=TIMEOUT_SECONDS)
        self.loaded_note = f"{self.model_id}, API key found (not shown); each answer records the model version Anthropic reports"

    def describe(self) -> dict:
        return {"provider": "anthropic", "model_id": self.model_id, "sdk": "anthropic",
                "sdk_version": getattr(self.sdk, "__version__", None),
                "generation_config": {"max_tokens": self.max_tokens, "thinking": self.thinking, "effort": self.effort,
                                      "sampling": "provider default (temperature, top_p and top_k are not sent)"}}

    def _failure(self, exc) -> Exception:
        """The run's own kind of error for an API refusal: retry later, stop until credit is added, or fix a setting."""
        status, kind, code = exc.status_code, getattr(exc, "type", None), getattr(exc, "error_code", None)
        detail = getattr(exc, "api_message", None) or str(exc)
        if kind == "billing_error" or code == "enforced_spend_limit_reached":
            return QuotaExhausted(f"Anthropic cannot charge for more requests ({detail}). Add credit or raise the spend limit in the "
                                  "Anthropic console, then run the same command again.")
        if status in (408, 409, 429) or status >= 500:
            return TransientError(f"HTTP {status} {kind or ''}: {detail}".replace("  ", " "))
        if status == 401:
            return FatalError(f"Anthropic refused the API key (HTTP 401: {detail}). Check the key in .env.", status)
        if status == 403:
            return FatalError(f"Anthropic refused the request (HTTP 403: {detail}). The key may lack permission, or the account's country "
                              "or organisation may not be allowed to use the API.", status)
        if status == 404:
            return FatalError(f"Anthropic does not know the model {self.model_id!r} (HTTP 404: {detail}). Check the id in the config.", status)
        return FatalError(f"Anthropic refused the request (HTTP {status} {kind or ''}: {detail}). Fix the setting the message names.",
                          status)

    def predict(self, image: bytes, mime_type: str, prompt: str) -> dict:
        if mime_type not in IMAGE_TYPES:
            raise FatalError(f"Claude accepts JPEG, PNG, GIF and WebP images, not {mime_type}")
        content = [{"type": "image", "source": {"type": "base64", "media_type": mime_type,
                                                "data": base64.standard_b64encode(image).decode("ascii")}},
                   {"type": "text", "text": prompt}]
        try:
            response = self.client.messages.create(model=self.model_id, max_tokens=self.max_tokens,
                                                   thinking={"type": self.thinking}, output_config={"effort": self.effort},
                                                   messages=[{"role": "user", "content": content}])
        except self.sdk.APIStatusError as exc:
            raise self._failure(exc) from exc
        except self.sdk.APIConnectionError as exc:  # also a timeout
            raise TransientError(f"cannot reach the Anthropic API ({type(exc).__name__}: {exc})") from exc
        except self.sdk.APIError as exc:
            raise FatalError(f"the Anthropic SDK failed ({type(exc).__name__}: {exc})") from exc
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip() or None
        usage, stop = response.usage, response.stop_reason
        return {"text": text, "finish_reason": ANTHROPIC_FINISH.get(stop, str(stop).upper()), "model_version": response.model,
                "response_id": response.id,
                "usage": {"prompt_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                          "total_tokens": usage.input_tokens + usage.output_tokens}}


# ---------------------------------------------------------------------------------------------------------------------------------
# Gemini (Google)
# The key is GEMINI_API_KEY or GOOGLE_API_KEY. The SDK (google-genai) is imported only when a Gemini model is loaded, so the
# other models do not need it.

GEMINI_RETRYABLE = {429, 500, 502, 503, 504}


def _gemini_daily_quota(exc) -> str | None:
    """A readable message when a 429 is a per-day quota (waiting minutes cannot help); None for any other error."""
    message = str(exc.message or "")
    if exc.code != 429 or "per day" not in message.lower():
        return None
    limit = re.search(r"limit: (\d+)", message)
    model = re.search(r"model: ([\w.\-]+)", message)
    retry = re.search(r"retry in ((?:\d+h)?(?:\d+m)?\d+(?:\.\d+)?s)", message, re.I)
    return ("Daily quota used up" + (f" for {model.group(1)}" if model else "")
            + (f" (limit {limit.group(1)} requests per day" + (" on the free tier)" if "free tier" in message.lower() else ")") if limit else "")
            + (f"; Google says to retry in {re.sub(r'[.][0-9]+', '', retry.group(1))}" if retry else "")
            + ". Continue after the quota resets, with a paid key, or with another model; https://ai.dev/rate-limit shows your limits")


def _gemini_explain(exc) -> str:
    """Google's refusal in one readable message, with what to do about the cases seen so far."""
    text = f"Google refused the request (HTTP {exc.code} {exc.status}): {exc.message}"
    said = str(exc.message).lower()
    if exc.code == 404:
        hint = ("this model is not available to your key, or the id is misspelled; compare it with the model list at "
                "https://ai.google.dev/gemini-api/docs/models and put the right id in a new config")
    elif exc.code == 400 and "thinking" in said:
        hint = "this model refuses the thinking_level setting; set another level (MINIMAL, LOW, MEDIUM, HIGH) in a new config file"
    elif exc.code == 400 and "api key" in said:
        hint = "the API key is wrong: check the GEMINI_API_KEY line in .env"
    elif exc.code in (401, 403):
        hint = "the key was refused: check .env and your account in Google AI Studio"
    else:
        hint = ""
    return text + (f"\nWhat to do: {hint}." if hint else "")


class GeminiModel:
    packages = ("google-genai", "pillow", "httpx")

    def __init__(self, settings: dict):
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise ValueError("no Gemini API key found: put the line GEMINI_API_KEY=<your key> in a file named .env "
                             "in the repository folder (next to README.md)")
        _require(settings, ("id", "temperature", "seed", "max_output_tokens"), "gemini")
        if "thinking_budget" in settings:  # Google marks it deprecated for Gemini 3 and later; it can cause 400 errors
            raise ValueError("thinking_budget is not supported for Gemini 3 and later models: use thinking_level")
        try:
            import httpx  # google-genai 2.x sends its requests with httpx
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise ValueError("The google-genai package is not installed: run pip install -r requirements.txt") from exc
        self.model_id, self.httpx, self.errors, self.types = settings["id"], httpx, errors, types
        self.pinned = not self.model_id.endswith("-latest")  # gemini-flash-latest and the like move to newer models
        level = settings.get("thinking_level")
        if level is not None and level not in {x.value for x in types.ThinkingLevel}:  # the SDK only warns about a typo
            raise ValueError(f"thinking_level {level!r} is not one of {[x.value for x in types.ThinkingLevel]}")
        self.config = types.GenerateContentConfig(
            temperature=settings["temperature"], seed=settings["seed"],
            max_output_tokens=settings["max_output_tokens"],  # thinking tokens count toward this limit
            thinking_config=types.ThinkingConfig(thinking_level=level) if level is not None else None)
        if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY") \
                and os.environ["GOOGLE_API_KEY"] != os.environ["GEMINI_API_KEY"]:
            print("WARNING: GOOGLE_API_KEY and GEMINI_API_KEY are both set and differ; the Gemini SDK uses GOOGLE_API_KEY, "
                  "not the key in .env. Remove GOOGLE_API_KEY from the environment.", file=sys.stderr)
        self.client = genai.Client(http_options=types.HttpOptions(timeout=int(TIMEOUT_SECONDS * 1000)))  # milliseconds

    def describe(self) -> dict:
        return {"provider": "gemini", "model_id": self.model_id, "sdk": "google-genai",
                "sdk_version": importlib.metadata.version("google-genai"),
                "generation_config": self.config.model_dump(exclude_none=True, mode="json")}

    def predict(self, image: bytes, mime_type: str, prompt: str) -> dict:
        try:
            response = self.client.models.generate_content(
                model=self.model_id, contents=[self.types.Part.from_bytes(data=image, mime_type=mime_type), prompt], config=self.config)
        except self.errors.APIError as exc:
            if daily := _gemini_daily_quota(exc):
                raise QuotaExhausted(daily) from exc
            if exc.code in GEMINI_RETRYABLE:
                raise TransientError(f"HTTP {exc.code} {exc.status}: {exc.message}") from exc
            raise FatalError(_gemini_explain(exc), exc.code) from exc
        except self.httpx.TransportError as exc:  # no connection, connection lost, or no answer within the timeout
            raise TransientError(f"network problem ({type(exc).__name__}: {exc})") from exc
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content and candidate.content.parts else []
        text = "".join(p.text for p in parts if p.text and not p.thought) or None
        finish = candidate.finish_reason.value if candidate and candidate.finish_reason else None
        feedback = response.prompt_feedback
        if finish is None and feedback and feedback.block_reason:
            finish = f"PROMPT_BLOCKED_{feedback.block_reason.value}"
        usage = response.usage_metadata
        return {"text": text, "finish_reason": finish, "model_version": response.model_version, "response_id": response.response_id,
                "usage": {"prompt_tokens": usage.prompt_token_count if usage else None,
                          "output_tokens": usage.candidates_token_count if usage else None,
                          "thinking_tokens": usage.thoughts_token_count if usage else None,
                          "total_tokens": usage.total_token_count if usage else None}}


# ---------------------------------------------------------------------------------------------------------------------------------
# OpenAI-style chat API: GPT-4o at OpenAI, and any host that speaks the same API (for example Llama at OpenRouter or Hugging Face)
# The host and the key come from the config: base_url (absent = OpenAI) and api_key_env, the name of the variable in .env that
# holds the key (default OPENAI_API_KEY). Decoding is as the base paper's: temperature 0 and a fixed seed; the host's seed is
# best effort, so a repeated run can still differ slightly. A host may serve one model from several back ends; extra_body passes
# a host-specific choice through (for OpenRouter, provider = {order = ["<host>"], allow_fallbacks = false} keeps every answer on
# one back end), and the back end's name is recorded with every answer when the host reports it.

OPENAI_DEFAULT_KEY = "OPENAI_API_KEY"
OPENAI_DETAIL = ("low", "high", "auto")  # how closely OpenAI looks at the image (more detail = more tokens)
OPENAI_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "content_filter": "CONTENT_FILTER"}
DATED = re.compile(r"\d{4}-\d{2}-\d{2}$")  # a model id ending in a date is a fixed snapshot; one without may move
DAILY = re.compile(r"per[- ]day|daily", re.IGNORECASE)  # waiting minutes cannot help with a per-day limit


class OpenAICompatibleModel:
    packages = ("openai", "pillow")
    probe_note = "it is one request: a fraction of a cent of credit, or one of a free model's daily requests"

    def __init__(self, settings: dict):
        _require(settings, ("id", "temperature", "seed", "max_tokens"), "openai_compatible")
        self.model_id = settings["id"]
        self.temperature = settings["temperature"]
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not 0 <= self.temperature <= 2:
            raise ValueError(f"[model] temperature must be a number from 0 to 2 (got {self.temperature!r})")
        self.seed = settings["seed"]
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"[model] seed must be a whole number (got {self.seed!r})")
        self.max_tokens = _whole(settings, "max_tokens")
        self.base_url = settings.get("base_url")
        if self.base_url is not None and (not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://"))):
            raise ValueError(f"[model] base_url must be a web address starting with https:// (got {self.base_url!r})")
        self.key_name = settings.get("api_key_env", OPENAI_DEFAULT_KEY)
        if not isinstance(self.key_name, str) or not self.key_name:
            raise ValueError("[model] api_key_env must be the name of a variable in .env")
        self.image_detail = settings.get("image_detail")
        if self.image_detail is not None and self.image_detail not in OPENAI_DETAIL:
            raise ValueError(f"[model] image_detail must be one of {list(OPENAI_DETAIL)} (got {self.image_detail!r})")
        self.extra_body = settings.get("extra_body")
        if self.extra_body is not None and not isinstance(self.extra_body, dict):
            raise ValueError("[model] extra_body must be a table")
        self.pinned = settings.get("pinned", bool(DATED.search(self.model_id)))  # a test run needs a fixed version
        if not isinstance(self.pinned, bool):
            raise ValueError("[model] pinned must be true or false")
        try:
            import openai
        except ImportError as exc:
            raise ValueError("The openai package is not installed: run pip install -r requirements.txt") from exc
        key = os.environ.get(self.key_name, "").strip()
        if not key:
            raise ValueError(f"no API key found: put the line {self.key_name}=<your key> in the .env file next to README.md")
        self.sdk = openai
        self.client = openai.OpenAI(api_key=key, base_url=self.base_url, max_retries=0, timeout=TIMEOUT_SECONDS)
        host = urlparse(self.base_url).netloc if self.base_url else "api.openai.com"
        self.loaded_note = f"{self.model_id} at {host}, API key {self.key_name} found (not shown); each answer records the model name the host reports"

    def describe(self) -> dict:
        return {"provider": "openai_compatible", "model_id": self.model_id, "base_url": str(self.client.base_url), "sdk": "openai",
                "sdk_version": getattr(self.sdk, "__version__", None),
                "generation_config": {"temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
                                      "image_detail": self.image_detail, "extra_body": self.extra_body}}

    def _failure(self, exc) -> Exception:
        """The run's own kind of error for a refusal by the host: retry later, stop until credit or quota returns, or fix a setting."""
        status, code = exc.status_code, getattr(exc, "code", None)
        detail = getattr(exc, "message", None) or str(exc)
        if code == "insufficient_quota" or status == 402 or (status == 429 and DAILY.search(detail)):
            return QuotaExhausted(f"The host cannot serve more requests ({detail}). Add credit, wait for the daily limit to reset, "
                                  "or use another model; then run the same command again.")
        if status in (408, 409, 429) or status >= 500:
            return TransientError(f"HTTP {status}: {detail}")
        if status == 401:
            return FatalError(f"The host refused the API key (HTTP 401: {detail}). Check {self.key_name} in .env.", status)
        if status == 403:
            return FatalError(f"The host refused the request (HTTP 403: {detail}). The key may lack permission, or the account's "
                              "country or organisation may not be allowed to use this model.", status)
        if status == 404:
            return FatalError(f"The host does not know the model {self.model_id!r} (HTTP 404: {detail}). Check the id and base_url "
                              "in the config.", status)
        return FatalError(f"The host refused the request (HTTP {status}: {detail}). Fix the setting the message names.", status)

    def predict(self, image: bytes, mime_type: str, prompt: str) -> dict:
        if mime_type not in IMAGE_TYPES:
            raise FatalError(f"This adapter accepts JPEG, PNG, GIF and WebP images, not {mime_type}")
        picture = {"url": f"data:{mime_type};base64,{base64.standard_b64encode(image).decode('ascii')}"}
        if self.image_detail:
            picture["detail"] = self.image_detail
        request = {"model": self.model_id, "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": picture},
                                                              {"type": "text", "text": prompt}]}]}
        if self.extra_body:
            request["extra_body"] = self.extra_body
        try:
            response = self.client.chat.completions.create(**request)
        except self.sdk.APIStatusError as exc:
            raise self._failure(exc) from exc
        except self.sdk.APIConnectionError as exc:  # also a timeout
            raise TransientError(f"cannot reach the host ({type(exc).__name__}: {exc})") from exc
        except self.sdk.APIError as exc:
            raise FatalError(f"the OpenAI SDK failed ({type(exc).__name__}: {exc})") from exc
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        text = ((message.content or "").strip() or None) if message else None
        finish = OPENAI_FINISH.get(choice.finish_reason, str(choice.finish_reason).upper()) if choice else "NO_CHOICE"
        if message and getattr(message, "refusal", None) and not text:
            finish = "REFUSAL"
        backend = (getattr(response, "model_extra", None) or {}).get("provider")  # OpenRouter says which back end answered
        usage = response.usage
        return {"text": text, "finish_reason": finish, "model_version": f"{response.model}@{backend}" if backend else response.model,
                "response_id": response.id,
                "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None),
                          "total_tokens": getattr(usage, "total_tokens", None)}}


# ==============================================================================================================================
# The pipeline and its command line
# ==============================================================================================================================


DATA = Path("data")  # one folder per dataset: data/image/ (data/text/ and data/multimodal/ in later phases)
DATASETS = ("image",)  # the datasets a config can name
TABLE_NAME, FILES = "dataset.csv", "dataset"  # in a dataset's folder: the table of the prepared dataset, and its files
RESULTS = Path("results")  # results/<dataset>/<model>/<run>/
SPLITS = ("train", "val", "test")  # the splits of the prepared dataset
RUN_SPLITS = ("val", "test")  # the splits a model answers; the training split is only ever looked up, never answered
COLUMNS = ("split", "label", "path", "sha256")  # required in the table; other columns are kept there but not needed here
GROUP = "lesion_group"  # optional column: the lesion each image shows (photos of one lesion are resampled together for the intervals)
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
SYSTEM_FILES = {"thumbs.db", "desktop.ini"}  # written by Windows Explorer into folders it shows; never part of the dataset
SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
PLACEHOLDER = "{labels}"
LEAD_INS = ("diagnosis", "answer", "final answer")  # the words the answer reader forgives in front of a colon and the class name
MAX_ATTEMPTS = 8  # waits of 10, 20, 40, 80, 120, 120, 120 seconds between them: about 8.5 minutes in all
PROBE_ATTEMPTS = 3
PROBE_PROMPT = "Reply with the single word: ok."


# ---------------------------------------------------------------------------------------------------------------------------------
# The prepared dataset


@dataclass(frozen=True)
class Item:
    split: str
    label: str
    path: str  # inside the dataset's folder of files, with forward slashes: <split>/<label>/<file>
    sha256: str
    lesion_group: str = ""  # the lesion the image shows; empty when the table does not say


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    """The table and the folder of files of one prepared dataset: data/<dataset>/dataset.csv and data/<dataset>/dataset/."""
    return DATA / dataset / TABLE_NAME, DATA / dataset / FILES


TABLE, IMAGES = dataset_paths("image")  # the defaults below


def read_table(table: str | Path = TABLE) -> list[Item]:
    """The rows of the table, after checking that they describe one consistent dataset (no image is opened)."""
    table = Path(table)
    if not table.is_file():
        raise FileNotFoundError(f"{table.as_posix()} not found: the prepared dataset does not exist yet (run the dataset "
                                "preparation first)")
    with table.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{table.as_posix()} lacks the column(s): {', '.join(missing)}")
        items = [Item(*(row[c] for c in COLUMNS), lesion_group=(row.get(GROUP) or "").strip()) for row in reader]
    if not items:
        raise ValueError(f"{table.as_posix()} lists no image")
    first_line = {}
    for line, item in enumerate(items, start=2):  # line 1 is the header
        where = f"{table.as_posix()} line {line}"
        if item.split not in SPLITS:
            raise ValueError(f"{where}: split {item.split!r} is not one of {list(SPLITS)}")
        parts = item.path.split("/")
        if len(parts) != 3 or parts[:2] != [item.split, item.label] or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"{where}: path {item.path!r} must be <split>/<label>/<file>, here {item.split}/{item.label}/<file>")
        if Path(item.path).suffix.lower() not in MEDIA_TYPES:
            raise ValueError(f"{where}: {item.path} is not a JPEG or PNG file")
        if not SHA256.fullmatch(item.sha256):
            raise ValueError(f"{where}: sha256 must be 64 lowercase hexadecimal characters")
        for key, what in ((item.path, "path"), (item.sha256, "image")):
            if key in first_line:  # a second copy of one image could sit in two splits: a leak between them
                raise ValueError(f"{where}: the same {what} is also on line {first_line[key]}")
            first_line[key] = line
    return items


def load_dataset(table: str | Path = TABLE, root: str | Path = IMAGES) -> list[Item]:
    """The table's rows, after checking that the images on disk are exactly the ones it lists, byte for byte."""
    items = read_table(table)
    root = Path(root)
    changed, missing = [], []
    for item in items:
        file = root / item.path
        if not file.is_file():
            missing.append(item.path)
        elif hashlib.sha256(file.read_bytes()).hexdigest() != item.sha256:
            changed.append(item.path)
    listed = {item.path for item in items}
    extra = sorted(p.relative_to(root).as_posix() for split in SPLITS if (root / split).is_dir() for p in (root / split).rglob("*")
                   if p.is_file() and p.relative_to(root).as_posix() not in listed and not p.name.startswith(".")
                   and p.name.lower() not in SYSTEM_FILES)
    if changed or missing or extra:
        examples = "; ".join(f"{why}: {paths[0]}" for why, paths in (("changed", changed), ("missing", missing),
                                                                       ("not in the table", extra)) if paths)
        raise ValueError(f"the images in {root.as_posix()} differ from {Path(table).as_posix()}: {len(changed)} changed, "
                         f"{len(missing)} missing, {len(extra)} not in the table ({examples}). Run the dataset preparation "
                         "again, or restore the images")
    return items


def classes(items: list[Item]) -> list[str]:
    """The answer options: the classes of the training split. A validation or test class without training images is refused."""
    train = sorted({item.label for item in items if item.split == "train"})
    if not train:
        raise ValueError("the training split is empty, so the classes are unknown")
    unknown = sorted({item.label for item in items if item.split != "train"} - set(train))
    if unknown:
        raise ValueError(f"classes without training images: {unknown}")
    return train


def items_hash(items: list[Item]) -> str:
    """One hash for a list of images: it changes if any file, label, path, split or lesion group changes."""
    lines = sorted(f"{i.split}\t{i.label}\t{i.path}\t{i.sha256}" + (f"\t{i.lesion_group}" if i.lesion_group else "") for i in items)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_image(item: Item, root: str | Path = IMAGES) -> tuple[bytes, str, dict]:
    """The bytes every model receives for one image (the file exactly as stored), its media type, and a record of what was sent."""
    content = (Path(root) / item.path).read_bytes()
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except Exception as exc:  # any decoding failure stops the run with a message, never a silent skip
        raise ValueError(f"{item.path} is not a readable image ({exc})") from exc
    return content, MEDIA_TYPES[Path(item.path).suffix.lower()], {
        "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "width": width, "height": height}


# ---------------------------------------------------------------------------------------------------------------------------------
# The prompt and the answer reader: the same for every model, so differences in results come from the models, not from how their
# answers were read


def load_template(path: str | Path) -> str:
    """The prompt template; it must contain the label placeholder exactly once."""
    text = Path(path).read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f"{path} must contain {PLACEHOLDER} exactly once")
    return text


def build_prompt(template: str, labels: list[str]) -> str:
    """The prompt for one image: the class names, sorted, one per line."""
    return template.replace(PLACEHOLDER, "\n".join(f"- {label}" for label in sorted(labels))).strip()


def _normalise(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text.strip().lower())  # bold, italic and code marks, wherever they stand
    text = re.sub(r"^(final\s+)?(diagnosis|answer)\s*:\s*", "", text.strip())
    text = text.strip(" \t\r\n\"'-.")
    return re.sub(r"\s+", " ", text)


def parse_answer(text: str | None, labels: list[str]) -> str | None:
    """The class the answer names, or None when it is not exactly one class name.

    Only formatting is forgiven (case, spaces, quotes, bold, italic and code marks, a trailing full stop, a leading
    "Diagnosis:", "Answer:" or "Final answer:"). An answer that adds words or names two classes counts as unreadable, which the
    metrics score as wrong.
    """
    if not text:
        return None
    answer = _normalise(text)
    for label in labels:
        if answer == _normalise(label):
            return label
    return None


def can_still_become_readable(text: str | None, labels: list[str]) -> bool:
    """For an answer that was cut off at the output limit: could the rest of it have made it a readable answer?

    True when what is there, after the formatting that parse_answer() forgives, is empty or only the beginning of a class name or
    of a "Diagnosis:"-style lead-in. False when it already differs from every class name (an essay, a second class name): no
    continuation can then make it exactly one class name, so the cut-off did not change how the answer is scored.
    """
    answer = _normalise(text or "")
    return any(start.startswith(answer) for start in [_normalise(label) for label in labels] + list(LEAD_INS))


# ---------------------------------------------------------------------------------------------------------------------------------
# Crash-safe files: a run can be stopped at any moment (power cut, Ctrl+C) and continued without losing or damaging answers


def _split_last_line(content: bytes) -> tuple[str, bytes]:
    """The complete lines of a JSON-lines file (decoded strictly) and the bytes after its last end of line."""
    cut = content.rfind(b"\n") + 1
    return content[:cut].decode("utf-8"), content[cut:]


def _complete_row(tail: bytes) -> dict | None:
    """The row in bytes that lack their end of line, or None when they are a cut-off line (even one cut inside a character)."""
    try:
        row = json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


def read_jsonl(path: Path) -> list[dict]:
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


def repair_jsonl(path: Path) -> None:
    """Before answers are appended: remove an incomplete last line, or add the missing end of line after a complete one."""
    if not path.exists():
        return
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return
    head, tail = _split_last_line(content)
    if _complete_row(tail) is not None:
        fixed = content + b"\n"
    else:
        fixed = head.encode("utf-8")
        print(f"{path}: removed an incomplete last line (the program was stopped while writing it); that image is answered again")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(fixed)
    os.replace(tmp, path)


def write_json(path: Path, content: dict) -> None:
    """Written to a temporary file first, so a stop in the middle never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


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
    rate, seed = config["run"]["requests_per_minute"], config["run"]["order_seed"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"{path}: [run] requests_per_minute must be a number above 0 (got {rate!r})")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path}: [run] order_seed must be a whole number (got {seed!r})")
    return config


# ---------------------------------------------------------------------------------------------------------------------------------
# The setup check (python -m src.evaluate check): each line says OK or FAIL, and a FAIL says what to do


def _blank_image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "PNG")
    return buffer.getvalue()


def _probe(model, say) -> tuple[dict | None, str | None]:
    """The answer to one blank-image request, or (None, reason) when it failed. Retries transient failures a few times;
    a used-up quota and a refusal are not retried."""
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return model.predict(_blank_image(), "image/png", PROBE_PROMPT), None
        except TransientError as exc:
            if attempt == PROBE_ATTEMPTS:
                return None, f"temporary problem, not a setting: {exc}"
            say(f"       attempt {attempt}/{PROBE_ATTEMPTS} failed ({exc}); waiting {5 * attempt} s")
            time.sleep(5 * attempt)
        except (QuotaExhausted, FatalError) as exc:
            return None, str(exc)
    raise AssertionError("unreachable")


def check_setup(config: dict, config_path: str, say=print) -> bool:
    """Prints one line per check and returns True when none failed. No dataset image is sent."""
    failed = False

    def line(status: str, what: str, detail: str = "") -> None:
        nonlocal failed
        failed = failed or status == "FAIL"
        say(f"[{status:>4}] {what}" + (f": {detail}" if detail else ""))

    line("OK", "Config", config_path)
    try:
        load_template(config["prompt"]["file"])
        line("OK", "Prompt file", config["prompt"]["file"])
    except (OSError, ValueError) as exc:
        line("FAIL", "Prompt file", str(exc))
    try:  # every image is hashed and compared with the committed table, as a run does
        items = load_dataset(*dataset_paths(config["dataset"]))
        labels = classes(items)
        counts = Counter(item.split for item in items)
        line("OK", "Prepared dataset", f"{len(items)} images ({', '.join(f'{s} {counts[s]}' for s in SPLITS)}), {len(labels)} classes")
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
        say(f"       asking {settings['id']} one question with a blank image (nothing from the dataset is sent; "
            f"{getattr(model, 'probe_note', 'it counts as one request against the daily quota')}) ...")
        answer, problem = _probe(model, say)
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
# Runs (python -m src.evaluate run): one folder per run, every answer saved as it arrives


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


def _ask(model, image: bytes, mime: str, text: str, throttle, say=print) -> tuple[dict, int]:
    """One answer and the number of attempts it took. Waits longer after each transient failure; raises when all fail."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        throttle()
        try:
            return model.predict(image, mime, text), attempt
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
    table, root = dataset_paths(config["dataset"])
    everything = load_dataset(table, root)  # stops if any image differs from the committed table
    labels = classes(everything)
    items = sorted((item for item in everything if item.split == split), key=lambda item: item.path)
    if not items:
        raise ValueError(f"the {split} split of the prepared dataset is empty")
    # A seeded random order, so that any drift of a hosted model during a run is not lined up with the classes.
    random.Random(config["run"]["order_seed"]).shuffle(items)
    items = items[: limit or None]

    prompt_text = build_prompt(load_template(config["prompt"]["file"]), labels)
    config_bytes = Path(config_path).read_bytes()
    fingerprint = hashlib.sha256(config_bytes + prompt_text.encode("utf-8") + items_hash(items).encode()).hexdigest()
    load_env()
    model_dir = results / config["dataset"] / config["name"]
    job = {"config": config, "config_path": config_path, "split": split, "items": items, "labels": labels,
           "prompt_text": prompt_text, "fingerprint": fingerprint, "model_dir": model_dir, "git": git,
           "table": table, "root": root, "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest()}
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
            raise SystemExit(f"a finished test run already exists for this config, made with other settings, prompt or images: "
                             f"{folder} (results are not overwritten). A changed setting needs a new config file")
        raise SystemExit(f"an unfinished test run made with other settings, prompt or images exists: {folder}. Delete that "
                         "folder to start over, or undo the change")


def _run_once(job: dict, run_id: str) -> Path:
    """One run over the job's images: answers are saved as they arrive in ``<run_id>.partial``, renamed when complete."""
    config, items, labels, prompt_text, fingerprint = (job["config"], job["items"], job["labels"], job["prompt_text"],
                                                       job["fingerprint"])
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
                "prompt": prompt_text, "labels": labels, "n_images": len(items), "images_sha256": items_hash(items),
                "dataset_table": {"file": job["table"].as_posix(), "sha256": job["table_sha256"]}, "git": job["git"],
                "versions": _versions(getattr(model, "packages", ("pillow",))),
                "started_utc": _now(), "finished_utc": None, "model": None, "model_versions_returned": []}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["fingerprint"] != fingerprint:
            raise SystemExit(f"{out} was started with a different config, prompt or image set; delete that folder to start "
                             "over, or choose another --run-id")
    card.setdefault("sessions", []).append({"utc": _now(), "commit": job["git"]["commit"],
                                            "uncommitted_changes": job["git"]["uncommitted_changes"]})
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
                image, mime, sent = read_image(item, job["root"])
                answer, attempt = _ask(model, image, mime, prompt_text, throttle, say)
                parsed = parse_answer(answer["text"], labels)
                line = {"path": item.path, "label": item.label, "lesion_group": item.lesion_group, "prediction": parsed, "raw_text": answer["text"],
                        "finish_reason": answer["finish_reason"], "model_version": answer["model_version"],
                        "response_id": answer["response_id"], "usage": answer["usage"], "attempts": attempt, "image_sent": sent,
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
                         "program showing it and run the same command again; no image is sent again.") from exc
    return final


# ---------------------------------------------------------------------------------------------------------------------------------
# The report of one run (python -m src.evaluate report): metrics.json and report.md, from the saved answers only


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(pair) -> str:
    return "–" if pair is None else f"{_pct(pair[0])} to {_pct(pair[1])}"


def _share(x: float | None) -> str:
    """A share as a percentage; a dash when the score is undefined."""
    return "–" if x is None else _pct(x)


def _coefficient(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _scoring(title: str, m: dict) -> list[str]:
    """The metrics of one scoring as report lines: the summary table, the per-class table and the confusion matrix."""
    counted = [c for c, v in m["per_class"].items() if v["support"]]
    by_lesion = m["interval_method"] == "bootstrap over lesions"
    simple = "bootstrap over lesions" if by_lesion else "Wilson"
    lines = [f"## {title}", "", "| Metric | Value | 95% interval |", "|---|---|---|",
             f"| **Balanced accuracy (main score: the mean recall of the classes)** | **{_share(m['balanced_accuracy'])}** | "
             f"{_ci(m['balanced_accuracy_ci95'])} (bootstrap{' over lesions' if by_lesion else ''}) |",
             f"| Accuracy | {_pct(m['accuracy'])} ({m['correct']}/{m['n']}) | {_ci(m['accuracy_ci95'])} ({simple}) |",
             f"| Unreadable answers (counted as wrong) | {m['unreadable']} | |",
             f"| Macro-F1 (classes with images: {len(counted)}) | {_share(m['macro']['f1'])} | {_ci(m['macro']['f1_ci95'])} "
             f"(bootstrap{' over lesions' if by_lesion else ''}) |",
             f"| Macro-precision | {_share(m['macro']['precision'])} | |",
             f"| Weighted F1 (by number of true images) | {_share(m['weighted']['f1'])} | |",
             f"| Matthews correlation (MCC, -1 to 1) | {_coefficient(m['mcc'])} | |",
             f"| Cohen's kappa (-1 to 1) | {_coefficient(m['kappa'])} | |", "",
             "### Per class", "", f"| Class | True images | Answered as this class | Precision | Recall (95% interval: {simple}) | F1 |",
             "|---|---|---|---|---|---|"]
    for label, c in m["per_class"].items():
        recall = _share(c["recall"]) + (f" ({_ci(c['recall_ci95'])})" if c["recall_ci95"] else "")
        lines.append(f"| {label} | {c['support']} | {c['answered']} | {_share(c['precision'])} | {recall} | {_share(c['f1'])} |")
    matrix = m["confusion_matrix"]
    numbers = range(1, len(matrix["rows"]) + 1)
    lines += ["", "### Confusion matrix", "",
              "Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.", "",
              "| True class | " + " | ".join(str(i) for i in numbers) + " | unreadable |",
              "|---|" + "---|" * (len(matrix["rows"]) + 1)]
    lines += [f"| {i}. {label} | " + " | ".join(str(v) for v in row) + " |"
              for i, label, row in zip(numbers, matrix["rows"], matrix["counts"])]
    absent = m["classes_without_test_images"]
    supports = {v["support"] for v in m["per_class"].values() if v["support"]}
    lines += ["", "Notes on these scores:",
              "- Balanced accuracy is the main score: every class counts the same, whatever its number of test images (the measure "
              "of the ISIC 2018 lesion-diagnosis challenge).",
              *([f"- Every class has the same number of test images ({supports.pop()}), so plain accuracy equals balanced accuracy."]
                if len(supports) == 1 and len(counted) > 1 else []),
              "- Precision and F1 depend on how common each class is in the test set; with the same number of images per class, the "
              "precision of a rarer disease is higher than it would be in a clinic, so it is not a clinical figure. Recall and "
              "balanced accuracy do not depend on it.",
              (f"- Photos of one lesion are not independent: the intervals resample whole lesions ({m['lesions']} lesions for "
               f"{m['n']} images)." if by_lesion else "- Every image counts as its own lesion (the run recorded no lesion groups)."),
              *([f"- No image has the true class {', '.join(absent)}: its recall is undefined and it is left out of the averages."]
                if absent else []),
              "- A dash means undefined: precision when a class was never answered, recall when no image has that class. A class "
              "that was never answered counts as precision 0 in the macro-precision.",
              "- With one answer per image, micro-averaged precision, recall and F1 equal accuracy, so they are not listed.",
              "- No ROC curve or AUC: the models answer with one class name, not a score per class, so there is nothing to rank.", ""]
    return lines


def write_report(run_dir: str | Path) -> Path:
    """Writes metrics.json and report.md from a run's saved answers."""
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    preds = read_jsonl(run_dir / "predictions.jsonl")
    labels = card["labels"]
    main = summarize([p["label"] for p in preds], [p["prediction"] for p in preds], labels,
                     [p.get("lesion_group") or p["path"] for p in preds])
    write_json(run_dir / "metrics.json", {"main": main})

    complete = card["status"] == "complete" and len(preds) == card["n_images"]
    lines = [f"# {card['name']}: {card['split']} split, run {card['run_id']}", ""]
    if card["split"] == "val":
        lines += ["**Validation run: for tuning the method, not a reported result.**", ""]
    if not complete:
        lines += [f"**Incomplete: {len(preds)} of {card['n_images']} images answered.**", ""]
    model = card.get("model") or {}
    table = card["dataset_table"]
    lines += [f"- Model requested: `{model.get('model_id')}`; versions returned by the API: "
              f"{', '.join(f'`{v}`' for v in card['model_versions_returned']) or 'not recorded yet'}",
              f"- Generation settings: `{json.dumps(model.get('generation_config'))}`",
              f"- Started {card['started_utc']}, finished {card['finished_utc']} (UTC)",
              f"- Prompt: `{card['prompt_file']}` (SHA-256 `{card['prompt_sha256'][:12]}`); images: {card['n_images']} "
              f"(hash `{card['images_sha256'][:12]}`), sent exactly as stored; dataset table `{table['file']}` "
              f"(SHA-256 `{table['sha256'][:12]}`); Git commit `{(card['git']['commit'] or 'unknown')[:10]}`", ""]
    lines += _scoring("Results", main)

    sessions = card.get("sessions", [])
    commits = sorted({s["commit"] or "unknown" for s in sessions})
    finish = Counter(p["finish_reason"] for p in preds)
    other = {k: v for k, v in finish.items() if k not in ("STOP", "MAX_TOKENS")}
    cut = [p for p in preds if p["finish_reason"] == "MAX_TOKENS"]
    harmful = sum(can_still_become_readable(p["raw_text"], labels) for p in cut)  # the limit may have decided these
    lines += ["## Notes", "",
              f"- How answers ended (finish reason): {', '.join(f'{k}: {v}' for k, v in sorted(finish.items(), key=str))}",
              *([f"- **{harmful} answers were cut off at the output limit (finish reason MAX_TOKENS) and count as wrong: "
                 "the output limit in the config (max_tokens or max_output_tokens) is too low for this model (hidden reasoning "
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
              + "".join(f"; {i['utc']} after {i['answered']} answers ({i['kind']})" for i in card.get("interruptions", []))]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------------------------------------------------------------
# The command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="test everything a run needs, without sending any dataset image")
    c.add_argument("--config", required=True)
    c.add_argument("--model", help="test this model id instead of the config's (the config file is not changed)")
    r = sub.add_parser("run", help="run one model on the validation or the test split")
    r.add_argument("--config", required=True)
    r.add_argument("--split", choices=RUN_SPLITS, required=True)
    r.add_argument("--run-id", help="folder name; by default one that follows from the settings, so a stopped run resumes")
    r.add_argument("--limit", type=int, help="only the first N images of the validation split, in the seeded order")
    rep = sub.add_parser("report", help="rewrite metrics.json and report.md from a run's saved answers")
    rep.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            config = load_config(args.config)
            if args.model:
                config["model"] = {**config["model"], "id": args.model}
            sys.exit(0 if check_setup(config, args.config) else 1)
        elif args.command == "run":
            run(args.config, args.split, args.run_id, args.limit)
        else:
            print(write_report(args.run_dir))
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
