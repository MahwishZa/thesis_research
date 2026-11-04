"""Model adapters. Each adapter answers one image plus one prompt and returns the raw answer and its metadata.

An adapter is one file in this folder with a class that has ``describe()`` (what was asked: provider, model id, settings) and
``predict(image, mime_type, prompt)``, plus a branch in ``load``. ``predict`` returns a dict with ``text`` (the answer, or None),
``finish_reason`` ("STOP" for a normal end, "MAX_TOKENS" when the answer was cut off at the output limit, any other value
passes through), ``model_version`` (as the API returns it), ``response_id`` and ``usage`` (token counts). It raises
TransientError for a failure worth retrying, QuotaExhausted when a per-day quota is used up, and FatalError (with a message that
says what to do) for anything retrying cannot fix. An adapter may also offer ``list_models()`` for ``check --list-models``.

Adapters: ``gemini`` (src/models/gemini.py).
"""

from __future__ import annotations

import os
from pathlib import Path


class TransientError(Exception):
    """A failure worth retrying later (rate limit, server overload, network). The image is left unanswered, not scored."""


class QuotaExhausted(Exception):
    """A per-day quota is used up, so retrying within minutes cannot help. Answers already saved are kept."""


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
    """The adapter for ``provider`` (a branch per provider, importing its module only when used)."""
    if provider == "gemini":
        from .gemini import GeminiModel  # imported here so tests and other providers do not need the Gemini SDK

        return GeminiModel(settings)
    raise ValueError(f"unknown model provider: {provider!r} (adapters: gemini)")
