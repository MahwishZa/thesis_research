"""Model adapters. Each adapter answers one image plus one prompt and returns the raw answer and its metadata.

To add a model later: write one file here with a class that has ``predict`` and ``describe``, and add it to ``load``.
"""

from __future__ import annotations

import os
from pathlib import Path


class TransientError(Exception):
    """A failure worth retrying later (rate limit, server overload, network). The image is left unanswered, not scored."""


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


def settings_variants(provider: str, settings: dict) -> list[tuple[str, dict]]:
    """Alternative settings worth probing for a provider (name, settings); empty when the provider has none."""
    if provider == "gemini":
        from .gemini import thinking_variants

        return thinking_variants(settings)
    return []


def load(provider: str, settings: dict):
    if provider == "gemini":
        from .gemini import GeminiModel  # imported here so tests and other models do not need the Gemini SDK

        return GeminiModel(settings)
    raise ValueError(f"unknown model provider: {provider!r}")
