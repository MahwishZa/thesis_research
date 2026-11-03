"""Model adapters. Each adapter answers one image plus one prompt and returns the raw answer and its metadata.

To add a model later: write one file here with a class that has ``predict`` and ``describe``, and add it to ``load``.
"""

from __future__ import annotations


class TransientError(Exception):
    """A failure worth retrying later (rate limit, server overload). The image is left unanswered, not scored."""


def load(provider: str, settings: dict):
    if provider == "gemini":
        from .gemini import GeminiModel  # imported here so tests and other models do not need the Gemini SDK

        return GeminiModel(settings)
    raise ValueError(f"unknown model provider: {provider!r}")
