"""The diagnosis prompt and the strict reader for the model's answer.

Every model gets the same prompt and is read by the same rule, so differences in results come from the models,
not from how their answers were interpreted.
"""

from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER = "{labels}"


def load_template(path: str | Path) -> str:
    """The prompt template; it must contain the label placeholder exactly once."""
    text = Path(path).read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f"{path} must contain {PLACEHOLDER} exactly once")
    return text


def build(template: str, labels: list[str]) -> str:
    """The prompt for one image: the class names, sorted, one per line."""
    return template.replace(PLACEHOLDER, "\n".join(f"- {label}" for label in sorted(labels))).strip()


def _normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^(final\s+)?(diagnosis|answer)\s*:\s*", "", text)
    text = text.strip(" \t\r\n\"'`*-.")
    return re.sub(r"\s+", " ", text)


def parse(text: str | None, labels: list[str]) -> str | None:
    """The class the answer names, or None when it is not exactly one class name.

    Only formatting is forgiven (case, spaces, quotes, bold marks, a trailing full stop, a leading "Diagnosis:").
    An answer that adds words or names two classes counts as unreadable, which the metrics score as wrong.
    """
    if not text:
        return None
    answer = _normalise(text)
    for label in labels:
        if answer == _normalise(label):
            return label
    return None
