"""The prompt and the answer reader: the same for every model, so differences in results come from the models, not from how
their answers were read."""

from __future__ import annotations

import re
from pathlib import Path

from .datasets import DATASETS
from .metrics import ABSTAIN


PLACEHOLDER = "{labels}"
CASE = "{case}"  # in a text prompt: where each case goes
FACTS = "{facts}"  # in a multimodal prompt: where the patient's facts go
LEAD_INS = ("diagnosis", "answer", "final answer")  # the words the answer reader forgives in front of a colon and the class name


# ---------------------------------------------------------------------------------------------------------------------------------
# The prompt and the answer reader: the same for every model, so differences in results come from the models, not from how their
# answers were read


def load_template(path: str | Path, dataset: str = "image", abstain: bool = False) -> str:
    """The prompt template: the label placeholder exactly once; the case placeholder exactly once in a prompt for text cases
    (never in one for images, which go next to the prompt); the facts placeholder exactly once in a multimodal prompt (never in
    another); and, only in a prompt of a run that may abstain, the word "uncertain" (the answer that means no diagnosis)."""
    text = Path(path).read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f"{path} must contain {PLACEHOLDER} exactly once")
    if DATASETS[dataset].in_prompt and text.count(CASE) != 1:
        raise ValueError(f"{path} must contain {CASE} exactly once: it is where each case goes")
    if not DATASETS[dataset].in_prompt and CASE in text:
        raise ValueError(f"{path} must not contain {CASE}: the {dataset} dataset sends no text case")
    if DATASETS[dataset].facts_column and text.count(FACTS) != 1:
        raise ValueError(f"{path} must contain {FACTS} exactly once: it is where each patient's facts go")
    if not DATASETS[dataset].facts_column and FACTS in text:
        raise ValueError(f"{path} must not contain {FACTS}: the {dataset} dataset has no patient facts")
    says = re.search(rf"\b{ABSTAIN}\b", text, re.IGNORECASE) is not None
    if abstain and not says:
        raise ValueError(f"{path} must tell the model it may answer {ABSTAIN!r}: the config says abstain = true")
    if not abstain and says:
        raise ValueError(f"{path} must not offer the answer {ABSTAIN!r}: the config forces a choice (abstain is not true)")
    return text


def build_prompt(template: str, labels: list[str], abstain: bool = False) -> str:
    """The prompt of a run: the class names, sorted, one per line, and for a run that may abstain the answer "uncertain" after
    them."""
    options = [*sorted(labels), *([ABSTAIN] if abstain else [])]
    return template.replace(PLACEHOLDER, "\n".join(f"- {option}" for option in options)).strip()


def fill(prompt: str, placeholder: str, value: str) -> str:
    """The prompt for one question: the value (a text case, or a patient's facts) word for word where the prompt has the
    placeholder. A placeholder-like text inside the value is left as it is."""
    before, after = prompt.split(placeholder)
    return before + value + after


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


def parse_last_line(text: str | None, labels: list[str]) -> str | None:
    """The class named alone on the last line of a longer answer, or None. Used only for the sensitivity analysis of the report:
    the main score never reads an answer this way, because the prompt asks for the diagnosis name only."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return parse_answer(lines[-1], labels) if len(lines) > 1 else None


def can_still_become_readable(text: str | None, labels: list[str]) -> bool:
    """For an answer that was cut off at the output limit: could the rest of it have made it a readable answer?

    True when what is there, after the formatting that parse_answer() forgives, is empty or only the beginning of a class name or
    of a "Diagnosis:"-style lead-in. False when it already differs from every class name (an essay, a second class name): no
    continuation can then make it exactly one class name, so the cut-off did not change how the answer is scored.
    """
    answer = _normalise(text or "")
    return any(start.startswith(answer) for start in [_normalise(label) for label in labels] + list(LEAD_INS))
