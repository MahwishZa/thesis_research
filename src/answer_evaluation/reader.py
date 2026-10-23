"""Reading the final answer from the text a model wrote, strictly (docs/methodology.md, decision D16).

The paper's case study shows Llama-3 ending its answers "Therefore, the answer is (D) Needle decompression." (Figure 4, p.8).
A generation is readable when it names exactly one option letter as its answer in that form: the words "answer is", then the
letter in round brackets. Anything else is unreadable, and the reasons are kept apart so that the unreadable rate can be looked
at on validation data before any test run:

    none      no "answer is (X)" in the text
    conflict  two different letters, or another bracketed letter in the same sentence ("the answer is (D), not (A)")
    outside   a letter that is not one of the options

An unreadable answer counts as wrong when systems are scored, and the pair is skipped when labels are made. The old code
guessed the last stray A to D in the text and that put wrong labels into the training set; this reader never guesses.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

LETTERS = "ABCD"

FINAL_ANSWER = re.compile(r"(?i:\banswer\s+is)\s*:?\s*\(([A-Z])\)")
_ANOTHER_LETTER = re.compile(r"\([A-Z]\)")
_SENTENCE_END = re.compile(r"\.\s|\n")


def reading(text: str, letters: str = LETTERS) -> tuple[str | None, str]:
    """``(letter, "ok")`` for a readable answer, else ``(None, reason)`` with the reason ``none``, ``conflict`` or ``outside``."""
    if not isinstance(text, str):
        raise TypeError(f"the text must be a string, got {type(text).__name__}")
    found: set[str] = set()
    for match in FINAL_ANSWER.finditer(text):
        end = _SENTENCE_END.search(text, match.end())
        tail = text[match.end() : end.start() if end else len(text)]
        if _ANOTHER_LETTER.search(tail):
            return None, "conflict"
        found.add(match.group(1))
    if not found:
        return None, "none"
    if len(found) > 1:
        return None, "conflict"
    (letter,) = found
    if letter not in letters:
        return None, "outside"
    return letter, "ok"


def read_final_answer(text: str, letters: str = LETTERS) -> str | None:
    """The option letter the text names as its answer, or None if it is unreadable."""
    return reading(text, letters)[0]


def scored_correct(text: str, gold: str, letters: str = LETTERS) -> bool:
    """Correct or not for scoring a system: an unreadable answer is wrong."""
    return read_final_answer(text, letters) == _gold(gold, letters)


def labelling_correct(text: str, gold: str, letters: str = LETTERS) -> bool | None:
    """Correct or not for making a label: an unreadable answer gives None and the pair is skipped."""
    letter = read_final_answer(text, letters)
    return None if letter is None else letter == _gold(gold, letters)


def unreadable_summary(texts: Iterable[str], letters: str = LETTERS) -> dict[str, object]:
    """How many texts are unreadable, the share, and the reasons. Meant for validation data, before any test run."""
    reasons: Counter[str] = Counter()
    total = 0
    for text in texts:
        total += 1
        reasons[reading(text, letters)[1]] += 1
    unreadable = total - reasons["ok"]
    return {
        "texts": total,
        "unreadable": unreadable,
        "share": round(unreadable / total, 4) if total else None,
        "reasons": {name: reasons[name] for name in ("none", "conflict", "outside")},
    }


def _gold(gold: str, letters: str) -> str:
    if gold not in tuple(letters):
        raise ValueError(f"the answer key must be one of {', '.join(letters)}, got {gold!r}")
    return gold
