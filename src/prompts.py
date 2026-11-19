"""The prompt and the answer reader: the same for every model, so differences in results come from the models, not from how
their answers were read.

Answer format (since the protocol of 2026-11-19, docs/method.md): one JSON object and nothing else,
    {"differential": [three different classes, the most likely first], "evidence_sufficient": "yes" or "no",
     "reason": "<why, in at most three sentences>", "evidence": [ids of the retrieved passages used; empty without retrieval]}
read by parse_json_answer. The answers of the first phase were a single class name, read by parse_answer."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .datasets import DATASETS


PLACEHOLDER = "{labels}"
CASE = "{case}"  # in a text prompt: where each case goes
FACTS = "{facts}"  # in an image + metadata prompt: where the patient's facts go
ANSWER_FORMAT = "json-v1"  # the answer format of the runs; a run card without it holds single class names (the first phase)
ANSWER_KEYS = ("differential", "evidence_sufficient", "reason", "evidence")
DIFFERENTIAL = 3  # the number of different classes in a differential (all of them when a dataset has fewer)
FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n\s*```\Z", re.S | re.I)  # a Markdown code block around the whole answer


# ---------------------------------------------------------------------------------------------------------------------------------
# The prompt and the answer reader: the same for every model, so differences in results come from the models, not from how their
# answers were read


def load_template(path: str | Path, dataset: str = "image") -> str:
    """The prompt template: the label placeholder exactly once; the case placeholder exactly once in a prompt for text cases
    (never in one for images, which go next to the prompt); the facts placeholder exactly once in an image + metadata prompt (never in
    another)."""
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
    return text


def build_prompt(template: str, labels: list[str]) -> str:
    """The prompt of a run: the class names, sorted, one per line."""
    return template.replace(PLACEHOLDER, "\n".join(f"- {label}" for label in sorted(labels))).strip()


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


def parse_json_answer(text: str | None, labels: list[str], evidence_ids: set[str] = frozenset()) -> tuple[dict | None, str | None]:
    """The answer of the JSON form, read strictly: (the answer, None), or (None, why it is invalid).

    Forgiven: spaces around the answer, a Markdown code block around it, and the case and spacing of a class name or of yes/no.
    Invalid: anything else around the object, a key missing or added, a differential that is not three different classes of the
    list, a flag that is not yes or no, an empty reason, and evidence that names a passage the run did not retrieve (without
    retrieval, any evidence). The diagnosis is the first class of the differential.
    """
    if not text or not text.strip():
        return None, "empty answer"
    body = text.strip()
    fenced = FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        answer = json.loads(body)
    except json.JSONDecodeError:
        return None, "not one JSON object"
    if not isinstance(answer, dict):
        return None, "not one JSON object"
    if set(answer) != set(ANSWER_KEYS):
        missing, extra = sorted(set(ANSWER_KEYS) - set(answer)), sorted(set(answer) - set(ANSWER_KEYS))
        return None, "; ".join([*(f"missing key {k}" for k in missing), *(f"key not asked for: {k}" for k in extra)])
    by_name = {_normalise(label): label for label in labels}
    size = min(DIFFERENTIAL, len(labels))
    differential = answer["differential"]
    if not isinstance(differential, list) or len(differential) != size or not all(isinstance(d, str) for d in differential):
        return None, f"the differential is not a list of {size} class names"
    named = [by_name.get(re.sub(r"\s+", " ", d.strip().lower())) for d in differential]
    if None in named:
        return None, f"not a class of the list: {differential[named.index(None)]!r}"
    if len(set(named)) != size:
        return None, "the differential names a class twice"
    flag = answer["evidence_sufficient"]
    if not isinstance(flag, str) or flag.strip().lower() not in ("yes", "no"):
        return None, "evidence_sufficient is not yes or no"
    reason = answer["reason"]
    if not isinstance(reason, str) or not reason.strip():
        return None, "no reason given"
    evidence = answer["evidence"]
    if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
        return None, "evidence is not a list of passage ids"
    unknown = [e for e in evidence if e not in evidence_ids]
    if unknown:
        return None, f"evidence names a passage the run did not give: {unknown[0]!r}"
    return {"diagnosis": named[0], "differential": named, "evidence_sufficient": flag.strip().lower(), "reason": reason.strip(),
            "evidence": evidence}, None
