"""What the model reads from one PubMed record, and how a record's study design is classified.

``study_snippet`` is the text the realigned study's filter judges for each candidate abstract: the title, then the
RESULTS and CONCLUSIONS sections (an abstract without labelled sections contributes its last three sentences),
at most ``MAX_SNIPPET_WORDS`` words. ``study_type`` assigns a record to one of the three evidence types used for
balanced retrieval and for labelling studies in the verification prompt: systematic review or meta-analysis, trial,
other. Both were first written for the earlier stages and are unchanged.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

MAX_SNIPPET_WORDS = 200
SR_MA_TYPES = ("Meta-Analysis", "Systematic Review")
RCT_TYPES = ("Randomized Controlled Trial", "Controlled Clinical Trial", "Clinical Trial")


_LABEL = re.compile(r"(?:(?<=\s)|^)([A-Z][A-Z0-9 ,'’/&()-]{2,60}?):\s")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
KEY_RESULTS = ("RESULT", "FINDING")
KEY_CONCLUSIONS = ("CONCLUSION", "INTERPRETATION")
#: A capitalised word followed by a colon only counts as a section label if it contains one of
#: these words; "BMI:" or "HR:" inside running text must not cut a section.
BOUNDARY_WORDS = KEY_RESULTS + KEY_CONCLUSIONS + (
    "BACKGROUND", "OBJECTIVE", "AIM", "PURPOSE", "METHOD", "DESIGN", "SETTING", "PARTICIPANT",
    "PATIENT", "SUBJECT", "INTERVENTION", "MEASUREMENT", "OUTCOME", "INTRODUCTION", "CONTEXT",
    "DISCUSSION", "SEARCH", "SELECTION", "DATA", "STUDY", "MATERIAL", "REVIEW", "ELIGIBILITY",
    "LIMITATION", "FUNDING", "REGISTRATION", "TRIAL", "IMPLICATION", "SIGNIFICANCE", "RATIONALE",
    "EXPOSURE", "POPULATION", "SAMPLE", "CRITERIA", "SOURCE", "STRATEGY", "RELEVANCE")


def split_sections(abstract: str) -> list[tuple[str, str]]:
    """[(LABEL, text)] for an abstract whose sections are marked ``LABEL: text`` (PubMed's
    structured abstracts, as stored in the frozen pools); [] when it has no recognised labels."""
    marks = []
    for m in _LABEL.finditer(abstract or ""):
        label = m.group(1).strip()
        if any(word in label for word in BOUNDARY_WORDS):
            marks.append((m.start(1), m.end(), label))
    out = []
    for i, (_, end, label) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(abstract)
        out.append((label, abstract[end:stop].strip()))
    return out


def key_text(abstract: str) -> tuple[str, str]:
    """(results text, conclusions text); an unlabelled abstract gives ("", its last three sentences)."""
    sections = split_sections(abstract)
    results = [t for label, t in sections if any(w in label for w in KEY_RESULTS) and t]
    conclusions = [t for label, t in sections if any(w in label for w in KEY_CONCLUSIONS) and t]
    if results or conclusions:
        return " ".join(results), " ".join(conclusions)
    sentences = [x for x in _SENTENCE.split((abstract or "").strip()) if x]
    return "", " ".join(sentences[-3:])


def study_snippet(candidate: dict, max_words: int = MAX_SNIPPET_WORDS) -> str:
    """Title, then results, then conclusions, at most ``max_words`` words; the conclusions keep at
    least half of what is left after the title."""
    title = (candidate.get("title") or "").strip()
    results, conclusions = key_text(candidate.get("abstract") or "")
    budget = max(20, max_words - len(title.split()))
    r_words, c_words = results.split(), conclusions.split()
    c_keep = min(len(c_words), max(budget // 2, budget - len(r_words)))
    r_keep = min(len(r_words), budget - c_keep)
    parts = [title.rstrip(".")] if title else []
    if r_keep:
        parts.append("Results: " + " ".join(r_words[:r_keep]))
    if c_keep:
        parts.append("Conclusions: " + " ".join(c_words[:c_keep]))
    if not parts:
        return "No abstract available."
    return ". ".join(p.rstrip(".") for p in parts) + "."


def study_type(pubtypes: Optional[Sequence[str]]) -> str:
    """"SR/MA", "RCT" or "other", from PubMed publication types."""
    kinds = set(pubtypes or ())
    if kinds & set(SR_MA_TYPES):
        return "SR/MA"
    if kinds & set(RCT_TYPES):
        return "RCT"
    return "other"
