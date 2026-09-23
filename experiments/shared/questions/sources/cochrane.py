"""Candidate questions from Cochrane systematic reviews.

Source of record: the MedRevQA release (Vladika, Dhaini & Matthes, *Facts Fade
Fast*, Findings of EMNLP 2025), <https://github.com/jvladika/MedChange>. Every
row is one Cochrane Database of Systematic Reviews record and carries the
review's own question, its structured abstract, a PMID, and the Cochrane
citation with its CD number and date.

Why this source. A Cochrane review is a peer-reviewed evidence synthesis with
an identifiable author group, a publication date, a stable citation and an
explicit bottom-line conclusion. The question and the verdict both come from
the review; this module only reformats them. That is what keeps the chain
independent: nothing here originates with an LLM or with the thesis.

**The reference answer is a short verbatim extract of the review's own
conclusion**, kept to one sentence and always paired with its PMID. It is
emitted as a *candidate* and must be verified by a human against the cited
record before the question can be approved. The adapter never paraphrases,
because a paraphrase is an interpretation the citation would no longer support.

Licence note: MedRevQA publishes no licence file and the underlying text is
Cochrane Library abstract content. The source file is therefore not vendored,
and only a short attributed extract is retained per item.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from src.evaluation.questions import EvaluationQuestion

#: Applied to the review's own question and objectives, then to the whole
#: record. Mirrors the corpus rule: related dementias qualify a record only
#: when an Alzheimer's anchor is present somewhere in it.
AD_ANCHOR = re.compile(r"alzheimer", re.I)
AD_RELATED = re.compile(
    r"alzheimer|dementia|cognitive impairment|cognitive decline|mild cognitive",
    re.I,
)

_CD = re.compile(r"(CD\d+)(?:\.(pub\d+))?", re.I)
_YEAR = re.compile(r"Syst Rev\.\s*(\d{4})")
_MONTH = re.compile(r"Syst Rev\.\s*\d{4}\s+([A-Z][a-z]{2})")
_MONTHS = {m: f"{i:02d}" for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1)}

#: Keyword -> (topic, subtopic). First match wins, so order matters: the more
#: specific patterns are listed before the general ones.
_TOPICS: Sequence[tuple[str, tuple[str, str]]] = (
    (r"\bbiomarker|amyloid|abeta|a-beta|tau\b|csf|pet\b|neuroimag",
     ("diagnosis", "biomarkers")),
    (r"accura|detect|screen|diagnos|sensitiv|specific",
     ("diagnosis", "diagnostic_accuracy")),
    (r"\bgene|genetic|apoe|heritab|mutation",
     ("mechanism", "genetics")),
    (r"prevent|risk reduction|prophylax",
     ("prevention", "risk_reduction")),
    (r"risk factor|associated with|incidence|prevalence|epidemiolog",
     ("epidemiology", "risk_and_frequency")),
    (r"caregiv|carer|respite|home care|famil",
     ("management", "caregiving")),
    (r"agitation|behaviour|behavior|psychosis|depress|sleep|apath|neuropsychiatric",
     ("management", "neuropsychiatric_symptoms")),
    (r"exercis|cognitive train|stimulat|therapy|rehabilit|music|reminiscen|aromather",
     ("treatment", "non_pharmacological")),
    (r"drug|inhibitor|donepezil|memantine|rivastigmine|galantamine|"
     r"supplement|vitamin|statin|antipsychotic|antidepress|treat",
     ("treatment", "pharmacological")),
    (r"progress|prognos|surviv|mortalit|outcome",
     ("disease_course", "progression")),
)


class SourceUnavailable(FileNotFoundError):
    """Raised when the source file has not been fetched."""


def _classify(text: str) -> tuple[str, str]:
    lowered = text.lower()
    for pattern, labels in _TOPICS:
        if re.search(pattern, lowered):
            return labels
    return ("general", "unclassified")


def _reference_date(citation: str) -> Optional[str]:
    year = _YEAR.search(citation or "")
    if not year:
        return None
    month = _MONTH.search(citation or "")
    if month and month.group(1) in _MONTHS:
        return f"{year.group(1)}-{_MONTHS[month.group(1)]}"
    return year.group(1)


def _locator(citation: str, pmid: str) -> str:
    """A concrete locator: PMID plus the Cochrane review id and version."""
    parts = []
    clean = (pmid or "").strip().strip("/")
    if clean:
        parts.append(f"PMID:{clean}")
    cd = _CD.search(citation or "")
    if cd:
        pub = _PUB.search(citation or "")
        suffix = f".pub{pub.group(1)}" if pub else ""
        parts.append(cd.group(1).upper() + suffix)
        parts.append(f"doi:10.1002/14651858.{cd.group(1).upper()}{suffix}")
    return " | ".join(parts)


#: Openers that describe the review's methods rather than its finding. The
#: first sentence of a Cochrane conclusion is often "We included N studies...",
#: which cites correctly but does not answer the question, so it is skipped.
_PREAMBLE = re.compile(
    r"^(this (review|update)|we (included|identified|found \d)|the (review|"
    r"search) (included|identified)|\d+ (studies|trials|rcts) (were |was )?"
    r"(included|identified)|no (studies|trials|rcts) (were |was )?(found|"
    r"included|identified))",
    re.I,
)


def _sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if s]


def bottom_line(text: str, *, limit: int = 240) -> str:
    """The review's own finding, as one verbatim sentence.

    Skips methodological preamble and returns the first sentence that states
    something about the question. Returns "" when the conclusion is entirely
    methodological, which drops the candidate rather than passing on a
    reference answer that does not answer anything.
    """
    for sentence in _sentences(text):
        if _PREAMBLE.match(sentence):
            continue
        # Some source pages open a section by restating its heading. A
        # question is never an answer.
        if sentence.rstrip().endswith("?"):
            continue
        if len(sentence) < 25:
            continue
        if len(sentence) > limit:
            sentence = sentence[:limit].rsplit(" ", 1)[0] + "..."
        return sentence
    return ""


#: The version suffix is attached to the CD number inside the DOI, not to the
#: first CD occurrence in the citation, so it is matched on its own.
_PUB = re.compile(r"\.pub(\d+)", re.I)


def _is_update(citation: str) -> bool:
    match = _PUB.search(citation or "")
    return bool(match) and int(match.group(1)) >= 2

def read_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        raise SourceUnavailable(
            f"MedRevQA source not found: {path}\n"
            "It is not vendored (no licence file upstream; Cochrane abstract "
            "text). Fetch Datasets/MedRevQA.csv from "
            "https://github.com/jvladika/MedChange and record the commit."
        )
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidates(
    path: Path,
    *,
    require_ad_anchor: bool = True,
) -> Iterator[EvaluationQuestion]:
    """Emit one candidate per AD-anchored Cochrane review.

    ``require_ad_anchor`` keeps only records naming Alzheimer's somewhere,
    which is the corpus's own rule: a dementia record qualifies when an
    Alzheimer's anchor is present, never on the dementia term alone.
    """
    for row in read_rows(path):
        question = " ".join((row.get("Question") or "").split())
        conclusions = row.get("conclusions") or ""
        citation = row.get("DOI_Date") or ""
        if not question or not conclusions:
            continue

        whole = " ".join(
            (row.get(k) or "") for k in
            ("Question", "objectives", "background", "conclusions")
        )
        anchored = bool(AD_ANCHOR.search(whole))
        if require_ad_anchor and not anchored:
            continue
        if not AD_RELATED.search(f"{question} {row.get('objectives', '')}"):
            continue

        date = _reference_date(citation)
        if not date:
            continue

        answer = bottom_line(conclusions)
        if not answer:
            continue

        topic, subtopic = _classify(f"{question} {row.get('objectives', '')}")
        label = (row.get("Label") or "").strip()

        yield EvaluationQuestion(
            question=question,
            topic=topic,
            subtopic=subtopic,
            reference_answer=answer,
            reference_source=f"Cochrane Database of Systematic Reviews; {citation.strip()}",
            reference_locator=_locator(citation, row.get("PMID", "")),
            reference_date=date,
            AD_anchor=anchored,
            # The review reaches an explicit verdict, which is what makes the
            # question determinate. "NOT ENOUGH INFORMATION" is still a
            # determinate finding about the evidence base.
            determinate=bool(label),
            corpus_support_expected=True,
            # A ".pub2" or higher suffix means Cochrane republished this
            # review, so its conclusion has been revisited at least once.
            # That makes the item a temporal candidate for error analysis; it
            # does not assert that the verdict changed.
            temporal_candidate=_is_update(citation),
            ambiguity_candidate=(label == "NOT ENOUGH INFORMATION"),
            status="candidate",
            metadata={
                "reference_source_type": "peer_reviewed_evidence_synthesis",
                "source_dataset": "MedRevQA (Vladika et al., EMNLP 2025 Findings)",
                "source_verdict_label": label,
                "reference_answer_provenance":
                    "verbatim_first_sentence_of_review_conclusions",
                "verification_required": True,
            },
        )
