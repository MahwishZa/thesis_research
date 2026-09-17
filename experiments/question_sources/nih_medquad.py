"""Candidate questions from NIH consumer-health pages, via MedQuAD.

Source of record: MedQuAD (Ben Abacha & Demner-Fushman, *A Question-Entailment
Approach to Question Answering*, BMC Bioinformatics 20(1):511, 2019),
<https://github.com/abachaa/MedQuAD>, released under **CC BY 4.0** (licence
file present and checked).

Why this source, alongside Cochrane. It is a different source category -
government and public-health material rather than peer-reviewed synthesis - so
the pool does not rest on one provider. Each document carries the originating
NIH site, the page URL and a UMLS CUI, which gives an ontology-backed
Alzheimer's criterion rather than a string match.

Two limitations are recorded rather than worked around:

* MedQuAD carries **no dates**. Items from here have no `reference_date` from
  the source, so the adapter records the page as undated and the caller must
  supply a retrieval date. Undated items are weaker evidence and reviewers
  should prefer Cochrane items where both exist.
* Answers for three subsets were stripped upstream for MedlinePlus copyright.
  Documents with empty answers are skipped, not reconstructed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from ..evaluation.questions import EvaluationQuestion
from .cochrane import SourceUnavailable, bottom_line

#: UMLS Concept Unique Identifier for Alzheimer's Disease. An ontology code is
#: a stronger anchor than a keyword: it is assigned externally and does not
#: match a passing mention.
AD_CUI = "C0002395"

#: Question types whose answers state a checkable fact. "support groups",
#: "research" and similar are dropped: they describe services, not findings.
_USABLE_QTYPES = {
    "information", "causes", "symptoms", "treatment", "prevention",
    "exams and tests", "outlook", "frequency", "genetic changes",
    "inheritance", "susceptibility", "stages", "complications",
}

_QTYPE_TOPIC = {
    "treatment": ("treatment", "pharmacological"),
    "prevention": ("prevention", "risk_reduction"),
    "exams and tests": ("diagnosis", "diagnostic_workup"),
    "causes": ("mechanism", "aetiology"),
    "genetic changes": ("mechanism", "genetics"),
    "inheritance": ("mechanism", "genetics"),
    "susceptibility": ("epidemiology", "risk_and_frequency"),
    "frequency": ("epidemiology", "risk_and_frequency"),
    "symptoms": ("disease_course", "symptoms"),
    "stages": ("disease_course", "progression"),
    "outlook": ("disease_course", "progression"),
    "complications": ("disease_course", "progression"),
    "information": ("disease_characteristics", "definition"),
}


def candidates(
    root: Path,
    *,
    retrieved_on: str,
) -> Iterator[EvaluationQuestion]:
    """Emit AD candidates from a local MedQuAD checkout.

    ``retrieved_on`` (YYYY-MM) is recorded as the reference date because the
    source supplies none. It is the date the page content was obtained, not a
    publication date, and the metadata says so explicitly so the distinction
    survives into the review file.
    """
    root = Path(root)
    if not root.exists():
        raise SourceUnavailable(
            f"MedQuAD checkout not found: {root}\n"
            "Clone https://github.com/abachaa/MedQuAD (CC BY 4.0)."
        )

    for path in sorted(root.glob("*/*.xml")):
        try:
            document = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        cuis = [c.text for c in document.findall(".//CUI") if c.text]
        focus = (document.findtext("Focus") or "").strip()
        if AD_CUI not in cuis:
            continue
        # "Alzheimer - resources" and similar are navigation pages.
        if "resource" in focus.lower():
            continue

        collection = path.parent.name
        source_name = document.get("source") or collection
        url = document.get("url") or ""

        for pair in document.findall(".//QAPair"):
            question_el, answer_el = pair.find("Question"), pair.find("Answer")
            if question_el is None or answer_el is None:
                continue
            qtype = (question_el.get("qtype") or "").strip().lower()
            if qtype not in _USABLE_QTYPES:
                continue
            question = " ".join((question_el.text or "").split())
            answer = bottom_line(answer_el.text or "")
            if not question or not answer:
                continue
            # Caregiving pages carry the AD CUI but ask about care logistics.
            if "caregiv" in focus.lower() or "caregiver" in question.lower():
                topic, subtopic = ("management", "caregiving")
            else:
                topic, subtopic = _QTYPE_TOPIC.get(
                    qtype, ("general", "unclassified"))

            yield EvaluationQuestion(
                question=question,
                topic=topic,
                subtopic=subtopic,
                reference_answer=answer,
                reference_source=f"NIH {source_name} (via MedQuAD, CC BY 4.0)",
                reference_locator=(
                    f"MedQuAD:{collection}/{path.stem} | "
                    f"qid:{question_el.get('qid')} | CUI:{AD_CUI} | {url}"
                ),
                reference_date=retrieved_on,
                AD_anchor=True,
                determinate=True,
                corpus_support_expected=True,
                temporal_candidate=False,
                ambiguity_candidate=(qtype == "information"),
                status="candidate",
                metadata={
                    "reference_source_type": "government_public_health",
                    "source_dataset": "MedQuAD (Ben Abacha & Demner-Fushman, 2019)",
                    "source_qtype": qtype,
                    "focus": focus,
                    "reference_answer_provenance":
                        "verbatim_leading_sentence_of_nih_page_section",
                    "reference_date_is_retrieval_date": True,
                    "verification_required": True,
                },
            )
