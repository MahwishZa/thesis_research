"""The fixed specification of the Alzheimer's-specific question set (``docs/protocol.md`` §8, amendment of 2026-10-09 on
templates). Everything here is declared before any question is drafted; the builder and the tests read it from here and a
guard test compares it with the protocol text. No network, no model."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from experiments.medchange.pubmed_asof import query_terms

#: The three question templates. Every fixed word is a stop word of the frozen query builder (``pubmed_asof._STOP``) except
#: "Alzheimer's disease", which closes the question so that the copied terms come first in the 8-term query.
TEMPLATES = {
    "effect": "Is {x} effective for {y} in people with Alzheimer's disease?",
    "association": "Is there any effect of {x} on {y} in people with Alzheimer's disease?",
    "test": "Can {x} be used for {y} in people with Alzheimer's disease?",
}
CONDITION_WORDS = ("alzheimer", "disease")
MAX_COPIED_CONTENT_WORDS = 6          # x and y together, after stop words; leaves room for the two condition words
MAX_TERM_WORDS = 8                    # words in one copied span
VERDICTS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")

SPLIT_SEED = "adkqa-split-v1"
DEV_FRACTION = 0.25                   # share of topic clusters whose records form the development pool
DRAFT_N = 150                         # development records drafted in the trial run, in hash order
DEV_N = 60                            # development questions kept (the first 60 that survive), 40% of DRAFT_N
TEST_MIN, TEST_MAX = 200, 300
SOURCE_ANCHOR = "experiments/adkqa/results/stage0_counts_r2.json"   # the frozen source list (``pool_ids``, 2026-10-09)

#: Cue phrases of the rule-based reading of the conclusion (lower case, regular expressions).
CUES = {
    "negative": [
        r"no (statistically )?significant (effects?|differences?|improvements?|benefits?|associations?|reductions?|changes?)",
        r"not (significantly )?(effective|beneficial|associated|superior|supported|useful|accurate)",
        r"(did|does|do) not (improve|reduce|show|demonstrate|differ|prevent|slow|affect|increase|decrease)",
        r"(no|without) (clear |apparent |definite )?(benefits?|effects?|differences?|associations?|advantages?|improvements?)",
        r"ineffective", r"failed to", r"similar to placebo", r"not recommended",
        r"(poor|low) (diagnostic )?(accuracy|performance|sensitivity|specificity)",
    ],
    "positive": [
        r"significantly (improved?|improves|reduced?|reduces|increased?|increases|enhanced?|slowed?|delayed?|lower|higher|better)",
        r"\b(effective|efficacious|beneficial|benefits?)\b",
        r"associated with (a |an )?(higher|increased|lower|reduced|greater|decreased|elevated|increase|decrease|risk)",
        r"(high|good|excellent|acceptable|promising) (diagnostic )?(accuracy|sensitivity|specificity|performance)",
        r"positive (effects?|associations?)", r"superior to",
    ],
    "insufficient": [
        r"insufficient (evidence|data)", r"(evidence|data) (is|are|remains?) (insufficient|lacking|scarce|inconclusive)",
        r"not enough (evidence|data|studies)", r"inconclusive", r"too few (studies|trials)", r"lack of (evidence|data)",
        r"no (firm|definitive) conclusions?", r"(cannot|can not|could not) be (drawn|made|determined)",
        r"remains? unclear", r"unclear whether", r"no (eligible|relevant) (studies|trials)",
    ],
    "hedge": [r"\bhowever\b", r"\bmixed\b", r"\binconsistent\b"],
}
_LABEL_OF = {"positive": "SUPPORTED", "negative": "REFUTED", "insufficient": "NOT ENOUGH INFORMATION"}
CONCLUSION_SECTIONS = ("conclusion", "conclusions", "interpretation", "authors' conclusions", "author conclusions")
OBJECTIVE_SECTIONS = ("objective", "objectives", "aim", "aims", "purpose", "background")
UNSTRUCTURED_CONCLUSION = re.compile(
    r"^(in conclusion|we conclude|overall|taken together|these (results|findings)|our (results|findings)|"
    r"this (systematic )?(review|meta-analysis|study))", re.IGNORECASE)


def fill(template: str, x: str, y: str) -> str:
    return TEMPLATES[template].format(x=x.strip(), y=y.strip())


def question_checks(template: str, x: str, y: str) -> list[str]:
    """Why a drafted (template, x, y) cannot be a question; empty when it can. Term presence in the abstract is checked by the
    builder, which has the abstract."""
    problems = []
    if template not in TEMPLATES:
        return ["unknown template"]
    for name, span in (("x", x), ("y", y)):
        n = len(span.split())
        if not 1 <= n <= MAX_TERM_WORDS:
            problems.append(f"{name} has {n} words")
    q = fill(template, x, y)
    terms = query_terms(q)
    content = [t for t in terms if t not in CONDITION_WORDS]
    if any(w not in terms for w in CONDITION_WORDS):
        problems.append("the condition words fall outside the first 8 query terms")
    if len([t for t in query_terms(f"{x} {y}", max_terms=99)]) > MAX_COPIED_CONTENT_WORDS:
        problems.append("more than 6 content words copied")
    if not content:
        problems.append("no content word besides the condition")
    return problems


def read_conclusion(text: str) -> Optional[str]:
    """The verdict the conclusion states, or None when it states none or states more than one (the question is then not made).
    Negative phrases are removed first so that "not effective" is not also read as positive."""
    t = " ".join((text or "").lower().split())
    hits = {k: [] for k in CUES}
    rest = t
    for pat in CUES["negative"]:
        if re.search(pat, rest):
            hits["negative"].append(pat)
            rest = re.sub(pat, " ", rest)
    for k in ("insufficient", "positive", "hedge"):
        hits[k] = [p for p in CUES[k] if re.search(p, rest)]
    if hits["hedge"]:
        return None
    classes = [k for k in ("positive", "negative", "insufficient") if hits[k]]
    return _LABEL_OF[classes[0]] if len(classes) == 1 else None


def cluster_key(major_descriptors: Iterable[str], pmid: str) -> str:
    """The topic cluster of a record: its first major-topic MeSH descriptor other than Alzheimer Disease (alphabetical,
    case-insensitive), or the record itself when it has none."""
    others = sorted((d.strip() for d in major_descriptors if d.strip().lower() != "alzheimer disease"), key=str.lower)
    return others[0].lower() if others else f"pmid:{pmid}"


def _unit(*parts: str) -> float:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32


def split_of(cluster: str) -> str:
    return "dev" if _unit(SPLIT_SEED, "cluster", cluster) < DEV_FRACTION else "test"


def draft_order(pmids: Iterable[str]) -> list[str]:
    """Hash order in which development records are drafted (first DRAFT_N)."""
    return sorted(pmids, key=lambda p: _unit(SPLIT_SEED, "draft", p))
