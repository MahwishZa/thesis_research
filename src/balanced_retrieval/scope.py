"""The default scope of the PubMed and PMC corpora (decision D9, still open): Alzheimer's disease and its neighbours.

The scope is one search string, written here and recorded with every download. It defines the PubMed corpus; the PMC corpus
is the open-access full text of those same records (the ones that carry a PMC identifier), so no second search is needed.
The core is Alzheimer's disease, the other dementias, mild cognitive impairment and amyloid or tau disease. The conditions that
must be told apart from it or that coexist (delirium, depression, normal-pressure hydrocephalus, vitamin B12 and thyroid
causes, stroke) are included only where the record is also about dementia or cognitive decline, because each of them alone
is a field of millions of records.
"""

from __future__ import annotations

CORE_TERMS = (
    '"Alzheimer Disease"[Mesh]',
    '"Dementia"[Mesh]',
    '"Cognitive Dysfunction"[Mesh]',
    '"Lewy Body Disease"[Mesh]',
    '"Frontotemporal Lobar Degeneration"[Mesh]',
    '"tau Proteins"[Mesh]',
    '"Amyloid beta-Peptides"[Mesh]',
    "Alzheimer*[tiab]",
    "dementia[tiab]",
    '"mild cognitive impairment"[tiab]',
    '"Lewy bod*"[tiab]',
    '"frontotemporal dementia"[tiab]',
)

NEIGHBOUR_TERMS = (
    '"Delirium"[Mesh]',
    '"Depression"[Mesh]',
    '"Hydrocephalus, Normal Pressure"[Mesh]',
    '"Vitamin B 12 Deficiency"[Mesh]',
    '"Hypothyroidism"[Mesh]',
    '"Parkinson Disease"[Mesh]',
    '"Stroke"[Mesh]',
    "delirium[tiab]",
    '"normal pressure hydrocephalus"[tiab]',
)

COGNITIVE_TERMS = ('"Dementia"[Mesh]', '"Cognitive Dysfunction"[Mesh]', "dementia[tiab]", "cognitive[tiab]")


def _any(terms: tuple[str, ...]) -> str:
    return "(" + " OR ".join(terms) + ")"


def pubmed_scope() -> str:
    """The search string of the PubMed corpus: the scope, English, with an abstract."""
    return f"({_any(CORE_TERMS)} OR ({_any(NEIGHBOUR_TERMS)} AND {_any(COGNITIVE_TERMS)})) AND English[lang] AND hasabstract"
