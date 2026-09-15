"""Shared helpers for the Alzheimer's corpus pipeline.

This module provides:
- project paths
- configuration loading
- logging
- UTC timestamps
- CSV reporting
- JSONL I/O
- licensing checks
- downstream Alzheimer's relevance assessment

Important pipeline principle:
Stage 01 PubMed retrieval is query-driven. The Alzheimer's relevance
gate is NOT used to decide PubMed retrieval membership. It is available
for later validation/filtering stages only.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


# ==========================================================================
# Project paths
# ==========================================================================

BASE = Path(__file__).resolve().parents[1]

CONFIG = BASE / "config"
DATA = BASE / "data"
LOGS = BASE / "logs"
METADATA = BASE / "metadata"
REPORTS = BASE / "reports"


# ==========================================================================
# Configuration
# ==========================================================================

def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Accepts either:
    - a filename relative to config/, e.g. 'search_queries.yaml'
    - an explicit Path, e.g. CONFIG / 'search_queries.yaml'
    """
    path = Path(name_or_path)

    if not path.is_absolute():
        path = CONFIG / path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file must contain a top-level mapping: {path}"
        )

    return data


# ==========================================================================
# Logging
# ==========================================================================

def get_logger(
    stage: str,
    logfile: str | Path | None = None,
) -> logging.Logger:
    """Create or return an append-only stage logger.

    If logfile is omitted, '<stage>.log' is used.

    Logs are treated as provenance and are therefore never truncated.
    """
    LOGS.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(stage)

    # Prevent duplicate handlers if get_logger() is called repeatedly.
    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    log.propagate = False

    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)-7s %(name)s %(message)s"
    )

    # Ensure timestamps are UTC.
    formatter.converter = (
        lambda *args: datetime.now(timezone.utc).timetuple()
    )

    if logfile is None:
        logfile = f"{stage}.log"

    logfile = Path(logfile)

    if not logfile.is_absolute():
        logfile = LOGS / logfile

    logfile.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        logfile,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    return log


# ==========================================================================
# Time
# ==========================================================================

def utcnow() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


# ==========================================================================
# CSV reporting
# ==========================================================================

def write_report(
    name_or_path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> Path:
    """Write a CSV report.

    Supports either:
        write_report("retrieval_report.csv", rows, fieldnames)

    or:
        write_report(REPORTS / "retrieval_report.csv", rows, fieldnames)

    If fieldnames are omitted, they are inferred from the first row.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)

    path = Path(name_or_path)

    if not path.is_absolute():
        path = REPORTS / path

    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


# ==========================================================================
# JSONL
# ==========================================================================

def read_jsonl(
    path: Path,
) -> Iterable[dict[str, Any]]:
    """Read newline-delimited JSON records."""
    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        for line in fh:
            line = line.strip()

            if line:
                yield json.loads(line)


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> int:
    """Write records as newline-delimited JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


# ==========================================================================
# Licensing
# ==========================================================================

DISTRIBUTABLE = {
    "CC0",
    "CC-BY",
    "CC-BY-SA",
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "CC-BY-ND",
    "public-domain",
    "us-government-work",
}


def redistribution_allowed(
    license_code: str | None,
) -> bool:
    """Return whether a known licence permits redistribution.

    Unknown or missing licences fail closed.
    """
    if not license_code:
        return False

    normalized = str(license_code).strip().upper()

    return normalized in {
        code.upper()
        for code in DISTRIBUTABLE
    }


# ==========================================================================
# Alzheimer's relevance assessment
# ==========================================================================
#
# IMPORTANT:
#
# This section is NOT part of Stage 01 PubMed retrieval.
#
# Stage 01 membership is determined by the independently executed queries
# in config/search_queries.yaml.
#
# This relevance gate can be used later when validating retrieved records,
# resolving ambiguous records, or auditing corpus quality.
#
# Supporting concepts such as dementia, amyloid, tau, and comparator
# diseases NEVER qualify a record by themselves.
# ==========================================================================


# Surface forms are preserved verbatim in stored text.
# Unicode folding is used ONLY for matching.

PRESERVE_VERBATIM = (
    "Aβ42",
    "Aβ40",
    "Aβ",
    "p-tau181",
    "p-tau217",
    "p-tau231",
    "ARIA-E",
    "ARIA-H",
    "APOE ε4",
)


AD_TERMS = (
    "Alzheimer disease",
    "Alzheimer's disease",
    "Alzheimer’s disease",
    "Alzheimer dementia",
    "Alzheimer's dementia",
    "Alzheimer’s dementia",
    "AD dementia",
    "Alzheimer-type dementia",
    "dementia of the Alzheimer type",
    "senile dementia of the Alzheimer type",
    "Alzheimers disease",
)


PATHOLOGY = (
    "amyloid beta",
    "abeta",
    "amyloid plaques",
    "tau",
    "p-tau",
    "phosphorylated tau",
    "neurofibrillary tangles",
)


COMPARATOR = (
    "vascular dementia",
    "vascular cognitive impairment",
    "lewy body",
    "dementia with lewy bodies",
    "frontotemporal dementia",
    "frontotemporal lobar degeneration",
    "normal pressure hydrocephalus",
    "progressive supranuclear palsy",
    "corticobasal",
)


GENERIC = (
    "dementia",
    "cognitive impairment",
    "cognitive dysfunction",
    "cognitive decline",
    "memory impairment",
    "neurodegeneration",
    "neurocognitive disorder",
)


def fold(text: str) -> str:
    """Normalize text for matching only.

    Stored source text must never be replaced by folded text.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize(
        "NFKD",
        text,
    )

    normalized = "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )

    return normalized.casefold()


def _rx(term: str) -> re.Pattern[str]:
    """Compile a word-bounded matching regex."""
    return re.compile(
        r"(?<!\w)"
        + re.escape(fold(term))
        + r"(?!\w)"
    )


_AD_RX = [
    (term, _rx(term))
    for term in AD_TERMS
]

_PATH_RX = [
    (term, _rx(term))
    for term in PATHOLOGY
]

_COMP_RX = [
    (term, _rx(term))
    for term in COMPARATOR
]

_GEN_RX = [
    (term, _rx(term))
    for term in GENERIC
]


# "AD" is ambiguous.
#
# It is deliberately not sufficient by itself to establish Alzheimer's
# relevance. A long-form Alzheimer's anchor must also appear.

_ABBREV_RX = re.compile(
    r"(?<!\w)AD(?!\w)"
)


@dataclass
class Relevance:
    """Complete relevance decision trace."""

    document_id: str

    ad_relevant: bool

    score: float

    rules_fired: list[str] = field(
        default_factory=list
    )

    exclusion_reason: str | None = None

    anchor_evidence: list[str] = field(
        default_factory=list
    )

    supporting_evidence: list[str] = field(
        default_factory=list
    )

    comparator_evidence: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert decision trace to a dictionary."""
        return asdict(self)


def _hits(
    text: str,
    pairs: list[tuple[str, re.Pattern[str]]],
) -> list[str]:
    """Return matched source terms."""
    folded = fold(text)

    return [
        term
        for term, regex in pairs
        if regex.search(folded)
    ]


def assess_ad_relevance(
    record: dict[str, Any],
    mesh_anchors: set[str] | None = None,
) -> Relevance:
    """Assess whether a record contains Alzheimer's evidence.

    This is a downstream validation/classification helper.

    Supporting concepts such as:
    - dementia
    - cognitive dysfunction
    - amyloid
    - tau
    - comparator diseases

    qualify a record ONLY when an Alzheimer's anchor is also present.

    They never qualify a record on their own.
    """
    if mesh_anchors is None:
        mesh_anchors = {"alzheimer disease"}

    # Normalize supplied MeSH anchors so comparisons are robust.
    normalized_mesh_anchors = {
        fold(str(anchor))
        for anchor in mesh_anchors
    }

    document_id = str(
        record.get("document_id")
        or record.get("pmid")
        or ""
    )

    title = str(
        record.get("title")
        or ""
    )

    abstract = str(
        record.get("abstract")
        or ""
    )

    whole = f"{title}\n{abstract}"

    decision = Relevance(
        document_id=document_id,
        ad_relevant=False,
        score=0.0,
    )

    # ------------------------------------------------------------------
    # MeSH Alzheimer's anchor
    # ------------------------------------------------------------------

    mesh_hits = []

    for mesh_term in (
        record.get("mesh_terms")
        or []
    ):
        # Support values such as:
        #   Alzheimer Disease
        #   *Alzheimer Disease
        #   Alzheimer Disease/diagnosis
        base_term = (
            str(mesh_term)
            .split("/", 1)[0]
            .lstrip("*")
            .strip()
        )

        if fold(base_term) in normalized_mesh_anchors:
            mesh_hits.append(
                str(mesh_term)
            )

    # ------------------------------------------------------------------
    # Text Alzheimer's anchors
    # ------------------------------------------------------------------

    title_hits = _hits(
        title,
        _AD_RX,
    )

    abstract_hits = _hits(
        abstract,
        _AD_RX,
    )

    longform_anchor = bool(
        mesh_hits
        or title_hits
        or abstract_hits
    )

    # ------------------------------------------------------------------
    # Ambiguous abbreviation
    # ------------------------------------------------------------------

    abbreviation_present = bool(
        _ABBREV_RX.search(whole)
    )

    decision.anchor_evidence = (
        [f"mesh:{term}" for term in mesh_hits]
        + [f"title:{term}" for term in title_hits]
        + [f"abstract:{term}" for term in abstract_hits]
    )

    # ------------------------------------------------------------------
    # Supporting concepts
    # ------------------------------------------------------------------

    pathology_hits = _hits(
        whole,
        _PATH_RX,
    )

    comparator_hits = _hits(
        whole,
        _COMP_RX,
    )

    generic_hits = _hits(
        whole,
        _GEN_RX,
    )

    decision.supporting_evidence = [
        f"pathology:{term}"
        for term in pathology_hits
    ]

    decision.comparator_evidence = [
        f"comparator:{term}"
        for term in comparator_hits
    ]

    # ------------------------------------------------------------------
    # No Alzheimer's anchor
    # ------------------------------------------------------------------

    if not longform_anchor:

        if abbreviation_present:
            decision.exclusion_reason = (
                "ambiguous_abbreviation_only"
            )

        elif comparator_hits:
            decision.exclusion_reason = (
                "comparator_disease_no_ad_anchor"
            )

        elif generic_hits:
            decision.exclusion_reason = (
                "generic_dementia_no_ad_anchor"
            )

        else:
            decision.exclusion_reason = (
                "no_ad_evidence"
            )

        return decision

    # ------------------------------------------------------------------
    # Alzheimer's anchor exists
    # ------------------------------------------------------------------

    score = 0.0

    if mesh_hits:
        decision.rules_fired.append(
            "R1_mesh_anchor"
        )
        score = max(score, 1.0)

    if title_hits:
        decision.rules_fired.append(
            "R2_title_anchor"
        )
        score = max(score, 1.0)

    if abstract_hits:
        decision.rules_fired.append(
            "R3_abstract_anchor"
        )
        score = max(score, 0.8)

    if pathology_hits:
        decision.rules_fired.append(
            "R4_pathology_with_anchor"
        )
        score = max(score, 0.7)

    if comparator_hits:
        decision.rules_fired.append(
            "R5_differential_with_anchor"
        )
        score = max(score, 0.7)

    decision.score = round(
        score,
        3,
    )

    decision.ad_relevant = (
        decision.score >= 0.7
    )

    return decision