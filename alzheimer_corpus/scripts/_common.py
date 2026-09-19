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
import os
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

    The directory is ``LOGS`` unless ``ALZHEIMER_CORPUS_LOGS`` is set, which
    exists for one reason: a test that calls a stage's ``main()``
    **in-process** against a temp corpus resolves this path from the loaded
    module's own ``__file__`` and therefore appends fixture-scale lines to
    the real, git-tracked logs under ``alzheimer_corpus/logs/`` - which are
    the corpus's provenance record, cited by
    ``docs/status_and_decisions.md`` §2. A real pipeline run never sets the
    variable and is byte-for-byte unaffected.
    """
    log_dir = Path(os.environ.get("ALZHEIMER_CORPUS_LOGS") or LOGS)
    log_dir.mkdir(parents=True, exist_ok=True)

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
        logfile = log_dir / logfile

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
    *,
    mode: str = "w",
) -> int:
    """Write records as newline-delimited JSON.

    ``rows`` is only ever iterated, never materialised here - passing a
    generator instead of a list keeps this function's own memory bounded to
    one record at a time regardless of what the caller does upstream.

    ``mode="a"`` appends instead of truncating, for a caller implementing
    its own resume-after-interruption logic (Stage 06). The default stays
    "w" so every existing caller's behaviour - always start from a fresh
    file - is unchanged.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with path.open(
        mode,
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
            fh.flush()

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

    PMC's own ``license_code`` field (read by ``iter_pmc_records``) writes
    Creative Commons codes space-delimited - "CC BY", "CC BY-NC" - not
    hyphenated like DISTRIBUTABLE's "CC-BY", "CC-BY-NC". Matching hyphens
    literally would fail every real PMC record regardless of licence,
    silently marking fully open content as restricted. Spaces are folded to
    hyphens before comparison so both spellings of the same licence match;
    the set of licence families this treats as distributable is unchanged.
    """
    if not license_code:
        return False

    normalized = str(license_code).strip().upper().replace(" ", "-")

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

# ==========================================================================
# PMC JATS XML extraction
# ==========================================================================
#
# Stage 02 downloads full-text JATS XML and a manifest (metadata/pmc.csv)
# carrying pmid/pmcid/doi/title/license/retraction status - but no article
# text. This section is the bridge: it reads the finalized manifest, parses
# each article's XML, and yields records in the shape Stage 04 normalizes
# and every later stage consumes.
#
# xml.etree.ElementTree (stdlib) is used deliberately rather than lxml or
# BeautifulSoup: PMC's Open Access XML is well-formed by construction (it is
# generated, not scraped), a new dependency buys nothing a trusted, already
# MD5-verified file needs, and the project avoids dependencies it does not
# need. Namespaces are stripped rather than declared, because PMC JATS files
# are not consistent about declaring one.

import xml.etree.ElementTree as ET


class CorpusPipelineError(RuntimeError):
    """Raised when pipeline input cannot be processed safely.

    Distinct from a per-record parse failure (logged and skipped so one bad
    file does not stop a 100000-document batch): this is for conditions that
    make the whole run untrustworthy, such as a missing manifest.
    """


# Fields the finalized PMC manifest is expected to carry (`02_pmc_download.py
# --finalize`'s write_manifest fieldnames). Read here, not re-declared, so
# the two files cannot drift silently the way the log-evidence reader once
# did (ledger D-41).
PMC_MANIFEST_FIELDS = (
    "pmid", "pmcid", "version", "doi", "title", "citation",
    "is_pmc_openaccess", "is_manuscript", "license_code", "is_retracted",
    "json_path", "xml_path", "status",
)

#: PMC's manifest carries no publication-date field (verified against
#: 02_pmc_download.py: no key named 'date' or similar is ever read from the
#: article-version JSON). The date has to come from the XML's <pub-date>.
#: Preference order when more than one <pub-date> is present, most to least
#: authoritative for "when this version became available":
PUB_DATE_TYPE_PRIORITY = ("epub", "pub", "ppub", "collection")


def _local_tag(element: ET.Element) -> str:
    """An element's tag without a namespace prefix, if any."""
    tag = element.tag
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _find_local(root: ET.Element, *path: str) -> ET.Element | None:
    """Walk a path of local (namespace-stripped) tag names from root.

    Only descends through the first matching child at each step - JATS does
    not repeat structural elements like article-meta, so this is unambiguous
    for the paths this module uses.
    """
    node = root
    for name in path:
        found = None
        for child in node:
            if _local_tag(child) == name:
                found = child
                break
        if found is None:
            return None
        node = found
    return node


def _iter_local(root: ET.Element, name: str):
    for child in root:
        if _local_tag(child) == name:
            yield child


def _text_content(element: ET.Element | None) -> str:
    """All text inside an element, tags stripped, whitespace collapsed.

    JATS marks up inline formatting (<italic>, <sub>, <sup>, cross-refs)
    inside prose; ``itertext()`` walks past all of it and concatenates the
    text nodes, which is what normalization needs. Surface forms such as
    Aβ42 survive because they are Unicode text content, not markup.
    """
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _paragraph_text(container: ET.Element | None) -> str:
    """Join every <p> under a container with blank lines between them."""
    if container is None:
        return ""
    paragraphs = [
        _text_content(p) for p in container.iter()
        if _local_tag(p) == "p"
    ]
    return "\n\n".join(p for p in paragraphs if p)


def _parse_pub_date(article_meta: ET.Element) -> tuple[str, str]:
    """Extract the best available publication date.

    Returns ``(date, precision)`` where date is ``YYYY``, ``YYYY-MM`` or
    ``YYYY-MM-DD`` and precision is ``"day"``, ``"month"``, ``"year"`` or
    ``""`` if no date was found. Several <pub-date> elements can be present
    (epub, print, collection); PUB_DATE_TYPE_PRIORITY picks one deterministically
    rather than taking "whichever XML lists first".
    """
    candidates: dict[str, ET.Element] = {}
    for pub_date in _iter_local(article_meta, "pub-date"):
        kind = (
            pub_date.get("pub-type")
            or pub_date.get("date-type")
            or "unspecified"
        ).lower()
        candidates.setdefault(kind, pub_date)

    chosen = None
    for kind in PUB_DATE_TYPE_PRIORITY:
        if kind in candidates:
            chosen = candidates[kind]
            break
    if chosen is None and candidates:
        chosen = next(iter(candidates.values()))
    if chosen is None:
        return "", ""

    def _num(tag: str) -> str:
        node = next(
            (c for c in chosen if _local_tag(c) == tag), None)
        text = _text_content(node)
        return text if text.isdigit() else ""

    year, month, day = _num("year"), _num("month"), _num("day")
    if not year:
        return "", ""
    if month and day:
        return f"{year}-{int(month):02d}-{int(day):02d}", "day"
    if month:
        return f"{year}-{int(month):02d}", "month"
    return year, "year"


@dataclass(frozen=True)
class ParsedArticle:
    """One article's extracted content, ready to merge with manifest fields."""

    title: str
    abstract: str
    sections: tuple[dict[str, str], ...]
    publication_date: str
    date_precision: str


def parse_jats_xml(xml_bytes: bytes) -> ParsedArticle:
    """Parse one PMC JATS article into title, abstract and body sections.

    Raises ``CorpusPipelineError`` on structurally invalid XML or a missing
    ``<article-meta>`` - both mean the file is not the article it claims to
    be, which the caller must not silently skip past into a claimed record
    with fabricated content.

    Reference elements (``<back>``) are deliberately excluded: citation
    lists are not evidence text and chunking them would manufacture passages
    with no scientific content.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise CorpusPipelineError(f"invalid XML: {exc}") from exc

    front = _find_local(root, "front")
    article_meta = _find_local(front, "article-meta") if front is not None else None
    if article_meta is None:
        raise CorpusPipelineError(
            "no <front><article-meta> found - not a recognisable JATS article"
        )

    title_group = _find_local(article_meta, "title-group")
    title = _text_content(
        _find_local(title_group, "article-title") if title_group is not None
        else None
    )

    abstract = _paragraph_text(_find_local(article_meta, "abstract"))

    publication_date, date_precision = _parse_pub_date(article_meta)

    sections: list[dict[str, str]] = []
    body = _find_local(root, "body")
    if body is not None:
        for sec in _iter_local(body, "sec"):
            heading = _text_content(_find_local(sec, "title"))
            text = _paragraph_text(sec)
            if text:
                sections.append({
                    "section": heading or "body",
                    "text": text,
                })
        if not sections:
            # No <sec> wrapping (short articles, editorials) - the body's
            # own paragraphs are the content.
            direct_text = _paragraph_text(body)
            if direct_text:
                sections.append({"section": "body", "text": direct_text})

    return ParsedArticle(
        title=title,
        abstract=abstract,
        sections=tuple(sections),
        publication_date=publication_date,
        date_precision=date_precision,
    )


def read_pmc_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read the finalized PMC manifest, verifying its schema first.

    Refuses a manifest whose columns do not match what this module expects
    to read, rather than silently reading None/empty values from mismatched
    columns - the exact failure class ledger D-41 diagnosed.
    """
    if not manifest_path.exists():
        raise CorpusPipelineError(
            f"PMC manifest not found: {manifest_path}. Run "
            "'02_pmc_download.py --finalize' first."
        )
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != PMC_MANIFEST_FIELDS:
            raise CorpusPipelineError(
                f"{manifest_path} has header {header}, expected "
                f"{PMC_MANIFEST_FIELDS}. The manifest schema has changed "
                "since this reader was written; update PMC_MANIFEST_FIELDS "
                "and the field mapping together, deliberately."
            )
        return list(reader)


def iter_pmc_records(
    manifest_path: Path,
    corpus_root: Path,
    log,
) -> Iterable[dict[str, Any]]:
    """Yield one normalize-ready record per verified PMC article version.

    A record that fails to parse is logged and skipped, not fabricated and
    not silently dropped without a trace: the skip count is the caller's to
    report. ``already_verified`` is the only status read - ``
    unavailable_current_dataset`` rows carry no local files to parse.
    """
    rows = read_pmc_manifest(manifest_path)
    verified = [r for r in rows if r["status"] == "already_verified"]
    parsed = failed = 0

    for row in verified:
        xml_path = corpus_root / row["xml_path"]
        try:
            xml_bytes = xml_path.read_bytes()
            article = parse_jats_xml(xml_bytes)
        except (OSError, CorpusPipelineError) as exc:
            failed += 1
            log.warning(
                "%s.%s: could not parse %s: %s",
                row["pmcid"], row["version"], xml_path, exc,
            )
            continue

        title = article.title or row["title"]
        sections = [{"section": "title", "text": title}]
        if article.abstract:
            sections.append({"section": "abstract", "text": article.abstract})
        sections.extend(dict(s) for s in article.sections)

        parsed += 1
        yield {
            "document_id": f"{row['pmcid']}.{row['version']}",
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "doi": row["doi"],
            "title": title,
            "abstract": article.abstract,
            "sections": sections,
            "mesh_terms": [],
            "publication_date": article.publication_date,
            "date_precision": article.date_precision,
            "license": row["license_code"],
            "source_tier": "peer_reviewed_primary",
            "retracted": row["is_retracted"].strip().lower() == "true",
        }

    log.info(
        "PMC extraction | verified=%d | parsed=%d | failed=%d",
        len(verified), parsed, failed,
    )


# ==========================================================================
# Official documents: clinical guidelines and textbooks
# ==========================================================================
#
# Guidelines and textbooks are not available through a retrieval API the way
# PubMed and PMC are (README, Stage 03's own docstring: "never scraped...
# acquisition of restricted documents is a manual step by design"). A human
# adds one row per document to metadata/guidelines.csv or
# metadata/textbooks.csv with a real source_url and a licence verified for
# THAT document - this module never invents either. What was missing before
# this section existed was everything downstream of that row: downloading
# the file, extracting its text, and feeding it into the same normalized
# shape PMC records already use. That is what these functions do.
#
# Scope: PDF only. A guideline or textbook is a specific, deliberately
# chosen official artifact - not a scraped web page - and PDF is what
# official guidance is actually published as. Extracting an HTML page's
# *article* text from its navigation, footers and boilerplate reliably needs
# either a hand-verified per-site scraper (which does not generalise) or a
# much heavier dependency than this project uses anywhere else; PDF avoids
# the problem entirely because pypdf reads exactly the document's own pages.

from io import BytesIO
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import time as _time

GUIDELINE_REGISTRY_FIELDS = (
    "document_id", "organization", "title", "version", "publication_date",
    "last_updated", "retrieval_date", "source_url", "document_type", "topic",
    "supersedes", "superseded_by", "license", "license_url", "reuse_status",
    "redistribution_allowed", "local_file", "notes",
)

TEXTBOOK_REGISTRY_FIELDS = (
    "document_id", "title", "edition", "authors", "publisher", "year",
    "isbn", "access_method", "source_url", "local_file", "license",
    "permission", "processing_status", "notes",
)

#: document_type / edition values map onto the taxonomy's evidence_levels
#: vocabulary (config/claim_taxonomy.yaml) so Stage 07's evidence-level
#: tagging can eventually recognise these without guessing from prose.
#: Unrecognised values fall back to the registry's own name, not a fabricated
#: default - see _guess_source_tier.
_GUIDELINE_TYPE_TIER = {
    "clinical_guideline": "clinical_guideline",
    "consensus_statement": "consensus_statement",
    "regulatory_document": "regulatory_document",
    "professional_society_statement": "professional_society_statement",
}

DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_RETRY_DELAY = 2.0


def _guess_source_tier(row: dict[str, str], *, default: str) -> str:
    value = (row.get("document_type") or row.get("access_method") or "").strip().lower()
    return _GUIDELINE_TYPE_TIER.get(value, value or default)


def read_document_registry(
    path: Path,
    expected_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    """Read a guideline/textbook registry, verifying its schema first.

    Same discipline as read_pmc_manifest, generalised: a registry whose
    header does not match what the caller expects is refused rather than
    silently read with the wrong columns.
    """
    if not path.exists():
        raise CorpusPipelineError(f"registry not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != expected_fields:
            raise CorpusPipelineError(
                f"{path} has header {header}, expected {expected_fields}. "
                "The registry schema has changed since this reader was "
                "written; update the field list and the mapping together, "
                "deliberately."
            )
        return list(reader)


def write_document_registry(
    path: Path,
    expected_fields: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    """Write a guideline/textbook registry back, atomically.

    Same temp-file-then-replace pattern as 02_pmc_download.py's
    write_manifest: a crash mid-write must never leave a truncated registry
    in place of a good one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def download_document(url: str, destination: Path, log) -> bytes:
    """Download one document, verifying it is actually a PDF.

    Refuses non-HTTPS URLs and non-PDF content rather than saving whatever
    came back: a redirect to an HTML login/paywall page must not be silently
    stored as if it were the document.
    """
    if not url.lower().startswith("https://"):
        raise CorpusPipelineError(f"refusing a non-HTTPS source_url: {url}")

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            request = Request(url, headers={
                "User-Agent": "AlzheimerCorpus-MS-Thesis/1.0"})
            with urlopen(request, timeout=120) as response:
                data = response.read()
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == DOWNLOAD_MAX_RETRIES:
                raise CorpusPipelineError(
                    f"could not download {url}: {exc}") from exc
            _time.sleep(DOWNLOAD_RETRY_DELAY * (2 ** (attempt - 1)))
    else:  # pragma: no cover - loop always breaks or raises
        raise CorpusPipelineError(f"could not download {url}: {last_error}")

    if not data.startswith(b"%PDF-"):
        raise CorpusPipelineError(
            f"{url} did not return a PDF (no %PDF- signature - likely an "
            "HTML error, login or paywall page)"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(destination)
    log.info("downloaded %s -> %s (%d bytes)", url, destination, len(data))
    return data


def extract_pdf_text(pdf_bytes: bytes) -> tuple[dict[str, str], ...]:
    """Extract per-page text from a PDF.

    Raises CorpusPipelineError on an unreadable, corrupt or encrypted PDF -
    the caller must skip and log, never fabricate content for a document it
    could not actually read. Page-level granularity, not full
    structure-aware section detection: PDF carries no reliable heading
    markup the way JATS XML does, and guessing headings from font size
    heuristics is exactly the kind of "looks structure-aware, isn't
    verified" gap this project avoids elsewhere.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except (PdfReadError, ValueError) as exc:
        raise CorpusPipelineError(f"unreadable PDF: {exc}") from exc

    if reader.is_encrypted:
        raise CorpusPipelineError(
            "PDF is encrypted/password-protected and cannot be read")

    sections = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf can raise a range of parser errors
            raise CorpusPipelineError(
                f"failed to extract text from page {index}: {exc}") from exc
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            sections.append({"section": f"page {index}", "text": text})

    if not sections:
        raise CorpusPipelineError(
            "PDF produced no extractable text (likely a scanned image "
            "without OCR - not supported)"
        )
    return tuple(sections)


def iter_official_documents(
    registry_path: Path,
    expected_fields: tuple[str, ...],
    corpus_root: Path,
    log,
    *,
    default_source_tier: str,
) -> Iterable[dict[str, Any]]:
    """Yield normalize-ready records from a downloaded guideline/textbook
    registry.

    Mirrors iter_pmc_records' contract: a row that fails is logged and
    skipped, never fabricated. Two conditions exclude a row from the
    corpus without that being a failure - both expected, both counted
    separately in the log line:

      - no local_file yet (row is curated but not downloaded - run
        '03_guidelines.py --download' first)
      - license does not permit redistribution ("Restricted guidance is
        recorded as metadata only; its text never enters the distributable
        corpus" - 03_guidelines.py's own docstring)
    """
    rows = read_document_registry(registry_path, expected_fields)
    not_downloaded = restricted = failed = parsed = 0

    for row in rows:
        local_file = (row.get("local_file") or "").strip()
        if not local_file:
            not_downloaded += 1
            continue
        if not redistribution_allowed(row.get("license")):
            restricted += 1
            log.info(
                "%s excluded: licence %r does not permit redistribution",
                row.get("document_id", "?"), row.get("license", ""),
            )
            continue

        pdf_path = corpus_root / local_file
        try:
            pdf_bytes = pdf_path.read_bytes()
            sections = extract_pdf_text(pdf_bytes)
        except (OSError, CorpusPipelineError) as exc:
            failed += 1
            log.warning("%s: could not extract %s: %s",
                       row.get("document_id", "?"), pdf_path, exc)
            continue

        parsed += 1
        title = row.get("title", "")
        publication_date = (row.get("publication_date")
                            or row.get("year") or "")
        # assess_ad_relevance() (Stage 04) only ever inspects title and
        # abstract, never sections - correct for PMC records, which always
        # carry a real <abstract>. A guideline/textbook has no abstract
        # field of its own, so leaving this blank would starve the gate of
        # the document's actual content and let a title using only the
        # ambiguous "AD" abbreviation (rather than spelling out Alzheimer)
        # get excluded regardless of what the document is actually about.
        # The first extracted page stands in for it - the closest thing a
        # PDF has to an abstract/executive summary.
        abstract = sections[0]["text"] if sections else ""
        yield {
            "document_id": row["document_id"],
            "pmid": "",
            "pmcid": "",
            "doi": "",
            "title": title,
            "abstract": abstract,
            "sections": [{"section": "title", "text": title}, *sections],
            "mesh_terms": [],
            "publication_date": str(publication_date),
            "date_precision": "year" if str(publication_date).strip().isdigit() else "",
            "license": row.get("license", ""),
            "source_tier": _guess_source_tier(row, default=default_source_tier),
            "retracted": False,
        }

    log.info(
        "%s | rows=%d | parsed=%d | not_downloaded=%d | restricted=%d | failed=%d",
        registry_path.name, len(rows), parsed, not_downloaded, restricted, failed,
    )
