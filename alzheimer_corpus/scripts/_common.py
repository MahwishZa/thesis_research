"""Shared helpers for the Alzheimer's corpus pipeline.

One module rather than seven copies of config loading, logging and the
Alzheimer's relevance gate. Every stage script imports from here.
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

BASE = Path(__file__).resolve().parents[1]          # alzheimer_corpus/
CONFIG = BASE / "config"
DATA = BASE / "data"
LOGS = BASE / "logs"
METADATA = BASE / "metadata"
REPORTS = BASE / "reports"


# --------------------------------------------------------------------------
# config + logging
# --------------------------------------------------------------------------
def load_config(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def get_logger(stage: str, logfile: str) -> logging.Logger:
    """Append-only stage logger. Never truncates: logs are provenance."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(stage)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)-7s %(name)s %(message)s")
    fmt.converter = lambda *a: datetime.now(timezone.utc).timetuple()
    fh = logging.FileHandler(LOGS / logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_report(name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / name
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return p


def read_jsonl(p: Path) -> Iterable[dict[str, Any]]:
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(p: Path, rows: Iterable[dict[str, Any]]) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------
# licensing: fails closed
# --------------------------------------------------------------------------
DISTRIBUTABLE = {"CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA",
                 "CC-BY-ND", "public-domain", "us-government-work"}


def redistribution_allowed(license_code: str | None) -> bool:
    """Unknown licence => NOT redistributable. Never assume permission."""
    if not license_code:
        return False
    return str(license_code).strip().upper() in {d.upper() for d in DISTRIBUTABLE}


# --------------------------------------------------------------------------
# Alzheimer's relevance gate
# --------------------------------------------------------------------------
# Surface forms preserved verbatim in stored text; folding is for MATCHING ONLY.
PRESERVE_VERBATIM = ("Aβ42", "Aβ40", "Aβ", "p-tau181", "p-tau217", "p-tau231",
                     "ARIA-E", "ARIA-H", "APOE ε4")

AD_TERMS = ("Alzheimer disease", "Alzheimer's disease", "Alzheimer’s disease",
            "Alzheimer dementia", "Alzheimer's dementia", "AD dementia",
            "Alzheimer-type dementia", "dementia of the Alzheimer type",
            "senile dementia of the Alzheimer type", "Alzheimers disease")
PATHOLOGY = ("amyloid beta", "abeta", "amyloid plaques", "tau", "p-tau",
             "phosphorylated tau", "neurofibrillary tangles")
COMPARATOR = ("vascular dementia", "vascular cognitive impairment", "lewy body",
              "dementia with lewy bodies", "frontotemporal dementia",
              "frontotemporal lobar degeneration", "normal pressure hydrocephalus",
              "progressive supranuclear palsy", "corticobasal")
GENERIC = ("dementia", "cognitive impairment", "cognitive dysfunction",
           "cognitive decline", "memory impairment", "neurodegeneration",
           "neurocognitive disorder")


def fold(text: str) -> str:
    """Casefold + strip accents FOR MATCHING ONLY. Never written back to text."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.casefold()


def _rx(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(fold(term)) + r"(?!\w)")


_AD_RX = [(t, _rx(t)) for t in AD_TERMS]
_PATH_RX = [(t, _rx(t)) for t in PATHOLOGY]
_COMP_RX = [(t, _rx(t)) for t in COMPARATOR]
_GEN_RX = [(t, _rx(t)) for t in GENERIC]
# "AD" is ambiguous (atopic dermatitis, autosomal dominant): case-sensitive,
# word-bounded, and only counted when a long form appears in the same record.
_ABBREV_RX = re.compile(r"(?<!\w)AD(?!\w)")


@dataclass
class Relevance:
    """Full decision trace. No verdict is left implicit."""
    document_id: str
    ad_relevant: bool
    score: float
    rules_fired: list[str] = field(default_factory=list)
    exclusion_reason: str | None = None
    anchor_evidence: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    comparator_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hits(text: str, pairs) -> list[str]:
    f = fold(text)
    return [t for t, rx in pairs if rx.search(f)]


def assess_ad_relevance(record: dict[str, Any], mesh_anchors: set[str] | None = None) -> Relevance:
    """Does this record materially support Alzheimer's disease research?

    Supporting concepts (dementia, cognitive dysfunction, amyloid/tau,
    comparator diseases) qualify a record ONLY alongside an Alzheimer's anchor.
    They never qualify one alone - that is what keeps this an Alzheimer's corpus.
    """
    mesh_anchors = mesh_anchors or {"alzheimer disease"}
    doc_id = str(record.get("document_id") or record.get("pmid") or "")
    title = str(record.get("title") or "")
    abstract = str(record.get("abstract") or "")
    whole = f"{title}\n{abstract}"

    d = Relevance(document_id=doc_id, ad_relevant=False, score=0.0)

    mesh_hits = [m for m in (record.get("mesh_terms") or [])
                 if fold(str(m).split("/")[0].lstrip("*").strip()) in mesh_anchors]
    title_hits = _hits(title, _AD_RX)
    abs_hits = _hits(abstract, _AD_RX)
    longform = bool(mesh_hits or title_hits or abs_hits)
    abbrev = bool(_ABBREV_RX.search(whole))

    d.anchor_evidence = ([f"mesh:{m}" for m in mesh_hits]
                         + [f"title:{t}" for t in title_hits]
                         + [f"abstract:{t}" for t in abs_hits])
    path_hits = _hits(whole, _PATH_RX)
    comp_hits = _hits(whole, _COMP_RX)
    gen_hits = _hits(whole, _GEN_RX)
    d.supporting_evidence = [f"pathology:{p}" for p in path_hits]
    d.comparator_evidence = [f"comparator:{c}" for c in comp_hits]

    if not longform:
        if abbrev:
            d.exclusion_reason = "ambiguous_abbreviation_only"
        elif comp_hits:
            d.exclusion_reason = "comparator_disease_no_ad_anchor"
        elif gen_hits:
            d.exclusion_reason = "generic_dementia_no_ad_anchor"
        else:
            d.exclusion_reason = "no_ad_evidence"
        return d

    score = 0.0
    if mesh_hits:
        d.rules_fired.append("R1_mesh_anchor"); score = max(score, 1.0)
    if title_hits:
        d.rules_fired.append("R2_title_anchor"); score = max(score, 1.0)
    if abs_hits:
        d.rules_fired.append("R3_abstract_anchor"); score = max(score, 0.8)
    if path_hits:
        d.rules_fired.append("R4_pathology_with_anchor"); score = max(score, 0.7)
    if comp_hits:
        d.rules_fired.append("R5_differential_with_anchor"); score = max(score, 0.7)

    d.score = round(score, 3)
    d.ad_relevant = d.score >= 0.7
    return d
