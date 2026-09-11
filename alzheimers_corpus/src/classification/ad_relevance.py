#!/usr/bin/env python3
"""Alzheimer's-disease relevance gate.

Answers one question per record, with a full decision trace:

    "Does this document materially support Alzheimer's disease research?"

The gate exists because the corpus must not become a generic dementia corpus
(spec section 32). Supporting concepts - dementia, cognitive dysfunction,
amyloid/tau pathology, comparator diseases - can qualify a record only when an
Alzheimer's ANCHOR is also present. They never qualify one alone.

Every rule is loaded from config, never hardcoded here, so the inclusion logic
is inspectable and changeable without touching code.

No network access. Pure function of (record, config).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "alzheimers_corpus" / "config"

# Comparator diseases that are NOT Alzheimer's. Presence without an AD anchor is
# what rule X2 excludes. Kept beside the differential MeSH tier in mesh_terms.yaml.
COMPARATOR_TERMS = (
    "vascular dementia", "vascular cognitive impairment",
    "lewy body", "dementia with lewy bodies", "dlb",
    "frontotemporal dementia", "frontotemporal lobar degeneration", "ftd", "ftld",
    "normal pressure hydrocephalus", "huntington", "creutzfeldt",
    "progressive supranuclear palsy", "corticobasal",
)
GENERIC_TERMS = (
    "dementia", "cognitive impairment", "cognitive dysfunction",
    "cognitive decline", "memory impairment", "neurodegeneration",
    "neurocognitive disorder",
)


def _fold(text: str) -> str:
    """Casefold for matching only. Never applied to stored text.

    Biomedical surface forms (Abeta42, p-tau217, ARIA-E, APOE e4) must survive
    in the corpus verbatim; this folding exists solely to make matching
    case- and accent-insensitive.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.casefold()


def _word_re(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(_fold(term)) + r"(?!\w)")


@dataclass
class Decision:
    """The full trace for one record. Nothing about the verdict is implicit."""
    document_id: str
    ad_relevant: bool
    score: float
    rules_fired: list[str] = field(default_factory=list)
    exclusion_fired: str | None = None
    exclusion_reason: str | None = None
    anchor_evidence: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    comparator_evidence: list[str] = field(default_factory=list)
    config_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ADRelevanceClassifier:
    def __init__(self, config_dir: Path | None = None) -> None:
        cd = Path(config_dir) if config_dir else CONFIG_DIR
        self.corpus_cfg = yaml.safe_load((cd / "corpus_config.yaml").read_text())
        self.vocab = yaml.safe_load((cd / "controlled_vocabulary.yaml").read_text())
        self.mesh = yaml.safe_load((cd / "mesh_terms.yaml").read_text())
        # Source pools, tiers and licensing policy are owned by sources.yaml.
        self.sources = yaml.safe_load((cd / "sources.yaml").read_text())

        rel = self.corpus_cfg["ad_relevance"]
        self.min_score: float = float(rel["min_score"])
        self.rules = {r["id"]: r for r in rel["inclusion_rules"]}
        self.exclusions = {r["id"]: r for r in rel["exclusion_rules"]}
        self.version = str(self.corpus_cfg["version"])

        groups = self.vocab["groups"]
        anchor_terms: list[str] = []
        for g in groups.values():
            if g.get("anchor"):
                anchor_terms.extend(g.get("terms", []))
        self._anchor = [(t, _word_re(t)) for t in anchor_terms]

        # Ambiguous abbreviations: matched separately and never decisive on their own.
        self._abbrev: list[tuple[str, re.Pattern[str], bool]] = []
        for g in groups.values():
            for a in g.get("abbreviations", []) or []:
                self._abbrev.append((a["token"],
                                     re.compile(r"(?<!\w)" + re.escape(a["token"]) + r"(?!\w)"),
                                     bool(a.get("requires_longform_in_document", True))))

        self._pathology = [(t, _word_re(t)) for t in groups.get("pathology", {}).get("terms", [])]
        self._comparator = [(t, _word_re(t)) for t in COMPARATOR_TERMS]
        self._generic = [(t, _word_re(t)) for t in GENERIC_TERMS]

        mesh_primary = self.mesh["tiers"]["primary"]["terms"]
        self._mesh_anchor = {_fold(t["descriptor"]) for t in mesh_primary}

    # -- matching helpers -------------------------------------------------
    @staticmethod
    def _hits(text: str, pairs: Iterable[tuple[str, re.Pattern[str]]]) -> list[str]:
        folded = _fold(text)
        return [term for term, rx in pairs if rx.search(folded)]

    def _mesh_anchor_hit(self, mesh_terms: Iterable[str]) -> list[str]:
        out = []
        for m in mesh_terms or []:
            base = _fold(str(m).split("/")[0].lstrip("*").strip())
            if base in self._mesh_anchor:
                out.append(str(m))
        return out

    # -- main -------------------------------------------------------------
    def classify(self, record: dict[str, Any]) -> Decision:
        doc_id = str(record.get("document_id") or record.get("pmid") or record.get("source_id") or "")
        title = str(record.get("title") or "")
        abstract = str(record.get("abstract") or "")
        mesh_terms = record.get("mesh_terms") or []
        whole = f"{title}\n{abstract}"

        d = Decision(document_id=doc_id, ad_relevant=False, score=0.0,
                     config_version=self.version)

        mesh_hits = self._mesh_anchor_hit(mesh_terms)
        title_hits = self._hits(title, self._anchor)
        abs_hits = self._hits(abstract, self._anchor)
        longform_anywhere = bool(mesh_hits or title_hits or abs_hits)

        # Ambiguous abbreviations only count when a long form is present.
        abbrev_hits: list[str] = []
        abbrev_only: list[str] = []
        for token, rx, needs_longform in self._abbrev:
            if not rx.search(whole):
                continue
            if needs_longform and not longform_anywhere:
                abbrev_only.append(token)
            else:
                abbrev_hits.append(token)

        d.anchor_evidence = (
            [f"mesh:{m}" for m in mesh_hits]
            + [f"title:{t}" for t in title_hits]
            + [f"abstract:{t}" for t in abs_hits]
            + [f"abbrev:{a}" for a in abbrev_hits]
        )
        path_hits = self._hits(whole, self._pathology)
        comp_hits = self._hits(whole, self._comparator)
        gen_hits = self._hits(whole, self._generic)
        d.supporting_evidence = [f"pathology:{p}" for p in path_hits]
        d.comparator_evidence = [f"comparator:{c}" for c in comp_hits]

        any_anchor = bool(mesh_hits or title_hits or abs_hits or abbrev_hits)

        # ---- exclusions first: they are decisive ------------------------
        if not any_anchor:
            if abbrev_only:
                x = self.exclusions["X3_ambiguous_abbreviation_only"]
                d.exclusion_fired, d.exclusion_reason = x["id"], x["reason_code"]
                return d
            if comp_hits:
                x = self.exclusions["X2_comparator_only"]
                d.exclusion_fired, d.exclusion_reason = x["id"], x["reason_code"]
                return d
            if gen_hits:
                x = self.exclusions["X1_generic_dementia_only"]
                d.exclusion_fired, d.exclusion_reason = x["id"], x["reason_code"]
                return d
            return d  # no AD evidence of any kind

        # ---- inclusion rules --------------------------------------------
        score = 0.0
        if mesh_hits:
            d.rules_fired.append("R1_mesh_anchor")
            score = max(score, float(self.rules["R1_mesh_anchor"]["weight"]))
        if title_hits:
            d.rules_fired.append("R2_title_anchor")
            score = max(score, float(self.rules["R2_title_anchor"]["weight"]))
        if abs_hits:
            d.rules_fired.append("R3_abstract_anchor")
            score = max(score, float(self.rules["R3_abstract_anchor"]["weight"]))
        if path_hits and any_anchor:
            d.rules_fired.append("R4_pathology_with_anchor")
            score = max(score, float(self.rules["R4_pathology_with_anchor"]["weight"]))
        if comp_hits and any_anchor:
            d.rules_fired.append("R5_differential_with_anchor")
            score = max(score, float(self.rules["R5_differential_with_anchor"]["weight"]))

        # An anchor found only via a disambiguated abbreviation is supporting-strength.
        if not d.rules_fired and abbrev_hits:
            score = max(score, float(self.rules["R4_pathology_with_anchor"]["weight"]))
            d.rules_fired.append("R4_pathology_with_anchor")

        d.score = round(score, 3)
        d.ad_relevant = d.score >= self.min_score
        return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score records for Alzheimer's relevance.")
    ap.add_argument("--input", required=True, help="JSONL of records")
    ap.add_argument("--output", help="JSONL of decisions (default: stdout summary only)")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args(argv)

    clf = ADRelevanceClassifier(args.config_dir)
    inc = exc = 0
    reasons: dict[str, int] = {}
    out = open(args.output, "w", encoding="utf-8", newline="\n") if args.output else None
    try:
        with open(args.input, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                dec = clf.classify(json.loads(line))
                if dec.ad_relevant:
                    inc += 1
                else:
                    exc += 1
                    reasons[dec.exclusion_reason or "below_min_score"] = \
                        reasons.get(dec.exclusion_reason or "below_min_score", 0) + 1
                if out:
                    out.write(json.dumps(dec.to_dict(), sort_keys=True) + "\n")
    finally:
        if out:
            out.close()
    print(json.dumps({"included": inc, "excluded": exc,
                      "exclusion_reasons": reasons}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
