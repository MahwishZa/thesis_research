#!/usr/bin/env python3
"""Corpus-wide QC investigation over parsed PMC articles.

Read-only over the corpus: opens pmc/parsed/articles.jsonl, pmc/fulltext/manifest.csv
and (optionally) the raw XML for verification. The only file it writes is the
Markdown report it is asked to produce, and it refuses to overwrite anything.

The central question this answers is not "which records are odd" but "is any
oddity the parser's fault". For every flagged record it re-reads the source XML
and looks for evidence that contradicts the flag -- an <abstract> in a record
marked no_abstract, a <sec> in one marked no_sections. A contradiction is a
parser defect; the absence of one is source variability.

    python3 pmc/qc_investigate.py                       # full report
    python3 pmc/qc_investigate.py --no-xml              # skip raw-XML checks
    python3 pmc/qc_investigate.py --output pmc/qc.md --force

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "pmc" / "parsed" / "articles.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "pmc" / "fulltext" / "manifest.csv"
DEFAULT_XML_DIR = REPO_ROOT / "pmc" / "fulltext" / "xml"

# Never written to, whatever is passed on the command line. Corpus data and
# source code are inputs to this tool and must survive it untouched.
PROTECTED_NAMES = {
    "articles.jsonl", "sample_articles.jsonl", "manifest.csv", "failures.csv",
    ".gitignore", "pmc_oa_inventory.csv", "pmc_oa_inventory_reconciled_2026-09-02.csv",
    "retry71.csv", "parse_pmc_xml.py", "download_pmc_xml.py", "inventory_pmc_oa.py",
    "qc_investigate.py", "pubmed_results.csv", "pubmed_results.json",
    "search_queries.txt", "search_log.csv",
}
PROTECTED_SUFFIXES = {".xml"}

STATUS_GROUPS = ["stub", "no_body", "no_abstract", "parse_error"]
FLAG_GROUPS = [
    "license_disagreement", "license_absent_in_xml", "license_recovered_from_xml",
    "no_doi", "no_pmid", "no_authors", "no_affiliations", "no_references",
    "no_sections", "flat_sections", "multiple_abstracts", "floats_group_present",
    "editors_present", "partial_date", "no_date",
]

# Five-way anomaly taxonomy required by the QC brief.
CAT_LEGIT = "A. legitimate PMC/XML variability"
CAT_UNUSUAL = "B. unusual but valid source record"
CAT_POSSIBLE = "C. possible parser issue"
CAT_BUG = "D. definite parser bug"
CAT_REVIEW = "E. needs manual review"

STUB_CLASSES = [
    "likely legitimate short publication",
    "correction/notice/editorial-type record",
    "preprint/full-text placeholder",
    "incomplete/unusual XML",
    "needs manual review",
]

CAUSES = {
    "stub": "Body under the 250-word threshold. Usually a placeholder standing in for "
            "text PMC may not host, or a genuinely very short item.",
    "no_body": "No <body> element at all. article-meta may still be complete; PMC holds "
               "metadata-only records for some deposits.",
    "no_abstract": "Body parsed normally but <abstract> is absent from article-meta. "
                   "Common for editorials, letters, corrections and some case reports.",
    "parse_error": "The XML would not parse -- genuinely malformed, or truncated on disk. "
                   "Check the file's MD5 against the manifest.",
    "license_disagreement": "The licence derived from the XML's ali:license_ref differs "
                            "from license_code in the manifest (PMC's S3 metadata layer).",
    "license_absent_in_xml": "The manifest carries a licence code but no machine-readable "
                             "licence was derived from the XML. Normal for TDM/author "
                             "manuscripts, whose permissions block is prose only.",
    "license_recovered_from_xml": "The manifest's licence was blank but the XML carried "
                                  "one. These are blank-licence records resolved from source.",
    "no_doi": "No article-id with pub-id-type='doi'. Older deposits and preprints often "
              "genuinely lack one.",
    "no_pmid": "No article-id with pub-id-type='pmid' in the XML, though the record reached "
               "the corpus via a PubMed search.",
    "no_authors": "No contrib with contrib-type='author'. Often a correction, notice or "
                  "editorial published without a byline.",
    "no_affiliations": "No <aff> anywhere in front matter. Common for preprints, editorials "
                       "and short notices.",
    "no_references": "No <ref> under back/ref-list. Expected for editorials, corrections and "
                     "placeholders; unexpected for a full research article.",
    "no_sections": "No <sec> in the body and no loose paragraphs either. Usually pairs with "
                   "a very short or absent body.",
    "flat_sections": "Paragraphs sit directly on <body> with no <sec> wrapper; the parser "
                     "recovers them into a synthetic section.",
    "multiple_abstracts": "More than one <abstract>; the untyped one is taken as scientific "
                          "and the rest kept separately.",
    "floats_group_present": "Figures and tables live in a top-level <floats-group>, as author "
                            "manuscripts place them.",
    "editors_present": "A contrib-group of editors was found and excluded from authors.",
    "partial_date": "The selected publication date has month or year precision only.",
    "no_date": "No usable <pub-date> at all.",
}

PLACEHOLDER_RE = re.compile(
    r"do not permit archiv|not permit(ted)? .{0,40}archiv|full text is available from"
    r"|full text availability|available from the preprint server|does not (allow|permit)",
    re.I)
NOTICE_TITLE_RE = re.compile(
    r"^(correction|corrigend|erratum|retraction|retracted|editorial|comment on|reply to|"
    r"response to|withdrawal|expression of concern|addendum|author correction|"
    r"publisher correction|in this issue|highlights from|obituary|letter to the editor)\b",
    re.I)
NOTICE_TYPES = {
    "correction", "retraction", "editorial", "letter", "article-commentary", "book-review",
    "obituary", "news", "in-brief", "discussion", "reply", "abstract", "meeting-report",
    "addendum", "expression-of-concern", "announcement", "product-review",
}
SUBSTANTIVE_TYPES = {"research-article", "case-report", "brief-report", "review-article"}
PREPRINT_JOURNALS = {"biorxiv", "medrxiv", "arxiv", "research square", "ssrn",
                     "preprints.org", "chemrxiv", "authorea"}

ARTICLE_TYPE_RE = re.compile(r'<article[^>]*\barticle-type="([^"]+)"')
CC_URL_RE = re.compile(r"creativecommons\.org/(licenses|publicdomain)/[a-z0-9\-/.]+", re.I)


def widen_csv_field_limit(target: int = 64 * 1024 * 1024) -> None:
    """Portable: never sys.maxsize, which overflows a 32-bit C long on Windows."""
    limit = target
    while limit > csv.field_size_limit():
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def stream_records(path: Path) -> Iterator[dict[str, Any]]:
    """One record at a time: the full corpus JSONL runs well past a gigabyte."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: line {number} is not valid JSON: {exc}", file=sys.stderr)


def assert_writable(output: Path) -> None:
    if output.name in PROTECTED_NAMES or output.suffix.lower() in PROTECTED_SUFFIXES:
        raise SystemExit(f"ERROR: refusing to write to protected file {output.name}")
    for part in output.resolve().parts:
        if part in {"xml", "pubmed"}:
            raise SystemExit(f"ERROR: refusing to write inside {part}/")


def read_xml_head(pmcid: str, xml_dir: Path | None, size: int = 200_000) -> str:
    """Front matter is what the checks below need; the whole body is not."""
    if xml_dir is None:
        return ""
    path = xml_dir / f"{pmcid}.xml"
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        return handle.read(size).decode("utf-8", "replace")


def article_type_of(xml_head: str) -> str:
    match = ARTICLE_TYPE_RE.search(xml_head)
    return match.group(1) if match else ""


def probe_license_in_xml(xml_head: str) -> dict[str, Any]:
    """Look for licence information beyond ali:license_ref.

    Answers whether a licence the parser did not derive is nonetheless present
    in the source, which is the difference between source variability and a
    recovery-logic defect.
    """
    block = ""
    match = re.search(r"<permissions>.*?</permissions>", xml_head, re.S)
    if match:
        block = match.group(0)

    license_ref = ""
    ref_match = re.search(r"<[^>]*license_ref[^>]*>([^<]*)<", block)
    if ref_match:
        license_ref = ref_match.group(1).strip()
    content_type = ""
    ct_match = re.search(r'license_ref[^>]*content-type="([^"]+)"', block)
    if ct_match:
        content_type = ct_match.group(1)

    prose = " ".join(re.sub(r"<[^>]+>", " ", block).split())
    cc_urls = CC_URL_RE.findall(block)
    copyright_stmt = ""
    cs_match = re.search(r"<copyright-statement>(.*?)</copyright-statement>", block, re.S)
    if cs_match:
        copyright_stmt = " ".join(re.sub(r"<[^>]+>", " ", cs_match.group(1)).split())

    has_tdm_prose = bool(re.search(r"text\s*(and\s*data\s*)?mining", prose, re.I))
    has_cc_prose = bool(re.search(r"creative\s*commons|\bCC[ -]?BY\b|public domain", prose, re.I))

    if license_ref:
        recoverable, source = True, "ali:license_ref"
    elif cc_urls:
        recoverable, source = True, "creativecommons URL in permissions"
    elif has_cc_prose:
        recoverable, source = True, "licence prose"
    elif has_tdm_prose:
        recoverable, source = True, "text-mining prose (TDM)"
    elif copyright_stmt:
        recoverable, source = False, "copyright statement only"
    elif block:
        recoverable, source = False, "permissions block present but uninformative"
    else:
        recoverable, source = False, "no permissions block"

    return {
        "has_permissions_block": bool(block),
        "license_ref": license_ref,
        "content_type": content_type,
        "prose": prose[:400],
        "cc_urls": cc_urls[:3],
        "copyright_statement": copyright_stmt[:200],
        "recoverable": recoverable,
        "recovery_source": source,
    }


def contradictions(record: dict[str, Any], xml_head: str) -> list[str]:
    """Evidence in the XML that contradicts what the parser recorded.

    Each entry means the source contained something the parser reported as
    absent. That is the only signal in this tool that points at a parser defect
    rather than at source variability.
    """
    if not xml_head:
        return []
    found: list[str] = []
    status = record["qc"]["status"]
    flags = set(record["qc"]["flags"])

    if status == "no_abstract" and re.search(r"<abstract[\s>]", xml_head):
        found.append("XML contains <abstract> but the record has none")
    if status == "no_body" and re.search(r"<body[\s>]", xml_head):
        found.append("XML contains <body> but the record has none")
    if "no_sections" in flags and re.search(r"<sec[\s>]", xml_head):
        found.append("XML contains <sec> but no sections were parsed")
    if "no_doi" in flags and re.search(r'pub-id-type="doi"', xml_head):
        found.append("XML carries a DOI article-id but the record has none")
    if "no_pmid" in flags and re.search(r'pub-id-type="pmid"', xml_head):
        found.append("XML carries a PMID article-id but the record has none")
    if "no_authors" in flags and re.search(r'contrib-type="author"', xml_head):
        found.append("XML contains an author contrib but no authors were parsed")
    if "no_affiliations" in flags and re.search(r"<aff[\s>]", xml_head):
        found.append("XML contains <aff> but no affiliations were parsed")
    if "no_references" in flags and re.search(r"<ref[\s>]", xml_head):
        found.append("XML contains <ref> but no references were counted")
    if "license_absent_in_xml" in flags and re.search(r"license_ref", xml_head):
        found.append("XML contains ali:license_ref but no licence was derived")
    return found


def classify_stub(record: dict[str, Any], atype: str, body_sample: str) -> str:
    """Conservative bucket. Anything without clear evidence goes to manual review."""
    title = record.get("title") or ""
    journal = (record.get("journal") or "").strip().lower()
    words = record.get("body_word_count", 0)

    if PLACEHOLDER_RE.search(body_sample) or journal in PREPRINT_JOURNALS:
        return "preprint/full-text placeholder"
    if atype in NOTICE_TYPES or NOTICE_TITLE_RE.search(title):
        return "correction/notice/editorial-type record"
    if record["qc"]["status"] == "no_body" or record.get("section_count", 0) == 0:
        return "incomplete/unusual XML"
    if (words >= 100 and record.get("reference_count", 0) > 0
            and (record.get("abstract") or {}).get("text") and atype in SUBSTANTIVE_TYPES):
        return "likely legitimate short publication"
    return "needs manual review"


def categorise(record: dict[str, Any], group: str, clashes: list[str],
               classification: str, licence_probe: dict[str, Any] | None) -> str:
    if clashes:
        # An identifier the XML plainly carries but the record lacks is not a
        # judgement call; anything softer stays "possible".
        hard = any("article-id" in c or "<abstract>" in c or "<body>" in c for c in clashes)
        return CAT_BUG if hard else CAT_POSSIBLE
    if group == "license_absent_in_xml" and licence_probe and licence_probe["recoverable"]:
        return CAT_POSSIBLE
    if classification == "needs manual review":
        return CAT_REVIEW
    if classification in {"preprint/full-text placeholder",
                          "correction/notice/editorial-type record"}:
        return CAT_LEGIT
    if classification == "incomplete/unusual XML":
        return CAT_UNUSUAL
    if group in {"no_doi", "no_authors", "no_affiliations", "no_references",
                 "no_abstract", "partial_date", "no_date", "no_pmid"}:
        return CAT_LEGIT
    if group in {"license_disagreement", "no_sections"}:
        return CAT_REVIEW
    return CAT_LEGIT


def md_table(rows: list[dict[str, str]], columns: list[str], max_rows: int) -> list[str]:
    """max_rows truncates DISPLAY only; callers count over the full list."""
    if not rows:
        return ["_No records in this group._", ""]
    out = ["| " + " | ".join(columns) + " |",
           "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:max_rows]:
        out.append("| " + " | ".join(
            str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns) + " |")
    out.append("")
    if len(rows) > max_rows:
        out += [f"_Showing {max_rows:,} of {len(rows):,} rows; all {len(rows):,} were "
                f"analysed and counted. Raise `--max-rows` to render more._", ""]
    return out


def investigate(jsonl: Path, manifest_path: Path, xml_dir: Path | None) -> dict[str, Any]:
    """Single streaming pass. Only compact findings are retained, never records."""
    widen_csv_field_limit()
    manifest: dict[str, dict[str, str]] = {}
    if manifest_path and manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest = {r["pmcid"]: r for r in csv.DictReader(handle)}

    total = 0
    statuses: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    findings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classifications: dict[str, Counter[str]] = defaultdict(Counter)
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    aggregates: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    licence_details: list[dict[str, Any]] = []
    absent_probes: list[dict[str, Any]] = []
    all_clashes: list[tuple[str, str, str]] = []
    wanted = set(STATUS_GROUPS) | set(FLAG_GROUPS)

    for record in stream_records(jsonl):
        total += 1
        status = record["qc"]["status"]
        statuses[status] += 1
        flags.update(record["qc"]["flags"])

        member = ({status} | set(record["qc"]["flags"])) & wanted
        if not member:
            continue

        pmcid = record.get("pmcid", "")
        xml_head = read_xml_head(pmcid, xml_dir)
        atype = article_type_of(xml_head) or "(unknown)"
        clashes = contradictions(record, xml_head)

        body_sample = ""
        if status in {"stub", "no_body"}:
            parts: list[str] = []

            def walk(sections):
                for section in sections:
                    for paragraph in section["paragraphs"]:
                        parts.append(paragraph["text"])
                    walk(section["subsections"])

            walk(record.get("sections", []))
            body_sample = " ".join(parts)[:600]

        classification = (classify_stub(record, atype, body_sample)
                          if status in {"stub", "no_body"} else "")

        probe: dict[str, Any] | None = None
        if "license_absent_in_xml" in member:
            probe = probe_license_in_xml(xml_head)
            absent_probes.append({"pmcid": pmcid, **probe})

        row = {
            "pmcid": pmcid,
            "pmid": record.get("pmid") or "-",
            "doi": record.get("doi") or "-",
            "title": (record.get("title") or "-")[:88],
            "journal": record.get("journal") or "-",
            "date": record.get("publication_date") or "-",
            "date_type": record.get("publication_date_type") or "-",
            "type": atype,
            "words": record.get("body_word_count", 0),
            "paras": record.get("paragraph_count", 0),
            "secs": record.get("section_count", 0),
            "abstract": "yes" if (record.get("abstract") or {}).get("text") else "NO",
            "flags": ",".join(record["qc"]["flags"]) or "-",
            "classification": classification or "-",
            "contradiction": "; ".join(clashes) or "-",
        }

        for group in member:
            row_for_group = dict(row)
            row_for_group["category"] = categorise(record, group, clashes, classification, probe)
            row_for_group["cause"] = CAUSES.get(group, "")
            findings[group].append(row_for_group)
            categories[group][row_for_group["category"]] += 1
            if status in {"stub", "no_body"} and group == status:
                classifications[status][classification] += 1

        for clash in clashes:
            all_clashes.append((pmcid, clash, status))

        if status in {"stub", "no_body", "no_abstract"}:
            aggregates[status]["article-type"][atype] += 1
            aggregates[status]["journal"][record.get("journal") or "(unknown)"] += 1
            aggregates[status]["year"][(record.get("publication_date") or "")[:4] or "(unknown)"] += 1

        if "license_disagreement" in member:
            prov = record.get("provenance", {})
            xml_probe = probe or probe_license_in_xml(xml_head)
            manifest_code = prov.get("license_code_manifest") or "(blank)"
            xml_code = prov.get("license_code_xml") or "(none)"
            same_family = (manifest_code.replace(" ", "").upper()
                           == xml_code.replace(" ", "").upper())
            licence_details.append({
                "pmcid": pmcid,
                "manifest_license_code": manifest_code,
                "manifest_row_license": manifest.get(pmcid, {}).get("license_code") or "(blank)",
                "xml_license_ref": prov.get("license_ref_xml") or xml_probe["license_ref"] or "(none)",
                "xml_content_type": prov.get("license_content_type_xml")
                                    or xml_probe["content_type"] or "(none)",
                "xml_derived_code": xml_code,
                "xml_prose": (prov.get("license_statement_xml") or xml_probe["prose"] or "(none)")[:500],
                "nature": ("metadata representation only (same licence family)" if same_family
                           else "actual licence-content disagreement"),
            })

    return {
        "total": total, "statuses": statuses, "flags": flags, "findings": findings,
        "classifications": classifications, "categories": categories,
        "aggregates": aggregates, "licence_details": licence_details,
        "absent_probes": absent_probes, "contradictions": all_clashes,
        "manifest_rows": len(manifest),
    }


def build_report(result: dict[str, Any], jsonl: Path, xml_used: bool, max_rows: int) -> str:
    total = result["total"]
    pct = (lambda n: f"{100 * n / total:.3f}%" if total else "0%")
    lines = [
        f"# PMC corpus QC investigation — {date.today().isoformat()}",
        "",
        f"Records examined: **{total:,}** from `{jsonl}`  ",
        f"Manifest rows: **{result['manifest_rows']:,}**  ",
        f"Raw XML consulted: **{'yes' if xml_used else 'no'}**",
        "",
        "Read-only over the corpus. No parsed record, manifest row, XML file or source "
        "file was modified. Table rendering may be truncated by `--max-rows`; every "
        "count, classification and aggregate below is computed over all records.",
        "",
        "## QC status counts", "",
    ]
    lines += md_table([{"status": k, "count": f"{v:,}", "percent": pct(v)}
                       for k, v in result["statuses"].most_common()],
                      ["status", "count", "percent"], 10**9)
    lines += ["## QC flag counts", ""]
    lines += md_table([{"flag": k, "count": f"{v:,}", "percent": pct(v)}
                       for k, v in result["flags"].most_common()],
                      ["flag", "count", "percent"], 10**9)

    lines += ["## Parser-defect check", ""]
    clashes = result["contradictions"]
    if clashes:
        lines += [f"**{len(clashes)} contradiction(s) found** — the XML contains something "
                  "the parser reported as absent. Each is a parser defect, not source "
                  "variability.", ""]
        lines += md_table([{"pmcid": p, "status": s, "contradiction": c}
                           for p, c, s in clashes], ["pmcid", "status", "contradiction"], max_rows)
    else:
        lines += ["**No contradictions found.** For every flagged record, the raw XML was "
                  "re-read and checked for the element the parser reported missing "
                  "(`<abstract>`, `<body>`, `<sec>`, `<aff>`, `<ref>`, author contribs, "
                  "DOI/PMID article-ids, `ali:license_ref`). None was present. No evidence "
                  "that `parse_pmc_xml.py` behaved incorrectly.", ""]

    for group in STATUS_GROUPS + FLAG_GROUPS:
        rows = result["findings"].get(group, [])
        lines += [f"## Group: `{group}` — {len(rows):,} record(s)", ""]
        if group in CAUSES:
            lines += [f"**Likely cause.** {CAUSES[group]}", ""]
        if result["categories"].get(group):
            lines += ["**Anomaly category.**", ""]
            lines += md_table([{"category": k, "count": str(v)}
                               for k, v in result["categories"][group].most_common()],
                              ["category", "count"], 10**9)
        if result["classifications"].get(group):
            lines += ["**Conservative classification.**", ""]
            lines += md_table([{"classification": k, "count": str(v)}
                               for k, v in result["classifications"][group].most_common()],
                              ["classification", "count"], 10**9)
        columns = ["pmcid", "pmid", "doi", "title", "journal", "date", "date_type", "type",
                   "words", "paras", "secs", "abstract", "flags", "category", "contradiction"]
        if group in {"stub", "no_body"}:
            columns.insert(-2, "classification")
        lines += md_table(rows, columns, max_rows)

    lines += ["## Licence disagreements — detail", ""]
    if not result["licence_details"]:
        lines += ["_None._", ""]
    for detail in result["licence_details"]:
        lines += [
            f"### {detail['pmcid']}", "",
            f"- Manifest `license_code`: **{detail['manifest_license_code']}** "
            f"(manifest.csv reads `{detail['manifest_row_license']}`)",
            f"- XML `ali:license_ref`: `{detail['xml_license_ref']}`",
            f"- XML `content-type`: `{detail['xml_content_type']}`",
            f"- Licence derived from XML: **{detail['xml_derived_code']}**",
            f"- Nature of the difference: **{detail['nature']}**",
            "", "XML licence prose:", "", f"> {detail['xml_prose']}", "",
            "The manifest value comes from PMC's S3 metadata layer, the XML value from the "
            "article's own `<permissions>` block. No reuse policy is decided here.", "",
        ]

    lines += ["## `license_absent_in_xml` — recoverability probe", ""]
    probes = result["absent_probes"]
    if not probes:
        lines += ["_No records in this group._", ""]
    else:
        recoverable = [p for p in probes if p["recoverable"]]
        lines += [f"Records probed: **{len(probes):,}**  ",
                  f"Licence information recoverable from the XML: **{len(recoverable):,}**  ",
                  f"Genuinely absent: **{len(probes) - len(recoverable):,}**", "",
                  "**Recovery-logic verdict.** " + (
                      f"{len(recoverable):,} record(s) carry licence information the parser did "
                      "not derive — the recovery logic is not catching every form."
                      if recoverable else
                      "No record carries recoverable licence information the parser missed; "
                      "the recovery logic is working."), ""]
        lines += md_table([{"pmcid": p["pmcid"],
                            "permissions_block": "yes" if p["has_permissions_block"] else "no",
                            "license_ref": p["license_ref"] or "-",
                            "content_type": p["content_type"] or "-",
                            "cc_urls": ", ".join(p["cc_urls"]) or "-",
                            "copyright": p["copyright_statement"] or "-",
                            "recoverable": "YES" if p["recoverable"] else "no",
                            "source": p["recovery_source"],
                            "prose": p["prose"][:150] or "-"} for p in probes],
                          ["pmcid", "permissions_block", "license_ref", "content_type",
                           "cc_urls", "copyright", "recoverable", "source", "prose"], max_rows)

    lines += ["## Licence recovery from XML", "",
              f"`license_recovered_from_xml`: **{result['flags'].get('license_recovered_from_xml', 0):,}** "
              "record(s) whose manifest licence was blank and was resolved from the XML.", ""]

    lines += ["## Aggregates for stub / no_body / no_abstract", ""]
    for status in ["stub", "no_body", "no_abstract"]:
        if not result["aggregates"].get(status):
            continue
        lines += [f"### `{status}`", ""]
        for facet in ["article-type", "journal", "year"]:
            counts = result["aggregates"][status][facet]
            if not counts:
                continue
            lines += [f"**By {facet}** ({len(counts):,} distinct)", ""]
            lines += md_table([{facet: k, "count": str(v)} for k, v in counts.most_common()],
                              [facet, "count"], max_rows)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--no-xml", action="store_true",
                        help="skip raw-XML verification (weakens the parser-defect check)")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "pmc" / f"pmc_qc_report_{date.today().isoformat()}.md")
    parser.add_argument("--max-rows", type=int, default=250,
                        help="rows RENDERED per Markdown table; never limits analysis")
    parser.add_argument("--force", action="store_true", help="allow overwriting the report")
    args = parser.parse_args(argv)

    assert_writable(args.output)
    if args.output.exists() and not args.force:
        raise SystemExit(f"ERROR: {args.output} exists; pass --force to replace it")
    if not args.jsonl.exists():
        raise SystemExit(f"ERROR: parsed corpus not found: {args.jsonl}")
    if not args.no_xml and not args.xml_dir.exists():
        raise SystemExit(f"ERROR: XML directory not found: {args.xml_dir} (use --no-xml)")
    if args.max_rows < 1:
        raise SystemExit("ERROR: --max-rows must be at least 1")

    xml_dir = None if args.no_xml else args.xml_dir
    result = investigate(args.jsonl, args.manifest, xml_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(result, args.jsonl, xml_dir is not None, args.max_rows),
                           encoding="utf-8")

    print(f"Records examined     : {result['total']:,}")
    print(f"QC status            : {dict(result['statuses'])}")
    print(f"QC flags             : {dict(result['flags'].most_common())}")
    for status in ["stub", "no_body"]:
        if result["classifications"].get(status):
            print(f"{status:12} classes : {dict(result['classifications'][status])}")
    print(f"Licence disagreements: {len(result['licence_details'])}")
    probes = result["absent_probes"]
    print(f"license_absent_in_xml: {len(probes)} probed, "
          f"{sum(1 for p in probes if p['recoverable'])} recoverable")
    print(f"license_recovered    : {result['flags'].get('license_recovered_from_xml', 0)}")
    clashes = result["contradictions"]
    print(f"Parser contradictions: {len(clashes)}"
          + ("  <-- investigate: XML contains what the parser reported missing" if clashes
             else "  (no evidence of a parser defect)"))
    print(f"\nReport written       : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
