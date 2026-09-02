#!/usr/bin/env python3
"""Offline tests for qc_investigate.py.

Every fixture is a small synthetic corpus built in a temporary directory. The
real 25,742-record corpus is never opened, and nothing touches the network.

    python3 -m unittest pmc.test_qc_investigate -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import qc_investigate as qc


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def record(
    pmcid: str = "PMC1", *, status: str = "ok", flags: list[str] | None = None,
    title: str = "A Study", journal: str = "Test Journal", pmid: str = "111",
    doi: str = "10.1/x", date_str: str = "2024-05-06", date_type: str = "epub",
    words: int = 4000, paragraphs: int = 30, sections: int = 12, refs: int = 40,
    abstract: str = "An abstract.", body: str = "Body text.",
    license_manifest: str = "CC BY", license_xml: str = "CC BY",
    license_ref: str = "https://creativecommons.org/licenses/by/4.0/",
    content_type: str = "ccbylicense", prose: str = "Open access under CC BY.",
) -> dict:
    return {
        "schema_version": "1.0", "pmcid": pmcid, "pmcid_versioned": f"{pmcid}.1",
        "pmid": pmid, "doi": doi, "journal": journal, "title": title,
        "publication_date": date_str, "publication_date_type": date_type,
        "publication_date_precision": "day", "publication_dates_all": {},
        "authors": [], "author_count": 0, "affiliations": [],
        "abstract": {"text": abstract, "is_structured": False, "sections": []},
        "abstract_other": [],
        "sections": [{
            "section_id": f"{pmcid}.1#s1", "path": [1], "depth": 1,
            "title_raw": "Introduction", "title_normalized": "introduction",
            "imrad": "introduction", "imrad_source": "title_match",
            "sec_type_attr": None, "xml_id_attr": None, "is_content": True,
            "paragraphs": [{"paragraph_id": f"{pmcid}.1#s1.p1", "text": body,
                            "word_count": len(body.split()), "section_title_path": ["Introduction"],
                            "ordinal_in_article": 1, "ordinal_in_section": 1,
                            "section_path": [1], "imrad": "introduction",
                            "text_sha256": "x", "inline_stripped": {},
                            "contains_float_reference": False}],
            "subsections": [],
        }],
        "provenance": {
            "xml_md5": "abc", "xml_bytes": 1000, "pmc_version": "1",
            "is_manuscript": "no", "is_retracted": "no",
            "license_code_manifest": license_manifest, "license_code_xml": license_xml,
            "license_ref_xml": license_ref, "license_content_type_xml": content_type,
            "license_statement_xml": prose,
        },
        "body_word_count": words, "paragraph_count": paragraphs,
        "section_count": sections, "reference_count": refs,
        "figure_count": 0, "table_count": 0,
        "qc": {"status": status, "flags": flags or [], "error": "",
               "parser_version": "1.0.0", "parsed_at_utc": "2026-09-02T00:00:00+00:00"},
    }


XML_TEMPLATE = (
    '<article article-type="{atype}" dtd-version="1.4"><front><article-meta>'
    "{permissions}</article-meta></front><body>{body}</body></article>"
)
PERMISSIONS_CC = (
    "<permissions><copyright-statement>(c) 2024 Author</copyright-statement><license>"
    '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
    ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/</ali:license_ref>'
    "<license-p>Open access under CC BY.</license-p></license></permissions>"
)
PERMISSIONS_TDM = (
    "<permissions><license><license-p>This file is available for text mining."
    "</license-p></license></permissions>"
)
PERMISSIONS_COPYRIGHT_ONLY = (
    "<permissions><copyright-statement>(c) 2024 The Publisher</copyright-statement>"
    "</permissions>"
)


class Corpus:
    """A temporary corpus directory: articles.jsonl, manifest.csv and XML."""

    def __init__(self, records, xml=None, manifest_rows=None):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.jsonl = root / "articles.jsonl"
        self.jsonl.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        self.xml_dir = root / "xml"
        self.xml_dir.mkdir()
        for pmcid, content in (xml or {}).items():
            (self.xml_dir / f"{pmcid}.xml").write_text(content, encoding="utf-8")
        self.manifest = root / "manifest.csv"
        rows = manifest_rows or [{"pmcid": r["pmcid"], "license_code": "CC BY"} for r in records]
        header = "pmcid,license_code\n"
        self.manifest.write_text(
            header + "".join(f"{r['pmcid']},{r['license_code']}\n" for r in rows),
            encoding="utf-8")
        self.output = root / "report.md"

    def run(self, xml=True):
        return qc.investigate(self.jsonl, self.manifest, self.xml_dir if xml else None)

    def close(self):
        self._tmp.cleanup()


class CorpusCase(unittest.TestCase):
    def make(self, records, xml=None, manifest_rows=None) -> Corpus:
        corpus = Corpus(records, xml, manifest_rows)
        self.addCleanup(corpus.close)
        return corpus


# ---------------------------------------------------------------------------
# 1-2. Counting
# ---------------------------------------------------------------------------


class StatusAndFlagCounting(CorpusCase):
    def test_status_counts_every_record(self):
        corpus = self.make([
            record("PMC1"), record("PMC2"),
            record("PMC3", status="stub", words=20),
            record("PMC4", status="no_body", words=0),
            record("PMC5", status="no_abstract", abstract=""),
        ])
        result = corpus.run()
        self.assertEqual(result["total"], 5)
        self.assertEqual(dict(result["statuses"]),
                         {"ok": 2, "stub": 1, "no_body": 1, "no_abstract": 1})

    def test_flag_counts_accumulate_across_records(self):
        corpus = self.make([
            record("PMC1", flags=["no_doi", "no_authors"]),
            record("PMC2", flags=["no_doi"]),
            record("PMC3", flags=["no_references", "no_doi"]),
        ])
        flags = corpus.run()["flags"]
        self.assertEqual(flags["no_doi"], 3)
        self.assertEqual(flags["no_authors"], 1)
        self.assertEqual(flags["no_references"], 1)

    def test_records_appear_in_every_group_they_belong_to(self):
        corpus = self.make([record("PMC1", status="stub", words=10,
                                   flags=["no_references", "no_affiliations"])])
        findings = corpus.run()["findings"]
        for group in ("stub", "no_references", "no_affiliations"):
            self.assertEqual(len(findings[group]), 1, group)

    def test_blank_lines_in_jsonl_are_skipped(self):
        corpus = self.make([record("PMC1"), record("PMC2")])
        corpus.jsonl.write_text(corpus.jsonl.read_text() + "\n\n", encoding="utf-8")
        self.assertEqual(corpus.run()["total"], 2)


# ---------------------------------------------------------------------------
# 3-4. Classification
# ---------------------------------------------------------------------------


class StubClassification(CorpusCase):
    def test_placeholder_body_text_recognised(self):
        corpus = self.make([record(
            "PMC1", status="stub", words=26, sections=1,
            body="The license terms selected by the author(s) do not permit archiving in PMC.")])
        self.assertEqual(dict(corpus.run()["classifications"]["stub"]),
                         {"preprint/full-text placeholder": 1})

    def test_preprint_journal_recognised_without_placeholder_prose(self):
        corpus = self.make([record("PMC1", status="stub", words=30, journal="bioRxiv")])
        self.assertEqual(dict(corpus.run()["classifications"]["stub"]),
                         {"preprint/full-text placeholder": 1})

    def test_correction_recognised_by_article_type(self):
        corpus = self.make(
            [record("PMC1", status="stub", words=40, title="Something", journal="J")],
            xml={"PMC1": XML_TEMPLATE.format(atype="correction", permissions="", body="<p>x</p>")})
        self.assertEqual(dict(corpus.run()["classifications"]["stub"]),
                         {"correction/notice/editorial-type record": 1})

    def test_correction_recognised_by_title_when_type_unknown(self):
        corpus = self.make([record("PMC1", status="stub", words=40,
                                   title="Correction to: Amyloid staging in AD")])
        self.assertEqual(dict(corpus.run()["classifications"]["stub"]),
                         {"correction/notice/editorial-type record": 1})

    def test_short_research_article_is_legitimate_not_guessed(self):
        corpus = self.make(
            [record("PMC1", status="stub", words=180, refs=12, sections=3,
                    title="A brief report", journal="J", body="Real prose here.")],
            xml={"PMC1": XML_TEMPLATE.format(
                atype="research-article", permissions="", body="<p>x</p>")})
        self.assertEqual(dict(corpus.run()["classifications"]["stub"]),
                         {"likely legitimate short publication": 1})

    def test_insufficient_evidence_goes_to_manual_review(self):
        corpus = self.make([record("PMC1", status="stub", words=120, refs=0,
                                   sections=2, title="Untitled", journal="Unknown")])
        self.assertEqual(dict(corpus.run()["classifications"]["stub"]),
                         {"needs manual review": 1})

    def test_every_classification_is_from_the_agreed_vocabulary(self):
        corpus = self.make([
            record("PMC1", status="stub", words=10, sections=0),
            record("PMC2", status="stub", words=120, refs=0, title="x", journal="y"),
            record("PMC3", status="stub", words=30, journal="medRxiv"),
        ])
        for name in corpus.run()["classifications"]["stub"]:
            self.assertIn(name, qc.STUB_CLASSES)


class NoBodyClassification(CorpusCase):
    def test_no_body_with_no_sections_is_incomplete_xml(self):
        corpus = self.make([record("PMC1", status="no_body", words=0, sections=0)])
        self.assertEqual(dict(corpus.run()["classifications"]["no_body"]),
                         {"incomplete/unusual XML": 1})

    def test_no_body_correction_still_classified_as_notice(self):
        corpus = self.make([record("PMC1", status="no_body", words=0, sections=0,
                                   title="Erratum: an earlier paper")])
        self.assertEqual(dict(corpus.run()["classifications"]["no_body"]),
                         {"correction/notice/editorial-type record": 1})


# ---------------------------------------------------------------------------
# 5-6. Licences
# ---------------------------------------------------------------------------


class LicenceComparison(CorpusCase):
    def test_disagreement_detail_captures_both_sides(self):
        corpus = self.make(
            [record("PMC1", flags=["license_disagreement"],
                    license_manifest="CC BY-NC", license_xml="CC BY")],
            manifest_rows=[{"pmcid": "PMC1", "license_code": "CC BY-NC"}])
        detail = corpus.run()["licence_details"][0]
        self.assertEqual(detail["manifest_license_code"], "CC BY-NC")
        self.assertEqual(detail["xml_derived_code"], "CC BY")
        self.assertEqual(detail["manifest_row_license"], "CC BY-NC")
        self.assertEqual(detail["nature"], "actual licence-content disagreement")

    def test_same_licence_written_differently_is_representation_only(self):
        corpus = self.make([record("PMC1", flags=["license_disagreement"],
                                   license_manifest="CCBY", license_xml="CC BY")])
        self.assertEqual(corpus.run()["licence_details"][0]["nature"],
                         "metadata representation only (same licence family)")

    def test_no_disagreements_when_flag_absent(self):
        corpus = self.make([record("PMC1"), record("PMC2")])
        self.assertEqual(corpus.run()["licence_details"], [])


class LicenceRecoveryProbe(CorpusCase):
    def test_ali_license_ref_is_found_in_permissions(self):
        probe = qc.probe_license_in_xml(
            XML_TEMPLATE.format(atype="research-article", permissions=PERMISSIONS_CC, body=""))
        self.assertTrue(probe["recoverable"])
        self.assertEqual(probe["recovery_source"], "ali:license_ref")
        self.assertEqual(probe["content_type"], "ccbylicense")
        self.assertEqual(probe["copyright_statement"], "(c) 2024 Author")

    def test_text_mining_prose_is_recoverable_as_tdm(self):
        probe = qc.probe_license_in_xml(
            XML_TEMPLATE.format(atype="research-article", permissions=PERMISSIONS_TDM, body=""))
        self.assertTrue(probe["recoverable"])
        self.assertEqual(probe["recovery_source"], "text-mining prose (TDM)")

    def test_copyright_statement_alone_is_not_recoverable(self):
        probe = qc.probe_license_in_xml(XML_TEMPLATE.format(
            atype="research-article", permissions=PERMISSIONS_COPYRIGHT_ONLY, body=""))
        self.assertFalse(probe["recoverable"])
        self.assertEqual(probe["recovery_source"], "copyright statement only")

    def test_absent_permissions_block_reported(self):
        probe = qc.probe_license_in_xml(
            XML_TEMPLATE.format(atype="research-article", permissions="", body=""))
        self.assertFalse(probe["recoverable"])
        self.assertFalse(probe["has_permissions_block"])

    def test_recovery_probe_runs_for_every_flagged_record(self):
        corpus = self.make(
            [record("PMC1", flags=["license_absent_in_xml"]),
             record("PMC2", flags=["license_absent_in_xml"])],
            xml={"PMC1": XML_TEMPLATE.format(atype="research-article",
                                             permissions=PERMISSIONS_TDM, body=""),
                 "PMC2": XML_TEMPLATE.format(atype="research-article",
                                             permissions=PERMISSIONS_COPYRIGHT_ONLY, body="")})
        probes = corpus.run()["absent_probes"]
        self.assertEqual(len(probes), 2)
        self.assertEqual(sum(1 for p in probes if p["recoverable"]), 1)

    def test_license_recovered_from_xml_flag_is_counted(self):
        corpus = self.make([record("PMC1", flags=["license_recovered_from_xml"]),
                            record("PMC2", flags=["license_recovered_from_xml"])])
        self.assertEqual(corpus.run()["flags"]["license_recovered_from_xml"], 2)


# ---------------------------------------------------------------------------
# Parser-defect detection
# ---------------------------------------------------------------------------


class ParserDefectDetection(CorpusCase):
    def test_abstract_present_in_xml_but_missing_from_record_is_a_bug(self):
        corpus = self.make(
            [record("PMC1", status="no_abstract", abstract="")],
            xml={"PMC1": '<article article-type="research-article"><front><article-meta>'
                         "<abstract><p>Real abstract.</p></abstract></article-meta></front>"
                         "<body><p>x</p></body></article>"})
        result = corpus.run()
        self.assertEqual(len(result["contradictions"]), 1)
        self.assertIn("abstract", result["contradictions"][0][1])
        self.assertEqual(result["findings"]["no_abstract"][0]["category"], qc.CAT_BUG)

    def test_doi_present_in_xml_but_missing_from_record_is_a_bug(self):
        corpus = self.make(
            [record("PMC1", doi="", flags=["no_doi"])],
            xml={"PMC1": '<article article-type="research-article"><front><article-meta>'
                         '<article-id pub-id-type="doi">10.1/y</article-id>'
                         "</article-meta></front><body><p>x</p></body></article>"})
        result = corpus.run()
        self.assertEqual(result["findings"]["no_doi"][0]["category"], qc.CAT_BUG)

    def test_sections_present_in_xml_but_none_parsed_is_possible_issue(self):
        corpus = self.make(
            [record("PMC1", flags=["no_sections"], sections=0)],
            xml={"PMC1": '<article article-type="research-article"><front><article-meta>'
                         "</article-meta></front><body><sec><p>x</p></sec></body></article>"})
        result = corpus.run()
        self.assertEqual(result["findings"]["no_sections"][0]["category"], qc.CAT_POSSIBLE)

    def test_no_contradiction_when_xml_genuinely_lacks_the_element(self):
        corpus = self.make(
            [record("PMC1", doi="", flags=["no_doi"])],
            xml={"PMC1": XML_TEMPLATE.format(atype="research-article",
                                             permissions="", body="<p>x</p>")})
        result = corpus.run()
        self.assertEqual(result["contradictions"], [])
        self.assertEqual(result["findings"]["no_doi"][0]["category"], qc.CAT_LEGIT)

    def test_no_xml_mode_produces_no_false_contradictions(self):
        corpus = self.make(
            [record("PMC1", status="no_abstract", abstract="")],
            xml={"PMC1": '<article><front><article-meta><abstract><p>a</p></abstract>'
                         "</article-meta></front><body><p>x</p></body></article>"})
        self.assertEqual(corpus.run(xml=False)["contradictions"], [])

    def test_categories_come_from_the_agreed_taxonomy(self):
        corpus = self.make([
            record("PMC1", status="stub", words=10, sections=0),
            record("PMC2", flags=["no_doi"], doi=""),
            record("PMC3", flags=["license_disagreement"]),
        ])
        valid = {qc.CAT_LEGIT, qc.CAT_UNUSUAL, qc.CAT_POSSIBLE, qc.CAT_BUG, qc.CAT_REVIEW}
        for group, counts in corpus.run()["categories"].items():
            for name in counts:
                self.assertIn(name, valid, group)


# ---------------------------------------------------------------------------
# 7. Aggregates
# ---------------------------------------------------------------------------


class Aggregates(CorpusCase):
    def test_aggregates_by_type_journal_and_year(self):
        corpus = self.make(
            [record("PMC1", status="stub", words=10, journal="J One", date_str="2023-01-02"),
             record("PMC2", status="stub", words=12, journal="J One", date_str="2023-07-08"),
             record("PMC3", status="stub", words=14, journal="J Two", date_str="2024-02-03")],
            xml={"PMC1": XML_TEMPLATE.format(atype="correction", permissions="", body=""),
                 "PMC2": XML_TEMPLATE.format(atype="correction", permissions="", body=""),
                 "PMC3": XML_TEMPLATE.format(atype="editorial", permissions="", body="")})
        agg = corpus.run()["aggregates"]["stub"]
        self.assertEqual(dict(agg["journal"]), {"J One": 2, "J Two": 1})
        self.assertEqual(dict(agg["year"]), {"2023": 2, "2024": 1})
        self.assertEqual(dict(agg["article-type"]), {"correction": 2, "editorial": 1})

    def test_missing_date_aggregates_as_unknown(self):
        corpus = self.make([record("PMC1", status="stub", words=10, date_str="")])
        self.assertEqual(dict(corpus.run()["aggregates"]["stub"]["year"]), {"(unknown)": 1})

    def test_aggregates_cover_no_abstract_group(self):
        corpus = self.make([record("PMC1", status="no_abstract", abstract="",
                                   journal="J", date_str="2022-03-04")])
        self.assertEqual(dict(corpus.run()["aggregates"]["no_abstract"]["year"]), {"2022": 1})


# ---------------------------------------------------------------------------
# 8-9. Report generation and --max-rows
# ---------------------------------------------------------------------------


class ReportGeneration(CorpusCase):
    def test_report_contains_every_required_section(self):
        corpus = self.make([record("PMC1", status="stub", words=10,
                                   flags=["license_disagreement"])])
        text = qc.build_report(corpus.run(), corpus.jsonl, True, 250)
        for heading in ["# PMC corpus QC investigation", "## QC status counts",
                        "## QC flag counts", "## Parser-defect check",
                        "## Group: `stub`", "## Licence disagreements — detail",
                        "## `license_absent_in_xml` — recoverability probe",
                        "## Licence recovery from XML", "## Aggregates"]:
            self.assertIn(heading, text)

    def test_pipe_characters_in_titles_do_not_break_tables(self):
        corpus = self.make([record("PMC1", status="stub", words=10,
                                   title="Weird | title | here")])
        self.assertIn(r"Weird \| title \| here",
                      qc.build_report(corpus.run(), corpus.jsonl, True, 250))

    def test_report_states_when_no_contradictions_found(self):
        corpus = self.make([record("PMC1", flags=["no_doi"], doi="")])
        text = qc.build_report(corpus.run(), corpus.jsonl, True, 250)
        self.assertIn("No contradictions found", text)
        self.assertIn("No evidence", text)


class MaxRowsTruncatesDisplayOnly(CorpusCase):
    def setUp(self):
        self.corpus = self.make([
            record(f"PMC{i}", status="stub", words=10 + i, journal=f"J{i % 3}")
            for i in range(1, 21)
        ])
        self.result = self.corpus.run()

    def test_all_records_are_analysed_regardless_of_max_rows(self):
        self.assertEqual(self.result["statuses"]["stub"], 20)
        self.assertEqual(len(self.result["findings"]["stub"]), 20)
        self.assertEqual(sum(self.result["classifications"]["stub"].values()), 20)

    def test_counts_in_report_are_full_while_rows_are_truncated(self):
        text = qc.build_report(self.result, self.corpus.jsonl, True, 5)
        self.assertIn("20 record(s)", text)
        self.assertIn("Showing 5 of 20 rows", text)
        self.assertIn("all 20 were", text)
        rendered = sum(1 for line in text.splitlines() if line.startswith("| PMC"))
        self.assertEqual(rendered, 5)

    def test_larger_max_rows_renders_everything(self):
        text = qc.build_report(self.result, self.corpus.jsonl, True, 250)
        self.assertNotIn("Showing", text)
        self.assertEqual(sum(1 for line in text.splitlines() if line.startswith("| PMC")), 20)

    def test_aggregate_totals_are_unaffected_by_max_rows(self):
        small = qc.build_report(self.result, self.corpus.jsonl, True, 2)
        self.assertIn("| stub | 20 |", small.replace("| 20 | ", "| 20 |"))


# ---------------------------------------------------------------------------
# 10-11. Output protection and safeguards
# ---------------------------------------------------------------------------


class OutputProtection(CorpusCase):
    def setUp(self):
        self.corpus = self.make([record("PMC1")])

    def _argv(self, output: Path, *extra: str) -> list[str]:
        return ["--jsonl", str(self.corpus.jsonl), "--manifest", str(self.corpus.manifest),
                "--xml-dir", str(self.corpus.xml_dir), "--output", str(output), *extra]

    def test_existing_report_is_not_overwritten_without_force(self):
        self.corpus.output.write_text("ORIGINAL", encoding="utf-8")
        with self.assertRaises(SystemExit):
            qc.main(self._argv(self.corpus.output))
        self.assertEqual(self.corpus.output.read_text(encoding="utf-8"), "ORIGINAL")

    def test_force_allows_overwrite(self):
        self.corpus.output.write_text("ORIGINAL", encoding="utf-8")
        self.assertEqual(qc.main(self._argv(self.corpus.output, "--force")), 0)
        self.assertIn("QC investigation", self.corpus.output.read_text(encoding="utf-8"))

    def test_fresh_output_is_written(self):
        self.assertEqual(qc.main(self._argv(self.corpus.output)), 0)
        self.assertTrue(self.corpus.output.exists())

    def test_missing_corpus_fails_clearly(self):
        with self.assertRaises(SystemExit) as caught:
            qc.main(["--jsonl", str(self.corpus.jsonl.parent / "absent.jsonl"),
                     "--output", str(self.corpus.output)])
        self.assertIn("not found", str(caught.exception))

    def test_missing_xml_dir_fails_unless_no_xml(self):
        argv = ["--jsonl", str(self.corpus.jsonl), "--manifest", str(self.corpus.manifest),
                "--xml-dir", str(self.corpus.jsonl.parent / "absent"),
                "--output", str(self.corpus.output)]
        with self.assertRaises(SystemExit):
            qc.main(argv)
        self.assertEqual(qc.main(argv + ["--no-xml"]), 0)

    def test_zero_max_rows_rejected(self):
        with self.assertRaises(SystemExit):
            qc.main(self._argv(self.corpus.output, "--max-rows", "0"))


class ProtectedFileSafeguards(unittest.TestCase):
    def test_corpus_and_source_files_are_refused_as_output(self):
        for name in ["articles.jsonl", "manifest.csv", ".gitignore", "retry71.csv",
                     "parse_pmc_xml.py", "pmc_oa_inventory.csv", "qc_investigate.py",
                     "pubmed_results.csv", "search_queries.txt"]:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                qc.assert_writable(qc.REPO_ROOT / "pmc" / name)

    def test_raw_xml_is_refused_as_output(self):
        with self.assertRaises(SystemExit):
            qc.assert_writable(qc.REPO_ROOT / "pmc" / "anything.xml")

    def test_writing_inside_the_xml_directory_is_refused(self):
        with self.assertRaises(SystemExit):
            qc.assert_writable(qc.DEFAULT_XML_DIR / "report.md")

    def test_writing_under_pubmed_is_refused(self):
        with self.assertRaises(SystemExit):
            qc.assert_writable(qc.REPO_ROOT / "pubmed" / "report.md")

    def test_ordinary_report_path_is_allowed(self):
        qc.assert_writable(qc.REPO_ROOT / "pmc" / "pmc_qc_report_2026-09-02.md")


# ---------------------------------------------------------------------------
# 12. Malformed and missing metadata
# ---------------------------------------------------------------------------


class MalformedAndMissingMetadata(CorpusCase):
    def test_malformed_json_line_is_warned_about_not_fatal(self):
        corpus = self.make([record("PMC1"), record("PMC2")])
        corpus.jsonl.write_text(
            corpus.jsonl.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
        self.assertEqual(corpus.run()["total"], 2)

    def test_missing_fields_render_as_placeholders(self):
        corpus = self.make([record("PMC1", pmid="", doi="", journal="",
                                   date_str="", date_type="", flags=["no_doi"])])
        row = corpus.run()["findings"]["no_doi"][0]
        self.assertEqual(row["pmid"], "-")
        self.assertEqual(row["doi"], "-")
        self.assertEqual(row["journal"], "-")
        self.assertEqual(row["date"], "-")
        self.assertEqual(row["date_type"], "-")

    def test_absent_xml_file_does_not_crash_the_run(self):
        corpus = self.make([record("PMC1", flags=["no_doi"], doi="")])
        result = corpus.run()
        self.assertEqual(result["findings"]["no_doi"][0]["type"], "(unknown)")
        self.assertEqual(result["contradictions"], [])

    def test_missing_manifest_is_tolerated(self):
        corpus = self.make([record("PMC1")])
        result = qc.investigate(corpus.jsonl, corpus.manifest.parent / "absent.csv",
                                corpus.xml_dir)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["manifest_rows"], 0)

    def test_parse_error_records_are_counted_and_grouped(self):
        broken = record("PMC1", status="parse_error")
        broken["qc"]["flags"] = ["xml_parse_error"]
        corpus = self.make([broken])
        result = corpus.run()
        self.assertEqual(result["statuses"]["parse_error"], 1)
        self.assertEqual(len(result["findings"]["parse_error"]), 1)

    def test_record_without_sections_key_does_not_crash(self):
        stripped = record("PMC1", status="stub", words=10)
        del stripped["sections"]
        corpus = self.make([stripped])
        self.assertEqual(corpus.run()["statuses"]["stub"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
