#!/usr/bin/env python3
"""Offline tests for parse_pmc_xml.py.

Every fixture is synthetic JATS built in-memory. No file in pmc/fulltext/xml/
is read, and nothing touches the network.

    python3 -m unittest pmc.test_parse_pmc_xml -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import parse_pmc_xml as pp


def article(front: str = "", body: str = "", back: str = "", floats: str = "") -> str:
    return (
        '<article article-type="research-article" dtd-version="1.4">'
        f"<front><journal-meta><journal-title-group><journal-title>Test Journal"
        f"</journal-title></journal-title-group></journal-meta>"
        f"<article-meta>{front}</article-meta></front>"
        f"<body>{body}</body>"
        + (f"<back>{back}</back>" if back else "")
        + (f"<floats-group>{floats}</floats-group>" if floats else "")
        + "</article>"
    )


IDS = (
    '<article-id pub-id-type="pmcid">PMC123</article-id>'
    '<article-id pub-id-type="pmcid-ver">PMC123.1</article-id>'
    '<article-id pub-id-type="pmid">999</article-id>'
    '<article-id pub-id-type="doi">10.1/xyz</article-id>'
)
TITLE = "<title-group><article-title>A Study</article-title></title-group>"
ABSTRACT = "<abstract><p>An abstract with enough words.</p></abstract>"


def long_body(words: int = 400) -> str:
    return f"<sec><title>Introduction</title><p>{' '.join(['word'] * words)}</p></sec>"


def parse(xml: str, row: dict[str, str] | None = None, threshold: int = 250) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "PMC123.xml"
        path.write_text(xml, encoding="utf-8")
        return pp.parse_article(path, row or {}, threshold)


class TextFlattening(unittest.TestCase):
    def test_bibliographic_xref_becomes_ref_placeholder(self):
        el = ET.fromstring('<p>Shown before<xref ref-type="bibr" rid="b1">12</xref> here.</p>')
        self.assertEqual(pp.flat_text(el), "Shown before[REF] here.")

    def test_non_bibliographic_xref_keeps_its_text(self):
        el = ET.fromstring('<p>See <xref ref-type="fig" rid="f1">Fig. 1</xref>.</p>')
        self.assertEqual(pp.flat_text(el), "See Fig. 1.")

    def test_figure_caption_is_never_spliced_into_prose(self):
        el = ET.fromstring(
            "<p>Before<fig><caption><p>CAPTION TEXT</p></caption></fig>after.</p>"
        )
        text = pp.flat_text(el)
        self.assertNotIn("CAPTION", text)
        self.assertEqual(text, "Beforeafter.")

    def test_inline_markup_is_unwrapped_and_counted(self):
        from collections import Counter
        stripped: Counter = Counter()
        el = ET.fromstring("<p>The <italic>APOE</italic> gene<sup>4</sup>.</p>")
        self.assertEqual(pp.flat_text(el, stripped), "The APOE gene4.")
        self.assertEqual(stripped["italic"], 1)
        self.assertEqual(stripped["sup"], 1)

    def test_tail_text_after_skipped_element_is_kept(self):
        el = ET.fromstring("<p>a<table-wrap><label>T1</label></table-wrap>b</p>")
        self.assertEqual(pp.flat_text(el), "ab")


class Sections(unittest.TestCase):
    def test_nested_sections_are_recursive_with_paths(self):
        body = (
            "<sec><title>Methods</title><p>Top level.</p>"
            "<sec><title>Participants</title><p>Nested one.</p>"
            "<sec><title>Screening</title><p>Nested two.</p></sec></sec></sec>"
        )
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        top = record["sections"][0]
        self.assertEqual(top["depth"], 1)
        self.assertEqual(top["subsections"][0]["depth"], 2)
        deepest = top["subsections"][0]["subsections"][0]
        self.assertEqual(deepest["depth"], 3)
        self.assertEqual(deepest["section_id"], "PMC123.1#s1.1.1")
        self.assertEqual(record["section_count"], 3)

    def test_paragraph_id_uses_versioned_pmcid_and_path_without_md5(self):
        body = "<sec><title>Methods</title><sec><title>Sub</title><p>a</p><p>b</p></sec></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        sub = record["sections"][0]["subsections"][0]
        self.assertEqual(sub["paragraphs"][1]["paragraph_id"], "PMC123.1#s1.1.p2")
        for paragraph in sub["paragraphs"]:
            self.assertNotIn("#s1.1.p1:", paragraph["paragraph_id"])
            self.assertEqual(paragraph["paragraph_id"].count("#"), 1)

    def test_classification_prefers_title_over_sec_type(self):
        body = '<sec sec-type="results"><title>Materials and Methods</title><p>x</p></sec>'
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        section = record["sections"][0]
        self.assertEqual(section["imrad"], "methods")
        self.assertEqual(section["imrad_source"], "title_match")
        self.assertEqual(section["sec_type_attr"], "results")

    def test_sec_type_used_when_title_is_unrecognised(self):
        body = '<sec sec-type="intro"><title>Setting the Scene</title><p>x</p></sec>'
        section = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)["sections"][0]
        self.assertEqual(section["imrad"], "introduction")
        self.assertEqual(section["imrad_source"], "sec_type")

    def test_supplementary_and_coi_sections_marked_not_content(self):
        body = (
            "<sec><title>Introduction</title><p>x</p></sec>"
            '<sec sec-type="supplementary-material"><title>Supplementary Material</title><p>y</p></sec>'
            "<sec><title>CONFLICT OF INTEREST STATEMENT</title><p>z</p></sec>"
        )
        sections = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)["sections"]
        self.assertTrue(sections[0]["is_content"])
        self.assertFalse(sections[1]["is_content"])
        self.assertFalse(sections[2]["is_content"])

    def test_section_title_path_is_denormalised_onto_paragraphs(self):
        body = "<sec><title>Methods</title><sec><title>Participants</title><p>x</p></sec></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        paragraph = record["sections"][0]["subsections"][0]["paragraphs"][0]
        self.assertEqual(paragraph["section_title_path"], ["Methods", "Participants"])

    def test_paragraphs_directly_on_body_are_captured_and_flagged(self):
        record = parse(article(IDS + TITLE + ABSTRACT, "<p>Loose paragraph.</p>"), threshold=0)
        self.assertIn("flat_sections", record["qc"]["flags"])
        self.assertEqual(record["paragraph_count"], 1)


class Authors(unittest.TestCase):
    def test_editors_are_excluded_and_flagged(self):
        front = IDS + TITLE + ABSTRACT + (
            '<contrib-group><contrib contrib-type="editor"><name>'
            "<surname>Muacevic</surname><given-names>Alexander</given-names></name></contrib>"
            "</contrib-group>"
            '<contrib-group><contrib contrib-type="author"><name>'
            "<surname>Charron</surname><given-names>Lily</given-names></name></contrib>"
            "</contrib-group>"
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["author_count"], 1)
        self.assertEqual(record["authors"][0]["surname"], "Charron")
        self.assertIn("editors_present", record["qc"]["flags"])

    def test_collab_authors_are_supported(self):
        front = IDS + TITLE + ABSTRACT + (
            '<contrib-group><contrib contrib-type="author">'
            "<collab>The ADNI Consortium</collab></contrib></contrib-group>"
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["authors"][0]["collab"], "The ADNI Consortium")
        self.assertEqual(record["authors"][0]["surname"], "")

    def test_affiliation_links_and_orcid_preserved(self):
        front = IDS + TITLE + ABSTRACT + (
            '<contrib-group><contrib contrib-type="author">'
            '<contrib-id contrib-id-type="orcid">https://orcid.org/0000-0003-2851-9763</contrib-id>'
            "<name><surname>Counts</surname><given-names>Scott</given-names></name>"
            '<xref ref-type="aff" rid="Aff1"/><xref ref-type="aff" rid="Aff2"/>'
            "</contrib></contrib-group>"
            '<aff id="Aff1"><label>1</label>Michigan State University</aff>'
            '<aff id="Aff2">Second Institute</aff>'
        )
        record = parse(article(front, long_body()))
        author = record["authors"][0]
        self.assertEqual(author["affiliation_ids"], ["Aff1", "Aff2"])
        self.assertEqual(author["orcid"], "0000-0003-2851-9763")
        self.assertEqual(record["affiliations"][0]["text"], "Michigan State University")


class Abstracts(unittest.TestCase):
    def test_scientific_abstract_chosen_over_plain_language_summary(self):
        front = IDS + TITLE + (
            "<abstract><sec><title>Background</title><p>Real background.</p></sec>"
            "<sec><title>Methods</title><p>Real methods.</p></sec></abstract>"
            '<abstract abstract-type="plain-language-summary"><p>Lay summary.</p></abstract>'
        )
        record = parse(article(front, long_body()))
        self.assertIn("Real background.", record["abstract"]["text"])
        self.assertNotIn("Lay summary", record["abstract"]["text"])
        self.assertTrue(record["abstract"]["is_structured"])
        self.assertEqual(record["abstract"]["sections"][0]["label"], "Background")
        self.assertEqual(record["abstract_other"][0]["type"], "plain-language-summary")
        self.assertIn("multiple_abstracts", record["qc"]["flags"])

    def test_unstructured_abstract(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()))
        self.assertFalse(record["abstract"]["is_structured"])
        self.assertEqual(record["abstract"]["sections"], [])


class Dates(unittest.TestCase):
    def test_epub_beats_ppub_and_precision_recorded(self):
        front = IDS + TITLE + ABSTRACT + (
            '<pub-date pub-type="ppub"><year>2025</year><month>1</month></pub-date>'
            '<pub-date pub-type="epub"><year>2024</year><month>11</month><day>7</day></pub-date>'
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["publication_date"], "2024-11-07")
        self.assertEqual(record["publication_date_type"], "epub")
        self.assertEqual(record["publication_date_precision"], "day")
        self.assertEqual(record["publication_dates_all"]["ppub"], "2025-01")

    def test_preprint_date_never_wins_over_pub(self):
        front = IDS + TITLE + ABSTRACT + (
            '<pub-date pub-type="preprint"><year>2018</year><month>12</month><day>20</day></pub-date>'
            '<pub-date pub-type="pub"><year>2022</year><month>5</month><day>3</day></pub-date>'
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["publication_date"], "2022-05-03")
        self.assertEqual(record["publication_date_type"], "pub")

    def test_year_only_date_flagged_partial(self):
        front = IDS + TITLE + ABSTRACT + '<pub-date pub-type="collection"><year>2022</year></pub-date>'
        record = parse(article(front, long_body()))
        self.assertEqual(record["publication_date"], "2022")
        self.assertEqual(record["publication_date_precision"], "year")
        self.assertIn("partial_date", record["qc"]["flags"])

    def test_named_month_is_understood(self):
        front = IDS + TITLE + ABSTRACT + (
            '<pub-date pub-type="epub"><year>2023</year><month>Mar</month><day>4</day></pub-date>'
        )
        self.assertEqual(parse(article(front, long_body()))["publication_date"], "2023-03-04")


class Licences(unittest.TestCase):
    def test_ali_license_ref_is_read(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' specific-use="textmining" content-type="ccbynclicense">'
            "https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref>"
            "<license-p>Open access under CC BY-NC.</license-p></license></permissions>"
        )
        prov = parse(article(front, long_body()), {"license_code": "CC BY-NC"})["provenance"]
        self.assertEqual(prov["license_code_xml"], "CC BY-NC")
        self.assertEqual(prov["license_content_type_xml"], "ccbynclicense")
        self.assertIn("creativecommons.org", prov["license_ref_xml"])

    def test_nc_nd_not_confused_with_nc(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbyncndlicense">https://creativecommons.org/licenses/by-nc-nd/4.0/'
            "</ali:license_ref></license></permissions>"
        )
        self.assertEqual(
            parse(article(front, long_body()))["provenance"]["license_code_xml"], "CC BY-NC-ND")

    def test_text_mining_prose_classified_as_tdm(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license><license-p>This file is available for text mining."
            "</license-p></license></permissions>"
        )
        self.assertEqual(
            parse(article(front, long_body()))["provenance"]["license_code_xml"], "TDM")

    def test_manifest_code_kept_separately_and_disagreement_flagged(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/'
            "</ali:license_ref></license></permissions>"
        )
        record = parse(article(front, long_body()), {"license_code": "CC BY-NC"})
        self.assertEqual(record["provenance"]["license_code_manifest"], "CC BY-NC")
        self.assertEqual(record["provenance"]["license_code_xml"], "CC BY")
        self.assertIn("license_disagreement", record["qc"]["flags"])

    def test_blank_manifest_licence_recovered_from_xml_is_flagged(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/'
            "</ali:license_ref></license></permissions>"
        )
        record = parse(article(front, long_body()), {"license_code": ""})
        self.assertIn("license_recovered_from_xml", record["qc"]["flags"])


class FiguresTablesAndFloats(unittest.TestCase):
    def test_floats_group_figures_and_tables_are_counted(self):
        floats = "<fig id='f1'/><fig id='f2'/><table-wrap id='t1'/>"
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(), floats=floats))
        self.assertEqual(record["figure_count"], 2)
        self.assertEqual(record["table_count"], 1)
        self.assertIn("floats_group_present", record["qc"]["flags"])

    def test_inline_body_figures_are_counted(self):
        body = "<sec><title>Results</title><p>text</p><fig id='f1'/><table-wrap id='t1'/></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        self.assertEqual(record["figure_count"], 1)
        self.assertEqual(record["table_count"], 1)

    def test_references_counted_from_back(self):
        back = "<ref-list><ref id='r1'/><ref id='r2'/><ref id='r3'/></ref-list>"
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(), back=back))
        self.assertEqual(record["reference_count"], 3)
        self.assertNotIn("no_references", record["qc"]["flags"])


class QualityControl(unittest.TestCase):
    def test_short_body_is_stub_but_record_is_kept_whole(self):
        body = "<sec><title>Full Text Availability</title><p>The license terms do not permit archiving.</p></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body))
        self.assertEqual(record["qc"]["status"], "stub")
        self.assertEqual(record["pmcid"], "PMC123")
        self.assertEqual(record["title"], "A Study")
        self.assertTrue(record["sections"], "stub records must keep their sections")

    def test_body_just_above_threshold_is_ok(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(260)))
        self.assertEqual(record["qc"]["status"], "ok")

    def test_missing_abstract_reported(self):
        record = parse(article(IDS + TITLE, long_body()))
        self.assertEqual(record["qc"]["status"], "no_abstract")

    def test_malformed_xml_recorded_not_raised(self):
        record = parse("<article><front>unclosed")
        self.assertEqual(record["qc"]["status"], "parse_error")
        self.assertIn("xml_parse_error", record["qc"]["flags"])
        self.assertEqual(record["pmcid"], "PMC123")

    def test_missing_identifiers_flagged(self):
        front = '<article-id pub-id-type="pmcid">PMC123</article-id>' + TITLE + ABSTRACT
        record = parse(article(front, long_body()))
        self.assertIn("no_pmid", record["qc"]["flags"])
        self.assertIn("no_doi", record["qc"]["flags"])


class Provenance(unittest.TestCase):
    def test_manifest_fields_are_joined_through(self):
        row = {
            "actual_md5": "abc123", "bytes": "107631", "version": "1",
            "is_manuscript": "no", "is_retracted": "yes", "license_code": "CC BY",
            "status": "ok", "downloaded_at_utc": "2026-09-01T22:14:03+00:00",
            "source_xml_url": "s3://pmc-oa-opendata/PMC123.1/PMC123.1.xml?md5=abc123",
        }
        prov = parse(article(IDS + TITLE + ABSTRACT, long_body()), row)["provenance"]
        self.assertEqual(prov["xml_md5"], "abc123")
        self.assertEqual(prov["pmc_version"], "1")
        self.assertEqual(prov["is_retracted"], "yes")
        self.assertEqual(prov["license_code_manifest"], "CC BY")
        self.assertEqual(prov["downloaded_at_utc"], "2026-09-01T22:14:03+00:00")

    def test_retracted_article_is_kept_not_dropped(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()), {"is_retracted": "yes"})
        self.assertEqual(record["qc"]["status"], "ok")
        self.assertEqual(record["provenance"]["is_retracted"], "yes")

    def test_parser_version_and_timestamp_present(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()))
        self.assertEqual(record["qc"]["parser_version"], pp.PARSER_VERSION)
        self.assertTrue(record["qc"]["parsed_at_utc"].endswith("+00:00"))
        self.assertEqual(record["schema_version"], pp.SCHEMA_VERSION)


class OutputSafety(unittest.TestCase):
    def test_refuses_to_write_under_pubmed(self):
        with self.assertRaises(SystemExit):
            pp.assert_safe_output(pp.REPO_ROOT / "pubmed" / "x.jsonl", pp.DEFAULT_XML_DIR)

    def test_refuses_to_write_inside_the_xml_directory(self):
        with self.assertRaises(SystemExit):
            pp.assert_safe_output(pp.DEFAULT_XML_DIR / "out.jsonl", pp.DEFAULT_XML_DIR)

    def test_record_is_json_serialisable(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()))
        self.assertIsInstance(json.dumps(record, ensure_ascii=False), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
