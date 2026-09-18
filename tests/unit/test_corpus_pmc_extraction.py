"""The PMC JATS XML -> normalize-ready record bridge (_common.py additions).

Stage 04 previously had no way to consume the real corpus: Stage 01 (PubMed)
produces PMIDs only, and Stage 02 (PMC) produces a manifest plus raw XML/JSON
files, never the flat title/abstract JSONL Stage 04 expected. This module's
functions - ``parse_jats_xml``, ``read_pmc_manifest``, ``iter_pmc_records`` -
are that bridge, and these tests are what stands in for "run it against
114,000 real files": synthetic JATS XML built to the same structure PMC's
Open Access XML actually uses, run through the real parser.

Nothing here reads or writes the real, already-finalized
``alzheimer_corpus/metadata/pmc.csv`` except the one test that explicitly
says so and only reads it - never writes.
"""

import csv
import importlib.util
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "alzheimer_corpus" / "scripts" / "_common.py"
REAL_MANIFEST = ROOT / "alzheimer_corpus" / "metadata" / "pmc.csv"


def load_module():
    # _common.py declares @dataclass classes under `from __future__ import
    # annotations`, so dataclass needs to resolve string annotations via
    # sys.modules[cls.__module__] during class creation - the module must be
    # registered there before exec_module runs, not just after.
    spec = importlib.util.spec_from_file_location("corpus_common", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quiet_logger(name="corpus_extraction_test"):
    log = logging.getLogger(name)
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


ARTICLE_XML = (
    b'<article xmlns:xlink="http://www.w3.org/1999/xlink">'
    b'<front><article-meta>'
    b'<article-id pub-id-type="pmid">12345678</article-id>'
    b'<title-group><article-title>Plasma p-tau217 and A\xce\xb242/40 in '
    b'Alzheimer disease</article-title></title-group>'
    b'<pub-date pub-type="ppub"><year>2024</year></pub-date>'
    b'<pub-date pub-type="epub"><year>2024</year><month>8</month><day>15</day></pub-date>'
    b'<abstract><p>Background text.</p><p>Second paragraph.</p></abstract>'
    b'</article-meta></front>'
    b'<body>'
    b'<sec><title>Introduction</title><p>Intro <italic>text</italic> with markup.</p></sec>'
    b'<sec><title>Methods</title><p>Methods text here.</p></sec>'
    b'</body>'
    b'<back><ref-list><ref>Must not appear in body text.</ref></ref-list></back>'
    b'</article>'
)


class ParseJATSXMLTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()

    def test_extracts_title_abstract_dates_and_sections(self):
        article = self.c.parse_jats_xml(ARTICLE_XML)
        self.assertIn("p-tau217", article.title)
        self.assertIn("Aβ42/40", article.title)
        self.assertEqual(article.abstract,
                         "Background text.\n\nSecond paragraph.")
        self.assertEqual(len(article.sections), 2)
        self.assertEqual(article.sections[0]["section"], "Introduction")

    def test_prefers_epub_date_over_print_date(self):
        """PUB_DATE_TYPE_PRIORITY: epub before ppub."""
        article = self.c.parse_jats_xml(ARTICLE_XML)
        self.assertEqual(article.publication_date, "2024-08-15")
        self.assertEqual(article.date_precision, "day")

    def test_inline_markup_is_stripped_but_text_kept(self):
        article = self.c.parse_jats_xml(ARTICLE_XML)
        self.assertIn("Intro text with markup", article.sections[0]["text"])
        self.assertNotIn("<italic>", article.sections[0]["text"])

    def test_reference_list_is_excluded_from_body_text(self):
        article = self.c.parse_jats_xml(ARTICLE_XML)
        joined = " ".join(s["text"] for s in article.sections)
        self.assertNotIn("Must not appear", joined)

    def test_malformed_xml_raises_not_silently_skips(self):
        with self.assertRaises(self.c.CorpusPipelineError):
            self.c.parse_jats_xml(b"<article><unclosed>")

    def test_missing_article_meta_raises(self):
        with self.assertRaises(self.c.CorpusPipelineError):
            self.c.parse_jats_xml(b"<article><front></front></article>")

    def test_short_article_with_no_sec_wrapping_still_extracts_body(self):
        """Editorials/short pieces put paragraphs directly under <body>."""
        xml = (b"<article><front><article-meta>"
              b"<title-group><article-title>Editorial</article-title></title-group>"
              b"</article-meta></front>"
              b"<body><p>Direct paragraph one.</p><p>Direct paragraph two.</p></body>"
              b"</article>")
        article = self.c.parse_jats_xml(xml)
        self.assertEqual(len(article.sections), 1)
        self.assertEqual(article.sections[0]["section"], "body")
        self.assertIn("Direct paragraph one", article.sections[0]["text"])

    def test_no_pub_date_yields_empty_not_fabricated(self):
        xml = (b"<article><front><article-meta>"
              b"<title-group><article-title>No date</article-title></title-group>"
              b"</article-meta></front></article>")
        article = self.c.parse_jats_xml(xml)
        self.assertEqual(article.publication_date, "")
        self.assertEqual(article.date_precision, "")

    def test_year_only_precision_is_recorded_as_year(self):
        xml = (b"<article><front><article-meta>"
              b"<title-group><article-title>Year only</article-title></title-group>"
              b"<pub-date><year>2019</year></pub-date>"
              b"</article-meta></front></article>")
        article = self.c.parse_jats_xml(xml)
        self.assertEqual(article.publication_date, "2019")
        self.assertEqual(article.date_precision, "year")

    def test_year_month_no_day_precision_is_month(self):
        xml = (b"<article><front><article-meta>"
              b"<title-group><article-title>t</article-title></title-group>"
              b"<pub-date><year>2019</year><month>6</month></pub-date>"
              b"</article-meta></front></article>")
        article = self.c.parse_jats_xml(xml)
        self.assertEqual(article.publication_date, "2019-06")
        self.assertEqual(article.date_precision, "month")


def write_manifest_row(handle_path, fieldnames, rows):
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    with handle_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ReadPMCManifestTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()

    def test_missing_manifest_raises_with_the_fix_instruction(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(self.c.CorpusPipelineError) as ctx:
                self.c.read_pmc_manifest(Path(tmp) / "nope.csv")
            self.assertIn("--finalize", str(ctx.exception))

    def test_a_schema_mismatch_is_refused_not_silently_misread(self):
        """The exact failure class ledger D-41 diagnosed, guarded here too."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pmc.csv"
            path.write_text("wrong,header\n1,2\n", encoding="utf-8")
            with self.assertRaises(self.c.CorpusPipelineError) as ctx:
                self.c.read_pmc_manifest(path)
            self.assertIn("header", str(ctx.exception))

    def test_reads_rows_matching_the_declared_schema(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pmc.csv"
            write_manifest_row(path, self.c.PMC_MANIFEST_FIELDS, [{
                "pmid": "1", "pmcid": "PMC1", "version": "1", "doi": "",
                "title": "t", "citation": "", "is_pmc_openaccess": "True",
                "is_manuscript": "False", "license_code": "CC BY",
                "is_retracted": "False", "json_path": "", "xml_path": "",
                "status": "already_verified",
            }])
            rows = self.c.read_pmc_manifest(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pmcid"], "PMC1")

    def test_the_real_committed_manifest_matches_the_declared_schema(self):
        """Read-only check against the actual finalized corpus manifest."""
        if not REAL_MANIFEST.exists():
            self.skipTest("real PMC manifest not present in this environment")
        rows = self.c.read_pmc_manifest(REAL_MANIFEST)
        self.assertEqual(len(rows), 114256)


class IterPMCRecordsTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()
        self.tmpdir = TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.addCleanup(self.tmpdir.cleanup)

    def _xml_dir(self, pmcid, version):
        d = self.root / "data" / "raw" / "pmc" / f"{pmcid}.{version}"
        d.mkdir(parents=True)
        return d

    def test_yields_one_record_per_verified_row_with_the_normalize_shape(self):
        xml_dir = self._xml_dir("PMC1000001", "1")
        (xml_dir / "PMC1000001.1.xml").write_bytes(ARTICLE_XML)
        manifest = self.root / "metadata" / "pmc.csv"
        write_manifest_row(manifest, self.c.PMC_MANIFEST_FIELDS, [{
            "pmid": "111", "pmcid": "PMC1000001", "version": "1", "doi": "",
            "title": "Manifest title", "citation": "", "is_pmc_openaccess": "True",
            "is_manuscript": "False", "license_code": "CC BY",
            "is_retracted": "False",
            "json_path": "data/raw/pmc/PMC1000001.1/PMC1000001.1.json",
            "xml_path": "data/raw/pmc/PMC1000001.1/PMC1000001.1.xml",
            "status": "already_verified",
        }])
        records = list(self.c.iter_pmc_records(manifest, self.root, quiet_logger()))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["document_id"], "PMC1000001.1")
        # XML title is preferred over the manifest's when both are present.
        self.assertIn("p-tau217", record["title"])
        self.assertEqual(record["pmid"], "111")
        self.assertEqual(record["retracted"], False)
        self.assertIsInstance(record["retracted"], bool)
        for field in ("document_id", "pmid", "pmcid", "doi", "title",
                     "abstract", "sections", "mesh_terms", "publication_date",
                     "license", "source_tier", "retracted"):
            self.assertIn(field, record)

    def test_a_broken_xml_file_is_skipped_not_fatal(self):
        good_dir = self._xml_dir("PMC1000001", "1")
        (good_dir / "PMC1000001.1.xml").write_bytes(ARTICLE_XML)
        bad_dir = self._xml_dir("PMC1000002", "1")
        (bad_dir / "PMC1000002.1.xml").write_bytes(b"<not><valid")

        manifest = self.root / "metadata" / "pmc.csv"
        write_manifest_row(manifest, self.c.PMC_MANIFEST_FIELDS, [
            {"pmid": "1", "pmcid": "PMC1000001", "version": "1", "doi": "",
             "title": "Good", "citation": "", "is_pmc_openaccess": "True",
             "is_manuscript": "False", "license_code": "CC BY",
             "is_retracted": "False",
             "json_path": "data/raw/pmc/PMC1000001.1/PMC1000001.1.json",
             "xml_path": "data/raw/pmc/PMC1000001.1/PMC1000001.1.xml",
             "status": "already_verified"},
            {"pmid": "2", "pmcid": "PMC1000002", "version": "1", "doi": "",
             "title": "Broken", "citation": "", "is_pmc_openaccess": "True",
             "is_manuscript": "False", "license_code": "CC BY",
             "is_retracted": "False",
             "json_path": "data/raw/pmc/PMC1000002.1/PMC1000002.1.json",
             "xml_path": "data/raw/pmc/PMC1000002.1/PMC1000002.1.xml",
             "status": "already_verified"},
        ])
        records = list(self.c.iter_pmc_records(manifest, self.root, quiet_logger()))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["document_id"], "PMC1000001.1")

    def test_unavailable_rows_are_never_read_as_files(self):
        """unavailable_current_dataset rows have no xml_path to open."""
        manifest = self.root / "metadata" / "pmc.csv"
        write_manifest_row(manifest, self.c.PMC_MANIFEST_FIELDS, [{
            "pmid": "", "pmcid": "PMC9999999", "version": "", "doi": "",
            "title": "", "citation": "", "is_pmc_openaccess": "", "is_manuscript": "",
            "license_code": "", "is_retracted": "", "json_path": "", "xml_path": "",
            "status": "unavailable_current_dataset",
        }])
        records = list(self.c.iter_pmc_records(manifest, self.root, quiet_logger()))
        self.assertEqual(records, [])

    def test_missing_pmid_survives_as_blank_not_fabricated(self):
        """A real, legitimate case: some PMC metadata carries no PMID."""
        xml_dir = self._xml_dir("PMC1000003", "1")
        (xml_dir / "PMC1000003.1.xml").write_bytes(ARTICLE_XML)
        manifest = self.root / "metadata" / "pmc.csv"
        write_manifest_row(manifest, self.c.PMC_MANIFEST_FIELDS, [{
            "pmid": "", "pmcid": "PMC1000003", "version": "1", "doi": "",
            "title": "No PMID article", "citation": "", "is_pmc_openaccess": "True",
            "is_manuscript": "False", "license_code": "CC BY",
            "is_retracted": "False",
            "json_path": "data/raw/pmc/PMC1000003.1/PMC1000003.1.json",
            "xml_path": "data/raw/pmc/PMC1000003.1/PMC1000003.1.xml",
            "status": "already_verified",
        }])
        records = list(self.c.iter_pmc_records(manifest, self.root, quiet_logger()))
        self.assertEqual(records[0]["pmid"], "")


class RedistributionAllowedTests(unittest.TestCase):
    """PMC's real license_code values are space-delimited ('CC BY'), not
    hyphenated ('CC-BY') like the DISTRIBUTABLE set. Both must match."""

    def setUp(self):
        self.c = load_module()

    def test_space_delimited_pmc_style_codes_are_recognised(self):
        self.assertTrue(self.c.redistribution_allowed("CC BY"))
        self.assertTrue(self.c.redistribution_allowed("CC BY-NC"))
        self.assertTrue(self.c.redistribution_allowed("CC0"))

    def test_hyphenated_codes_still_work(self):
        self.assertTrue(self.c.redistribution_allowed("CC-BY"))
        self.assertTrue(self.c.redistribution_allowed("CC-BY-NC-SA"))

    def test_case_insensitive(self):
        self.assertTrue(self.c.redistribution_allowed("cc by"))

    def test_non_distributable_and_missing_licences_fail_closed(self):
        self.assertFalse(self.c.redistribution_allowed("TDM"))
        self.assertFalse(self.c.redistribution_allowed(""))
        self.assertFalse(self.c.redistribution_allowed(None))
        self.assertFalse(self.c.redistribution_allowed("CC BY-NC-ND"))


if __name__ == "__main__":
    unittest.main()
