"""Guideline and textbook acquisition: download, PDF extraction, and the
normalize-bridge that merges them with PMC records.

Both registries (metadata/guidelines.csv, metadata/textbooks.csv) are
manually curated by design - a human adds a row with a real source_url and a
licence verified for that specific document (03_guidelines.py's own
docstring: "never scraped"). Nothing here invents a document, a licence, or
a URL. What these tests lock is everything downstream of a curated row:
downloading it safely, refusing anything that isn't actually a PDF,
extracting its text, and including it in the corpus only when its licence
actually permits redistribution.

Network is mocked throughout - nothing here makes a real HTTP request.
"""

import importlib.util
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "corpus" / "scripts" / "_common.py"


def load_module():
    spec = importlib.util.spec_from_file_location("corpus_common", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quiet_logger(name="official_docs_test"):
    log = logging.getLogger(name)
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


#: A minimal, valid one-page PDF with a real, extractable content stream.
#: Built by hand (no dependency needed to construct one) so tests never need
#: a real document.
def make_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>"
        b"/MediaBox[0 0 200 200]/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length " + str(len(content)).encode() + b">>\n"
        b"stream\n" + content + b"\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )


class ExtractPDFTextTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()

    def test_extracts_real_text_from_a_valid_pdf(self):
        sections = self.c.extract_pdf_text(make_pdf("Guideline body text."))
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["section"], "page 1")
        self.assertIn("Guideline body text.", sections[0]["text"])

    def test_not_a_pdf_raises(self):
        with self.assertRaises(self.c.CorpusPipelineError):
            self.c.extract_pdf_text(b"not a pdf at all, just bytes")

    def test_empty_bytes_raises(self):
        with self.assertRaises(self.c.CorpusPipelineError):
            self.c.extract_pdf_text(b"")


class DownloadDocumentTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()
        self.log = quiet_logger()

    class _FakeResponse:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_non_https_url_is_refused_before_any_request(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(self.c.CorpusPipelineError) as ctx:
                self.c.download_document(
                    "http://example.org/doc.pdf", Path(tmp) / "d.pdf", self.log)
            self.assertIn("HTTPS", str(ctx.exception))

    def test_a_real_pdf_response_is_saved_and_returned(self):
        pdf = make_pdf("Real content.")
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "doc.pdf"
            with patch.object(self.c, "urlopen", return_value=self._FakeResponse(pdf)):
                data = self.c.download_document(
                    "https://example.org/doc.pdf", dest, self.log)
            self.assertEqual(data, pdf)
            self.assertEqual(dest.read_bytes(), pdf)

    def test_an_html_paywall_response_is_refused_and_nothing_is_written(self):
        html = b"<html><body>Please log in to continue</body></html>"
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "doc.pdf"
            with patch.object(self.c, "urlopen", return_value=self._FakeResponse(html)):
                with self.assertRaises(self.c.CorpusPipelineError) as ctx:
                    self.c.download_document(
                        "https://example.org/doc.pdf", dest, self.log)
            self.assertIn("did not return a PDF", str(ctx.exception))
            self.assertFalse(dest.exists())

    def test_a_network_error_raises_after_retrying(self):
        from urllib.error import URLError
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "doc.pdf"
            with patch.object(self.c, "urlopen", side_effect=URLError("refused")):
                with patch.object(self.c, "_time") as fake_time:
                    with self.assertRaises(self.c.CorpusPipelineError):
                        self.c.download_document(
                            "https://example.org/doc.pdf", dest, self.log)
                    # Retried, not just failed once.
                    self.assertGreater(fake_time.sleep.call_count, 0)


class RegistryReadWriteTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()

    def test_a_schema_mismatch_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "guidelines.csv"
            path.write_text("wrong,header\n1,2\n", encoding="utf-8")
            with self.assertRaises(self.c.CorpusPipelineError):
                self.c.read_document_registry(path, self.c.GUIDELINE_REGISTRY_FIELDS)

    def test_the_real_committed_guidelines_registry_matches_the_schema(self):
        real = ROOT / "corpus" / "metadata" / "guidelines.csv"
        if not real.exists():
            self.skipTest("real guidelines.csv not present")
        rows = self.c.read_document_registry(real, self.c.GUIDELINE_REGISTRY_FIELDS)
        self.assertIsInstance(rows, list)

    def test_the_real_committed_textbooks_registry_matches_the_schema(self):
        real = ROOT / "corpus" / "metadata" / "textbooks.csv"
        if not real.exists():
            self.skipTest("real textbooks.csv not present")
        rows = self.c.read_document_registry(real, self.c.TEXTBOOK_REGISTRY_FIELDS)
        self.assertIsInstance(rows, list)

    def test_write_then_read_round_trips(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "guidelines.csv"
            row = {f: "" for f in self.c.GUIDELINE_REGISTRY_FIELDS}
            row.update(document_id="G-1", title="Test", license="CC BY")
            self.c.write_document_registry(path, self.c.GUIDELINE_REGISTRY_FIELDS, [row])
            rows = self.c.read_document_registry(path, self.c.GUIDELINE_REGISTRY_FIELDS)
            self.assertEqual(rows, [row])

    def test_write_is_atomic_no_temp_file_left_behind(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "guidelines.csv"
            self.c.write_document_registry(path, self.c.GUIDELINE_REGISTRY_FIELDS, [])
            self.assertTrue(path.exists())
            self.assertFalse((path.with_suffix(".csv.part")).exists())


class IterOfficialDocumentsTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module()
        self.log = quiet_logger()
        self.tmpdir = TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.addCleanup(self.tmpdir.cleanup)

    def _row(self, **overrides):
        row = {f: "" for f in self.c.GUIDELINE_REGISTRY_FIELDS}
        row.update(document_id="G-1", organization="NIA", title="AD guidance",
                  publication_date="2023", document_type="clinical_guideline",
                  license="us-government-work")
        row.update(overrides)
        return row

    def _write_registry(self, rows):
        path = self.root / "metadata" / "guidelines.csv"
        self.c.write_document_registry(path, self.c.GUIDELINE_REGISTRY_FIELDS, rows)
        return path

    def test_a_downloaded_openly_licensed_row_yields_one_record(self):
        pdf_dir = self.root / "data" / "raw" / "guidelines"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "G-1.pdf").write_bytes(
            make_pdf("Alzheimer disease diagnostic guidance text."))
        registry = self._write_registry([self._row(
            local_file="data/raw/guidelines/G-1.pdf")])

        records = list(self.c.iter_official_documents(
            registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.root, self.log,
            default_source_tier="clinical_guideline"))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["document_id"], "G-1")
        self.assertEqual(record["source_tier"], "clinical_guideline")
        self.assertIn("guidance text", record["abstract"])
        self.assertEqual(record["sections"][0]["section"], "title")

    def test_abstract_is_populated_from_the_first_page_not_left_blank(self):
        """Regression: assess_ad_relevance only reads title+abstract. A
        blank abstract would starve the relevance gate of the document's
        actual content and could wrongly exclude a real guideline whose
        title only uses the ambiguous 'AD' abbreviation."""
        pdf_dir = self.root / "data" / "raw" / "guidelines"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "G-1.pdf").write_bytes(make_pdf("Body content here."))
        registry = self._write_registry([self._row(
            title="Criteria for AD",  # abbreviation only, deliberately
            local_file="data/raw/guidelines/G-1.pdf")])

        records = list(self.c.iter_official_documents(
            registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.root, self.log,
            default_source_tier="clinical_guideline"))
        self.assertNotEqual(records[0]["abstract"], "")

    def test_a_row_with_no_local_file_yields_nothing_not_an_error(self):
        registry = self._write_registry([self._row(local_file="")])
        records = list(self.c.iter_official_documents(
            registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.root, self.log,
            default_source_tier="clinical_guideline"))
        self.assertEqual(records, [])

    def test_a_restricted_licence_yields_nothing_even_when_downloaded(self):
        """'Restricted guidance is recorded as metadata only; its text
        never enters the distributable corpus' - 03_guidelines.py."""
        pdf_dir = self.root / "data" / "raw" / "guidelines"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "G-1.pdf").write_bytes(make_pdf("Some text."))
        registry = self._write_registry([self._row(
            license="all-rights-reserved",
            local_file="data/raw/guidelines/G-1.pdf")])

        records = list(self.c.iter_official_documents(
            registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.root, self.log,
            default_source_tier="clinical_guideline"))
        self.assertEqual(records, [])

    def test_a_missing_or_corrupt_local_file_is_skipped_not_fatal(self):
        registry = self._write_registry([self._row(
            local_file="data/raw/guidelines/does-not-exist.pdf")])
        records = list(self.c.iter_official_documents(
            registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.root, self.log,
            default_source_tier="clinical_guideline"))
        self.assertEqual(records, [])

    def test_mixed_batch_counts_each_outcome_separately(self):
        pdf_dir = self.root / "data" / "raw" / "guidelines"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "good.pdf").write_bytes(make_pdf("Good content."))
        (pdf_dir / "restricted.pdf").write_bytes(make_pdf("Restricted content."))
        registry = self._write_registry([
            self._row(document_id="good", local_file="data/raw/guidelines/good.pdf"),
            self._row(document_id="restricted", license="proprietary",
                      local_file="data/raw/guidelines/restricted.pdf"),
            self._row(document_id="not-yet", local_file=""),
        ])
        records = list(self.c.iter_official_documents(
            registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.root, self.log,
            default_source_tier="clinical_guideline"))
        self.assertEqual([r["document_id"] for r in records], ["good"])


if __name__ == "__main__":
    unittest.main()
