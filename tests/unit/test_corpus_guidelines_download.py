"""03_guidelines.py's --download flow: downloads curated rows, never
invents them, and is safe to rerun.

Network is mocked (via _common.urlopen, which 03_guidelines.py's downloads
go through) - no real HTTP request happens in this suite.
"""

import importlib.util
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "corpus" / "scripts"


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), SCRIPT_DIR / name)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quiet_logger(name="guidelines_download_test"):
    log = logging.getLogger(name)
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


def make_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>"
        b"/MediaBox[0 0 200 200]/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length " + str(len(content)).encode() + b">>\n"
        b"stream\n" + content + b"\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \ntrailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )


class _FakeResponse:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DownloadRegistryTests(unittest.TestCase):

    def setUp(self):
        self.c = load_module("_common.py")
        self.g = load_module("03_guidelines.py")
        self.log = quiet_logger()
        self.tmpdir = TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.addCleanup(self.tmpdir.cleanup)
        self.registry = self.root / "guidelines.csv"
        self.raw_dir = self.root / "raw" / "guidelines"

    def _row(self, **overrides):
        row = {f: "" for f in self.c.GUIDELINE_REGISTRY_FIELDS}
        row.update(document_id="G-1", organization="NIA", title="Guidance",
                  publication_date="2023", document_type="clinical_guideline",
                  license="us-government-work",
                  source_url="https://example.org/doc.pdf")
        row.update(overrides)
        return row

    def _write(self, rows):
        self.c.write_document_registry(self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, rows)

    def test_downloads_a_curated_row_and_updates_local_file(self):
        self._write([self._row()])
        with patch.object(self.c, "urlopen",
                          return_value=_FakeResponse(make_pdf("text"))):
            downloaded, skipped, failed = self.g.download_registry(
                self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.raw_dir,
                "guidelines", self.log, corpus_root=self.root)
        self.assertEqual((downloaded, skipped, failed), (1, 0, 0))
        rows = self.c.read_document_registry(self.registry, self.c.GUIDELINE_REGISTRY_FIELDS)
        self.assertTrue(rows[0]["local_file"])
        self.assertTrue((self.root / rows[0]["local_file"]).exists())

    def test_a_row_without_source_url_is_skipped_not_a_failure(self):
        """Most textbook rows have no fetchable URL - manual acquisition,
        by the registry's own access_method design."""
        self._write([self._row(source_url="")])
        downloaded, skipped, failed = self.g.download_registry(
            self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.raw_dir,
            "guidelines", self.log, corpus_root=self.root)
        self.assertEqual((downloaded, skipped, failed), (0, 1, 0))

    def test_an_already_downloaded_row_is_not_re_fetched(self):
        """Safe to rerun: rows with local_file already set are skipped."""
        self._write([self._row(local_file="already/here.pdf")])
        with patch.object(self.c, "urlopen") as mock_urlopen:
            downloaded, skipped, failed = self.g.download_registry(
                self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.raw_dir,
                "guidelines", self.log, corpus_root=self.root)
        mock_urlopen.assert_not_called()
        self.assertEqual((downloaded, skipped, failed), (0, 0, 0))

    def test_a_row_missing_document_id_fails_rather_than_guessing_a_filename(self):
        self._write([self._row(document_id="")])
        with patch.object(self.c, "urlopen",
                          return_value=_FakeResponse(make_pdf("text"))):
            downloaded, skipped, failed = self.g.download_registry(
                self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.raw_dir,
                "guidelines", self.log, corpus_root=self.root)
        self.assertEqual(failed, 1)

    def test_a_failed_download_does_not_corrupt_the_registry(self):
        """One bad row must not prevent a good row's local_file from being
        recorded, and the registry must stay valid CSV throughout."""
        self._write([
            self._row(document_id="good", source_url="https://example.org/good.pdf"),
            self._row(document_id="bad", source_url="https://example.org/bad.pdf"),
        ])

        def fake_urlopen(request, timeout=None):
            if "good" in request.full_url:
                return _FakeResponse(make_pdf("good text"))
            return _FakeResponse(b"<html>404</html>")

        with patch.object(self.c, "urlopen", side_effect=fake_urlopen):
            self.g.download_registry(
                self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.raw_dir,
                "guidelines", self.log, corpus_root=self.root)

        rows = self.c.read_document_registry(self.registry, self.c.GUIDELINE_REGISTRY_FIELDS)
        by_id = {r["document_id"]: r for r in rows}
        self.assertTrue(by_id["good"]["local_file"])
        self.assertFalse(by_id["bad"]["local_file"])

    def test_redistribution_allowed_is_stamped_after_download(self):
        self._write([self._row(license="CC BY")])
        with patch.object(self.c, "urlopen",
                          return_value=_FakeResponse(make_pdf("text"))):
            self.g.download_registry(
                self.registry, self.c.GUIDELINE_REGISTRY_FIELDS, self.raw_dir,
                "guidelines", self.log, corpus_root=self.root)
        rows = self.c.read_document_registry(self.registry, self.c.GUIDELINE_REGISTRY_FIELDS)
        self.assertEqual(rows[0]["redistribution_allowed"], "true")


class MainCLITests(unittest.TestCase):
    """The --download flag wired into main() and its exit-code contract.

    Run as a subprocess against an isolated temp copy of corpus/,
    never in-process: main() calls _common.get_logger(), which resolves its
    log file from the loaded module's own __file__ - in-process, that would
    write test output into the real, tracked
    corpus/logs/retrieval.log. A subprocess launched with the
    isolated copy as its script path has its own __file__ and therefore its
    own, disposable log file.
    """

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.copy = Path(self.tmpdir.name) / "corpus"
        import shutil
        shutil.copytree(ROOT / "corpus", self.copy,
                        ignore=shutil.ignore_patterns("__pycache__"))

    def _run(self, *args):
        import subprocess, sys
        return subprocess.run(
            [sys.executable, str(self.copy / "scripts" / "03_guidelines.py"), *args],
            cwd=self.copy, capture_output=True, text=True, timeout=60,
        )

    def test_validate_only_default_exits_zero_on_empty_registries(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_download_flag_exits_zero_when_nothing_to_download(self):
        """Both registries in a fresh copy are currently empty - no rows,
        nothing to fetch, no failure."""
        result = self._run("--download")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_does_not_touch_the_real_repository(self):
        """The isolated copy is where all writes land, never the real tree."""
        real_log = ROOT / "corpus" / "logs" / "retrieval.log"
        before = real_log.stat().st_mtime if real_log.exists() else None
        self._run("--download")
        after = real_log.stat().st_mtime if real_log.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
