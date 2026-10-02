"""Stages 04-06's real-data path, and the determinism of the offline fixture path.

Two things are locked here:

1. The offline fixture path (``--input records.example.jsonl`` -> 05 -> 06 -> 07) must
   be deterministic, and Stage 07 must only ADD its claim-classification fields to what
   Stages 04-06 produced. This is checked hermetically: two independent runs in lean
   scaffold copies (``evaluation/tests/corpus_scaffold.py``) must agree byte for byte.
   It deliberately does NOT compare against ``corpus/data/`` or ``corpus/metadata/``:
   ``corpus/data/**`` is gitignored and on a machine that has built the real corpus
   holds many GB of unrelated data, and ``corpus/metadata/duplicates.csv`` is the
   real corpus's committed registry, not a fixture result. (An earlier version of
   these tests compared against both, so it failed on every clean clone and would
   have tried to load the real corpus on the researcher's machine.)
2. The real-data path (reading metadata/pmc.csv + XML directly) must produce
   internally consistent records: the AD-relevance decision made once in Stage 04
   must be the same one that appears on every chunk in Stage 06, not a null or a
   recomputation.

Every test runs against isolated temporary copies - nothing here writes to the real tree.
"""

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.tests.corpus_scaffold import copy_corpus_scaffold

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "corpus"


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), CORPUS / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_stage(cwd, script, *args):
    """Run a stage script as a subprocess against an isolated corpus copy.

    Subprocess, not import, because each stage script re-derives its own
    paths from __file__ at import time - running in-process against a copied
    tree would still resolve to the real corpus/.
    """
    result = subprocess.run(
        [sys.executable, str(cwd / "scripts" / script), *args],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )
    return result


class FixturePathDeterminismTests(unittest.TestCase):
    """04 --input <fixture> -> 05 -> 06 (-> 07) twice: identical output, and 07 only adds
    its own fields. This is the offline path the corpus README documents."""

    CLAIM_FIELDS = {"claim_classes", "claim_confidence", "claim_method",
                    "claim_evidence_levels", "claim_evidence_confidence"}

    @classmethod
    def _run_pipeline(cls, root):
        fixture = "data/raw/pubmed/records.example.jsonl"
        for script, args in (("04_normalize.py", ("--input", fixture)),
                             ("05_deduplicate.py", ()),
                             ("06_chunk.py", ("--tokenizer", "whitespace"))):
            r = run_stage(root, script, *args)
            assert r.returncode == 0, f"{script}: {r.stderr}"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TemporaryDirectory()
        base = Path(cls.tmpdir.name)
        cls.a, cls.b = base / "a", base / "b"
        for root in (cls.a, cls.b):
            copy_corpus_scaffold(root)
            cls._run_pipeline(root)
        cls.chunks_before_07 = cls._rows_in(cls.a, "data/chunks/chunks.jsonl")
        r = run_stage(cls.a, "07_claim_classification.py")
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @staticmethod
    def _rows_in(root, relative_path):
        path = root / relative_path
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    def test_the_scaffold_does_not_copy_local_corpus_data(self):
        """Only the committed fixture (plus empty skeleton dirs) may be present."""
        files = sorted(p.relative_to(self.b).as_posix()
                       for p in (self.b / "data").rglob("*") if p.is_file()
                       and p.name != ".gitkeep")
        self.assertTrue(all(f.startswith("data/raw/pubmed/records.example")
                            or f.startswith("data/normalized/")
                            or f.startswith("data/deduplicated/")
                            or f.startswith("data/chunks/") for f in files), files)

    def test_normalized_output_is_deterministic_and_nonempty(self):
        a = self._rows_in(self.a, "data/normalized/documents.jsonl")
        b = self._rows_in(self.b, "data/normalized/documents.jsonl")
        self.assertEqual(a, b)
        self.assertTrue(a and all(str(r.get("document_id", "")).startswith("FIXTURE-") for r in a))

    def test_deduplicated_output_is_deterministic(self):
        self.assertEqual(self._rows_in(self.a, "data/deduplicated/documents.jsonl"),
                         self._rows_in(self.b, "data/deduplicated/documents.jsonl"))

    def test_duplicates_registry_is_deterministic_and_valid_csv(self):
        fa = (self.a / "metadata" / "duplicates.csv").read_text(encoding="utf-8")
        fb = (self.b / "metadata" / "duplicates.csv").read_text(encoding="utf-8")
        self.assertEqual(fa, fb)
        rows = list(csv.reader(fa.splitlines()))
        self.assertEqual(rows[0][:2], ["canonical_document_id", "duplicate_document_id"])
        self.assertTrue(all(len(r) == len(rows[0]) for r in rows))

    def test_chunk_output_is_deterministic(self):
        self.assertEqual(self.chunks_before_07,
                         self._rows_in(self.b, "data/chunks/chunks.jsonl"))

    def test_stage_07_only_adds_its_claim_fields(self):
        """Stage 07 (run on copy A only) must change nothing Stages 04-06 produced -
        including ad_relevant / ad_relevance_score - beyond the claim fields."""
        after = self._rows_in(self.a, "data/chunks/chunks.jsonl")
        self.assertEqual(len(after), len(self.chunks_before_07))
        diffs = set()
        for x, y in zip(self.chunks_before_07, after):
            for key in set(x) | set(y):
                if x.get(key) != y.get(key):
                    diffs.add(key)
        self.assertEqual(diffs, self.CLAIM_FIELDS)


class RealDataPathTests(unittest.TestCase):
    """The new default path: metadata/pmc.csv + XML, no --input given."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.copy = Path(self.tmpdir.name) / "corpus"
        shutil.copytree(CORPUS, self.copy, ignore=shutil.ignore_patterns("__pycache__"))
        # Replace the real, finalized manifest/data with a small synthetic
        # one so this test never depends on (or risks) the real corpus.
        # data/raw/pmc may not even exist here (it is gitignored - real
        # corpus data lives only on the machine that built it).
        shutil.rmtree(self.copy / "data" / "raw" / "pmc", ignore_errors=True)
        (self.copy / "data" / "raw" / "pmc").mkdir(parents=True)
        xml_dir = self.copy / "data" / "raw" / "pmc" / "PMC7000001.1"
        xml_dir.mkdir()
        (xml_dir / "PMC7000001.1.xml").write_bytes(
            b"<article><front><article-meta>"
            b"<title-group><article-title>Donepezil in Alzheimer disease</article-title></title-group>"
            b'<pub-date pub-type="epub"><year>2022</year><month>5</month><day>1</day></pub-date>'
            b"<abstract><p>A randomized trial of donepezil.</p></abstract>"
            b"</article-meta></front>"
            b"<body><sec><title>Results</title><p>Cognitive scores improved.</p></sec></body>"
            b"</article>")
        with (self.copy / "metadata" / "pmc.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "pmid", "pmcid", "version", "doi", "title", "citation",
                "is_pmc_openaccess", "is_manuscript", "license_code",
                "is_retracted", "json_path", "xml_path", "status"])
            writer.writeheader()
            writer.writerow({
                "pmid": "999", "pmcid": "PMC7000001", "version": "1", "doi": "",
                "title": "manifest title", "citation": "", "is_pmc_openaccess": "True",
                "is_manuscript": "False", "license_code": "CC BY",
                "is_retracted": "False",
                "json_path": "data/raw/pmc/PMC7000001.1/PMC7000001.1.json",
                "xml_path": "data/raw/pmc/PMC7000001.1/PMC7000001.1.xml",
                "status": "already_verified",
            })

    def test_the_full_chain_runs_end_to_end_on_real_shaped_data(self):
        for script, args in (
            ("04_normalize.py", []),
            ("05_deduplicate.py", []),
            ("06_chunk.py", ["--tokenizer", "whitespace"]),
            ("07_claim_classification.py", []),
        ):
            result = run_stage(self.copy, script, *args)
            self.assertEqual(result.returncode, 0,
                             f"{script} failed:\n{result.stderr}")

        chunks = [json.loads(l) for l in
                 (self.copy / "data" / "chunks" / "chunks.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk["ad_relevant"], True)
            self.assertEqual(chunk["ad_relevance_score"], 1.0)
            self.assertIn("claim_classes", chunk)

    def test_missing_manifest_fails_clearly_instead_of_using_the_fixture(self):
        """The dangerous behaviour the audit flagged: silently falling back
        to the 10-record fixture when the real manifest is absent."""
        (self.copy / "metadata" / "pmc.csv").unlink()
        normalized = self.copy / "data" / "normalized" / "documents.jsonl"
        normalized.unlink(missing_ok=True)  # the copied tree may already
        # carry the committed fixture-run output; start from "absent".

        result = run_stage(self.copy, "04_normalize.py")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--finalize", result.stderr)
        self.assertFalse(normalized.exists(),
                         "must not silently produce output from the fixture "
                         "when the real manifest is missing")


if __name__ == "__main__":
    unittest.main()
