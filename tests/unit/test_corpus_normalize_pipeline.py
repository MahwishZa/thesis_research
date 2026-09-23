"""Stages 04-06's real-data path and their regression safety against
whatever fixture output is currently sitting in the local working tree.

Two things are locked here:

1. The offline fixture path (--input records.example.jsonl) must keep
   producing byte-identical output to what is currently on disk under
   corpus/data/ - a rewrite that changes it without anyone
   deciding to would be a silent regression. IMPORTANT: corpus/
   data/** is gitignored by design ("research data is never committed" -
   see corpus/.gitignore), so this is NOT a comparison against a
   git-tracked golden file, despite this module's name. On a fresh clone
   with no prior local pipeline run, corpus/data/normalized/,
   deduplicated/ and chunks/ do not exist, and the tests below that read
   `tree=CORPUS` will fail with FileNotFoundError rather than skip. Run
   04_normalize.py --input records.example.jsonl -> 05_deduplicate.py ->
   06_chunk.py --tokenizer whitespace once against the real
   corpus/ tree first (see README.md and
   _archive/docs_legacy/status_and_decisions.md) to
   populate a local baseline before these tests are meaningful; they then
   catch drift within this working copy over time, not against history.
2. The new real-data path (reading metadata/pmc.csv + XML directly) must
   produce internally consistent records: the AD-relevance decision made
   once in Stage 04 must be the same one that appears on every chunk in
   Stage 06, not a null or a recomputation.

Every test runs against an isolated temporary copy of corpus/ -
nothing here writes to the real tree. (The `tree=CORPUS` comparison reads
the real tree; it never writes to it.)
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

ROOT = Path(__file__).resolve().parents[2]
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


class FixturePathRegressionTests(unittest.TestCase):
    """04 --input <fixture> -> 05 -> 06 must reproduce the committed output
    exactly. This is the same offline path the corpus README documents."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TemporaryDirectory()
        cls.copy = Path(cls.tmpdir.name) / "corpus"
        shutil.copytree(CORPUS, cls.copy, ignore=shutil.ignore_patterns("__pycache__"))

        fixture = "data/raw/pubmed/records.example.jsonl"
        r = run_stage(cls.copy, "04_normalize.py", "--input", fixture)
        assert r.returncode == 0, r.stderr
        r = run_stage(cls.copy, "05_deduplicate.py")
        assert r.returncode == 0, r.stderr
        r = run_stage(cls.copy, "06_chunk.py", "--tokenizer", "whitespace")
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _rows(self, relative_path, tree=None):
        path = (tree or self.copy) / relative_path
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    def test_normalized_output_matches_the_committed_fixture_run(self):
        fresh = self._rows("data/normalized/documents.jsonl")
        committed = self._rows("data/normalized/documents.jsonl", tree=CORPUS)
        self.assertEqual(fresh, committed)

    def test_deduplicated_output_matches_the_committed_fixture_run(self):
        fresh = self._rows("data/deduplicated/documents.jsonl")
        committed = self._rows("data/deduplicated/documents.jsonl", tree=CORPUS)
        self.assertEqual(fresh, committed)

    def test_duplicates_registry_matches_and_is_valid_csv(self):
        fresh = (self.copy / "metadata" / "duplicates.csv").read_text(encoding="utf-8")
        committed = (CORPUS / "metadata" / "duplicates.csv").read_text(encoding="utf-8")
        self.assertEqual(fresh, committed)

    def test_chunk_output_differs_only_by_stage_07_fields(self):
        """This copy only runs 04->06 (see setUpClass), while the local
        reference under corpus/data/ has also had Stage 07 run -
        so only Stage 07's fields (claim_classes/claim_confidence/
        claim_method for the topical dimension, claim_evidence_levels/
        claim_evidence_confidence for the evidence-level dimension) should
        differ; everything Stage 04-06 produce, including ad_relevant/
        ad_relevance_score, must be identical since both sides run the same
        current code."""
        fresh = self._rows("data/chunks/chunks.jsonl")
        committed = self._rows("data/chunks/chunks.jsonl", tree=CORPUS)
        self.assertEqual(len(fresh), len(committed))
        diffs = set()
        for a, b in zip(fresh, committed):
            for key in set(a) | set(b):
                if a.get(key) != b.get(key):
                    diffs.add(key)
        self.assertEqual(
            diffs,
            {"claim_classes", "claim_confidence", "claim_method",
             "claim_evidence_levels", "claim_evidence_confidence"},
        )


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
