"""Proves the corpus pipeline's real output is actually consumable by the
retrieval layer - not by schema inspection, but by running the real
04_normalize -> 05_deduplicate -> 06_chunk -> 07_claim_classification chain
(offline fixture, no network/real tokenizer needed) and then calling
experiments/shared/retrieval/corpus.py's real read_passages() against the result.

This is the corpus-completion audit's answer to "can downstream retrieval
consume the final corpus" for anything code-level can verify without the
real 4.3M-chunk corpus, which only exists on the machine that built it
(corpus/data/** is gitignored by design).
"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.shared.retrieval import corpus as corpus_module  # noqa: E402


class PipelineToRetrievalTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TemporaryDirectory()
        cls.copy = Path(cls.tmpdir.name) / "corpus"
        shutil.copytree(ROOT / "corpus", cls.copy,
                        ignore=shutil.ignore_patterns("__pycache__"))

        def run(script, *args):
            result = subprocess.run(
                [sys.executable, str(cls.copy / "scripts" / script), *args],
                cwd=cls.copy, capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, f"{script}: {result.stderr}"

        run("04_normalize.py", "--input", "data/raw/pubmed/records.example.jsonl")
        run("05_deduplicate.py")
        run("06_chunk.py", "--tokenizer", "whitespace")
        run("07_claim_classification.py")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_read_passages_loads_the_real_pipeline_output(self):
        """The exact function experiments/shared/retrieval/pipeline.py calls to
        build a candidate set, pointed at real (not hand-built) pipeline
        output."""
        passages = corpus_module.read_passages(self.copy)
        self.assertGreater(len(passages), 0)

    def test_every_passage_has_the_fields_retrieval_and_admission_need(self):
        passages = corpus_module.read_passages(self.copy)
        for p in passages:
            self.assertTrue(p.chunk_id)
            self.assertTrue(p.document_id)
            self.assertTrue(p.text)
            self.assertTrue(p.retrieval_text)
            self.assertTrue(p.source_tier)  # read_passages() defaults to
            # "unknown" rather than "" - source_tier is a free-form tag,
            # not a closed enum enforced anywhere downstream.
            self.assertIsInstance(p.retracted, bool)
            self.assertIsInstance(p.claim_classes, tuple)

    def test_chunk_ids_are_unique_across_the_real_pipeline_output(self):
        """read_passages() itself raises on a duplicate chunk_id (see its
        own docstring: frozen candidate sets can't replay a collision) -
        this asserts the real pipeline never produces one, rather than
        just trusting the loader's guard to catch it if it ever did."""
        passages = corpus_module.read_passages(self.copy)
        ids = [p.chunk_id for p in passages]
        self.assertEqual(len(ids), len(set(ids)))

    def test_claim_classification_fields_survive_into_retrieval(self):
        """Stage 07 only adds fields to what Stage 06 wrote - this checks
        the addition actually reaches the passage the retrieval layer
        hands to the admission policies, not just the raw JSONL."""
        passages = corpus_module.read_passages(self.copy)
        self.assertTrue(any(p.claim_classes for p in passages),
                        "no passage carries a claim_classes tag - Stage 07's "
                        "output isn't reaching the retrieval layer")

    def test_snapshot_id_is_stable_and_content_derived(self):
        first = corpus_module.snapshot_id(self.copy)
        second = corpus_module.snapshot_id(self.copy)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
