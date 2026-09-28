"""build_textbook_index's cache: encoding 50,000 textbook passages took
~2 hours on CPU on a real run (2026-09-28) - every retry of a later,
unrelated failure (e.g. the scorer) must not redo it. load_textbook_passages
is deterministic given the same (n, seed) (evaluation.tests already covers
that directly); this file covers only the caching decision itself, with
encoding mocked out - it must never run when a matching cache is present.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.baseline.filter_training import build_labels
from experiments.shared.retrieval.corpus import CorpusPassage
from experiments.shared.retrieval.encoders import HashingEncoder


class _FakeEncoder:
    """HashingEncoder plus the on_progress kwarg build_index() always
    passes through - build_textbook_index always supplies a real
    on_progress callback, unlike most of this suite's other HashingEncoder
    uses."""

    name = "fake-textbook-encoder"

    def __init__(self):
        self._real = HashingEncoder(dim=8)

    def encode(self, texts, *, on_progress=None):
        if on_progress is not None:
            on_progress(1, 1, 0.0)
        return self._real.encode(texts)


def _fake_passages(n, seed):
    # Mirrors load_textbook_passages' determinism contract (same n/seed ->
    # same passages) without touching the real dataset or needing the
    # `datasets` package, which this test environment does not have.
    return tuple(
        CorpusPassage(
            chunk_id=f"textbook_train_{seed}_{i}", document_id=f"doc-{i}",
            text=f"passage {i}", retrieval_text=f"[TITLE] t{i}\npassage {i}",
            publication_date=None, source_tier="textbook", retracted=False,
        )
        for i in range(n)
    )


class TextbookIndexCacheTests(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root_patch = mock.patch.object(
            build_labels, "TEXTBOOK_INDEX_CACHE_ROOT", Path(self._tmp.name),
        )
        self._root_patch.start()
        self._passages_patch = mock.patch.object(
            build_labels, "load_textbook_passages", side_effect=_fake_passages,
        )
        self._passages_patch.start()

    def tearDown(self):
        self._passages_patch.stop()
        self._root_patch.stop()
        self._tmp.cleanup()

    def test_builds_and_caches_on_first_call(self):
        with mock.patch(
            "experiments.shared.retrieval.encoders.medcpt_article_encoder",
            return_value=_FakeEncoder(),
        ):
            index, passages = build_labels.build_textbook_index(5, seed=42, device=None)
        self.assertEqual(len(index.passage_ids), 5)
        self.assertEqual(len(passages), 5)
        cache_dir = Path(self._tmp.name) / "n5_seed42"
        self.assertTrue((cache_dir / "manifest.json").exists())

    def test_second_call_reuses_the_cache_without_encoding(self):
        with mock.patch(
            "experiments.shared.retrieval.encoders.medcpt_article_encoder",
            return_value=_FakeEncoder(),
        ):
            first_index, _ = build_labels.build_textbook_index(5, seed=42, device=None)

        def _must_not_run(*a, **k):
            raise AssertionError("encoding must not run on a cache hit")

        with mock.patch(
            "experiments.shared.retrieval.encoders.medcpt_article_encoder",
            side_effect=_must_not_run,
        ):
            second_index, passages = build_labels.build_textbook_index(
                5, seed=42, device=None)

        self.assertEqual(first_index.passage_ids, second_index.passage_ids)
        self.assertTrue((first_index.vectors == second_index.vectors).all())
        self.assertEqual(len(passages), 5)

    def test_different_seed_does_not_reuse_a_mismatched_cache(self):
        with mock.patch(
            "experiments.shared.retrieval.encoders.medcpt_article_encoder",
            return_value=_FakeEncoder(),
        ):
            build_labels.build_textbook_index(5, seed=1, device=None)
            # Different seed -> different passage sample -> must rebuild,
            # not silently reuse seed=1's cache.
            index, _ = build_labels.build_textbook_index(5, seed=2, device=None)
        self.assertEqual(len(index.passage_ids), 5)


if __name__ == "__main__":
    unittest.main()
