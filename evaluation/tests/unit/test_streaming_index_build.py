"""Memory-bounded, checkpointed index build (streaming_index_build.py).

Locks the property the whole module exists for: identical output to the
in-memory ``build_index()``/``DenseIndex.save()`` path, plus the safety
properties an unattended multi-hour job on consumer hardware needs -
checkpointing, resume, and refusing to silently continue against a
changed corpus, a different encoder, or a different filtering policy.

All fixtures are tiny synthetic corpora; nothing here needs the real
4.3M-chunk corpus or a real model (``HashingEncoder`` throughout, same
stand-in ``test_retrieval.py`` uses).
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.shared.retrieval.corpus import (
    dated_only, read_passages, snapshot_id,
)
from experiments.shared.retrieval.encoders import HashingEncoder
from experiments.shared.retrieval.index import DenseIndex, build_index
from experiments.shared.retrieval.streaming_index_build import (
    BuildResult, StreamingBuildError, encode_index_streaming, plan_index_build,
)


def chunk(i, *, dated=True, text=None):
    return {
        "chunk_id": f"C-{i:03d}",
        "document_id": f"D-{i // 3}",
        "text": text or f"passage number {i} about treatment",
        "retrieval_text": f"[TITLE] doc {i}\n{text or f'passage {i}'}",
        "publication_date": "2023-05" if dated else "",
        "source_tier": "peer_reviewed",
        "retracted": False,
        "section": "abstract",
        "claim_classes": ["AD-CRIT-01"],
    }


def write_corpus(root, records):
    chunks = Path(root) / "data" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return root


class StreamingBuildMatchesInMemoryTests(unittest.TestCase):

    def test_output_is_bit_identical_to_the_in_memory_build(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(
                Path(tmp) / "corpus",
                [chunk(i, dated=(i % 7 != 0)) for i in range(63)],
            )
            old_index = build_index(
                dated_only(read_passages(corpus)), HashingEncoder(dim=12),
                corpus_snapshot=snapshot_id(corpus),
            )

            out = Path(tmp) / "index_out"
            encode_index_streaming(
                corpus, out, HashingEncoder(dim=12), batch_size=8,
                checkpoint_every_batches=3,
            )
            new_index = DenseIndex.load(out)

        self.assertEqual(old_index.passage_ids, new_index.passage_ids)
        self.assertEqual(old_index.corpus_snapshot, new_index.corpus_snapshot)
        self.assertTrue(np.array_equal(old_index.vectors, new_index.vectors))

    def test_dated_only_drops_undated_passages_the_same_way(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [
                chunk(1), chunk(2, dated=False), chunk(3),
            ])
            out = Path(tmp) / "index_out"
            encode_index_streaming(corpus, out, HashingEncoder(dim=8))
            index = DenseIndex.load(out)
        self.assertEqual(index.passage_ids, ("C-001", "C-003"))

    def test_include_undated_keeps_everything(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [
                chunk(1), chunk(2, dated=False), chunk(3),
            ])
            out = Path(tmp) / "index_out"
            encode_index_streaming(
                corpus, out, HashingEncoder(dim=8), dated_only=False,
            )
            index = DenseIndex.load(out)
        self.assertEqual(len(index.passage_ids), 3)

    def test_returns_a_build_result_matching_the_written_manifest(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(5)])
            out = Path(tmp) / "index_out"
            result = encode_index_streaming(corpus, out, HashingEncoder(dim=8))
        self.assertIsInstance(result, BuildResult)
        self.assertEqual(result.n_passages, 5)
        self.assertEqual(result.dim, 8)
        self.assertEqual(result.encoder_name, "hashing-stub")
        self.assertEqual(result.duplicates_count, 0)


class DuplicateHandlingTests(unittest.TestCase):

    def test_raise_is_the_default_and_fails_loudly(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(1), chunk(1)])
            out = Path(tmp) / "index_out"
            with self.assertRaises(Exception) as ctx:
                encode_index_streaming(corpus, out, HashingEncoder(dim=8))
        self.assertIn("duplicate chunk_id", str(ctx.exception))

    def test_keep_first_matches_the_in_memory_report(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [
                chunk(1, text="first version"),
                chunk(2),
                chunk(1, text="second, different version"),
            ])
            out = Path(tmp) / "index_out"
            result = encode_index_streaming(
                corpus, out, HashingEncoder(dim=8), on_duplicate="keep_first",
            )
            index = DenseIndex.load(out)
        self.assertEqual(len(index.passage_ids), 2)
        self.assertEqual(result.duplicates_count, 1)


class CheckpointAndResumeTests(unittest.TestCase):

    def _crashing_encoder(self, dim, crash_after_batches):
        real = HashingEncoder(dim=dim)
        state = {"n": 0}

        class CrashAfter:
            name = "stable-name"

            def encode(self, texts):
                state["n"] += 1
                if state["n"] > crash_after_batches:
                    raise RuntimeError("simulated interruption")
                return real.encode(texts)

        return CrashAfter()

    def test_interrupted_build_leaves_a_checkpoint_not_a_partial_manifest(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(50)])
            out = Path(tmp) / "index_out"
            encoder = self._crashing_encoder(8, crash_after_batches=2)
            with self.assertRaises(RuntimeError):
                encode_index_streaming(
                    corpus, out, encoder, batch_size=10,
                    checkpoint_every_batches=1,
                )
            self.assertFalse((out / "manifest.json").exists())
            state = json.loads((out / "_build" / "build_state.json").read_text())
            self.assertEqual(state["rows_done"], 20)

    def test_resume_without_the_flag_is_refused(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(30)])
            out = Path(tmp) / "index_out"
            encoder = self._crashing_encoder(8, crash_after_batches=1)
            with self.assertRaises(RuntimeError):
                encode_index_streaming(
                    corpus, out, encoder, batch_size=10,
                    checkpoint_every_batches=1,  # force a checkpoint before the crash
                )
            with self.assertRaises(StreamingBuildError) as ctx:
                encode_index_streaming(
                    corpus, out, HashingEncoder(dim=8), batch_size=10,
                )
        self.assertIn("resume", str(ctx.exception))

    def test_resume_completes_the_build_with_correct_order_and_no_rework(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(50)])
            out = Path(tmp) / "index_out"
            encoder = self._crashing_encoder(8, crash_after_batches=2)
            with self.assertRaises(RuntimeError):
                encode_index_streaming(
                    corpus, out, encoder, batch_size=10,
                    checkpoint_every_batches=1,
                )

            # Resume must use an encoder identity matching the checkpoint
            # ("stable-name", set on _crashing_encoder) - a genuinely
            # different encoder is refused, exercised separately in
            # test_resume_refuses_on_encoder_mismatch.
            class SameNameEncoder:
                name = "stable-name"

                def __init__(self):
                    self._real = HashingEncoder(dim=8)

                def encode(self, texts):
                    return self._real.encode(texts)

            resumed = encode_index_streaming(
                corpus, out, SameNameEncoder(), batch_size=10,
                checkpoint_every_batches=1, resume=True,
            )
            index = DenseIndex.load(out)

            self.assertEqual(resumed.n_passages, 50)
            self.assertEqual(index.passage_ids, tuple(f"C-{i:03d}" for i in range(50)))
            self.assertTrue((out / "_build_completed").exists())
            self.assertFalse((out / "_build").exists())

    def test_resume_refuses_on_corpus_drift(self):
        with TemporaryDirectory() as tmp:
            corpus_root = Path(tmp) / "corpus"
            corpus = write_corpus(corpus_root, [chunk(i) for i in range(30)])
            out = Path(tmp) / "index_out"
            encoder = self._crashing_encoder(8, crash_after_batches=1)
            with self.assertRaises(RuntimeError):
                encode_index_streaming(
                    corpus, out, encoder, batch_size=10, checkpoint_every_batches=1,
                )
            chunk_file = corpus_root / "data" / "chunks" / "chunks.jsonl"
            chunk_file.write_text(
                chunk_file.read_text(encoding="utf-8") + json.dumps(chunk(999)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(StreamingBuildError) as ctx:
                encode_index_streaming(
                    corpus, out, HashingEncoder(dim=8), batch_size=10, resume=True,
                )
        self.assertIn("corpus file", str(ctx.exception))

    def test_resume_refuses_on_dated_only_mismatch(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(30)])
            out = Path(tmp) / "index_out"
            encoder = self._crashing_encoder(8, crash_after_batches=1)
            with self.assertRaises(RuntimeError):
                encode_index_streaming(
                    corpus, out, encoder, batch_size=10, checkpoint_every_batches=1,
                )
            with self.assertRaises(StreamingBuildError) as ctx:
                encode_index_streaming(
                    corpus, out, HashingEncoder(dim=8), batch_size=10,
                    resume=True, dated_only=False,
                )
        self.assertIn("dated_only", str(ctx.exception))

    def test_resume_refuses_on_encoder_mismatch(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(30)])
            out = Path(tmp) / "index_out"
            encoder = self._crashing_encoder(8, crash_after_batches=1)
            with self.assertRaises(RuntimeError):
                encode_index_streaming(
                    corpus, out, encoder, batch_size=10, checkpoint_every_batches=1,
                )

            class DifferentEncoder:
                name = "a-different-encoder"

                def encode(self, texts):
                    return HashingEncoder(dim=8).encode(texts)

            with self.assertRaises(StreamingBuildError) as ctx:
                encode_index_streaming(
                    corpus, out, DifferentEncoder(), batch_size=10, resume=True,
                )
        self.assertIn("encoder", str(ctx.exception))


class SafetyGuardTests(unittest.TestCase):

    def test_refuses_to_build_into_a_directory_with_a_complete_index(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(10)])
            out = Path(tmp) / "index_out"
            encode_index_streaming(corpus, out, HashingEncoder(dim=8))
            with self.assertRaises(StreamingBuildError) as ctx:
                encode_index_streaming(corpus, out, HashingEncoder(dim=8))
        self.assertIn("already exists", str(ctx.exception))

    def test_a_degenerate_all_zero_encoder_is_caught_by_the_final_check(self):
        class ZeroEncoder:
            name = "zero-encoder"

            def encode(self, texts):
                return np.zeros((len(texts), 8), dtype=np.float32)

        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(10)])
            out = Path(tmp) / "index_out"
            with self.assertRaises(StreamingBuildError) as ctx:
                encode_index_streaming(corpus, out, ZeroEncoder())
        self.assertIn("all-zero", str(ctx.exception))
        # And it must not leave a manifest.json behind for a caller to
        # mistake for a valid, complete index.
        self.assertFalse((out / "manifest.json").exists())


class PlanCachingTests(unittest.TestCase):

    def test_plan_is_reused_when_the_corpus_file_is_unchanged(self):
        with TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "corpus", [chunk(i) for i in range(200)])
            out = Path(tmp) / "index_out"

            calls = {"n": 0}
            import experiments.shared.retrieval.streaming_index_build as mod
            original = mod.StreamingCorpusReader.__iter__

            def counting_iter(self):
                calls["n"] += 1
                return original(self)

            mod.StreamingCorpusReader.__iter__ = counting_iter
            try:
                plan_index_build(corpus, out)
                first_calls = calls["n"]
                plan_index_build(corpus, out)  # unchanged corpus: must be cached
                second_calls = calls["n"]
            finally:
                mod.StreamingCorpusReader.__iter__ = original

        self.assertEqual(first_calls, 1)
        self.assertEqual(second_calls, 1, "a second plan call re-read the corpus "
                                          "instead of reusing the cached plan")


if __name__ == "__main__":
    unittest.main()
