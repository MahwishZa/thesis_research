"""The upstream retrieval stage: corpus reading, index, retrieval, reranking.

No model is loaded anywhere here. The MedCPT wrappers import torch lazily, so
these tests exercise the whole stage on a machine without it, using the
deterministic stand-ins - which is the point of keeping those stand-ins
clearly labelled as never being a result.

The properties locked here are the ones the comparison depends on: one
candidate set per question, the same size for every question, built once,
deterministic, and never written back into the corpus.
"""

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.retrieval.corpus import (
    CorpusError, CorpusPassage, dated_only, parse_publication_date,
    read_passages, read_passages_with_snapshot, snapshot_id,
)
from experiments.retrieval.encoders import HashingEncoder, LexicalOverlapReranker
from experiments.retrieval.index import (
    DenseIndex, IndexError_, build_index, l2_normalize,
)
from experiments.retrieval.pipeline import (
    RetrievalConfig, RetrievalError, RetrievalPipeline,
)


def chunk(i, *, dated=True, retracted=False, text=None):
    return {
        "chunk_id": f"C-{i:03d}",
        "document_id": f"D-{i // 3}",
        "text": text or f"alzheimer passage number {i} about treatment",
        "retrieval_text": f"[TITLE] doc {i}\n{text or f'passage {i}'}",
        "publication_date": "2023-05" if dated else "",
        "source_tier": "peer_reviewed",
        "retracted": retracted,
        "section": "abstract",
        "claim_classes": ["AD-CRIT-01"],
    }


def write_corpus(root, records):
    chunks = Path(root) / "data" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return root


class CorpusReadingTests(unittest.TestCase):

    def test_reads_chunks_in_file_order(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(5)])
            passages = read_passages(tmp)
        self.assertEqual([p.chunk_id for p in passages],
                         [f"C-{i:03d}" for i in range(5)])

    def test_missing_corpus_says_step_one_must_finish(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(CorpusError) as ctx:
                read_passages(tmp)
        self.assertIn("Step 1", str(ctx.exception))

    def test_duplicate_evidence_ids_are_refused(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(1), chunk(1)])
            with self.assertRaises(CorpusError) as ctx:
                read_passages(tmp)
        self.assertIn("duplicate chunk_id", str(ctx.exception))

    def test_never_reads_the_whole_file_into_memory_at_once(self):
        """Regression test: ``Path.read_text().splitlines()`` loads the
        entire chunk file as one string, which raises
        ``OSError: [Errno 22] Invalid argument`` on Windows once the file
        exceeds ~2 GB (a real failure hit on the student's machine against
        the real 4.3M-chunk corpus). ``read_passages`` must stream the file
        line by line instead - this fails loudly if it ever regresses back
        to a whole-file read."""
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(5)])
            path = Path(tmp)

            original_read_text = Path.read_text

            def _guard(self, *a, **kw):
                if self.name == "chunks.jsonl":
                    raise AssertionError(
                        "read_passages must not read the whole chunk file "
                        "into memory with Path.read_text() - stream it line "
                        "by line instead (Windows >2GB OSError regression)"
                    )
                return original_read_text(self, *a, **kw)

            Path.read_text = _guard
            try:
                passages = read_passages(path)
            finally:
                Path.read_text = original_read_text
        self.assertEqual(len(passages), 5)

    def test_combined_reader_matches_the_standalone_snapshot_id(self):
        """read_passages_with_snapshot hashes the file line by line while
        parsing; snapshot_id hashes it in 1MB binary blocks. A streaming
        SHA-256's digest depends only on the byte sequence and its order,
        not the chunking, so these two must always agree."""
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(7)])
            passages, combined_snapshot, duplicates = read_passages_with_snapshot(tmp)
            standalone_snapshot = snapshot_id(tmp)
        self.assertEqual(combined_snapshot, standalone_snapshot)
        self.assertEqual(len(passages), 7)
        self.assertEqual(duplicates, ())

    def test_combined_reader_also_reads_only_once(self):
        """Same Windows->2GB guard as read_passages, for the function
        build_index.py actually calls."""
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(5)])
            path = Path(tmp)

            original_read_text = Path.read_text

            def _guard(self, *a, **kw):
                if self.name == "chunks.jsonl":
                    raise AssertionError(
                        "read_passages_with_snapshot must not read the whole "
                        "chunk file into memory with Path.read_text()"
                    )
                return original_read_text(self, *a, **kw)

            Path.read_text = _guard
            try:
                passages, _, _ = read_passages_with_snapshot(path)
            finally:
                Path.read_text = original_read_text
        self.assertEqual(len(passages), 5)

    def test_combined_reader_still_catches_duplicate_ids_by_default(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(1), chunk(1)])
            with self.assertRaises(CorpusError) as ctx:
                read_passages_with_snapshot(tmp)
        self.assertIn("duplicate chunk_id", str(ctx.exception))

    def test_rejects_an_unknown_on_duplicate_policy(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(1)])
            with self.assertRaises(CorpusError):
                read_passages_with_snapshot(tmp, on_duplicate="something_else")

    def test_keep_first_keeps_the_first_occurrence_and_reports_the_rest(self):
        """The actual policy diagnosed against the real corpus's duplicate
        chunk_ids (2026-09-21g): keep file order, never drop silently."""
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [
                chunk(1, text="first version of the text"),
                chunk(2),
                chunk(1, text="second, different version"),  # same chunk_id
            ])
            passages, _, duplicates = read_passages_with_snapshot(
                tmp, on_duplicate="keep_first"
            )
        self.assertEqual(len(passages), 2)  # not 3 - the duplicate was dropped
        self.assertEqual(passages[0].text, "first version of the text")
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["chunk_id"], "C-001")
        self.assertEqual(duplicates[0]["kept_line"], 1)
        self.assertEqual(duplicates[0]["dropped_line"], 3)
        self.assertIn("second, different version",
                      duplicates[0]["dropped_text_preview"])

    def test_keep_first_reports_nothing_when_there_are_no_duplicates(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(5)])
            _, _, duplicates = read_passages_with_snapshot(
                tmp, on_duplicate="keep_first"
            )
        self.assertEqual(duplicates, ())

    def test_progress_callback_fires_at_the_configured_interval(self):
        seen = []
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(10)])
            read_passages_with_snapshot(
                tmp, on_progress=seen.append, progress_every=3
            )
        self.assertEqual(seen, [3, 6, 9])

    def test_no_progress_callback_by_default(self):
        """The library function stays silent unless a caller opts in -
        build_index.py wires the printing, not this module."""
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(5)])
            read_passages_with_snapshot(tmp)  # must not raise / require one

    def test_snapshot_id_tracks_content_not_a_typed_version(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(1)])
            first = snapshot_id(tmp)
            same = snapshot_id(tmp)
            (Path(tmp) / "data" / "chunks" / "chunks.jsonl").write_text(
                json.dumps(chunk(2)) + "\n", encoding="utf-8")
            changed = snapshot_id(tmp)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_partial_dates_parse_without_inventing_precision(self):
        self.assertEqual(parse_publication_date("2024"), date(2024, 1, 1))
        self.assertEqual(parse_publication_date("2024-08"), date(2024, 8, 1))
        self.assertEqual(parse_publication_date("2024-08-15"), date(2024, 8, 15))
        self.assertIsNone(parse_publication_date(""))
        self.assertIsNone(parse_publication_date(None))

    def test_dated_only_drops_undated_passages(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(1), chunk(2, dated=False), chunk(3)])
            kept = dated_only(read_passages(tmp))
        self.assertEqual([p.chunk_id for p in kept], ["C-001", "C-003"])

    def test_evidence_context_text_excludes_retrieval_markers(self):
        """Section headers help the retriever; they must not reach the arms."""
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(1)])
            evidence = read_passages(tmp)[0].to_evidence()
        self.assertNotIn("[TITLE]", evidence.text)
        self.assertEqual(evidence.publication_date, date(2023, 5, 1))

    def test_reading_does_not_write_to_the_corpus(self):
        with TemporaryDirectory() as tmp:
            write_corpus(tmp, [chunk(i) for i in range(3)])
            before = {p: p.stat().st_mtime_ns
                      for p in Path(tmp).rglob("*") if p.is_file()}
            read_passages(tmp)
            snapshot_id(tmp)
            after = {p: p.stat().st_mtime_ns
                     for p in Path(tmp).rglob("*") if p.is_file()}
        self.assertEqual(before, after)


class IndexTests(unittest.TestCase):

    def setUp(self):
        self.passages = tuple(
            CorpusPassage(chunk_id=f"C-{i}", document_id="D", text=f"text {i}",
                          retrieval_text=f"text {i}", publication_date="2023",
                          source_tier="t", retracted=False)
            for i in range(10)
        )
        self.encoder = HashingEncoder(dim=16)

    def test_build_and_search_is_deterministic(self):
        index = build_index(self.passages, self.encoder, corpus_snapshot="s")
        query = self.encoder.encode(["text 3"])[0]
        first = index.search(query, top_k=5)
        second = index.search(query, top_k=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_vectors_are_normalised_so_scores_are_cosine(self):
        index = build_index(self.passages, self.encoder, corpus_snapshot="s")
        norms = np.linalg.norm(index.vectors, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_identical_vectors_break_ties_by_corpus_order(self):
        duplicated = self.passages[:1] * 3
        duplicated = tuple(
            CorpusPassage(chunk_id=f"DUP-{i}", document_id="D", text="same",
                          retrieval_text="same", publication_date="2023",
                          source_tier="t", retracted=False)
            for i in range(3)
        )
        index = build_index(duplicated, self.encoder, corpus_snapshot="s")
        hits = index.search(self.encoder.encode(["same"])[0], top_k=3)
        self.assertEqual([row for row, _ in hits], [0, 1, 2])

    def test_a_mismatched_query_dimension_is_refused(self):
        index = build_index(self.passages, self.encoder, corpus_snapshot="s")
        with self.assertRaises(IndexError_) as ctx:
            index.search(np.zeros(4, dtype=np.float32), top_k=3)
        self.assertIn("does not match", str(ctx.exception))

    def test_round_trips_through_disk_unchanged(self):
        index = build_index(self.passages, self.encoder, corpus_snapshot="s@1")
        with TemporaryDirectory() as tmp:
            index.save(tmp)
            loaded = DenseIndex.load(tmp)
        self.assertEqual(loaded.passage_ids, index.passage_ids)
        self.assertEqual(loaded.corpus_snapshot, "s@1")
        self.assertEqual(loaded.encoder_name, "hashing-stub")
        self.assertTrue(np.array_equal(loaded.vectors, index.vectors))

    def test_manifest_records_the_encoder_and_snapshot(self):
        index = build_index(self.passages, self.encoder, corpus_snapshot="s@1")
        with TemporaryDirectory() as tmp:
            index.save(tmp)
            manifest = json.loads(
                (Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["corpus_snapshot"], "s@1")
        self.assertEqual(manifest["encoder_name"], "hashing-stub")
        self.assertEqual(manifest["n_passages"], 10)

    def test_on_progress_is_passed_through_to_an_encoder_that_accepts_it(self):
        """build_index() must forward on_progress to encoders that support
        it (MedCPTEncoder), without breaking ones that don't (HashingEncoder,
        used everywhere else in this test file - hence a dedicated stub
        here rather than torch, which this suite deliberately avoids)."""
        calls = []

        class ProgressAwareStub:
            name = "progress-stub"

            def encode(self, texts, *, on_progress=None):
                if on_progress is not None:
                    on_progress(1, 1, 0.01)
                return np.zeros((len(texts), 4), dtype=np.float32)

        build_index(self.passages, ProgressAwareStub(), corpus_snapshot="s",
                    on_progress=lambda *a: calls.append(a))
        self.assertEqual(calls, [(1, 1, 0.01)])

    def test_on_progress_omitted_still_works_with_an_encoder_that_lacks_it(self):
        """HashingEncoder.encode has no on_progress parameter; build_index()
        must not pass the keyword unless it was actually given one."""
        index = build_index(self.passages, self.encoder, corpus_snapshot="s")
        self.assertEqual(len(index.passage_ids), 10)

    def test_empty_corpus_is_refused(self):
        with self.assertRaises(IndexError_):
            build_index((), self.encoder, corpus_snapshot="s")

    def test_l2_normalize_survives_a_zero_vector(self):
        out = l2_normalize(np.zeros((1, 4), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(out)))


def make_pipeline(n=30, *, config=None, retracted_ids=()):
    passages = tuple(
        CorpusPassage(
            chunk_id=f"C-{i:03d}", document_id=f"D-{i}",
            text=f"alzheimer evidence {i} donepezil memantine",
            retrieval_text=f"alzheimer evidence {i}",
            publication_date="2023-05", source_tier="peer_reviewed",
            retracted=(f"C-{i:03d}" in retracted_ids),
        )
        for i in range(n)
    )
    encoder = HashingEncoder(dim=16)
    index = build_index(passages, encoder, corpus_snapshot="corpus@test")
    return RetrievalPipeline(
        index=index, passages=passages, query_encoder=encoder,
        reranker=LexicalOverlapReranker(),
        config=config or RetrievalConfig(retrieval_depth=20,
                                         candidate_count=5),
    )


class PipelineTests(unittest.TestCase):

    def test_builds_a_candidate_set_of_exactly_the_configured_size(self):
        pipeline = make_pipeline()
        result = pipeline.build_candidate_set(
            question_id="ADQ-001", question="does donepezil help?")
        self.assertEqual(len(result.candidates), 5)

    def test_rerank_ranks_are_contiguous_from_one(self):
        """rho is a within-set rank; a gap would distort it silently."""
        pipeline = make_pipeline()
        result = pipeline.build_candidate_set(
            question_id="ADQ-001", question="does donepezil help?")
        self.assertEqual([c.rerank_rank for c in result.candidates],
                         [1, 2, 3, 4, 5])

    def test_the_same_question_produces_the_same_set_twice(self):
        pipeline = make_pipeline()
        a = pipeline.build_candidate_set(question_id="Q", question="memantine")
        b = pipeline.build_candidate_set(question_id="Q", question="memantine")
        self.assertEqual([c.evidence_id for c in a.candidates],
                         [c.evidence_id for c in b.candidates])
        self.assertEqual([c.rerank_score for c in a.candidates],
                         [c.rerank_score for c in b.candidates])

    def test_a_short_candidate_set_is_refused_not_padded(self):
        """A shorter set would put that question on a different theta scale."""
        pipeline = make_pipeline(n=6, config=RetrievalConfig(
            retrieval_depth=6, candidate_count=6),
            retracted_ids={"C-000", "C-001"})
        with self.assertRaises(RetrievalError) as ctx:
            pipeline.build_candidate_set(question_id="ADQ-9", question="q")
        self.assertIn("same candidate-set size", str(ctx.exception))

    def test_retracted_passages_are_excluded_by_default(self):
        pipeline = make_pipeline(retracted_ids={f"C-{i:03d}" for i in range(5)})
        result = pipeline.build_candidate_set(question_id="Q", question="q")
        ids = {c.evidence_id for c in result.candidates}
        self.assertFalse(ids & {f"C-{i:03d}" for i in range(5)})

    def test_provenance_records_both_encoders_and_the_reranker(self):
        pipeline = make_pipeline()
        result = pipeline.build_candidate_set(question_id="Q", question="q")
        self.assertEqual(result.provenance["query_encoder"], "hashing-stub")
        self.assertEqual(result.provenance["article_encoder"], "hashing-stub")
        self.assertEqual(result.provenance["reranker"], "lexical-overlap-stub")
        self.assertEqual(result.corpus_snapshot, "corpus@test")

    def test_candidates_carry_the_dates_the_proposed_policy_needs(self):
        pipeline = make_pipeline()
        result = pipeline.build_candidate_set(question_id="Q", question="q")
        for candidate in result.candidates:
            self.assertEqual(candidate.publication_date, "2023-05")

    def test_rationale_querying_is_opt_in_and_requires_a_rationale(self):
        pipeline = make_pipeline(config=RetrievalConfig(
            retrieval_depth=20, candidate_count=5, query_with_rationale=True))
        with self.assertRaises(RetrievalError) as ctx:
            pipeline.build_candidate_set(question_id="Q", question="q")
        self.assertIn("no rationale", str(ctx.exception))

    def test_rationale_is_used_for_retrieval_when_configured(self):
        pipeline = make_pipeline(config=RetrievalConfig(
            retrieval_depth=20, candidate_count=5, query_with_rationale=True))
        result = pipeline.build_candidate_set(
            question_id="Q", question="q", rationale="a longer rationale")
        self.assertEqual(result.query_text, "a longer rationale")
        self.assertTrue(result.provenance["config"]["query_with_rationale"])

    def test_index_and_passage_order_must_agree(self):
        pipeline = make_pipeline()
        with self.assertRaises(RetrievalError) as ctx:
            RetrievalPipeline(
                index=pipeline.index,
                passages=tuple(reversed(pipeline.passages)),
                query_encoder=pipeline.query_encoder,
                reranker=pipeline.reranker, config=pipeline.config)
        self.assertIn("does not match the index", str(ctx.exception))


class RetrievalConfigTests(unittest.TestCase):

    def test_depth_below_candidate_count_is_refused(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(retrieval_depth=5, candidate_count=20)

    def test_defaults_match_the_documented_engineering_constants(self):
        config = RetrievalConfig()
        self.assertEqual(config.retrieval_depth, 50)
        self.assertEqual(config.candidate_count, 20)
        self.assertTrue(config.exclude_retracted)
        self.assertFalse(config.query_with_rationale)


if __name__ == "__main__":
    unittest.main()
