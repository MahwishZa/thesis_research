"""Stage 06's batched/streaming rewrite: proves it produces byte-identical
output to the original per-section encode/decode implementation, and that
its performance-relevant properties (call counts, memory, resumability)
actually hold.

No real tokenizer is available in this environment (no network to Hugging
Face), so a FakeTokenizer stands in - a simple, deterministic word-level
tokenizer that implements exactly the interface real use exercises
(__call__ for batched encode, .decode, .batch_decode, .is_fast). What is
under test is the *chunking and batching logic*, not MedCPT's specific
subword vocabulary - that part is fixed, external, and unaffected by this
refactor. The reference implementation below is the original script's
chunk_units(), preserved verbatim so the comparison is against what
actually shipped, not a re-derived approximation of it.
"""

import importlib.util
import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "corpus" / "scripts"


def load_module(name="06_chunk.py"):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                   SCRIPT_DIR / name)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quiet_logger(name="chunk_perf_test"):
    log = logging.getLogger(name)
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


class FakeTokenizer:
    """A deterministic word-level stand-in for a real HF tokenizer.

    Implements exactly the surface this script actually calls:
    __call__(texts, add_special_tokens=False, padding=False, truncation=False)
    -> {"input_ids": [[...], ...]}, .decode(ids), .batch_decode(list_of_ids),
    .is_fast. Each distinct word gets a stable integer id (vocabulary grows
    as new words are seen), so encode/decode round-trip exactly for the
    whitespace-tokenized text this test constructs - real subword behaviour
    is irrelevant to what's being tested here (windowing and call batching).
    """

    is_fast = True

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverse: dict[int, str] = {}
        self.calls = 0
        self.decode_calls = 0

    def _id_for(self, word: str) -> int:
        if word not in self.vocab:
            i = len(self.vocab)
            self.vocab[word] = i
            self.inverse[i] = word
        return self.vocab[word]

    def _encode_one(self, text: str) -> list[int]:
        return [self._id_for(w) for w in text.split()]

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False):
        assert add_special_tokens is False
        assert padding is False
        assert truncation is False
        self.calls += 1
        return {"input_ids": [self._encode_one(t) for t in texts]}

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        return self._encode_one(text)

    def decode(self, ids):
        self.decode_calls += 1
        return " ".join(self.inverse[i] for i in ids)

    def batch_decode(self, list_of_ids, skip_special_tokens=False):
        self.decode_calls += 1  # one batched call, not one per window
        return [" ".join(self.inverse[i] for i in ids) for ids in list_of_ids]


def reference_chunk_units(units, size, stride, tok):
    """The ORIGINAL implementation's chunk_units(), preserved verbatim as
    the ground truth this refactor must match exactly."""
    text = " ".join(u for u in units if u)
    if tok is None:
        toks = text.split()
        join = lambda xs: " ".join(xs)
    else:
        toks = tok.encode(text, add_special_tokens=False)
        join = lambda xs: tok.decode(xs)
    out = []
    i = 0
    while i < len(toks):
        piece = toks[i:i + size]
        out.append((join(piece), i, i + len(piece)))
        if i + size >= len(toks):
            break
        i += stride
    return out


class WindowBoundariesTests(unittest.TestCase):
    """Pure index arithmetic - no tokenizer needed to test it exhaustively."""

    def setUp(self):
        self.m = load_module()

    def test_matches_the_original_for_a_wide_range_of_lengths(self):
        tok = FakeTokenizer()
        for n_tokens in list(range(0, 20)) + [255, 256, 257, 480, 500, 1000, 2003]:
            words = [f"w{i}" for i in range(n_tokens)]
            text = " ".join(words)
            expected = reference_chunk_units([text], 256, 224, tok)
            expected_bounds = [(a, b) for (_, a, b) in expected]

            toks = tok._encode_one(text)
            actual_bounds = self.m.window_boundaries(len(toks), 256, 224)
            self.assertEqual(actual_bounds, expected_bounds,
                             f"mismatch at n_tokens={n_tokens}")

    def test_empty_section_yields_no_windows(self):
        self.assertEqual(self.m.window_boundaries(0, 256, 224), [])

    def test_a_short_section_yields_one_window_covering_all_tokens(self):
        self.assertEqual(self.m.window_boundaries(10, 256, 224), [(0, 10)])

    def test_the_last_window_is_never_dropped(self):
        # 300 tokens: window0=[0,256), window1=[224,300) - tail not lost.
        bounds = self.m.window_boundaries(300, 256, 224)
        self.assertEqual(bounds[-1][1], 300)

    def test_no_window_is_ever_skipped_or_duplicated(self):
        bounds = self.m.window_boundaries(1000, 256, 224)
        # Consecutive windows must advance by exactly `stride`, except
        # possibly the final one (which may be shorter).
        for (a1, _), (a2, _) in zip(bounds, bounds[1:]):
            self.assertEqual(a2 - a1, 224)


class BatchedEncodeDecodeTests(unittest.TestCase):

    def setUp(self):
        self.m = load_module()

    def test_encode_batch_makes_one_call_for_many_texts(self):
        tok = FakeTokenizer()
        texts = [f"word{i} word{i+1} word{i+2}" for i in range(50)]
        result = self.m.encode_batch(texts, tok)
        self.assertEqual(tok.calls, 1)
        self.assertEqual(len(result), 50)

    def test_decode_batch_makes_one_call_for_many_windows(self):
        tok = FakeTokenizer()
        tok._encode_one("a b c d e f g h")
        windows = [[0, 1, 2], [3, 4, 5], [6, 7]]
        result = self.m.decode_batch(windows, tok)
        self.assertEqual(tok.decode_calls, 1)
        self.assertEqual(len(result), 3)

    def test_padding_is_never_requested(self):
        """Padding would insert pad-token ids into input_ids and corrupt the
        true per-text lengths window_boundaries depends on."""
        tok = FakeTokenizer()
        self.m.encode_batch(["short", "a much longer piece of text here"], tok)
        # FakeTokenizer's __call__ asserts padding is False - no assertion
        # error means the contract held.


class ChunkBatchEquivalenceTests(unittest.TestCase):
    """The real test: batched chunk_batch() vs the original's per-section
    chunk_units(), run over the same synthetic documents, must agree
    exactly on every chunk's text, token_start, token_end and exception
    flag."""

    def setUp(self):
        self.m = load_module()

    def _reference_chunks(self, records, size, stride, tok, tok_used):
        """Reproduces the ORIGINAL script's per-record, per-section loop
        exactly, using reference_chunk_units()."""
        out = []
        for rec in records:
            doc_id = str(rec.get("document_id") or rec.get("pmid") or "")
            tier = rec.get("source_tier", "")
            sections = rec.get("sections") or [
                {"section": "title", "text": rec.get("title", "")},
                {"section": "abstract", "text": rec.get("abstract", "")}]
            for sec in sections:
                body = sec.get("text", "")
                if not body:
                    continue
                pieces = reference_chunk_units([body], size, stride, tok)
                exception = tier in self.m.STRUCTURE_AWARE and len(pieces) > 1
                for idx, (text, a, b) in enumerate(pieces):
                    out.append({
                        "chunk_id": f"{doc_id}#{sec.get('section','')}.{idx}",
                        "document_id": doc_id, "chunk_index": idx, "text": text,
                        "retrieval_text": f"[TITLE] {rec.get('title','')}\n"
                                          f"[SECTION] {sec.get('section','')}\n[TEXT] {text}",
                        "section": sec.get("section", ""),
                        "subsection": sec.get("subsection", ""),
                        "token_start": a, "token_end": b, "tokenizer_used": tok_used,
                        "publication_date": rec.get("publication_date", ""),
                        "source_tier": tier, "claim_classes": [],
                        "retracted": rec.get("retracted", ""),
                        "ad_relevant": rec.get("ad_relevant", ""),
                        "ad_relevance_score": rec.get("ad_relevance_score", ""),
                        "chunk_size_exception": exception,
                        "chunk_size_exception_reason":
                            "guideline recommendation kept with its qualifying conditions"
                            if exception else "",
                    })
        return out

    def _make_records(self, n=12):
        records = []
        for i in range(n):
            body_len = [5, 10, 300, 500, 1, 0, 260][i % 7]
            body = " ".join(f"tok{i}_{j}" for j in range(body_len))
            records.append({
                "document_id": f"DOC-{i}", "title": f"Title of document {i}",
                "sections": [
                    {"section": "title", "text": f"Title of document {i}"},
                    {"section": "abstract", "text": f"Abstract sentence {i}."},
                    {"section": "Results", "text": body},
                ] if body else [
                    {"section": "title", "text": f"Title of document {i}"},
                ],
                "publication_date": "2023", "source_tier": "peer_reviewed_primary",
                "retracted": False, "ad_relevant": True, "ad_relevance_score": 1.0,
            })
        return records

    def test_batched_output_matches_reference_exactly(self):
        records = self._make_records()
        tok = FakeTokenizer()
        expected = self._reference_chunks(records, 256, 224, tok, "fake-tok")

        tok2 = FakeTokenizer()
        actual = self.m.chunk_batch(records, 256, 224, tok2, "fake-tok")

        self.assertEqual(actual, expected)

    def test_batched_output_matches_reference_across_multiple_batches(self):
        """Chunking must not depend on batch boundaries - the same corpus
        split into two batches must produce the same chunks as one batch."""
        records = self._make_records(n=12)
        tok = FakeTokenizer()
        expected = self._reference_chunks(records, 256, 224, tok, "fake-tok")

        tok2 = FakeTokenizer()
        actual = (self.m.chunk_batch(records[:6], 256, 224, tok2, "fake-tok")
                 + self.m.chunk_batch(records[6:], 256, 224, tok2, "fake-tok"))
        self.assertEqual(actual, expected)

    def test_structure_aware_exception_flag_matches_reference(self):
        records = [{
            "document_id": "G-1", "title": "Guideline",
            "sections": [{"section": "recommendation",
                         "text": " ".join(f"w{i}" for i in range(600))}],
            "source_tier": "clinical_guideline", "publication_date": "2023",
            "retracted": False, "ad_relevant": True, "ad_relevance_score": 1.0,
        }]
        tok = FakeTokenizer()
        expected = self._reference_chunks(records, 256, 224, tok, "fake-tok")
        tok2 = FakeTokenizer()
        actual = self.m.chunk_batch(records, 256, 224, tok2, "fake-tok")
        self.assertEqual(actual, expected)
        self.assertTrue(any(c["chunk_size_exception"] for c in actual))

    def test_whitespace_tokenizer_mode_still_matches_reference(self):
        records = self._make_records()
        expected = self._reference_chunks(records, 256, 224, None, "whitespace")
        actual = self.m.chunk_batch(records, 256, 224, None, "whitespace")
        self.assertEqual(actual, expected)

    def test_empty_batch_yields_no_chunks(self):
        self.assertEqual(self.m.chunk_batch([], 256, 224, FakeTokenizer(), "t"), [])

    def test_call_count_is_two_per_batch_regardless_of_document_count(self):
        """The entire point of batching: O(1) tokenizer calls per batch,
        not O(documents) or O(sections)."""
        records = self._make_records(n=50)
        tok = FakeTokenizer()
        self.m.chunk_batch(records, 256, 224, tok, "fake-tok")
        self.assertEqual(tok.calls, 1)
        self.assertEqual(tok.decode_calls, 1)


class IterChunksStreamingTests(unittest.TestCase):
    """iter_chunks() is a generator - peak memory during iteration must stay
    bounded to one batch, and write_jsonl() must receive it as such."""

    def setUp(self):
        self.m = load_module()
        self.log = quiet_logger()

    def _records(self, n):
        for i in range(n):
            yield {"document_id": f"D-{i}", "title": f"Title {i}",
                  "sections": [{"section": "abstract", "text": f"content {i} words here"}],
                  "publication_date": "2023", "source_tier": "peer_reviewed_primary",
                  "retracted": False, "ad_relevant": True, "ad_relevance_score": 1.0}

    def test_is_a_generator_not_a_list(self):
        import types
        result = self.m.iter_chunks(self._records(5), 256, 224, FakeTokenizer(),
                                    "t", batch_size=2, log=self.log)
        self.assertIsInstance(result, types.GeneratorType)

    def test_yields_the_same_chunks_regardless_of_batch_size(self):
        tok_a, tok_b = FakeTokenizer(), FakeTokenizer()
        a = list(self.m.iter_chunks(self._records(10), 256, 224, tok_a, "t",
                                    batch_size=3, log=self.log))
        b = list(self.m.iter_chunks(self._records(10), 256, 224, tok_b, "t",
                                    batch_size=100, log=self.log))
        self.assertEqual(a, b)

    def test_write_jsonl_streams_without_materialising_a_full_list(self):
        """Integration: write_jsonl(OUT, generator) writes every record
        without the caller ever building a `chunks = [...]` list."""
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        import _common

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks.jsonl"
            n = _common.write_jsonl(
                out,
                self.m.iter_chunks(self._records(20), 256, 224, FakeTokenizer(),
                                   "t", batch_size=4, log=self.log),
            )
            self.assertEqual(n, 20)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 20)
            for line in lines:
                json.loads(line)  # every line is valid JSON on its own


class ResumeTests(unittest.TestCase):
    """Interruption safety: resuming must never duplicate or corrupt output,
    and must refuse to resume under different chunking parameters."""

    def setUp(self):
        self.m = load_module()

    def test_resume_with_matching_params_appends_without_duplicating(self):
        with TemporaryDirectory() as tmp:
            self.m.PROGRESS_FILE = Path(tmp) / ".chunk_progress.json"
            self.m.OUT = Path(tmp) / "chunks.jsonl"
            src = Path(tmp) / "documents.jsonl"
            src.write_text("\n".join(
                json.dumps({"document_id": f"D-{i}", "title": f"t{i}",
                           "sections": [{"section": "abstract", "text": f"x{i} y{i}"}],
                           "publication_date": "2023", "source_tier": "peer_reviewed_primary",
                           "retracted": False, "ad_relevant": True,
                           "ad_relevance_score": 1.0})
                for i in range(6)
            ), encoding="utf-8")

            # First "run": process only the first 4 documents, as if
            # interrupted after that batch.
            rc = self.m.main(["--input", str(src), "--tokenizer", "whitespace",
                              "--batch-size", "4"])
            self.assertEqual(rc, 0)
            first_run_lines = self.m.OUT.read_text(encoding="utf-8").splitlines()

            # Simulate interruption: truncate progress to pretend only the
            # first batch (4 docs) completed, and truncate OUT to match -
            # exactly the state a real Ctrl+C after one flushed batch would
            # leave (this test controls that state directly rather than
            # timing an actual interrupt).
            partial_chunks = [c for c in
                              (json.loads(l) for l in first_run_lines)
                              if c["document_id"] in {"D-0", "D-1", "D-2", "D-3"}]
            self.m.OUT.write_text(
                "\n".join(json.dumps(c, sort_keys=True) for c in partial_chunks) + "\n",
                encoding="utf-8")
            self.m._save_progress(4, self.m._run_params(
                self.m.argparse.Namespace(tokenizer="whitespace", size=256, overlap=32)))

            rc = self.m.main(["--input", str(src), "--tokenizer", "whitespace",
                              "--batch-size", "4", "--resume"])
            self.assertEqual(rc, 0)
            resumed = [json.loads(l) for l in
                      self.m.OUT.read_text(encoding="utf-8").splitlines()]
            doc_ids = [c["document_id"] for c in resumed]
            # Every document appears - none duplicated, none missing.
            self.assertEqual(sorted(set(doc_ids)), sorted({f"D-{i}" for i in range(6)}))
            self.assertEqual(len(doc_ids), len(set(
                (c["chunk_id"] for c in resumed))))  # chunk_ids are unique

    def test_resume_refuses_a_parameter_mismatch(self):
        with TemporaryDirectory() as tmp:
            self.m.PROGRESS_FILE = Path(tmp) / ".chunk_progress.json"
            self.m.OUT = Path(tmp) / "chunks.jsonl"
            self.m.OUT.write_text('{"chunk_id": "x"}\n', encoding="utf-8")
            self.m._save_progress(4, {"tokenizer": "whitespace", "size": 256, "overlap": 32})

            src = Path(tmp) / "documents.jsonl"
            src.write_text(json.dumps({"document_id": "D-0", "title": "t",
                                       "sections": [{"section": "a", "text": "x y"}],
                                       "publication_date": "2023",
                                       "source_tier": "peer_reviewed_primary",
                                       "retracted": False, "ad_relevant": True,
                                       "ad_relevance_score": 1.0}) + "\n",
                          encoding="utf-8")

            rc = self.m.main(["--input", str(src), "--tokenizer", "whitespace",
                              "--size", "128",  # different from the saved run
                              "--resume"])
            self.assertEqual(rc, 2)

    def test_resume_with_no_prior_progress_starts_fresh_not_an_error(self):
        with TemporaryDirectory() as tmp:
            self.m.PROGRESS_FILE = Path(tmp) / ".chunk_progress.json"
            self.m.OUT = Path(tmp) / "chunks.jsonl"
            src = Path(tmp) / "documents.jsonl"
            src.write_text(json.dumps({"document_id": "D-0", "title": "t",
                                       "sections": [{"section": "a", "text": "x y"}],
                                       "publication_date": "2023",
                                       "source_tier": "peer_reviewed_primary",
                                       "retracted": False, "ad_relevant": True,
                                       "ad_relevance_score": 1.0}) + "\n",
                          encoding="utf-8")
            rc = self.m.main(["--input", str(src), "--tokenizer", "whitespace",
                              "--resume"])
            self.assertEqual(rc, 0)
            self.assertTrue(self.m.OUT.exists())


class FixtureRegressionTests(unittest.TestCase):
    """Against whatever fixture chunks.jsonl is currently on disk under
    corpus/data/chunks/ - the same offline path the project
    already tests everything else through. That file is gitignored (see
    corpus/.gitignore: research data is never committed), so this
    test needs a prior local pipeline run to be meaningful - see the note
    at the top of test_corpus_normalize_pipeline.py."""

    def test_matches_the_local_chunks_output_for_the_fixture_corpus(self):
        import shutil, subprocess, sys as _sys
        with TemporaryDirectory() as tmp:
            copy = Path(tmp) / "corpus"
            shutil.copytree(ROOT / "corpus", copy,
                            ignore=shutil.ignore_patterns("__pycache__"))
            # Build the same deduplicated fixture the committed chunks.jsonl
            # was produced from.
            subprocess.run([_sys.executable, "scripts/04_normalize.py",
                           "--input", "data/raw/pubmed/records.example.jsonl"],
                          cwd=copy, check=True, capture_output=True)
            subprocess.run([_sys.executable, "scripts/05_deduplicate.py"],
                          cwd=copy, check=True, capture_output=True)
            result = subprocess.run(
                [_sys.executable, "scripts/06_chunk.py", "--tokenizer", "whitespace"],
                cwd=copy, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

            fresh = [json.loads(l) for l in
                    (copy / "data" / "chunks" / "chunks.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            committed = [json.loads(l) for l in
                        (ROOT / "corpus" / "data" / "chunks" / "chunks.jsonl")
                        .read_text(encoding="utf-8").splitlines()]
            # Stage 07's claim_classes/claim_confidence/claim_method (topical
            # dimension) and claim_evidence_levels/claim_evidence_confidence
            # (evidence-level dimension) differ because Stage 07 hasn't run
            # in this fresh copy, only in the local reference - same known
            # diff set as test_corpus_normalize_pipeline.py's equivalent
            # regression test for this fixture. ad_relevant/ad_relevance_score
            # are NOT in this set: both sides run the same current Stage 04,
            # so those fields agree.
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


if __name__ == "__main__":
    unittest.main()
