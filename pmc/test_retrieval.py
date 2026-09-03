#!/usr/bin/env python3
"""Tests for embed_chunks.py and retrieve.py.

Run from the pmc/ directory:
    cd pmc && python3 -m unittest test_retrieval

Offline: uses the deterministic stub encoder, no models, no network, no GPU.
The production MedCPT paths are import-guarded and not exercised here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import embed_chunks as ec
import retrieve as rt


def chunk(cid, cat="pmc-fulltext", text="alpha beta gamma", **kw):
    base = {
        "chunk_id": cid, "document_id": cid.split("#")[0],
        "pmcid": "PMC1", "pmid": "111", "doi": "10.1/x", "title": "T",
        "source_category": cat, "eligibility_status": "eligible",
        "canonical_date": "2024-08", "date_precision": "month",
        "date_source": "jats:epub", "split_june_2024": "post",
        "authority_tier_label": "", "guideline_family": "", "in_currency_pack": "no",
        "retracted": "no", "license_code": "CC BY", "license_band": "open",
        "location": "body", "section_id": "s1", "section_heading": "Intro",
        "imrad": "introduction", "word_count": 3, "text_sha256": "h",
        "duplicate_of": "", "text": text,
    }
    base.update(kw)
    return base


def write_chunks(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


class StubEncoderDeterminism(unittest.TestCase):
    def test_same_text_same_vector(self):
        e = ec.StubEncoder(16)
        self.assertEqual(e.encode(["hello"]), e.encode(["hello"]))

    def test_different_text_different_vector(self):
        e = ec.StubEncoder(16)
        self.assertNotEqual(e.encode(["a"]), e.encode(["b"]))

    def test_vectors_are_unit_norm_and_right_dim(self):
        v = ec.StubEncoder(32).encode(["some text"])[0]
        self.assertEqual(len(v), 32)
        self.assertAlmostEqual(sum(x * x for x in v) ** 0.5, 1.0, places=5)

    def test_stub_is_marked_non_production(self):
        self.assertFalse(ec.StubEncoder(8).production)


class ModelPinning(unittest.TestCase):
    def test_medcpt_models_are_the_pinned_ones(self):
        self.assertEqual(ec.ARTICLE_ENCODER, "ncbi/MedCPT-Article-Encoder")
        self.assertEqual(ec.QUERY_ENCODER, "ncbi/MedCPT-Query-Encoder")
        self.assertEqual(ec.CROSS_ENCODER, "ncbi/MedCPT-Cross-Encoder")
        self.assertEqual(ec.MEDCPT_DIM, 768)

    def test_stub_index_refused_without_explicit_flag(self):
        with self.assertRaises(SystemExit):
            ec.main(["--encoder", "stub", "--chunks", "/nonexistent"])


class IndexBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.chunks = self.tmp / "chunks.jsonl"
        write_chunks(self.chunks, [
            chunk("PMC1.1#s1.w1", "pmc-fulltext", "amyloid biomarker evidence"),
            chunk("PMC1.1#s1.w2", "pmc-fulltext", "tau imaging findings"),
            chunk("PMC2.1#abs.w1", "pubmed-abstract", "plasma p-tau217 assay"),
            chunk("PMC3.1#abs.w1", "currency-pack", "anti-amyloid contested efficacy"),
            chunk("PMC4.1#s1.w1", "pmc-fulltext", "duplicate text", duplicate_of="PMC1.1#s1.w1"),
        ])
        self.out = self.tmp / "index"

    def build(self, out=None, skip_dupes=True):
        return ec.build(self.chunks, out or self.out, ec.StubEncoder(16), 2,
                        skip_duplicates=skip_dupes)

    def test_vector_count_matches_manifest(self):
        meta = self.build()
        idx = rt.Index(self.out)
        self.assertEqual(idx.n, meta["vectors"])
        self.assertEqual(len(idx.rows), meta["vectors"])

    def test_duplicates_skipped_by_default_but_chunk_layer_untouched(self):
        meta = self.build()
        self.assertEqual(meta["vectors"], 4)                    # the dupe is skipped
        self.assertEqual(len(self.chunks.read_text().splitlines()), 5)   # source intact

    def test_duplicates_included_on_request(self):
        meta = self.build(self.tmp / "i2", skip_dupes=False)
        self.assertEqual(meta["vectors"], 5)

    def test_rows_are_sorted_and_contiguous(self):
        self.build()
        idx = rt.Index(self.out)
        ids = [r["chunk_id"] for r in idx.rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual([r["row"] for r in idx.rows], list(range(idx.n)))

    def test_no_orphan_vectors_or_missing_mappings(self):
        self.build()
        idx = rt.Index(self.out)
        raw = (self.out / "embeddings.f32").read_bytes()
        self.assertEqual(len(raw), idx.n * idx.dim * 4)          # exact, no stragglers
        for r in idx.rows:
            self.assertTrue(r["chunk_id"])

    def test_provenance_fields_survive_into_the_manifest(self):
        self.build()
        idx = rt.Index(self.out)
        for r in idx.rows:
            for f in ["canonical_date", "date_precision", "split_june_2024",
                      "source_category", "authority_tier_label", "in_currency_pack",
                      "retracted", "pmid", "pmcid", "document_id", "text_sha256"]:
                self.assertIn(f, r)

    def test_manifest_does_not_carry_chunk_text(self):
        # the index is a mapping layer; text stays in the frozen chunk file
        self.build()
        self.assertNotIn("text", rt.Index(self.out).rows[0])

    def test_build_is_deterministic(self):
        a = self.build(self.tmp / "d1")
        b = self.build(self.tmp / "d2")
        self.assertEqual(a["content_digest"], b["content_digest"])
        self.assertEqual((self.tmp / "d1" / "embeddings.f32").read_bytes(),
                         (self.tmp / "d2" / "embeddings.f32").read_bytes())
        self.assertEqual((self.tmp / "d1" / "index_manifest.jsonl").read_bytes(),
                         (self.tmp / "d2" / "index_manifest.jsonl").read_bytes())

    def test_meta_records_stub_as_non_production(self):
        meta = self.build()
        self.assertFalse(meta["production"])
        self.assertIn("exact flat scan", meta["search"])


class Retrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.chunks = self.tmp / "chunks.jsonl"
        rows = []
        for i in range(6):
            rows.append(chunk(f"PMCA.1#s1.w{i}", "pmc-fulltext", f"body text {i}"))
        for i in range(6):
            rows.append(chunk(f"PMCB.1#abs.w{i}", "pubmed-abstract", f"abstract text {i}"))
        rows.append(chunk("PMCC.1#abs.w1", "currency-pack", "contested anti-amyloid"))
        write_chunks(self.chunks, rows)
        self.out = self.tmp / "index"
        ec.build(self.chunks, self.out, ec.StubEncoder(16), 4)
        self.index = rt.Index(self.out)
        self.q = ec.StubEncoder(16).encode(["body text 1"])[0]

    def test_balanced_quota_per_category(self):
        cands = rt.balanced_retrieve(self.index, self.q, per_category=2)
        got = {}
        for c in cands:
            got[c["retrieved_from_category"]] = got.get(c["retrieved_from_category"], 0) + 1
        self.assertEqual(got["pmc-fulltext"], 2)
        self.assertEqual(got["pubmed-abstract"], 2)
        self.assertEqual(got["currency-pack"], 1)     # only one exists; quota is a cap

    def test_small_category_is_not_drowned_out(self):
        cands = rt.balanced_retrieve(self.index, self.q, per_category=3)
        self.assertIn("currency-pack", {c["retrieved_from_category"] for c in cands})

    def test_retrieval_is_deterministic(self):
        a = rt.balanced_retrieve(self.index, self.q, 3)
        b = rt.balanced_retrieve(self.index, self.q, 3)
        self.assertEqual([c["chunk_id"] for c in a], [c["chunk_id"] for c in b])

    def test_ranks_are_dense_and_ordered(self):
        cands = rt.balanced_retrieve(self.index, self.q, 3)
        self.assertEqual([c["retrieval_rank"] for c in cands],
                         list(range(1, len(cands) + 1)))
        scores = [c["retrieval_score"] for c in cands]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_candidates_keep_recency_and_authority_metadata(self):
        for c in rt.balanced_retrieve(self.index, self.q, 2):
            for f in ["canonical_date", "date_precision", "split_june_2024",
                      "source_category", "in_currency_pack", "retracted"]:
                self.assertIn(f, c)

    def test_query_dim_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            rt.score_all(self.index, [0.1, 0.2])

    def test_ties_break_deterministically_on_chunk_id(self):
        scores = [1.0] * self.index.n              # everything tied
        got = rt.top_k(self.index, scores, 4)
        ids = [self.index.rows[i]["chunk_id"] for i, _ in got]
        self.assertEqual(ids, sorted(ids))


class CandidateReplay(unittest.TestCase):
    """Validity control V3: the candidate set must replay byte-identically."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        chunks = self.tmp / "chunks.jsonl"
        write_chunks(chunks, [chunk(f"PMCA.1#s1.w{i}", "pmc-fulltext", f"t {i}")
                              for i in range(5)])
        self.out = self.tmp / "index"
        ec.build(chunks, self.out, ec.StubEncoder(16), 4)
        self.index = rt.Index(self.out)
        self.q = ec.StubEncoder(16).encode(["t 2"])[0]
        self.cands = rt.balanced_retrieve(self.index, self.q, 3)
        self.path = self.tmp / "cand" / "q1.json"

    def test_save_then_replay_round_trips(self):
        saved = rt.save_candidates(self.path, "q1", "t 2", self.cands,
                                   self.index.meta, {"per_category": 3})
        back = rt.replay_candidates(self.path)
        self.assertEqual(back["candidate_digest"], saved["candidate_digest"])
        self.assertEqual([c["chunk_id"] for c in back["candidates"]],
                         [c["chunk_id"] for c in self.cands])

    def test_fresh_retrieval_matches_saved_set(self):
        rt.save_candidates(self.path, "q1", "t 2", self.cands, self.index.meta, {})
        again = rt.balanced_retrieve(self.index, self.q, 3)
        self.assertTrue(rt.verify_replay(self.path, again))

    def test_different_candidate_population_is_detected(self):
        rt.save_candidates(self.path, "q1", "t 2", self.cands, self.index.meta, {})
        smaller = rt.balanced_retrieve(self.index, self.q, 1)
        self.assertFalse(rt.verify_replay(self.path, smaller))

    def test_tampered_candidate_file_is_rejected(self):
        rt.save_candidates(self.path, "q1", "t 2", self.cands, self.index.meta, {})
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["candidates"][0]["chunk_id"] = "PMCX.1#s9.w9"
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            rt.replay_candidates(self.path)

    def test_digest_depends_on_order_not_just_membership(self):
        reordered = list(reversed(self.cands))
        self.assertNotEqual(rt.candidate_digest(self.cands),
                            rt.candidate_digest(reordered))

    def test_saved_payload_records_index_identity(self):
        saved = rt.save_candidates(self.path, "q1", "t 2", self.cands,
                                   self.index.meta, {})
        self.assertEqual(saved["index"]["content_digest"],
                         self.index.meta["content_digest"])
        self.assertIs(saved["index"]["production"], False)

    def test_corrupt_index_is_detected(self):
        (self.out / "embeddings.f32").write_bytes(b"\x00" * 8)
        with self.assertRaises(SystemExit):
            rt.Index(self.out)


if __name__ == "__main__":
    unittest.main()
