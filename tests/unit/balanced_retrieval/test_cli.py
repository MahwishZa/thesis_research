"""Tests of src/balanced_retrieval/cli.py: the five commands, run in this process with stand-ins for the models.

The stand-in encoder turns a text into word counts over a small vocabulary, so a passage is found by the words it shares
with the query; the stand-in cross-encoder scores a passage by the question words it contains. That makes the
outcome of a run predictable. The tests need numpy and skip without it.
"""

import contextlib
import gc
import io
import json
import unittest

from balanced_retrieval.cli import main
from rationale_query.cli import Streams
from rationale_query.cli import main as rationale_main
from rationale_query.store import RationaleRecord, RationaleStore, question_hash
from tests.support import QUESTIONS, TempDirTestCase, np, question_lines

VOCABULARY = ["donepezil", "amyloid", "insulin", "aspirin", "memory", "warfarin"]

DOCUMENTS = {
    "pubmed": [("PMID:1", "donepezil improves memory"), ("PMID:2", "amyloid plaques form in the brain"), ("PMID:3", "insulin lowers glucose")],
    "pmc": [("PMID:1", "donepezil trial memory outcome"), ("PMC:9", "memory clinic cohort")],
    "guidelines": [("G1", "aspirin is not recommended for dementia prevention"), ("G2", "warfarin interacts with many drugs")],
    "textbooks": [("T1", "donepezil is a cholinesterase inhibitor"), ("T2", "amyloid and tau in alzheimer disease")],
}

RATIONALES = {
    "q1": "memory loss and amyloid plaques point to alzheimer disease",
    "q2": "donepezil donepezil is the cholinesterase inhibitor",
    "q3": "amyloid amyloid beta forms plaques",
}


class BagEncoder:
    """Word counts over VOCABULARY; stands in for both MedCPT encoders."""

    max_length = 512

    def __init__(self, name="ncbi/MedCPT-Article-Encoder", **info):
        self.info = {"encoder": name, "revision": None, "max_length": 512, "pooling": "bag of words", "normalised": False, "dtype": "float32", "device": "cpu", "batch_size": 1}
        self.info.update(info)
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        return np.array([[text.lower().split().count(word) for word in VOCABULARY] for text in texts], dtype=np.float32)

    def token_counts(self, texts):
        return [len(text.split()) + 2 for text in texts]


class BagScorer:
    """The number of distinct query words that a passage contains."""

    device = "cpu"

    def __init__(self):
        self.info = {"cross_encoder": "fake-cross-encoder", "revision": None, "max_length": 512, "device": "cpu", "batch_size": 16}
        self.calls = []

    def __call__(self, query, candidates):
        self.calls.append((query, len(candidates)))
        words = set(query.lower().split())
        return [float(len(words & set(c["text"].lower().split()))) for c in candidates]


def run(argv, **factories):
    out, err = io.StringIO(), io.StringIO()
    status = main(argv, streams=Streams(out, err), **factories)
    return status, out.getvalue(), err.getvalue()


@unittest.skipIf(np is None, "numpy is not installed")
class CliTestCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(gc.collect)  # release memory maps before the folder is removed
        self.root = self.dir / "corpora"
        self.questions = self.write("questions.jsonl", question_lines())
        self.rationales = self.dir / "rationales"
        store = RationaleStore(self.rationales)
        store.bind({"model_name": "fake"}, [])
        for question in QUESTIONS:
            store.append(RationaleRecord(id=question.id, question_sha256=question_hash(question.text), status="ok", rationale=RATIONALES[question.id]))
        self.scorer = BagScorer()

    def build(self, corpora=("pubmed", "pmc", "guidelines", "textbooks"), words=100):
        for name in corpora:
            lines = "".join(json.dumps({"doc_key": key, "text": text}) + "\n" for key, text in DOCUMENTS[name])
            path = self.write(f"{name}.jsonl", lines)
            status, _, err = run(["chunk", "--corpus", name, "--documents", str(path), "--root", str(self.root), "--words", str(words), "--overlap", "10"])
            self.assertEqual(status, 0, err)

    def embed(self, name, **extra):
        return run(["embed", "--root", str(self.root), "--corpus", name, "--batch-size", "2", "--save-every", "2", "--quiet"], article_encoder_factory=lambda args: BagEncoder(**extra))

    def build_and_embed(self, corpora=("pubmed", "pmc", "guidelines", "textbooks")):
        self.build(corpora)
        for name in corpora:
            status, _, err = self.embed(name)
            self.assertEqual(status, 0, err)

    def retrieve(self, *extra, out="out", **factories):
        factories.setdefault("query_encoder_factory", lambda args: BagEncoder(args.query_encoder))
        factories.setdefault("scorer_factory", lambda args: self.scorer)
        argv = ["retrieve", "--questions", str(self.questions), "--rationales", str(self.rationales), "--root", str(self.root), "--out", str(self.dir / out), "--top-k", "1", "--quiet", *extra]
        return run(argv, **factories)

    def records(self, out="out"):
        text = (self.dir / out / "retrieved.jsonl").read_text(encoding="utf-8")
        return {r["id"]: r for r in map(json.loads, text.splitlines())}


class ChunkCommandTests(CliTestCase):
    def test_chunk_writes_a_corpus_and_says_what_it_made(self):
        path = self.write("docs.jsonl", '{"doc_key": "k", "text": "' + "word " * 150 + '"}\n{"text": "x"}\n{"doc_key": "e", "text": " "}\n')
        status, out, err = run(["chunk", "--corpus", "pmc", "--documents", str(path), "--root", str(self.root)])
        self.assertEqual(status, 0, err)
        self.assertIn("3 documents gave 3 passages of 128 words (overlap 32)", out)
        self.assertIn("1 documents without any word were skipped", out)
        self.assertIn("1 documents have no doc_key", out)
        self.assertIn("46,300,000", out)
        self.assertTrue((self.root / "pmc" / "manifest.json").exists())

    def test_an_existing_corpus_is_not_replaced_without_overwrite(self):
        self.build(("pmc",))
        path = self.write("again.jsonl", '{"doc_key": "k", "text": "alpha"}\n')
        argv = ["chunk", "--corpus", "pmc", "--documents", str(path), "--root", str(self.root)]
        status, _, err = run(argv)
        self.assertEqual(status, 1)
        self.assertIn("--overwrite", err)
        self.assertEqual(run([*argv, "--overwrite"])[0], 0)

    def test_bad_window_settings_and_missing_files_are_reported(self):
        path = self.write("docs.jsonl", '{"text": "alpha"}\n')
        status, _, err = run(["chunk", "--corpus", "pmc", "--documents", str(path), "--root", str(self.root), "--words", "4", "--overlap", "4"])
        self.assertEqual((status, "overlap" in err), (1, True))
        status, _, err = run(["chunk", "--corpus", "pmc", "--documents", str(self.dir / "absent.jsonl"), "--root", str(self.root)])
        self.assertEqual((status, "cannot read" in err), (1, True))

    def test_an_unknown_corpus_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            run(["chunk", "--corpus", "web", "--documents", "x", "--root", "y"])
        self.assertEqual(raised.exception.code, 2)


class EmbedAndCheckTests(CliTestCase):
    def test_check_lists_what_is_built_and_what_is_embedded(self):
        self.build(("pubmed", "textbooks"))
        self.assertEqual(self.embed("pubmed")[0], 0)
        status, out, err = run(["check", "--root", str(self.root)])
        self.assertEqual(status, 0, err)
        lines = {line.split(":")[0]: line for line in out.splitlines()}
        self.assertIn("pubmed: 3 passages from 3 documents (100 words, overlap 10); embedded", lines["pubmed"])
        self.assertIn("paper: 69,700,000", lines["pubmed"])
        self.assertIn("not embedded", lines["textbooks"])
        self.assertEqual(lines["pmc"], "pmc: not built")

    def test_check_without_any_corpus_is_an_error(self):
        status, _, err = run(["check", "--root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertIn("no corpus", err)

    def test_embed_writes_one_vector_per_passage_and_says_so(self):
        self.build(("pubmed",))
        status, out, err = self.embed("pubmed")
        self.assertEqual(status, 0, err)
        self.assertIn("pubmed: 3 vectors written", out)
        from balanced_retrieval.corpus import Corpus

        matrix = np.array(Corpus(self.root, "pubmed").embeddings())
        np.testing.assert_array_equal(matrix[0], [1, 0, 0, 0, 1, 0])  # "donepezil improves memory"

    def test_embed_again_does_not_encode_again(self):
        self.build(("pubmed",))
        self.embed("pubmed")
        encoders = []

        def factory(args):
            encoders.append(BagEncoder())
            return encoders[-1]

        status, _, _ = run(["embed", "--root", str(self.root), "--corpus", "pubmed", "--quiet"], article_encoder_factory=factory)
        self.assertEqual((status, encoders[0].encoded), (0, 0))

    def test_embed_with_other_settings_is_refused(self):
        self.build(("pubmed",))
        self.embed("pubmed")
        status, _, err = self.embed("pubmed", max_length=64)
        self.assertEqual(status, 1)
        self.assertIn("other settings", err)

    def test_embed_of_a_corpus_that_was_not_built_is_an_error(self):
        status, _, err = self.embed("pmc")
        self.assertEqual(status, 1)
        self.assertIn("no usable corpus", err)

    def test_embed_prints_progress_unless_quiet(self):
        self.build(("pubmed",))
        status, _, err = run(["embed", "--root", str(self.root), "--corpus", "pubmed", "--batch-size", "2"], article_encoder_factory=lambda a: BagEncoder())
        self.assertEqual(status, 0)
        self.assertIn("embedded 3 of 3 passages", err)
        self.assertIn("encoding on the cpu", err)


class PrecisionCheckCommandTests(CliTestCase):
    def test_both_kinds_of_arithmetic_are_asked_for_and_the_comparison_is_reported(self):
        self.build(("pubmed",))
        asked = []

        def factory(args):
            asked.append(args.compute_dtype)
            return BagEncoder()

        status, out, err = run(["precision-check", "--root", str(self.root), "--corpus", "pubmed", "--samples", "3"], article_encoder_factory=factory)
        self.assertEqual(status, 0, err)
        self.assertEqual(asked, ["float32", "float16"])
        self.assertIn("pubmed: 3 passages encoded twice on the cpu", out)
        self.assertIn("cosine between the two vectors of a passage: mean 1.000000, lowest 1.000000", out)
        self.assertIn("the same nearest passage 100.0% of the time", out)

    def test_embed_passes_the_chosen_arithmetic_to_the_encoder(self):
        self.build(("pubmed",))
        asked = []

        def factory(args):
            asked.append(args.compute_dtype)
            return BagEncoder()

        for extra, expected in (([], "float32"), (["--compute-dtype", "float16"], "float16")):
            asked.clear()
            status, out, err = run(["embed", "--root", str(self.root), "--corpus", "pubmed", "--quiet", *extra], article_encoder_factory=factory)
            self.assertEqual((status, asked), (0, [expected]), err)
            (self.root / "pubmed" / "embeddings.npy").unlink()
            (self.root / "pubmed" / "embeddings.json").unlink()
        self.assertIn("float16 arithmetic", out)


class RetrieveCommandTests(CliTestCase):
    def test_every_question_gets_its_best_passages_from_all_corpora(self):
        self.build_and_embed()
        status, out, err = self.retrieve()
        self.assertEqual(status, 0, err)
        records = self.records()
        self.assertEqual(sorted(records), ["q1", "q2", "q3"])
        for record in records.values():
            self.assertEqual(record["pool_size"], 4)  # one candidate from each corpus at k = 1
            self.assertEqual(len(record["evidence"]), 1)
        self.assertIn("3 questions: 0 already done, 3 written", out)

    def test_the_question_not_the_rationale_decides_the_final_order_by_default(self):
        self.build_and_embed()
        self.retrieve()
        best = self.records()["q2"]["evidence"][0]
        self.assertEqual((best["corpus"], best["doc_key"]), ("textbooks", "T1"))  # shares four words with the question
        self.assertEqual([call[0] for call in self.scorer.calls], [q.text for q in QUESTIONS])

    def test_the_rationale_variant_scores_against_the_rationale(self):
        self.build_and_embed()
        self.retrieve("--rerank-against", "rationale")
        self.assertTrue(all(call[0] in RATIONALES.values() for call in self.scorer.calls))

    def test_the_search_uses_the_rationale_alone_and_finds_passages_by_its_words(self):
        self.build_and_embed(("pubmed",))
        self.retrieve("--corpora", "pubmed")
        found = self.records()["q3"]["evidence"][0]  # "amyloid amyloid beta forms plaques" shares two counts with one passage
        self.assertEqual((found["doc_key"], found["row"], found["chunk_index"], found["dense_score"]), ("PMID:2", 1, 0, 2.0))

    def test_a_record_shows_where_each_passage_came_from_and_how_it_scored(self):
        self.build_and_embed()
        self.retrieve("--top-k", "2", out="two")
        for record in self.records("two").values():
            scores = [item["rerank_score"] for item in record["evidence"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for item in record["evidence"]:
                self.assertEqual(sorted(item), ["chunk_index", "corpus", "dense_score", "doc_key", "rerank_score", "row", "text"])
                self.assertIn(item["corpus"], DOCUMENTS)

    def test_a_subset_of_the_corpora_can_be_searched(self):
        self.build_and_embed(("pubmed", "pmc"))
        status, _, err = self.retrieve("--corpora", "pubmed", "pmc")
        self.assertEqual(status, 0, err)
        self.assertEqual({r["pool_size"] for r in self.records().values()}, {2})

    def test_a_stopped_run_continues_with_the_questions_that_are_missing(self):
        self.build_and_embed()
        self.assertEqual(self.retrieve("--limit", "2")[0], 0)
        self.assertEqual(sorted(self.records()), ["q1", "q2"])
        first = self.records()["q1"]
        status, out, err = self.retrieve()
        self.assertEqual(status, 0, err)
        self.assertIn("3 questions: 2 already done, 1 written", out)
        self.assertEqual(self.records()["q1"], first)
        self.assertEqual(sorted(self.records()), ["q1", "q2", "q3"])

    def test_a_finished_run_says_so_and_loads_no_model(self):
        self.build_and_embed()
        self.retrieve()

        def forbidden(args):
            raise AssertionError("no model should be loaded")

        status, out, _ = self.retrieve(query_encoder_factory=forbidden, scorer_factory=forbidden)
        self.assertEqual(status, 0)
        self.assertIn("nothing to do", out)
        self.assertIn("mean share of the final list", out)

    def test_other_settings_are_refused_for_the_same_folder(self):
        self.build_and_embed()
        self.assertEqual(self.retrieve("--limit", "1")[0], 0)
        status, _, err = self.retrieve("--top-k", "2")
        self.assertEqual(status, 1)
        self.assertIn("other settings", err)

    def test_other_settings_are_refused_even_when_nothing_is_left_to_do(self):
        self.build_and_embed()
        self.retrieve()
        for extra in (["--top-k", "2"], ["--rerank-against", "rationale"], ["--corpora", "pubmed"]):
            status, out, err = self.retrieve(*extra)
            with self.subTest(extra=extra):
                self.assertEqual(status, 1)
                self.assertIn("other settings", err)
                self.assertNotIn("nothing to do", out)

    def test_evidence_found_for_another_rationale_is_not_reused(self):
        self.build_and_embed()
        self.retrieve("--limit", "1")
        store = RationaleStore(self.rationales)
        store.load(repair=True)
        store.append(RationaleRecord(id="q1", question_sha256=question_hash(QUESTIONS[0].text), status="ok", rationale="insulin and warfarin"))
        status, _, err = self.retrieve()
        self.assertEqual(status, 1)
        self.assertIn("another rationale", err)

    def test_a_corpus_that_is_not_embedded_is_an_error_before_any_model_is_loaded(self):
        self.build(("pubmed", "pmc", "guidelines", "textbooks"))

        def forbidden(args):
            raise AssertionError("no model should be loaded")

        status, _, err = self.retrieve(query_encoder_factory=forbidden, scorer_factory=forbidden)
        self.assertEqual(status, 1)
        self.assertIn("not embedded", err)

    def test_the_query_vectors_of_part_1_are_reused_when_they_fit(self):
        self.build_and_embed()
        status = rationale_main(
            ["encode", "--rationales", str(self.rationales), "--questions", str(self.questions)],
            encoder_factory=lambda args: BagEncoder("ncbi/MedCPT-Query-Encoder"),
            streams=Streams(io.StringIO(), io.StringIO()),
        )
        self.assertEqual(status, 0)

        def forbidden(args):
            raise AssertionError("the query encoder should not be loaded")

        status, _, err = self.retrieve(query_encoder_factory=forbidden)
        self.assertEqual(status, 0, err)
        self.assertIn("using the query vectors", err)
        self.assertEqual(sorted(self.records()), ["q1", "q2", "q3"])

    def test_vectors_made_from_other_rationales_are_not_reused(self):
        self.build_and_embed()
        rationale_main(
            ["encode", "--rationales", str(self.rationales), "--questions", str(self.questions)],
            encoder_factory=lambda args: BagEncoder("ncbi/MedCPT-Query-Encoder"),
            streams=Streams(io.StringIO(), io.StringIO()),
        )
        store = RationaleStore(self.rationales)
        store.load(repair=True)
        store.append(RationaleRecord(id="q1", question_sha256=question_hash(QUESTIONS[0].text), status="ok", rationale="insulin"))
        encoders = []

        def factory(args):
            encoders.append(BagEncoder(args.query_encoder))
            return encoders[-1]

        status, _, err = self.retrieve(query_encoder_factory=factory)
        self.assertEqual(status, 0, err)
        self.assertNotIn("using the query vectors", err)
        self.assertEqual(encoders[0].encoded, 3)

    def test_a_bad_top_k_or_limit_is_an_error(self):
        self.build_and_embed()
        for option, value in (("--top-k", "0"), ("--limit", "0")):
            status, _, err = self.retrieve(option, value)
            self.assertEqual(status, 1, option)
            self.assertIn("at least 1", err)

    def test_missing_rationales_are_an_error(self):
        self.build_and_embed()
        status, _, err = run(
            ["retrieve", "--questions", str(self.questions), "--rationales", str(self.dir / "none"), "--root", str(self.root), "--out", str(self.dir / "o"), "--top-k", "1"]
        )
        self.assertEqual(status, 1)
        self.assertIn("rationale", err.lower())


class ReportCommandTests(CliTestCase):
    def test_report_describes_the_lists(self):
        self.build_and_embed()
        self.retrieve("--top-k", "2")
        status, out, err = run(["report", str(self.dir / "out"), "--json", str(self.dir / "r.json")])
        self.assertEqual(status, 0, err)
        self.assertIn("questions with evidence: 3", out)
        self.assertIn("mean share of the final list: pubmed", out)
        self.assertIn("mean duplicate rate across corpora", out)
        self.assertIn("settings: top_k 2, re-ranker reads the question", out)
        saved = json.loads((self.dir / "r.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["summary"]["questions"], 3)
        self.assertEqual(saved["config"]["top_k"], 2)
        self.assertTrue((self.dir / "out" / "summary.json").exists())

    def test_report_of_a_folder_without_records_is_an_error(self):
        status, _, err = run(["report", str(self.dir)])
        self.assertEqual(status, 1)
        self.assertIn("retrieved.jsonl", err)


if __name__ == "__main__":
    unittest.main()
