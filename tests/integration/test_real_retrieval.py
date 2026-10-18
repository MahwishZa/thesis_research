"""Tests of src/balanced_retrieval against the real libraries, using tiny synthetic BERT models (tests/integration/tiny_models.py).

They need torch, transformers and numpy; a class skips, with the reason, when a package is missing. Nothing is
downloaded and no real model is needed. They check that the encoders, the cross-encoder and the whole retrieve command
do with real torch what the stand-ins of the unit tests assume, and that they agree with the authors' own lines of code.
"""

import contextlib
import gc
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from balanced_retrieval.cli import main
from rationale_query.cli import Streams
from rationale_query.store import RationaleRecord, RationaleStore, question_hash
from tests.integration.test_real_libraries import unusable
from tests.integration.tiny_models import write_tiny_bert, write_tiny_cross_encoder
from tests.support import QUESTIONS, question_lines

DOCUMENTS = {
    "pubmed": ["the patient has memory loss", "alzheimer disease is likely", "donepezil option a"],
    "pmc": ["memory loss and alzheimer disease", "the patient has answer b"],
    "guidelines": ["donepezil is likely", "option c option d"],
    "textbooks": ["alzheimer disease the patient", "memory loss ?"],
}
RATIONALES = {"q1": "memory loss and alzheimer disease", "q2": "donepezil is likely", "q3": "the patient has alzheimer disease"}


class TinyRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = unusable("torch", "transformers", "numpy")
        if reason:
            raise unittest.SkipTest(reason)
        import numpy
        import torch

        cls.np, cls.torch = numpy, torch
        cls.base = Path(tempfile.mkdtemp(prefix="br tiny "))
        cls.encoder_folder = cls.base / "tiny encoder"
        cls.cross_folder = cls.base / "tiny cross encoder"
        write_tiny_bert(cls.encoder_folder)
        write_tiny_cross_encoder(cls.cross_folder)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        shutil.rmtree(cls.base, ignore_errors=True)

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="br work ", dir=self.base))
        self.addCleanup(gc.collect)

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        status = main(argv, streams=Streams(out, err))
        return status, out.getvalue(), err.getvalue()

    def test_the_article_encoder_gives_the_first_token_vectors_of_the_authors_code(self):
        from transformers import AutoModel, AutoTokenizer

        from balanced_retrieval.embedding import ArticleEncoder

        texts = list(DOCUMENTS["pubmed"])
        encoder = ArticleEncoder.from_pretrained(str(self.encoder_folder), device="cpu", local_files_only=True)
        ours = encoder.encode(texts)
        tokenizer = AutoTokenizer.from_pretrained(str(self.encoder_folder))
        model = AutoModel.from_pretrained(str(self.encoder_folder)).eval()
        with self.torch.no_grad():  # the lines of the authors' query_encode.py, with the texts of an article encoder
            theirs = [
                model(**tokenizer(texts[i : i + 1], truncation=True, padding=True, return_tensors="pt", max_length=512))
                .last_hidden_state[:, 0, :]
                .numpy()[0]
                for i in range(len(texts))
            ]
        self.assertTrue(self.np.array_equal(ours, self.np.vstack(theirs)))
        self.assertEqual((ours.dtype, ours.shape), (self.np.float32, (3, 32)))

    def test_half_precision_arithmetic_gives_vectors_close_to_the_full_kind_and_the_check_reports_it(self):
        root = self.work / "corpora"
        path = self.work / "pubmed.jsonl"
        path.write_text("".join(json.dumps({"doc_key": f"k{i}", "text": t}) + "\n" for i, t in enumerate(DOCUMENTS["pubmed"] + DOCUMENTS["pmc"])), encoding="utf-8", newline="\n")
        status, _, err = self.run_main(["chunk", "--corpus", "pubmed", "--documents", str(path), "--root", str(root)])
        self.assertEqual(status, 0, err)
        status, out, err = self.run_main(
            ["precision-check", "--root", str(root), "--corpus", "pubmed", "--samples", "5", "--encoder", str(self.encoder_folder), "--device", "cpu", "--local-files-only"]
        )
        self.assertEqual(status, 0, err)
        self.assertIn("5 passages encoded twice on the cpu", out)
        self.assertRegex(out, r"mean (0\.99\d+|1\.000000), lowest (0\.99\d+|1\.000000)")

    def test_the_cross_encoder_gives_the_scores_of_the_authors_code_whatever_the_batch_size(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from balanced_retrieval.scorer import CrossEncoderScorer

        candidates = [{"corpus": "pmc", "doc_key": str(i), "text": t} for i, t in enumerate(DOCUMENTS["pubmed"] + DOCUMENTS["pmc"])]
        query = "memory loss and alzheimer disease"
        tokenizer = AutoTokenizer.from_pretrained(str(self.cross_folder))
        model = AutoModelForSequenceClassification.from_pretrained(str(self.cross_folder)).eval()
        with self.torch.no_grad():  # the lines of the authors' rerank.py: the whole pool in one call
            encoded = tokenizer([[query, c["text"]] for c in candidates], truncation=True, padding=True, return_tensors="pt", max_length=512)
            theirs = model(**encoded).logits.squeeze(dim=1).numpy()
        for batch_size in (1, 2, 5):
            with self.subTest(batch_size=batch_size):
                scorer = CrossEncoderScorer.from_pretrained(str(self.cross_folder), device="cpu", batch_size=batch_size, local_files_only=True)
                self.np.testing.assert_allclose(scorer(query, candidates), theirs, atol=1e-5)

    def test_the_whole_command_chain_runs_on_real_torch_and_resumes(self):
        root = self.work / "corpora"
        for name, texts in DOCUMENTS.items():
            path = self.work / f"{name}.jsonl"
            path.write_text("".join(json.dumps({"doc_key": f"{name}:{i}", "text": t}) + "\n" for i, t in enumerate(texts)), encoding="utf-8", newline="\n")
            status, _, err = self.run_main(["chunk", "--corpus", name, "--documents", str(path), "--root", str(root)])
            self.assertEqual(status, 0, err)
            status, _, err = self.run_main(
                ["embed", "--root", str(root), "--corpus", name, "--encoder", str(self.encoder_folder), "--device", "cpu", "--local-files-only", "--batch-size", "2", "--quiet"]
            )
            self.assertEqual(status, 0, err)
        questions = self.work / "questions.jsonl"
        questions.write_text(question_lines(), encoding="utf-8", newline="\n")
        rationales = self.work / "rationales"
        store = RationaleStore(rationales)
        store.bind({"model_name": "tiny"}, [])
        for question in QUESTIONS:
            store.append(RationaleRecord(id=question.id, question_sha256=question_hash(question.text), status="ok", rationale=RATIONALES[question.id]))
        argv = [
            "retrieve", "--questions", str(questions), "--rationales", str(rationales), "--root", str(root), "--out", str(self.work / "out"),
            "--top-k", "2", "--query-encoder", str(self.encoder_folder), "--cross-encoder", str(self.cross_folder),
            "--device", "cpu", "--local-files-only", "--quiet",
        ]  # fmt: skip
        status, out, err = self.run_main([*argv, "--limit", "2"])
        self.assertEqual(status, 0, err)
        status, out, err = self.run_main(argv)
        self.assertEqual(status, 0, err)
        self.assertIn("3 questions: 2 already done, 1 written", out)
        records = [json.loads(line) for line in (self.work / "out" / "retrieved.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["id"] for r in records], ["q1", "q2", "q3"])
        for record in records:
            self.assertEqual((record["pool_size"], len(record["evidence"])), (8, 2))
            scores = [item["rerank_score"] for item in record["evidence"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
        config = json.loads((self.work / "out" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["top_k"], 2)
        self.assertEqual(config["cross_encoder"]["cross_encoder"], str(self.cross_folder))
        with contextlib.suppress(Exception):
            self.assertTrue(self.torch.is_grad_enabled())


if __name__ == "__main__":
    unittest.main()
