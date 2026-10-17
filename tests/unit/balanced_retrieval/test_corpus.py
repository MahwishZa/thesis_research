"""Tests of src/balanced_retrieval/corpus.py: writing and reading a corpus folder. They need numpy and skip without it."""

import gc
import hashlib
import json
import unittest

from tests.support import TempDirTestCase, np

try:
    from balanced_retrieval.corpus import (
        EMBEDDINGS_FILE,
        EMBEDDINGS_STATE,
        OFFSETS_FILE,
        PASSAGES_FILE,
        Corpus,
        CorpusError,
        read_documents,
        write_corpus,
    )
except ImportError:  # numpy is not installed
    Corpus = None


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


DOCUMENTS = [("PMID:1", words(130)), ("PMID:2", "short text"), ("", "no key here")]


@unittest.skipIf(Corpus is None, "numpy is not installed")
class CorpusTestCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(gc.collect)  # release memory maps before the folder is removed (Windows refuses otherwise)


class WriteCorpusTests(CorpusTestCase):
    def test_the_manifest_counts_documents_and_passages(self):
        manifest = write_corpus(self.dir, "pubmed", DOCUMENTS)
        self.assertEqual(manifest["corpus"], "pubmed")
        self.assertEqual(manifest["documents"], 3)
        self.assertEqual(manifest["documents_without_key"], 1)
        self.assertEqual(manifest["documents_without_words"], 0)
        self.assertEqual(manifest["passages"], 4)  # 130 words give two passages, the others one each
        self.assertEqual(manifest["chunking"], {"unit": "words", "size": 128, "overlap": 32})

    def test_the_manifest_is_saved_and_holds_the_hash_of_the_passage_file(self):
        manifest = write_corpus(self.dir, "pmc", DOCUMENTS)
        saved = json.loads((self.dir / "pmc" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)
        digest = hashlib.sha256((self.dir / "pmc" / PASSAGES_FILE).read_bytes()).hexdigest()
        self.assertEqual(manifest["passages_sha256"], digest)

    def test_the_same_documents_give_the_same_files(self):
        first = write_corpus(self.dir / "a", "pmc", DOCUMENTS)
        second = write_corpus(self.dir / "b", "pmc", DOCUMENTS)
        self.assertEqual(first, second)

    def test_documents_without_words_are_skipped_and_counted(self):
        manifest = write_corpus(self.dir, "pmc", [("k", "  "), ("j", "word")])
        self.assertEqual((manifest["passages"], manifest["documents_without_words"]), (1, 1))

    def test_a_corpus_without_any_passage_is_an_error_and_leaves_no_folder_content(self):
        with self.assertRaises(CorpusError):
            write_corpus(self.dir, "pmc", [("k", "  ")])
        self.assertFalse((self.dir / "pmc" / PASSAGES_FILE).exists())
        self.assertFalse((self.dir / "pmc" / (PASSAGES_FILE + ".tmp")).exists())

    def test_an_unknown_corpus_name_is_refused(self):
        with self.assertRaises(ValueError):
            write_corpus(self.dir, "web", DOCUMENTS)

    def test_an_existing_corpus_is_not_replaced_without_the_flag(self):
        write_corpus(self.dir, "pmc", DOCUMENTS)
        with self.assertRaises(CorpusError):
            write_corpus(self.dir, "pmc", DOCUMENTS)

    def test_overwriting_removes_the_embeddings_that_no_longer_fit(self):
        write_corpus(self.dir, "pmc", DOCUMENTS)
        (self.dir / "pmc" / EMBEDDINGS_FILE).write_bytes(b"x")
        (self.dir / "pmc" / EMBEDDINGS_STATE).write_text("{}", encoding="utf-8")
        write_corpus(self.dir, "pmc", [("k", "only one")], overwrite=True)
        self.assertFalse((self.dir / "pmc" / EMBEDDINGS_FILE).exists())
        self.assertFalse((self.dir / "pmc" / EMBEDDINGS_STATE).exists())
        self.assertEqual(len(Corpus(self.dir, "pmc")), 1)

    def test_other_window_settings_are_recorded(self):
        manifest = write_corpus(self.dir, "pmc", [("k", "a b c d e")], words=3, overlap=1)
        self.assertEqual(manifest["passages"], 2)
        self.assertEqual(manifest["chunking"], {"unit": "words", "size": 3, "overlap": 1})


class ReadCorpusTests(CorpusTestCase):
    def test_a_passage_comes_back_as_it_was_written(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        corpus = Corpus(self.dir, "pubmed")
        self.assertEqual(len(corpus), 4)
        self.assertEqual(corpus.passage(0), {"corpus": "pubmed", "doc_key": "PMID:1", "chunk_index": 0, "text": words(128)})
        self.assertEqual(corpus.passage(1)["text"], words(34, start=96))
        self.assertEqual(corpus.passage(1)["chunk_index"], 1)
        self.assertEqual(corpus.passage(3), {"corpus": "pubmed", "doc_key": "", "chunk_index": 0, "text": "no key here"})

    def test_text_with_quotes_unicode_and_line_breaks_survives(self):
        text = 'β-amyloid "plaques" ≥ 5 µm\nsecond line\twith a tab'
        write_corpus(self.dir, "pmc", [("k", text)])
        self.assertEqual(Corpus(self.dir, "pmc").passage(0)["text"], 'β-amyloid "plaques" ≥ 5 µm second line with a tab')

    def test_passages_come_in_the_order_asked(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        corpus = Corpus(self.dir, "pubmed")
        self.assertEqual([p["doc_key"] for p in corpus.passages([3, 0, 2])], ["", "PMID:1", "PMID:2"])
        self.assertEqual(corpus.passages([]), [])

    def test_numpy_row_numbers_are_accepted(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        corpus = Corpus(self.dir, "pubmed")
        self.assertEqual(corpus.passages(np.array([2, 1]))[0]["doc_key"], "PMID:2")

    def test_a_row_outside_the_corpus_or_of_the_wrong_kind_is_an_error(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        corpus = Corpus(self.dir, "pubmed")
        for bad in (-1, 4, 2.0, True, None):
            with self.subTest(row=bad), self.assertRaises(IndexError):
                corpus.passages([bad])

    def test_iter_passages_reads_from_a_start_row_to_the_end(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        corpus = Corpus(self.dir, "pubmed")
        self.assertEqual([p["doc_key"] for p in corpus.iter_passages()], ["PMID:1", "PMID:1", "PMID:2", ""])
        self.assertEqual([p["doc_key"] for p in corpus.iter_passages(2)], ["PMID:2", ""])
        self.assertEqual(list(corpus.iter_passages(4)), [])
        with self.assertRaises(IndexError):
            list(corpus.iter_passages(5))

    def test_a_folder_without_a_manifest_is_an_error(self):
        with self.assertRaises(CorpusError):
            Corpus(self.dir, "pubmed")

    def test_an_unknown_corpus_name_is_refused(self):
        with self.assertRaises(ValueError):
            Corpus(self.dir, "web")

    def test_offsets_that_do_not_match_the_manifest_are_an_error(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        np.save(self.dir / "pubmed" / OFFSETS_FILE, np.array([0, 5], dtype=np.uint64))
        with self.assertRaises(CorpusError):
            Corpus(self.dir, "pubmed")

    def test_the_embeddings_are_refused_until_the_corpus_is_embedded(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        corpus = Corpus(self.dir, "pubmed")
        self.assertIsNone(corpus.embedding_state())
        with self.assertRaises(CorpusError):
            corpus.embeddings()

    def test_an_embedding_matrix_with_the_wrong_number_of_rows_is_an_error(self):
        write_corpus(self.dir, "pubmed", DOCUMENTS)
        folder = self.dir / "pubmed"
        np.save(folder / EMBEDDINGS_FILE, np.zeros((3, 4), dtype=np.float32))
        (folder / EMBEDDINGS_STATE).write_text(json.dumps({"complete": True, "rows_done": 3}), encoding="utf-8")
        with self.assertRaises(CorpusError):
            Corpus(self.dir, "pubmed").embeddings()


class ReadDocumentsTests(CorpusTestCase):
    def test_documents_come_with_their_keys(self):
        path = self.write("docs.jsonl", '{"doc_key": "PMID:1", "text": "alpha"}\n\n{"text": "beta"}\n{"doc_key": 42, "text": "gamma"}\n{"doc_key": null, "text": "delta"}\n')
        self.assertEqual(list(read_documents(path)), [("PMID:1", "alpha"), ("", "beta"), ("42", "gamma"), ("", "delta")])

    def test_a_byte_order_mark_is_accepted(self):
        path = self.dir / "docs.jsonl"
        path.write_bytes(b'\xef\xbb\xbf{"doc_key": "k", "text": "alpha"}\n')
        self.assertEqual(list(read_documents(path)), [("k", "alpha")])

    def test_a_damaged_line_names_its_number(self):
        path = self.write("docs.jsonl", '{"text": "ok"}\n{broken\n')
        with self.assertRaisesRegex(CorpusError, "line 2"):
            list(read_documents(path))

    def test_a_document_without_text_is_an_error(self):
        for line in ('{"doc_key": "k"}', '{"text": 5}', "[1, 2]"):
            path = self.write("docs.jsonl", line + "\n")
            with self.subTest(line=line), self.assertRaises(CorpusError):
                list(read_documents(path))

    def test_a_key_of_the_wrong_kind_is_an_error(self):
        for key in ("true", "[1]", "1.5"):
            path = self.write("docs.jsonl", '{"doc_key": ' + key + ', "text": "alpha"}\n')
            with self.subTest(key=key), self.assertRaises(CorpusError):
                list(read_documents(path))

    def test_a_missing_file_is_an_error(self):
        with self.assertRaises(CorpusError):
            list(read_documents(self.dir / "absent.jsonl"))


if __name__ == "__main__":
    unittest.main()
