"""Tests of src/balanced_retrieval/chunking.py: windows of 128 words that overlap by 32."""

import itertools
import unittest

from balanced_retrieval.chunking import (
    DEFAULT_OVERLAP,
    DEFAULT_WORDS,
    check_window,
    chunk_words,
    passages_of,
)


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


class WindowTests(unittest.TestCase):
    def test_the_defaults_are_the_self_biorag_values(self):
        self.assertEqual((DEFAULT_WORDS, DEFAULT_OVERLAP), (128, 32))

    def test_a_text_that_fits_in_one_window_is_one_passage(self):
        self.assertEqual(chunk_words(words(128)), [words(128)])
        self.assertEqual(chunk_words("one two"), ["one two"])

    def test_one_word_more_gives_a_second_passage_that_starts_96_words_in(self):
        passages = chunk_words(words(129))
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[1], words(33, start=96))

    def test_the_last_window_is_the_one_that_reaches_the_end(self):
        self.assertEqual(len(chunk_words(words(224))), 2)  # 0-127 and 96-223
        self.assertEqual(len(chunk_words(words(225))), 3)  # a third window starts at 192
        self.assertEqual(chunk_words(words(225))[2], words(33, start=192))

    def test_neighbours_share_exactly_32_words(self):
        passages = chunk_words(words(400))
        for before, after in itertools.pairwise(passages):
            self.assertEqual(before.split()[-32:], after.split()[:32])

    def test_every_word_is_in_some_passage_and_no_passage_is_longer_than_the_window(self):
        passages = chunk_words(words(1000))
        self.assertLessEqual(max(len(p.split()) for p in passages), 128)
        self.assertEqual({w for p in passages for w in p.split()}, set(words(1000).split()))

    def test_white_space_of_any_kind_becomes_single_spaces(self):
        self.assertEqual(chunk_words("a  b\n\nc\td "), ["a b c d"])

    def test_a_text_without_words_gives_no_passage(self):
        for empty in ("", "   ", "\n\t"):
            self.assertEqual(chunk_words(empty), [])

    def test_other_sizes(self):
        self.assertEqual(chunk_words("a b c d e", size=2, overlap=0), ["a b", "c d", "e"])
        self.assertEqual(chunk_words("a b c d e", size=3, overlap=1), ["a b c", "c d e"])
        self.assertEqual(chunk_words("a b c", size=1, overlap=0), ["a", "b", "c"])

    def test_bad_settings_are_errors(self):
        for size, overlap in ((0, 0), (-1, 0), (4, 4), (4, 5), (4, -1)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                check_window(size, overlap)
        for size, overlap in ((4.0, 1), (4, True), ("4", 1)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(TypeError):
                check_window(size, overlap)


class PassageRecordTests(unittest.TestCase):
    def test_a_record_names_the_corpus_the_document_and_the_chunk(self):
        records = list(passages_of("pmc", "PMC123", words(130)))
        self.assertEqual([r["chunk_index"] for r in records], [0, 1])
        self.assertEqual({r["corpus"] for r in records}, {"pmc"})
        self.assertEqual({r["doc_key"] for r in records}, {"PMC123"})
        self.assertEqual(records[0]["text"], words(128))

    def test_an_empty_text_gives_no_record(self):
        self.assertEqual(list(passages_of("pmc", "k", "  ")), [])


if __name__ == "__main__":
    unittest.main()
