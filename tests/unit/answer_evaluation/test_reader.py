"""answer_evaluation.reader: the strict reading of the final answer and the two ways an unreadable answer counts."""

import unittest

from answer_evaluation import reader


class ReadingTests(unittest.TestCase):
    def test_the_form_of_the_papers_case_study_is_read(self):
        self.assertEqual(reader.reading("Therefore, the answer is (D) Needle decompression."), ("D", "ok"))
        self.assertEqual(reader.reading("Therefore, the answer is (A) BiPAP."), ("A", "ok"))

    def test_small_variations_of_the_phrase_are_read(self):
        for text in (
            "The answer is (B)",
            "the correct answer is (B).",
            "Answer is (B)",
            "ANSWER IS (B)",
            "The answer is: (B)",
            "So the answer is  (B)\nand that is all.",
            "Long reasoning...\n\nTherefore, the answer is (B) Option text",
        ):
            with self.subTest(text=text):
                self.assertEqual(reader.read_final_answer(text), "B")

    def test_a_letter_without_brackets_or_in_lower_case_is_not_read(self):
        for text in ("the answer is B", "The answer is B.", "the answer is (b)", "the answer is [B]", "answer: (B)", "I choose (B)"):
            with self.subTest(text=text):
                self.assertEqual(reader.reading(text), (None, "none"))

    def test_no_phrase_at_all_is_none(self):
        self.assertEqual(reader.reading(""), (None, "none"))
        self.assertEqual(reader.reading("The patient has pneumonia. A) is wrong."), (None, "none"))

    def test_two_different_letters_are_a_conflict_and_the_same_letter_twice_is_not(self):
        self.assertEqual(reader.reading("The answer is (A). On reflection, the answer is (C)."), (None, "conflict"))
        self.assertEqual(reader.reading("The answer is (C). Yes, the answer is (C) indeed."), ("C", "ok"))

    def test_another_bracketed_letter_in_the_same_sentence_is_a_conflict(self):
        self.assertEqual(reader.reading("The answer is (D) Needle decompression, not (A) BiPAP."), (None, "conflict"))
        self.assertEqual(reader.reading("The answer is (D) or (A)"), (None, "conflict"))

    def test_a_bracketed_letter_in_the_next_sentence_is_not_a_conflict(self):
        self.assertEqual(reader.reading("The answer is (D) Needle decompression. Option (A) is wrong."), ("D", "ok"))
        self.assertEqual(reader.reading("The answer is (D)\nOption (A) is wrong."), ("D", "ok"))

    def test_a_letter_that_is_not_an_option_is_outside(self):
        self.assertEqual(reader.reading("The answer is (E)"), (None, "outside"))
        self.assertEqual(reader.reading("The answer is (E)", letters="ABCDE"), ("E", "ok"))

    def test_the_text_must_be_a_string(self):
        with self.assertRaises(TypeError):
            reader.reading(None)

    def test_it_never_guesses_a_stray_letter(self):
        # The old code took the last standalone A to D, here the article "A".
        self.assertIsNone(reader.read_final_answer("A patient presents with fever. B cells are involved. Final choice D"))


class ScoringTests(unittest.TestCase):
    def test_a_readable_answer_is_compared_with_the_key(self):
        self.assertTrue(reader.scored_correct("the answer is (B)", "B"))
        self.assertFalse(reader.scored_correct("the answer is (B)", "C"))

    def test_an_unreadable_answer_is_wrong_for_scoring_and_skipped_for_labelling(self):
        for text in ("no answer here", "The answer is (A) and the answer is (B)", "the answer is (E)"):
            with self.subTest(text=text):
                self.assertFalse(reader.scored_correct(text, "A"))
                self.assertIsNone(reader.labelling_correct(text, "A"))

    def test_labelling_gives_true_or_false_for_a_readable_answer(self):
        self.assertIs(reader.labelling_correct("the answer is (C)", "C"), True)
        self.assertIs(reader.labelling_correct("the answer is (C)", "A"), False)

    def test_a_key_that_is_not_an_option_is_an_error(self):
        for bad in ("", "E", "a", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reader.scored_correct("the answer is (A)", bad)


class SummaryTests(unittest.TestCase):
    def test_the_summary_counts_the_reasons(self):
        texts = ["the answer is (A)", "nothing", "The answer is (A) then (B)", "The answer is (A). The answer is (C).", "the answer is (E)", "the answer is (D)"]
        summary = reader.unreadable_summary(texts)
        self.assertEqual(summary, {"texts": 6, "unreadable": 4, "share": 0.6667, "reasons": {"none": 1, "conflict": 2, "outside": 1}})

    def test_no_texts_have_no_share(self):
        self.assertEqual(reader.unreadable_summary([]), {"texts": 0, "unreadable": 0, "share": None, "reasons": {"none": 0, "conflict": 0, "outside": 0}})


if __name__ == "__main__":
    unittest.main()
