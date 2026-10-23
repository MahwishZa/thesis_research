"""data_roles.overlap: exact repeats, near repeats, and what is left out."""

import unittest

from data_roles import overlap
from data_roles.questions import Item


def item(identifier, stem):
    return Item(identifier, "medmcqa", "train", stem, ("a", "b", "c", "d"), "A")


LONG = "A 58 year old man presents with five days of productive cough and fever and a chest radiograph shows a right lower lobe infiltrate"


class ShingleTests(unittest.TestCase):
    def test_punctuation_case_and_spacing_do_not_matter(self):
        self.assertEqual(overlap.shingles("Which  drug, if any?"), overlap.shingles("which drug if any"))

    def test_a_short_text_is_one_shingle_and_an_empty_text_none(self):
        self.assertEqual(overlap.shingles("Which drug?"), {"which drug"})
        self.assertEqual(overlap.shingles("?!"), set())

    def test_a_long_text_has_one_shingle_per_start_position(self):
        self.assertEqual(len(overlap.shingles(LONG, 5)), len(overlap.words(LONG)) - 4)


class FindTests(unittest.TestCase):
    def test_an_exact_repeat_is_found_with_similarity_one(self):
        found = overlap.find_overlaps([item("r1", LONG)], [item("c1", LONG.upper() + "?")])
        self.assertEqual([(o.candidate, o.reference, o.similarity, o.exact) for o in found], [("c1", "r1", 1.0, True)])

    def test_a_near_repeat_is_found_and_marked_not_exact(self):
        changed = LONG.replace("five days", "six days")
        found = overlap.find_overlaps([item("r1", LONG)], [item("c1", changed)], threshold=0.6)
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].exact)
        self.assertTrue(0.6 <= found[0].similarity < 1.0)

    def test_a_different_question_is_not_reported(self):
        other = "Which of the following drugs is a cholinesterase inhibitor used in the treatment of mild dementia of the Alzheimer type"
        self.assertEqual(overlap.find_overlaps([item("r1", LONG)], [item("c1", other)]), [])

    def test_a_high_threshold_leaves_out_a_loose_match(self):
        changed = LONG.replace("five days", "six days")
        self.assertEqual(overlap.find_overlaps([item("r1", LONG)], [item("c1", changed)], threshold=0.95), [])

    def test_short_questions_match_only_exactly(self):
        found = overlap.find_overlaps([item("r1", "Which drug?")], [item("c1", "Which drug"), item("c2", "Which dose?")])
        self.assertEqual([(o.candidate, o.exact) for o in found], [("c1", True)])

    def test_a_phrase_found_in_too_many_reference_questions_is_ignored(self):
        common = [item(f"r{i}", "which of the following is true about " + f"topic{i} detail{i} extra{i}") for i in range(30)]
        candidate = item("c1", "which of the following is true about something else entirely different")
        self.assertEqual(overlap.find_overlaps(common, [candidate], threshold=0.2, max_postings=5), [])

    def test_the_best_match_of_a_candidate_comes_first(self):
        near = LONG.replace("five days", "six days")
        found = overlap.find_overlaps([item("r-near", near), item("r-exact", LONG)], [item("c1", LONG)], threshold=0.5)
        self.assertEqual([o.reference for o in found], ["r-exact", "r-near"])

    def test_an_empty_side_gives_no_overlaps(self):
        self.assertEqual(overlap.find_overlaps([], [item("c1", LONG)]), [])
        self.assertEqual(overlap.find_overlaps([item("r1", LONG)], []), [])

    def test_a_bad_threshold_is_an_error(self):
        for bad in (0, -0.1, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                overlap.find_overlaps([], [], threshold=bad)


if __name__ == "__main__":
    unittest.main()
