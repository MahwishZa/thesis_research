"""What the filter reads from a PubMed record, and how a record's study design is classified (no model, no network)."""

import unittest

from experiments.medchange import abstracts as AB

ABSTRACT = ("BACKGROUND: Cranberries may prevent urinary tract infection. METHODS: We randomised 120 women "
            "(BMI: 25). RESULTS: Infection occurred in 18% versus 32% (RR 0.56, 95% CI 0.35 to 0.89). "
            "CONCLUSIONS: Cranberry juice reduced recurrence.")


def cand(abstract=ABSTRACT, title="Cranberry for infection."):
    return {"pmid": "1", "rank": 1, "title": title, "abstract": abstract}


class SnippetTests(unittest.TestCase):

    def test_labelled_sections_are_found_and_running_text_acronyms_are_not_boundaries(self):
        sections = AB.split_sections(ABSTRACT)
        self.assertEqual([label for label, _ in sections], ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"])
        self.assertIn("(BMI: 25)", dict(sections)["METHODS"])

    def test_key_text_takes_results_and_conclusions(self):
        results, conclusions = AB.key_text(ABSTRACT)
        self.assertTrue(results.startswith("Infection occurred"))
        self.assertEqual(conclusions, "Cranberry juice reduced recurrence.")

    def test_an_unlabelled_abstract_contributes_its_last_three_sentences(self):
        results, conclusions = AB.key_text("One is first. Two is second. Three is third. Four is fourth.")
        self.assertEqual(results, "")
        self.assertEqual(conclusions, "Two is second. Three is third. Four is fourth.")

    def test_snippet_has_title_results_and_conclusions(self):
        text = AB.study_snippet(cand())
        self.assertTrue(text.startswith("Cranberry for infection."))
        self.assertIn("Results: Infection occurred", text)
        self.assertIn("Conclusions: Cranberry juice reduced recurrence.", text)

    def test_snippet_respects_the_word_budget_and_keeps_half_for_conclusions(self):
        long = ("RESULTS: " + " ".join(f"r{i}" for i in range(400)) + ". CONCLUSIONS: "
                + " ".join(f"c{i}" for i in range(400)) + ".")
        text = AB.study_snippet(cand(abstract=long), max_words=100)
        self.assertLessEqual(len(text.split()), 100 + 4)
        self.assertGreaterEqual(text.count(" c"), 45)

    def test_a_missing_abstract_does_not_crash(self):
        self.assertEqual(AB.study_snippet({"title": "", "abstract": ""}), "No abstract available.")
        self.assertEqual(AB.study_snippet({"title": "Only a title", "abstract": None}), "Only a title.")


class StudyTypeTests(unittest.TestCase):

    def test_study_types_prefer_reviews_then_trials(self):
        self.assertEqual(AB.study_type(["Journal Article", "Systematic Review", "Randomized Controlled Trial"]), "SR/MA")
        self.assertEqual(AB.study_type(["Meta-Analysis"]), "SR/MA")
        self.assertEqual(AB.study_type(["Clinical Trial"]), "RCT")
        self.assertEqual(AB.study_type(["Case Reports"]), "other")
        self.assertEqual(AB.study_type(None), "other")


if __name__ == "__main__":
    unittest.main()
