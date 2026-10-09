"""The fixed specification of the Alzheimer's-specific question set: templates, reading cues, split. No network, no model."""

import re
import unittest
from pathlib import Path

from experiments.adkqa import spec as S
from experiments.medchange.pubmed_asof import _STOP, query_terms

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = (ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")


class TemplateTests(unittest.TestCase):
    def test_every_fixed_template_word_is_a_stop_word_of_the_query_builder(self):
        for name, t in S.TEMPLATES.items():
            fixed = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", t.replace("{x}", "").replace("{y}", "").lower())
            extra = [w for w in fixed if len(w) > 2 and w not in _STOP]
            self.assertEqual(extra, list(S.CONDITION_WORDS), name)

    def test_short_forms_use_only_stop_words_and_the_span_carries_the_condition(self):
        for name, t in S.TEMPLATES_SHORT.items():
            fixed = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", t.replace("{x}", "").replace("{y}", "").lower())
            self.assertEqual([w for w in fixed if len(w) > 2 and w not in _STOP], [], name)
        self.assertEqual(set(S.TEMPLATES), set(S.TEMPLATES_SHORT))

    def test_a_span_that_names_the_disease_gives_the_short_form_and_never_names_it_twice(self):
        q = S.fill("association", "CHASERR expression", "Alzheimer's disease")
        self.assertEqual(q, "Is there any effect of CHASERR expression on Alzheimer's disease?")
        self.assertEqual(q.lower().count("alzheimer"), 1)
        self.assertEqual(S.question_checks("association", "CHASERR expression", "Alzheimer's disease"), [])
        self.assertEqual(query_terms(q), ["chaserr", "expression", "alzheimer", "disease"])
        self.assertTrue(S.fill("effect", "donepezil", "cognition").endswith("in people with Alzheimer's disease?"))

    def test_the_short_form_still_needs_a_content_word_and_obeys_the_length_rule(self):
        self.assertIn("no content word besides the condition", S.question_checks("effect", "Alzheimer's disease", "Alzheimer disease"))
        self.assertIn("more than 6 content words copied",
                      S.question_checks("association", "amyloid beta 42 total tau", "Alzheimer's disease risk"))

    def test_the_query_of_a_question_is_its_copied_terms_plus_the_condition(self):
        q = S.fill("effect", "donepezil", "cognitive function")
        self.assertEqual(query_terms(q), ["donepezil", "cognitive", "function", "alzheimer", "disease"])
        self.assertEqual(S.question_checks("effect", "donepezil", "cognitive function"), [])

    def test_too_many_copied_words_push_the_condition_out_of_the_query(self):
        problems = S.question_checks("test", "plasma p-tau217 assay", "early detection of amyloid pathology")
        self.assertTrue(problems)
        self.assertIn("more than 6 content words copied", problems)

    def test_an_unknown_template_or_empty_span_is_refused(self):
        self.assertTrue(S.question_checks("nonsense", "a", "b"))
        self.assertTrue(S.question_checks("effect", "", "memory"))


class ReadingTests(unittest.TestCase):
    def test_each_verdict_is_read_from_a_plain_conclusion(self):
        self.assertEqual(S.read_conclusion("Donepezil significantly improved cognition."), "SUPPORTED")
        self.assertEqual(S.read_conclusion("There was no significant difference between groups."), "REFUTED")
        self.assertEqual(S.read_conclusion("The evidence is insufficient to draw conclusions."), "NOT ENOUGH INFORMATION")

    def test_a_negated_positive_word_is_not_read_as_positive(self):
        self.assertEqual(S.read_conclusion("Treatment was not effective."), "REFUTED")

    def test_mixed_or_hedged_conclusions_make_no_question(self):
        self.assertIsNone(S.read_conclusion("It was effective, however results were mixed."))
        self.assertIsNone(S.read_conclusion("No significant effect on memory but beneficial for mood."))
        self.assertIsNone(S.read_conclusion("Further research is needed."))
        self.assertIsNone(S.read_conclusion(""))


class SplitTests(unittest.TestCase):
    def test_the_cluster_is_the_first_other_major_descriptor_or_the_record(self):
        self.assertEqual(S.cluster_key(["Alzheimer Disease", "Memantine", "Donepezil"], "1"), "donepezil")
        self.assertEqual(S.cluster_key(["Alzheimer Disease"], "42"), "pmid:42")

    def test_split_is_deterministic_and_close_to_the_declared_share(self):
        keys = [f"topic-{i}" for i in range(4000)]
        share = sum(S.split_of(k) == "dev" for k in keys) / len(keys)
        self.assertAlmostEqual(share, S.DEV_FRACTION, delta=0.03)
        self.assertEqual([S.split_of(k) for k in keys[:50]], [S.split_of(k) for k in keys[:50]])

    def test_the_numbers_are_consistent(self):
        self.assertEqual(S.DEV_N / S.DRAFT_N, 0.40, "the 40% survival gate is the share needed for 60 of 150")
        self.assertLessEqual(S.TEST_MIN, S.TEST_MAX)
        self.assertEqual(len(S.draft_order(["1", "2", "3"])), 3)


class ProtocolMatchesCodeTests(unittest.TestCase):
    def test_templates_cues_and_constants_in_the_protocol_are_those_of_the_code(self):
        for t in list(S.TEMPLATES.values()) + list(S.TEMPLATES_SHORT.values()):
            self.assertIn(t, PROTOCOL)
        self.assertIn(f"`{S.SPLIT_SEED}`", PROTOCOL)
        for group in S.CUES.values():
            for pat in group:
                self.assertIn(pat, PROTOCOL, pat)
        self.assertIn(f"{int(S.DEV_FRACTION * 100)}%", PROTOCOL)
        self.assertIn(S.SOURCE_ANCHOR, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
