"""answer_evaluation.evidence: the snippets each system reads, the filter's decisions reused, and the refusals."""

import unittest

from answer_evaluation.evidence import NO_FILTER, WITH_FILTER, EvidenceProvider
from answer_evaluation.runner import NO_RETRIEVAL, Arm, EvalQuestion

QUESTION = EvalQuestion("q1", "Which drug? A) x B) y", "A", "Therefore, the answer is (A).")
LISTS = {
    2: {"q1": ["good one", "bad one", "good two"]},
    4: {"q1": ["good one", "bad one", "good two", "bad two", "good three"]},
}


class Judge:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs):
        self.calls.append(list(inputs))
        return ["Evidence: good" in text for text in inputs]


class ProviderTests(unittest.TestCase):
    def test_no_filter_takes_the_first_k_of_the_list_for_k(self):
        provider = EvidenceProvider(LISTS)
        self.assertEqual(provider(QUESTION, Arm(NO_FILTER, 2)), ["good one", "bad one"])
        self.assertEqual(provider(QUESTION, Arm(NO_FILTER, 4)), ["good one", "bad one", "good two", "bad two"])

    def test_filter_keeps_the_helpful_snippets_in_order(self):
        provider = EvidenceProvider(LISTS, Judge())
        self.assertEqual(provider(QUESTION, Arm(WITH_FILTER, 4)), ["good one", "good two"])

    def test_the_same_snippet_is_judged_once_for_a_question(self):
        judge = Judge()
        provider = EvidenceProvider(LISTS, judge)
        provider(QUESTION, Arm(WITH_FILTER, 2))
        provider(QUESTION, Arm(WITH_FILTER, 4))
        judged = [text for call in judge.calls for text in call]
        self.assertEqual(len(judged), 4)
        self.assertEqual(len(set(judged)), 4)

    def test_summary_counts_considered_and_kept(self):
        provider = EvidenceProvider(LISTS, Judge())
        provider(QUESTION, Arm(WITH_FILTER, 4))
        self.assertEqual(provider.summary(), {4: {"considered": 4, "kept": 2, "kept_share": 0.5}})

    def test_refusals(self):
        with self.assertRaises(ValueError):
            EvidenceProvider(LISTS)(QUESTION, Arm(NO_RETRIEVAL, 0))
        with self.assertRaises(ValueError):
            EvidenceProvider(LISTS)(QUESTION, Arm(WITH_FILTER, 2))
        with self.assertRaises(ValueError):
            EvidenceProvider(LISTS)(QUESTION, Arm(NO_FILTER, 8))
        with self.assertRaises(ValueError):
            EvidenceProvider(LISTS)(EvalQuestion("other", "q", "A", "r"), Arm(NO_FILTER, 2))

    def test_a_wrong_number_of_decisions_is_an_error(self):
        provider = EvidenceProvider(LISTS, lambda inputs: [True])
        with self.assertRaises(ValueError):
            provider(QUESTION, Arm(WITH_FILTER, 2))


if __name__ == "__main__":
    unittest.main()
