"""Acceptance tests for docs/methodology.md section 5 (filter input and decision). They skip until src/rationale_filter exists."""

import random

from tests.spec_targets import FILTER_INSTRUCTION, SpecTestCase, resolve

NAN = float("nan")
INF = float("inf")


def expected_input(evidence, question):
    return FILTER_INSTRUCTION + "\n\nEvidence: " + evidence + "\n\nQuestion: " + question


def build(evidence, question):
    return resolve("build_filter_input")(evidence=evidence, question=question)


class FilterInputTests(SpecTestCase):
    """The released training examples all follow one template (the five samples in the authors' repository)."""

    requires = ("build_filter_input",)

    QUESTION = "A 58-year-old man has memory loss. Which test is most appropriate? A) MRI B) EEG C) CT D) PET"

    def test_the_template_is_exact(self):
        evidence = "Donepezil is a cholinesterase inhibitor used in mild to moderate dementia."
        self.assertEqual(build(evidence, self.QUESTION), expected_input(evidence, self.QUESTION))

    def test_surrounding_whitespace_is_removed_from_both_parts(self):
        evidence = "  \n Donepezil is a cholinesterase inhibitor. \t\n"
        question = "\n\n  " + self.QUESTION + "  \n"
        self.assertEqual(build(evidence, question), expected_input(evidence.strip(), question.strip()))

    def test_inner_whitespace_and_options_are_kept(self):
        evidence = "Line one.\nLine two  with two spaces.\n\nNew paragraph."
        question = "Which drug?\nA) one\nB) two\nC) three\nD) four"
        self.assertEqual(build(evidence, question), expected_input(evidence, question))

    def test_the_input_has_exactly_one_evidence_and_one_question_marker(self):
        text = build("Some evidence.", self.QUESTION)
        self.assertTrue(text.startswith(FILTER_INSTRUCTION + "\n\nEvidence: "))
        self.assertEqual(text.count("\n\nEvidence: "), 1)
        self.assertEqual(text.count("\n\nQuestion: "), 1)
        self.assertTrue(text.endswith(self.QUESTION))

    def test_swapping_the_parts_changes_the_input(self):
        self.assertNotEqual(build("alpha", "beta"), build("beta", "alpha"))

    def test_the_same_arguments_give_the_same_text(self):
        self.assertEqual(build("alpha", "beta"), build("alpha", "beta"))


class DecisionRuleTests(SpecTestCase):
    """Softmax over the two label logits, larger wins, a tie counts as helpful (the authors' argmax picks index 0)."""

    requires = ("decide_helpful",)

    @staticmethod
    def decide(helpful, not_helpful):
        return resolve("decide_helpful")(logit_helpful=helpful, logit_not_helpful=not_helpful)

    def test_the_larger_logit_wins(self):
        self.assertIs(self.decide(2.0, 1.0), True)
        self.assertIs(self.decide(1.0, 2.0), False)
        self.assertIs(self.decide(-1.0, -3.0), True)
        self.assertIs(self.decide(-3.0, -1.0), False)

    def test_a_tie_counts_as_helpful(self):
        for value in (0.0, 5.5, -3.0, 1e30, -1e30):
            with self.subTest(value=value):
                self.assertIs(self.decide(value, value), True)

    def test_very_large_logits_do_not_overflow(self):
        self.assertIs(self.decide(1000.0, 0.0), True)
        self.assertIs(self.decide(0.0, 1000.0), False)
        self.assertIs(self.decide(1000.0, 999.0), True)
        self.assertIs(self.decide(-1000.0, -999.0), False)
        self.assertIs(self.decide(-1e4, -1e4), True)

    def test_it_agrees_with_a_plain_comparison_on_random_pairs(self):
        rng = random.Random(5)
        for _ in range(500):
            helpful, not_helpful = rng.uniform(-30, 30), rng.uniform(-30, 30)
            with self.subTest(helpful=helpful, not_helpful=not_helpful):
                self.assertIs(self.decide(helpful, not_helpful), helpful >= not_helpful)

    def test_non_finite_logits_are_an_error(self):
        for bad in (NAN, INF, -INF):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.decide(bad, 0.0)
                with self.assertRaises(ValueError):
                    self.decide(0.0, bad)


class JudgeSnippetsTests(SpecTestCase):
    """Each snippet is judged on its own, from the question and the snippet only (Paper p.9)."""

    requires = ("judge_snippets",)

    QUESTION = "  Which drug? A) one B) two C) three D) four \n"
    SNIPPETS = ("first snippet", "second snippet", "third snippet", "fourth snippet")

    @staticmethod
    def judge_all(question, snippets, judge):
        return resolve("judge_snippets")(question=question, snippets=snippets, judge=judge)

    def test_the_judge_sees_one_filter_input_per_snippet_in_order(self):
        seen = []

        def judge(text):
            seen.append(text)
            return True

        self.judge_all(self.QUESTION, self.SNIPPETS, judge)
        self.assertEqual(seen, [expected_input(snippet, self.QUESTION.strip()) for snippet in self.SNIPPETS])

    def test_the_decisions_follow_the_judge_in_order(self):
        answers = {expected_input(snippet, self.QUESTION.strip()): flag for snippet, flag in zip(self.SNIPPETS, (True, False, False, True))}
        result = self.judge_all(self.QUESTION, self.SNIPPETS, lambda text: answers[text])
        self.assertEqual(result, [True, False, False, True])

    def test_a_snippet_is_judged_the_same_whatever_its_neighbours_are(self):
        def judge(text):
            return "third" in text

        alone = self.judge_all(self.QUESTION, ["third snippet"], judge)
        among_others = self.judge_all(self.QUESTION, self.SNIPPETS, judge)
        self.assertEqual(alone, [True])
        self.assertEqual(among_others, [False, False, True, False])

    def test_no_snippets_means_no_calls(self):
        calls = []
        self.assertEqual(self.judge_all(self.QUESTION, [], lambda text: calls.append(text) or True), [])
        self.assertEqual(calls, [])

    def test_a_tuple_of_snippets_works(self):
        self.assertEqual(self.judge_all(self.QUESTION, tuple(self.SNIPPETS), lambda text: True), [True] * 4)

    def test_a_failing_judge_is_not_swallowed(self):
        def judge(text):
            raise RuntimeError("model failed")

        with self.assertRaises(RuntimeError):
            self.judge_all(self.QUESTION, self.SNIPPETS, judge)

    def test_the_results_are_plain_booleans_whatever_the_judge_returns(self):
        result = self.judge_all(self.QUESTION, self.SNIPPETS, lambda text: 1 if "first" in text else 0)
        self.assertEqual(result, [True, False, False, False])
        self.assertTrue(all(type(flag) is bool for flag in result))


class KeepHelpfulTests(SpecTestCase):
    requires = ("keep_helpful",)

    @staticmethod
    def keep(items, decisions):
        return resolve("keep_helpful")(items, decisions)

    def test_only_helpful_items_remain_in_their_original_order(self):
        self.assertEqual(self.keep(["a", "b", "c", "d"], [True, False, True, False]), ["a", "c"])
        self.assertEqual(self.keep(["a", "b", "c", "d"], [False, True, True, True]), ["b", "c", "d"])

    def test_nothing_helpful_gives_an_empty_list(self):
        self.assertEqual(self.keep(["a", "b"], [False, False]), [])

    def test_everything_helpful_gives_all_items_in_a_new_list(self):
        items = ["a", "b"]
        kept = self.keep(items, [True, True])
        self.assertEqual(kept, ["a", "b"])
        self.assertIsNot(kept, items)

    def test_empty_input_gives_an_empty_list(self):
        self.assertEqual(self.keep([], []), [])

    def test_repeated_items_are_kept_each_time(self):
        self.assertEqual(self.keep(["x", "x", "y"], [True, True, False]), ["x", "x"])

    def test_the_very_same_objects_come_back(self):
        records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        kept = self.keep(records, [True, False, True])
        self.assertIs(kept[0], records[0])
        self.assertIs(kept[1], records[2])

    def test_the_input_is_not_changed_and_a_new_list_is_returned(self):
        items, decisions = ["a", "b", "c"], [True, False, True]
        kept = self.keep(items, decisions)
        self.assertEqual(items, ["a", "b", "c"])
        self.assertEqual(decisions, [True, False, True])
        self.assertIsNot(kept, items)

    def test_a_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            self.keep(["a", "b", "c"], [True, False])
        with self.assertRaises(ValueError):
            self.keep(["a"], [True, False])
