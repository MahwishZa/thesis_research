"""answer_evaluation.prompt: the paper's wording kept, an Evidence section added only when snippets are kept."""

import unittest

from answer_evaluation import prompt
from rationale_query.prompt import build_rationale_prompt
from tests.spec.targets import RATIONALE_PROMPT

QUESTION = "  Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin \n"


class PromptTests(unittest.TestCase):
    def test_without_snippets_it_is_the_rationale_prompt_of_part_one(self):
        self.assertEqual(prompt.build_answer_prompt(QUESTION), build_rationale_prompt(QUESTION))
        self.assertEqual(prompt.build_answer_prompt(QUESTION, []), build_rationale_prompt(QUESTION))
        self.assertEqual(prompt.build_answer_prompt(QUESTION, ()), build_rationale_prompt(QUESTION))

    def test_with_snippets_the_layout_is_exact(self):
        text = prompt.build_answer_prompt(QUESTION, ["  First snippet.\n", "Second snippet."])
        stem = "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"
        self.assertEqual(
            text,
            "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, "
            "starting by summarizing the available information. Output your explanation and single option from the given "
            "options as the final answer.\n"
            "Evidence:\nFirst snippet.\n\nSecond snippet.\n"
            "Here is the question: " + stem,
        )

    def test_the_papers_words_are_all_still_there_in_order(self):
        text = prompt.build_answer_prompt(QUESTION, ["A snippet."])
        words = " ".join(text.split())
        self.assertTrue(words.startswith(" ".join(RATIONALE_PROMPT.split()[:-4])))  # up to "as the final answer."
        self.assertIn("Here is the question:", words)

    def test_the_question_comes_after_the_evidence(self):
        text = prompt.build_answer_prompt(QUESTION, ["The evidence text."])
        self.assertLess(text.index("The evidence text."), text.index("Which drug"))

    def test_one_snippet_has_no_blank_line(self):
        self.assertNotIn("\n\n", prompt.build_answer_prompt(QUESTION, ["Only one."]))

    def test_the_order_of_the_snippets_is_kept(self):
        text = prompt.build_answer_prompt(QUESTION, ["zebra", "apple"])
        self.assertLess(text.index("zebra"), text.index("apple"))

    def test_bad_snippets_are_errors(self):
        for bad in (["fine", ""], ["fine", "   "], ["fine", None], [5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prompt.build_answer_prompt(QUESTION, bad)

    def test_one_text_instead_of_a_list_is_a_type_error(self):
        with self.assertRaises(TypeError):
            prompt.build_answer_prompt(QUESTION, "a single text")

    def test_a_bad_question_is_an_error_whatever_the_snippets(self):
        with self.assertRaises(ValueError):
            prompt.build_answer_prompt("  ", ["x"])
        with self.assertRaises(TypeError):
            prompt.build_answer_prompt(None, ["x"])

    def test_the_instruction_ends_before_the_question_marker(self):
        self.assertTrue(prompt.INSTRUCTION.endswith("as the final answer.\n"))
        self.assertNotIn("Here is the question", prompt.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
