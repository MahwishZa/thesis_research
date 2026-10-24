"""rationale_query.prompt: the paper's prompt, word for word, with the question and its options in place of [initial_query]."""

import unittest

from rationale_query import prompt
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

# Written out again here, from the box on p.4 of the paper, so that a change in the module cannot change the expectation.
PAPER_TEXT = (
    "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, "
    "starting by summarizing the available information. Output your explanation and single option from the given "
    "options as the final answer.\n"
    "Here is the question: "
)


class PromptTextTests(unittest.TestCase):
    def test_the_prompt_is_the_papers_box_followed_by_the_question(self):
        self.assertEqual(prompt.build_rationale_prompt("Which drug? (A) x (B) y"), PAPER_TEXT + "Which drug? (A) x (B) y")

    def test_one_line_break_stands_before_here_is_the_question(self):
        text = prompt.build_rationale_prompt("Q")
        self.assertIn("as the final answer.\nHere is the question: Q", text)
        self.assertNotIn("\n\n", text)

    def test_the_wording_is_the_papers_not_a_paraphrase(self):
        text = prompt.build_rationale_prompt("Q")
        for phrase in (
            "multiple choice questions",  # no hyphen
            "about medical knowledge. Solve them",  # full stop and capital S
            "step-by-step fashion",
            "starting by summarizing the available information",
            "Output your explanation and single option from the given options as the final answer.",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("[initial_query]", text)
        self.assertNotIn("multiple-choice", text)

    def test_the_question_and_its_options_are_passed_on_unchanged(self):
        text = prompt.build_rationale_prompt(FIGURE_4_QUESTION)
        self.assertTrue(text.endswith(FIGURE_4_QUESTION))
        self.assertEqual(len(text), len(PAPER_TEXT) + len(FIGURE_4_QUESTION))
        for option in ("(A) BiPAP", "(B) Chest tube placement", "(C) Intubation", "(D) Needle Decompression"):
            self.assertIn(option, text)

    def test_the_question_is_not_stripped_or_reformatted(self):
        question = "  Line one\nLine two (A) a (B) b  \n"
        self.assertEqual(prompt.build_rationale_prompt(question), PAPER_TEXT + question)

    def test_the_separator_is_a_single_setting(self):
        self.assertEqual(prompt.SEPARATOR, "\n")
        self.assertEqual(prompt.INSTRUCTION + prompt.SEPARATOR + prompt.LEAD_IN + " ", PAPER_TEXT)


class PromptRefusalTests(unittest.TestCase):
    def test_a_question_that_is_not_text_is_refused(self):
        for bad in (None, 5, ["Q"], b"Q"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                prompt.build_rationale_prompt(bad)

    def test_an_empty_or_blank_question_is_refused(self):
        for bad in ("", " ", "\n\t "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prompt.build_rationale_prompt(bad)


if __name__ == "__main__":
    unittest.main()
