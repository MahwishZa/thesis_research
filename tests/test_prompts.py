"""The prompt and the answer reader (src/prompts.py)."""

import tempfile
import unittest
from pathlib import Path

from src import metrics, prompts

from tests.fixtures import LABELS

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------------------------------
# The prompt and the answer reader


class PromptTests(unittest.TestCase):
    def test_build_lists_sorted_labels(self):
        text = prompts.build_prompt("Pick one:\n{labels}\nAnswer.", LABELS)
        self.assertEqual(text, "Pick one:\n- melanoma\n- nevus\n- vascular lesion\nAnswer.")

    def test_template_needs_one_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.txt"
            path.write_text("no placeholder", encoding="utf-8")
            with self.assertRaises(ValueError):
                prompts.load_template(path)

    def test_a_text_prompt_has_the_case_placeholder_once_and_an_image_prompt_has_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.txt"
            for template, dataset in (("Pick:\n{labels}", "text"), ("{case}\n{case}\n{labels}", "text"),
                                      ("Case: {case}\n{labels}", "image")):
                with self.subTest(template):
                    path.write_text(template, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "{case}"):
                        prompts.load_template(path, dataset)
            path.write_text("Case: {case}\nPick one:\n{labels}\n", encoding="utf-8")
            prompt = prompts.build_prompt(prompts.load_template(path, "text"), LABELS)
        case = "A man of 70 had a {labels} lesion – “café”.\nIt grew."  # put in as it is, even text that looks like a placeholder
        self.assertEqual(prompts.fill(prompt, prompts.CASE, case), f"Case: {case}\nPick one:\n- melanoma\n- nevus\n- vascular lesion")
        self.assertEqual(prompts.fill("Case: {case}\nPick", prompts.CASE, " a papule \n"), "Case:  a papule \n\nPick")  # nothing trimmed

    def test_the_answer_uncertain_exists_only_for_a_run_that_may_abstain(self):
        self.assertEqual(prompts.build_prompt("{labels}", LABELS, abstain=True), "- melanoma\n- nevus\n- vascular lesion\n- uncertain")
        self.assertEqual(prompts.build_prompt("{labels}", LABELS), "- melanoma\n- nevus\n- vascular lesion")
        self.assertEqual(prompts.parse_answer(" **Uncertain.** ", [*LABELS, metrics.ABSTAIN]), "uncertain")
        self.assertIsNone(prompts.parse_answer("Uncertain", LABELS))  # a forced-choice run does not accept it
        self.assertIsNone(prompts.parse_answer("uncertain, probably melanoma", [*LABELS, metrics.ABSTAIN]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.txt"
            for text, abstain, message in (("{labels}\nAnswer uncertain if unsure.", False, "must not offer the answer"),
                                           ("{labels}\nAnswer with a name.", True, "must tell the model it may answer")):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text), self.assertRaisesRegex(ValueError, message):
                    prompts.load_template(path, "image", abstain)
            path.write_text("{labels}\nAnswer Uncertain if unsure.", encoding="utf-8")
            self.assertEqual(prompts.load_template(path, "image", True), "{labels}\nAnswer Uncertain if unsure.")

    def test_parse_forgives_only_formatting(self):
        for answer in ["melanoma", " Melanoma.", "**melanoma**", '"melanoma"', "Diagnosis: melanoma", "MELANOMA\n"]:
            self.assertEqual(prompts.parse_answer(answer, LABELS), "melanoma", answer)
        self.assertEqual(prompts.parse_answer("Vascular  lesion", LABELS), "vascular lesion")

    def test_parse_forgives_markup_around_a_leading_label(self):
        for answer in ["**Diagnosis:** Melanoma", "**Diagnosis**: Melanoma", "***Diagnosis:*** melanoma", "Answer: **melanoma**",
                       "Final answer: melanoma", "_melanoma_", "`melanoma`", "- melanoma", "**Final Answer:** `Melanoma`."]:
            self.assertEqual(prompts.parse_answer(answer, LABELS), "melanoma", answer)

    def test_parse_rejects_everything_else(self):
        for answer in [None, "", "melanoma or nevus", "probably melanoma", "basal cell carcinoma", "The answer is nevus.",
                       "The diagnosis is melanoma", "1. melanoma", "MELANOMA!", "melanoma (in situ)", "<end_of_turn>melanoma",
                       "**Diagnosis:** melanoma, nevus", "Melanoma\n\nThis lesion shows irregular pigment."]:
            self.assertIsNone(prompts.parse_answer(answer, LABELS), answer)

    def test_a_cut_off_answer_could_still_have_become_readable_while_it_looks_like_the_start_of_one(self):
        for answer in [None, "", "  ", "**", "nev", "Nevus", "**mela", "Vascular  les", '"melan', "Diagnosis", "Diagnosis:", "**Diagnosis:** mel",
                       "Final", "Final answer", "Final answer: vasc", "Answer: nev"]:
            self.assertTrue(prompts.can_still_become_readable(answer, LABELS), repr(answer))

    def test_class_names_are_compared_the_way_the_reader_compares_them(self):
        labels = ["Nevus", "Vascular  Lesion"]  # capitals and a double space in the dataset's own spelling
        self.assertEqual(prompts.parse_answer("vascular lesion", labels), "Vascular  Lesion")
        for answer in ["nev", "NEVUS", "vascular l", "Vascular  Les"]:
            self.assertTrue(prompts.can_still_become_readable(answer, labels), repr(answer))
        self.assertFalse(prompts.can_still_become_readable("nevi", labels))

    def test_a_cut_off_answer_that_already_differs_from_every_class_name_could_not_have_become_readable(self):
        for answer in ["Based on the image, the most likely diagnosis is **melanoma**.", "melanoma or nevus", "The answer is nevus",
                       "Melanoma\n\nThis lesion", "nevi", "basal cell", "1. melanoma", "Diagnosis is melanoma", "Final answers: nevus"]:
            self.assertFalse(prompts.can_still_become_readable(answer, LABELS), repr(answer))

    def test_whatever_is_readable_or_a_start_of_a_readable_answer_is_never_called_harmless(self):
        """The two functions agree: every answer the reader accepts, and every start of one, could still have become readable."""
        for answer in ["melanoma", "**Diagnosis:** Melanoma", "Final answer: `vascular lesion`."]:
            self.assertIsNotNone(prompts.parse_answer(answer, LABELS))
            for end in range(len(answer) + 1):
                self.assertTrue(prompts.can_still_become_readable(answer[:end], LABELS), repr(answer[:end]))


if __name__ == "__main__":
    unittest.main()
