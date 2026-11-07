import tempfile
import unittest
from pathlib import Path

from src import prompt

LABELS = ["nevus", "melanoma", "vascular lesion"]


class PromptTests(unittest.TestCase):
    def test_build_lists_sorted_labels(self):
        text = prompt.build("Pick one:\n{labels}\nAnswer.", LABELS)
        self.assertEqual(text, "Pick one:\n- melanoma\n- nevus\n- vascular lesion\nAnswer.")

    def test_template_needs_one_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.txt"
            path.write_text("no placeholder", encoding="utf-8")
            with self.assertRaises(ValueError):
                prompt.load_template(path)

    def test_parse_forgives_only_formatting(self):
        for answer in ["melanoma", " Melanoma.", "**melanoma**", '"melanoma"', "Diagnosis: melanoma", "MELANOMA\n"]:
            self.assertEqual(prompt.parse(answer, LABELS), "melanoma", answer)
        self.assertEqual(prompt.parse("Vascular  lesion", LABELS), "vascular lesion")

    def test_parse_forgives_markup_around_a_leading_label(self):
        for answer in ["**Diagnosis:** Melanoma", "**Diagnosis**: Melanoma", "***Diagnosis:*** melanoma", "Answer: **melanoma**",
                       "Final answer: melanoma", "_melanoma_", "`melanoma`", "- melanoma", "**Final Answer:** `Melanoma`."]:
            self.assertEqual(prompt.parse(answer, LABELS), "melanoma", answer)

    def test_parse_rejects_everything_else(self):
        for answer in [None, "", "melanoma or nevus", "probably melanoma", "basal cell carcinoma", "The answer is nevus.",
                       "The diagnosis is melanoma", "1. melanoma", "MELANOMA!", "melanoma (in situ)", "<end_of_turn>melanoma",
                       "**Diagnosis:** melanoma, nevus", "Melanoma\n\nThis lesion shows irregular pigment."]:
            self.assertIsNone(prompt.parse(answer, LABELS), answer)

    def test_a_cut_off_answer_could_still_have_become_readable_while_it_looks_like_the_start_of_one(self):
        for answer in [None, "", "  ", "**", "nev", "Nevus", "**mela", "Vascular  les", '"melan', "Diagnosis", "Diagnosis:", "**Diagnosis:** mel",
                       "Final", "Final answer", "Final answer: vasc", "Answer: nev"]:
            self.assertTrue(prompt.can_still_become_readable(answer, LABELS), repr(answer))

    def test_class_names_are_compared_the_way_parse_compares_them(self):
        labels = ["Nevus", "Vascular  Lesion"]  # capitals and a double space in the dataset's own spelling
        self.assertEqual(prompt.parse("vascular lesion", labels), "Vascular  Lesion")
        for answer in ["nev", "NEVUS", "vascular l", "Vascular  Les"]:
            self.assertTrue(prompt.can_still_become_readable(answer, labels), repr(answer))
        self.assertFalse(prompt.can_still_become_readable("nevi", labels))

    def test_a_cut_off_answer_that_already_differs_from_every_class_name_could_not_have_become_readable(self):
        for answer in ["Based on the image, the most likely diagnosis is **melanoma**.", "melanoma or nevus", "The answer is nevus",
                       "Melanoma\n\nThis lesion", "nevi", "basal cell", "1. melanoma", "Diagnosis is melanoma", "Final answers: nevus"]:
            self.assertFalse(prompt.can_still_become_readable(answer, LABELS), repr(answer))

    def test_whatever_is_readable_or_a_start_of_a_readable_answer_is_never_called_harmless(self):
        """The two functions agree: every answer parse() accepts, and every start of one, could still have become readable."""
        for answer in ["melanoma", "**Diagnosis:** Melanoma", "Final answer: `vascular lesion`."]:
            self.assertIsNotNone(prompt.parse(answer, LABELS))
            for end in range(len(answer) + 1):
                self.assertTrue(prompt.can_still_become_readable(answer[:end], LABELS), repr(answer[:end]))


if __name__ == "__main__":
    unittest.main()
