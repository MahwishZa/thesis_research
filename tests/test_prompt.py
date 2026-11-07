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


if __name__ == "__main__":
    unittest.main()
