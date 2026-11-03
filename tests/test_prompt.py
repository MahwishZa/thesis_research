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

    def test_parse_rejects_everything_else(self):
        for answer in [None, "", "melanoma or nevus", "probably melanoma", "basal cell carcinoma", "The answer is nevus."]:
            self.assertIsNone(prompt.parse(answer, LABELS), answer)


if __name__ == "__main__":
    unittest.main()
