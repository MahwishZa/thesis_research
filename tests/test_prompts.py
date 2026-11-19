"""The prompt and the answer reader (src/prompts.py)."""

import json
import tempfile
import unittest
from pathlib import Path

from src import prompts

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

    def test_the_options_are_the_classes_only(self):
        self.assertEqual(prompts.build_prompt("{labels}", LABELS), "- melanoma\n- nevus\n- vascular lesion")


class JsonAnswerTests(unittest.TestCase):
    """The answer of the current form (prompts.parse_json_answer), read strictly. LABELS: nevus, melanoma, vascular lesion."""

    def answer(self, **change):
        content = {"differential": ["melanoma", "nevus", "vascular lesion"], "evidence_sufficient": "no",
                   "reason": "Irregular border.", "evidence": []} | change
        return json.dumps(content)

    def test_a_valid_answer_gives_the_diagnosis_the_differential_and_the_flag(self):
        read, problem = prompts.parse_json_answer(self.answer(), LABELS)
        self.assertIsNone(problem)
        self.assertEqual(read, {"diagnosis": "melanoma", "differential": ["melanoma", "nevus", "vascular lesion"],
                                "evidence_sufficient": "no", "reason": "Irregular border.", "evidence": []})

    def test_only_spaces_a_code_block_and_the_case_of_names_are_forgiven(self):
        loose = self.answer(differential=[" Melanoma", "NEVUS", "vascular  lesion"], evidence_sufficient=" Yes ")
        for text in (loose, f"  {loose}\n", f"```json\n{loose}\n```", f"```\n{loose}\n```"):
            with self.subTest(text=text):
                read, problem = prompts.parse_json_answer(text, LABELS)
                self.assertIsNone(problem)
                self.assertEqual((read["differential"], read["evidence_sufficient"]), (["melanoma", "nevus", "vascular lesion"], "yes"))

    def test_everything_else_is_invalid_with_its_reason(self):
        good = json.loads(self.answer())
        cases = {
            None: "empty answer", "  ": "empty answer",
            "melanoma": "not one JSON object", f"Here it is: {self.answer()}": "not one JSON object",
            f"{self.answer()} I hope this helps.": "not one JSON object", "[1, 2]": "not one JSON object",
            json.dumps({k: v for k, v in good.items() if k != "reason"}): "missing key reason",
            json.dumps(good | {"confidence": 0.9}): "key not asked for: confidence",
            self.answer(differential=["melanoma", "nevus"]): "the differential is not a list of 3 class names",
            self.answer(differential="melanoma"): "the differential is not a list of 3 class names",
            self.answer(differential=["melanoma", "nevus", 3]): "the differential is not a list of 3 class names",
            self.answer(differential=["melanoma", "nevus", "basal cell carcinoma"]): "not a class of the list: 'basal cell carcinoma'",
            self.answer(differential=["**melanoma**", "nevus", "vascular lesion"]): "not a class of the list: '**melanoma**'",
            self.answer(differential=["melanoma", "Melanoma", "nevus"]): "the differential names a class twice",
            self.answer(evidence_sufficient="maybe"): "evidence_sufficient is not yes or no",
            self.answer(evidence_sufficient=True): "evidence_sufficient is not yes or no",
            self.answer(reason="  "): "no reason given", self.answer(reason=None): "no reason given",
            self.answer(evidence="P1"): "evidence is not a list of passage ids",
            self.answer(evidence=["P1"]): "evidence names a passage the run did not give: 'P1'",
        }
        for text, why in cases.items():
            with self.subTest(text=text):
                self.assertEqual(prompts.parse_json_answer(text, LABELS), (None, why))

    def test_evidence_must_be_among_the_passages_the_run_gave(self):
        text = self.answer(evidence=["P1", "P2"])
        self.assertEqual(prompts.parse_json_answer(text, LABELS, {"P1", "P2", "P3"})[0]["evidence"], ["P1", "P2"])
        self.assertEqual(prompts.parse_json_answer(text, LABELS, {"P1"})[1], "evidence names a passage the run did not give: 'P2'")

    def test_with_fewer_than_three_classes_the_differential_holds_them_all(self):
        read, problem = prompts.parse_json_answer(self.answer(differential=["nevus", "melanoma"]), ["melanoma", "nevus"])
        self.assertEqual((problem, read["diagnosis"]), (None, "nevus"))

    def test_each_prompt_asks_for_exactly_the_keys_the_reader_requires(self):
        for path in sorted((ROOT / "prompts").glob("*/closed_book.txt")):
            with self.subTest(path=path.parent.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual(text.count("{labels}"), 1)
                for key in prompts.ANSWER_KEYS:
                    self.assertIn(f'"{key}"', text)
                self.assertIn("three", text)


if __name__ == "__main__":
    unittest.main()
