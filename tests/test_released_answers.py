"""The closed-book answers of other models that the MedChange release contains, read as verdicts (synthetic files)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange.released_answers import FILES, find_label, read_answers


class ReleasedAnswersTests(unittest.TestCase):
    def test_the_first_verdict_word_of_an_answer_is_its_label(self):
        self.assertEqual(find_label("LABEL: not enough information |||"), "NOT ENOUGH INFORMATION")
        self.assertEqual(find_label("Answer: REFUTED: no ||| benefit"), "REFUTED")
        self.assertIsNone(find_label("no verdict"))

    def test_three_line_files_are_read_as_the_authors_do(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / FILES["biomistral"]).write_text(
                "Question: a?\nAnswer: REFUTED: no ||| benefit\n---\nQuestion: b?\nAnswer: unclear text\n---\n"
                "Question: c?\nAnswer: SUPPORTED\n---", encoding="utf-8")
            (folder / FILES["qwen25-7b"]).write_text("LABEL: SUPPORTED\nNOT ENOUGH INFORMATION: x\n", encoding="utf-8")
            self.assertEqual(read_answers(folder, "biomistral"), ["REFUTED", None, "SUPPORTED"])
            self.assertEqual(read_answers(folder, "qwen25-7b"), ["SUPPORTED", "NOT ENOUGH INFORMATION", None])
            self.assertIsNone(read_answers(folder, "pmcllama"))


if __name__ == "__main__":
    unittest.main()
