"""The tables and the command line of the analysis of the realigned study (synthetic answers; no model)."""

import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import analyze_rag2 as AR
from experiments.medchange.analysis import build_report
from tests.fixtures import world, write


class TablesTests(unittest.TestCase):
    def test_the_markdown_has_the_tables_and_the_reading(self):
        items, answers = world()
        rep = build_report(items, answers, "confirm")
        text = AR.to_markdown(rep)
        for needle in ("verdict accuracy", "R2V against the adapted RAG² baseline", "Secondary comparisons", "Retrieval", "**confirmed**"):
            self.assertIn(needle, text)
        self.assertNotIn("requirement", text.lower())

    def test_the_markdown_mentions_answers_written_after_the_abstracts_were_shortened(self):
        items, answers = world(n=6)
        answers[("MC-000", "R2")]["context_truncated"] = True
        self.assertIn("shortened", AR.to_markdown(build_report(items, answers, "confirm")))


class CliTests(unittest.TestCase):
    def test_nothing_to_analyse_is_reported_and_a_complete_split_writes_the_json_and_the_markdown(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(AR.main(["--split", "confirm", "--data-dir", tmp]), 2)
            items, answers = world(n=12)
            write(d / "benchmark.jsonl", list(items.values()))
            write(d / "answers_confirm.jsonl", [r for (i, a), r in answers.items() if a in ("B0", "B1")])
            write(d / "rag2_answers_confirm.jsonl", [r for (i, a), r in answers.items() if a.startswith("R2")])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(AR.main(["--split", "confirm", "--data-dir", tmp, "--out-dir", tmp]), 0)
            self.assertTrue((d / "rag2_analysis_confirm.md").is_file())
            self.assertTrue((d / "rag2_analysis_confirm.json").is_file())


if __name__ == "__main__":
    unittest.main()
