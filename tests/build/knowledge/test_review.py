"""The reading of the appearance passages (src/build/knowledge/review.py), on invented passages: no network, no real data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.build.knowledge import review

CLASSES = {"melanoma": 6, "nevus": 4, "vascular lesion": 2}  # passages per class (by first topic)


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name)
        (self.data / "texts").mkdir()
        self.table = [{"passage_id": f"appearance-{label[:3]}-{i}", "topics": f"{label}; other{i}", "source": f"PMC{i}", "title": "T", "section": "S"}
                      for label, n in CLASSES.items() for i in range(n)]
        with (self.data / review.TABLE).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.table)
        (self.data / review.TEXTS).write_text("".join(json.dumps({"passage_id": r["passage_id"], "text": f"text of {r['passage_id']}"}) + "\n"
                                                      for r in self.table), encoding="utf-8")

    def fill(self, answers):
        """Writes the readings: ``answers`` is a list of (about_appearance, about_its_disease) per row of the drawn file."""
        path = self.data / review.REVIEW_FILE
        rows = read(path)
        for row, (a, b) in zip(rows, answers):
            row["about_appearance"], row["about_its_disease"] = a, b
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=review.COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


class DrawTests(ReviewTests):
    def test_the_draw_takes_the_classes_in_turn_so_that_a_small_class_is_read(self):
        done = review.draw(self.data, size=6)
        rows = read(self.data / review.REVIEW_FILE)
        self.assertEqual((done["passages"], len(rows)), (6, 6))
        self.assertEqual([r["topics"].split(";")[0] for r in rows], ["melanoma", "nevus", "vascular lesion"] * 2)  # a turn each, in class order
        self.assertEqual(list(rows[0]), review.COLUMNS)
        self.assertTrue(all(r["text"] == f"text of {r['passage_id']}" and r["about_appearance"] == "" for r in rows))
        self.assertEqual(len({r["passage_id"] for r in rows}), 6)

    def test_a_class_that_runs_out_is_skipped_and_a_draw_never_exceeds_the_passages_there_are(self):
        review.draw(self.data, size=100)
        rows = read(self.data / review.REVIEW_FILE)
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["topics"].split(";")[0] for r in rows].count("vascular lesion"), 2)

    def test_the_same_seed_draws_the_same_passages_and_another_seed_others(self):
        review.draw(self.data, size=6)
        first = read(self.data / review.REVIEW_FILE)
        (self.data / review.REVIEW_FILE).unlink()
        review.draw(self.data, size=6)
        self.assertEqual(read(self.data / review.REVIEW_FILE), first)
        (self.data / review.REVIEW_FILE).unlink()
        review.draw(self.data, size=6, seed=1)
        self.assertNotEqual([r["passage_id"] for r in read(self.data / review.REVIEW_FILE)], [r["passage_id"] for r in first])

    def test_a_file_that_may_hold_readings_is_never_overwritten(self):
        review.draw(self.data, size=6)
        self.fill([("yes", "yes")] * 6)
        before = (self.data / review.REVIEW_FILE).read_bytes()
        with self.assertRaisesRegex(ValueError, "may hold your readings"):
            review.draw(self.data, size=6)
        self.assertEqual((self.data / review.REVIEW_FILE).read_bytes(), before)

    def test_missing_texts_and_a_missing_table_are_named(self):
        (self.data / review.TEXTS).write_text(json.dumps({"passage_id": "other", "text": "x"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "12 passages have no text"):
            review.draw(self.data)
        (self.data / review.TEXTS).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "only on the computer that built them"):
            review.draw(self.data)
        (self.data / review.TABLE).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "enrich appearance first"):
            review.draw(self.data)


class ScoreTests(ReviewTests):
    def setUp(self):
        super().setUp()
        review.draw(self.data, size=10)

    def test_the_bar_is_met_at_nine_of_ten_and_the_report_holds_numbers_not_texts(self):
        self.fill([("yes", "yes")] * 9 + [("yes", "no")])
        done = review.score(self.data)
        self.assertEqual((done["n"], done["both"], done["meets"]), (10, 9, True))
        report = (self.data / review.REPORT).read_text(encoding="utf-8")
        for expected in ("| Does it describe how a lesion looks? | 10 of 10 | 100% |", "| Is it about the disease named in its topics? | 9 of 10 | 90% |",
                         "| **Both** | **9 of 10** | **90%** (95% interval 60% to 98%)", "**The bar is met**", "adding them is a separate decision"):
            self.assertIn(expected, report)
        self.assertNotIn("text of", report)

    def test_the_bar_is_not_met_below_ninety_percent_and_they_are_not_added(self):
        self.fill([("yes", "yes")] * 8 + [("no", "yes"), ("yes", "no")])
        done = review.score(self.data)
        self.assertEqual((done["both"], done["meets"]), (8, False))
        report = (self.data / review.REPORT).read_text(encoding="utf-8")
        self.assertIn("**The bar is not met**: 80% pass both questions (less than 90%). The passages are not added to the index.", report)

    def test_the_class_table_counts_what_each_class_passed(self):
        rows = read(self.data / review.REVIEW_FILE)
        answers = [("yes", "yes") if r["topics"].startswith("melanoma") else ("no", "yes") for r in rows]
        self.fill(answers)
        review.score(self.data)
        report = (self.data / review.REPORT).read_text(encoding="utf-8")
        self.assertIn("| melanoma | 4 | 4 |", report)
        self.assertIn("| nevus | 4 | 0 |", report)
        self.assertIn("| vascular lesion | 2 | 0 |", report)

    def test_answers_may_be_written_in_any_case_but_a_gap_or_another_word_stops_the_score(self):
        self.fill([(" YES", "Yes")] * 10)
        self.assertEqual(review.score(self.data)["both"], 10)
        self.fill([("yes", "yes")] * 4 + [("yes", "")] + [("yes", "yes")] * 5)
        with self.assertRaisesRegex(ValueError, "line 6 .*: about_its_disease must be yes or no, here ''"):
            review.score(self.data)
        self.fill([("maybe", "yes")] + [("yes", "yes")] * 9)
        with self.assertRaisesRegex(ValueError, "line 2 .*: about_appearance must be yes or no, here 'maybe'"):
            review.score(self.data)

    def test_a_file_that_was_not_drawn_yet_or_is_empty_is_named(self):
        (self.data / review.REVIEW_FILE).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run draw first"):
            review.score(self.data)
        (self.data / review.REVIEW_FILE).write_text(",".join(review.COLUMNS) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "holds no passage"):
            review.score(self.data)

    def test_the_command_line_turns_expected_problems_into_messages(self):
        with mock.patch("src.build.knowledge.review.KNOWLEDGE", self.data), mock.patch.object(review.draw, "__defaults__", (self.data, review.SIZE, review.SEED)):
            with self.assertRaisesRegex(SystemExit, "ERROR: .*may hold your readings"):
                review.main(["draw"])
        with mock.patch.object(review.score, "__defaults__", (self.data,)):
            with self.assertRaisesRegex(SystemExit, "ERROR: line 2"):
                review.main(["score"])


if __name__ == "__main__":
    unittest.main()
