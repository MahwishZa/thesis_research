"""The result tables (src/tables.py): filled from finished test runs, empty where a run is missing, and the same as docs/results.md."""

import contextlib
import io
import json
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from src import datasets, evaluate, tables

from tests.fixtures import RunProject
from tests.test_evaluate import write_run

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ["a", "a", "a", "a", "b", "b", "b", "b"]
GROUPS = [f"P{i}" for i in range(8)]
CLAUDE, GPT = "claude-sonnet-5-5", "gpt-4o"
CLAUDE_NAME = "Claude Sonnet 5.5"


class TableTests(RunProject):
    def put(self, dataset, model, predicted, name="test_aaaa1111", answers=None):
        folder = write_run(self.base / "results" / dataset / "closed_book" / model / name, TRUTH, predicted, GROUPS, dataset, answers)
        if datasets.DATASETS[dataset].missing_fact:  # a run of such a dataset saves the patient's facts with each answer
            rows = [json.loads(line) | {"facts": "unknown" if i == 0 else "sex: female"} for i, line in enumerate((folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines())]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return folder

    def runs(self):
        return tables.Runs(self.base / "results")

    def test_a_finished_run_fills_its_cells_and_a_missing_run_leaves_them_empty(self):
        # Truth a a a a b b b b; answers a a a b b b b and one invalid. Recall: a 3/4, b 3/4, so balanced accuracy 75.0% (and top-3, as each
        # differential holds one class). a: precision 3/3, F1 6/7; b: precision 3/4, F1 3/4; macro-F1 = (6/7 + 3/4) / 2 = 80.4%.
        # All 7 valid answers say "yes" (coverage 7/8 = 87.5%); one of them is wrong (selective risk 1/7 = 14.3%).
        self.put("clinic_photo", CLAUDE, ["a", "a", "a", "b", "b", "b", "b", None])
        runs = self.runs()
        cells = [c.strip() for c in tables.by_dataset(runs, "clinic_photo")[2].strip("|").split("|")]
        self.assertEqual(cells[:2] + cells[3:], ["Claude Sonnet 5.5", "75.0%", "80.4%", "75.0%", "1 (12.5%)", "0", "87.5%", "14.3% (1 of 7)", "`test_aaaa1111`"])
        self.assertEqual(tables.by_dataset(runs, "clinic_photo")[3], "| GPT-4o |  |  |  |  |  |  |  |  |  |")
        overview = {row.split("|")[1].strip(): row for row in tables.overview(runs)[2:]}
        self.assertEqual(overview["`published_text`"], "| `published_text` |  |  |  |  |")
        self.assertRegex(overview["`clinic_photo`"], r"^\| `clinic_photo` \| 75\.0% \(\d+\.\d to \d+\.\d\) \|  \|  \|  \|$")
        self.assertEqual(tables.per_class(runs, "clinic_photo")[2:], ["| a | 4 | 75.0 |  |  |  |", "| b | 4 | 75.0 |  |  |  |"])

    def test_questions_the_host_refused_are_counted_apart_and_are_among_the_invalid_answers(self):
        folder = self.put("clinic_photo", GPT, ["a", "a", "a", "a", "b", "b", "b", "b"])
        rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0] |= {"prediction": None, "differential": None, "evidence_sufficient": None, "reason": None, "claims": None, "evidence": None,
                    "invalid": "refused by the host's content moderation (HTTP 403: flagged)", "raw_text": None, "finish_reason": "REFUSED",
                    "model_version": None}
        (folder / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        cells = [c.strip() for c in tables.by_dataset(self.runs(), "clinic_photo")[3].strip("|").split("|")]
        self.assertEqual((cells[0], cells[5], cells[6]), ("GPT-4o", "1 (12.5%)", "1"))  # one invalid answer, and it is a refusal
        self.put("clinic_photo", CLAUDE, TRUTH)  # a model with no refusal: its main score stands, nothing was left out
        self.put("clinic_record", CLAUDE, TRUTH)  # a dataset where nothing was refused is not listed
        rows = tables.without_refused(self.runs())[2:]
        self.assertEqual([r.split("|")[1].strip() for r in rows], ["`clinic_photo`"] * 4)  # only a dataset with a refusal is listed
        by_model = {r.split("|")[2].strip(): [c.strip() for c in r.strip("|").split("|")] for r in rows}
        self.assertEqual((by_model["GPT-4o"][2], by_model["GPT-4o"][3]), ("7", "100.0%"))  # the 7 questions it saw, all right
        self.assertEqual((by_model[CLAUDE_NAME][2], by_model[CLAUDE_NAME][3]), ("8", "100.0%"))
        self.assertEqual(by_model["Gemini 3.8 Flash"][2:], ["", "", "", ""])  # no run

    def test_a_flag_that_is_never_yes_has_no_selective_risk(self):
        self.put("clinic_photo", CLAUDE, TRUTH, answers=[([t], "no") for t in TRUTH])
        self.assertIn("| 0.0% | n/a |", tables.by_dataset(self.runs(), "clinic_photo")[2])

    def test_runs_in_progress_and_validation_runs_are_not_results_and_two_finished_runs_are_refused(self):
        base = self.base / "results" / "clinic_photo" / "closed_book" / CLAUDE
        (base / "test_bbbb2222.partial").mkdir(parents=True)
        (base / "val_cccc3333").mkdir()
        self.assertIsNone(self.runs().folder("clinic_photo", CLAUDE))
        self.put("clinic_photo", CLAUDE, TRUTH)
        self.assertEqual(self.runs().folder("clinic_photo", CLAUDE).name, "test_aaaa1111")
        self.put("clinic_photo", CLAUDE, TRUTH, name="test_dddd4444")
        with self.assertRaisesRegex(ValueError, "more than one finished test run"):
            self.runs().folder("clinic_photo", CLAUDE)

    def test_the_difference_is_run_b_minus_run_a_and_the_runs_are_named_in_that_order(self):
        self.put("clinic_photo", CLAUDE, ["a", "a", "b", "b", "b", "b", "a", "a"])  # balanced 50.0%
        self.put("clinic_photo_record", CLAUDE, TRUTH)  # balanced 100.0%
        rows = tables.same_model(self.runs())
        row = next(r for r in rows if r.startswith("| Claude Sonnet 5.5 | `clinic_photo` | `clinic_photo_record`"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[3:5], ["8", "+50.0"])
        self.assertEqual(cells[7], "4 / 0")  # right only in B / only in A
        pending = next(r for r in rows if r.startswith("| GPT-4o | `clinic_photo` | `clinic_photo_record`"))
        self.assertEqual(pending, "| GPT-4o | `clinic_photo` | `clinic_photo_record` | 8 |  |  |  |  |  |")  # the question count is known; nothing else

    def test_models_are_compared_with_the_first_model_as_run_a(self):
        self.put("clinic_photo", CLAUDE, ["a", "a", "b", "b", "b", "b", "a", "a"])
        self.put("clinic_photo", GPT, TRUTH)
        row = next(r for r in tables.models_against_each_other(self.runs()) if r.startswith("| `clinic_photo` | GPT-4o"))
        self.assertTrue(row.startswith("| `clinic_photo` | GPT-4o | 8 | +50.0 |"), row)

    def test_the_beside_tables_follow_the_dataset_not_the_run(self):
        self.assertEqual([r.split("|")[1].strip() for r in tables.without_small_classes(self.runs())[2::4]],
                         ["`published_text`", "`published_text_photos`", "`published_photos`"])
        self.assertEqual([r.split("|")[1].strip() for r in tables.complete_record(self.runs())[2::4]], ["`clinic_record`", "`clinic_photo_record`"])

    def test_the_command_writes_the_tables_in_utf8_and_prints_them_without_a_file(self):
        self.put("clinic_photo", CLAUDE, TRUTH)
        target = self.base / "tables.md"
        with mock.patch("src.tables.RESULTS", self.base / "results"), contextlib.redirect_stdout(io.StringIO()) as shown:
            evaluate.main(["tables", "--output", str(target)])
        text = target.read_text(encoding="utf-8")
        self.assertIn("The tables are written to", shown.getvalue())
        self.assertTrue(text.startswith("## 2. Overview") and text.endswith("|\n"))
        self.assertIn("| Claude Sonnet 5.5 | 100.0% |", text)
        self.assertNotIn("\r", target.read_bytes().decode("utf-8"))


class RepositoryTests(unittest.TestCase):
    def test_the_models_and_datasets_of_the_tables_are_those_of_the_configuration(self):
        names = [m["name"] for m in tomllib.loads((ROOT / "configs" / "models.toml").read_text(encoding="utf-8"))["models"]]
        self.assertEqual([m for m, _ in tables.MODELS], names)
        self.assertEqual(sorted(tables.SEES), sorted(datasets.DATASETS))
        self.assertEqual({d for pair in tables.PAIRS for d in pair}, set(datasets.DATASETS))

    @unittest.skipUnless((ROOT / "docs" / "results.md").is_file(), "docs/ is removed when the research is final")
    def test_every_table_line_of_the_command_is_in_the_results_page(self):
        """The page and the runs agree: after a run is committed, `python -m src.evaluate tables` and docs/results.md must be brought
        together (a drift fails here)."""
        page = set((ROOT / "docs" / "results.md").read_text(encoding="utf-8").split("\n"))
        missing = [line for line in tables.render(ROOT / "results").split("\n") if line.startswith("|") and line not in page]
        self.assertEqual(missing, [], "docs/results.md lacks lines of `python -m src.evaluate tables`")


if __name__ == "__main__":
    unittest.main()
