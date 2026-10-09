"""Paper-style tables and figures (synthetic data; no experiment is run)."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import report as R

import importlib.util

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None

S, RF, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"


def make_items(n_changed=6, n_unchanged=4):
    items = {}
    labels = [S, RF, N]
    for k in range(n_changed + n_unchanged):
        kind = "changed" if k < n_changed else "unchanged"
        items[f"i{k}"] = {"item_id": f"i{k}", "kind": kind, "split": "dev", "likely_label_noise": False,
                          "newest": {"label": labels[k % 3], "row": k, "date": "2020-01-01"},
                          "previous": {"label": S, "date": "2010-01-01"}}
    return items


def make_answers(items, perfect=("B1",), wrong=("B0",)):
    out = {}
    for i, it in items.items():
        for arm in perfect:
            out[(i, arm)] = {"verdict": it["newest"]["label"]}
        for arm in wrong:
            out[(i, arm)] = {"verdict": S}
    return out


class NumbersTests(unittest.TestCase):

    def setUp(self):
        self.items = make_items()
        self.answers = make_answers(self.items)

    def test_accuracy_cell_counts_and_interval(self):
        c = R.accuracy_cell(self.items, self.answers, "B1", ("changed",))
        self.assertEqual((c["n"], c["correct"], c["accuracy"]), (6, 6, 1.0))
        self.assertLessEqual(c["ci"][0], 1.0)
        self.assertIsNone(R.accuracy_cell(self.items, self.answers, "absent", ("changed",)))

    def test_system_rows_are_grouped_and_skip_missing_arms(self):
        rows = R.system_rows(self.items, self.answers, ["B0", "B1"], [])
        self.assertEqual([r["arm"] for r in rows], ["B0", "B1"])
        self.assertTrue(all(r["group"].startswith("Llama-3-8B") for r in rows))
        self.assertEqual(rows[1]["all"]["accuracy"], 1.0)

    def test_realigned_arms_get_their_own_group_table_rows_and_class_rows(self):
        answers = make_answers(self.items, perfect=("B1", "R2V"), wrong=("B0", "R2"))
        rows = R.system_rows(self.items, answers, ["B0", "B1", "R2", "R2V"], [])
        self.assertEqual([r["arm"] for r in rows], ["B0", "B1", "R2", "R2V"])
        self.assertTrue(rows[2]["group"].startswith("Adapted RAG²") and rows[3]["group"] == rows[2]["group"])
        self.assertEqual(rows[3]["all"]["accuracy"], 1.0)
        self.assertEqual([r["arm"] for r in R.filtering_rows(self.items, answers, ["B1", "R2", "R2V"])], ["B1", "R2", "R2V"])
        self.assertIn("R2V", [r["arm"] for r in R.class_rows(self.items, answers, ["B0", "R2V"])])

    def test_released_verdicts_are_read_by_row_and_feed_the_closed_book_rows(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Code" / "GeneratedAnswers"
            folder.mkdir(parents=True)
            lines = ["LABEL: SUPPORTED"] * 10 + [""]
            (folder / "qwen25-7b_answers.txt").write_text("\n".join(lines), encoding="utf-8")
            extra = R.released_verdicts(self.items, Path(tmp))
        self.assertEqual(extra[("i0", "R:qwen25-7b")]["verdict"], S)
        self.assertNotIn(("i0", "R:gpt4o-mini"), extra)                      # a missing file is skipped
        answers = dict(self.answers)
        answers.update(extra)
        rows = R.system_rows(self.items, answers, ["B1"], ["qwen25-7b"])
        self.assertEqual(rows[0]["name"], "Qwen2.5-7B")
        self.assertAlmostEqual(rows[0]["all"]["accuracy"], 4 / 10)            # 4 of 10 gold labels are SUPPORTED

    def test_filtering_rows_pair_each_arm_with_b1(self):
        answers = make_answers(make_items(40, 0), perfect=("B1",), wrong=("B2",))
        rows = R.filtering_rows(make_items(40, 0), answers, ["B1", "B2"])
        self.assertEqual([r["arm"] for r in rows], ["B1", "B2"])
        self.assertIsNone(rows[0]["diff_vs_baseline"])
        self.assertLess(rows[1]["diff_vs_baseline"]["diff_a_minus_b"], 0)

    def test_class_rows_and_constant_baselines(self):
        rows = R.class_rows(self.items, self.answers, ["B0", "B1"])
        self.assertEqual(rows[0]["recall"][S], 1.0)
        self.assertEqual(rows[0]["recall"][RF], 0.0)
        self.assertEqual(rows[1]["recall"][RF], 1.0)
        const = R.constant_baselines(self.items)
        self.assertAlmostEqual(const["changed"][S], 2 / 6)
        self.assertAlmostEqual(sum(const["all"].values()), 1.0)


class TextTests(unittest.TestCase):

    def setUp(self):
        self.items = make_items()
        self.answers = make_answers(self.items)
        self.rows = R.system_rows(self.items, self.answers, ["B0", "B1"], [])

    def test_systems_table_bolds_the_best_value_per_column_and_states_n(self):
        text = R.table_systems(self.rows, {"changed": 6, "unchanged": 4, "all": 10})
        self.assertIn("Changed (n = 6)", text)
        self.assertIn("| B1: + MedCPT retrieval, top-5 (standard RAG) | **100.0** |", text)
        self.assertNotIn("**", text.split("| B0:")[1].split("\n")[0])

    def test_filtering_and_class_tables_render_every_row(self):
        answers = make_answers(make_items(40, 0), perfect=("B1",), wrong=("B2",))
        items = make_items(40, 0)
        t = R.table_filtering(R.filtering_rows(items, answers, ["B1", "B2"]))
        self.assertIn("exact McNemar p", t)
        self.assertEqual(t.count("\n"), 4)
        c = R.table_classes(R.class_rows(items, answers, ["B1", "B2"]))
        self.assertIn("macro-F1 (%)", c)

    def test_latex_escapes_and_marks_best(self):
        tex = R.to_latex(self.rows, "Accuracy (%) & more", "tab:x")
        self.assertIn("\\toprule", tex)
        self.assertIn("\\textbf{100.0}", tex)
        self.assertIn("Accuracy (\\%) \\& more", tex)
        self.assertIn("\\label{tab:x}", tex)


class AssemblyTests(unittest.TestCase):

    def build(self, split="dev"):
        items = make_items(40, 20)
        answers = make_answers(items, perfect=("B1", "R2V"), wrong=("B0", "B2", "B3", "P", "C1", "R2"))
        audit = {"agreement": 0.83, "kappa": 0.74, "n_items": 226, "n_stable": 188, "label_change_reproduced": 0.68,
                 "per_gold_class": {}}
        return R.build(split, items, answers, [], audit)

    def test_markdown_is_honest_about_status_and_the_base_paper(self):
        text = R.report_markdown(self.build(), {"fig1": True})
        for needle in ("dev split, exploratory", "not comparable with theirs", "Table 1", "Table 2", "Table 3",
                       "Label reproducibility", "Relation to the base paper", "fig1_accuracy_by_system.png",
                       "R2 is an adapted RAG², not a reproduction"):
            self.assertIn(needle, text)
        self.assertNotIn("fig2_class_recall.png", text)                          # only figures that exist are linked
        self.assertNotIn("Table 4", text)

    def test_write_report_without_figures_writes_markdown_latex_and_data(self):
        with TemporaryDirectory() as tmp:
            figs = R.write_report(self.build(), Path(tmp), make_figures=False)
            names = {p.name for p in Path(tmp).iterdir()}
            data = json.loads((Path(tmp) / "report_data_dev.json").read_text(encoding="utf-8"))
        self.assertFalse(any(figs.values()))
        self.assertEqual(names, {"REPORT.md", "tables.tex", "report_data_dev.json"})
        self.assertEqual(data["split"], "dev")

    @unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
    def test_write_report_makes_the_two_figures(self):
        with TemporaryDirectory() as tmp:
            figs = R.write_report(self.build(), Path(tmp))
            pngs = sorted(p.name for p in Path(tmp).glob("*.png"))
        self.assertTrue(all(figs.values()), figs)
        self.assertEqual(pngs, ["fig1_accuracy_by_system.png", "fig2_class_recall.png"])

    def test_confirmatory_report_has_no_dev_only_sections(self):
        items = make_items(40, 20)
        d = R.build("confirm", items, make_answers(items), [], None)
        text = R.report_markdown(d, {})
        self.assertIn("confirm split, confirmatory", text)
        self.assertNotIn("Dev split, exploratory", text)
        self.assertNotIn("Closed-book rows", text)                               # no released answers were given


class CliTests(unittest.TestCase):

    def test_cli_end_to_end_on_synthetic_files(self):
        items = make_items(40, 20)
        answers = make_answers(items)
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, res = tmp / "data", tmp / "results"
            data.mkdir()
            res.mkdir()
            (data / "benchmark.jsonl").write_text("".join(json.dumps(v) + "\n" for v in items.values()), encoding="utf-8")
            (res / "answers_dev.jsonl").write_text(
                "".join(json.dumps({"item_id": i, "arm": a, "verdict": r["verdict"]}) + "\n" for (i, a), r in answers.items()),
                encoding="utf-8")
            realigned = make_answers(items, perfect=("R2V",), wrong=("R2",))
            (res / "rag2_answers_dev.jsonl").write_text(
                "".join(json.dumps({"item_id": i, "arm": a, "verdict": r["verdict"]}) + "\n" for (i, a), r in realigned.items()),
                encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = R.main(["--data-dir", str(data), "--results-dir", str(res), "--no-figures"])
                missing = R.main(["--data-dir", str(tmp / "none"), "--results-dir", str(res)])
            exists = (res / "report_dev" / "REPORT.md").is_file()
            arms = {r["arm"] for r in json.loads((res / "report_dev" / "report_data_dev.json").read_text(encoding="utf-8"))["systems"]}
        self.assertEqual((code, missing, exists), (0, 2, True))
        self.assertTrue({"R2", "R2V"} <= arms)                                   # the realigned arms are read too


class ReleasedAnswerFilesAndUnchangedOnlySplitsTests(unittest.TestCase):
    def test_three_line_files_are_read_as_the_authors_do(self):
        import tempfile
        from experiments.medchange.headroom import FILES, read_answers
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / FILES["biomistral"]).write_text(
                "Question: a?\nAnswer: REFUTED: no ||| benefit\n---\nQuestion: b?\nAnswer: unclear text\n---\n"
                "Question: c?\nAnswer: SUPPORTED\n---", encoding="utf-8")
            (folder / FILES["qwen25-7b"]).write_text("LABEL: SUPPORTED\nNOT ENOUGH INFORMATION: x\n", encoding="utf-8")
            self.assertEqual(read_answers(folder, "biomistral"), ["REFUTED", None, "SUPPORTED"])
            self.assertEqual(read_answers(folder, "qwen25-7b"), ["SUPPORTED", "NOT ENOUGH INFORMATION", None])
            self.assertIsNone(read_answers(folder, "pmcllama"))

    def test_a_split_without_changed_questions_has_no_changed_column_and_no_changed_only_tables(self):
        items = {f"q{i}": {"item_id": f"q{i}", "kind": "unchanged", "newest": {"label": "NOT ENOUGH INFORMATION", "row": i}}
                 for i in range(6)}
        answers = {(i, "B1"): {"verdict": "NOT ENOUGH INFORMATION" if n % 2 else "SUPPORTED"} for n, i in enumerate(items)}
        d = R.build("ad", items, answers, [], None)
        text = R.report_markdown(d, {})
        self.assertEqual((d["filtering"], d["classes"], d["status"]), ([], [], "secondary held-out test"))
        self.assertNotIn("Changed", text)
        self.assertIn("All (n = 6)", text)
        self.assertNotIn("Unchanged", text)
        self.assertIn("the best constant answer is NOT ENOUGH INFORMATION", text)


if __name__ == "__main__":
    unittest.main()
