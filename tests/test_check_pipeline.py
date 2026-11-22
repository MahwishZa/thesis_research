"""Checks of src/check_pipeline.py on small made-up folders: each step is reported done, not run or with a problem."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import check_pipeline as cp  # noqa: E402
from common import ALL_CASES, CLASSES, PROCESSED_CASES, question_file  # noqa: E402

DATA = ROOT / "data"


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.results = Path(self.tmp.name, "data"), Path(self.tmp.name, "results")

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_empty_workspace_is_all_not_run(self):
        rows = {name: status for name, status, _ in cp.check(str(self.data), str(self.results))}
        self.assertEqual(set(rows.values()), {cp.NOT_RUN})

    def test_reference_texts_old_form_or_empty_are_a_problem(self):
        path = self.data / "knowledge_corpus" / "reference_texts.json"
        path.parent.mkdir(parents=True)
        good = {"texts": [{"section": "Dermoscopic criteria", "text": "x", "n_words": 120}]}
        path.write_text(json.dumps({**{c: good for c in CLASSES}, "BCC": {"texts": good["texts"] * 3}}), encoding="utf-8")
        status, note = cp.step_references(str(self.data))
        self.assertEqual(status, cp.DONE)
        self.assertEqual(note, "10 reference texts (MEL 1, NV 1, BCC 3, AK 1, BKL 1, DF 1, VASC 1, SCC 1)")
        path.write_text(json.dumps({**{c: good for c in CLASSES}, "NV": {"texts": []}}), encoding="utf-8")
        self.assertEqual(cp.step_references(str(self.data)), (cp.PROBLEM, "no text: ['NV']"))
        path.write_text(json.dumps({c: good for c in CLASSES if c != "DF"}), encoding="utf-8")
        self.assertEqual(cp.step_references(str(self.data)), (cp.PROBLEM, "missing ['DF']"))
        path.write_text(json.dumps({c: {"text": "x", "n_words": 120} for c in CLASSES}), encoding="utf-8")
        status, note = cp.step_references(str(self.data))
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("old form", note)

    def test_missing_and_left_over_images_are_named(self):
        cases = self.data / PROCESSED_CASES
        cases.mkdir(parents=True)
        for f in cp.PROCESSED_FILES:
            (cases / f).write_text("isic_id,image_path\n", encoding="utf-8")
        (self.data / ALL_CASES).write_text("isic_id,image_path\nISIC_1,processed_images/ISIC_1.jpg\n"
                                           "ISIC_2,processed_images/ISIC_2.jpg\n", encoding="utf-8")
        images = self.data / "processed_images"
        images.mkdir()
        for name in ("ISIC_1.jpg", "ISIC_2.jpg"):
            (images / name).write_bytes(b"")
        self.assertEqual(cp.step_clean(str(self.data)), (cp.DONE, "2 cases, every image present"))
        (images / "ISIC_9.jpg").write_bytes(b"")
        (images / "notes.txt").write_text("not an image", encoding="utf-8")
        status, note = cp.step_clean(str(self.data))
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("1 images not in the case table", note)
        self.assertIn("ISIC_9.jpg", note)
        self.assertNotIn("missing", note)
        (images / "ISIC_2.jpg").unlink()
        status, note = cp.step_clean(str(self.data))
        self.assertIn("1 images missing", note)
        self.assertIn("ISIC_2", note)
        self.assertIn("ISIC_9.jpg", note)

    @unittest.skipUnless((DATA / ALL_CASES).is_file(), "data/processed_cases/all_cases.csv is not there")
    def test_views_that_differ_from_a_fresh_build_are_a_problem(self):
        shutil.copytree(DATA / PROCESSED_CASES, self.data / PROCESSED_CASES)
        shutil.copytree(DATA / "views", self.data / "views")
        self.assertEqual(cp.step_views(str(self.data))[0], cp.DONE)
        with open(self.data / question_file("text_only", "test"), "a", encoding="utf-8") as f:
            f.write("\n")
        self.assertEqual(cp.step_views(str(self.data))[0], cp.PROBLEM)

    def run_folder(self, partial=False, sha="x", report=False):
        folder = self.results / "llm_only" / "gpt-4o" / ("text_only_test" + (".partial" if partial else ""))
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"settings": {"questions_sha256": sha}, "status": "stopped",
                                                     "answered": 3, "of": 315}), encoding="utf-8")
        if report:
            (folder / "report.json").write_text("{}", encoding="utf-8")
        return folder

    def runs(self):
        return {n: (s, note) for n, s, note in cp.step_runs(str(self.data), str(self.results), ["gpt-4o"])}

    def write_questions(self):
        q = self.data / question_file("text_only", "test")
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text('{"id": "a"}\n', encoding="utf-8")
        return cp.sha256(q)

    def test_an_unfinished_run_is_a_problem_with_its_progress(self):
        self.run_folder(partial=True)
        status, note = self.runs()["llm_only gpt-4o text_only test"]
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("3/315", note)

    def test_a_run_on_old_questions_or_without_a_report_is_a_problem(self):
        sha = self.write_questions()
        folder = self.run_folder(sha="old")
        self.assertIn("question file changed", self.runs()["llm_only gpt-4o text_only test"][1])
        shutil.rmtree(folder)
        folder = self.run_folder(sha=sha)
        self.assertIn("not evaluated", self.runs()["llm_only gpt-4o text_only test"][1])
        (folder / "report.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.runs()["llm_only gpt-4o text_only test"][0], cp.DONE)

    def test_the_exit_code_is_1_only_when_a_step_has_a_problem(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cp.main(["--data", str(self.data), "--results", str(self.results)]), 0)
            self.run_folder(partial=True)
            self.assertEqual(cp.main(["--data", str(self.data), "--results", str(self.results)]), 1)


if __name__ == "__main__":
    unittest.main()
