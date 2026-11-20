"""The no-cost pipeline checks (src/validate.py): each check on small hand-made views, then the whole run on the committed tables."""

import contextlib
import csv
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import validate
from src.datasets import Item

ROOT = Path(__file__).resolve().parents[1]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item(split, label, name, group, ending="jpg", content=None, **extra):
    return Item(split, label, f"{split}/{label}/{name}.{ending}", sha(content or name + ending), group=group, **extra)


def views():
    """Three clinic views of 4 questions (a patient P1 with two photos in train; P2 in val; P3 in test), aligned as the real ones."""
    rows = [("train", "melanoma", "P1_1", "P1"), ("train", "nevus", "P1_2", "P1"), ("val", "melanoma", "P2_1", "P2"), ("test", "nevus", "P3_1", "P3")]
    photo = [item(s, c, n, g) for s, c, n, g in rows]
    record = [item(s, c, n, g, "txt", content=f"record of {n}") for s, c, n, g in rows]
    both = [item(s, c, n, g, facts=f"record of {n}") for s, c, n, g in rows]
    return photo, record, both


class TableTests(unittest.TestCase):
    def test_a_consistent_table_has_no_problem(self):
        self.assertEqual(validate.table_problems("clinic_photo", views()[0] + [item("train", "melanoma", "P4_1", "P4"), item("val", "nevus", "P5_1", "P5"),
                                                                              item("test", "melanoma", "P6_1", "P6")]), [])

    def test_a_patient_in_two_splits_is_a_leak(self):
        photo = views()[0] + [item("test", "nevus", "P1_3", "P1")]
        self.assertEqual(validate.table_problems("clinic_photo", photo), ["patients in more than one split: P1"])

    def test_a_missing_group_a_split_without_questions_and_an_unknown_class_are_named(self):
        photo = [item("train", "melanoma", "a", ""), item("val", "melanoma", "b", "P2")]
        self.assertEqual(validate.table_problems("clinic_photo", photo),
                         ["a question has no group (patient or article)", "no question in the split(s): test"])
        stray = views()[0] + [item("test", "vascular lesion", "P9_1", "P9")]
        self.assertIn("classes without training files", validate.table_problems("clinic_photo", stray)[0])


class AlignmentTests(unittest.TestCase):
    def test_aligned_views_have_no_problem_in_any_order(self):
        photo, record, both = views()
        self.assertEqual(validate.pair_problems("clinic_photo", photo, "clinic_record", record), [])
        self.assertEqual(validate.pair_problems("clinic_record", record, "clinic_photo_record", both), [])
        self.assertEqual(validate.same_content_problems("clinic_photo", photo, "clinic_photo_record", both, "photo"), [])
        self.assertEqual(validate.record_problems(record, both), [])

    def test_a_view_with_another_question_split_class_or_group_is_found(self):
        photo, record, _ = views()
        self.assertIn("1 only in clinic_photo, 0 only in clinic_record", validate.pair_problems("clinic_photo", photo, "clinic_record", record[:3])[0])
        moved = [item("test", "melanoma", "P2_1", "P2", "txt", content="record of P2_1") if r.path.startswith("val/") else r for r in record]
        self.assertIn("the questions differ: 1 only in clinic_photo, 1 only in clinic_record",  # another split is another question
                      validate.pair_problems("clinic_photo", photo, "clinic_record", moved)[0])
        renamed = [item("val", "nevus", "P2_1", "P2", "txt") if r.path.startswith("val/") else r for r in record]
        self.assertIn("the questions differ: 1 only in clinic_photo, 1 only in clinic_record",  # and another class too
                      validate.pair_problems("clinic_photo", photo, "clinic_record", renamed)[0])
        regrouped = [item("train", "melanoma", "P1_1", "P9", "txt") if r.path.startswith("train/melanoma/P1_1") else r for r in record]
        self.assertTrue(any("group is 'P1' in clinic_photo and 'P9' in clinic_record" in p for p in validate.pair_problems(
            "clinic_photo", photo, "clinic_record", regrouped)))

    def test_views_with_other_answer_options_are_found(self):
        photo, record, _ = views()
        only_melanoma = [r for r in record if "/melanoma/" in r.path]
        problems = validate.pair_problems("clinic_photo", photo, "clinic_record", only_melanoma)
        self.assertIn("the answer options differ", problems)
        self.assertIn("the questions differ: 2 only in clinic_photo, 0 only in clinic_record", problems[0])

    def test_another_photo_or_another_record_text_in_the_combined_view_is_found(self):
        photo, record, both = views()
        other_photo = [Item(b.split, b.label, b.path, "0" * 64, group=b.group, facts=b.facts) if b.path.startswith("val/") else b for b in both]
        self.assertEqual(validate.same_content_problems("clinic_photo", photo, "clinic_photo_record", other_photo, "photo"),
                         ["val/melanoma/P2_1: not the same photo in clinic_photo and clinic_photo_record"])
        other_record = [Item(b.split, b.label, b.path, b.sha256, group=b.group, facts=b.facts + " ") if b.path.startswith("test/") else b for b in both]
        self.assertEqual(validate.record_problems(record, other_record),
                         ["test/nevus/P3_1: the record in the photo + record view is not the record file of the record view"])

    def test_a_view_of_the_cases_with_photos_may_ask_some_of_the_cases_but_not_others(self):
        text = [item(s, c, n, g, "txt") for s, c, n, g in (("train", "melanoma", "1-1", "1"), ("train", "melanoma", "1-2", "4"),
                                                           ("train", "nevus", "2-1", "2"), ("test", "nevus", "3-1", "3"))]
        some = [t for t in text if "1-1" not in t.path]
        self.assertEqual(validate.pair_problems("published_text", text, "published_photos", some), [])
        self.assertEqual(validate.pair_problems("published_photos", some, "published_text", text), [])
        extra = some + [item("test", "nevus", "9-1", "9", "txt")]
        self.assertIn("1 questions of published_photos are not questions of published_text (e.g. test/nevus/9-1)",
                      validate.pair_problems("published_text", text, "published_photos", extra)[0])

    def test_the_same_case_and_the_same_pictures_in_the_views_of_a_case(self):
        text = [item("train", "nevus", "2-1", "2", "txt")]
        pictures = (("a/f1.jpg", sha("f1")), ("a/f2.jpg", sha("f2")))
        both = [Item(t.split, t.label, t.path, t.sha256, group=t.group, pictures=pictures) for t in text]
        only = [Item(t.split, t.label, t.path, t.sha256, group=t.group, pictures=pictures[::-1]) for t in text]
        self.assertEqual(validate.case_problems(text, both), [])
        self.assertEqual(validate.picture_problems("published_text_photos", both, "published_photos", both), [])
        self.assertEqual(validate.picture_problems("published_text_photos", both, "published_photos", only),
                         ["train/nevus/2-1: other pictures in published_text_photos than in published_photos"])
        other = [Item(t.split, t.label, t.path, "1" * 64, group=t.group, pictures=pictures) for t in text]
        self.assertEqual(validate.case_problems(text, other), ["train/nevus/2-1: not the same case in the text view"])


class PromptTests(unittest.TestCase):
    """The prompts of the repository, and copies of them changed one way at a time."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.prompts = self.base / "prompts"
        for folder in (ROOT / "prompts").iterdir():
            (self.prompts / folder.name).mkdir(parents=True)
            for file in folder.iterdir():
                (self.prompts / folder.name / file.name).write_bytes(file.read_bytes())
        patcher = mock.patch("src.validate.prompt_file", lambda dataset, approach: self.prompts / dataset / f"{approach}.txt")
        patcher.start()
        self.addCleanup(patcher.stop)

    def edit(self, dataset, old, new):
        path = self.prompts / dataset / "closed_book.txt"
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new), encoding="utf-8")

    def test_the_prompts_of_the_repository_pass(self):
        self.assertEqual(validate.prompt_problems(), [])

    def test_a_prompt_whose_answer_block_differs_from_the_others_is_found(self):
        self.edit("clinic_record", "reason: at most three sentences", "reason: at most four sentences")
        problems = validate.prompt_problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("clinic_record/closed_book.txt: the answer-format block differs", problems[0])

    def test_an_example_answer_with_other_keys_than_the_reader_requires_is_found(self):
        for dataset in ("clinic_photo", "clinic_record", "clinic_photo_record", "published_text", "published_text_photos", "published_photos"):
            self.edit(dataset, '"reason": "<why>", "claims"', '"claims": [{"statement": "<one short claim>", "evidence": []}], "reason": "<why>", "claims_"')
        problems = validate.prompt_problems()
        self.assertEqual(len(problems), 6)
        self.assertIn("the example answer does not have the keys", problems[0])

    def test_a_prompt_without_its_placeholder_or_answer_block_is_found(self):
        self.edit("published_text", "{case}", "the case")
        self.edit("clinic_photo", "Answer with one JSON object and nothing else", "Answer")
        problems = validate.prompt_problems()
        self.assertTrue(any("must contain {case} exactly once" in p for p in problems), problems)
        self.assertTrue(any("clinic_photo/closed_book.txt has no answer-format block" in p for p in problems), problems)


class ModelListTests(unittest.TestCase):
    def setUp(self):
        previous = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, previous)

    def list_of(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "models.toml"
        path.write_text(text, encoding="utf-8")
        return mock.patch("src.validate.MODELS", path)

    def test_the_model_list_of_the_repository_passes_and_every_entry_is_tried_on_every_dataset(self):
        self.assertEqual(validate.config_problems(), [])

    def test_an_entry_with_a_missing_setting_is_named_for_every_dataset(self):
        with self.list_of('[[models]]\nname = "m"\nrequests_per_minute = 1\n[models.settings]\nprovider = "anthropic"\nid = "x"\n'):
            problems = validate.config_problems()
        self.assertEqual(len(problems), len(validate.DATASETS) * len(validate.APPROACHES))
        self.assertTrue(all("missing order_seed" in p for p in problems), problems)
        self.assertEqual(sorted(p.split(":")[0] for p in problems), sorted(f"m on {d}/closed_book" for d in validate.DATASETS))

    def test_an_empty_model_list_is_a_problem(self):
        with self.list_of("models = []\n"):
            self.assertEqual(len(validate.config_problems()), 1)


class WholeRunTests(unittest.TestCase):
    def setUp(self):
        previous = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, previous)

    def run_validate(self, **kwargs):
        lines = []
        return validate.validate(say=lines.append, **kwargs), "\n".join(lines)

    def test_the_committed_tables_prompts_and_model_list_pass_every_check(self):
        ok, text = self.run_validate()
        self.assertTrue(ok, text)
        for expected in ("Table of clinic_photo", "Alignment clinic_record / clinic_photo_record: the record", "Alignment published_text_photos / published_text: the case",
                         "Prompts", "Model list", "Knowledge index: no validation or test case in it", "All checks passed."):
            self.assertIn(expected, text)
        self.assertNotIn("FAIL", text)

    def test_a_failed_check_is_shown_with_its_reason_and_fails_the_run(self):
        with mock.patch("src.validate.config_problems", return_value=["fake on published_text/closed_book: missing settings.id"]):
            ok, text = self.run_validate()
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model list", text)
        self.assertIn("       - fake on published_text/closed_book: missing settings.id", text)
        self.assertIn("Fix the lines marked FAIL before any experiment.", text)

    def test_files_that_are_not_on_this_computer_are_skipped_not_failed(self):
        with mock.patch("src.validate.load_dataset") as load, mock.patch("pathlib.Path.is_dir", return_value=False):
            ok, text = self.run_validate(files=True)
        self.assertTrue(ok, text)
        self.assertIn("[SKIP] Files of clinic_photo", text)
        load.assert_not_called()

    def test_files_that_are_here_are_hashed_and_a_changed_one_fails(self):
        with mock.patch("src.validate.load_dataset", side_effect=ValueError("1 changed (test/x.jpg)")), mock.patch("pathlib.Path.is_dir", return_value=True):
            ok, text = self.run_validate(files=True)
        self.assertFalse(ok)
        self.assertIn("[FAIL] Files of clinic_photo", text)

    def test_the_knowledge_index_check_finds_a_test_case_a_changed_case_and_a_foreign_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.csv"
            with (ROOT / "data" / "published_cases" / "dataset.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            train, val, test = (next(r for r in rows if r["split"] == split) for split in ("train", "val", "test"))

            def write(*cases):
                with index.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, ["item_id", "kind", "path", "sha256"], lineterminator="\n")
                    writer.writeheader()
                    writer.writerows({"item_id": f"case:{i}", "kind": "case", "path": f"data/published_cases/files/text/{p}", "sha256": s}
                                     for i, (p, s) in enumerate(cases))
            write((train["path"], train["sha256"]))
            self.assertEqual(validate.knowledge_problems(index), [])
            write((test["path"], test["sha256"]))
            self.assertEqual(validate.knowledge_problems(index), ["case:0: a test case is in the knowledge index (a leak)"])
            write((val["path"], val["sha256"]))
            self.assertEqual(validate.knowledge_problems(index), ["case:0: a val case is in the knowledge index (a leak)"])
            write((train["path"], "0" * 64))
            self.assertEqual(validate.knowledge_problems(index), ["case:0: the index holds another version of the case than the dataset"])
            write(("train/nevus/not-a-case.txt", "0" * 64))
            self.assertEqual(validate.knowledge_problems(index), ["case:0: not a case of the published cases"])

    def test_the_command_line_exits_with_the_result_and_has_one_option(self):
        with mock.patch("src.validate.validate", return_value=False) as check, self.assertRaises(SystemExit) as stop:
            validate.main(["--files"])
        self.assertEqual((stop.exception.code, check.call_args.kwargs), (1, {"files": True}))
        with mock.patch("src.validate.validate", return_value=True), self.assertRaises(SystemExit) as stop:
            validate.main([])
        self.assertEqual(stop.exception.code, 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate.main(["--other"])


if __name__ == "__main__":
    unittest.main()
