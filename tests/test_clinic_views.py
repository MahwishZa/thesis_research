"""The three views of the clinic patients cohort end to end: the invented PAD-UFES-20 download is prepared by every step of
src/build/clinic_patients, then loaded, asked and compared through the code a real run uses (src/datasets.py, src/run.py,
src/evaluate.py), with a stand-in model."""

import contextlib
import dataclasses
import io
import json
import os
import shutil
from unittest import mock

from src import datasets, evaluate, prompts, run
from src.build.clinic_patients import prepare
from src.build.clinic_patients.table import CLASS_NAMES
from tests.build.clinic_patients.project import Cohort
from tests.fixtures import MODEL_LIST, RecordingModel, ROOT

VIEWS = ("clinic_photo", "clinic_record", "clinic_photo_record")


def question(path: str) -> str:
    return path.rsplit(".", 1)[0]


class ClinicViews(Cohort):
    def setUp(self):
        super().setUp()
        previous = os.getcwd()
        os.chdir(self.base)  # data/, configs/ and prompts/ point into the invented project
        self.addCleanup(os.chdir, previous)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        (self.base / "configs").mkdir()
        (self.base / "configs" / "models.toml").write_text(MODEL_LIST, encoding="utf-8")
        for view in VIEWS:  # the prompts of the repository, unchanged
            shutil.copytree(ROOT / "prompts" / view, self.base / "prompts" / view)
        for name, value in (("time.sleep", None), ("_git", {"commit": "c" * 40, "uncommitted_changes": False})):
            patcher = mock.patch(f"src.run.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        prepare.main(["all"])
        self.results = self.base / "results"

    def n_test(self):
        """How many questions the test split holds, from the table step split wrote."""
        n = sum(r["split"] == "test" for r in self.table("split.csv"))
        self.assertGreater(n, 0)
        return n

    def load(self, view):
        return datasets.load_dataset(*datasets.dataset_paths(view), view)

    def ask(self, view, split="test", model=None):
        model = model or RecordingModel()
        with mock.patch("src.run.load", return_value=model):
            folder = run.run("fake", view, split, results=self.results)
        return folder, model

    def test_the_three_views_load_and_hold_the_same_questions(self):
        loaded = {view: self.load(view) for view in VIEWS}
        self.assertEqual({len(items) for items in loaded.values()}, {121})
        keys = {view: {question(i.path): (i.split, i.label, i.group) for i in items} for view, items in loaded.items()}
        self.assertEqual(keys["clinic_photo"], keys["clinic_record"])
        self.assertEqual(keys["clinic_photo"], keys["clinic_photo_record"])
        self.assertEqual({i.path.rsplit(".", 1)[1] for i in loaded["clinic_photo"]}, {"jpg"})
        self.assertEqual({i.path.rsplit(".", 1)[1] for i in loaded["clinic_record"]}, {"txt"})
        for view, items in loaded.items():
            self.assertEqual(datasets.classes(items), sorted(CLASS_NAMES.values()), view)  # the six class names, never the codes
        self.assertTrue(all(i.group.startswith("P") for i in loaded["clinic_photo"]))  # the group is the patient
        # a photo + record question is the photo with its record text, which is the record view's file, word for word
        records = {question(i.path): i for i in loaded["clinic_record"]}
        for item in loaded["clinic_photo_record"]:
            text = (self.data / "files" / "records" / records[question(item.path)].path).read_text(encoding="utf-8")
            self.assertEqual(item.facts, text)
        self.assertEqual({i.facts for i in loaded["clinic_photo"]}, {""})  # the photo alone carries no facts

    def test_identical_records_are_allowed_in_the_record_view_only(self):
        records = self.load("clinic_record")
        self.assertLess(len({i.sha256 for i in records}), len(records))  # two photos of one visit share one record
        strict = dataclasses.replace(datasets.DATASETS["clinic_record"], unique_content=True)
        with mock.patch.dict(datasets.DATASETS, {"clinic_record": strict}):
            with self.assertRaisesRegex(ValueError, "the same record is also on line"):
                self.load("clinic_record")
        photos = self.load("clinic_photo")
        self.assertEqual(len({i.sha256 for i in photos}), len(photos))  # no photo is in the table twice

    def test_each_view_sends_what_it_says_and_the_record_is_the_same_words_in_two_of_them(self):
        sent = {}
        for view in VIEWS:
            folder, model = self.ask(view)
            sent[view] = (folder, model.requests)
        counts = {len(requests) for _, requests in sent.values()}
        self.assertEqual(len(counts), 1)
        photo, record, both = (sent[v][1] for v in VIEWS)
        self.assertTrue(all(image is not None and mime == "image/jpeg" for image, mime, _ in photo))  # a photo, as stored
        self.assertTrue(all(image is None for image, _, _ in record))  # the record view sends no photo
        self.assertTrue(all(image is not None for image, _, _ in both))
        template = prompts.load_template(run.prompt_file("clinic_photo_record", "closed_book"), "clinic_photo_record")
        for item in self.load("clinic_photo_record"):
            if item.split != "test":
                continue
            stored = (self.data / "files" / "photos" / item.path).read_bytes()
            prompt = next(p for image, _, p in both if image == stored)
            self.assertIn(item.facts, prompt)  # the record, word for word
            self.assertEqual(sum(p.count(item.facts) for image, _, p in photo if image == stored), 0)  # not in the photo view
            self.assertEqual(prompt.count("1. Age in years:"), 1)
        self.assertTrue(all("1. Age in years:" not in p for _, _, p in photo))
        self.assertTrue(all(p.count("1. Age in years:") == 1 and "There is no photograph" in p for _, _, p in record))
        self.assertIn("{facts}", template)
        for view, (folder, _) in sent.items():
            card = json.loads((folder / "run_card.json").read_text(encoding="utf-8"))
            self.assertEqual((card["status"], card["labels"]), ("complete", sorted(CLASS_NAMES.values())), view)
            self.assertEqual(card[f"n_{datasets.DATASETS[view].item}s"], self.n_test())

    def test_the_three_runs_are_compared_question_by_question_whatever_the_file_ending(self):
        folders = {view: self.ask(view)[0] for view in VIEWS}
        for a, b in (("clinic_photo", "clinic_record"), ("clinic_photo", "clinic_photo_record"), ("clinic_record", "clinic_photo_record")):
            found = evaluate.compare(folders[a], folders[b])
            self.assertEqual((found["n"], found["only_a"], found["only_b"]), (self.n_test(), 0, 0), (a, b))  # one stand-in answer for every view
            self.assertEqual(found["dataset"], f"{a} and {b}")
            self.assertIn(f"{a} and {b} dataset, test split, {self.n_test()} questions", evaluate.comparison_text(found))
        report = (folders["clinic_photo_record"] / "report.md").read_text(encoding="utf-8")
        self.assertIn("each with its patient's facts in the prompt", report)
        self.assertIn("## Questions with all facts recorded", report)
        self.assertIn("(bootstrap over patients)", report)

    def test_the_scores_on_complete_records_leave_out_the_questions_with_an_unknown(self):
        test_ids = [r["img_id"] for r in self.table("split.csv") if r["split"] == "test"]
        for row in self.rows:
            if row["img_id"] in test_ids[:2]:
                row["itch"], row["hurt"] = "UNK", "UNK"  # two questions, each with two gaps
        self.write()
        prepare.main(["all"])
        folder, _ = self.ask("clinic_photo_record")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["main"]["n"], metrics["all_facts_recorded"]["questions"]), (self.n_test(), self.n_test() - 2))
        report = (folder / "report.md").read_text(encoding="utf-8")
        self.assertIn("A missing fact (\"unknown\") can itself point to the class", report)
        self.assertIn(f"count only the {self.n_test() - 2} of {self.n_test()} questions with every fact recorded", report)

    def test_a_changed_record_file_or_photo_is_found_before_anything_is_sent(self):
        record = next((self.data / "files" / "records").rglob("*.txt"))
        record.write_bytes(record.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "1 changed"):
            self.load("clinic_record")
        photo = next((self.data / "files" / "photos").rglob("*.jpg"))
        photo.write_bytes(photo.read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "1 changed"):
            self.load("clinic_photo")
        shutil.rmtree(self.data / "files" / "records")
        with self.assertRaisesRegex(ValueError, "121 missing"):
            self.load("clinic_record")
