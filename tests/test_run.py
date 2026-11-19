"""Runs (src/run.py) on tiny synthetic projects: the run folder, resuming, refusals, the setup check, and each kind of dataset."""

import contextlib
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src import common, datasets, evaluate, models, prompts, run

from tests.fixtures import FakeModel, MultimodalProject, ProbeModel, Project, RecordingModel, RunProject, TextProject, UnpinnedModel

ROOT = Path(__file__).resolve().parents[1]


class RunTests(RunProject):
    def test_test_run_writes_predictions_metrics_and_report(self):
        out = self._run(FakeModel(fail_on=[2]), "test", run_id="t1")
        self.assertEqual(out, self.model_dir / "t1")  # one test run: its report is the result
        preds = self._preds(out)
        self.assertEqual(len(preds), 4)
        self.assertTrue(all(p["prediction"] == "melanoma" for p in preds))
        self.assertEqual(preds[1]["attempts"], 2)
        sent = preds[0]["photo_sent"]  # the file exactly as stored, nothing resized or re-encoded
        self.assertEqual(sent["sha256"], hashlib.sha256((self.image_dir / preds[0]["path"]).read_bytes()).hexdigest())
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["main"]["n"], metrics["main"]["correct"]), (4, 2))
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["status"], card["model_versions_returned"], card["run_id"]), ("complete", ["fake-1-001"], "t1"))
        self.assertEqual((card["n_photos"], card["photos_sha256"]),  # the names of the photo dataset: its item is a photo
                         (4, datasets.items_hash([i for i in datasets.load_dataset() if i.split == "test"])))
        self.assertEqual(card["dataset_table"]["sha256"], hashlib.sha256(self.table.read_bytes()).hexdigest())
        self.assertEqual({p["patient_id"] for p in preds}, {"patient-melanoma", "patient-nevus"})
        self.assertEqual((metrics["main"]["groups"], metrics["main"]["interval_method"]), (2, "bootstrap over patients"))
        self.assertEqual(list(metrics), ["answer_format", "main"])  # no class of the image dataset is set apart
        self.assertEqual((metrics["answer_format"], card["answer_format"]), (prompts.ANSWER_FORMAT, prompts.ANSWER_FORMAT))
        self.assertEqual(preds[0]["differential"], ["melanoma", "nevus"])  # two classes: a differential of two
        self.assertEqual((preds[0]["evidence_sufficient"], preds[0]["evidence"], preds[0]["invalid"]), ("yes", [], None))
        self.assertEqual(metrics["main"]["top3_balanced_accuracy"], 1.0)
        detection = metrics["main"]["detection"]  # the stand-in always says "yes": no wrong answer is marked
        self.assertEqual((detection["sensitivity"], detection["specificity"], detection["coverage"], detection["selective_risk"]),
                         (0.0, 1.0, 1.0, 0.5))
        text = (out / "report.md").read_text(encoding="utf-8")
        for expected in ("| **Balanced accuracy (main score)** | **50.0%** |", "(bootstrap over patients)", "| Macro-F1 |",
                         "| Top-3 balanced accuracy (the true class in the differential) | 100.0% |",
                         "| Invalid answers (counted as wrong) | 0.0% (0/4) |",
                         "### Misdiagnosis detection and selective prediction",
                         "| Sensitivity: wrong diagnoses marked \"not sufficient\" | 0.0% (0/2) |",
                         "| Specificity: right diagnoses not marked | 100.0% (2/2) |",
                         "| Coverage: questions answered with \"sufficient\" | 100.0% (4/4) |",
                         "| Confident misdiagnosis (selective risk): wrong among them | 50.0% (2/4) |",
                         "### Per class", "### Confusion matrix", "(2 patients for 4 photos)", "sent exactly as stored",
                         "docs/method.md", f"answer format `{prompts.ANSWER_FORMAT}`"):
            self.assertIn(expected, text)
        for gone in ("| Accuracy |", "Hallucination", "Matthews", "kappa", "malignancy", "first phase"):
            self.assertNotIn(gone, text)
        self.assertNotIn("## Without", text)
        self.assertNotIn("Abstention rate", text)  # a forced-choice run has no abstention to report
        again = FakeModel()  # the same command again: nothing is sent, nothing is overwritten
        self.assertEqual((self._run(again, "test", run_id="t1"), again.calls), (out, 0))
        self.assertEqual(sorted(p.name for p in self.model_dir.iterdir()), ["t1"])  # no .partial folder left

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_a_finished_test_run_does_not_stop_the_next_one_but_changed_code_does(self):
        """With the real Git check (no stand-in): the results folder of the first test run is new and uncommitted, and the second
        test run, of another model, must still start; a changed or new file outside results/ must still stop it."""
        self.clean.stop()  # RunTests.setUp replaced _git with a clean stand-in: use the real one here
        def git(*args):
            return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=self.base, capture_output=True, check=True)

        git("init", "-q")
        (self.base / ".gitignore").write_text("*.partial/\nresults/**/val_*/\n", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "code and data")
        self.assertEqual(run._git()["uncommitted_changes"], False)
        first = self._run(FakeModel(), "test")
        self.assertTrue(first.is_dir())
        self.assertEqual(run._git()["uncommitted_changes"], False)  # only a results folder is new
        listed = self.config.read_text(encoding="utf-8")
        self.config.write_text(listed + listed.replace('name = "fake"', 'name = "other"'), encoding="utf-8")
        git("add", "configs/models.toml")
        git("commit", "-qm", "another model")
        with mock.patch("src.run.load", return_value=FakeModel()), mock.patch("src.run.time.sleep"):
            second = run.run("other", self.dataset, "test", results=self.results)
        card = json.loads((second / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual(card["git"]["uncommitted_changes"], False)
        self.assertEqual(len(card["git"]["commit"]), 40)
        self.prompt.write_text("changed {labels}", encoding="utf-8")  # tracked code or settings changed
        self.assertEqual(run._git()["uncommitted_changes"], True)
        git("checkout", "--", self.prompt.relative_to(self.base).as_posix())
        (self.base / "new_code.py").write_text("x = 1\n", encoding="utf-8")  # a new file outside results/
        self.assertEqual(run._git()["uncommitted_changes"], True)

    def test_the_report_names_the_revision_a_model_says_it_served(self):
        class Revised(FakeModel):
            def describe(self):
                return super().describe() | {"model_revision": "9.9-test-01-2026"}

        revised = (self._run(Revised(), "val", run_id="v1") / "report.md").read_text(encoding="utf-8")
        plain = (self._run(FakeModel(), "val", run_id="v2") / "report.md").read_text(encoding="utf-8")
        self.assertIn("- Model requested: `fake-1` (revision `9.9-test-01-2026`); versions returned by the API: `fake-1-001`", revised)
        self.assertIn("- Model requested: `fake-1`; versions returned by the API: `fake-1-001`", plain)

    def test_only_the_validation_and_test_splits_are_run(self):
        with self.assertRaisesRegex(ValueError, "split must be one of"):
            self._run(FakeModel(), "train")

    def test_the_run_card_records_the_versions_of_the_packages_the_adapter_names(self):
        model = FakeModel()
        model.packages = ("pillow", "no-such-package")
        self._run(model, "val", run_id="v1", limit=1)
        card = json.loads((self.model_dir / "v1" / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(card["versions"]), ["no-such-package", "pillow", "python"])
        self.assertIsNone(card["versions"]["no-such-package"])
        self.assertRegex(card["versions"]["pillow"], r"^\d+\.\d+")

    def test_test_split_runs_once_and_refuses_limit(self):
        self._run(FakeModel(), "test", run_id="t1")
        with self.assertRaises(SystemExit):
            self._run(FakeModel(), "test", run_id="t2")
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t3", limit=1)

    def test_a_stopped_run_keeps_its_answers_apart_and_resumes(self):
        with mock.patch("src.run.MAX_ATTEMPTS", 2):
            partial = self._run(FakeModel(fail_from=3), "val", run_id="v1")
        self.assertEqual(partial, self.model_dir / "v1.partial")  # Git ignores *.partial
        self.assertFalse((self.model_dir / "v1").exists())
        self.assertEqual(len(self._preds(partial)), 2)
        self.assertEqual(json.loads((partial / "run_card.json").read_text(encoding="utf-8"))["status"], "running")
        second = FakeModel()
        out = self._run(second, "val", run_id="v1")
        self.assertEqual(second.calls, 2)  # only the two unanswered images
        self.assertEqual(out, self.model_dir / "v1")
        self.assertEqual(len(self._preds(out)), 4)
        self.assertEqual([p.name for p in self.model_dir.iterdir()], ["v1"])
        stops = json.loads((out / "run_card.json").read_text(encoding="utf-8"))["interruptions"]  # the stop stays on record
        self.assertEqual([(s["answered"], s["kind"]) for s in stops], [(2, "TransientError")])
        self.assertIn("Interruptions before the run finished: 1", (out / "report.md").read_text(encoding="utf-8"))

    def test_a_run_whose_first_answers_are_all_invalid_stops_early_keeps_them_and_can_go_on(self):
        class NotJson(FakeModel):
            def predict(self, images, prompt):
                return super().predict(images, prompt) | {"text": "I cannot say."}

        with mock.patch("src.run.EARLY_CHECK", 2):
            with self.assertRaisesRegex(SystemExit, "the first 2 answers are all invalid .*not one JSON object.*run the same command again"):
                self._run(NotJson(), "val", run_id="v1")
            partial = self.model_dir / "v1.partial"
            self.assertEqual(len(self._preds(partial)), 2)  # what was asked is kept, nothing more is asked
            interruption = json.loads((partial / "run_card.json").read_text(encoding="utf-8"))["interruptions"]
            self.assertEqual([(i["answered"], i["kind"]) for i in interruption], [(2, "FatalError")])
            again = NotJson()
            out = self._run(again, "val", run_id="v1")  # the researcher decided to go on: the check is not made a second time
        self.assertEqual((again.calls, len(self._preds(out))), (2, 4))

    def test_one_valid_answer_among_the_first_ones_does_not_stop_the_run(self):
        class FirstInvalid(FakeModel):
            def predict(self, images, prompt):
                answer = super().predict(images, prompt)
                return answer | {"text": "no"} if self.calls == 1 else answer

        with mock.patch("src.run.EARLY_CHECK", 2):
            out = self._run(FirstInvalid(), "val", run_id="v1")
        self.assertEqual(len(self._preds(out)), 4)

    def test_a_run_whose_answer_file_is_inconsistent_is_not_marked_finished(self):
        with mock.patch("src.run.verify_answers", return_value=["test/a/1.jpg is answered 2 times"]):
            with self.assertRaisesRegex(SystemExit, "not consistent, so the run is not marked finished: test/a/1.jpg is answered 2 times"):
                self._run(FakeModel(), "val", run_id="v1")
        self.assertTrue((self.model_dir / "v1.partial").is_dir())
        self.assertFalse((self.model_dir / "v1").exists())
        self.assertEqual(json.loads((self.model_dir / "v1.partial" / "run_card.json").read_text(encoding="utf-8"))["status"], "running")

    def test_a_run_that_answered_nothing_leaves_no_folder(self):
        with mock.patch("src.run.MAX_ATTEMPTS", 2):
            self._run(FakeModel(fail_from=1), "val", run_id="v1")  # overloaded from the first image
        self.assertEqual(list(self.model_dir.iterdir()) if self.model_dir.exists() else [], [])
        with self.assertRaises(SystemExit) as raised:  # a refused setting, from the first image
            self._run(FakeModel(fatal_from=1), "val", run_id="v1")
        self.assertIn("STOPPED, 0 of 4 answered", str(raised.exception))
        self.assertIn("HTTP 404", str(raised.exception))
        self.assertEqual(list(self.model_dir.iterdir()) if self.model_dir.exists() else [], [])

    def test_a_daily_quota_stops_at_once_keeps_the_answers_and_continues_later(self):
        model = FakeModel(quota_from=3)
        partial = self._run(model, "val", run_id="v1")
        self.assertEqual((partial, model.calls), (self.model_dir / "v1.partial", 3))  # no retries: waiting minutes cannot help
        self.assertEqual(len(self._preds(partial)), 2)
        self.assertFalse([c for c in self.sleep.call_args_list if c.args and c.args[0] >= 10])  # and no waiting either
        later = FakeModel()
        self.assertEqual(self._run(later, "val", run_id="v1"), self.model_dir / "v1")
        self.assertEqual(later.calls, 2)
        self._run(FakeModel(quota_from=1), "val", run_id="v2")  # quota used up before the first answer: nothing is left behind
        self.assertEqual([p.name for p in self.model_dir.iterdir()], ["v1"])

    def test_a_run_stopped_by_a_quota_and_resumed_says_whether_the_model_changed_meanwhile(self):
        def revised(revision, **kwargs):
            class Revised(FakeModel):
                def describe(self):
                    return super().describe() | {"model_revision": revision}

            return Revised(**kwargs)

        for second, changed in (("r1", False), ("r2", True)):
            with self.subTest(second):
                self._run(revised("r1", quota_from=3), "val", run_id=f"v_{second}")  # day one: the quota stops it
                final = self._run(revised(second), "val", run_id=f"v_{second}")  # day two: it continues
                card = json.loads((final / "run_card.json").read_text(encoding="utf-8"))
                self.assertEqual([s["model_revision"] for s in card["sessions"]], ["r1", second])  # one per session
                report = (final / "report.md").read_text(encoding="utf-8")
                self.assertIn("- Model revision at the start of each session: " + ("`r1`, `r2`" if changed else "`r1`"), report)
                self.assertEqual("the model changed while the run was stopped" in report, changed)
        plain = self._run(FakeModel(), "val", run_id="v_plain")  # a model that names no revision: nothing to report
        self.assertNotIn("Model revision at the start", (plain / "report.md").read_text(encoding="utf-8"))

    def test_a_refusal_after_some_answers_keeps_them(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "val", run_id="v1")
        self.assertEqual(len(self._preds(self.model_dir / "v1.partial")), 2)
        card = json.loads((self.model_dir / "v1.partial" / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual([s["kind"] for s in card["interruptions"]], ["FatalError"])

    def test_the_default_folder_follows_from_the_settings(self):
        with mock.patch("src.run.MAX_ATTEMPTS", 1):
            partial = self._run(FakeModel(fail_on=[2]), "val")
        self.assertRegex(partial.name, r"^val_[0-9a-f]{8}\.partial$")
        again = self._run(FakeModel(), "val")  # the same command continues the same run
        self.assertEqual(again, self.model_dir / partial.name.removesuffix(".partial"))
        self.assertEqual(len(self._preds(again)), 4)
        with self.assertRaises(SystemExit):  # finished: the same command does not run it again
            self._run(FakeModel(), "val")
        self.prompt.write_text("Choose:\n{labels}", encoding="utf-8")  # other settings: another folder, no clash
        self.assertNotEqual(self._run(FakeModel(), "val"), again)

    def test_a_changed_prompt_cannot_continue_an_old_explicit_run(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "val", run_id="v1")
        self.prompt.write_text("Choose:\n{labels}", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            self._run(FakeModel(), "val", run_id="v1")
        self.assertIn("different config, prompt or set of files", str(raised.exception))

    def test_an_unfinished_test_run_must_be_continued_not_restarted(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "test", run_id="t1")
        with self.assertRaises(SystemExit) as raised:
            self._run(FakeModel(), "test", run_id="t2")
        self.assertIn("--run-id t1", str(raised.exception))
        self.assertEqual(self._run(FakeModel(), "test", run_id="t1"), self.model_dir / "t1")

    def test_images_are_sent_in_the_seeded_random_order(self):
        original, by_seed = self.config.read_text(encoding="utf-8"), {}
        for seed in (0, 1):  # each seed in a results folder of its own: one model, dataset and approach have one test run
            self.config.write_text(original.replace("order_seed = 0", f"order_seed = {seed}"), encoding="utf-8")
            with mock.patch("src.run.load", return_value=FakeModel()):
                out = run.run("fake", self.dataset, "test", run_id="t1", results=self.results / f"seed{seed}")
            by_seed[seed] = [p["path"] for p in self._preds(out)]
            expected = sorted(i.path for i in datasets.load_dataset() if i.split == "test")
            random.Random(seed).shuffle(expected)
            self.assertEqual(by_seed[seed], expected, f"order_seed = {seed}")
        self.assertNotEqual(by_seed[0], by_seed[1])
        self.assertNotEqual(by_seed[0], sorted(by_seed[0]))  # not simply the sorted order

    def test_a_changed_setting_cannot_add_a_second_test_run(self):
        self._run(FakeModel(), "test")
        self.config.write_text(self.config.read_text(encoding="utf-8").replace('id = "fake-1"', 'id = "fake-1"\nseed = 1'),
                               encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "a finished test run already exists"):
            self._run(FakeModel(), "test")

    def test_a_test_run_needs_a_clean_git_state_but_a_validation_run_does_not(self):
        dirty = {"commit": "a" * 40, "uncommitted_changes": True}
        with mock.patch("src.run._git", return_value=dirty):
            with self.assertRaisesRegex(ValueError, "clean Git state"):
                self._run(FakeModel(), "test", run_id="t1")
            self.assertFalse(self.model_dir.exists())  # nothing was started
            self._run(FakeModel(), "val", run_id="v1")  # allowed, and the run card records the state
        card = json.loads((self.model_dir / "v1" / "run_card.json").read_text(encoding="utf-8"))
        self.assertTrue(card["git"]["uncommitted_changes"])
        with mock.patch("src.run._git", return_value={"commit": None, "uncommitted_changes": None}), \
                contextlib.redirect_stdout(io.StringIO()) as shown:  # without Git the run goes on, with a warning
            self._run(FakeModel(), "test", run_id="t1")
        self.assertIn("WARNING: Git is not available", shown.getvalue())

    def test_a_test_run_of_an_unpinned_model_is_refused_before_anything_is_written(self):
        with self.assertRaisesRegex(SystemExit, "pinned to one version"):
            self._run(UnpinnedModel(), "test", run_id="t1")
        self.assertFalse(self.model_dir.exists())
        self._run(UnpinnedModel(), "val", run_id="v1")  # a validation run may use any version
        self._run(FakeModel(), "test", run_id="t1")
        self.assertTrue((self.model_dir / "t1").exists())

    def test_every_session_and_every_step_is_logged(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "val", run_id="v1")
        out = self._run(FakeModel(), "val", run_id="v1")
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual(len(card["sessions"]), 2)
        log = (out / "run.log").read_text(encoding="utf-8")
        self.assertEqual(log.count("session start: Git commit"), 2)
        for expected in ("STOPPED, 2 of 4 answered.", "[3/4]", "FINISHED: 4 of 4 answered"):
            self.assertIn(expected, log)
        self.assertTrue(all(line[:4].isdigit() for line in log.splitlines()))  # every line starts with a UTC time
        self.assertIn("Sessions (starts and restarts of this run): 2", (out / "report.md").read_text(encoding="utf-8"))

    def test_a_finished_run_says_whether_its_folder_belongs_in_git(self):
        val = (self._run(FakeModel(), "val", run_id="v1") / "run.log").read_text(encoding="utf-8")
        self.assertIn("A validation run stays on this computer", val)
        self.assertNotIn("can be committed", val)
        self.assertIn("Validation run: for tuning the method", (self.model_dir / "v1" / "report.md").read_text(encoding="utf-8"))
        test = (self._run(FakeModel(), "test", run_id="t1") / "run.log").read_text(encoding="utf-8")
        self.assertIn("This folder is complete and can be committed.", test)
        self.assertNotIn("stays on this computer", test)

    def test_a_run_that_spans_two_commits_says_so_in_its_report(self):
        with mock.patch("src.run._git", return_value={"commit": "a" * 40, "uncommitted_changes": False}), \
                self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "val", run_id="v1")
        with mock.patch("src.run._git", return_value={"commit": "b" * 40, "uncommitted_changes": False}):
            out = self._run(FakeModel(), "val", run_id="v1")
        text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("the run spanned more than one commit", text)
        self.assertIn("aaaaaaaaaa, bbbbbbbbbb", text)

    def test_a_run_that_stops_before_its_first_answer_leaves_no_log(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=1), "val", run_id="v1")
        self.assertFalse(self.model_dir.exists() and any(self.model_dir.iterdir()))

    def test_a_stop_in_the_middle_of_a_write_does_not_stop_the_run_from_continuing(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "val", run_id="v1")
        partial = self.model_dir / "v1.partial" / "predictions.jsonl"
        partial.write_text(partial.read_text(encoding="utf-8") + '{"path": "val/mel', encoding="utf-8")  # cut off mid-line
        later = FakeModel()
        out = self._run(later, "val", run_id="v1")
        self.assertEqual((out, later.calls), (self.model_dir / "v1", 2))  # the cut-off image is answered again
        self.assertEqual(len(self._preds(out)), 4)
        # a complete last line that lacks only its end of line is kept, and the next answer starts on a new line
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "val", run_id="v2")
        partial = self.model_dir / "v2.partial" / "predictions.jsonl"
        partial.write_text(partial.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
        self.assertEqual(len(self._preds(self._run(FakeModel(), "val", run_id="v2"))), 4)
        # a line cut off inside a multi-byte character ("é" is two bytes) is also removed, and the report reads the partial run
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "val", run_id="v4")
        partial = self.model_dir / "v4.partial" / "predictions.jsonl"
        partial.write_bytes(partial.read_bytes() + '{"path": "x", "raw_text": "café"}'.encode("utf-8")[:-3])
        self.assertEqual(len(common.read_jsonl(partial)), 2)  # the cut-off line is ignored, nothing crashes
        evaluate.write_report(partial.parent)
        self.assertEqual(len(self._preds(self._run(FakeModel(), "val", run_id="v4"))), 4)
        # a damaged line in the middle is an error, never skipped
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "val", run_id="v3")
        partial = self.model_dir / "v3.partial" / "predictions.jsonl"
        first, second = partial.read_text(encoding="utf-8").splitlines()
        partial.write_text(first + "\n{broken\n" + second + "\n", encoding="utf-8")
        with self.assertRaises(ValueError) as raised:
            self._run(FakeModel(), "val", run_id="v3")
        self.assertIn("line 2 is not valid JSON", str(raised.exception))

    def test_files_are_written_through_a_temporary_file(self):
        target = self.results / "x.json"
        self.results.mkdir(parents=True, exist_ok=True)
        common.write_json(target, {"a": 1})
        with mock.patch("src.common.os.replace", side_effect=OSError("stopped")), self.assertRaises(OSError):
            common.write_json(target, {"a": 2})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": 1})  # the old file is whole

    def test_unsafe_names_and_impossible_settings_are_refused_with_a_message(self):
        for bad in ("../escape", "a/b", "a\\b", "", ".hidden", "x" * 81, "a b"):
            with self.assertRaises(ValueError, msg=bad):
                self._run(FakeModel(), "val", run_id=bad)
        self.assertFalse(self.results.exists())  # nothing was written
        text = self.config.read_text(encoding="utf-8")
        for old, new in (("requests_per_minute = 600000", "requests_per_minute = 0"), ("requests_per_minute = 600000", 'requests_per_minute = "5"'),
                         ("order_seed = 0", "order_seed = 1.5"), ("order_seed = 0", ""), ('name = "fake"', 'name = "../fake"'),
                         ('name = "fake"', ""), ('id = "fake-1"', "")):
            self.config.write_text(text.replace(old, new), encoding="utf-8")
            with self.assertRaises(ValueError, msg=new):
                run.load_config("fake", self.dataset)
        self.config.write_text(text + text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "a model name is listed twice"):
            run.load_config("fake", self.dataset)
        self.config.write_text(text, encoding="utf-8")
        for model, dataset, approach in (("other", "clinic_photo", "closed_book"), ("fake", "multimodal", "closed_book"),
                                         ("fake", "clinic_photo", "open_book")):
            with self.assertRaises(ValueError, msg=(model, dataset, approach)):
                run.load_config(model, dataset, approach)
        for limit in (0, -3):
            with self.assertRaises(ValueError):
                self._run(FakeModel(), "val", limit=limit)

    def test_answers_that_did_not_end_normally_are_flagged_in_the_report(self):
        class Blocked(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": None, "finish_reason": "SAFETY"}

        text = (self._run(Blocked(), "val", run_id="v1") / "report.md").read_text(encoding="utf-8")
        self.assertIn("4 answers did not end normally (SAFETY: 4)", text)
        self.assertNotIn("did not end normally", (self._run(FakeModel(), "val", run_id="v2") / "report.md").read_text(encoding="utf-8"))

    def test_a_folder_that_cannot_be_renamed_is_finished_by_running_the_command_again(self):
        real_rename = Path.rename
        with mock.patch.object(Path, "rename", side_effect=PermissionError("in use")), self.assertRaises(SystemExit) as raised:
            self._run(FakeModel(), "test", run_id="t1")
        self.assertIn("could not be renamed", str(raised.exception))
        self.assertTrue((self.model_dir / "t1.partial" / "report.md").exists())
        self.assertFalse((self.model_dir / "t1").exists())
        again = FakeModel()
        with mock.patch.object(Path, "rename", real_rename):
            out = self._run(again, "test", run_id="t1")  # the test-run guard must not mistake the finished answers for a finished run
        self.assertEqual((out, again.calls), (self.model_dir / "t1", 0))
        self.assertEqual(len(self._preds(self.model_dir / "t1")), 4)

    def test_failed_attempts_wait_longer_each_time(self):
        self._run(FakeModel(fail_on=[1, 2, 3]), "val", run_id="v1")
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list if c.args and c.args[0] >= 10][:3], [10, 20, 40])

    def test_limit_takes_the_first_images_of_the_seeded_order(self):
        preds = self._preds(self._run(FakeModel(), "val", run_id="v2", limit=3))
        self.assertEqual(len(preds), 3)
        self.assertEqual({p["label"] for p in preds}, {"melanoma", "nevus"})

    def test_images_that_differ_from_the_dataset_table_stop_the_run_before_anything_is_written(self):
        Image.new("RGB", (60, 45), (1, 2, 3)).save(self.image_dir / "test" / "nevus" / "PAT_0000013.jpg")
        with self.assertRaises(ValueError) as raised:
            self._run(FakeModel(), "test", run_id="t1")
        self.assertIn("1 changed", str(raised.exception))
        self.assertFalse(self.model_dir.exists())
        (self.image_dir / "test" / "nevus" / "PAT_0000013.jpg").unlink()
        (self.image_dir / "test" / "nevus" / "broken.jpg").write_bytes(b"not a jpeg")
        with self.assertRaisesRegex(ValueError, "1 missing, 1 not in the table"):
            self._run(FakeModel(), "test", run_id="t1")

    def test_answers_cut_off_at_the_output_limit_make_the_results_invalid(self):
        class Truncated(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": '{"differential": ["mela', "finish_reason": "MAX_TOKENS"}

        self._run(Truncated(), "test", run_id="t1")
        text = (self.model_dir / "t1" / "report.md").read_text(encoding="utf-8")
        self.assertIn("4 answers were cut off at the output limit", text)
        self.assertIn("Results are not valid until it is raised", text)
        self.assertIn("| Invalid answers (counted as wrong) | 100.0% (4/4) |", text)
        self.assertNotIn("cut off", (self._run(FakeModel(), "val", run_id="v1") / "report.md").read_text(encoding="utf-8"))

    def test_a_complete_answer_that_reached_the_limit_does_not_make_the_results_invalid(self):
        class Exact(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"finish_reason": "MAX_TOKENS"}

        self._run(Exact(), "test", run_id="t1")
        text = (self.model_dir / "t1" / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("Results are not valid", text)
        self.assertIn("4 answers reached the output limit but were complete and valid all the same", text)

    def test_a_missing_key_stops_the_run_before_anything_is_written(self):
        with mock.patch("src.run.load", side_effect=ValueError("no key")), self.assertRaises(ValueError):
            run.run("fake", self.dataset, "test", results=self.results)
        self.assertFalse(self.results.exists())

    def test_the_command_line_turns_expected_problems_into_messages(self):
        with self.assertRaises(SystemExit) as raised:
            run.main(["run", "--model", "missing", "--dataset", "clinic_photo", "--split", "val"])
        self.assertTrue(str(raised.exception).startswith("ERROR: "))
        self.table.unlink()  # the dataset is not prepared yet
        with self.assertRaises(SystemExit) as raised:
            run.main(["run", "--model", "fake", "--dataset", "clinic_photo", "--split", "test"])
        self.assertIn("does not exist yet", str(raised.exception))


class CheckTests(Project):
    def setUp(self):
        super().setUp()
        patcher = mock.patch("src.run.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _check(self, model=None, load_error=None):
        lines = []
        with mock.patch("src.run.load", return_value=model, side_effect=load_error):
            ok = run.check_setup(run.load_config("fake", self.dataset), say=lines.append)
        return ok, "\n".join(lines)

    def test_everything_in_order(self):
        model = ProbeModel()
        ok, text = self._check(model)
        self.assertTrue(ok, text)
        for expected in ("[  OK] Settings: model fake (configs/models.toml), dataset clinic_photo, approach closed_book", "[  OK] Prompt file: prompts/clinic_photo/closed_book.txt",
                         "[  OK] Prepared dataset: 14 photos (train 6, val 4, test 4), 2 classes",
                         "[  OK] Model adapter and access", "[  OK] Model fake-1", "All checks passed"):
            self.assertIn(expected, text)
        image, mime, _ = model.images[0]  # only a blank synthetic image is ever sent
        self.assertEqual((mime, Image.open(io.BytesIO(image)).size), ("image/png", (64, 64)))

    def test_an_adapter_can_say_how_it_is_loaded_and_what_the_probe_costs(self):
        model = ProbeModel()
        model.loaded_note, model.probe_note = "model files found and loaded", "it runs on this computer, so it uses no quota"
        ok, text = self._check(model)
        self.assertTrue(ok, text)
        self.assertIn("[  OK] Model adapter and access: fake, model files found and loaded", text)
        self.assertIn("it runs on this computer, so it uses no quota) ...", text)
        self.assertNotIn("daily quota)", text)
        ok, text = self._check(ProbeModel())  # an adapter without notes keeps the wording for hosted models
        self.assertIn("fake, key found (not shown)", text)
        self.assertIn("it counts as one request against the daily quota) ...", text)

    def test_an_unavailable_model_is_reported(self):
        ok, text = self._check(ProbeModel(error=models.FatalError("Provider refused the request (HTTP 404 NOT_FOUND): gone", 404)))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model fake-1: Provider refused the request (HTTP 404", text)
        self.assertIn("Fix the lines marked FAIL", text)

    def test_a_temporary_overload_is_not_blamed_on_the_setup(self):
        model = ProbeModel(error=models.TransientError("HTTP 503 UNAVAILABLE: high demand"))
        ok, text = self._check(model)
        self.assertFalse(ok)
        self.assertEqual(model.calls, run.PROBE_ATTEMPTS)
        self.assertIn("temporary problem, not a setting", text)

    def test_a_used_up_daily_quota_fails_at_once_and_spends_nothing_more(self):
        model = ProbeModel(error=models.QuotaExhausted("Daily quota used up for fake-1 (limit 20 requests per day)"))
        ok, text = self._check(model)
        self.assertFalse(ok)
        self.assertEqual(model.calls, 1)  # no retries
        self.assertIn("[FAIL] Model fake-1: Daily quota used up for fake-1", text)
        self.assertIn("a used-up daily quota needs the reset, a paid key or another model", text)

    def test_a_cut_off_answer_fails(self):
        answer = {"text": "o", "finish_reason": "MAX_TOKENS", "model_version": "v", "response_id": "r", "usage": {"output_tokens": 60}}
        ok, text = self._check(ProbeModel(answer))
        self.assertFalse(ok)
        self.assertIn("raise the output limit, max_tokens or max_output_tokens", text)

    def test_a_missing_key_fails_and_still_says_what_to_do_next(self):
        ok, text = self._check(load_error=ValueError("no API key found"))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model adapter and access: no API key found", text)
        self.assertTrue(text.endswith("check --model <id> tries one)."), text)  # the closing advice is always shown

    def test_a_dataset_that_is_not_prepared_or_differs_from_its_table_fails_but_the_model_is_still_checked(self):
        next((self.image_dir / "test" / "nevus").iterdir()).unlink()
        model = ProbeModel()
        ok, text = self._check(model)
        self.assertFalse(ok)
        self.assertIn("[FAIL] Prepared dataset: the photos in data/clinic_patients/files/photos differ from data/clinic_patients/dataset.csv", text)
        self.assertEqual(model.calls, 1)  # the model access is still tested, so both problems show at once
        self.table.unlink()
        ok, text = self._check(ProbeModel())
        self.assertIn("[FAIL] Prepared dataset: data/clinic_patients/dataset.csv not found", text)
        self.assertIn("[  OK] Model fake-1", text)

    def test_a_broken_prompt_file_fails(self):
        self.prompt.write_text("no placeholder", encoding="utf-8")
        ok, text = self._check(ProbeModel())
        self.assertFalse(ok)
        self.assertIn("[FAIL] Prompt file", text)

    def test_the_check_can_try_another_model_id_without_changing_the_model_list(self):
        seen = []

        def load(provider, settings):
            seen.append(settings["id"])
            return ProbeModel()

        before = self.config.read_text(encoding="utf-8")
        with mock.patch("src.run.load", side_effect=load), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            run.main(["check", "--model", "fake", "--dataset", "clinic_photo", "--try-id", "other-model"])
        self.assertEqual((seen, raised.exception.code), (["other-model"], 0))
        self.assertEqual(self.config.read_text(encoding="utf-8"), before)  # the file itself is unchanged


class TextDatasetTests(TextProject):
    def _run(self, model, *args, **kwargs):
        with mock.patch("src.run.load", return_value=model):
            return run.run("fake", self.dataset, *args, results=self.results, **kwargs)

    def test_the_cases_load_with_their_article_as_their_group_and_only_text_files_are_accepted(self):
        items = datasets.load_dataset(*datasets.dataset_paths("published_text"), "published_text")
        self.assertEqual((len(items), datasets.classes(items)), (16, ["actinic keratosis", "melanoma", "nevus"]))
        self.assertEqual(sorted(i.group for i in items if i.split == "test" and i.label == "melanoma"), ["203", "203", "204"])
        with self.assertRaisesRegex(ValueError, "is not a JPEG or PNG file"):
            datasets.read_table(self.table, "clinic_photo")  # a text table is never read as photos
        with self.table.open("a", encoding="utf-8") as f:
            f.write(f"train,nevus,train/nevus/photo.jpg,{'0' * 64},x,9,PMC-Patients\n")
        with self.assertRaisesRegex(ValueError, "is not a .txt file"):
            datasets.read_table(self.table, "published_text")

    def test_a_case_is_sent_exactly_as_stored_and_an_unreadable_one_stops_the_run(self):
        item = next(i for i in datasets.load_dataset(*datasets.dataset_paths("published_text"), "published_text") if i.split == "test")
        text, sent = datasets.read_case(item, self.case_dir)
        self.assertEqual(text, self.texts[item.path])
        self.assertEqual((sent["sha256"], sent["words"]), (item.sha256, len(self.texts[item.path].split())))
        for content, message in ((b"caf\xe9", "not UTF-8 text"), (b" \n", "is empty")):
            with self.subTest(message):
                (self.case_dir / item.path).write_bytes(content)
                with self.assertRaisesRegex(ValueError, message):
                    datasets.read_case(item, self.case_dir)

    def test_each_case_goes_into_the_prompt_word_for_word_and_no_image_is_sent(self):
        model = RecordingModel()
        out = self._run(model, "test", run_id="t1")
        labels = ["actinic keratosis", "melanoma", "nevus"]
        prompt = "A case.\nCase: {case}\nPick one:\n- actinic keratosis\n- melanoma\n- nevus\nAnswer with the name only."
        preds = [json.loads(x) for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(model.requests), 8)
        for (image, mime, sent), pred in zip(model.requests, preds):
            self.assertEqual((image, mime), (None, None))
            self.assertEqual(sent, prompt.replace("{case}", self.texts[pred["path"]]))
            self.assertEqual(pred["case_sent"]["sha256"], hashlib.sha256(self.texts[pred["path"]].encode("utf-8")).hexdigest())
            self.assertEqual(pred["pmid"], pred["path"].split("/")[2].split("-")[0])
            self.assertNotIn("photo_sent", pred)
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["labels"], card["n_cases"], card["group_column"], card["prompt"]), (labels, 8, "pmid", prompt))
        self.assertEqual(card["cases_sha256"], datasets.items_hash([i for i in datasets.load_dataset(*datasets.dataset_paths("published_text"), "published_text")
                                                                    if i.split == "test"]))

    def test_nothing_of_a_case_is_written_into_the_run_folder_that_goes_into_git(self):
        """The case texts have research-only licences and stay on this computer; a test run's folder is committed."""
        out = self._run(FakeModel(), "test", run_id="t1")
        written = {f.name: f.read_text(encoding="utf-8") for f in out.iterdir()}
        self.assertEqual(sorted(written), ["metrics.json", "predictions.jsonl", "report.md", "run.log", "run_card.json"])
        for text in self.texts.values():
            for piece in (text, "café au lait", "It grew."):
                for name, content in written.items():
                    self.assertNotIn(piece, content, f"{name} holds a piece of a case text")

    def test_the_report_resamples_articles_and_gives_the_scores_without_actinic_keratosis_beside_the_main_score(self):
        out = self._run(FakeModel(), "test", run_id="t1")  # "melanoma" for every case: right for the 3 melanomas only
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        main, apart = metrics["main"], metrics["without_small_classes"]
        self.assertEqual((main["n"], main["groups"], main["interval_method"]), (8, 7, "bootstrap over articles"))
        self.assertAlmostEqual(main["balanced_accuracy"], 1 / 3)  # recall 0, 1, 0
        self.assertEqual((apart["left_out"], apart["metrics"]["n"], apart["metrics"]["classes_absent"]),
                         (["actinic keratosis"], 6, ["actinic keratosis"]))
        self.assertEqual((apart["metrics"]["balanced_accuracy"], apart["metrics"]["correct"]), (0.5, 3))  # recall 1, 0
        self.assertEqual([(f["label"], f["prediction"]) for f in apart["files"]], [("actinic keratosis", "melanoma")] * 2)
        text = (out / "report.md").read_text(encoding="utf-8")
        for expected in ("| **Balanced accuracy (main score)** | **33.3%** |", "(bootstrap over articles)", "| Macro-F1 |",
                         "Patients described in one article are not independent: the intervals resample whole articles (7 articles "
                         "for 8 cases).", "cases: 8 (hash", "each put into the prompt word for word",
                         "## Without actinic keratosis",
                         "The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet "
                         "it weighs as much as any other class in the balanced accuracy above (one class in 3).",
                         "never instead of it", "answering actinic keratosis for another case still counts as wrong",
                         "| **Balanced accuracy (main score)** | **50.0%** |",
                         "| `test/actinic keratosis/201-9.txt` | actinic keratosis | melanoma |",
                         "| `test/actinic keratosis/202-10.txt` | actinic keratosis | melanoma |"):
            self.assertIn(expected, text)
        self.assertLess(text.index("## Results"), text.index("## Without actinic keratosis"))  # the main score comes first
        self.assertLess(text.index("## Without actinic keratosis"), text.index("## Notes"))
        self.assertNotIn("image", text.split("## Notes")[0].lower().replace("multimodal", ""))

    def test_a_run_whose_cases_hold_no_actinic_keratosis_has_no_second_scoring(self):
        out = self._run(FakeModel(), "val", run_id="v1")
        rows = [json.loads(x) for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        kept = [r for r in rows if r["label"] != "actinic keratosis"]
        (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        card["n_cases"] = len(kept)  # the run is rewritten as one that never had those cases
        (out / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
        evaluate.write_report(out)
        self.assertEqual(list(json.loads((out / "metrics.json").read_text(encoding="utf-8"))), ["answer_format", "main"])
        self.assertNotIn("## Without", (out / "report.md").read_text(encoding="utf-8"))

    def test_the_check_asks_the_model_without_an_image_and_counts_cases(self):
        model, lines = ProbeModel(), []
        with mock.patch("src.run.load", return_value=model):
            ok = run.check_setup(run.load_config("fake", self.dataset), say=lines.append)
        text = "\n".join(lines)
        self.assertTrue(ok, text)
        self.assertIn("[  OK] Prepared dataset: 16 cases (train 5, val 3, test 8), 3 classes", text)
        self.assertIn("one question with no image (nothing from the dataset is sent", text)
        self.assertEqual(model.images, [(None, None, run.PROBE_PROMPT)])
        self.prompt.write_text("Pick one:\n{labels}\n", encoding="utf-8")  # a text prompt without the case
        ok, lines = None, []
        with mock.patch("src.run.load", return_value=ProbeModel()):
            ok = run.check_setup(run.load_config("fake", self.dataset), say=lines.append)
        self.assertFalse(ok)
        self.assertIn("[FAIL] Prompt file: ", "\n".join(lines))


class MultimodalTests(MultimodalProject):
    def _run(self, model, *args, **kwargs):
        with mock.patch("src.run.load", return_value=model):
            return run.run("fake", self.dataset, *args, results=self.results, **kwargs)

    def test_the_questions_are_the_clinic_photos_with_their_records(self):
        self.assertEqual(datasets.dataset_paths("clinic_photo_record"),
                         (Path("data/clinic_patients/dataset.csv"), Path("data/clinic_patients/files/photos")))
        self.assertEqual(datasets.dataset_paths("clinic_photo"), datasets.dataset_paths("clinic_photo_record"))  # one table, two views
        items = datasets.load_dataset(*datasets.dataset_paths("clinic_photo_record"), "clinic_photo_record")
        self.assertEqual(len(items), 14)
        missing = next(i for i in items if i.path == self.MISSING)
        self.assertTrue(missing.facts.endswith("3. The lesion itches: unknown"))
        changed = [datasets.Item(**{**items[0].__dict__, "facts": "1. Age in years: 99\n2. Body site: trunk\n3. The lesion itches: no"}), *items[1:]]
        self.assertNotEqual(datasets.items_hash(changed), datasets.items_hash(items))  # a changed fact is a changed question
        text = self.questions.read_text(encoding="utf-8")
        self.questions.write_text(text.replace('"1. Age in years: 30\n2. Body site: trunk\n3. The lesion itches: no"', '""'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "the facts column is empty \\(a missing fact is written \"unknown\"\\)"):
            datasets.read_table(self.questions, "clinic_photo_record")
        self.questions.write_text(text.replace(",facts\n", "\n", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks the column"):
            datasets.read_table(self.questions, "clinic_photo_record")

    def test_a_photo_record_prompt_needs_the_facts_placeholder_and_no_other_prompt_may_have_it(self):
        for template, dataset in (("Pick:\n{labels}", "clinic_photo_record"), ("Patient: {facts}\n{labels}", "clinic_photo"),
                                  ("Case: {case}\nPatient: {facts}\n{labels}", "published_text")):
            with self.subTest(dataset):
                self.prompt.write_text(template, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "{facts}"):
                    prompts.load_template(self.prompt, dataset)

    def test_each_photo_is_sent_as_stored_with_its_facts_in_the_prompt_and_scored_also_on_complete_facts(self):
        model = RecordingModel()
        out = self._run(model, "test", run_id="t1")
        preds = {p["path"]: p for p in (json.loads(x) for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        self.assertEqual(len(model.requests), 4)
        with self.questions.open(encoding="utf-8", newline="") as f:
            facts = {row["path"]: row["facts"] for row in csv.DictReader(f)}
        for image, mime, prompt in model.requests:
            path = next(p for p in preds if (self.image_dir / p).read_bytes() == image)  # the photo, exactly as stored
            self.assertEqual(mime, "image/jpeg")
            self.assertEqual(prompt, "A photo.\nRecord:\n" + facts[path] + "\nPick one:\n- melanoma\n- nevus\nAnswer with the name only.")
            self.assertEqual(preds[path]["facts"], facts[path])
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["main"]["n"], metrics["all_facts_recorded"]["questions"], metrics["all_facts_recorded"]["metrics"]["n"]),
                         (4, 3, 3))
        report = (out / "report.md").read_text(encoding="utf-8")
        for expected in ("sent exactly as stored, each with its patient's facts in the prompt", "## Questions with all facts recorded",
                         "These scores count only the 3 of 4 questions with every fact recorded", "(bootstrap over patients)"):
            self.assertIn(expected, report)
        self.assertLess(report.index("## Results"), report.index("## Questions with all facts recorded"))

    def test_the_check_sends_a_blank_image_and_counts_the_questions(self):
        model, lines = ProbeModel(), []
        with mock.patch("src.run.load", return_value=model):
            ok = run.check_setup(run.load_config("fake", self.dataset), say=lines.append)
        self.assertTrue(ok, "\n".join(lines))
        self.assertIn("[  OK] Prepared dataset: 14 photos (train 6, val 4, test 4), 2 classes", "\n".join(lines))
        self.assertEqual(model.images[0][1:], ("image/png", run.PROBE_PROMPT))


if __name__ == "__main__":
    unittest.main()
