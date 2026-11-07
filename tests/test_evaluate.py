import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import evaluate, models
from tests.support import make_project, write_corrections


class FakeModel:
    """Says "melanoma" for every image. Calls are counted, including retries.

    ``fail_on``: call numbers that fail once with a transient error; ``fail_from``: every call from this number on fails
    transiently; ``fatal_from``: every call from this number on fails with an error that retrying cannot fix;
    ``quota_from``: every call from this number on finds the daily quota used up.
    """

    def __init__(self, fail_on=(), fail_from=None, fatal_from=None, quota_from=None):
        self.calls, self.fail_on, self.fail_from, self.fatal_from, self.quota_from = 0, set(fail_on), fail_from, fatal_from, quota_from

    def describe(self):
        return {"model_id": "fake-1", "generation_config": {}}

    def predict(self, image, mime_type, prompt):
        self.calls += 1
        if self.quota_from and self.calls >= self.quota_from:
            raise models.QuotaExhausted("Daily quota used up for fake-1 (limit 20 requests per day on the free tier)")
        if self.fatal_from and self.calls >= self.fatal_from:
            raise models.FatalError("Provider refused the request (HTTP 404 NOT_FOUND): gone", 404)
        if self.calls in self.fail_on or (self.fail_from and self.calls >= self.fail_from):
            raise models.TransientError("rate limited")
        return {"text": "Melanoma.", "finish_reason": "STOP", "model_version": "fake-1-001", "response_id": str(self.calls),
                "usage": {}}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = make_project(Path(self.tmp.name))
        self.config, self.results = self.project.config, self.project.results
        self.model_dir = self.results / "fake"
        patcher = mock.patch("src.evaluate.time.sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, model, *args, **kwargs):
        with mock.patch("src.evaluate.models.load", return_value=model):
            return evaluate.run(str(self.config), *args, results=self.results, **kwargs)

    def _preds(self, folder):
        return [json.loads(x) for x in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_test_run_writes_predictions_metrics_and_report(self):
        out = self._run(FakeModel(fail_on=[2]), "test", run_id="t1")
        self.assertEqual(out, self.model_dir / "t1_r1")  # one test run (test_runs = 1): its report is the result
        preds = self._preds(out)
        self.assertEqual(len(preds), 4)
        self.assertTrue(all(p["prediction"] == "melanoma" for p in preds))
        self.assertEqual(preds[1]["attempts"], 2)
        self.assertEqual((preds[0]["image_sent"]["resized"], preds[0]["image_sent"]["width"]), (True, 50))  # 60x45 > 50
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["folder_labels"]["n"], metrics["folder_labels"]["correct"]), (4, 2))
        self.assertIsNone(metrics["corrected_labels"])  # no corrections listed: scored once
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["status"], card["model_versions_returned"], card["run_id"]), ("complete", ["fake-1-001"], "t1_r1"))
        text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| Accuracy | 50.0% (2/4) |", text)
        self.assertNotIn("ISIC-corrected labels", text)
        self.assertNotIn("summary across the runs", text)
        again = FakeModel()  # the same command again: nothing is sent, nothing is overwritten
        self.assertEqual((self._run(again, "test", run_id="t1"), again.calls), (out, 0))
        self.assertEqual(sorted(p.name for p in self.model_dir.iterdir()), ["t1_r1"])  # no .partial, no summary

    def test_the_run_card_records_the_versions_of_the_packages_the_adapter_names(self):
        model = FakeModel()
        model.packages = ("pillow", "no-such-package")
        self._run(model, "dev", run_id="d1", limit=1)
        card = json.loads((self.model_dir / "d1" / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(card["versions"]), ["no-such-package", "pillow", "python"])
        self.assertIsNone(card["versions"]["no-such-package"])
        self.assertRegex(card["versions"]["pillow"], r"^\d+\.\d+")

    def test_the_same_answers_are_scored_against_folder_and_corrected_labels(self):
        write_corrections(self.project.corrections, ("Test/nevus/ISIC_0000009.jpg", "nevus", "melanoma"),
                          ("Test/nevus/ISIC_0000010.jpg", "nevus", "melanoma"),
                          ("Train/nevus/ISIC_0000004.jpg", "nevus", "melanoma"))  # not in the test run: ignored
        self._run(FakeModel(), "test", run_id="t1")
        out = self.model_dir / "t1_r1"
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["folder_labels"]["correct"], 2)
        self.assertEqual(metrics["corrected_labels"]["correct"], 4)
        self.assertEqual(len(metrics["label_corrections"]["applied"]), 2)
        text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| Accuracy | 50.0% (2/4) |", text)
        self.assertIn("| Accuracy | 100.0% (4/4) |", text)
        for other in ("Balanced", "Macro-F1", "kappa", "Recall", "Confusion"):  # accuracy is the only metric reported
            self.assertNotIn(other, text)
        self.assertIn("| Test/nevus/ISIC_0000009.jpg | nevus | melanoma | melanoma |", text)
        main, comparison = text.index("## Main result: scored against the ISIC-corrected labels"), text.index("## For comparison")
        self.assertLess(main, comparison)  # the corrected labels give the headline figure
        self.assertLess(text.index("| Accuracy | 100.0% (4/4) |"), comparison)

    def test_corrections_written_for_other_data_are_refused(self):
        write_corrections(self.project.corrections, ("Test/nevus/ISIC_0000009.jpg", "melanoma", "nevus"))  # folder label differs
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t1")
        write_corrections(self.project.corrections, ("Test/nevus/ISIC_0000009.jpg", "nevus", "vascular lesion"))  # not an option
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t1")

    def test_test_split_runs_once_and_refuses_limit(self):
        self._run(FakeModel(), "test", run_id="t1")
        with self.assertRaises(SystemExit):
            self._run(FakeModel(), "test", run_id="t2")
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t3", limit=1)

    def test_a_stopped_run_keeps_its_answers_apart_and_resumes(self):
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 2):
            partial = self._run(FakeModel(fail_from=3), "dev", run_id="d1")
        self.assertEqual(partial, self.model_dir / "d1.partial")  # Git ignores *.partial
        self.assertFalse((self.model_dir / "d1").exists())
        self.assertEqual(len(self._preds(partial)), 2)
        self.assertEqual(json.loads((partial / "run_card.json").read_text(encoding="utf-8"))["status"], "running")
        second = FakeModel()
        out = self._run(second, "dev", run_id="d1")
        self.assertEqual(second.calls, 2)  # only the two unanswered images
        self.assertEqual(out, self.model_dir / "d1")
        self.assertEqual(len(self._preds(out)), 4)
        self.assertEqual([p.name for p in self.model_dir.iterdir()], ["d1"])
        self.assertTrue((out / "report.md").exists())
        stops = json.loads((out / "run_card.json").read_text(encoding="utf-8"))["interruptions"]  # the stop stays on record
        self.assertEqual([(s["answered"], s["kind"]) for s in stops], [(2, "TransientError")])
        self.assertIn("Interruptions before the run finished: 1", (out / "report.md").read_text(encoding="utf-8"))

    def test_a_run_that_answered_nothing_leaves_no_folder(self):
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 2):
            self._run(FakeModel(fail_from=1), "dev", run_id="d1")  # overloaded from the first image
        self.assertEqual(list(self.model_dir.iterdir()) if self.model_dir.exists() else [], [])
        with self.assertRaises(SystemExit) as raised:  # a refused setting, from the first image
            self._run(FakeModel(fatal_from=1), "dev", run_id="d1")
        self.assertIn("STOPPED, 0 of 4 answered", str(raised.exception))
        self.assertIn("HTTP 404", str(raised.exception))
        self.assertEqual(list(self.model_dir.iterdir()) if self.model_dir.exists() else [], [])

    def test_a_daily_quota_stops_at_once_keeps_the_answers_and_continues_later(self):
        model = FakeModel(quota_from=3)
        partial = self._run(model, "dev", run_id="d1")
        self.assertEqual((partial, model.calls), (self.model_dir / "d1.partial", 3))  # no retries: waiting minutes cannot help
        self.assertEqual(len(self._preds(partial)), 2)
        self.assertFalse([c for c in self.sleep.call_args_list if c.args and c.args[0] >= 10])  # and no waiting either
        later = FakeModel()
        self.assertEqual(self._run(later, "dev", run_id="d1"), self.model_dir / "d1")
        self.assertEqual(later.calls, 2)
        self._run(FakeModel(quota_from=1), "dev", run_id="d2")  # quota used up before the first answer: nothing is left behind
        self.assertEqual([p.name for p in self.model_dir.iterdir()], ["d1"])

    def test_the_check_can_try_another_model_without_changing_the_config(self):
        seen = []

        def load(provider, settings):
            seen.append(settings["id"])
            return mock.Mock(predict=mock.Mock(return_value={"text": "ok", "finish_reason": "STOP", "model_version": "v", "response_id": "r",
                                                             "usage": {}}))

        before = self.config.read_text(encoding="utf-8")
        with mock.patch("src.preflight.models.load", side_effect=load), self.assertRaises(SystemExit) as raised:
            evaluate.main(["check", "--config", str(self.config), "--model", "other-model"])
        self.assertEqual((seen, raised.exception.code), (["other-model"], 0))
        self.assertEqual(self.config.read_text(encoding="utf-8"), before)  # the file itself is unchanged

    def test_a_refusal_after_some_answers_keeps_them(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "dev", run_id="d1")
        self.assertEqual(len(self._preds(self.model_dir / "d1.partial")), 2)
        card = json.loads((self.model_dir / "d1.partial" / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual([s["kind"] for s in card["interruptions"]], ["FatalError"])

    def test_the_default_folder_follows_from_the_settings(self):
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 1):
            partial = self._run(FakeModel(fail_on=[2]), "dev")
        self.assertRegex(partial.name, r"^dev_[0-9a-f]{8}\.partial$")
        again = self._run(FakeModel(), "dev")  # the same command continues the same run
        self.assertEqual(again, self.model_dir / partial.name.removesuffix(".partial"))
        self.assertEqual(len(self._preds(again)), 4)
        with self.assertRaises(SystemExit):  # finished: the same command does not run it again
            self._run(FakeModel(), "dev")
        self.project.prompt.write_text("Choose:\n{labels}", encoding="utf-8")  # other settings: another folder, no clash
        other = self._run(FakeModel(), "dev")
        self.assertNotEqual(other, again)

    def test_a_changed_prompt_cannot_continue_an_old_explicit_run(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "dev", run_id="d1")
        self.project.prompt.write_text("Choose:\n{labels}", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            self._run(FakeModel(), "dev", run_id="d1")
        self.assertIn("different config, prompt or image set", str(raised.exception))

    def test_an_unfinished_test_run_must_be_continued_not_restarted(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "test", run_id="t1")
        with self.assertRaises(SystemExit) as raised:
            self._run(FakeModel(), "test", run_id="t2")
        self.assertIn("--run-id t1", str(raised.exception))
        self.assertEqual(self._run(FakeModel(), "test", run_id="t1"), self.model_dir / "t1_r1")

    def _three_runs(self):
        """Config with 3 test runs and a seed; returns a model loader that records the seed of every model it loads."""
        text = self.config.read_text(encoding="utf-8")
        self.config.write_text(text.replace("test_runs = 1", "test_runs = 3").replace('id = "fake-1"', 'id = "fake-1"\nseed = 7'),
                               encoding="utf-8")
        seeds = []

        def load(models_list):
            def loader(provider, settings):
                seeds.append(settings["seed"])
                return models_list.pop(0)
            return loader
        return seeds, load

    def test_the_test_split_is_run_three_times_with_three_seeds_and_summarized(self):
        seeds, load = self._three_runs()

        class NevusOnce(FakeModel):  # the third run answers nevus for its first image: the runs disagree on one image
            def predict(self, *args):
                answer = super().predict(*args)
                return answer | {"text": "nevus"} if self.calls == 1 else answer

        with mock.patch("src.evaluate.models.load", side_effect=load([FakeModel(), FakeModel(), NevusOnce()])):
            summary = evaluate.run(str(self.config), "test", results=self.results)
        self.assertEqual(seeds, [7, 8, 9])
        group = summary.name.removesuffix("_summary.md")
        self.assertRegex(group, r"^test_[0-9a-f]{8}$")
        cards = [json.loads((self.model_dir / f"{group}_r{n}" / "run_card.json").read_text(encoding="utf-8")) for n in (1, 2, 3)]
        self.assertEqual([(c["repeat"], c["repeats"], c["seed"], c["repeat_group"]) for c in cards],
                         [(1, 3, 7, group), (2, 3, 8, group), (3, 3, 9, group)])
        text = summary.read_text(encoding="utf-8")
        self.assertIn("Images that got the same answer in all 3 runs: 3 of 4 (75.0%)", text)
        self.assertIn(f"| {group}_r1 | 7 | 50.0% (2/4) |", text)
        self.assertIn("| **Mean ± SD** | | 58.3% ± 14.4% (lowest 50.0%, highest 75.0%) |", text)
        self.assertIn("| **Mean ± SD** | |", text)
        data = json.loads((self.model_dir / f"{group}_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["folder_labels"]["accuracy"]["per_run"], [0.5, 0.5, 0.75])  # run 3's nevus answer was a nevus image
        self.assertAlmostEqual(data["folder_labels"]["accuracy"]["mean"], 0.5833333, places=6)
        self.assertAlmostEqual(data["folder_labels"]["accuracy"]["sd"], 0.1443376, places=6)  # sample SD (n - 1)
        self.assertEqual(set(data["folder_labels"]), {"accuracy"})
        self.assertIn("Test run 2 of 3 (seed 8)", (self.model_dir / f"{group}_r2" / "report.md").read_text(encoding="utf-8"))
        again = FakeModel()  # the same command once all runs are finished: no model call, the summary is written again
        with mock.patch("src.evaluate.models.load", return_value=again):
            self.assertEqual(evaluate.run(str(self.config), "test", results=self.results), summary)
        self.assertEqual(again.calls, 0)
        self.config.write_text(self.config.read_text(encoding="utf-8").replace("seed = 7", "seed = 1"), encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:  # a changed setting cannot add runs to the finished ones
            self._run(FakeModel(), "test")
        self.assertIn("a finished test run already exists", str(raised.exception))

    def test_a_stop_in_the_second_test_run_continues_there(self):
        seeds, load = self._three_runs()
        second = FakeModel(quota_from=3)
        with mock.patch("src.evaluate.models.load", side_effect=load([FakeModel(), second])):
            partial = evaluate.run(str(self.config), "test", results=self.results)
        self.assertRegex(partial.name, r"^test_[0-9a-f]{8}_r2\.partial$")
        self.assertEqual(len(self._preds(partial)), 2)
        rest = [FakeModel(), FakeModel()]
        with mock.patch("src.evaluate.models.load", side_effect=load(rest)):
            summary = evaluate.run(str(self.config), "test", results=self.results)
        self.assertEqual(seeds, [7, 8, 8, 9])  # run 1 is not repeated; run 2 continues with its own seed
        self.assertTrue(summary.exists())
        self.assertFalse(list(self.model_dir.glob("*.partial")))

    def test_the_number_of_test_runs_is_checked(self):
        text = self.config.read_text(encoding="utf-8")
        for bad in ("0", "11", '"3"', "true", "2.5"):
            self.config.write_text(text.replace("test_runs = 1", f"test_runs = {bad}"), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_config(self.config)
        self.config.write_text(text.replace("test_runs = 1\n", ""), encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate.load_config(self.config)

    def test_a_stop_in_the_middle_of_a_write_does_not_stop_the_run_from_continuing(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "dev", run_id="d1")
        partial = self.model_dir / "d1.partial" / "predictions.jsonl"
        complete = partial.read_text(encoding="utf-8")
        partial.write_text(complete + '{"path": "Train/mel', encoding="utf-8")  # cut off in the middle of a line
        later = FakeModel()
        out = self._run(later, "dev", run_id="d1")
        self.assertEqual((out, later.calls), (self.model_dir / "d1", 2))  # the cut-off image is answered again
        self.assertEqual(len(self._preds(out)), 4)
        # a complete last line that lacks only its end of line is kept, and the next answer starts on a new line
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "dev", run_id="d2")
        partial = self.model_dir / "d2.partial" / "predictions.jsonl"
        partial.write_text(partial.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
        out = self._run(FakeModel(), "dev", run_id="d2")
        self.assertEqual(len(self._preds(out)), 4)
        # a line cut off inside a multi-byte character ("é" is two bytes) is also removed, and the report reads the partial run
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "dev", run_id="d4")
        partial = self.model_dir / "d4.partial" / "predictions.jsonl"
        partial.write_bytes(partial.read_bytes() + '{"path": "x", "raw_text": "caf\u00e9"}'.encode("utf-8")[:-3])
        self.assertEqual(len(evaluate._read_jsonl(partial)), 2)  # the cut-off line is ignored, nothing crashes
        evaluate.report(partial.parent)
        out = self._run(FakeModel(), "dev", run_id="d4")
        self.assertEqual(len(self._preds(out)), 4)
        # a damaged line in the middle is an error, never skipped
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "dev", run_id="d3")
        partial = self.model_dir / "d3.partial" / "predictions.jsonl"
        first, second = partial.read_text(encoding="utf-8").splitlines()
        partial.write_text(first + "\n{broken\n" + second + "\n", encoding="utf-8")
        with self.assertRaises(ValueError) as raised:
            self._run(FakeModel(), "dev", run_id="d3")
        self.assertIn("line 2 is not valid JSON", str(raised.exception))

    def test_files_are_written_through_a_temporary_file(self):
        target = self.results / "x.json"
        self.results.mkdir(parents=True, exist_ok=True)
        evaluate._write_json(target, {"a": 1})
        with mock.patch("src.evaluate.os.replace", side_effect=OSError("stopped")), self.assertRaises(OSError):
            evaluate._write_json(target, {"a": 2})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": 1})  # the old file is whole

    def test_unsafe_names_and_impossible_settings_are_refused_with_a_message(self):
        for bad in ("../escape", "a/b", "a\\b", "", ".hidden", "x" * 81, "a b"):
            with self.assertRaises(ValueError, msg=bad):
                self._run(FakeModel(), "dev", run_id=bad)
        self.assertFalse(self.results.exists())  # nothing was written
        project_text = self.config.read_text(encoding="utf-8")
        for old, new in (("requests_per_minute = 600000", "requests_per_minute = 0"), ("requests_per_minute = 600000", 'requests_per_minute = "5"'),
                         ("dev_per_class = 2", "dev_per_class = 0"), ("max_image_side = 50", "max_image_side = -1"),
                         ("dev_seed = 0", "dev_seed = 1.5"), ('name = "fake"', 'name = "../fake"'), ('name = "fake"', "")):
            self.config.write_text(project_text.replace(old, new), encoding="utf-8")
            with self.assertRaises(ValueError, msg=new):
                evaluate.load_config(self.config)
        self.config.write_text(project_text, encoding="utf-8")
        for limit in (0, -3):
            with self.assertRaises(ValueError):
                self._run(FakeModel(), "dev", limit=limit)

    def test_development_images_that_copy_a_test_image_are_left_out(self):
        copy = self.project.data / "Train" / "nevus" / "ISIC_9999999.jpg"
        copy.write_bytes(next((self.project.data / "Test" / "nevus").glob("*.jpg")).read_bytes())  # a training copy of a test image
        from src import dataset
        records, _ = dataset.scan(self.project.data, {"train": "Train", "test": "Test"})
        dataset.write_manifest(records, self.project.manifest)
        self.config.write_text(self.config.read_text(encoding="utf-8").replace("dev_per_class = 2", "dev_per_class = 99"),
                               encoding="utf-8")  # every training image is in the sample
        out = self._run(FakeModel(), "dev", run_id="d1")
        paths = [p["path"] for p in self._preds(out)]
        self.assertEqual(len(paths), 6)  # 7 training images, one of them a copy of a test image
        self.assertNotIn("Train/nevus/ISIC_9999999.jpg", paths)

    def test_answers_that_did_not_end_normally_are_flagged_in_the_report(self):
        class Blocked(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": None, "finish_reason": "SAFETY"}

        out = self._run(Blocked(), "dev", run_id="d1")
        text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("4 answers did not end normally (SAFETY: 4)", text)
        clean = (self._run(FakeModel(), "dev", run_id="d2") / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("did not end normally", clean)

    def test_a_folder_that_cannot_be_renamed_is_finished_by_running_the_command_again(self):
        real_rename = Path.rename
        with mock.patch.object(Path, "rename", side_effect=PermissionError("in use")), self.assertRaises(SystemExit) as raised:
            self._run(FakeModel(), "test", run_id="t1")
        self.assertIn("could not be renamed", str(raised.exception))
        self.assertTrue((self.model_dir / "t1_r1.partial" / "report.md").exists())
        self.assertFalse((self.model_dir / "t1_r1").exists())
        again = FakeModel()
        with mock.patch.object(Path, "rename", real_rename):
            out = self._run(again, "test", run_id="t1")  # the test-run guard must not mistake the finished answers for a finished run
        self.assertEqual((out, again.calls), (self.model_dir / "t1_r1", 0))
        self.assertEqual(len(self._preds(self.model_dir / "t1_r1")), 4)

    def test_failed_attempts_wait_longer_each_time(self):
        self._run(FakeModel(fail_on=[1, 2, 3]), "dev", run_id="d1")
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list if c.args and c.args[0] >= 10][:3], [10, 20, 40])

    def test_limit_takes_a_mixed_sample(self):
        out = self._run(FakeModel(), "dev", run_id="d2", limit=3)
        preds = self._preds(out)
        self.assertEqual(len(preds), 3)
        self.assertEqual({p["label"] for p in preds}, {"melanoma", "nevus"})

    def test_images_that_differ_from_the_committed_profile_stop_the_run(self):
        from PIL import Image

        Image.new("RGB", (60, 45), (1, 2, 3)).save(self.project.data / "Test" / "nevus" / "ISIC_0000009.jpg")  # replaced file
        with self.assertRaises(ValueError) as raised:
            self._run(FakeModel(), "test", run_id="t1")
        self.assertIn("1 changed", str(raised.exception))
        self.assertFalse(self.model_dir.exists())

    def test_answers_cut_off_at_the_output_limit_are_flagged_in_the_report(self):
        class Truncated(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": "mela", "finish_reason": "MAX_TOKENS"}

        self._run(Truncated(), "test", run_id="t1")
        text = (self.model_dir / "t1_r1" / "report.md").read_text(encoding="utf-8")
        self.assertIn("4 answers were cut off at the output limit", text)
        self.assertIn("| Unreadable answers (counted as wrong) | 4 |", text)
        self.assertNotIn("cut off", (self._run(FakeModel(), "dev", run_id="d1") / "report.md").read_text(encoding="utf-8"))

    def test_a_missing_key_stops_the_run_before_anything_is_written(self):
        with mock.patch("src.evaluate.models.load", side_effect=ValueError("no key")), self.assertRaises(ValueError):
            evaluate.run(str(self.config), "test", results=self.results)
        self.assertFalse(self.results.exists())

    def test_unreadable_image_stops_the_run(self):
        (self.project.data / "Test" / "nevus" / "broken.jpg").write_bytes(b"not a jpeg")
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t1")

    def test_empty_split_names_are_refused(self):
        self.config.write_text(self.config.read_text(encoding="utf-8").replace('train = "Train"', 'train = ""'), encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate.load_config(self.config)

    def test_the_command_line_turns_expected_problems_into_messages(self):
        with self.assertRaises(SystemExit) as raised:
            evaluate.main(["run", "--config", str(self.config.with_name("missing.toml")), "--split", "dev"])
        self.assertTrue(str(raised.exception).startswith("ERROR: "))


if __name__ == "__main__":
    unittest.main()
