import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import evaluate, models
from tests.support import make_dataset

CONFIG = """name = "fake"
[data]
root = "{root}"
train = "Train"
test = "Test"
corrections = "{corrections}"
[model]
provider = "fake"
id = "fake-1"
[prompt]
file = "{prompt}"
[run]
requests_per_minute = 600000
dev_per_class = 2
dev_seed = 0
order_seed = 0
max_image_side = 50
"""


class FakeModel:
    """Says "melanoma" for every image; fails transiently once on the second call."""

    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call

    def describe(self):
        return {"model_id": "fake-1", "generation_config": {}}

    def predict(self, image, mime_type, prompt):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise models.TransientError("rate limited")
        return {"text": "Melanoma.", "finish_reason": "STOP", "model_version": "fake-1-001", "response_id": str(self.calls),
                "usage": {}}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        make_dataset(base / "data", {"Train": {"melanoma": 3, "nevus": 3}, "Test": {"melanoma": 2, "nevus": 2}})
        (base / "prompt.txt").write_text("Pick one:\n{labels}\nAnswer with the name only.", encoding="utf-8")
        self.corrections = base / "label_corrections.csv"
        self._corrections()
        self.config = base / "fake.toml"
        self.config.write_text(CONFIG.format(root=(base / "data").as_posix(), prompt=(base / "prompt.txt").as_posix(),
                                             corrections=self.corrections.as_posix()), encoding="utf-8")
        self.results = base / "results"
        patcher = mock.patch("src.evaluate.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _corrections(self, *rows):
        """Writes the corrections file: (path, folder label, corrected label) per row."""
        lines = ["path,isic_id,folder_label,corrected_label,isic_fields"]
        lines += [f"{path},{Path(path).stem},{folder},{corrected},diagnosis={corrected}" for path, folder, corrected in rows]
        self.corrections.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run(self, model, *args, **kwargs):
        with mock.patch("src.evaluate.models.load", return_value=model):
            return evaluate.run(str(self.config), *args, results=self.results, **kwargs)

    def test_test_run_writes_predictions_metrics_and_report(self):
        out = self._run(FakeModel(fail_on_call=2), "test", run_id="t1")
        preds = [json.loads(x) for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(preds), 4)
        self.assertTrue(all(p["prediction"] == "melanoma" for p in preds))
        self.assertEqual(preds[1]["attempts"], 2)
        self.assertEqual((preds[0]["image_sent"]["resized"], preds[0]["image_sent"]["width"]), (True, 50))  # 60x45 > 50
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["folder_labels"]["n"], metrics["folder_labels"]["correct"]), (4, 2))
        self.assertIsNone(metrics["corrected_labels"])  # no corrections listed: scored once
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["status"], card["model_versions_returned"]), ("complete", ["fake-1-001"]))
        text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| Accuracy | 50.0% (2/4) |", text)
        self.assertNotIn("ISIC-corrected labels", text)

    def test_the_same_answers_are_scored_against_folder_and_corrected_labels(self):
        self._corrections(("Test/nevus/ISIC_0000009.jpg", "nevus", "melanoma"), ("Test/nevus/ISIC_0000010.jpg", "nevus", "melanoma"),
                          ("Train/nevus/ISIC_0000004.jpg", "nevus", "melanoma"))  # not in the test run: ignored
        out = self._run(FakeModel(), "test", run_id="t1")
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["folder_labels"]["correct"], 2)
        self.assertEqual(metrics["corrected_labels"]["correct"], 4)
        self.assertEqual(metrics["corrected_labels"]["balanced_accuracy"], 1.0)  # averaged over melanoma only
        self.assertEqual(len(metrics["label_corrections"]["applied"]), 2)
        text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| Accuracy | 50.0% (2/4) |", text)
        self.assertIn("| Accuracy | 100.0% (4/4) |", text)
        self.assertIn("| nevus | 0 | n/a | n/a | n/a | 0 |", text)  # an answer option with no image under corrected labels
        self.assertIn("any one of the 2 largest classes (melanoma, nevus) 50.0%", text)  # folder labels: a tie
        self.assertIn("the largest class (melanoma) 100.0%", text)  # corrected labels
        self.assertIn("answer options with no image: nevus", text)
        self.assertIn("| Test/nevus/ISIC_0000009.jpg | nevus | melanoma | melanoma |", text)

    def test_corrections_written_for_other_data_are_refused(self):
        self._corrections(("Test/nevus/ISIC_0000009.jpg", "melanoma", "nevus"))  # the folder label does not match
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t1")
        self._corrections(("Test/nevus/ISIC_0000009.jpg", "nevus", "vascular lesion"))  # not an answer option
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t1")

    def test_test_split_runs_once_and_refuses_limit(self):
        self._run(FakeModel(), "test", run_id="t1")
        with self.assertRaises(SystemExit):
            self._run(FakeModel(), "test", run_id="t2")
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t3", limit=1)

    def test_interrupted_run_resumes_without_repeating(self):
        model = FakeModel(fail_on_call=1)
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 1):
            out = self._run(model, "dev", run_id="d1")
        self.assertFalse((out / "predictions.jsonl").read_text(encoding="utf-8").strip())
        self.assertEqual(json.loads((out / "run_card.json").read_text(encoding="utf-8"))["status"], "running")
        second = FakeModel()
        self._run(second, "dev", run_id="d1")
        self.assertEqual(second.calls, 4)
        self.assertEqual(len((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()), 4)

    def test_changed_prompt_cannot_continue_an_old_run(self):
        model = FakeModel(fail_on_call=1)
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 1):
            self._run(model, "dev", run_id="d1")
        (Path(self.tmp.name) / "prompt.txt").write_text("Choose:\n{labels}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self._run(FakeModel(), "dev", run_id="d1")

    def test_limit_takes_a_mixed_sample(self):
        out = self._run(FakeModel(), "dev", run_id="d2", limit=3)
        preds = [json.loads(x) for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(preds), 3)
        self.assertEqual({p["label"] for p in preds}, {"melanoma", "nevus"})

    def test_unreadable_image_stops_the_run(self):
        (Path(self.tmp.name) / "data" / "Test" / "nevus" / "broken.jpg").write_bytes(b"not a jpeg")
        with self.assertRaises(ValueError):
            self._run(FakeModel(), "test", run_id="t1")

    def test_empty_split_names_are_refused(self):
        self.config.write_text(self.config.read_text(encoding="utf-8").replace('train = "Train"', 'train = ""'), encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate.load_config(self.config)


if __name__ == "__main__":
    unittest.main()
