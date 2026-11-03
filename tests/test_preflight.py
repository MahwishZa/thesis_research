import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src import evaluate, models, preflight
from tests.support import make_project, write_corrections


class ProbeModel:
    """Answers the blank-image question as told: ``answer`` (a dict), or raises ``error``."""

    def __init__(self, answer=None, error=None, names=None):
        self.answer = answer or {"text": "ok", "finish_reason": "STOP", "model_version": "fake-1-001", "response_id": "r",
                                 "usage": {"output_tokens": 1}}
        self.error, self.calls, self.images = error, 0, []
        if names is not None:
            self.list_models = lambda: names

    def predict(self, image, mime_type, prompt):
        self.calls += 1
        self.images.append((image, mime_type, prompt))
        if self.error:
            raise self.error
        return self.answer


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = make_project(Path(self.tmp.name))
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch("src.preflight.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _check(self, model=None, load_error=None, list_models=False):
        lines = []
        with mock.patch("src.preflight.models.load", return_value=model, side_effect=load_error):
            ok = preflight.check(evaluate.load_config(self.project.config), str(self.project.config), say=lines.append,
                                 list_models=list_models)
        return ok, "\n".join(lines)

    def test_everything_in_order(self):
        model = ProbeModel()
        ok, text = self._check(model)
        self.assertTrue(ok, text)
        for expected in ("[  OK] Config", "[  OK] Prompt file", "[  OK] Classes", "[  OK] Images vs committed profile: 6 training and 4 testing",
                         "[  OK] Label corrections", "[  OK] Model adapter and API key", "[  OK] Model fake-1", "All checks passed"):
            self.assertIn(expected, text)
        image, mime, _ = model.images[0]  # only a blank synthetic image is ever sent
        self.assertEqual((mime, Image.open(io.BytesIO(image)).size), ("image/png", (64, 64)))

    def test_an_unavailable_model_is_reported_with_the_models_that_are(self):
        error = models.FatalError("Provider refused the request (HTTP 404 NOT_FOUND): gone", 404)
        ok, text = self._check(ProbeModel(error=error, names=["model-a", "model-b"]))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model fake-1: Provider refused the request (HTTP 404", text)
        self.assertIn("models your key can use", text)
        self.assertIn("model-a, model-b", text)
        self.assertIn("Fix the lines marked FAIL", text)

    def test_the_models_can_be_listed_while_the_model_works(self):
        ok, text = self._check(ProbeModel(names=["model-a", "model-b"]), list_models=True)
        self.assertTrue(ok)
        self.assertEqual(text.count("model-a, model-b"), 1)
        ok, text = self._check(ProbeModel(), list_models=True)  # a provider that cannot list them says so
        self.assertIn("cannot list its models", text)

    def test_a_temporary_overload_is_not_blamed_on_the_setup(self):
        model = ProbeModel(error=models.TransientError("HTTP 503 UNAVAILABLE: high demand"))
        ok, text = self._check(model)
        self.assertFalse(ok)
        self.assertEqual(model.calls, preflight.PROBE_ATTEMPTS)
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
        self.assertIn("raise max_output_tokens", text)

    def test_a_missing_key_or_adapter_stops_early(self):
        ok, text = self._check(load_error=ValueError("no API key found"))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model adapter and API key: no API key found", text)

    def test_a_dataset_that_differs_from_the_profile_fails(self):
        next((self.project.data / "Test" / "nevus").iterdir()).unlink()
        ok, text = self._check(ProbeModel())
        self.assertFalse(ok)
        self.assertIn("[FAIL] Images vs committed profile", text)

    def test_bad_corrections_and_missing_folders_fail(self):
        write_corrections(self.project.corrections, ("Test/nevus/ISIC_0000009.jpg", "nevus", "vascular lesion"))
        ok, text = self._check(ProbeModel())
        self.assertFalse(ok)
        self.assertIn("[FAIL] Label corrections", text)
        (self.project.data / "Test").rename(self.project.data / "Elsewhere")
        ok, text = self._check(ProbeModel())
        self.assertIn("[FAIL] Dataset test folder", text)


if __name__ == "__main__":
    unittest.main()
