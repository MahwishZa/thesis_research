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
                                 "usage": {"thinking_tokens": 0}}
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

    def _check(self, model=None, load_error=None, variants=False, list_models=False):
        lines = []
        with mock.patch("src.preflight.models.load", return_value=model, side_effect=load_error):
            ok = preflight.check(evaluate.load_config(self.project.config), str(self.project.config), variants, say=lines.append,
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
        error = models.FatalError("Google refused the request (HTTP 404 NOT_FOUND): gone", 404)
        ok, text = self._check(ProbeModel(error=error, names=["gemini-a", "gemini-b"]))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model fake-1: Google refused the request (HTTP 404", text)
        self.assertIn("models your key can use", text)
        self.assertIn("gemini-a, gemini-b", text)
        self.assertIn("Fix the lines marked FAIL", text)

    def test_the_models_can_be_listed_while_the_model_works(self):
        ok, text = self._check(ProbeModel(names=["gemini-a", "gemini-b"]), list_models=True)
        self.assertTrue(ok)
        self.assertEqual(text.count("gemini-a, gemini-b"), 1)
        ok, text = self._check(ProbeModel(), list_models=True)  # a provider that cannot list them says so
        self.assertIn("cannot list its models", text)

    def test_a_temporary_overload_is_not_blamed_on_the_setup(self):
        model = ProbeModel(error=models.TransientError("HTTP 503 UNAVAILABLE: high demand"))
        ok, text = self._check(model)
        self.assertFalse(ok)
        self.assertEqual(model.calls, preflight.PROBE_ATTEMPTS)
        self.assertIn("temporary problem, not a setting", text)

    def test_a_cut_off_answer_fails(self):
        answer = {"text": "o", "finish_reason": "MAX_TOKENS", "model_version": "v", "response_id": "r", "usage": {"thinking_tokens": 60}}
        ok, text = self._check(ProbeModel(answer))
        self.assertFalse(ok)
        self.assertIn("raise max_output_tokens", text)

    def test_thinking_despite_a_budget_of_zero_is_a_warning(self):
        self.project.config.write_text(self.project.config.read_text(encoding="utf-8").replace('id = "fake-1"', 'id = "fake-1"\nthinking_budget = 0'),
                                       encoding="utf-8")
        answer = {"text": "ok", "finish_reason": "STOP", "model_version": "v", "response_id": "r", "usage": {"thinking_tokens": 58}}
        ok, text = self._check(ProbeModel(answer))
        self.assertTrue(ok)
        self.assertIn("[WARN] Model fake-1", text)

    def test_a_missing_key_stops_early(self):
        ok, text = self._check(load_error=ValueError("no Gemini API key found"))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model adapter and API key: no Gemini API key found", text)

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

    def test_every_thinking_setting_can_be_probed(self):
        def load(provider, settings):
            if settings.get("thinking_level") == "MINIMAL":
                return ProbeModel(error=models.FatalError("Google refused the request (HTTP 400): MINIMAL not supported", 400))
            return ProbeModel()

        lines = []
        with mock.patch("src.preflight.models.load", side_effect=load), \
                mock.patch("src.preflight.models.settings_variants", return_value=[("thinking_level = MINIMAL", {"thinking_level": "MINIMAL"}),
                                                                                    ("thinking_level = LOW", {"thinking_level": "LOW"})]):
            preflight.check(evaluate.load_config(self.project.config), str(self.project.config), True, say=lines.append)
        text = "\n".join(lines)
        self.assertIn("- thinking_level = MINIMAL: REFUSED: Google refused the request (HTTP 400): MINIMAL not supported", text)
        self.assertIn("- thinking_level = LOW: accepted, thinking tokens 0, finish reason STOP", text)


    def _variants(self, loads, variants):
        """The check with ``variants`` probed by the models in ``loads`` (one per call of load, in order); returns (ok, text)."""
        lines, loaded = [], iter(loads)
        with mock.patch("src.preflight.models.load", side_effect=lambda provider, settings: next(loaded)), \
                mock.patch("src.preflight.models.settings_variants", return_value=variants):
            ok = preflight.check(evaluate.load_config(self.project.config), str(self.project.config), True, say=lines.append)
        return ok, "\n".join(lines)

    def test_a_used_up_daily_quota_fails_at_once_and_spends_nothing_more(self):
        quota = models.QuotaExhausted("Daily quota used up for fake-1 (limit 20 requests per day on the free tier)")
        model = ProbeModel(error=quota)
        ok, text = self._check(model, variants=True)
        self.assertFalse(ok)
        self.assertEqual(model.calls, 1)  # no retries, and the thinking settings are not probed
        self.assertIn("[FAIL] Model fake-1: Daily quota used up for fake-1", text)
        self.assertIn("the thinking settings were not tested: the daily quota is used up", text)
        self.assertIn("a used-up daily quota needs the reset, a paid key or another model", text)

    def test_a_quota_that_runs_out_during_the_thinking_probes_is_a_warning_and_stops_them(self):
        quota = models.QuotaExhausted("Daily quota used up for fake-1")
        third = ProbeModel()
        ok, text = self._variants([ProbeModel(), ProbeModel(), ProbeModel(error=quota), third],
                                  [("setting A", {"a": 1}), ("setting B", {"b": 1}), ("setting C", {"c": 1})])
        self.assertTrue(ok)
        self.assertIn("- setting A: accepted", text)
        self.assertIn("- setting B: not tested: Daily quota used up", text)
        self.assertNotIn("setting C", text)
        self.assertEqual(third.calls, 0)
        self.assertIn("[WARN] Thinking settings: the remaining ones were not tested", text)
        self.assertIn("Checks passed, but read the WARN lines", text)

    def test_a_temporary_problem_while_probing_is_not_called_a_refusal(self):
        busy = ProbeModel(error=models.TransientError("HTTP 503 UNAVAILABLE: busy"))
        ok, text = self._variants([ProbeModel(), busy, ProbeModel()], [("setting A", {"a": 1}), ("setting B", {"b": 1})])
        self.assertTrue(ok)
        self.assertIn("- setting A: not tested: temporary problem, not a setting: HTTP 503", text)
        self.assertIn("- setting B: accepted", text)

    def test_the_configured_setting_is_not_asked_twice(self):
        settings = evaluate.load_config(self.project.config)["model"]
        configured = ProbeModel()
        ok, text = self._variants([ProbeModel(), configured], [("SKIPPED-SETTING", dict(settings)), ("another", {"x": 1})])
        self.assertNotIn("SKIPPED-SETTING", text)  # skipped: it was tested above, and every request counts against the quota
        self.assertIn("- another: accepted", text)


if __name__ == "__main__":
    unittest.main()
