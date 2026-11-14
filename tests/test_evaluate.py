"""The evaluation code (src/evaluate.py) with no network beyond this computer, no real key, no real data and no cost: the metrics
(checked against values worked out by hand), the model adapters, the pipeline on tiny synthetic projects (coloured squares for the
image dataset, short invented cases for the text dataset), and the repository against itself (README, command line, configs and
committed tables must agree; a drift fails here).

Each model adapter is tested twice: with a stand-in for its SDK (to check exactly what is sent and how every kind of answer and
error is turned into the run's own form), and with the real SDK against a local server that returns the exact bodies the real
APIs sent (to check that the real SDK's exceptions are sorted correctly).
"""

import argparse
import ast
import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image

from src import evaluate

ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================================================================
# The metrics
# ==============================================================================================================================

METRIC_LABELS = ["a", "b", "c", "d"]  # d never occurs as a true class below
METRIC_TRUE = list("aaaabbbccc")
METRIC_PRED = ["a", "a", "b", None, "b", "b", "c", "c", "c", "d"]  # one unreadable answer, and one answer naming the absent class d
# Confusion matrix (rows: true a, b, c, d; columns: answered a, b, c, d, unreadable)
#   a: 2 1 0 0 1    b: 0 2 1 0 0    c: 0 0 2 1 0    d: 0 0 0 0 0
# Worked by hand: precision a 2/2, b 2/3, c 2/3; recall a 2/4, b 2/3, c 2/3; F1 a 4/6, b 4/6, c 4/6.
# MCC = (6*10 - (2*4 + 3*3 + 3*3)) / sqrt((100 - (4+9+9+1+1)) * (100 - (16+9+9))) = 34 / sqrt(76*66);
# kappa = (0.6 - 26/100) / (1 - 26/100). Both agree with scikit-learn on this example.


class MetricTests(unittest.TestCase):
    def test_wilson_known_value(self):
        low, high = evaluate.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(evaluate.wilson(0, 0), (0.0, 0.0))

    def test_accuracy_counts_unreadable_as_wrong_and_keeps_its_original_fields(self):
        m = evaluate.summarize(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"])
        self.assertEqual((m["n"], m["correct"], m["unreadable"]), (4, 2, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertEqual(m["accuracy_ci95"], evaluate.wilson(2, 4))

    def test_the_confusion_matrix_has_a_column_for_unreadable_answers(self):
        m = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["confusion_matrix"]
        self.assertEqual(m["rows"], METRIC_LABELS)
        self.assertEqual(m["columns"], METRIC_LABELS + [evaluate.UNREADABLE])
        self.assertEqual(m["counts"], [[2, 1, 0, 0, 1], [0, 2, 1, 0, 0], [0, 0, 2, 1, 0], [0, 0, 0, 0, 0]])

    def test_per_class_precision_recall_and_f1_by_hand(self):
        per_class = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["per_class"]
        a, b, c, d = (per_class[x] for x in METRIC_LABELS)
        self.assertEqual((a["support"], a["answered"], a["tp"], a["fp"], a["fn"]), (4, 2, 2, 0, 2))
        self.assertAlmostEqual(a["precision"], 1.0)
        self.assertAlmostEqual(a["recall"], 0.5)
        self.assertAlmostEqual(a["f1"], 4 / 6)
        for x in (b, c):
            self.assertAlmostEqual(x["precision"], 2 / 3)
            self.assertAlmostEqual(x["recall"], 2 / 3)
            self.assertAlmostEqual(x["f1"], 2 / 3)
        self.assertEqual(a["recall_ci95"], evaluate.wilson(2, 4))

    def test_a_class_without_true_images_has_no_recall_and_stays_out_of_the_averages(self):
        m = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        d = m["per_class"]["d"]
        self.assertEqual((d["support"], d["answered"], d["fp"]), (0, 1, 1))  # a wrong answer naming it is still a false positive
        self.assertIsNone(d["recall"])
        self.assertIsNone(d["recall_ci95"])
        self.assertEqual(m["classes_absent"], ["d"])
        self.assertAlmostEqual(m["macro"]["precision"], (1 + 2 / 3 + 2 / 3) / 3)  # a, b, c only
        self.assertAlmostEqual(m["macro"]["recall"], (0.5 + 2 / 3 + 2 / 3) / 3)
        self.assertAlmostEqual(m["macro"]["f1"], 2 / 3)

    def test_weighted_averages_use_the_number_of_true_images(self):
        w = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["weighted"]
        self.assertAlmostEqual(w["precision"], (4 * 1 + 3 * 2 / 3 + 3 * 2 / 3) / 10)
        self.assertAlmostEqual(w["recall"], (4 * 0.5 + 3 * 2 / 3 + 3 * 2 / 3) / 10)
        self.assertAlmostEqual(w["f1"], 2 / 3)

    def test_balanced_accuracy_mcc_and_kappa(self):
        m = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertAlmostEqual(m["balanced_accuracy"], m["macro"]["recall"])
        self.assertAlmostEqual(m["mcc"], 34 / (76 * 66) ** 0.5)
        self.assertAlmostEqual(m["kappa"], (0.6 - 0.26) / (1 - 0.26))

    def test_a_class_that_was_never_answered_has_no_precision_but_counts_as_zero_in_the_macro_precision(self):
        m = evaluate.summarize(list("aabb"), list("aaaa"), ["a", "b"])
        self.assertIsNone(m["per_class"]["b"]["precision"])
        self.assertAlmostEqual(m["per_class"]["b"]["recall"], 0.0)
        self.assertAlmostEqual(m["per_class"]["b"]["f1"], 0.0)
        self.assertAlmostEqual(m["macro"]["precision"], (0.5 + 0.0) / 2)  # a: 2 of 4 answers right; b: undefined, counted as 0

    def test_unreadable_answers_lower_recall_and_are_never_dropped(self):
        full = evaluate.summarize(list("aabb"), list("aabb"), ["a", "b"])
        gaps = evaluate.summarize(list("aabb"), ["a", None, "b", "b"], ["a", "b"])
        self.assertAlmostEqual(full["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(gaps["per_class"]["a"]["recall"], 0.5)
        self.assertAlmostEqual(gaps["per_class"]["a"]["precision"], 1.0)  # the unreadable answer is nobody's false positive
        self.assertEqual((gaps["unreadable"], gaps["confusion_matrix"]["counts"][0]), (1, [1, 0, 1]))

    def test_invalid_input_is_refused_instead_of_scored(self):
        for true, pred, labels in ((["a"], ["a", "a"], ["a"]),            # different lengths
                                   (["z"], ["a"], ["a", "b"]),            # a true label that is not a class
                                   (["a"], ["z"], ["a", "b"]),            # an answer that is neither a class nor None
                                   (["a"], ["a"], ["a", "a"])):           # a class listed twice
            with self.subTest(true=true, pred=pred, labels=labels), self.assertRaises(ValueError):
                evaluate.summarize(true, pred, labels)

    def test_no_answers_at_all_gives_accuracy_zero_and_no_other_score(self):
        m = evaluate.summarize([], [], ["a", "b"])
        self.assertEqual((m["n"], m["accuracy"], m["unreadable"]), (0, 0.0, 0))
        self.assertEqual(m["classes_absent"], ["a", "b"])
        for value in (m["macro"]["f1"], m["macro"]["f1_ci95"], m["weighted"]["f1"], m["balanced_accuracy"], m["mcc"], m["kappa"]):
            self.assertIsNone(value)

    def test_an_answer_pattern_that_cannot_be_scored_gives_none_not_a_crash(self):
        m = evaluate.summarize(list("aaaa"), list("aaaa"), ["a", "b"])  # one class only: MCC and kappa are 0/0
        self.assertIsNone(m["mcc"])
        self.assertIsNone(m["kappa"])
        self.assertAlmostEqual(m["accuracy"], 1.0)

    def test_the_bootstrap_interval_is_reproducible_stays_with_the_class_sizes_and_collapses_when_all_is_right(self):
        first = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        again = evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertEqual(first["macro"]["f1_ci95"], again["macro"]["f1_ci95"])  # seeded: the same answers give the same interval
        self.assertAlmostEqual(first["macro"]["f1_ci95"][0], 0.3, places=6)  # pinned: a change here means the resampling changed
        self.assertAlmostEqual(first["macro"]["f1_ci95"][1], 14 / 15, places=6)
        perfect = evaluate.summarize(METRIC_TRUE, METRIC_TRUE, METRIC_LABELS)
        self.assertEqual(perfect["macro"]["f1_ci95"], (1.0, 1.0))
        self.assertEqual(perfect["balanced_accuracy_ci95"], (1.0, 1.0))

    def test_photos_of_one_lesion_are_resampled_together_so_the_intervals_widen(self):
        true = ["a"] * 12 + ["b"] * 12
        pred = ["a"] * 8 + ["b"] * 4 + ["b"] * 6 + ["a"] * 6
        lesions = [f"L{i // 4}" for i in range(24)]  # four photos per lesion
        alone = evaluate.summarize(true, pred, ["a", "b"])
        grouped = evaluate.summarize(true, pred, ["a", "b"], groups=lesions)
        self.assertEqual(grouped["balanced_accuracy"], alone["balanced_accuracy"])  # the scores themselves do not change ...
        def width(pair):
            return pair[1] - pair[0]
        for key in ("balanced_accuracy_ci95", "accuracy_ci95", "macro"):  # ... only how certain they are
            a, g = (alone[key]["f1_ci95"], grouped[key]["f1_ci95"]) if key == "macro" else (alone[key], grouped[key])
            self.assertGreater(width(g), width(a), key)
        self.assertGreater(width(grouped["per_class"]["a"]["recall_ci95"]), width(alone["per_class"]["a"]["recall_ci95"]))
        self.assertEqual((grouped["groups"], grouped["interval_method"], alone["groups"]), (6, "bootstrap over lesions", 24))
        self.assertEqual(evaluate.summarize(true, pred, ["a", "b"], groups=[str(i) for i in range(24)]), alone)  # one image each
        with self.assertRaisesRegex(ValueError, "one group per answer"):
            evaluate.summarize(true, pred, ["a", "b"], groups=["L1"])

    def test_cases_of_one_article_are_resampled_together_and_the_summary_says_so(self):
        m = evaluate.summarize(["a", "a", "a", "b"], ["a", "b", "a", "b"], ["a", "b"], groups=["p1", "p1", "p2", "p3"], group_name="article")
        self.assertEqual((m["groups"], m["group_name"], m["interval_method"]), (3, "article", "bootstrap over articles"))

    def test_missed_malignancy_and_over_calls_are_counted_by_hand(self):
        labels = ["actinic keratosis", "basal cell carcinoma", "melanoma", "nevus"]
        true = ["melanoma", "melanoma", "melanoma", "basal cell carcinoma", "nevus", "nevus", "actinic keratosis"]
        pred = ["melanoma", "nevus", None, "actinic keratosis", "nevus", "melanoma", "nevus"]
        risk = evaluate.summarize(true, pred, labels)["risk"]
        # cancers: 4, of which 1 answered benign (melanoma -> nevus); unreadable and actinic keratosis count as neither
        self.assertEqual((risk["missed"], risk["cancers"], risk["missed_malignancy"]), (1, 4, 0.25))
        self.assertEqual((risk["overcalled"], risk["benign"], risk["overcall"]), (1, 2, 0.5))  # nevus -> melanoma
        self.assertEqual(risk["missed_malignancy_ci95"], evaluate.wilson(1, 4))  # every answer its own lesion: Wilson
        grouped = evaluate.summarize(true, pred, labels, groups=["L1", "L1", "L2", "L3", "L4", "L5", "L6"])["risk"]
        low, high = grouped["missed_malignancy_ci95"]
        self.assertTrue(0 <= low <= 0.25 <= high <= 1)  # bootstrap over lesions
        none = evaluate.summarize(["a", "b"], ["a", "a"], ["a", "b"])["risk"]  # classes outside both groups: undefined, not 0
        self.assertEqual((none["missed_malignancy"], none["missed_malignancy_ci95"], none["overcall"]), (None, None, None))

    def test_mcnemar_exact_values_by_hand(self):
        self.assertEqual(evaluate.mcnemar(0, 0), 1.0)  # no question decides between the runs
        self.assertEqual(evaluate.mcnemar(0, 5), 2 / 32)  # 2 x (1/2)^5
        self.assertEqual(evaluate.mcnemar(9, 1), 2 * 11 / 1024)  # 2 x (C(10,0) + C(10,1)) / 2^10
        self.assertEqual(evaluate.mcnemar(3, 3), 1.0)

    def test_the_summary_can_be_saved_as_json(self):
        saved = json.loads(json.dumps(evaluate.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)))
        self.assertEqual(saved["confusion_matrix"]["columns"][-1], evaluate.UNREADABLE)
        self.assertIsNone(saved["per_class"]["d"]["recall"])


# ==============================================================================================================================
# The model adapters
# ==============================================================================================================================


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # the parent package (for example "google") is missing too
        return False


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # a client that gives up is expected here: print nothing
        pass


def _send(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    data = json.dumps(body).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except ConnectionError:  # the client gave up (the timeout case): BrokenPipe, ConnectionReset and, on Windows, ConnectionAborted
        pass


def _serve(test: unittest.TestCase, handler: type) -> QuietServer:
    server = QuietServer(("127.0.0.1", 0), handler)
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return server


# ---------------------------------------------------------------------------------------------------------------------------------
# Shared parts: keys from .env and the choice of adapter


class LoadEnvTests(unittest.TestCase):
    def _load(self, content, existing=None):
        """Loads ``content`` as a .env file into an otherwise empty environment and returns that environment."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, existing or {}, clear=True):
            path = Path(d) / ".env"
            path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
            evaluate.load_env(path)
            return dict(os.environ)

    def test_reads_names_and_values(self):
        env = self._load('# keys\n\nEXAMPLE_API_KEY = abc123\r\nOTHER="quoted value"\nSINGLE=\'x\'\nno equals sign\nEMPTY=\n')
        self.assertEqual(env, {"EXAMPLE_API_KEY": "abc123", "OTHER": "quoted value", "SINGLE": "x"})

    def test_a_byte_order_mark_from_an_editor_is_ignored(self):
        self.assertEqual(self._load(b"\xef\xbb\xbfEXAMPLE_API_KEY=abc"), {"EXAMPLE_API_KEY": "abc"})

    def test_a_key_already_set_in_the_terminal_wins(self):
        self.assertEqual(self._load("EXAMPLE_API_KEY=from-file", {"EXAMPLE_API_KEY": "from-terminal"}),
                         {"EXAMPLE_API_KEY": "from-terminal"})

    def test_a_missing_file_is_fine_and_a_wrongly_encoded_file_is_explained(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            evaluate.load_env(Path(tempfile.gettempdir()) / "no-such-folder" / ".env")
            self.assertEqual(dict(os.environ), {})
        with self.assertRaises(ValueError):  # Windows PowerShell's ">" saves UTF-16
            self._load("EXAMPLE_API_KEY=abc".encode("utf-16"))


class LoadTests(unittest.TestCase):
    def test_a_provider_without_an_adapter_is_refused_with_an_explanation(self):
        with self.assertRaisesRegex(ValueError, r"unknown model provider: 'nothing' \(adapters: anthropic, gemini, openai_compatible\)"):
            evaluate.load("nothing", {})

    def test_a_claude_or_gpt_run_does_not_need_the_gemini_sdk(self):
        code = ("import sys, src.evaluate as m; m.AnthropicModel, m.OpenAICompatibleModel; "
                "print('google.genai' in sys.modules, 'anthropic' in sys.modules, 'openai' in sys.modules)")
        shown = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(shown, ["False", "False", "False"])  # every SDK is imported only when its adapter is loaded


# ---------------------------------------------------------------------------------------------------------------------------------
# Claude (Anthropic), with a stand-in for the Anthropic SDK

CLAUDE = {"id": "claude-test-1", "max_tokens": 64, "thinking": "between_tools", "effort": "low"}


class AnthropicAPIError(Exception):
    pass


class AnthropicStatusError(AnthropicAPIError):
    def __init__(self, status_code, type=None, error_code=None, message="the reason"):
        super().__init__(message)
        self.status_code, self.type, self.error_code, self.api_message = status_code, type, error_code, message


class AnthropicConnectionError(AnthropicAPIError):
    pass


class AnthropicTimeoutError(AnthropicConnectionError):
    pass


def claude_reply(text="Melanoma", stop="end_turn", extra=()):
    blocks = [*extra, types.SimpleNamespace(type="text", text=text)] if text is not None else list(extra)
    return types.SimpleNamespace(content=blocks, stop_reason=stop, model="claude-test-1-served", id="msg_1",
                                 usage=types.SimpleNamespace(input_tokens=1500, output_tokens=3))


class FakeClaudeClient:
    def __init__(self, **kwargs):
        self.kwargs, self.calls, self.answer, self.error = kwargs, [], claude_reply(), None
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.answer


class AnthropicTests(unittest.TestCase):
    def _load(self, settings=None, env=None, answer=None, error=None):
        sdk = types.ModuleType("anthropic")
        sdk.__version__ = "9.9.9"
        sdk.APIError, sdk.APIStatusError, sdk.APIConnectionError = AnthropicAPIError, AnthropicStatusError, AnthropicConnectionError
        self.client = None

        def make(**kwargs):
            self.client = FakeClaudeClient(**kwargs)
            self.client.answer = answer or claude_reply()
            self.client.error = error
            return self.client

        sdk.Anthropic = make
        with mock.patch.dict(sys.modules, {"anthropic": sdk}), mock.patch.dict(os.environ, env or {"CLAUDE_API_KEY": "key-1"}, clear=True):
            return evaluate.load("anthropic", CLAUDE | (settings or {}))

    def test_the_request_carries_the_picture_then_the_prompt_and_only_the_settings_of_the_config(self):
        adapter = self._load()
        adapter.predict(b"\xff\xd8picture", "image/jpeg", "Which one?")
        (sent,) = self.client.calls
        self.assertEqual({k: v for k, v in sent.items() if k != "messages"},
                         {"model": "claude-test-1", "max_tokens": 64, "thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}})
        (turn,) = sent["messages"]
        image, text = turn["content"]
        self.assertEqual(turn["role"], "user")
        self.assertEqual(image["type"], "image")
        self.assertEqual((image["source"]["type"], image["source"]["media_type"]), ("base64", "image/jpeg"))
        self.assertEqual(base64.b64decode(image["source"]["data"]), b"\xff\xd8picture")
        self.assertEqual(text, {"type": "text", "text": "Which one?"})
        for never_sent in ("temperature", "top_p", "top_k", "fallbacks", "system"):  # the answer must come from the named model
            self.assertNotIn(never_sent, sent)

    def test_a_text_case_is_sent_as_the_prompt_alone(self):
        self._load().predict(None, None, "Case: a pearly papule.\nWhich one?")
        (sent,) = self.client.calls
        self.assertEqual(sent["messages"], [{"role": "user", "content": [{"type": "text", "text": "Case: a pearly papule.\nWhich one?"}]}])

    def test_the_answer_is_the_text_blocks_with_the_version_the_vendor_reports_and_the_token_counts(self):
        thought = types.SimpleNamespace(type="thinking", thinking="hidden")
        adapter = self._load(answer=claude_reply("  Mela", extra=[thought]))
        self.client.answer.content.append(types.SimpleNamespace(type="text", text="noma \n"))
        answer = adapter.predict(b"x", "image/png", "?")
        self.assertEqual(answer, {"text": "Mela" + "noma", "finish_reason": "STOP", "model_version": "claude-test-1-served",
                                  "response_id": "msg_1",
                                  "usage": {"prompt_tokens": 1500, "output_tokens": 3, "total_tokens": 1503}})

    def test_how_an_answer_ended(self):
        expected = {"end_turn": "STOP", "stop_sequence": "STOP", "max_tokens": "MAX_TOKENS", "refusal": "REFUSAL", "pause_turn": "PAUSE_TURN"}
        for stop, finish in expected.items():
            with self.subTest(stop):
                self.assertEqual(self._load(answer=claude_reply(stop=stop)).predict(b"x", "image/png", "?")["finish_reason"], finish)
        nothing = self._load(answer=claude_reply(text=None, stop="refusal")).predict(b"x", "image/png", "?")
        self.assertEqual((nothing["text"], nothing["finish_reason"]), (None, "REFUSAL"))

    def test_the_key_comes_from_the_environment_and_the_sdks_own_retries_are_off(self):
        self._load(env={"ANTHROPIC_API_KEY": " first ", "CLAUDE_API_KEY": "second"})
        self.assertEqual(self.client.kwargs, {"api_key": "first", "max_retries": 0, "timeout": 120})
        self._load(env={"CLAUDE_API_KEY": "second"})
        self.assertEqual(self.client.kwargs["api_key"], "second")
        self._load(env={"ANTHROPIC_API_KEY": "  ", "CLAUDE_API_KEY": "second"})  # an empty line does not hide the other key
        self.assertEqual(self.client.kwargs["api_key"], "second")

    def test_a_missing_key_is_explained(self):
        with self.assertRaisesRegex(ValueError, r"CLAUDE_API_KEY=<your key>.*\.env"):
            self._load(env={"UNRELATED": "x"})

    def test_failures_are_sorted_into_retry_later_add_credit_and_fix_a_setting(self):
        retry = [AnthropicStatusError(429, "rate_limit_error"), AnthropicStatusError(529, "overloaded_error"),
                 AnthropicStatusError(500, "api_error"), AnthropicStatusError(408, "timeout_error"),
                 AnthropicStatusError(409, "conflict_error"), AnthropicConnectionError("down"), AnthropicTimeoutError("slow")]
        for error in retry:
            with self.subTest(repr(error)), self.assertRaises(evaluate.TransientError):
                self._load(error=error).predict(b"x", "image/png", "?")
        for error in (AnthropicStatusError(402, "billing_error"),
                      AnthropicStatusError(400, "invalid_request_error", "enforced_spend_limit_reached")):
            with self.subTest(repr(error)), self.assertRaisesRegex(evaluate.QuotaExhausted, "run the same command again"):
                self._load(error=error).predict(b"x", "image/png", "?")
        for status, kind, words in ((401, "authentication_error", "key"), (403, "permission_error", "country"), (404, "not_found_error", "model"),
                                    (400, "invalid_request_error", "Fix the setting"), (413, "request_too_large", "Fix the setting")):
            with self.subTest(status), self.assertRaisesRegex(evaluate.FatalError, words) as raised:
                self._load(error=AnthropicStatusError(status, kind)).predict(b"x", "image/png", "?")
            self.assertEqual(raised.exception.code, status)
        with self.assertRaisesRegex(evaluate.FatalError, "SDK failed"):
            self._load(error=AnthropicAPIError("odd")).predict(b"x", "image/png", "?")

    def test_settings_are_checked_with_the_name_of_the_wrong_one(self):
        for key in ("id", "max_tokens", "thinking", "effort"):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"missing.*{key}"):
                evaluate.load("anthropic", {k: v for k, v in CLAUDE.items() if k != key})
        for bad in ({"thinking": "disabled"}, {"effort": "extreme"}, {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": "64"}, {"id": ""}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._load(bad)

    def test_without_the_package_the_setup_is_explained(self):
        with mock.patch.dict(sys.modules, {"anthropic": None}), self.assertRaisesRegex(ValueError, "pip install -r requirements.txt"):
            evaluate.load("anthropic", CLAUDE)

    def test_a_picture_type_claude_does_not_take_is_refused_before_any_request(self):
        adapter = self._load()
        with self.assertRaisesRegex(evaluate.FatalError, "JPEG, PNG, GIF and WebP"):
            adapter.predict(b"x", "image/tiff", "?")
        self.assertEqual(self.client.calls, [])

    def test_the_run_card_records_what_was_asked_and_that_the_default_sampling_was_used(self):
        description = self._load().describe()
        self.assertEqual((description["provider"], description["model_id"], description["sdk_version"]), ("anthropic", "claude-test-1", "9.9.9"))
        self.assertEqual({k: description["generation_config"][k] for k in ("max_tokens", "thinking", "effort")},
                         {"max_tokens": 64, "thinking": "between_tools", "effort": "low"})
        self.assertIn("provider default", description["generation_config"]["sampling"])
        self.assertIn("anthropic", self._load().packages)

    def test_a_model_name_that_moves_is_not_a_pinned_version(self):
        self.assertTrue(self._load().pinned)
        self.assertFalse(self._load({"id": "claude-sonnet-latest"}).pinned)


# ---------------------------------------------------------------------------------------------------------------------------------
# Gemini (Google), with the real google-genai SDK: its API call replaced, then against a local server

GEMINI = {"id": "gemini-test", "temperature": 0.0, "seed": 0, "max_output_tokens": 64}
GEMINI_GOOD = {"candidates": [{"content": {"role": "model", "parts": [{"text": "melanoma"}]}, "finishReason": "STOP"}],
               "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11}, "modelVersion": "stub-1"}
GEMINI_ERRORS = {  # bodies seen from the Gemini API on the researcher's laptop
    "gone": (404, {"error": {"code": 404, "message": "This model models/gemini-2.5-flash is no longer available to new users.",
                             "status": "NOT_FOUND"}}),
    "level": (400, {"error": {"code": 400, "message": "Thinking level MINIMAL is not supported for this model. Please retry with "
                                                      "other thinking level.", "status": "INVALID_ARGUMENT"}}),
    "key": (400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}),
    "busy": (503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}),
    "daily": (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
        "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: "
        "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n"
        "* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: "
        "gemini-3.8-flash\nDescription: Generate content requests per day limit exceeded on Free Tier. See "
        "https://ai.google.dev/gemini-api/docs/rate-limits.\nPlease retry in 11h37m58.111642937s."}}),  # as Google sent it
    "minute": (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
        "Quota exceeded for metric: generate_content_free_tier_requests, limit: 5, model: gemini-3.8-flash. "
        "Generate content requests per minute limit exceeded. Please retry in 21s."}}),
}


class GeminiHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.seen.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
        model = self.path.split("/models/")[1].split(":")[0]
        if model == "slow":
            time.sleep(1.5)
        _send(self, *GEMINI_ERRORS.get(model, (200, GEMINI_GOOD)))

    def log_message(self, *args):
        pass


@unittest.skipUnless(_have("google.genai"), "google-genai not installed")
class GeminiTests(unittest.TestCase):
    def setUp(self):
        from google.genai import errors, types as genai_types

        self.genai_types, self.errors = genai_types, errors
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "dummy"}):
            self.model = evaluate.load("gemini", GEMINI | {"thinking_level": "MINIMAL"})

    def _respond(self, response):
        return mock.patch.object(self.model.client.models, "generate_content", return_value=response)

    def test_a_missing_key_is_explained(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            evaluate.load("gemini", GEMINI)

    def test_missing_settings_are_named(self):
        for key in ("temperature", "seed", "max_output_tokens", "id"):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "dummy"}), self.assertRaisesRegex(ValueError, f"missing {key}"):
                evaluate.load("gemini", {k: v for k, v in GEMINI.items() if k != key})

    def test_answer_text_version_and_usage(self):
        t = self.genai_types
        response = t.GenerateContentResponse(
            candidates=[t.Candidate(content=t.Content(role="model", parts=[t.Part(text="hidden", thought=True), t.Part(text="melanoma")]),
                                    finish_reason="STOP")],
            model_version="gemini-test-001", response_id="r1",
            usage_metadata=t.GenerateContentResponseUsageMetadata(prompt_token_count=300, candidates_token_count=2, total_token_count=302))
        with self._respond(response) as call:
            answer = self.model.predict(b"\xff\xd8\xff", "image/jpeg", "Which?")
        self.assertEqual((answer["text"], answer["finish_reason"], answer["model_version"]), ("melanoma", "STOP", "gemini-test-001"))
        self.assertEqual(answer["usage"]["total_tokens"], 302)
        sent = call.call_args.kwargs
        self.assertEqual(sent["model"], "gemini-test")
        self.assertEqual(sent["config"].temperature, 0.0)
        self.assertEqual(sent["contents"][1], "Which?")

    def test_a_text_case_is_sent_as_the_prompt_alone(self):
        t = self.genai_types
        response = t.GenerateContentResponse(
            candidates=[t.Candidate(content=t.Content(role="model", parts=[t.Part(text="nevus")]), finish_reason="STOP")])
        with self._respond(response) as call:
            answer = self.model.predict(None, None, "Case: a brown macule.\nWhich?")
        self.assertEqual(call.call_args.kwargs["contents"], ["Case: a brown macule.\nWhich?"])
        self.assertEqual(answer["text"], "nevus")

    def test_thinking_level_is_sent_and_a_deprecated_or_mistyped_setting_is_refused(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "dummy"}):
            level = evaluate.load("gemini", GEMINI | {"thinking_level": "MINIMAL"})
            self.assertEqual(level.config.thinking_config.thinking_level.value, "MINIMAL")
            self.assertIsNone(evaluate.load("gemini", GEMINI).config.thinking_config)  # neither set: the model's default
            with self.assertRaises(ValueError):
                evaluate.load("gemini", GEMINI | {"thinking_budget": 0})
            with self.assertRaises(ValueError):
                evaluate.load("gemini", GEMINI | {"thinking_level": "OFF"})

    def test_a_model_name_that_moves_is_not_a_pinned_version(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "dummy"}):
            self.assertTrue(evaluate.load("gemini", GEMINI | {"id": "gemini-3.1-flash-lite"}).pinned)
            self.assertFalse(evaluate.load("gemini", GEMINI | {"id": "gemini-flash-lite-latest"}).pinned)

    def test_blocked_prompt_gives_no_text(self):
        t = self.genai_types
        response = t.GenerateContentResponse(prompt_feedback=t.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
        with self._respond(response):
            answer = self.model.predict(b"x", "image/jpeg", "Which?")
        self.assertIsNone(answer["text"])
        self.assertEqual(answer["finish_reason"], "PROMPT_BLOCKED_SAFETY")

    def test_rate_limit_is_transient_but_bad_request_is_not(self):
        rate = self.errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        bad = self.errors.APIError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
        with mock.patch.object(self.model.client.models, "generate_content", side_effect=rate):
            with self.assertRaises(evaluate.TransientError):
                self.model.predict(b"x", "image/jpeg", "Which?")
        with mock.patch.object(self.model.client.models, "generate_content", side_effect=bad):
            with self.assertRaises(evaluate.FatalError):
                self.model.predict(b"x", "image/jpeg", "Which?")


@unittest.skipUnless(_have("google.genai"), "google-genai not installed")
class GeminiOverHttpTests(unittest.TestCase):
    def setUp(self):
        self.server = _serve(self, GeminiHandler)
        patcher = mock.patch.dict(os.environ, {"GOOGLE_GEMINI_BASE_URL": f"http://127.0.0.1:{self.server.server_port}",
                                               "GEMINI_API_KEY": "dummy"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _predict(self, model_id):
        return evaluate.load("gemini", GEMINI | {"id": model_id}).predict(b"\x89PNG", "image/png", "Which?")

    def test_an_answer(self):
        self.assertEqual(self._predict("fine")["text"], "melanoma")

    def test_the_real_sdk_sends_a_text_case_as_the_prompt_alone(self):
        answer = evaluate.load("gemini", GEMINI | {"id": "fine"}).predict(None, None, "Case: “café” – which?")
        (body,) = self.server.seen
        self.assertEqual(body["contents"], [{"parts": [{"text": "Case: “café” – which?"}], "role": "user"}])
        self.assertEqual(answer["text"], "melanoma")

    def test_an_overloaded_service_can_be_retried(self):
        with self.assertRaisesRegex(evaluate.TransientError, "HTTP 503 UNAVAILABLE: This model is currently experiencing high demand"):
            self._predict("busy")

    def test_refusals_cannot_be_retried_and_say_what_to_do(self):
        for model_id, code, advice in (("gone", 404, "not available to your key"), ("level", 400, "set another level"),
                                       ("key", 400, "GEMINI_API_KEY line in .env")):
            with self.subTest(model_id), self.assertRaises(evaluate.FatalError) as raised:
                self._predict(model_id)
            self.assertEqual(raised.exception.code, code)
            self.assertIn(advice, str(raised.exception))

    def test_a_used_up_daily_quota_is_not_retried_but_a_per_minute_limit_is(self):
        with self.assertRaises(evaluate.QuotaExhausted) as raised:
            self._predict("daily")
        for expected in ("Daily quota used up for gemini-3.8-flash", "limit 20 requests per day on the free tier",
                         "retry in 11h37m58s", "https://ai.dev/rate-limit"):
            self.assertIn(expected, str(raised.exception))
        with self.assertRaises(evaluate.TransientError):
            self._predict("minute")

    def test_a_stalled_connection_times_out_instead_of_waiting_forever(self):
        with mock.patch.object(evaluate, "TIMEOUT_SECONDS", 0.3), self.assertRaisesRegex(evaluate.TransientError, "network problem"):
            self._predict("slow")

    def test_no_connection_is_a_transient_problem(self):
        self.server.shutdown()
        self.server.server_close()
        with self.assertRaisesRegex(evaluate.TransientError, "network problem"):
            self._predict("fine")


# ---------------------------------------------------------------------------------------------------------------------------------
# GPT-4o and other OpenAI-style hosts, with a stand-in for the OpenAI SDK, then with the real SDK against a local server

OPENAI = {"id": "gpt-test-2026-01-01", "temperature": 0.0, "seed": 0, "max_tokens": 64}


class OpenAIAPIError(Exception):
    pass


class OpenAIStatusError(OpenAIAPIError):
    def __init__(self, status_code, code=None, message="the reason"):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message


class OpenAIConnectionError(OpenAIAPIError):
    pass


class OpenAITimeoutError(OpenAIConnectionError):
    pass


def openai_reply(text="Melanoma", finish="stop", refusal=None, extra=None, usage=True):
    message = types.SimpleNamespace(content=text, refusal=refusal)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)], model="gpt-test-2026-01-01",
                                 id="chatcmpl-1", model_extra=extra or {},
                                 usage=types.SimpleNamespace(prompt_tokens=900, completion_tokens=3, total_tokens=903) if usage else None)


class FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs, self.calls, self.answer, self.error = kwargs, [], openai_reply(), None
        self.base_url = kwargs.get("base_url") or "https://api.openai.com/v1/"
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.answer


class OpenAICompatibleTests(unittest.TestCase):
    def _load(self, settings=None, env=None, answer=None, error=None):
        sdk = types.ModuleType("openai")
        sdk.__version__ = "9.9.9"
        sdk.APIError, sdk.APIStatusError, sdk.APIConnectionError = OpenAIAPIError, OpenAIStatusError, OpenAIConnectionError

        def make(**kwargs):
            self.client = FakeOpenAIClient(**kwargs)
            self.client.answer = answer or openai_reply()
            self.client.error = error
            return self.client

        sdk.OpenAI = make
        with mock.patch.dict(sys.modules, {"openai": sdk}), mock.patch.dict(os.environ, env or {"OPENAI_API_KEY": "key-1"}, clear=True):
            return evaluate.load("openai_compatible", OPENAI | (settings or {}))

    def test_the_request_carries_the_picture_then_the_prompt_and_the_decoding_settings_of_the_config(self):
        adapter = self._load({"image_detail": "high"})
        adapter.predict(b"\xff\xd8picture", "image/jpeg", "Which one?")
        (sent,) = self.client.calls
        self.assertEqual({k: v for k, v in sent.items() if k != "messages"},
                         {"model": "gpt-test-2026-01-01", "temperature": 0.0, "seed": 0, "max_tokens": 64})
        (turn,) = sent["messages"]
        image, text = turn["content"]
        self.assertEqual((turn["role"], image["type"]), ("user", "image_url"))
        prefix = "data:image/jpeg;base64,"
        self.assertTrue(image["image_url"]["url"].startswith(prefix))
        self.assertEqual(base64.b64decode(image["image_url"]["url"][len(prefix):]), b"\xff\xd8picture")
        self.assertEqual(image["image_url"]["detail"], "high")
        self.assertEqual(text, {"type": "text", "text": "Which one?"})
        self.assertNotIn("system", [m["role"] for m in sent["messages"]])

    def test_a_text_case_is_sent_as_the_prompt_alone(self):
        self._load({"image_detail": "high"}).predict(None, None, "Case: a pearly papule.\nWhich one?")
        (sent,) = self.client.calls
        self.assertEqual(sent["messages"], [{"role": "user", "content": [{"type": "text", "text": "Case: a pearly papule.\nWhich one?"}]}])

    def test_options_that_are_not_set_are_not_sent(self):
        self._load().predict(b"x", "image/png", "?")
        (sent,) = self.client.calls
        self.assertNotIn("detail", sent["messages"][0]["content"][0]["image_url"])
        self.assertNotIn("extra_body", sent)
        self._load({"extra_body": {"provider": {"order": ["Host"], "allow_fallbacks": False}}}).predict(b"x", "image/png", "?")
        self.assertEqual(self.client.calls[0]["extra_body"], {"provider": {"order": ["Host"], "allow_fallbacks": False}})

    def test_the_answer_is_the_text_with_the_model_name_the_host_reports_and_the_token_counts(self):
        answer = self._load(answer=openai_reply("  Melanoma \n")).predict(b"x", "image/png", "?")
        self.assertEqual(answer, {"text": "Melanoma", "finish_reason": "STOP", "model_version": "gpt-test-2026-01-01",
                                  "response_id": "chatcmpl-1",
                                  "usage": {"prompt_tokens": 900, "output_tokens": 3, "total_tokens": 903}})

    def test_the_back_end_that_answered_is_recorded_when_the_host_names_it(self):
        answer = self._load(answer=openai_reply(extra={"provider": "SomeHost"})).predict(b"x", "image/png", "?")
        self.assertEqual(answer["model_version"], "gpt-test-2026-01-01@SomeHost")

    def test_how_an_answer_ended(self):
        expected = {"stop": "STOP", "length": "MAX_TOKENS", "content_filter": "CONTENT_FILTER", "tool_calls": "TOOL_CALLS"}
        for finish, shown in expected.items():
            with self.subTest(finish):
                self.assertEqual(self._load(answer=openai_reply(finish=finish)).predict(b"x", "image/png", "?")["finish_reason"], shown)
        refused = self._load(answer=openai_reply(text=None, refusal="I can't help with that.")).predict(b"x", "image/png", "?")
        self.assertEqual((refused["text"], refused["finish_reason"]), (None, "REFUSAL"))
        empty = self._load(answer=openai_reply(text="")).predict(b"x", "image/png", "?")
        self.assertEqual((empty["text"], empty["finish_reason"]), (None, "STOP"))
        silent = self._load(answer=openai_reply(usage=False)).predict(b"x", "image/png", "?")["usage"]
        self.assertEqual(silent, {"prompt_tokens": None, "output_tokens": None, "total_tokens": None})

    def test_the_key_comes_from_the_variable_the_config_names_and_the_sdks_own_retries_are_off(self):
        self._load(env={"OPENAI_API_KEY": " first "})
        self.assertEqual(self.client.kwargs, {"api_key": "first", "base_url": None, "max_retries": 0, "timeout": 120})
        self._load({"api_key_env": "OTHER_KEY", "base_url": "https://host.example/v1"}, env={"OTHER_KEY": "second", "OPENAI_API_KEY": "no"})
        self.assertEqual((self.client.kwargs["api_key"], self.client.kwargs["base_url"]), ("second", "https://host.example/v1"))

    def test_a_missing_key_is_explained_with_the_name_of_the_variable(self):
        with self.assertRaisesRegex(ValueError, r"OPENAI_API_KEY=<your key>.*\.env"):
            self._load(env={"UNRELATED": "x"})
        with self.assertRaisesRegex(ValueError, r"OTHER_KEY=<your key>"):
            self._load({"api_key_env": "OTHER_KEY"}, env={"OPENAI_API_KEY": "x"})

    def test_failures_are_sorted_into_retry_later_add_credit_and_fix_a_setting(self):
        retry = [OpenAIStatusError(429, "rate_limit_exceeded", "Rate limit reached for requests per min"), OpenAIStatusError(500),
                 OpenAIStatusError(503), OpenAIStatusError(408), OpenAIStatusError(409), OpenAIConnectionError("down"),
                 OpenAITimeoutError("slow")]
        for error in retry:
            with self.subTest(repr(error)), self.assertRaises(evaluate.TransientError):
                self._load(error=error).predict(b"x", "image/png", "?")
        stop = [OpenAIStatusError(429, "insufficient_quota", "You exceeded your current quota"),
                OpenAIStatusError(402, 402, "Insufficient credits"), OpenAIStatusError(429, 429, "Rate limit exceeded: free-models-per-day")]
        for error in stop:
            with self.subTest(repr(error)), self.assertRaisesRegex(evaluate.QuotaExhausted, "run the same command again"):
                self._load(error=error).predict(b"x", "image/png", "?")
        for status, words in ((401, "OPENAI_API_KEY"), (403, "country"), (404, "does not know the model"), (400, "Fix the setting"),
                              (413, "Fix the setting")):
            with self.subTest(status), self.assertRaisesRegex(evaluate.FatalError, words) as raised:
                self._load(error=OpenAIStatusError(status)).predict(b"x", "image/png", "?")
            self.assertEqual(raised.exception.code, status)
        with self.assertRaisesRegex(evaluate.FatalError, "SDK failed"):
            self._load(error=OpenAIAPIError("odd")).predict(b"x", "image/png", "?")

    def test_settings_are_checked_with_the_name_of_the_wrong_one(self):
        for key in ("id", "temperature", "seed", "max_tokens"):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"missing.*{key}"):
                evaluate.load("openai_compatible", {k: v for k, v in OPENAI.items() if k != key})
        for bad in ({"id": ""}, {"temperature": -1}, {"temperature": 3}, {"temperature": True}, {"seed": 1.5}, {"seed": True},
                    {"max_tokens": 0}, {"max_tokens": "64"}, {"image_detail": "ultra"}, {"base_url": "host.example"},
                    {"api_key_env": ""}, {"extra_body": "x"}, {"pinned": "yes"}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._load(bad)

    def test_without_the_package_the_setup_is_explained(self):
        with mock.patch.dict(sys.modules, {"openai": None}), self.assertRaisesRegex(ValueError, "pip install -r requirements.txt"):
            evaluate.load("openai_compatible", OPENAI)

    def test_a_picture_type_the_host_does_not_take_is_refused_before_any_request(self):
        adapter = self._load()
        with self.assertRaisesRegex(evaluate.FatalError, "JPEG, PNG, GIF and WebP"):
            adapter.predict(b"x", "image/tiff", "?")
        self.assertEqual(self.client.calls, [])

    def test_the_run_card_records_what_was_asked_and_where(self):
        description = self._load({"image_detail": "high", "base_url": "https://host.example/v1"}).describe()
        self.assertEqual((description["provider"], description["model_id"], description["sdk_version"], description["base_url"]),
                         ("openai_compatible", "gpt-test-2026-01-01", "9.9.9", "https://host.example/v1"))
        self.assertEqual(description["generation_config"],
                         {"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": "high", "extra_body": None})
        self.assertIn("openai", self._load().packages)
        self.assertIn("host.example", self._load({"base_url": "https://host.example/v1"}).loaded_note)
        self.assertNotIn("key-1", self._load().loaded_note)  # the key itself is never shown

    def test_only_a_dated_snapshot_counts_as_a_fixed_version_unless_the_config_says_otherwise(self):
        self.assertTrue(self._load().pinned)
        self.assertFalse(self._load({"id": "gpt-4o"}).pinned)  # the plain name moves
        self.assertTrue(self._load({"id": "meta-llama/some-model", "pinned": True}).pinned)
        self.assertFalse(self._load({"pinned": False}).pinned)


OPENAI_GOOD = {"id": "chatcmpl-9", "object": "chat.completion", "created": 1, "model": "stub-2026-01-01", "provider": "StubHost",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "melanoma"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
OPENAI_ERRORS = {  # the shapes OpenAI and OpenRouter send
    "quota": (429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                              "type": "insufficient_quota", "code": "insufficient_quota"}}),
    "daily": (429, {"error": {"message": "Rate limit exceeded: free-models-per-day. Add credits to unlock more.", "code": 429}}),
    "rate": (429, {"error": {"message": "Rate limit reached for requests per min.", "type": "requests", "code": "rate_limit_exceeded"}}),
    "credits": (402, {"error": {"message": "Insufficient credits. Add more using the billing page.", "code": 402}}),
    "key": (401, {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}),
    "gone": (404, {"error": {"message": "No endpoints found for this model.", "code": 404}}),
    "busy": (503, {"error": {"message": "The server is overloaded.", "type": "server_error"}}),
}


class OpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.server.seen.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": body})
        model = body["model"]
        if model == "slow":
            time.sleep(1.5)
        if model == "limit":
            choice = {"index": 0, "message": {"role": "assistant", "content": "Mela"}, "finish_reason": "length"}
            _send(self, 200, {**OPENAI_GOOD, "choices": [choice]})
        else:
            _send(self, *OPENAI_ERRORS.get(model, (200, OPENAI_GOOD)))

    def log_message(self, *args):
        pass


@unittest.skipUnless(_have("openai"), "openai not installed")
class OpenAICompatibleOverHttpTests(unittest.TestCase):
    def setUp(self):
        self.server = _serve(self, OpenAIHandler)
        patcher = mock.patch.dict(os.environ, {"STUB_KEY": "dummy-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _model(self, model_id, **more):
        settings = {"id": model_id, "temperature": 0.0, "seed": 7, "max_tokens": 64, "api_key_env": "STUB_KEY",
                    "base_url": f"http://127.0.0.1:{self.server.server_port}/v1", "image_detail": "high"}
        return evaluate.load("openai_compatible", settings | more)

    def test_the_real_sdk_sends_the_request_the_adapter_means_to_send(self):
        answer = self._model("stub-2026-01-01").predict(b"\x89PNGdata", "image/png", "Which?")
        (seen,) = self.server.seen
        self.assertEqual((seen["path"], seen["authorization"]), ("/v1/chat/completions", "Bearer dummy-key"))
        body = seen["body"]
        self.assertEqual((body["model"], body["temperature"], body["seed"], body["max_tokens"]), ("stub-2026-01-01", 0.0, 7, 64))
        image, text = body["messages"][0]["content"]
        self.assertEqual(image["image_url"]["detail"], "high")
        self.assertEqual(base64.b64decode(image["image_url"]["url"].split(",", 1)[1]), b"\x89PNGdata")
        self.assertEqual(text, {"type": "text", "text": "Which?"})
        self.assertNotIn("stream", body)
        self.assertEqual((answer["text"], answer["finish_reason"], answer["model_version"]), ("melanoma", "STOP", "stub-2026-01-01@StubHost"))
        self.assertEqual(answer["usage"], {"prompt_tokens": 10, "output_tokens": 1, "total_tokens": 11})

    def test_the_real_sdk_sends_a_text_case_as_the_prompt_alone(self):
        self._model("stub-2026-01-01").predict(None, None, "Case: “café” – which?")
        (seen,) = self.server.seen
        self.assertEqual(seen["body"]["messages"], [{"role": "user", "content": [{"type": "text", "text": "Case: “café” – which?"}]}])

    def test_extra_body_reaches_the_host_as_top_level_fields(self):
        self._model("stub-2026-01-01", extra_body={"provider": {"order": ["StubHost"], "allow_fallbacks": False}}).predict(b"x", "image/png", "?")
        self.assertEqual(self.server.seen[0]["body"]["provider"], {"order": ["StubHost"], "allow_fallbacks": False})

    def test_an_answer_cut_off_at_the_limit_says_so(self):
        answer = self._model("limit").predict(b"x", "image/png", "?")
        self.assertEqual((answer["text"], answer["finish_reason"]), ("Mela", "MAX_TOKENS"))

    def test_the_real_error_answers_are_sorted_correctly(self):
        for model_id, kind in (("quota", evaluate.QuotaExhausted), ("daily", evaluate.QuotaExhausted), ("credits", evaluate.QuotaExhausted),
                               ("rate", evaluate.TransientError), ("busy", evaluate.TransientError),
                               ("key", evaluate.FatalError), ("gone", evaluate.FatalError)):
            with self.subTest(model_id), self.assertRaises(kind):
                self._model(model_id).predict(b"x", "image/png", "?")

    def test_a_refused_key_and_an_unknown_model_say_what_to_do(self):
        with self.assertRaisesRegex(evaluate.FatalError, "STUB_KEY in .env") as raised:
            self._model("key").predict(b"x", "image/png", "?")
        self.assertEqual(raised.exception.code, 401)
        with self.assertRaisesRegex(evaluate.FatalError, "Check the id and base_url") as raised:
            self._model("gone").predict(b"x", "image/png", "?")
        self.assertEqual(raised.exception.code, 404)

    def test_a_stalled_connection_times_out_instead_of_waiting_forever(self):
        with mock.patch.object(evaluate, "TIMEOUT_SECONDS", 0.3), self.assertRaisesRegex(evaluate.TransientError, "cannot reach the host"):
            self._model("slow").predict(b"x", "image/png", "?")

    def test_no_connection_is_a_transient_problem(self):
        model = self._model("stub-2026-01-01")
        self.server.shutdown()
        self.server.server_close()
        with self.assertRaisesRegex(evaluate.TransientError, "cannot reach the host"):
            model.predict(b"x", "image/png", "?")


# ==============================================================================================================================
# The pipeline and the repository
# ==============================================================================================================================


COLOURS = {"melanoma": (120, 40, 30), "nevus": (200, 150, 120)}
SIZES = {"train": 3, "val": 2, "test": 2}  # images per class and split in the synthetic project
CONFIG = """name = "fake"
dataset = "image"
[model]
provider = "fake"
id = "fake-1"
[prompt]
file = "{prompt}"
[run]
requests_per_minute = 600000
order_seed = 0
"""
LABELS = ["nevus", "melanoma", "vascular lesion"]


def write_table(base: Path) -> Path:
    """data/image/dataset.csv for the images now in data/image/dataset (as the dataset preparation writes it)."""
    image_dir, table = base / "data" / "image" / "dataset", base / "data" / "image" / "dataset.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["split", "label", "path", "sha256", "lesion_group", "source"])  # and a further column, as the preparation adds
        for file in sorted(image_dir.rglob("*.jpg")):
            split, label, _ = file.relative_to(image_dir).parts
            group = f"lesion-{label}" if split == "test" else file.stem  # each class's two test images show one lesion
            writer.writerow([split, label, file.relative_to(image_dir).as_posix(), hashlib.sha256(file.read_bytes()).hexdigest(), group,
                             "kaggle"])
    return table


class Project(unittest.TestCase):
    """A prepared dataset (3+3 training, 2+2 validation, 2+2 test images: ISIC_0000001 to ISIC_0000014, in that order), its
    table, a prompt and a config, in a folder laid out like the repository; each test runs inside it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.image_dir = self.base / "data" / "image" / "dataset"
        number = 0
        for split, n in SIZES.items():
            for label, colour in COLOURS.items():
                folder = self.image_dir / split / label
                folder.mkdir(parents=True, exist_ok=True)
                for _ in range(n):
                    number += 1
                    image = Image.new("RGB", (60, 45), colour)
                    image.paste((255, 255, 255), (0, 0, 2 + number, 2 + number))  # every image differs from every other
                    image.save(folder / f"ISIC_{number:07d}.jpg", quality=95)
        self.table = write_table(self.base)
        self.prompt = self.base / "prompt.txt"
        self.prompt.write_text("Pick one:\n{labels}\nAnswer with the name only.", encoding="utf-8")
        self.config = self.base / "fake.toml"
        self.config.write_text(CONFIG.format(prompt=self.prompt.as_posix()), encoding="utf-8")
        self.results = self.base / "results"
        previous = os.getcwd()
        os.chdir(self.base)  # the paths data/image/dataset and data/image/dataset.csv point into the project
        self.addCleanup(os.chdir, previous)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))  # the progress lines of the runs inside the tests


# ---------------------------------------------------------------------------------------------------------------------------------
# The prepared dataset


class DatasetTests(Project):
    header = "split,label,path,sha256\n"

    def _table(self, *rows: str) -> None:
        self.table.write_text(self.header + "".join(r + "\n" for r in rows), encoding="utf-8")

    def _row(self, path: str) -> str:
        split, label, _ = path.split("/")
        return f"{split},{label},{path},{hashlib.sha256((self.image_dir / path).read_bytes()).hexdigest()}"

    def test_a_consistent_dataset_loads_with_its_classes_from_the_training_split(self):
        items = evaluate.load_dataset()
        self.assertEqual(len(items), 14)
        self.assertEqual(sorted({i.split for i in items}), ["test", "train", "val"])
        self.assertEqual(evaluate.classes(items), ["melanoma", "nevus"])

    def test_a_missing_table_says_the_dataset_is_not_prepared_yet(self):
        self.table.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "does not exist yet"):
            evaluate.load_dataset()

    def test_changed_missing_and_extra_images_stop_with_a_count_of_each(self):
        Image.new("RGB", (60, 45), (1, 2, 3)).save(self.image_dir / "test" / "nevus" / "ISIC_0000013.jpg")
        (self.image_dir / "val" / "nevus" / "ISIC_0000009.jpg").unlink()
        (self.image_dir / "train" / "nevus" / "stray.jpg").write_bytes(b"x")
        with self.assertRaises(ValueError) as raised:
            evaluate.load_dataset()
        for expected in ("1 changed", "1 missing", "1 not in the table", "changed: test/nevus/ISIC_0000013.jpg"):
            self.assertIn(expected, str(raised.exception))

    def test_files_that_windows_and_macos_put_in_folders_are_not_extra_images(self):
        for name in ("Thumbs.db", "desktop.ini", ".DS_Store"):
            (self.image_dir / "test" / "nevus" / name).write_bytes(b"x")
        self.assertEqual(len(evaluate.load_dataset()), 14)

    def test_a_table_that_contradicts_itself_is_refused(self):
        good = "train/nevus/ISIC_0000004.jpg"
        sha = hashlib.sha256((self.image_dir / good).read_bytes()).hexdigest()
        cases = {
            "lacks the column": ("split,label,path\n", ["train,nevus," + good]),
            "is not one of": (self.header, [f"holdout,nevus,{good},{sha}"]),
            "must be <split>/<label>/<file>": (self.header, [f"train,melanoma,{good},{sha}"]),  # the folder says nevus
            "not a JPEG or PNG": (self.header, [f"train,nevus,train/nevus/a.gif,{sha}"]),
            "64 lowercase": (self.header, [f"train,nevus,{good},{sha.upper()}"]),
            "same image is also on line 2": (self.header, [self._row(good), f"test,nevus,test/nevus/copy.jpg,{sha}"]),
            "same path is also on line 2": (self.header, [self._row(good), self._row(good)]),
            "lists no file": (self.header, []),
        }
        for message, (header, rows) in cases.items():
            with self.subTest(message):
                self.table.write_text(header + "".join(r + "\n" for r in rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    evaluate.read_table()
        for bad in ("train/../ISIC_0000004.jpg", "train/nevus/../x.jpg"):  # a path cannot point outside its class folder
            with self.subTest(bad):
                self._table(f"train,nevus,{bad},{sha}")
                with self.assertRaises(ValueError):
                    evaluate.read_table()

    def test_a_validation_or_test_class_without_training_images_is_refused(self):
        folder = self.image_dir / "test" / "vascular lesion"
        folder.mkdir()
        Image.new("RGB", (60, 45), (200, 0, 0)).save(folder / "ISIC_0000099.jpg")
        write_table(self.base)
        with self.assertRaisesRegex(ValueError, "vascular lesion"):
            evaluate.classes(evaluate.load_dataset())

    def test_the_image_is_sent_exactly_as_stored(self):
        item = next(i for i in evaluate.load_dataset() if i.split == "test")
        content, mime, sent = evaluate.read_image(item)
        self.assertEqual(content, (self.image_dir / item.path).read_bytes())
        self.assertEqual((mime, sent["sha256"], sent["width"], sent["height"]), ("image/jpeg", item.sha256, 60, 45))

    def test_an_unreadable_image_is_reported_not_skipped(self):
        (self.image_dir / "test" / "nevus" / "ISIC_0000013.jpg").write_bytes(b"not a jpeg")
        write_table(self.base)
        item = next(i for i in evaluate.load_dataset() if i.path == "test/nevus/ISIC_0000013.jpg")
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            evaluate.read_image(item)

    def test_the_hash_of_a_list_of_images_changes_with_any_file_label_or_split(self):
        items = evaluate.load_dataset()
        base = evaluate.items_hash(items)
        self.assertEqual(base, evaluate.items_hash(list(reversed(items))))  # the order does not matter
        first = items[0]
        for change in ({"sha256": "0" * 64}, {"label": "other"}, {"split": "val"}):
            with self.subTest(change):
                self.assertNotEqual(evaluate.items_hash([evaluate.Item(**{**first.__dict__, **change}), *items[1:]]), base)


# ---------------------------------------------------------------------------------------------------------------------------------
# The prompt and the answer reader


class PromptTests(unittest.TestCase):
    def test_build_lists_sorted_labels(self):
        text = evaluate.build_prompt("Pick one:\n{labels}\nAnswer.", LABELS)
        self.assertEqual(text, "Pick one:\n- melanoma\n- nevus\n- vascular lesion\nAnswer.")

    def test_template_needs_one_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.txt"
            path.write_text("no placeholder", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_template(path)

    def test_a_text_prompt_has_the_case_placeholder_once_and_an_image_prompt_has_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.txt"
            for template, dataset in (("Pick:\n{labels}", "text"), ("{case}\n{case}\n{labels}", "text"),
                                      ("Case: {case}\n{labels}", "image")):
                with self.subTest(template):
                    path.write_text(template, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "{case}"):
                        evaluate.load_template(path, dataset)
            path.write_text("Case: {case}\nPick one:\n{labels}\n", encoding="utf-8")
            prompt = evaluate.build_prompt(evaluate.load_template(path, "text"), LABELS)
        case = "A man of 70 had a {labels} lesion – “café”.\nIt grew."  # put in as it is, even text that looks like a placeholder
        self.assertEqual(evaluate.fill(prompt, evaluate.CASE, case), f"Case: {case}\nPick one:\n- melanoma\n- nevus\n- vascular lesion")
        self.assertEqual(evaluate.fill("Case: {case}\nPick", evaluate.CASE, " a papule \n"), "Case:  a papule \n\nPick")  # nothing trimmed

    def test_parse_forgives_only_formatting(self):
        for answer in ["melanoma", " Melanoma.", "**melanoma**", '"melanoma"', "Diagnosis: melanoma", "MELANOMA\n"]:
            self.assertEqual(evaluate.parse_answer(answer, LABELS), "melanoma", answer)
        self.assertEqual(evaluate.parse_answer("Vascular  lesion", LABELS), "vascular lesion")

    def test_parse_forgives_markup_around_a_leading_label(self):
        for answer in ["**Diagnosis:** Melanoma", "**Diagnosis**: Melanoma", "***Diagnosis:*** melanoma", "Answer: **melanoma**",
                       "Final answer: melanoma", "_melanoma_", "`melanoma`", "- melanoma", "**Final Answer:** `Melanoma`."]:
            self.assertEqual(evaluate.parse_answer(answer, LABELS), "melanoma", answer)

    def test_parse_rejects_everything_else(self):
        for answer in [None, "", "melanoma or nevus", "probably melanoma", "basal cell carcinoma", "The answer is nevus.",
                       "The diagnosis is melanoma", "1. melanoma", "MELANOMA!", "melanoma (in situ)", "<end_of_turn>melanoma",
                       "**Diagnosis:** melanoma, nevus", "Melanoma\n\nThis lesion shows irregular pigment."]:
            self.assertIsNone(evaluate.parse_answer(answer, LABELS), answer)

    def test_a_cut_off_answer_could_still_have_become_readable_while_it_looks_like_the_start_of_one(self):
        for answer in [None, "", "  ", "**", "nev", "Nevus", "**mela", "Vascular  les", '"melan', "Diagnosis", "Diagnosis:", "**Diagnosis:** mel",
                       "Final", "Final answer", "Final answer: vasc", "Answer: nev"]:
            self.assertTrue(evaluate.can_still_become_readable(answer, LABELS), repr(answer))

    def test_class_names_are_compared_the_way_the_reader_compares_them(self):
        labels = ["Nevus", "Vascular  Lesion"]  # capitals and a double space in the dataset's own spelling
        self.assertEqual(evaluate.parse_answer("vascular lesion", labels), "Vascular  Lesion")
        for answer in ["nev", "NEVUS", "vascular l", "Vascular  Les"]:
            self.assertTrue(evaluate.can_still_become_readable(answer, labels), repr(answer))
        self.assertFalse(evaluate.can_still_become_readable("nevi", labels))

    def test_a_cut_off_answer_that_already_differs_from_every_class_name_could_not_have_become_readable(self):
        for answer in ["Based on the image, the most likely diagnosis is **melanoma**.", "melanoma or nevus", "The answer is nevus",
                       "Melanoma\n\nThis lesion", "nevi", "basal cell", "1. melanoma", "Diagnosis is melanoma", "Final answers: nevus"]:
            self.assertFalse(evaluate.can_still_become_readable(answer, LABELS), repr(answer))

    def test_whatever_is_readable_or_a_start_of_a_readable_answer_is_never_called_harmless(self):
        """The two functions agree: every answer the reader accepts, and every start of one, could still have become readable."""
        for answer in ["melanoma", "**Diagnosis:** Melanoma", "Final answer: `vascular lesion`."]:
            self.assertIsNotNone(evaluate.parse_answer(answer, LABELS))
            for end in range(len(answer) + 1):
                self.assertTrue(evaluate.can_still_become_readable(answer[:end], LABELS), repr(answer[:end]))


# ---------------------------------------------------------------------------------------------------------------------------------
# Runs


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
            raise evaluate.QuotaExhausted("Daily quota used up for fake-1 (limit 20 requests per day on the free tier)")
        if self.fatal_from and self.calls >= self.fatal_from:
            raise evaluate.FatalError("Provider refused the request (HTTP 404 NOT_FOUND): gone", 404)
        if self.calls in self.fail_on or (self.fail_from and self.calls >= self.fail_from):
            raise evaluate.TransientError("rate limited")
        return {"text": "Melanoma.", "finish_reason": "STOP", "model_version": "fake-1-001", "response_id": str(self.calls),
                "usage": {}}


class UnpinnedModel(FakeModel):
    """A model whose name may move to a newer version (the adapter says ``pinned = False``)."""

    pinned = False


class RunTests(Project):
    def setUp(self):
        super().setUp()
        self.model_dir = self.results / "image" / "fake"
        patcher = mock.patch("src.evaluate.time.sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)
        clean = mock.patch("src.evaluate._git", return_value={"commit": "c" * 40, "uncommitted_changes": False})
        clean.start()
        self.addCleanup(clean.stop)

    def _run(self, model, *args, **kwargs):
        with mock.patch("src.evaluate.load", return_value=model):
            return evaluate.run(str(self.config), *args, results=self.results, **kwargs)

    def _preds(self, folder):
        return [json.loads(x) for x in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_test_run_writes_predictions_metrics_and_report(self):
        out = self._run(FakeModel(fail_on=[2]), "test", run_id="t1")
        self.assertEqual(out, self.model_dir / "t1")  # one test run: its report is the result
        preds = self._preds(out)
        self.assertEqual(len(preds), 4)
        self.assertTrue(all(p["prediction"] == "melanoma" for p in preds))
        self.assertEqual(preds[1]["attempts"], 2)
        sent = preds[0]["image_sent"]  # the file exactly as stored, nothing resized or re-encoded
        self.assertEqual(sent["sha256"], hashlib.sha256((self.image_dir / preds[0]["path"]).read_bytes()).hexdigest())
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["main"]["n"], metrics["main"]["correct"]), (4, 2))
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["status"], card["model_versions_returned"], card["run_id"]), ("complete", ["fake-1-001"], "t1"))
        self.assertEqual((card["n_images"], card["images_sha256"]),  # the names the image dataset's data sheet gives
                         (4, evaluate.items_hash([i for i in evaluate.load_dataset() if i.split == "test"])))
        self.assertEqual(card["dataset_table"]["sha256"], hashlib.sha256(self.table.read_bytes()).hexdigest())
        self.assertEqual({p["lesion_group"] for p in preds}, {"lesion-melanoma", "lesion-nevus"})
        self.assertEqual((metrics["main"]["groups"], metrics["main"]["interval_method"]), (2, "bootstrap over lesions"))
        self.assertEqual(list(metrics), ["main"])  # no class of the image dataset is set apart
        text = (out / "report.md").read_text(encoding="utf-8")
        for expected in ("| **Balanced accuracy (main score: the mean recall of the classes)** | **50.0%** |",
                         "| Accuracy | 50.0% (2/4) |",
                         "(bootstrap over lesions)", "Macro-F1 (classes with images: 2)", "Matthews correlation", "Cohen's kappa",
                         "### Per class", "Recall (95% interval: bootstrap over lesions)", "### Confusion matrix",
                         "Every class has the same number of test images (2), so plain accuracy equals balanced accuracy.",
                         "(2 lesions for 4 images)",
                         "it is not a clinical figure", "No ROC curve or AUC", "sent exactly as stored",
                         "| Missed malignancy: cancers answered as benign | 0.0% (0/2) |",
                         "| Over-calls: benign lesions answered as cancer | 100.0% (2/2) |"):
            self.assertIn(expected, text)
        self.assertLess(text.index("Balanced accuracy (main score"), text.index("| Accuracy |"))  # the main score comes first
        self.assertNotIn("## Without", text)
        again = FakeModel()  # the same command again: nothing is sent, nothing is overwritten
        self.assertEqual((self._run(again, "test", run_id="t1"), again.calls), (out, 0))
        self.assertEqual(sorted(p.name for p in self.model_dir.iterdir()), ["t1"])  # no .partial folder left

    def test_two_runs_on_the_same_questions_are_compared_and_others_are_refused(self):
        class Nevus(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": "nevus"}

        a, b = self._run(FakeModel(), "val", run_id="v1"), self._run(Nevus(), "val", run_id="v2")
        c = evaluate.compare(a, b)
        self.assertEqual((c["n"], c["both_right"], c["only_a"], c["only_b"], c["both_wrong"]), (4, 0, 2, 2, 0))
        self.assertEqual((c["mcnemar_p"], c["balanced_accuracy_difference"], c["accuracy_difference"]), (1.0, 0.0, 0.0))
        low, high = c["balanced_accuracy_difference_ci95"]
        self.assertTrue(low <= 0 <= high)
        text = evaluate.comparison_text(c)
        for expected in ("| **Run B right** | 0 | 2 |", "McNemar's exact test: p = 1.0000", "+0.0 points", "paired bootstrap over lesions"):
            self.assertIn(expected, text)
        same = evaluate.compare(a, a)
        self.assertEqual((same["only_a"], same["only_b"], same["balanced_accuracy_difference_ci95"]), (0, 0, (0.0, 0.0)))
        short = self._run(FakeModel(), "val", run_id="v3", limit=3)
        with self.assertRaisesRegex(ValueError, "differ in their questions"):
            evaluate.compare(a, short)
        test = self._run(FakeModel(), "test", run_id="t1")
        with self.assertRaisesRegex(ValueError, "differ in their split"):
            evaluate.compare(a, test)
        card = json.loads((b / "run_card.json").read_text(encoding="utf-8"))
        card["status"] = "running"
        (b / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a finished run"):
            evaluate.compare(a, b)
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            evaluate.main(["compare", str(a), str(a)])
        self.assertIn("McNemar's exact test: p = 1.0000", shown.getvalue())

    def test_the_paired_interval_resamples_whole_lesions_of_both_runs_together(self):
        """Run B misses all 4 photos of lesion a1 and nothing else. Resampling the two lesions of class a (with replacement) gives
        B's recall of a as 0, 1/2 or 1 with chances 1/4, 1/2, 1/4, so the difference in balanced accuracy is -50, -25 or 0 points
        and the 95% interval is exactly -50 to 0. Resampling the 8 photos one by one would make losing all of them rare (under 1%)
        and give an interval of about -44 to -6 instead."""
        photos = [("a", "a1")] * 4 + [("a", "a2")] * 4 + [("b", "b1")] * 2 + [("b", "b2")] * 2
        runs = []
        for name, wrong in (("A", set()), ("B", {0, 1, 2, 3})):
            folder = self.base / f"run_{name}"
            folder.mkdir()
            card = {"status": "complete", "config": {"dataset": "image"}, "n_images": 12, "images_sha256": "same", "split": "test",
                    "labels": ["a", "b"]}
            (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
            rows = [{"path": f"test/{label}/{i}.jpg", "label": label, "lesion_group": group,
                     "prediction": ("b" if label == "a" else "a") if i in wrong else label} for i, (label, group) in enumerate(photos)]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs.append(folder)
        c = evaluate.compare(*runs)
        self.assertEqual((c["only_a"], c["only_b"], c["balanced_accuracy_difference"]), (4, 0, -0.25))
        self.assertEqual(c["balanced_accuracy_difference_ci95"], (-0.5, 0.0))
        self.assertEqual(c["mcnemar_p"], 0.125)  # 2 x (1/2)^4

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
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 2):
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

    def test_a_run_that_answered_nothing_leaves_no_folder(self):
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 2):
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

    def test_a_refusal_after_some_answers_keeps_them(self):
        with self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=3), "val", run_id="v1")
        self.assertEqual(len(self._preds(self.model_dir / "v1.partial")), 2)
        card = json.loads((self.model_dir / "v1.partial" / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual([s["kind"] for s in card["interruptions"]], ["FatalError"])

    def test_the_default_folder_follows_from_the_settings(self):
        with mock.patch("src.evaluate.MAX_ATTEMPTS", 1):
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
        for seed in (0, 1):  # each seed in a results folder of its own: one config has one test run
            self.config.write_text(original.replace("order_seed = 0", f"order_seed = {seed}"), encoding="utf-8")
            with mock.patch("src.evaluate.load", return_value=FakeModel()):
                out = evaluate.run(str(self.config), "test", run_id="t1", results=self.results / f"seed{seed}")
            by_seed[seed] = [p["path"] for p in self._preds(out)]
            expected = sorted(i.path for i in evaluate.load_dataset() if i.split == "test")
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
        with mock.patch("src.evaluate._git", return_value=dirty):
            with self.assertRaisesRegex(ValueError, "clean Git state"):
                self._run(FakeModel(), "test", run_id="t1")
            self.assertFalse(self.model_dir.exists())  # nothing was started
            self._run(FakeModel(), "val", run_id="v1")  # allowed, and the run card records the state
        card = json.loads((self.model_dir / "v1" / "run_card.json").read_text(encoding="utf-8"))
        self.assertTrue(card["git"]["uncommitted_changes"])
        with mock.patch("src.evaluate._git", return_value={"commit": None, "uncommitted_changes": None}), \
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
        with mock.patch("src.evaluate._git", return_value={"commit": "a" * 40, "uncommitted_changes": False}), \
                self.assertRaises(SystemExit):
            self._run(FakeModel(fatal_from=2), "val", run_id="v1")
        with mock.patch("src.evaluate._git", return_value={"commit": "b" * 40, "uncommitted_changes": False}):
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
        self.assertEqual(len(evaluate.read_jsonl(partial)), 2)  # the cut-off line is ignored, nothing crashes
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
        evaluate.write_json(target, {"a": 1})
        with mock.patch("src.evaluate.os.replace", side_effect=OSError("stopped")), self.assertRaises(OSError):
            evaluate.write_json(target, {"a": 2})
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
                evaluate.load_config(self.config)
        self.config.write_text(text, encoding="utf-8")
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
        Image.new("RGB", (60, 45), (1, 2, 3)).save(self.image_dir / "test" / "nevus" / "ISIC_0000013.jpg")
        with self.assertRaises(ValueError) as raised:
            self._run(FakeModel(), "test", run_id="t1")
        self.assertIn("1 changed", str(raised.exception))
        self.assertFalse(self.model_dir.exists())
        (self.image_dir / "test" / "nevus" / "ISIC_0000013.jpg").unlink()
        (self.image_dir / "test" / "nevus" / "broken.jpg").write_bytes(b"not a jpeg")
        with self.assertRaisesRegex(ValueError, "1 missing, 1 not in the table"):
            self._run(FakeModel(), "test", run_id="t1")

    def test_answers_cut_off_at_the_output_limit_are_flagged_in_the_report(self):
        class Truncated(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": "mela", "finish_reason": "MAX_TOKENS"}

        self._run(Truncated(), "test", run_id="t1")
        text = (self.model_dir / "t1" / "report.md").read_text(encoding="utf-8")
        self.assertIn("4 answers were cut off at the output limit", text)
        self.assertIn("| Unreadable answers (counted as wrong) | 4 |", text)
        self.assertNotIn("cut off", (self._run(FakeModel(), "val", run_id="v1") / "report.md").read_text(encoding="utf-8"))

    def test_essays_cut_off_at_the_limit_do_not_make_the_results_invalid(self):
        class Essay(FakeModel):
            def predict(self, *args):
                text = "Based on the image, the most likely diagnosis is **melanoma**. Here is why:"
                return super().predict(*args) | {"text": text, "finish_reason": "MAX_TOKENS"}

        self._run(Essay(), "test", run_id="t1")
        text = (self.model_dir / "t1" / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("Results are not valid", text)
        self.assertIn("4 answers reached the output limit but already differed from every class name", text)
        self.assertIn("| Unreadable answers (counted as wrong) | 4 |", text)

    def test_only_the_cut_off_answers_that_could_have_become_readable_make_the_results_invalid(self):
        asked = []

        class Mixed(FakeModel):
            def predict(self, *args):
                asked.append(1)
                text = "mela" if len(asked) == 1 else "Based on the image, the most likely diagnosis is melanoma."
                return super().predict(*args) | {"text": text, "finish_reason": "MAX_TOKENS"}

        self._run(Mixed(), "test", run_id="t1")
        text = (self.model_dir / "t1" / "report.md").read_text(encoding="utf-8")
        self.assertIn("1 answers were cut off at the output limit", text)
        self.assertIn("Results are not valid until it is raised", text)
        self.assertIn("3 answers reached the output limit but already differed from every class name", text)

    def test_a_missing_key_stops_the_run_before_anything_is_written(self):
        with mock.patch("src.evaluate.load", side_effect=ValueError("no key")), self.assertRaises(ValueError):
            evaluate.run(str(self.config), "test", results=self.results)
        self.assertFalse(self.results.exists())

    def test_the_command_line_turns_expected_problems_into_messages(self):
        with self.assertRaises(SystemExit) as raised:
            evaluate.main(["run", "--config", str(self.config.with_name("missing.toml")), "--split", "val"])
        self.assertTrue(str(raised.exception).startswith("ERROR: "))
        self.table.unlink()  # the dataset is not prepared yet
        with self.assertRaises(SystemExit) as raised:
            evaluate.main(["run", "--config", str(self.config), "--split", "test"])
        self.assertIn("does not exist yet", str(raised.exception))


# ---------------------------------------------------------------------------------------------------------------------------------
# The setup check


class ProbeModel:
    """Answers the blank-image question as told: ``answer`` (a dict), or raises ``error``."""

    def __init__(self, answer=None, error=None):
        self.answer = answer or {"text": "ok", "finish_reason": "STOP", "model_version": "fake-1-001", "response_id": "r",
                                 "usage": {"output_tokens": 1}}
        self.error, self.calls, self.images = error, 0, []

    def predict(self, image, mime_type, prompt):
        self.calls += 1
        self.images.append((image, mime_type, prompt))
        if self.error:
            raise self.error
        return self.answer


class CheckTests(Project):
    def setUp(self):
        super().setUp()
        patcher = mock.patch("src.evaluate.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _check(self, model=None, load_error=None):
        lines = []
        with mock.patch("src.evaluate.load", return_value=model, side_effect=load_error):
            ok = evaluate.check_setup(evaluate.load_config(self.config), str(self.config), say=lines.append)
        return ok, "\n".join(lines)

    def test_everything_in_order(self):
        model = ProbeModel()
        ok, text = self._check(model)
        self.assertTrue(ok, text)
        for expected in ("[  OK] Config", "[  OK] Prompt file", "[  OK] Prepared dataset: 14 images (train 6, val 4, test 4), 2 classes",
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
        ok, text = self._check(ProbeModel(error=evaluate.FatalError("Provider refused the request (HTTP 404 NOT_FOUND): gone", 404)))
        self.assertFalse(ok)
        self.assertIn("[FAIL] Model fake-1: Provider refused the request (HTTP 404", text)
        self.assertIn("Fix the lines marked FAIL", text)

    def test_a_temporary_overload_is_not_blamed_on_the_setup(self):
        model = ProbeModel(error=evaluate.TransientError("HTTP 503 UNAVAILABLE: high demand"))
        ok, text = self._check(model)
        self.assertFalse(ok)
        self.assertEqual(model.calls, evaluate.PROBE_ATTEMPTS)
        self.assertIn("temporary problem, not a setting", text)

    def test_a_used_up_daily_quota_fails_at_once_and_spends_nothing_more(self):
        model = ProbeModel(error=evaluate.QuotaExhausted("Daily quota used up for fake-1 (limit 20 requests per day)"))
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
        self.assertIn("[FAIL] Prepared dataset: the images in data/image/dataset differ from data/image/dataset.csv", text)
        self.assertEqual(model.calls, 1)  # the model access is still tested, so both problems show at once
        self.table.unlink()
        ok, text = self._check(ProbeModel())
        self.assertIn("[FAIL] Prepared dataset: data/image/dataset.csv not found", text)
        self.assertIn("[  OK] Model fake-1", text)

    def test_a_broken_prompt_file_fails(self):
        self.prompt.write_text("no placeholder", encoding="utf-8")
        ok, text = self._check(ProbeModel())
        self.assertFalse(ok)
        self.assertIn("[FAIL] Prompt file", text)

    def test_the_check_can_try_another_model_without_changing_the_config(self):
        seen = []

        def load(provider, settings):
            seen.append(settings["id"])
            return ProbeModel()

        before = self.config.read_text(encoding="utf-8")
        with mock.patch("src.evaluate.load", side_effect=load), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            evaluate.main(["check", "--config", str(self.config), "--model", "other-model"])
        self.assertEqual((seen, raised.exception.code), (["other-model"], 0))
        self.assertEqual(self.config.read_text(encoding="utf-8"), before)  # the file itself is unchanged


# ---------------------------------------------------------------------------------------------------------------------------------
# The text dataset: each case goes into the prompt, word for word, with no image

TEXT_CASES = {  # (split, class): the article of each case
    ("train", "actinic keratosis"): ["101"], ("train", "melanoma"): ["102", "103"], ("train", "nevus"): ["104", "105"],
    ("val", "actinic keratosis"): ["106"], ("val", "melanoma"): ["107"], ("val", "nevus"): ["108"],
    ("test", "actinic keratosis"): ["201", "202"], ("test", "melanoma"): ["203", "203", "204"],  # one article, two patients
    ("test", "nevus"): ["205", "206", "207"],
}
TEXT_CONFIG = CONFIG.replace('dataset = "image"', 'dataset = "text"')


class TextProject(unittest.TestCase):
    """A prepared text dataset (16 cases: 5 training, 3 validation, 8 test from 7 articles), its table, a prompt with the case
    placeholder and a config, in a folder laid out like the repository; each test runs inside it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.case_dir, self.table = self.base / "data" / "text" / "dataset", self.base / "data" / "text" / "dataset.csv"
        self.texts, rows, number = {}, [], 0
        for (split, label), articles in TEXT_CASES.items():
            (self.case_dir / split / label).mkdir(parents=True)
            for pmid in articles:
                number += 1
                path = f"{split}/{label}/{pmid}-{number}.txt"
                self.texts[path] = f"A {40 + number}-year-old woman had a “café au lait” lesion – case {number}.\nIt grew."
                (self.case_dir / path).write_bytes(self.texts[path].encode("utf-8"))
                rows.append(f"{split},{label},{path},{hashlib.sha256(self.texts[path].encode('utf-8')).hexdigest()},{pmid}-{number},"
                            f"{pmid},PMC-Patients\n")
        self.table.write_text("split,label,path,sha256,patient_uid,pmid,source\n" + "".join(rows), encoding="utf-8")
        self.prompt = self.base / "prompt.txt"
        self.prompt.write_text("A case.\nCase: {case}\nPick one:\n{labels}\nAnswer with the name only.\n", encoding="utf-8")
        self.config = self.base / "fake.toml"
        self.config.write_text(TEXT_CONFIG.format(prompt=self.prompt.as_posix()), encoding="utf-8")
        self.results = self.base / "results"
        previous = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))  # the progress lines of the runs inside the tests
        for name, value in (("time.sleep", None), ("_git", {"commit": "c" * 40, "uncommitted_changes": False})):
            patcher = mock.patch(f"src.evaluate.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)


class RecordingModel(FakeModel):
    """Says "melanoma" for every case, and keeps what each request carried."""

    def __init__(self):
        super().__init__()
        self.requests = []

    def predict(self, image, mime_type, prompt):
        self.requests.append((image, mime_type, prompt))
        return super().predict(image, mime_type, prompt)


class TextDatasetTests(TextProject):
    def _run(self, model, *args, **kwargs):
        with mock.patch("src.evaluate.load", return_value=model):
            return evaluate.run(str(self.config), *args, results=self.results, **kwargs)

    def test_the_cases_load_with_their_article_as_their_group_and_only_text_files_are_accepted(self):
        items = evaluate.load_dataset(*evaluate.dataset_paths("text"), "text")
        self.assertEqual((len(items), evaluate.classes(items)), (16, ["actinic keratosis", "melanoma", "nevus"]))
        self.assertEqual(sorted(i.group for i in items if i.split == "test" and i.label == "melanoma"), ["203", "203", "204"])
        with self.assertRaisesRegex(ValueError, "is not a JPEG or PNG file"):
            evaluate.read_table(self.table, "image")  # a text table is never read as images
        with self.table.open("a", encoding="utf-8") as f:
            f.write(f"train,nevus,train/nevus/photo.jpg,{'0' * 64},x,9,PMC-Patients\n")
        with self.assertRaisesRegex(ValueError, "is not a .txt file"):
            evaluate.read_table(self.table, "text")

    def test_a_case_is_sent_exactly_as_stored_and_an_unreadable_one_stops_the_run(self):
        item = next(i for i in evaluate.load_dataset(*evaluate.dataset_paths("text"), "text") if i.split == "test")
        text, sent = evaluate.read_case(item, self.case_dir)
        self.assertEqual(text, self.texts[item.path])
        self.assertEqual((sent["sha256"], sent["words"]), (item.sha256, len(self.texts[item.path].split())))
        for content, message in ((b"caf\xe9", "not UTF-8 text"), (b" \n", "is empty")):
            with self.subTest(message):
                (self.case_dir / item.path).write_bytes(content)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate.read_case(item, self.case_dir)

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
            self.assertNotIn("image_sent", pred)
        card = json.loads((out / "run_card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["labels"], card["n_cases"], card["group_column"], card["prompt"]), (labels, 8, "pmid", prompt))
        self.assertEqual(card["cases_sha256"], evaluate.items_hash([i for i in evaluate.load_dataset(*evaluate.dataset_paths("text"), "text")
                                                                    if i.split == "test"]))

    def test_the_report_resamples_articles_and_gives_the_scores_without_actinic_keratosis_beside_the_main_score(self):
        out = self._run(FakeModel(), "test", run_id="t1")  # "melanoma" for every case: right for the 3 melanomas only
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        main, apart = metrics["main"], metrics["without_small_classes"]
        self.assertEqual((main["n"], main["groups"], main["interval_method"]), (8, 7, "bootstrap over articles"))
        self.assertAlmostEqual(main["balanced_accuracy"], 1 / 3)  # recall 0, 1, 0
        self.assertEqual((apart["left_out"], apart["metrics"]["n"], apart["metrics"]["classes_absent"]),
                         (["actinic keratosis"], 6, ["actinic keratosis"]))
        self.assertEqual((apart["metrics"]["balanced_accuracy"], apart["metrics"]["accuracy"]), (0.5, 0.5))  # recall 1, 0
        self.assertEqual([(f["label"], f["prediction"]) for f in apart["files"]], [("actinic keratosis", "melanoma")] * 2)
        text = (out / "report.md").read_text(encoding="utf-8")
        for expected in ("| **Balanced accuracy (main score: the mean recall of the classes)** | **33.3%** |",
                         "| Accuracy | 37.5% (3/8) |", "(bootstrap over articles)", "Macro-F1 (classes with cases: 3)",
                         "Patients described in one article are not independent: the intervals resample whole articles (7 articles "
                         "for 8 cases).", "cases: 8 (hash", "each put into the prompt word for word",
                         "## Without actinic keratosis",
                         "The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet "
                         "it weighs as much as any other class in the balanced accuracy above (one class in 3).",
                         "never instead of it", "answering actinic keratosis for another case still counts as wrong",
                         "| Balanced accuracy (2 classes) | 50.0% |", "| Accuracy | 50.0% (3/6) |",
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
        evaluate.write_report(out)
        self.assertEqual(list(json.loads((out / "metrics.json").read_text(encoding="utf-8"))), ["main"])
        self.assertNotIn("## Without", (out / "report.md").read_text(encoding="utf-8"))

    def test_the_check_asks_the_model_without_an_image_and_counts_cases(self):
        model, lines = ProbeModel(), []
        with mock.patch("src.evaluate.load", return_value=model):
            ok = evaluate.check_setup(evaluate.load_config(self.config), str(self.config), say=lines.append)
        text = "\n".join(lines)
        self.assertTrue(ok, text)
        self.assertIn("[  OK] Prepared dataset: 16 cases (train 5, val 3, test 8), 3 classes", text)
        self.assertIn("one question with no image (nothing from the dataset is sent", text)
        self.assertEqual(model.images, [(None, None, evaluate.PROBE_PROMPT)])
        self.prompt.write_text("Pick one:\n{labels}\n", encoding="utf-8")  # a text prompt without the case
        ok, lines = None, []
        with mock.patch("src.evaluate.load", return_value=ProbeModel()):
            ok = evaluate.check_setup(evaluate.load_config(self.config), str(self.config), say=lines.append)
        self.assertFalse(ok)
        self.assertIn("[FAIL] Prompt file: ", "\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------------------------
# The multimodal dataset: each photo with its patient's facts in the prompt

class MultimodalProject(Project):
    """The image project, with data/multimodal/questions.csv giving each photo its patient's facts (one test photo lacks a fact),
    a prompt with the facts placeholder and a multimodal config."""

    MISSING = "test/nevus/ISIC_0000013.jpg"

    def setUp(self):
        super().setUp()
        with self.table.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.questions = self.base / "data" / "multimodal" / "questions.csv"
        self.questions.parent.mkdir(parents=True)
        with self.questions.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "label", "path", "sha256", "lesion_group", "facts"], lineterminator="\n")
            writer.writeheader()
            for i, row in enumerate(rows):
                site = "not recorded" if row["path"] == self.MISSING else "trunk"
                writer.writerow({k: row[k] for k in ("split", "label", "path", "sha256", "lesion_group")}
                                | {"facts": f"Age: about {30 + 5 * i} years. Sex: female. Body site: {site}."})
        self.prompt.write_text("A photo.\nPatient: {facts}\nPick one:\n{labels}\nAnswer with the name only.\n", encoding="utf-8")
        self.config.write_text(CONFIG.replace('dataset = "image"', 'dataset = "multimodal"').format(prompt=self.prompt.as_posix()),
                               encoding="utf-8")
        for name, value in (("time.sleep", None), ("_git", {"commit": "c" * 40, "uncommitted_changes": False})):
            patcher = mock.patch(f"src.evaluate.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)


class MultimodalTests(MultimodalProject):
    def _run(self, model, *args, **kwargs):
        with mock.patch("src.evaluate.load", return_value=model):
            return evaluate.run(str(self.config), *args, results=self.results, **kwargs)

    def test_the_questions_are_the_image_dataset_s_photos_with_their_facts(self):
        self.assertEqual(evaluate.dataset_paths("multimodal"), (Path("data/multimodal/questions.csv"), Path("data/image/dataset")))
        items = evaluate.load_dataset(*evaluate.dataset_paths("multimodal"), "multimodal")
        self.assertEqual(len(items), 14)
        missing = next(i for i in items if i.path == self.MISSING)
        self.assertTrue(missing.facts.endswith("Body site: not recorded."))
        changed = [evaluate.Item(**{**items[0].__dict__, "facts": "Age: about 99 years. Sex: male. Body site: trunk."}), *items[1:]]
        self.assertNotEqual(evaluate.items_hash(changed), evaluate.items_hash(items))  # a changed fact is a changed question
        text = self.questions.read_text(encoding="utf-8")
        self.questions.write_text(text.replace("Age: about 30 years. Sex: female. Body site: trunk.", ""), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "the facts column is empty"):
            evaluate.read_table(self.questions, "multimodal")
        self.questions.write_text(text.replace(",facts\n", "\n", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks the column"):
            evaluate.read_table(self.questions, "multimodal")

    def test_a_multimodal_prompt_needs_the_facts_placeholder_and_no_other_prompt_may_have_it(self):
        for template, dataset in (("Pick:\n{labels}", "multimodal"), ("Patient: {facts}\n{labels}", "image"),
                                  ("Case: {case}\nPatient: {facts}\n{labels}", "text")):
            with self.subTest(dataset):
                self.prompt.write_text(template, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "{facts}"):
                    evaluate.load_template(self.prompt, dataset)

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
            self.assertEqual(prompt, "A photo.\nPatient: " + facts[path] + "\nPick one:\n- melanoma\n- nevus\nAnswer with the name only.")
            self.assertEqual(preds[path]["facts"], facts[path])
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((metrics["main"]["n"], metrics["all_facts_recorded"]["questions"], metrics["all_facts_recorded"]["metrics"]["n"]),
                         (4, 3, 3))
        report = (out / "report.md").read_text(encoding="utf-8")
        for expected in ("sent exactly as stored, each with its patient's facts in the prompt", "## Questions with all facts recorded",
                         "These scores count only the 3 of 4 questions with every fact recorded", "(bootstrap over lesions)"):
            self.assertIn(expected, report)
        self.assertLess(report.index("## Results"), report.index("## Questions with all facts recorded"))

    def test_the_check_sends_a_blank_image_and_counts_the_questions(self):
        model, lines = ProbeModel(), []
        with mock.patch("src.evaluate.load", return_value=model):
            ok = evaluate.check_setup(evaluate.load_config(self.config), str(self.config), say=lines.append)
        self.assertTrue(ok, "\n".join(lines))
        self.assertIn("[  OK] Prepared dataset: 14 images (train 6, val 4, test 4), 2 classes", "\n".join(lines))
        self.assertEqual(model.images[0][1:], ("image/png", evaluate.PROBE_PROMPT))


# ---------------------------------------------------------------------------------------------------------------------------------
# The repository against itself


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        previous = os.getcwd()
        os.chdir(ROOT)  # configs name their files relative to the repository folder
        self.addCleanup(os.chdir, previous)
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_every_command_in_the_usage_documentation_is_accepted_by_the_command_line(self):
        usage = re.findall(r"^\s+python -m src\.evaluate (.+?)$", evaluate.__doc__, re.M)
        self.assertGreaterEqual(len(usage), 3)
        for line in usage:
            for form in (re.sub(r"\[[^\]]*\]", "", line), re.sub(r"\[([^\]]*)\]", r"\1", line)):  # without / with the optional [parts]
                with self.subTest(form):
                    evaluate.build_parser().parse_args(shlex.split(form))
        commands = next(a for a in evaluate.build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
        for name in commands:
            self.assertIn(f"python -m src.evaluate {name}", evaluate.__doc__, f"the usage documentation omits '{name}'")

    def test_the_commands_written_in_the_readme_are_accepted_by_the_command_line(self):
        commands = re.findall(r"`python -m src\.evaluate ([^`]+)`", self.readme)
        self.assertGreaterEqual(len(commands), 2)
        for command in commands:
            with self.subTest(command):
                args = evaluate.build_parser().parse_args(shlex.split(command))
                if hasattr(args, "config"):
                    self.assertTrue((ROOT / args.config).is_file(), f"the README names a config that does not exist: {args.config}")

    def test_the_readme_names_every_code_and_test_file(self):
        for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py")):
            if path.name != "__init__.py":
                self.assertIn(path.name, self.readme, f"the README tree omits {path.name}")

    def test_the_results_written_in_the_readme_match_the_saved_results(self):
        def pct(x):
            return f"{100 * x:.1f}"

        header = next(line for line in self.readme.splitlines() if line.startswith("| Dataset |"))
        self.assertEqual([c.strip() for c in header.strip().strip("|").split("|")],
                         ["Dataset", "Model", "Setting", "Balanced accuracy (test)", "95% interval",
                          "Missed malignancy (cancers called benign)", "Report"])
        rows = [line for line in self.readme.splitlines() if line.startswith("| ") and "](results/" in line]
        for row in rows:  # no row until a model has a finished test run
            link = re.search(r"\]\((results/([^/]+)/([^/]+)/[^)]+/report\.md)\)", row)
            self.assertTrue((ROOT / link.group(1)).is_file(), f"the README links to a report that does not exist: {link.group(1)}")
            main = json.loads((ROOT / link.group(1)).with_name("metrics.json").read_text(encoding="utf-8"))["main"]
            low, high = main["balanced_accuracy_ci95"]  # the main score
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            self.assertEqual(cells[:2], [link.group(2), link.group(3)], "the dataset and model must be the run's own")
            self.assertEqual(cells[3:6], [pct(main["balanced_accuracy"]), f"{pct(low)} to {pct(high)}",
                                          pct(main["risk"]["missed_malignancy"]) if main["risk"]["missed_malignancy"] is not None else "–"])
        finished = [p for p in (ROOT / "results").glob("*/*/*") if p.is_dir() and p.name.startswith("test_") and not p.name.endswith(".partial")]
        self.assertEqual(len(rows), len(finished), "every finished test run in results/ must have exactly one README row")

    def test_configs_are_valid_and_point_to_files_that_exist(self):
        configs = sorted((ROOT / "configs").glob("*/*.toml"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(path.name):
                config = evaluate.load_config(path)
                self.assertEqual(config["name"], path.stem, "the config name is the results folder name and must match the file")
                self.assertEqual(config["dataset"], path.parent.name, "a config sits in the folder of the dataset it names")
                self.assertTrue((ROOT / config["prompt"]["file"]).is_file(), f"{path.name} names a prompt file that does not exist")

    def test_each_model_is_asked_the_same_way_on_every_dataset(self):
        """The same models, settings and pace on every dataset; only the image setting is left out for text."""
        configs = {}
        for path in sorted((ROOT / "configs").glob("*/*.toml")):
            config = evaluate.load_config(path)
            configs.setdefault(config["dataset"], {})[config["name"]] = config
        self.assertEqual(sorted(configs["text"]), sorted(configs["image"]))
        self.assertEqual(sorted(configs["multimodal"]), sorted(configs["image"]))
        for name, text in configs["text"].items():
            with self.subTest(name):
                image = configs["image"][name]
                self.assertEqual(text["model"], {k: v for k, v in image["model"].items() if k != "image_detail"})
                self.assertEqual(text["run"], image["run"])
                self.assertEqual((configs["multimodal"][name]["model"], configs["multimodal"][name]["run"]), (image["model"], image["run"]))

    def test_the_committed_dataset_tables_are_consistent(self):
        for dataset, kind in evaluate.DATASETS.items():
            table = evaluate.dataset_paths(dataset)[0]
            if table.exists():  # committed once the dataset is prepared; its files stay out of Git
                with self.subTest(dataset):
                    labels = evaluate.classes(evaluate.read_table(table, dataset))
                    self.assertEqual([c for c in kind.set_apart if c not in labels], [], "a class set apart must be a class")
                    self.assertTrue(all(item.group for item in evaluate.read_table(table, dataset)), "every file needs its group")

    def test_every_prompt_file_is_a_template_with_one_label_placeholder_and_a_final_line_break(self):
        files = sorted((ROOT / "prompts").glob("*/*.txt"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(path.name):
                evaluate.load_template(path, path.parent.name)  # refuses a file without exactly one {labels} (and {case} for text)
                self.assertTrue(path.read_bytes().endswith(b"\n"))
                self.assertNotIn(b"\r", path.read_bytes(), "Windows line endings (.gitattributes should prevent this)")
                self.assertNotIn("\n\n", path.read_text(encoding="utf-8"), "a blank line would change what the model is shown")

    def test_every_imported_package_is_in_the_one_requirements_file_and_pinned(self):
        listed = {}  # package -> how it is constrained (every package must be an exact "==" pin)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if line := line.split("#")[0].strip():
                name, operator, _ = re.fullmatch(r"([A-Za-z0-9._-]+)(==|>=)(\S+)", line).groups()
                listed[name.lower()] = operator
        self.assertFalse(list(ROOT.glob("requirements*.txt"))[1:], "there must be one requirements file")
        distribution = {"PIL": "pillow", "anthropic": "anthropic", "openai": "openai", "google": "google-genai",
                        "httpx": "httpx", "cv2": "opencv-python-headless", "numpy": "numpy"}  # import name -> package name
        imported = set()
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        outside = sorted(imported - set(sys.stdlib_module_names) - {"src"})
        self.assertEqual([name for name in outside if name not in distribution], [], "add this package to the map in this test and to requirements.txt")
        for name in outside:
            self.assertIn(distribution[name], listed, f"{name} is imported but {distribution[name]} is not in requirements.txt")
        for package, operator in listed.items():
            self.assertEqual(operator, "==", f"{package} must be pinned with ==")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_git_ignores_keys_images_unfinished_and_validation_runs_but_not_tables_and_test_results(self):
        def ignored(path):
            return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

        for path in (".env", ".env.txt", ".env.local", "data/image/raw/Train/melanoma/x.jpg", "data/image/downloads/isic/ISIC_0000001.json",
                     "data/image/downloads/isic_pictures/ISIC_0000001.jpg", "data/image/dataset/test/nevus/x.jpg",
                     "data/image/dataset/x.csv", "data/text/sources/pmc_patients/PMC-Patients-V2.json",
                     "data/text/sources/pmc_patients/download.json", "data/text/sources/x.csv", "data/text/dataset/test/melanoma/1-1.txt",
                     "data/text/sources/pubmed/abstracts.jsonl", "data/text/sources/audit_sample.txt",
                     "data/text/sources/library/statpearls/NBK1.html", "data/text/library/passages.jsonl", "data/anything_else.txt",
                     "data/x.csv",
                     "results/image/m/val_1/report.md", "results/image/m/test_1.partial/predictions.jsonl", "src/__pycache__/x.pyc",
                     "results/text/m/val_1/report.md", "results/text/m/test_1.partial/predictions.jsonl",
                     "src/image/__pycache__/x.pyc", ".venv/x", ".coverage"):
            self.assertTrue(ignored(path), f"{path} must be ignored by Git")
        for path in ("README.md", "data/image/dataset.csv", "data/image/raw_photos.csv", "data/image/dataset_report.md",
                     "data/image/datasheet.md", "data/text/dataset.csv", "data/text/case_candidates.csv", "data/text/article_mesh.csv", "data/text/abstract_candidates.csv",
                     "data/text/library.csv", "results/image/m/test_1/report.md",
                     "results/image/m/test_1/predictions.jsonl", "configs/image/example.toml", "prompts/image/example.txt",
                     "results/text/m/test_1/report.md", "configs/text/example.toml", "prompts/text/example.txt"):
            self.assertFalse(ignored(path), f"{path} must not be ignored by Git")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_no_key_image_or_unfinished_run_is_tracked(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual([p for p in tracked if p.startswith(".env") or ".partial/" in p], [])
        self.assertEqual([p for p in tracked if Path(p).suffix.lower() in evaluate.MEDIA_TYPES], [], "no image belongs in Git")
        key_formats = re.compile(r"AIza[0-9A-Za-z_-]{35}|\bsk-[A-Za-z0-9_-]{20,}|\bgsk_[A-Za-z0-9]{20,}|\bhf_[A-Za-z0-9]{30,}"
                                 r"|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")  # Google, OpenAI-style, Groq, Hugging Face, GitHub
        for path in tracked:
            if path and (ROOT / path).suffix in {".py", ".md", ".toml", ".txt", ".csv", ".json", ".jsonl"} and (ROOT / path).is_file():
                self.assertIsNone(key_formats.search((ROOT / path).read_text(encoding="utf-8")), f"key-like text in {path}")

    def test_the_readme_uses_unix_line_endings(self):
        self.assertNotIn(b"\r", (ROOT / "README.md").read_bytes(), "Windows line endings (.gitattributes should prevent this)")


if __name__ == "__main__":
    unittest.main()
