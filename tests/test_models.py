"""The model adapters (src/models.py), with no network beyond this computer, no real key and no cost.

Each adapter is tested twice: with a stand-in for its SDK (to check exactly what is sent and how every kind of answer and error
is turned into the run's own form), and with the real SDK against a local server that returns the exact bodies the real APIs
sent (to check that the real SDK's exceptions are sorted correctly)."""

import base64
import importlib.util
import json
import os
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

from src import models


ROOT = Path(__file__).resolve().parents[1]


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
            models.load_env(path)
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
            models.load_env(Path(tempfile.gettempdir()) / "no-such-folder" / ".env")
            self.assertEqual(dict(os.environ), {})
        with self.assertRaises(ValueError):  # Windows PowerShell's ">" saves UTF-16
            self._load("EXAMPLE_API_KEY=abc".encode("utf-16"))


class LoadTests(unittest.TestCase):
    def test_a_provider_without_an_adapter_is_refused_with_an_explanation(self):
        with self.assertRaisesRegex(ValueError, r"unknown model provider: 'nothing' \(adapters: anthropic, openai_compatible\)"):
            models.load("nothing", {})

    def test_an_sdk_is_imported_only_when_its_adapter_is_loaded(self):
        code = ("import sys, src.run, src.models as m; m.AnthropicModel, m.OpenAICompatibleModel; "
                "print('anthropic' in sys.modules, 'openai' in sys.modules)")
        shown = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(shown, ["False", "False"])


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
            return models.load("anthropic", CLAUDE | (settings or {}))

    def test_the_request_carries_the_picture_then_the_prompt_and_only_the_settings_of_the_config(self):
        adapter = self._load()
        adapter.predict([(b"\xff\xd8picture", "image/jpeg")], "Which one?")
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
        self._load().predict([], "Case: a pearly papule.\nWhich one?")
        (sent,) = self.client.calls
        self.assertEqual(sent["messages"], [{"role": "user", "content": [{"type": "text", "text": "Case: a pearly papule.\nWhich one?"}]}])

    def test_several_pictures_go_before_the_prompt_in_their_order(self):
        self._load().predict([(b"one", "image/jpeg"), (b"two", "image/png")], "Which one?")
        (sent,) = self.client.calls
        first, second, text = sent["messages"][0]["content"]
        self.assertEqual([base64.b64decode(p["source"]["data"]) for p in (first, second)], [b"one", b"two"])
        self.assertEqual((first["source"]["media_type"], second["source"]["media_type"]), ("image/jpeg", "image/png"))
        self.assertEqual(text, {"type": "text", "text": "Which one?"})

    def test_the_answer_is_the_text_blocks_with_the_version_the_vendor_reports_and_the_token_counts(self):
        thought = types.SimpleNamespace(type="thinking", thinking="hidden")
        adapter = self._load(answer=claude_reply("  Mela", extra=[thought]))
        self.client.answer.content.append(types.SimpleNamespace(type="text", text="noma \n"))
        answer = adapter.predict([(b"x", "image/png")], "?")
        self.assertEqual(answer, {"text": "Mela" + "noma", "finish_reason": "STOP", "model_version": "claude-test-1-served",
                                  "response_id": "msg_1",
                                  "usage": {"prompt_tokens": 1500, "output_tokens": 3, "total_tokens": 1503}})

    def test_how_an_answer_ended(self):
        expected = {"end_turn": "STOP", "stop_sequence": "STOP", "max_tokens": "MAX_TOKENS", "refusal": "REFUSAL", "pause_turn": "PAUSE_TURN"}
        for stop, finish in expected.items():
            with self.subTest(stop):
                self.assertEqual(self._load(answer=claude_reply(stop=stop)).predict([(b"x", "image/png")], "?")["finish_reason"], finish)
        nothing = self._load(answer=claude_reply(text=None, stop="refusal")).predict([(b"x", "image/png")], "?")
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
            with self.subTest(repr(error)), self.assertRaises(models.TransientError):
                self._load(error=error).predict([(b"x", "image/png")], "?")
        for error in (AnthropicStatusError(402, "billing_error"),
                      AnthropicStatusError(400, "invalid_request_error", "enforced_spend_limit_reached")):
            with self.subTest(repr(error)), self.assertRaisesRegex(models.QuotaExhausted, "run the same command again"):
                self._load(error=error).predict([(b"x", "image/png")], "?")
        for status, kind, words in ((401, "authentication_error", "key"), (403, "permission_error", "country"), (404, "not_found_error", "model"),
                                    (400, "invalid_request_error", "Fix the setting"), (413, "request_too_large", "Fix the setting")):
            with self.subTest(status), self.assertRaisesRegex(models.FatalError, words) as raised:
                self._load(error=AnthropicStatusError(status, kind)).predict([(b"x", "image/png")], "?")
            self.assertEqual(raised.exception.code, status)
        with self.assertRaisesRegex(models.FatalError, "SDK failed"):
            self._load(error=AnthropicAPIError("odd")).predict([(b"x", "image/png")], "?")

    def test_settings_are_checked_with_the_name_of_the_wrong_one(self):
        for key in ("id", "max_tokens", "thinking", "effort"):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"missing.*{key}"):
                models.load("anthropic", {k: v for k, v in CLAUDE.items() if k != key})
        for bad in ({"thinking": "disabled"}, {"effort": "extreme"}, {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": "64"}, {"id": ""}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._load(bad)

    def test_without_the_package_the_setup_is_explained(self):
        with mock.patch.dict(sys.modules, {"anthropic": None}), self.assertRaisesRegex(ValueError, "pip install -r requirements.txt"):
            models.load("anthropic", CLAUDE)

    def test_a_picture_type_claude_does_not_take_is_refused_before_any_request(self):
        adapter = self._load()
        with self.assertRaisesRegex(models.FatalError, "JPEG, PNG, GIF and WebP"):
            adapter.predict([(b"x", "image/tiff")], "?")
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
# GPT-4o and other OpenAI-style hosts, with a stand-in for the OpenAI SDK, then with the real SDK against a local server

OPENAI = {"id": "gpt-test-2026-01-01", "temperature": 0.0, "seed": 0, "max_tokens": 64}


class OpenAIAPIError(Exception):
    pass


class OpenAIStatusError(OpenAIAPIError):
    def __init__(self, status_code, code=None, message="the reason", body=None):
        super().__init__(message)
        self.status_code, self.code, self.message, self.body = status_code, code, message, body


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
            return models.load("openai_compatible", OPENAI | (settings or {}))

    def test_the_request_carries_the_picture_then_the_prompt_and_the_decoding_settings_of_the_config(self):
        adapter = self._load({"image_detail": "high"})
        adapter.predict([(b"\xff\xd8picture", "image/jpeg")], "Which one?")
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
        self._load({"image_detail": "high"}).predict([], "Case: a pearly papule.\nWhich one?")
        (sent,) = self.client.calls
        self.assertEqual(sent["messages"], [{"role": "user", "content": [{"type": "text", "text": "Case: a pearly papule.\nWhich one?"}]}])

    def test_several_pictures_go_before_the_prompt_in_their_order(self):
        self._load({"image_detail": "high"}).predict([(b"one", "image/jpeg"), (b"two", "image/png")], "Which one?")
        (sent,) = self.client.calls
        first, second, text = sent["messages"][0]["content"]
        self.assertEqual([p["image_url"]["url"] for p in (first, second)],
                         [f"data:image/jpeg;base64,{base64.b64encode(b'one').decode()}", f"data:image/png;base64,{base64.b64encode(b'two').decode()}"])
        self.assertEqual((first["image_url"]["detail"], second["image_url"]["detail"]), ("high", "high"))
        self.assertEqual(text, {"type": "text", "text": "Which one?"})

    def test_options_that_are_not_set_are_not_sent(self):
        self._load().predict([(b"x", "image/png")], "?")
        (sent,) = self.client.calls
        self.assertNotIn("detail", sent["messages"][0]["content"][0]["image_url"])
        self.assertNotIn("extra_body", sent)
        self._load({"extra_body": {"provider": {"order": ["Host"], "allow_fallbacks": False}}}).predict([(b"x", "image/png")], "?")
        self.assertEqual(self.client.calls[0]["extra_body"], {"provider": {"order": ["Host"], "allow_fallbacks": False}})

    def test_the_answer_is_the_text_with_the_model_name_the_host_reports_and_the_token_counts(self):
        answer = self._load(answer=openai_reply("  Melanoma \n")).predict([(b"x", "image/png")], "?")
        self.assertEqual(answer, {"text": "Melanoma", "finish_reason": "STOP", "model_version": "gpt-test-2026-01-01",
                                  "response_id": "chatcmpl-1",
                                  "usage": {"prompt_tokens": 900, "output_tokens": 3, "total_tokens": 903}})

    def test_the_back_end_that_answered_is_recorded_when_the_host_names_it(self):
        answer = self._load(answer=openai_reply(extra={"provider": "SomeHost"})).predict([(b"x", "image/png")], "?")
        self.assertEqual(answer["model_version"], "gpt-test-2026-01-01@SomeHost")

    def test_how_an_answer_ended(self):
        expected = {"stop": "STOP", "length": "MAX_TOKENS", "content_filter": "CONTENT_FILTER", "tool_calls": "TOOL_CALLS"}
        for finish, shown in expected.items():
            with self.subTest(finish):
                self.assertEqual(self._load(answer=openai_reply(finish=finish)).predict([(b"x", "image/png")], "?")["finish_reason"], shown)
        refused = self._load(answer=openai_reply(text=None, refusal="I can't help with that.")).predict([(b"x", "image/png")], "?")
        self.assertEqual((refused["text"], refused["finish_reason"]), (None, "REFUSAL"))
        empty = self._load(answer=openai_reply(text="")).predict([(b"x", "image/png")], "?")
        self.assertEqual((empty["text"], empty["finish_reason"]), (None, "STOP"))
        silent = self._load(answer=openai_reply(usage=False)).predict([(b"x", "image/png")], "?")["usage"]
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
            with self.subTest(repr(error)), self.assertRaises(models.TransientError):
                self._load(error=error).predict([(b"x", "image/png")], "?")
        stop = [OpenAIStatusError(429, "insufficient_quota", "You exceeded your current quota"),
                OpenAIStatusError(402, 402, "Insufficient credits"), OpenAIStatusError(429, 429, "Rate limit exceeded: free-models-per-day")]
        for error in stop:
            with self.subTest(repr(error)), self.assertRaisesRegex(models.QuotaExhausted, "run the same command again"):
                self._load(error=error).predict([(b"x", "image/png")], "?")
        for status, words in ((401, "OPENAI_API_KEY"), (403, "country"), (404, "does not know the model"), (400, "Fix the setting"),
                              (413, "Fix the setting")):
            with self.subTest(status), self.assertRaisesRegex(models.FatalError, words) as raised:
                self._load(error=OpenAIStatusError(status)).predict([(b"x", "image/png")], "?")
            self.assertEqual(raised.exception.code, status)
        with self.assertRaisesRegex(models.FatalError, "SDK failed"):
            self._load(error=OpenAIAPIError("odd")).predict([(b"x", "image/png")], "?")

    def test_a_refusal_names_the_provider_and_its_own_message_when_the_host_gives_them(self):
        body = {"error": {"message": "Provider returned error", "code": 400,
                          "metadata": {"provider_name": "Google AI Studio", "raw": "Content blocked: PROHIBITED_CONTENT"}}}
        for given in (body, body["error"]):  # the SDK may hand over the whole body or only its error part
            with self.subTest(given is body), self.assertRaisesRegex(
                    models.FatalError, r"Provider returned error \[Google AI Studio: Content blocked: PROHIBITED_CONTENT\]"):
                self._load(error=OpenAIStatusError(400, None, "Provider returned error", given)).predict([(b"x", "image/png")], "?")
        for given in (None, {}, {"error": {}}, {"error": {"metadata": "odd"}}, {"error": {"metadata": {}}}, "text"):
            with self.subTest(repr(given)), self.assertRaises(models.FatalError) as raised:
                self._load(error=OpenAIStatusError(400, None, "Provider returned error", given)).predict([(b"x", "image/png")], "?")
            self.assertNotIn("[", str(raised.exception))  # nothing more to say: the message is as before
        long = {"error": {"metadata": {"raw": "x" * 2000}}}
        with self.assertRaises(models.FatalError) as raised:
            self._load(error=OpenAIStatusError(400, None, "bad", long)).predict([(b"x", "image/png")], "?")
        self.assertLess(len(str(raised.exception)), 800)  # a long reply is cut

    def test_settings_are_checked_with_the_name_of_the_wrong_one(self):
        for key in ("id", "temperature", "seed", "max_tokens"):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"missing.*{key}"):
                models.load("openai_compatible", {k: v for k, v in OPENAI.items() if k != key})
        for bad in ({"id": ""}, {"temperature": -1}, {"temperature": 3}, {"temperature": True}, {"seed": 1.5}, {"seed": True},
                    {"max_tokens": 0}, {"max_tokens": "64"}, {"image_detail": "ultra"}, {"base_url": "host.example"},
                    {"api_key_env": ""}, {"extra_body": "x"}, {"pinned": "yes"}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._load(bad)

    def test_without_the_package_the_setup_is_explained(self):
        with mock.patch.dict(sys.modules, {"openai": None}), self.assertRaisesRegex(ValueError, "pip install -r requirements.txt"):
            models.load("openai_compatible", OPENAI)

    def test_a_picture_type_the_host_does_not_take_is_refused_before_any_request(self):
        adapter = self._load()
        with self.assertRaisesRegex(models.FatalError, "JPEG, PNG, GIF and WebP"):
            adapter.predict([(b"x", "image/tiff")], "?")
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
        return models.load("openai_compatible", settings | more)

    def test_the_real_sdk_sends_the_request_the_adapter_means_to_send(self):
        answer = self._model("stub-2026-01-01").predict([(b"\x89PNGdata", "image/png")], "Which?")
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
        self._model("stub-2026-01-01").predict([], "Case: “café” – which?")
        (seen,) = self.server.seen
        self.assertEqual(seen["body"]["messages"], [{"role": "user", "content": [{"type": "text", "text": "Case: “café” – which?"}]}])

    def test_extra_body_reaches_the_host_as_top_level_fields(self):
        self._model("stub-2026-01-01", extra_body={"provider": {"order": ["StubHost"], "allow_fallbacks": False}}).predict([(b"x", "image/png")], "?")
        self.assertEqual(self.server.seen[0]["body"]["provider"], {"order": ["StubHost"], "allow_fallbacks": False})

    def test_an_answer_cut_off_at_the_limit_says_so(self):
        answer = self._model("limit").predict([(b"x", "image/png")], "?")
        self.assertEqual((answer["text"], answer["finish_reason"]), ("Mela", "MAX_TOKENS"))

    def test_the_real_error_answers_are_sorted_correctly(self):
        for model_id, kind in (("quota", models.QuotaExhausted), ("daily", models.QuotaExhausted), ("credits", models.QuotaExhausted),
                               ("rate", models.TransientError), ("busy", models.TransientError),
                               ("key", models.FatalError), ("gone", models.FatalError)):
            with self.subTest(model_id), self.assertRaises(kind):
                self._model(model_id).predict([(b"x", "image/png")], "?")

    def test_a_refused_key_and_an_unknown_model_say_what_to_do(self):
        with self.assertRaisesRegex(models.FatalError, "STUB_KEY in .env") as raised:
            self._model("key").predict([(b"x", "image/png")], "?")
        self.assertEqual(raised.exception.code, 401)
        with self.assertRaisesRegex(models.FatalError, "Check the id and base_url") as raised:
            self._model("gone").predict([(b"x", "image/png")], "?")
        self.assertEqual(raised.exception.code, 404)

    def test_a_stalled_connection_times_out_instead_of_waiting_forever(self):
        with mock.patch.object(models, "TIMEOUT_SECONDS", 0.3), self.assertRaisesRegex(models.TransientError, "cannot reach the host"):
            self._model("slow").predict([(b"x", "image/png")], "?")

    def test_no_connection_is_a_transient_problem(self):
        model = self._model("stub-2026-01-01")
        self.server.shutdown()
        self.server.server_close()
        with self.assertRaisesRegex(models.TransientError, "cannot reach the host"):
            model.predict([(b"x", "image/png")], "?")


if __name__ == "__main__":
    unittest.main()
