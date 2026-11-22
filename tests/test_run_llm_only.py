"""Checks of src/run_llm_only.py without any API call: a fake model answers, and the three request formats are checked as data."""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate as ev  # noqa: E402
import run_llm_only as rl  # noqa: E402
from build_views import question  # noqa: E402
from common import question_file, write_jsonl  # noqa: E402


def cases(n, view_has_image):
    labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    return [{"isic_id": f"ISIC_{i:07d}", "split": "val", "label": labels[i % 8], "group_id": f"ISIC_{i:07d}",
             "clinical_text": "Patient: 50-year-old male. Skin lesion on the trunk.", "image_path": f"processed_images/ISIC_{i:07d}.jpg"}
            for i in range(n)]


class FakeClient:
    """Answers right for every second question, wrong for the others, with no letter for the third, and can fail on demand."""
    def __init__(self, questions, fail_at=None, refuse_at=None):
        self.by_text = {rl.user_text(q): q for q in questions}
        self.calls, self.images, self.fail_at, self.refuse_at = 0, [], fail_at, refuse_at

    def ask(self, system, text, image):
        self.calls += 1
        if self.fail_at and self.calls == self.fail_at:
            raise rl.ApiError("HTTP 402: no credit")
        if self.refuse_at and self.calls == self.refuse_at:
            raise rl.Refused("refused by the host's content moderation")
        self.images.append(image)
        q = self.by_text[text]
        k = int(q["id"][-3:])
        if k == 2:
            return {"text": "I cannot help with that.", "finish_reason": "refusal", "tokens_in": 10, "tokens_out": 5}
        letter = q["answer"] if k % 2 == 0 else next(x for x in q["options"] if x != q["answer"])
        others = [x for x in q["options"] if x not in (letter, q["answer"])][:2]
        if k == 1:
            others = [q["answer"], others[0]]        # wrong first, right second: right in the top 3
        flag = "yes" if k % 3 else "no"
        return {"text": f"Some reasoning.\nAnswer: {letter}, {others[0]}, {others[1]}\nEvidence sufficient: {flag}",
                "finish_reason": "stop", "tokens_in": 100, "tokens_out": 20}


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.results = os.path.join(self.tmp.name, "data"), os.path.join(self.tmp.name, "results")
        self.make("text_only", 6)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, view, n, images=True):
        self.questions = [question(c, view) for c in cases(n, view != "text_only")]
        write_jsonl(os.path.join(self.data, question_file(view, "val")), self.questions)
        if images and view != "text_only":
            for q in self.questions:
                Path(self.data, q["image_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(self.data, q["image_path"]).write_bytes(b"\xff\xd8\xff fake jpeg")

    def args(self, view="text_only", **kw):
        base = dict(model="gpt-4o", view=view, split="val", style="rationale", limit=0, sleep=0.0, dry_run=False,
                    env_file=os.path.join(self.tmp.name, ".env"), data=self.data, results=self.results)
        base.update(kw)
        return argparse.Namespace(**base)

    def go(self, args, client):
        log = rl.get_logger(f"test.{id(self)}.{args.view}")
        with contextlib.redirect_stdout(io.StringIO()):               # keep the test output quiet
            return rl.run(args, client, log)

    def folder(self, view="text_only", partial=False):
        final = os.path.join(self.results, "llm_only", "gpt-4o", f"{view}_val")
        return final + ".partial" if partial else final

    def test_a_full_run_saves_every_answer_and_ends_complete(self):
        out = self.go(self.args(), FakeClient(self.questions))
        self.assertEqual(out, self.folder())
        self.assertFalse(os.path.exists(self.folder(partial=True)))
        preds = rl.read_jsonl(os.path.join(out, "predictions.jsonl"))
        self.assertEqual([p["id"] for p in preds], [q["id"] for q in self.questions])
        info = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))
        self.assertEqual((info["status"], info["answered"], info["of"]), ("complete", 6, 6))
        self.assertEqual(info["settings"]["prompt_version"], rl.PROMPT_VERSION)
        refused = next(p for p in preds if p["id"].endswith("002"))
        self.assertEqual((refused["pred"], refused["valid"], refused["finish_reason"]), (None, False, "refusal"))
        first = preds[0]
        self.assertEqual((len(first["top3"]), first["pred"], first["evidence_sufficient"]), (3, first["top3"][0], "no"))

    def test_the_predictions_go_straight_into_evaluate(self):
        out = self.go(self.args(), FakeClient(self.questions))
        qa = {q["id"]: q for q in self.questions}
        pred = ev.load(os.path.join(out, "predictions.jsonl"))
        d = ev.diagnosis_metrics(qa, pred)
        self.assertEqual((d["accuracy"], d["invalid_answers"]), (0.3333, 1))    # ids 0 and 4 right; 2 has no letter
        self.assertEqual(d["top3_accuracy"], 0.5)                                  # ids 0, 4 first; id 1 second
        self.assertEqual((d["n_covered"], d["coverage"]), (3, 0.5))                # ids 1, 4, 5 say yes (k % 3); 2 is invalid
        self.assertEqual(d["selective_risk"], 0.6667)                              # ids 1 and 5 are wrong

    def test_an_interrupted_run_continues_where_it_stopped_without_repeating(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        self.assertTrue(os.path.isdir(self.folder(partial=True)))
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "unfinished")
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(client.calls, 4)                                       # only the 4 left
        preds = rl.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))
        self.assertEqual(len(preds), 6)
        self.assertEqual(len({p["id"] for p in preds}), 6)

    def test_an_api_error_stops_the_run_and_keeps_what_was_answered(self):
        with self.assertRaises(SystemExit):
            self.go(self.args(), FakeClient(self.questions, fail_at=4))
        preds = rl.read_jsonl(os.path.join(self.folder(partial=True), "predictions.jsonl"))
        self.assertEqual(len(preds), 3)
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "stopped")
        self.go(self.args(), FakeClient(self.questions))
        self.assertEqual(len(rl.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))), 6)

    def test_an_unfinished_run_is_not_continued_with_other_settings(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        with self.assertRaises(SystemExit):
            self.go(self.args(style="answer_only"), FakeClient(self.questions))

    def test_a_question_refused_by_the_host_is_saved_as_invalid_and_the_run_goes_on(self):
        self.go(self.args(), FakeClient(self.questions, refuse_at=2))
        preds = rl.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))
        self.assertEqual(len(preds), 6)
        refused = [p for p in preds if str(p["finish_reason"]).startswith("refused by host")]
        self.assertEqual(len(refused), 1)
        self.assertEqual((refused[0]["pred"], refused[0]["valid"]), (None, False))

    def test_a_complete_run_is_not_asked_again(self):
        self.go(self.args(), FakeClient(self.questions))
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(client.calls, 0)

    def test_image_views_send_the_image_and_the_text_view_does_not(self):
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(set(client.images), {None})
        self.make("multimodal", 3)
        client = FakeClient(self.questions)
        self.go(self.args(view="multimodal"), client)
        self.assertTrue(all(img and img.startswith("/9j") for img in client.images))   # base64 of a JPEG start

    def test_missing_images_stop_the_run_before_any_question(self):
        self.make("image_only", 3, images=False)
        client = FakeClient(self.questions)
        with self.assertRaises(SystemExit):
            self.go(self.args(view="image_only"), client)
        self.assertEqual(client.calls, 0)

    def test_a_dry_run_asks_nothing_and_writes_nothing(self):
        client = FakeClient(self.questions)
        self.go(self.args(dry_run=True), client)
        self.assertEqual(client.calls, 0)
        self.assertFalse(os.path.exists(self.results))

    def test_the_api_key_is_needed_and_never_written(self):
        argv = ["--model", "gpt-4o", "--view", "text_only", "--data", self.data, "--results", self.results,
                "--env-file", os.path.join(self.tmp.name, ".env")]
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit), \
                contextlib.redirect_stdout(io.StringIO()):
            rl.main(argv)
        self.assertFalse(os.path.exists(self.results))                  # nothing is left behind
        Path(self.tmp.name, ".env").write_text("# keys\nOPENROUTER_API_KEY = 'SECRET123'\n", encoding="utf-8")
        fake = FakeClient(self.questions)
        made = []

        def client(cfg, key, log=None):
            made.append(key)
            return fake

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(rl, "OpenRouterClient", client), \
                contextlib.redirect_stdout(io.StringIO()):
            rl.main(argv)
        self.assertEqual(made, ["SECRET123"])                         # read from .env, quotes removed
        written = "".join(p.read_text(encoding="utf-8") for p in Path(self.results).rglob("*") if p.is_file())
        self.assertIn("gpt-4o", written)
        self.assertNotIn("SECRET123", written)

    def test_claude_takes_its_key_from_either_name(self):
        with mock.patch.dict(os.environ, {"CLAUDE_API_KEY": "K1"}, clear=True):
            self.assertEqual(rl.api_key("anthropic"), "K1")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "K2", "CLAUDE_API_KEY": "K1"}, clear=True):
            self.assertEqual(rl.api_key("anthropic"), "K2")


class PromptAndAnswerTests(unittest.TestCase):
    def test_the_prompt_lists_the_question_the_eight_options_and_one_instruction(self):
        q = question(cases(1, False)[0], "text_only")
        text = rl.user_text(q)
        self.assertTrue(text.startswith(q["question"]))
        for letter, option in q["options"].items():
            self.assertIn(f"\n{letter}. {option}", text)
        self.assertTrue(text.endswith(rl.INSTRUCTIONS["rationale"]))
        self.assertNotIn(q["gold_label"], text.replace(q["options"][q["answer"]], ""))   # the code of the class is not shown

    def test_the_answer_rule(self):
        cases_ = {"Reasoning.\nAnswer: C": "C", "**Answer:** (D)": "D", "answer: b": None, "Answer: Actinic keratosis": None,
                  "Answer: A\nWait, Answer: B": "B", "E": "E", "(F).": "F", "The answer is probably G": None, "": None,
                  "Answer: Z": None}
        for text, letter in cases_.items():
            with self.subTest(text):
                self.assertEqual(rl.parse_answer(text)["pred"], letter)
        self.assertEqual(rl.parse_answer("Pigment network.\n**Answer:** C, A, F\nEvidence sufficient: Yes"),
                         {"pred": "C", "top3": ["C", "A", "F"], "sufficient": "yes", "rationale": "Pigment network."})
        self.assertEqual(rl.parse_answer("Answer: B, B, E, H\nEvidence sufficient: no")["top3"], ["B", "E", "H"])  # no repeats
        self.assertEqual(rl.parse_answer("Answer: C (Melanoma), A, F, H")["top3"], ["C", "A", "F"])           # at most three
        self.assertIsNone(rl.parse_answer("Answer: C")["sufficient"])
        self.assertEqual(rl.parse_answer("Evidence sufficient: yes\nOn reflection:\nAnswer: C\nEvidence sufficient: no")["sufficient"],
                         "no")                                                                    # the last flag counts
        self.assertEqual(rl.parse_answer("Answer: Z, Q\nEvidence sufficient: yes")["top3"], [])            # no option letter


class RequestTests(unittest.TestCase):
    def test_the_four_research_models_and_their_fixed_settings(self):
        self.assertEqual(sorted(rl.MODELS), ["claude-sonnet-5-5", "gemini-3.8-flash", "gpt-4o", "llama-4-maverick"])
        for name, cfg in rl.MODELS.items():
            with self.subTest(name):
                if cfg["api"] == "anthropic":
                    self.assertNotIn("temperature", cfg)                 # Claude rejects a temperature
                    self.assertEqual(cfg["thinking"], {"type": "between_tools"})  # no hidden reasoning
                else:
                    self.assertEqual((cfg["temperature"], cfg["seed"]), (0.0, 0))
                    self.assertEqual(len(cfg["route"]), 1)              # one pinned host

    def test_claude_request(self):
        p = rl.anthropic_params(rl.MODELS["claude-sonnet-5-5"], "sys", "txt", "IMG")
        self.assertEqual(p["model"], "claude-sonnet-5-5")
        self.assertEqual(p["system"], "sys")
        self.assertEqual(p["messages"][0]["content"][0]["source"], {"type": "base64", "media_type": "image/jpeg", "data": "IMG"})
        self.assertEqual(p["messages"][0]["content"][1], {"type": "text", "text": "txt"})
        self.assertEqual((p["thinking"], p["output_config"]), ({"type": "between_tools"}, {"effort": "low"}))
        self.assertNotIn("temperature", p)
        self.assertEqual(len(rl.anthropic_params(rl.MODELS["claude-sonnet-5-5"], "s", "t", None)["messages"][0]["content"]), 1)

    def test_claude_reply_and_refusal(self):
        ns = types.SimpleNamespace
        r = ns(content=[ns(type="thinking", text=""), ns(type="text", text="Answer: B")], stop_reason="end_turn",
               stop_details=None, model="claude-sonnet-5-5", usage=ns(input_tokens=7, output_tokens=3))
        self.assertEqual(rl.anthropic_reply(r), {"text": "Answer: B", "finish_reason": "end_turn", "served_by": "claude-sonnet-5-5",
                                                 "tokens_in": 7, "tokens_out": 3})
        r = ns(content=[], stop_reason="refusal", stop_details=ns(category="bio"), model="m", usage=ns(input_tokens=1, output_tokens=0))
        self.assertEqual(rl.anthropic_reply(r)["finish_reason"], "refusal: bio")

    def test_openrouter_request_pins_the_host(self):
        url, body = rl.openrouter_request(rl.MODELS["gpt-4o"], "sys", "txt", "IMG")
        self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(body["model"], "openai/gpt-4o-2024-08-06")
        self.assertEqual(body["provider"], {"order": ["openai"], "allow_fallbacks": False})
        self.assertEqual(body["messages"][1]["content"][1]["image_url"], {"url": "data:image/jpeg;base64,IMG", "detail": "high"})
        self.assertEqual((body["temperature"], body["seed"], body["max_tokens"]), (0.0, 0, 1024))
        _, body = rl.openrouter_request(rl.MODELS["gemini-3.8-flash"], "sys", "txt", None)
        self.assertEqual((body["reasoning"], len(body["messages"][1]["content"])), ({"effort": "minimal"}, 1))
        _, body = rl.openrouter_request(rl.MODELS["llama-4-maverick"], "sys", "txt", "IMG")
        self.assertNotIn("detail", body["messages"][1]["content"][1]["image_url"])
        self.assertNotIn("reasoning", body)

    def test_openrouter_replies(self):
        ok = rl.openrouter_reply({"model": "openai/gpt-4o-2024-08-06", "provider": "OpenAI",
                                  "choices": [{"message": {"content": "Answer: A"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
        self.assertEqual(ok, {"text": "Answer: A", "finish_reason": "stop", "served_by": "openai/gpt-4o-2024-08-06@OpenAI",
                              "tokens_in": 5, "tokens_out": 2})
        self.assertEqual(rl.openrouter_reply({"choices": [{"message": {"content": None, "refusal": "no"}}]})["finish_reason"],
                         "refusal")
        with self.assertRaises(rl.Refused):
            rl.openrouter_reply({"error": {"code": 403, "message": "Input was flagged by moderation"}})
        with self.assertRaises(rl.ApiError):
            rl.openrouter_reply({"error": {"code": 402, "message": "Insufficient credits"}})

    def test_the_claude_client_uses_the_official_package_and_stops_on_an_api_error(self):
        ns = types.SimpleNamespace
        sent = {}

        class StatusError(Exception):
            def __init__(self, status, message):
                super().__init__(message)
                self.status_code, self.message = status, message

        class Messages:
            def create(self, **kw):
                sent.update(kw)
                if kw["system"] == "fail":
                    raise StatusError(401, "invalid x-api-key")
                return ns(content=[ns(type="text", text="Answer: C")], stop_reason="end_turn", stop_details=None,
                          model="claude-sonnet-5-5", usage=ns(input_tokens=9, output_tokens=4))

        def make(api_key, max_retries, timeout):
            sent["client"] = (api_key, max_retries, timeout)
            return ns(messages=Messages())

        fake = ns(Anthropic=make, APIStatusError=StatusError, APIConnectionError=ConnectionError)
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            client = rl.AnthropicClient(rl.MODELS["claude-sonnet-5-5"], "KEY")
            self.assertEqual(client.ask("sys", "txt", None)["text"], "Answer: C")
            self.assertEqual(sent["client"], ("KEY", 6, 180.0))
            self.assertEqual(sent["model"], "claude-sonnet-5-5")
            with self.assertRaises(rl.ApiError):
                client.ask("fail", "txt", None)

    def test_load_env_reads_keys_without_overriding_the_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, ".env")
            path.write_text("# comment\nA_KEY=\"one\"\nB_KEY = two\nbroken line\nEMPTY=\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"B_KEY": "terminal"}, clear=True):
                rl.load_env(path)
                self.assertEqual((os.environ["A_KEY"], os.environ["B_KEY"]), ("one", "terminal"))
                self.assertNotIn("EMPTY", os.environ)
            rl.load_env(Path(d, "missing.env"))                       # a missing file is fine

    def test_busy_servers_are_retried_and_other_errors_stop_at_once(self):
        class Answer(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def error(code):
            return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b"detail"))

        calls = []

        def busy_then_ok(req, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise error(429)
            return Answer(b'{"ok": true}')

        with mock.patch("urllib.request.urlopen", busy_then_ok), mock.patch("time.sleep"):
            self.assertEqual(rl.post_json("https://x", {}, {}), {"ok": True})
        self.assertEqual(len(calls), 2)
        with mock.patch("urllib.request.urlopen", side_effect=error(400)) as bad, mock.patch("time.sleep"):
            with self.assertRaises(rl.ApiError):
                rl.post_json("https://x", {}, {})
        self.assertEqual(bad.call_count, 1)                           # a bad request is not sent again
        moderated = urllib.error.HTTPError("u", 403, "x", {}, io.BytesIO(b'{"error": "flagged by moderation"}'))
        with mock.patch("urllib.request.urlopen", side_effect=moderated), mock.patch("time.sleep"):
            with self.assertRaises(rl.Refused):
                rl.post_json("https://x", {}, {})


if __name__ == "__main__":
    unittest.main()
