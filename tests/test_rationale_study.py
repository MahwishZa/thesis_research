"""Tests of experiments/01_rationale_writers: its settings file and the runner that calls rationale_query's commands."""

import contextlib
import hashlib
import io
import json
import unittest

from tests.rationale_support import (
    STUDY_FOLDER,
    TempDirTestCase,
    load_study_runner,
    question_lines,
)
from tests.spec_targets import ROOT
from tests.test_rationale_models import pinned_sha256

ARCHIVED_RESULTS = ROOT / "_archive" / "rag2_adaptation_v1" / "experiments" / "medchange" / "results"

LLAMA3_FILE = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
QWEN_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"


run_study = load_study_runner()
SHA_A, SHA_B = "a" * 64, "b" * 64


def writer(name, **changes):
    base = {"name": name, "model_file": f"{name}.gguf", "sha256": SHA_A, "chat_format": "llama-3", "role": "a test writer"}
    return {**base, **changes}


class CommittedSettingsTests(unittest.TestCase):
    def test_the_settings_of_the_study_are_valid(self):
        settings = run_study.load_settings()
        self.assertEqual([w["name"] for w in settings["writers"]], ["llama3-8b-instruct-q4km", "qwen25-7b-instruct-q4km"])

    def test_the_baseline_writer_is_the_paper_model_in_the_llama3_format(self):
        first = run_study.load_settings()["writers"][0]
        self.assertEqual((first["model_file"], first["chat_format"]), (LLAMA3_FILE, "llama-3"))

    def test_the_second_writer_uses_the_template_of_its_own_file(self):
        second = run_study.load_settings()["writers"][1]
        self.assertEqual((second["model_file"], second["chat_format"]), (QWEN_FILE, "gguf"))

    def test_the_pinned_hashes_are_those_the_earlier_runs_recorded_for_the_same_files(self):
        if not ARCHIVED_RESULTS.is_dir():
            self.skipTest("the archived results are not in this copy of the repository")
        recorded: dict[str, set[str]] = {}
        for path in ARCHIVED_RESULTS.glob("*.config.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            name = config.get("model_file") or str(config.get("backend", "")).rpartition(":")[2]
            if name and config.get("model_sha256"):
                recorded.setdefault(name, set()).add(config["model_sha256"])
        for entry in run_study.load_settings()["writers"]:
            with self.subTest(writer=entry["name"]):
                self.assertEqual(recorded.get(entry["model_file"]), {entry["sha256"]})

    def test_the_model_checks_read_the_pins_of_the_same_file(self):
        for entry in run_study.load_settings()["writers"]:
            self.assertEqual(pinned_sha256(entry["model_file"]), entry["sha256"])
        self.assertIsNone(pinned_sha256("not-listed.gguf"))

    def test_the_folder_holds_what_the_experiment_convention_asks_for(self):
        for name in ("settings.json", "run_study.py", "NOTES.md"):
            self.assertTrue((STUDY_FOLDER / name).is_file(), name)


class LoadSettingsTests(TempDirTestCase):
    def load(self, data):
        text = data if isinstance(data, str) else json.dumps(data)
        return run_study.load_settings(self.write("settings.json", text))

    def test_a_valid_file_is_returned(self):
        data = {"writers": [writer("one"), writer("two", sha256=None, chat_format="gguf")]}
        self.assertEqual(self.load(data), data)

    def test_unusable_files_are_refused_with_a_message(self):
        good = writer("one")
        cases = {
            "not json": "{nope",
            "not an object": "[1]",
            "no writers": {"writers": []},
            "writers not a list": {"writers": "one"},
            "missing name": {"writers": [{k: v for k, v in good.items() if k != "name"}]},
            "missing file": {"writers": [{k: v for k, v in good.items() if k != "model_file"}]},
            "empty chat format": {"writers": [writer("one", chat_format="")]},
            "unknown chat format": {"writers": [writer("one", chat_format="chatml")]},
            "name with a slash": {"writers": [writer("a/b")]},
            "name with a space": {"writers": [writer("a b")]},
            "name starting with a dot": {"writers": [writer(".hidden")]},
            "name twice": {"writers": [writer("one"), writer("one", model_file="x.gguf")]},
            "short hash": {"writers": [writer("one", sha256="abc")]},
            "capital hash": {"writers": [writer("one", sha256="A" * 64)]},
            "hash not text": {"writers": [writer("one", sha256=5)]},
        }
        for label, data in cases.items():
            with self.subTest(case=label), self.assertRaises(run_study.StudyError):
                self.load(data)

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(run_study.StudyError):
            run_study.load_settings(self.dir / "absent.json")


class ChooseWritersTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.settings = {"writers": [writer("one"), writer("two"), writer("three")]}

    def have(self, *names):
        for name in names:
            self.write(f"{name}.gguf", "model")

    def test_without_names_every_writer_with_a_file_runs_and_the_others_are_passed_over(self):
        self.have("one", "three")
        chosen, skipped = run_study.choose_writers(self.settings, None, self.dir)
        self.assertEqual([w["name"] for w in chosen], ["one", "three"])
        self.assertEqual([w["name"] for w in skipped], ["two"])

    def test_named_writers_run_in_the_order_given_without_repeats(self):
        self.have("one", "two", "three")
        chosen, skipped = run_study.choose_writers(self.settings, ["three", "one", "three"], self.dir)
        self.assertEqual([w["name"] for w in chosen], ["three", "one"])
        self.assertEqual(skipped, [])

    def test_a_named_writer_that_is_not_in_the_settings_is_refused(self):
        self.have("one")
        with self.assertRaises(run_study.StudyError) as caught:
            run_study.choose_writers(self.settings, ["one", "nine"], self.dir)
        self.assertIn("nine", str(caught.exception))

    def test_a_named_writer_without_its_file_is_refused_not_passed_over(self):
        self.have("one")
        with self.assertRaises(run_study.StudyError) as caught:
            run_study.choose_writers(self.settings, ["one", "two"], self.dir)
        self.assertIn("two.gguf", str(caught.exception))

    def test_no_file_at_all_is_refused_and_the_names_are_listed(self):
        with self.assertRaises(run_study.StudyError) as caught:
            run_study.choose_writers(self.settings, None, self.dir)
        self.assertIn("one.gguf", str(caught.exception))
        self.assertIn(str(self.dir), str(caught.exception))


class GenerateCommandTests(TempDirTestCase):
    def test_the_command_names_everything_the_writer_needs(self):
        argv = run_study.generate_command(
            writer("one"), questions=self.dir / "q.jsonl", out=self.dir / "out", models_dir=self.dir / "models", limit=None
        )
        self.assertEqual(
            argv,
            [
                "generate", "--questions", str(self.dir / "q.jsonl"), "--out", str(self.dir / "out"),
                "--model", str(self.dir / "models" / "one.gguf"), "--model-name", "one",
                "--chat-format", "llama-3", "--model-sha256", SHA_A,
            ],
        )  # fmt: skip

    def test_a_writer_without_a_pinned_hash_gets_no_hash_check(self):
        argv = run_study.generate_command(
            writer("one", sha256=None), questions=self.dir / "q", out=self.dir / "o", models_dir=self.dir, limit=None
        )
        self.assertNotIn("--model-sha256", argv)

    def test_the_template_switch_is_passed_on_only_when_given(self):
        kwargs = {"questions": self.dir / "q", "out": self.dir / "o", "models_dir": self.dir, "limit": None}
        self.assertNotIn("--allow-template-mismatch", run_study.generate_command(writer("one"), **kwargs))
        argv = run_study.generate_command(writer("one"), allow_template_mismatch=True, **kwargs)
        self.assertEqual(argv[-1], "--allow-template-mismatch")

    def test_a_limit_is_passed_on(self):
        argv = run_study.generate_command(writer("one"), questions=self.dir / "q", out=self.dir / "o", models_dir=self.dir, limit=7)
        self.assertEqual(argv[-2:], ["--limit", "7"])


class MainTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.models = self.dir / "models"
        self.models.mkdir()
        for name in ("one", "two"):
            (self.models / f"{name}.gguf").write_bytes(b"model")
        self.questions = self.write("my questions.jsonl", question_lines())
        self.settings = self.write("settings.json", json.dumps({"writers": [writer("one"), writer("two", sha256=SHA_B)]}))
        self.calls = []

    def fake_cli(self, status_for=None):
        def cli_main(argv, *, streams=None):
            self.calls.append(argv)
            if argv[0] == "report" and streams is not None:
                streams.out.write("THE TABLE\n")
            return (status_for or {}).get(argv[0], 0) if not callable(status_for) else status_for(argv)

        return cli_main

    def run_main(self, *extra, cli_main=None, with_settings=True):
        argv = ["--questions", str(self.questions), "--models-dir", str(self.models)]
        argv += ["--outputs-dir", str(self.dir / "outputs"), "--results-dir", str(self.dir / "results")]
        if with_settings:
            argv += ["--settings", str(self.settings)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = run_study.main([*argv, *extra], cli_main=cli_main or self.fake_cli())
        return status, out.getvalue(), err.getvalue()

    def test_every_writer_writes_then_the_writers_are_compared(self):
        status, out, _ = self.run_main()
        self.assertEqual(status, 0)
        self.assertEqual([c[0] for c in self.calls], ["generate", "generate", "report"])
        folders = [self.dir / "outputs" / "my-questions" / "one", self.dir / "outputs" / "my-questions" / "two"]
        self.assertEqual([c[c.index("--out") + 1] for c in self.calls[:2]], [str(f) for f in folders])
        self.assertEqual(self.calls[2][1:3], [str(f) for f in folders])
        self.assertEqual(self.calls[2][3:], ["--json", str(self.dir / "results" / "my-questions" / "report.json")])
        self.assertIn("THE TABLE", out)

    def test_each_writer_is_started_with_its_pinned_hash_and_format(self):
        self.run_main()
        first, second = self.calls[:2]
        self.assertEqual((first[first.index("--model-sha256") + 1], first[first.index("--model-name") + 1]), (SHA_A, "one"))
        self.assertEqual((second[second.index("--model-sha256") + 1], second[second.index("--model-name") + 1]), (SHA_B, "two"))

    def test_the_run_says_which_questions_and_writers_the_report_is_about(self):
        self.run_main("--limit", "2", "--encode")
        saved = json.loads((self.dir / "results" / "my-questions" / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["questions_file"], "my questions.jsonl")
        self.assertEqual(saved["questions_sha256"], hashlib.sha256(self.questions.read_bytes()).hexdigest())
        self.assertEqual((saved["questions_used"], saved["encoded"]), (2, True))
        self.assertEqual([w["name"] for w in saved["writers"]], ["one", "two"])
        self.assertEqual([w["sha256"] for w in saved["writers"]], [SHA_A, SHA_B])
        self.assertEqual(set(saved["writers"][0]), {"name", "model_file", "sha256", "chat_format"})
        self.run_main()
        self.assertEqual(json.loads((self.dir / "results" / "my-questions" / "run.json").read_text(encoding="utf-8"))["questions_used"], 3)

    def test_the_report_is_saved_as_text(self):
        self.run_main()
        self.assertEqual((self.dir / "results" / "my-questions" / "report.txt").read_text(encoding="utf-8"), "THE TABLE\n")

    def test_the_template_switch_reaches_every_writer(self):
        self.run_main("--allow-template-mismatch")
        generates = [c for c in self.calls if c[0] == "generate"]
        self.assertEqual(len(generates), 2)
        self.assertTrue(all("--allow-template-mismatch" in c for c in generates))
        self.calls.clear()
        self.run_main()
        self.assertTrue(all("--allow-template-mismatch" not in c for c in self.calls))

    def test_the_name_of_the_run_can_be_chosen(self):
        self.run_main("--run-name", "trial.1")
        self.assertIn(str(self.dir / "outputs" / "trial.1" / "one"), self.calls[0])
        self.assertTrue((self.dir / "results" / "trial.1" / "report.txt").is_file())

    def test_a_bad_name_of_the_run_is_refused_before_anything_runs(self):
        for name in ("a/b", "..", "", "a b"):
            with self.subTest(name=name):
                status, _, err = self.run_main("--run-name", name)
                self.assertEqual(status, 1)
                self.assertIn("--run-name", err)
        self.assertEqual(self.calls, [])

    def test_a_writer_without_its_file_is_passed_over_and_the_others_run(self):
        (self.models / "two.gguf").unlink()
        status, out, _ = self.run_main()
        self.assertEqual(status, 0)
        self.assertIn("passed over two", out)
        self.assertEqual([c[0] for c in self.calls], ["generate", "report"])

    def test_no_file_at_all_is_an_error_and_nothing_runs(self):
        for path in self.models.iterdir():
            path.unlink()
        status, _, err = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("none of the model files", err)
        self.assertEqual(self.calls, [])

    def test_only_the_named_writers_run(self):
        status, _, _ = self.run_main("--writers", "two")
        self.assertEqual(status, 0)
        self.assertEqual([c[c.index("--model-name") + 1] for c in self.calls if c[0] == "generate"], ["two"])

    def test_an_unknown_writer_is_an_error(self):
        status, _, err = self.run_main("--writers", "nine")
        self.assertEqual(status, 1)
        self.assertIn("unknown writer nine", err)

    def test_encoding_follows_each_writer_and_overwrites_old_vectors(self):
        self.run_main("--encode", "--limit", "2")
        self.assertEqual([c[0] for c in self.calls], ["generate", "encode", "generate", "encode", "report"])
        encode = self.calls[1]
        self.assertEqual(encode[:5], ["encode", "--rationales", str(self.dir / "outputs" / "my-questions" / "one"), "--questions", str(self.questions)])
        self.assertIn("--overwrite", encode)
        self.assertEqual(encode[-2:], ["--limit", "2"])

    def test_the_first_step_that_fails_stops_the_run_and_says_how_to_go_on(self):
        status, _, err = self.run_main(cli_main=self.fake_cli({"generate": 2}))
        self.assertEqual(status, 2)
        self.assertEqual([c[0] for c in self.calls], ["generate"])
        self.assertIn("one: write the rationales", err)
        self.assertIn("run the same command again", err)
        self.assertFalse((self.dir / "results").exists())

    def test_a_step_with_a_wrong_setup_does_not_suggest_to_run_it_again(self):
        status, _, err = self.run_main(cli_main=self.fake_cli({"generate": 1}))
        self.assertEqual(status, 1)
        self.assertIn("the message above says what to change", err)
        self.assertNotIn("run the same command again", err)

    def test_an_interrupt_is_passed_on_and_can_be_continued(self):
        status, _, err = self.run_main(cli_main=self.fake_cli({"generate": 130}))
        self.assertEqual(status, 130)
        self.assertIn("run the same command again", err)

    def test_a_failing_report_gives_its_status_and_writes_no_text_file(self):
        status, _, _ = self.run_main(cli_main=self.fake_cli({"report": 1}))
        self.assertEqual(status, 1)
        self.assertFalse((self.dir / "results" / "my-questions" / "report.txt").exists())

    def test_the_settings_of_the_repository_are_used_by_default(self):
        (self.models / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf").write_bytes(b"model")
        status, _, _ = self.run_main(with_settings=False)
        self.assertEqual(status, 0)
        first = self.calls[0]
        self.assertEqual(first[first.index("--model-name") + 1], "llama3-8b-instruct-q4km")
        self.assertEqual(first[first.index("--model-sha256") + 1], pinned_sha256("Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"))


if __name__ == "__main__":
    unittest.main()
