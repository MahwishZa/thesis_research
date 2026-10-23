"""rationale_filter.cli: the label and labels commands end to end, with a scripted generator and scorer in place of llama.cpp."""

import io
import json
import unittest

from rationale_filter import cli
from rationale_filter.label_records import PairStore
from rationale_query.generator import GenerationResult
from tests.support import TempDirTestCase
from tests.unit.rationale_filter.test_labelling import Generator, Scorer, write_inputs

RIGHT = "Therefore, the answer is (A) Donepezil."
WRONG = "Therefore, the answer is (B) Aspirin."


class Models(Generator):
    """Answers a snippet-1 prompt rightly and every other prompt wrongly, so that the labels differ."""

    def __init__(self):
        super().__init__([RIGHT])

    def generate(self, user_message):
        self.prompts.append(user_message)
        text = RIGHT if "snippet 1" in user_message else WRONG
        return GenerationResult(text=text, finish_reason="stop", prompt_tokens=50, completion_tokens=9)


class CliTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        write_inputs(self)
        self.questions_file = self.write("questions.jsonl", "".join(json.dumps({"id": q.id, "question": q.text}) + "\n" for q in self.questions))

    def run_command(self, command, *extra, generator=None, scorer=None):
        out, err = io.StringIO(), io.StringIO()
        arguments = [
            command, "--questions", str(self.questions_file), "--answers", str(self.answers), "--rationales", str(self.dir / "rationales"),
            "--retrieved", str(self.dir / "retrieved"), "--out", str(self.dir / "pairs"), "--snippets", "2",
        ]
        if command == "label":
            model = self.write("model.gguf", "x")
            arguments += ["--dataset", "medqa", "--split", "train", "--model", str(model), "--quiet"]
        status = cli.main(
            arguments + list(extra), generator_factory=lambda args, streams: generator or Models(),
            scorer_factory=lambda args, streams: scorer or Scorer(), streams=cli.Streams(out, err),
        )
        return status, out.getvalue(), err.getvalue()

    def test_label_writes_one_record_per_pair_and_reports_the_time_per_pair(self):
        status, out, err = self.run_command("label")
        self.assertEqual((status, err), (0, ""))
        self.assertIn("4 pairs: 0 already done, 4 written, 0 failed", out)
        self.assertIn("s per pair", out)
        self.assertEqual(sorted(PairStore(self.dir / "pairs").load()), ["q1#1", "q1#2", "q2#1", "q2#2"])

    def test_a_second_run_does_nothing_and_a_limit_cuts_the_questions(self):
        self.run_command("label", "--limit", "1")
        self.assertEqual(sorted(PairStore(self.dir / "pairs").load()), ["q1#1", "q1#2"])
        status, out, _ = self.run_command("label")
        self.assertIn("2 already done, 2 written", out)
        status, out, _ = self.run_command("label")
        self.assertIn("4 already done, 0 written", out)
        self.assertNotIn("per pair", out)

    def test_the_config_binds_the_dataset_split_and_snippet_count(self):
        self.run_command("label")
        config = PairStore(self.dir / "pairs").read_config()
        self.assertEqual((config["dataset"], config["split"], config["snippets_per_question"]), ("medqa", "train", 2))
        self.assertEqual(config["generator"]["model_sha256"], "m" * 64)
        self.assertIn("reader_pattern", config)
        status, out, err = self.run_command("label", "--snippets", "3")
        self.assertEqual(status, 1)
        self.assertIn("snippets_per_question", err)

    def test_the_label_command_refuses_a_test_split(self):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(
            ["label", "--questions", str(self.questions_file), "--answers", str(self.answers), "--rationales", str(self.dir / "rationales"),
             "--retrieved", str(self.dir / "retrieved"), "--out", str(self.dir / "pairs"), "--dataset", "medqa", "--split", "test",
             "--model", str(self.write("m.gguf", "x"))],
            generator_factory=lambda a, s: Models(), scorer_factory=lambda a, s: Scorer(), streams=cli.Streams(out, err),
        )
        self.assertEqual(status, 1)
        self.assertIn("never from a test split", err.getvalue())
        self.assertFalse((self.dir / "pairs").exists())

    def test_failed_pairs_give_exit_status_2_and_are_tried_again(self):
        generator = Models()
        failing = Generator([RuntimeError("boom"), RIGHT, RIGHT, RIGHT])
        status, out, _ = self.run_command("label", generator=failing)
        self.assertEqual(status, 2)
        self.assertIn("failed q1#1", out)
        status, out, _ = self.run_command("label", generator=generator)
        self.assertEqual(status, 0)
        self.assertIn("3 already done, 1 written", out)

    def test_a_scorer_that_cannot_be_loaded_closes_the_generator(self):
        closed = []

        class Closing(Models):
            def close(self):
                closed.append(True)

        def broken(args, streams):
            raise ValueError("no such model")

        out, err = io.StringIO(), io.StringIO()
        model = self.write("model.gguf", "x")
        status = cli.main(
            ["label", "--questions", str(self.questions_file), "--answers", str(self.answers), "--rationales", str(self.dir / "rationales"),
             "--retrieved", str(self.dir / "retrieved"), "--out", str(self.dir / "pairs"), "--dataset", "medqa", "--split", "train",
             "--model", str(model)],
            generator_factory=lambda a, s: Closing(), scorer_factory=broken, streams=cli.Streams(out, err),
        )
        self.assertEqual((status, closed), (1, [True]))

    def test_labels_prints_tau_the_counts_and_writes_the_training_file(self):
        self.run_command("label")
        target = self.dir / "training.json"
        status, out, err = self.run_command("labels", "--training-file", str(target))
        self.assertEqual((status, err), (0, ""))
        report = json.loads(out)
        self.assertEqual(report["dataset"], "medqa")
        self.assertEqual(report["status"], {"ok": 4, "skipped": 0, "error": 0})
        self.assertIn("tau", report)
        self.assertEqual(report["training_file"]["path"], str(target))
        rows = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), report["training_file"]["records"])
        self.assertTrue(all(row["dataset_name"] == "medqa" for row in rows))
        self.assertTrue(all(list(row) == ["id", "answer", "dataset_name", "question"] for row in rows))

    def test_labels_never_overwrites_a_training_file(self):
        self.run_command("label")
        target = self.write("training.json", "keep")
        status, out, err = self.run_command("labels", "--training-file", str(target))
        self.assertEqual(status, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_labels_without_records_says_to_run_label_first(self):
        status, out, err = self.run_command("labels")
        self.assertEqual(status, 1)
        self.assertIn("run the label command first", err)

    def test_labels_leaves_out_pairs_that_have_no_finished_record_and_says_so(self):
        self.run_command("label", "--limit", "1")
        status, out, err = self.run_command("labels")
        self.assertEqual(status, 0)
        self.assertIn("2 pairs have no finished record yet", err)
        self.assertEqual(json.loads(out)["status"]["ok"], 2)

    def test_missing_arguments_exit_with_2(self):
        with self.assertRaises(SystemExit) as caught:
            cli.main(["label"], streams=cli.Streams(io.StringIO(), io.StringIO()))
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
