"""rationale_filter.filter_model and rationale_filter.training against real torch and transformers, on a tiny T5.

``write_tiny_t5`` writes a Flan-T5-shaped model with random weights and a word-level tokenizer (tests/integration/tiny_models.py).
Needs torch, transformers and tokenizers; the classes skip, with the reason, when they are missing. Nothing is downloaded.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path

from rationale_filter import cli, filter_model, inference, training
from rationale_filter.filter_model import FilterModel, FilterModelError
from rationale_query.store import ConfigMismatch
from tests.integration.test_real_libraries import unusable
from tests.integration.tiny_models import write_tiny_t5
from tests.support import TempDirTestCase

QUESTION = "Which drug is a donepezil ? A) one B) two C) three D) four"
H, N = filter_model.TOKEN_HELPFUL, filter_model.TOKEN_NOT_HELPFUL


def examples(count=16):
    """Evidence that starts with "good" is helpful, evidence that starts with "bad" is not: a rule a tiny model can learn."""
    rows = []
    for i in range(count):
        good = i % 2 == 0
        words = ("one", "two", "three", "four") if good else ("five", "six", "seven", "eight")
        evidence = f"{'good' if good else 'bad'} {words[i % 4]} dementia"
        rows.append(
            {"id": f"r{i}", "answer": H if good else N, "dataset_name": "tiny", "question": inference.build_filter_input(evidence=evidence, question=QUESTION)}
        )
    return rows


def skip_without_torch():
    reason = unusable("torch", "transformers", "tokenizers")
    if reason:
        raise unittest.SkipTest(reason)


class Fixture(TempDirTestCase):
    """A prepared tiny model shared by the tests of one class (it is only read)."""

    @classmethod
    def setUpClass(cls):
        skip_without_torch()
        holder = tempfile.TemporaryDirectory(prefix="rf filter ", ignore_cleanup_errors=True)
        cls.addClassCleanup(holder.cleanup)
        cls.root = Path(holder.name)
        write_tiny_t5(cls.root / "base")
        cls.info = filter_model.prepare_model(cls.root / "base", cls.root / "prepared")
        cls.model = FilterModel.load(cls.root / "prepared", device="cpu")

    def train_file(self, rows=None, name="train.json"):
        """The path of a labelled file. It is written when ``rows`` is given or when it does not exist yet, never rewritten otherwise."""
        path = self.dir / name
        if rows is not None or not path.exists():
            path.write_text(json.dumps(rows if rows is not None else examples()), encoding="utf-8")
        return path

    def settings(self, **changes):
        values = {
            "model_dir": str(self.root / "prepared"), "train_file": str(self.train_file()), "out_dir": str(self.dir / "run"),
            "epochs": 30, "batch_size": 4, "learning_rate": 5e-3,
        }
        values.update(changes)
        return training.TrainSettings(**values)


class PrepareTests(Fixture):
    def test_the_two_tokens_are_added_and_the_embedding_is_resized_to_the_tokenizer(self):
        self.assertEqual(self.info["label_token_ids"], {H: 54, N: 55})
        self.assertEqual(self.info["tokenizer_length"], [54, 56])
        self.assertEqual(self.info["embedding_rows"], [62, 56])
        self.assertEqual(self.info["model_class"], "T5ForConditionalGeneration")

    def test_a_folder_that_is_not_empty_is_refused(self):
        with self.assertRaises(FilterModelError):
            filter_model.prepare_model(self.root / "base", self.root / "prepared")

    def test_a_model_that_already_knows_the_tokens_is_refused(self):
        with self.assertRaises(FilterModelError) as caught:
            filter_model.prepare_model(self.root / "prepared", self.dir / "again")
        self.assertIn("not a base model", str(caught.exception))

    def test_the_base_model_without_the_tokens_cannot_be_used_as_a_filter(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.root / "base"))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.root / "base"))
        with self.assertRaises(FilterModelError) as caught:
            FilterModel(model, tokenizer, "cpu")
        self.assertIn("prepare-model", str(caught.exception))


class LogitTests(Fixture):
    TEXTS = [inference.build_filter_input(evidence=f"{word} snippet one dementia", question=QUESTION) for word in ("good", "bad", "donepezil aspirin insulin")]

    def test_the_logits_equal_the_scores_of_the_first_generated_token(self):
        import torch

        tokenizer, model = self.model.tokenizer, self.model.model
        batch = tokenizer(self.TEXTS, return_tensors="pt", padding=True)
        with torch.no_grad():
            scores = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict_in_generate=True, output_scores=True, max_length=30
            ).scores[0]
        ours = self.model.logits(self.TEXTS, batch_size=8)
        for row, (helpful, other) in enumerate(ours):
            self.assertAlmostEqual(helpful, float(scores[row, 54]), places=4)
            self.assertAlmostEqual(other, float(scores[row, 55]), places=4)

    def test_the_batch_size_does_not_change_the_numbers(self):
        together = self.model.logits(self.TEXTS, batch_size=8)
        for batch_size in (1, 2):
            for a, b in zip(together, self.model.logits(self.TEXTS, batch_size=batch_size), strict=True):
                self.assertAlmostEqual(a[0], b[0], places=4)
                self.assertAlmostEqual(a[1], b[1], places=4)

    def test_the_order_of_the_texts_is_kept_although_they_are_batched_by_length(self):
        forward = self.model.logits(self.TEXTS, batch_size=2)
        backward = self.model.logits(list(reversed(self.TEXTS)), batch_size=2)
        for a, b in zip(forward, reversed(backward), strict=True):
            self.assertAlmostEqual(a[0], b[0], places=4)

    def test_judging_follows_the_larger_logit_and_a_single_text_agrees_with_a_batch(self):
        pairs = self.model.logits(self.TEXTS)
        self.assertEqual(self.model.judge_many(self.TEXTS), [a >= b for a, b in pairs])
        self.assertEqual([self.model.judge(text) for text in self.TEXTS], self.model.judge_many(self.TEXTS))

    def test_no_texts_give_no_logits_and_bad_arguments_are_refused(self):
        self.assertEqual(self.model.logits([]), [])
        with self.assertRaises(ValueError):
            self.model.logits(self.TEXTS, batch_size=0)
        with self.assertRaises(ValueError):
            self.model.logits(["fine", "  "])

    def test_the_decisions_of_the_batch_function_equal_those_of_one_at_a_time(self):
        snippets = ["good one dementia", "bad five dementia", "good three dementia"]
        one_by_one = inference.judge_snippets(question=QUESTION, snippets=snippets, judge=self.model.judge)
        batched = inference.decisions_for(question=QUESTION, snippets=snippets, judge_many=self.model.judge_many)
        self.assertEqual(one_by_one, batched)


class WindowTests(Fixture):
    LONG = inference.build_filter_input(evidence=" ".join(["good one two three four five six seven eight nine ten"] * 4), question=QUESTION)

    def small(self):
        return FilterModel.load(self.root / "prepared", device="cpu", max_seq_length=24, doc_stride=6)

    def test_a_long_input_is_cut_into_windows_that_overlap_and_all_end_with_the_end_token(self):
        windows, owners = filter_model.tokenize_windows(self.model.tokenizer, [self.LONG], 24, 6)
        self.assertGreater(len(windows), 2)
        self.assertEqual(set(owners), {0})
        self.assertTrue(all(len(w) <= 24 and w[-1] == 1 for w in windows))
        for first, second in zip(windows, windows[1:], strict=False):
            self.assertEqual(first[-7:-1], second[:6])  # the stride: six tokens are repeated

    def test_the_logits_of_a_long_input_are_the_mean_of_the_logits_of_its_windows(self):
        import torch

        small = self.small()
        windows, _ = filter_model.tokenize_windows(small.tokenizer, [self.LONG], 24, 6)
        rows = []
        with torch.no_grad():
            for ids in windows:
                out = small.model(input_ids=torch.tensor([ids]), decoder_input_ids=torch.tensor([[small.decoder_start]]))
                rows.append((float(out.logits[0, 0, 54]), float(out.logits[0, 0, 55])))
        expected = (sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows))
        got = small.logits([self.LONG])[0]
        self.assertAlmostEqual(got[0], expected[0], places=4)
        self.assertAlmostEqual(got[1], expected[1], places=4)

    def test_the_share_of_inputs_that_needed_windows_is_counted(self):
        small = self.small()
        self.assertIsNone(small.windowed_share())
        short = inference.build_filter_input(evidence="good one", question="A ) one")
        small.logits([short, self.LONG, short, self.LONG])
        self.assertEqual((small.inputs_seen, small.windowed_inputs, small.windowed_share()), (4, 2, 0.5))

    def test_the_info_names_the_rule_for_long_inputs(self):
        info = self.model.info()
        self.assertIn("averaged over the windows", info["overflow_rule"])
        self.assertEqual(info["label_token_ids"], {H: 54, N: 55})


class FeatureTests(Fixture):
    def test_every_window_carries_the_target_of_its_example(self):
        rows = examples(4)
        rows[0]["question"] = WindowTests.LONG
        features = training.build_features(self.model.tokenizer, rows, 24, 6, 30)
        self.assertGreater(len(features), 4)
        for feature in features:
            self.assertEqual(feature.labels, [54 if rows[feature.example]["answer"] == H else 55, 1])
        windows = [sum(1 for f in features if f.example == i) for i in range(4)]
        self.assertGreater(windows[0], windows[1])  # the long input has the most windows

    def test_the_collate_pads_inputs_with_the_pad_id_and_labels_with_minus_100(self):
        features = training.build_features(self.model.tokenizer, examples(2), 512, 128, 30)
        features[1] = training.Feature(features[1].input_ids[:5], [55], features[1].example)
        batch = training.collate(features, 0)
        self.assertEqual(batch["input_ids"].shape[1], len(features[0].input_ids))
        self.assertEqual(batch["attention_mask"][1].tolist(), [1] * 5 + [0] * (len(features[0].input_ids) - 5))
        self.assertEqual(batch["labels"][1].tolist(), [55, training.IGNORE])
        self.assertTrue((batch["input_ids"][1, 5:] == 0).all())

    def test_a_target_that_is_not_the_label_token_and_the_end_token_is_refused(self):
        rows = examples(2)
        rows[0]["answer"] = H
        features = training.build_features(self.model.tokenizer, rows, 512, 128, 30)
        self.assertEqual(len(features), 2)
        with self.assertRaises(FilterModelError):
            training.build_features(self.model.tokenizer, rows, 512, 128, 1)  # a one-token answer cuts off the end token


class TrainTests(Fixture):
    def test_the_filter_learns_a_simple_rule_and_the_run_leaves_its_files(self):
        settings = self.settings(validation_file=str(self.train_file(name="validation.json")))
        events = []
        summary = training.train(settings, device="cpu", progress=events.append)
        out = self.dir / "run"
        for name in ("last", "best", "final"):
            self.assertTrue((out / name / "config.json").is_file(), name)
        self.assertTrue((out / "last" / "state.pt").is_file())
        self.assertTrue((out / "best" / "metrics.json").is_file())
        log = [json.loads(line) for line in (out / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([entry["epoch"] for entry in log], list(range(1, 31)))
        self.assertLess(log[-1]["mean_loss"], log[0]["mean_loss"] / 5)
        self.assertEqual(log[-1]["learning_rate"], 0.0)  # the schedule ends at zero after exactly the planned number of steps
        self.assertEqual(summary["total_steps"], 30 * 4)
        self.assertGreaterEqual(summary["best"]["accuracy"], 0.9)
        result = training.evaluate_examples(FilterModel.load(out / "final", device="cpu"), examples())
        self.assertGreaterEqual(result["report"]["accuracy"], 0.9)
        self.assertEqual(sum(1 for e in events if e["event"] == "epoch"), 30)

    def test_the_best_model_is_the_earliest_of_the_highest_validation_accuracy(self):
        training.train(self.settings(validation_file=str(self.train_file(name="validation.json")), epochs=12), device="cpu")
        log = [json.loads(line) for line in (self.dir / "run" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
        best = json.loads((self.dir / "run" / "best" / "metrics.json").read_text(encoding="utf-8"))
        top = max(entry["validation"]["accuracy"] for entry in log)
        self.assertEqual(best["accuracy"], top)
        self.assertEqual(best["epoch"], next(entry["epoch"] for entry in log if entry["validation"]["accuracy"] == top))

    def test_a_run_with_other_settings_or_other_data_is_refused_and_a_stopped_run_continues(self):
        training.train(self.settings(epochs=2), device="cpu")
        with self.assertRaises(ConfigMismatch) as caught:
            training.train(self.settings(epochs=2, learning_rate=1e-3), device="cpu")
        self.assertIn("learning_rate", str(caught.exception))
        self.train_file(rows=examples(12))
        with self.assertRaises(ConfigMismatch) as caught:
            training.train(self.settings(epochs=2), device="cpu")
        self.assertIn("train_sha256", str(caught.exception))

    def test_a_stopped_run_gives_exactly_the_model_of_an_uninterrupted_run(self):
        import torch

        training.train(self.settings(epochs=6, out_dir=str(self.dir / "straight")), device="cpu")

        class Stop(Exception):
            pass

        def stop_after_epoch_three(event):
            if event["event"] == "epoch" and event["epoch"] == 3:
                raise Stop

        with self.assertRaises(Stop):
            training.train(self.settings(epochs=6, out_dir=str(self.dir / "stopped")), device="cpu", progress=stop_after_epoch_three)
        summary = training.train(self.settings(epochs=6, out_dir=str(self.dir / "stopped")), device="cpu")
        self.assertEqual(summary["resumed_from_epoch"], 3)
        straight = FilterModel.load(self.dir / "straight" / "final", device="cpu").model.state_dict()
        resumed = FilterModel.load(self.dir / "stopped" / "final", device="cpu").model.state_dict()
        self.assertEqual(sorted(straight), sorted(resumed))
        for name in straight:
            self.assertTrue(torch.allclose(straight[name], resumed[name], atol=1e-6), name)
        log = (self.dir / "stopped" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["epoch"] for line in log], [1, 2, 3, 4, 5, 6])

    def test_gradient_accumulation_takes_one_step_for_several_batches_and_the_schedule_still_ends_at_zero(self):
        summary = training.train(self.settings(epochs=3, gradient_accumulation_steps=3, out_dir=str(self.dir / "acc")), device="cpu")
        self.assertEqual(summary["steps_per_epoch"], 2)  # 4 batches in groups of 3: a group of 3 and a group of 1
        log = [json.loads(line) for line in (self.dir / "acc" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(log[-1]["learning_rate"], 0.0)
        self.assertAlmostEqual(log[0]["learning_rate"], 5e-3 * (1 - 2 / 6))

    def test_the_models_of_chosen_epochs_are_kept(self):
        training.train(self.settings(epochs=4, save_every=2, out_dir=str(self.dir / "keep")), device="cpu")
        self.assertTrue((self.dir / "keep" / "epoch_2" / "config.json").is_file())
        self.assertTrue((self.dir / "keep" / "epoch_4" / "config.json").is_file())
        self.assertFalse((self.dir / "keep" / "epoch_3").exists())

    def test_training_on_a_model_without_the_label_tokens_is_refused(self):
        with self.assertRaises(FilterModelError):
            training.train(self.settings(model_dir=str(self.root / "base"), epochs=1, out_dir=str(self.dir / "bad")), device="cpu")


class MemoryTestTests(Fixture):
    def test_one_step_on_the_longest_inputs_reports_the_time_and_writes_nothing(self):
        result = training.memory_test(self.settings(batch_size=8), device="cpu")
        self.assertTrue(result["fits"])
        self.assertEqual(result["batch_size"], 8)
        self.assertGreater(result["seconds_per_step"], 0)
        self.assertGreater(result["estimated_seconds_per_epoch"], 0)
        self.assertEqual(result["steps_per_epoch"], 2)
        self.assertEqual(result["longest_input_tokens"], max(len(f.input_ids) for f in training.build_features(self.model.tokenizer, examples(), 512, 128, 30)))
        self.assertFalse((self.dir / "run").exists())

    def test_a_file_with_fewer_windows_than_the_batch_says_so(self):
        result = training.memory_test(self.settings(train_file=str(self.train_file(rows=examples(3), name="few.json"))), device="cpu")
        self.assertEqual(result["batch_size"], 3)
        self.assertIn("fewer than the batch size", result["note"])

    def test_running_out_of_memory_is_the_result_and_not_a_crash(self):
        import torch

        original = training.collate

        def exhausted(*arguments, **options):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.00 GiB")

        training.collate = exhausted
        try:
            result = training.memory_test(self.settings(), device="cpu")
        finally:
            training.collate = original
        self.assertFalse(result["fits"])
        self.assertIn("out of memory", result["error"])


class EvaluateTests(Fixture):
    def test_the_report_has_accuracy_per_class_the_majority_rate_and_the_auc_and_the_predictions_carry_the_dataset(self):
        training.train(self.settings(out_dir=str(self.dir / "run")), device="cpu")
        model = FilterModel.load(self.dir / "run" / "final", device="cpu")
        result = training.evaluate_examples(model, examples())
        report = result["report"]
        self.assertEqual((report["n"], report["majority_rate"]), (16, 0.5))
        self.assertGreaterEqual(report["accuracy"], 0.9)
        self.assertGreaterEqual(report["auc"], 0.9)
        self.assertEqual(result["predictions"]["r0"]["dataset_name"], "tiny")

    def test_one_class_only_has_no_auc(self):
        only_helpful = [row for row in examples() if row["answer"] == H]
        result = training.evaluate_examples(self.model, only_helpful)
        self.assertIsNone(result["report"]["auc"])


class CommandLineTests(Fixture):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(list(argv), streams=cli.Streams(out, err))
        return status, out.getvalue(), err.getvalue()

    def test_prepare_train_memory_test_and_evaluate_end_to_end(self):
        status, out, err = self.run_cli("prepare-model", "--base", str(self.root / "base"), "--out", str(self.dir / "prepared"), "--local-files-only")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(json.loads(out)["label_token_ids"], {H: 54, N: 55})
        train_file = self.train_file()
        common = ["--model", str(self.dir / "prepared"), "--train-file", str(train_file), "--batch-size", "4", "--learning-rate", "5e-3", "--device", "cpu"]
        status, out, err = self.run_cli("memory-test", *common)
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(out)["fits"])
        status, out, err = self.run_cli("train", *common, "--epochs", "30", "--out", str(self.dir / "run"), "--validation-file", str(train_file))
        self.assertEqual(status, 0, err)
        self.assertEqual(err.count("[epoch "), 30)
        self.assertEqual(json.loads(out)["epochs"], 30)
        target = self.dir / "report.json"
        status, out, err = self.run_cli("evaluate", "--model", str(self.dir / "run" / "final"), "--file", str(train_file), "--report-out", str(target), "--device", "cpu")
        self.assertEqual((status, err), (0, ""))
        report = json.loads(out)
        self.assertGreaterEqual(report["accuracy"], 0.9)
        self.assertEqual(report["windowed_inputs"], 0)
        self.assertEqual(sorted(json.loads(target.read_text(encoding="utf-8"))), ["predictions", "report"])
        status, out, err = self.run_cli("evaluate", "--model", str(self.dir / "run" / "final"), "--file", str(train_file), "--report-out", str(target), "--device", "cpu")
        self.assertEqual(status, 1)
        self.assertIn("refusing to overwrite", err)

    def test_a_memory_test_that_does_not_fit_exits_with_3(self):
        import torch

        original = training.collate
        training.collate = lambda *a, **k: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("CUDA out of memory"))
        try:
            status, out, err = self.run_cli(
                "memory-test", "--model", str(self.root / "prepared"), "--train-file", str(self.train_file()), "--device", "cpu"
            )
        finally:
            training.collate = original
        self.assertEqual(status, 3)
        self.assertFalse(json.loads(out)["fits"])

    def test_a_bad_setting_is_an_error_message_not_a_traceback(self):
        status, out, err = self.run_cli("train", "--model", str(self.root / "prepared"), "--train-file", str(self.train_file()), "--out", str(self.dir / "x"), "--epochs", "0")
        self.assertEqual(status, 1)
        self.assertIn("epochs must be a whole number from 1", err)


if __name__ == "__main__":
    unittest.main()
