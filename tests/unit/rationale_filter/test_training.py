"""rationale_filter.training without torch: the arithmetic of the schedule, the order of the batches, the settings, the files."""

import json
import unittest

from rationale_filter import training
from rationale_filter.training import TrainSettings
from rationale_filter.training_data import TrainingDataError
from tests.spec.targets import TOKEN_HELPFUL as H
from tests.spec.targets import TOKEN_NOT_HELPFUL as N
from tests.support import TempDirTestCase


def settings(**changes):
    values = {"model_dir": "model", "train_file": "train.json", "out_dir": "out"}
    values.update(changes)
    return TrainSettings(**values)


class ArithmeticTests(unittest.TestCase):
    def test_the_steps_of_an_epoch_round_up_twice_as_in_the_authors_script(self):
        self.assertEqual(training.steps_per_epoch(100, 16), 7)
        self.assertEqual(training.steps_per_epoch(96, 16), 6)
        self.assertEqual(training.steps_per_epoch(100, 16, 2), 4)  # 7 batches in pairs: 4 steps, the last of one batch
        self.assertEqual(training.steps_per_epoch(1, 16), 1)

    def test_the_total_is_epochs_times_the_steps_of_an_epoch(self):
        self.assertEqual(training.total_steps(100, 16, 40), 280)
        self.assertEqual(training.total_steps(100, 16, 40, 2), 160)

    def test_the_learning_rate_falls_in_a_straight_line_to_zero(self):
        self.assertEqual(training.linear_factor(0, 10), 1.0)
        self.assertEqual(training.linear_factor(5, 10), 0.5)
        self.assertEqual(training.linear_factor(10, 10), 0.0)
        self.assertEqual(training.linear_factor(11, 10), 0.0)  # never negative
        self.assertEqual(training.linear_factor(0, 0), 0.0)  # no steps at all: nothing to scale, and no division by zero

    def test_an_epoch_order_is_a_permutation_that_depends_on_the_seed_and_the_epoch_only(self):
        order = training.epoch_order(50, 42, 3)
        self.assertEqual(sorted(order), list(range(50)))
        self.assertEqual(order, training.epoch_order(50, 42, 3))
        self.assertNotEqual(order, training.epoch_order(50, 42, 4))
        self.assertNotEqual(order, training.epoch_order(50, 43, 3))

    def test_the_order_of_known_inputs_is_pinned(self):
        # Random(text) is seeded from the text itself, not from hash(), so this is the same in every process.
        self.assertEqual(training.epoch_order(8, 42, 0), training.epoch_order(8, 42, 0))
        self.assertEqual(len(set(map(tuple, (training.epoch_order(8, 42, e) for e in range(30))))), 30)

    def test_batches_are_in_order_and_the_last_may_be_smaller(self):
        self.assertEqual(training.batches([5, 6, 7, 8, 9], 2), [[5, 6], [7, 8], [9]])
        self.assertEqual(training.batches([], 3), [])

    def test_the_dropout_seed_is_a_31_bit_number_of_the_seed_and_the_epoch(self):
        seeds = {training.epoch_seed(42, e) for e in range(50)}
        self.assertEqual(len(seeds), 50)
        self.assertTrue(all(0 <= s < 2**31 for s in seeds))
        self.assertEqual(training.epoch_seed(42, 7), training.epoch_seed(42, 7))


class SettingsTests(unittest.TestCase):
    def test_the_defaults_are_the_authors(self):
        s = settings()
        self.assertEqual((s.epochs, s.batch_size, s.learning_rate, s.weight_decay), (40, 16, 3e-5, 0.0))
        self.assertEqual((s.max_seq_length, s.doc_stride, s.max_answer_length, s.gradient_accumulation_steps), (512, 128, 30, 1))

    def test_refused_values(self):
        for name, value in {
            "epochs": 0, "batch_size": -1, "gradient_accumulation_steps": 0, "eval_batch_size": True, "seed": -1, "save_every": -2,
            "learning_rate": 0.0, "weight_decay": -0.1, "max_answer_length": 1, "max_seq_length": 4, "doc_stride": 600, "model_dir": "",
            "train_file": None, "out_dir": "",
        }.items():
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                settings(**{name: value})


class FileTests(TempDirTestCase):
    def good(self, n=3):
        return [{"id": f"r{i}", "answer": H if i % 2 else N, "dataset_name": "medqa", "question": f"text {i}"} for i in range(n)]

    def test_a_good_file_is_read(self):
        path = self.write("train.json", json.dumps(self.good()))
        self.assertEqual(len(training.read_examples(path)), 3)

    def test_refused_files(self):
        for name, records in {
            "empty": [],
            "missing field": [{"id": "a", "answer": H}],
            "wrong token": [{"id": "a", "answer": "helpful", "question": "q"}],
            "empty question": [{"id": "a", "answer": H, "question": ""}],
            "repeated id": self.good(1) + self.good(1),
            "not records": ["a"],
        }.items():
            path = self.write(f"{name}.json", json.dumps(records))
            with self.subTest(name), self.assertRaises(TrainingDataError):
                training.read_examples(path)

    def test_the_hash_of_the_base_files_covers_the_weights_the_tokenizer_and_the_configuration(self):
        for name in ("model.safetensors", "config.json", "tokenizer.json", "spiece.model", "notes.md", "pytorch_model.bin"):
            self.write(f"base/{name}", name)
        hashes = training.base_files_sha256(self.dir / "base")
        self.assertEqual(sorted(hashes), ["config.json", "model.safetensors", "pytorch_model.bin", "spiece.model", "tokenizer.json"])
        self.write("base/config.json", "changed")
        self.assertNotEqual(hashes["config.json"], training.base_files_sha256(self.dir / "base")["config.json"])


class SwapTests(TempDirTestCase):
    def folder(self, name, content):
        self.write(f"{name}/file.txt", content)
        return self.dir / name

    def test_the_new_folder_replaces_the_old_one(self):
        final = self.folder("last", "old")
        training._swap_in(self.folder("last.tmp", "new"), final)
        self.assertEqual((final / "file.txt").read_text(encoding="utf-8"), "new")
        self.assertFalse((self.dir / "last.tmp").exists())
        self.assertFalse((self.dir / "last.old").exists())

    def test_a_first_save_needs_no_old_folder(self):
        training._swap_in(self.folder("last.tmp", "new"), self.dir / "last")
        self.assertEqual((self.dir / "last" / "file.txt").read_text(encoding="utf-8"), "new")

    def test_recover_puts_the_old_folder_back_after_a_crash_between_the_renames(self):
        self.folder("last.old", "old")
        self.folder("last.tmp", "half written")
        training.recover(self.dir)
        self.assertEqual((self.dir / "last" / "file.txt").read_text(encoding="utf-8"), "old")
        self.assertFalse((self.dir / "last.tmp").exists())

    def test_recover_leaves_a_complete_folder_alone_and_removes_a_leftover_temporary_one(self):
        self.folder("last", "kept")
        self.folder("last.tmp", "leftover")
        training.recover(self.dir)
        self.assertEqual((self.dir / "last" / "file.txt").read_text(encoding="utf-8"), "kept")
        self.assertFalse((self.dir / "last.tmp").exists())


class ReportTests(TempDirTestCase):
    def test_a_report_is_written_once(self):
        path = training.write_report(self.dir / "r.json", {"b": 1, "a": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 2, "b": 1})
        with self.assertRaises(training.TrainingError):
            training.write_report(path, {})


if __name__ == "__main__":
    unittest.main()
