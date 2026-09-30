"""Deployed-rule validation metrics, usability criterion, resume fingerprint,
local-training config deviations. Pure Python: no torch needed."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.baseline.filter_training.config import (
    BASE_MODEL, CheckpointRecord, ConfigError, FilterTrainingConfig,
)
from experiments.baseline.filter_training.metrics import (
    classification_metrics, two_way_prediction,
)
from experiments.baseline.filter_training.train import (
    _CHECKPOINT_REQUIRED, check_resume_fingerprint,
    latest_complete_checkpoint, training_fingerprint,
)


class MetricTests(unittest.TestCase):

    def test_reject_everything_matches_majority_and_is_chance_balanced(self):
        gold = [True] * 3 + [False] * 7
        m = classification_metrics([False] * 10, gold)
        self.assertAlmostEqual(m["accuracy"], 0.7)
        self.assertAlmostEqual(m["majority_baseline"], 0.7)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)
        self.assertEqual(m["recall_helpful"], 0.0)

    def test_perfect(self):
        gold = [True, False, True, False]
        m = classification_metrics(gold, gold)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)

    def test_length_mismatch_and_empty_are_refused(self):
        with self.assertRaises(ValueError):
            classification_metrics([True], [True, False])
        with self.assertRaises(ValueError):
            classification_metrics([], [])

    def test_two_way_rule_ties_go_to_helpful(self):
        self.assertTrue(two_way_prediction(1.0, 1.0))
        self.assertFalse(two_way_prediction(0.9, 1.0))


class UsabilityTests(unittest.TestCase):

    def rec(self, acc, bal=None, maj=None):
        return CheckpointRecord(
            checkpoint_path="/c", base_model=BASE_MODEL,
            n_training_examples=450, n_validation_examples=50,
            validation_accuracy=acc, epochs_run=3,
            balanced_accuracy=bal, majority_baseline=maj)

    def test_reject_everything_is_not_usable_despite_accuracy_above_half(self):
        self.assertFalse(self.rec(0.714, bal=0.5, maj=0.714).is_usable())

    def test_beating_the_majority_baseline_is_usable(self):
        self.assertTrue(self.rec(0.80, bal=0.72, maj=0.714).is_usable())

    def test_legacy_record_without_extra_metrics_still_works(self):
        self.assertTrue(self.rec(0.78).is_usable())


class LocalConfigTests(unittest.TestCase):

    def local(self, **kw):
        return FilterTrainingConfig(
            epochs=3, optimizer="adafactor", mixed_precision="fp32",
            gradient_checkpointing=True, use_cpu=True, **kw)

    def test_local_switches_are_reported_as_deviations(self):
        d = self.local().deviations()
        for key in ("optimizer", "mixed_precision", "gradient_checkpointing",
                    "use_cpu"):
            self.assertIn(key, d)

    def test_paper_defaults_report_no_local_deviations(self):
        d = FilterTrainingConfig(epochs=40).deviations()
        self.assertNotIn("optimizer", d)
        self.assertNotIn("gradient_checkpointing", d)

    def test_unknown_optimizer_or_precision_refused(self):
        with self.assertRaises(ConfigError):
            FilterTrainingConfig(epochs=1, optimizer="sgd").validate()
        with self.assertRaises(ConfigError):
            FilterTrainingConfig(epochs=1, mixed_precision="fp8").validate()


class ResumeFingerprintTests(unittest.TestCase):

    def fp(self, **kw):
        return training_fingerprint(
            FilterTrainingConfig(epochs=3, **kw), labels_sha256="abc",
            val_fraction=0.1)

    def test_identical_settings_resume(self):
        check_resume_fingerprint(self.fp(), self.fp())

    def test_changed_epochs_refused(self):
        other = training_fingerprint(
            FilterTrainingConfig(epochs=5), labels_sha256="abc",
            val_fraction=0.1)
        with self.assertRaises(ConfigError) as ctx:
            check_resume_fingerprint(other, self.fp())
        self.assertIn("epochs", str(ctx.exception))

    def test_changed_labels_refused(self):
        other = training_fingerprint(
            FilterTrainingConfig(epochs=3), labels_sha256="zzz",
            val_fraction=0.1)
        with self.assertRaises(ConfigError):
            check_resume_fingerprint(other, self.fp())

    def test_changed_optimizer_refused(self):
        with self.assertRaises(ConfigError):
            check_resume_fingerprint(self.fp(optimizer="adafactor"), self.fp())


class LatestCompleteCheckpointTests(unittest.TestCase):

    def make(self, root, n, files=_CHECKPOINT_REQUIRED):
        d = Path(root) / f"checkpoint-{n}"
        d.mkdir()
        for f in files:
            (d / f).write_bytes(b"x")
        return d

    def test_picks_newest_complete(self):
        with TemporaryDirectory() as tmp:
            self.make(tmp, 14)
            newest = self.make(tmp, 28)
            self.assertEqual(latest_complete_checkpoint(tmp), newest)

    def test_skips_a_partially_written_newest_checkpoint(self):
        with TemporaryDirectory() as tmp:
            good = self.make(tmp, 28)
            self.make(tmp, 42, files=("config.json",))  # killed mid-save
            self.assertEqual(latest_complete_checkpoint(tmp), good)

    def test_numeric_not_lexicographic_order(self):
        with TemporaryDirectory() as tmp:
            self.make(tmp, 9)
            newest = self.make(tmp, 100)
            self.assertEqual(latest_complete_checkpoint(tmp), newest)

    def test_none_when_nothing_complete_or_missing_dir(self):
        with TemporaryDirectory() as tmp:
            self.make(tmp, 5, files=("config.json",))
            self.assertIsNone(latest_complete_checkpoint(tmp))
            self.assertIsNone(latest_complete_checkpoint(Path(tmp) / "nope"))


if __name__ == "__main__":
    unittest.main()
