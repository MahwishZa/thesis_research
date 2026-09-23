"""RAG²'s label function and the filter training configuration.

The label function defines what the baseline *is*, so it is tested exactly:
every branch of the paper's decision tree, and the quantile the tie-break
depends on. No model is loaded - rationale scoring enters as plain data.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.baseline.filter_training.config import (
    BASE_MODEL, CheckpointRecord, ConfigError, FilterTrainingConfig,
)
from experiments.baseline.filter_training.labeling import (
    HELPFUL, NOT_HELPFUL, TAU, LabelingError, PairOutcome, RationaleOutcome,
    label_dataset, label_distribution, label_pair, perplexity_threshold,
    write_training_file,
)


def pair(pid, *, without=(False, 10.0), with_ev=(True, 5.0)):
    return PairOutcome(
        pair_id=pid, question=f"question {pid}", evidence=f"evidence {pid}",
        without=RationaleOutcome(correct=without[0], perplexity=without[1]),
        with_evidence=RationaleOutcome(correct=with_ev[0],
                                       perplexity=with_ev[1]),
    )


class DecisionTreeTests(unittest.TestCase):
    """The paper's tree: correctness flip first, perplexity only as tie-break."""

    def test_wrong_to_right_is_helpful(self):
        label, rule = label_pair(
            pair("a", without=(False, 9.0), with_ev=(True, 9.0)), None)
        self.assertEqual(label, HELPFUL)
        self.assertEqual(rule, "correctness_flip_to_correct")

    def test_right_to_wrong_is_not_helpful(self):
        label, rule = label_pair(
            pair("b", without=(True, 9.0), with_ev=(False, 1.0)), None)
        self.assertEqual(label, NOT_HELPFUL)
        self.assertEqual(rule, "correctness_flip_to_wrong")

    def test_a_flip_outranks_perplexity(self):
        """Even a large perplexity gain cannot override right -> wrong."""
        label, _ = label_pair(
            pair("c", without=(True, 100.0), with_ev=(False, 1.0)), 0.0)
        self.assertEqual(label, NOT_HELPFUL)

    def test_unchanged_correctness_uses_the_perplexity_tie_break(self):
        helpful, rule = label_pair(
            pair("d", without=(True, 10.0), with_ev=(True, 5.0)), 3.0)
        self.assertEqual(helpful, HELPFUL)
        self.assertEqual(rule, "perplexity_differential_top_tau")

        unhelpful, rule = label_pair(
            pair("e", without=(True, 10.0), with_ev=(True, 9.5)), 3.0)
        self.assertEqual(unhelpful, NOT_HELPFUL)
        self.assertEqual(rule, "perplexity_differential_below_tau")

    def test_unchanged_correctness_without_a_threshold_raises(self):
        with self.assertRaises(LabelingError) as ctx:
            label_pair(pair("f", without=(True, 10.0), with_ev=(True, 5.0)),
                       None)
        self.assertIn("no threshold", str(ctx.exception))

    def test_a_passage_that_raises_perplexity_is_not_helpful(self):
        label, _ = label_pair(
            pair("g", without=(True, 5.0), with_ev=(True, 20.0)), 0.0)
        self.assertEqual(label, NOT_HELPFUL)


class ThresholdTests(unittest.TestCase):

    def test_threshold_uses_only_pairs_the_tie_break_judges(self):
        """A flip-decided pair must not help set the quantile."""
        outcomes = [
            pair("flip", without=(False, 1.0), with_ev=(True, 100.0)),
            *[pair(f"t{i}", without=(True, 10.0), with_ev=(True, 10.0 - i))
              for i in range(4)],
        ]
        threshold = perplexity_threshold(outcomes, tau=0.25)
        # Reductions among unchanged pairs: 0, 1, 2, 3 -> top 25% is 3.0
        self.assertEqual(threshold, 3.0)

    def test_no_unchanged_pairs_yields_no_threshold(self):
        outcomes = [pair("x", without=(False, 5.0), with_ev=(True, 5.0))]
        self.assertIsNone(perplexity_threshold(outcomes))

    def test_tau_outside_the_unit_interval_is_refused(self):
        with self.assertRaises(LabelingError):
            perplexity_threshold([pair("a")], tau=0.0)
        with self.assertRaises(LabelingError):
            perplexity_threshold([pair("a")], tau=1.0)

    def test_tau_is_the_papers_value(self):
        self.assertEqual(TAU, 0.25)

    def test_roughly_tau_of_tie_broken_pairs_are_helpful(self):
        outcomes = [
            pair(f"p{i}", without=(True, 100.0), with_ev=(True, 100.0 - i))
            for i in range(100)
        ]
        labelled = label_dataset(outcomes, dataset_name="d")
        helpful = sum(1 for e in labelled if e.answer == HELPFUL)
        self.assertEqual(helpful, 25)


class TrainingFileTests(unittest.TestCase):

    def setUp(self):
        self.examples = label_dataset(
            [pair(f"p{i}", without=(True, 10.0), with_ev=(True, 10.0 - i))
             for i in range(8)],
            dataset_name="ad_filter_5pct",
        )

    def test_records_carry_exactly_the_released_format_fields(self):
        record = self.examples[0].to_training_record()
        self.assertEqual(set(record),
                         {"id", "answer", "dataset_name", "question"})
        self.assertNotIn("rule", record)

    def test_the_prompt_matches_the_released_training_format(self):
        question = self.examples[0].question
        self.assertTrue(question.startswith(
            "Given the following evidence, determine whether it helps answer"))
        self.assertIn("Evidence:", question)
        self.assertIn("Question:", question)

    def test_labels_are_only_the_two_tokens(self):
        for example in self.examples:
            self.assertIn(example.answer, {HELPFUL, NOT_HELPFUL})

    def test_writes_json_and_refuses_to_overwrite(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            write_training_file(self.examples, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded), 8)
            with self.assertRaises(LabelingError) as ctx:
                write_training_file(self.examples, path)
            self.assertIn("refusing to overwrite", str(ctx.exception))

    def test_distribution_reports_labels_and_rules(self):
        counts = label_distribution(self.examples)
        self.assertEqual(counts[HELPFUL] + counts[NOT_HELPFUL], 8)
        self.assertIn("rule:perplexity_differential_top_tau", counts)

    def test_an_empty_set_cannot_be_labelled(self):
        with self.assertRaises(LabelingError):
            label_dataset([], dataset_name="d")

    def test_non_positive_perplexity_is_refused(self):
        with self.assertRaises(LabelingError):
            RationaleOutcome(correct=True, perplexity=0.0)


class ConfigTests(unittest.TestCase):

    def test_hyperparameters_match_the_released_launch_script(self):
        config = FilterTrainingConfig(epochs=10)
        self.assertEqual(config.learning_rate, 3e-5)
        self.assertEqual(config.max_seq_length, 512)
        self.assertEqual(config.doc_stride, 128)
        self.assertEqual(config.optimizer, "adamw")
        self.assertEqual(config.base_model, BASE_MODEL)

    def test_effective_batch_size_preserves_the_papers_sixteen(self):
        config = FilterTrainingConfig(epochs=10)
        self.assertEqual(config.effective_batch_size, 16)
        self.assertEqual(config.per_device_batch_size, 4)

    def test_epochs_must_be_chosen_not_defaulted(self):
        with self.assertRaises(ConfigError) as ctx:
            FilterTrainingConfig().validate()
        self.assertIn("must be chosen and recorded", str(ctx.exception))

    def test_a_changed_effective_batch_is_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            FilterTrainingConfig(epochs=5, per_device_batch_size=2,
                                 gradient_accumulation_steps=2).validate()
        self.assertIn("effective batch size", str(ctx.exception))

    def test_deviations_are_reported_not_hidden(self):
        deviations = FilterTrainingConfig(epochs=10).deviations()
        self.assertIn("per_device_batch_size", deviations)
        self.assertIn("epochs", deviations)
        self.assertIn("40", deviations["epochs"])

    def test_the_recorded_config_carries_its_deviations(self):
        self.assertIn("deviations_from_paper",
                      FilterTrainingConfig(epochs=10).to_dict())


class CheckpointRecordTests(unittest.TestCase):

    def record(self, accuracy):
        return CheckpointRecord(
            checkpoint_path="/ckpt", base_model=BASE_MODEL,
            n_training_examples=5000, n_validation_examples=500,
            validation_accuracy=accuracy, epochs_run=10)

    def test_a_checkpoint_above_chance_is_usable(self):
        self.assertTrue(self.record(0.78).is_usable())

    def test_a_checkpoint_at_chance_is_not_usable(self):
        """At chance the filter has not learned the label function."""
        self.assertFalse(self.record(0.50).is_usable())

    def test_an_impossible_accuracy_is_refused(self):
        with self.assertRaises(ConfigError):
            self.record(1.4).validate()

    def test_a_checkpoint_must_record_its_training_size_and_epochs(self):
        with self.assertRaises(ConfigError):
            CheckpointRecord(
                checkpoint_path="/c", base_model=BASE_MODEL,
                n_training_examples=0, n_validation_examples=10,
                validation_accuracy=0.8, epochs_run=3).validate()


if __name__ == "__main__":
    unittest.main()
