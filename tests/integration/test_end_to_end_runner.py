"""Integration test for experiments/runners/run_end_to_end.py.

This is the test that proves objective 1 ("run the proposed system and
check its overall performance") is actually satisfiable in this repository:
before this script and this test existed, every piece was real and
unit-tested in isolation, but nothing exercised retriever candidates
-> admission (baseline AND proposed) -> generation -> standard-metric
scoring -> a proposed-vs-baseline comparison as one run.

It runs entirely on the module's built-in synthetic fixture (no network, no
downloaded model - this sandbox has neither), so it is not a scientific
result; it is proof the pipeline executes cleanly end-to-end and that the
comparison mechanism (objective 3) correctly detects an improvement when one
is designed into the fixture (a recency-weighted policy should recover the
current passage that a relevance-only ranking would miss).
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.runners import run_end_to_end as e2e


class EndToEndRunnerTests(unittest.TestCase):

    def test_runs_cleanly_with_no_errors(self):
        with TemporaryDirectory() as tmp:
            code = e2e.main([
                "--output-dir", tmp, "--n-questions", "5",
                "--ablation-lambdas", "0,0.5,1.0",
            ])
            self.assertEqual(code, 0)

            results = Path(tmp) / "results.jsonl"
            report = Path(tmp) / "metrics_report.json"
            self.assertTrue(results.exists())
            self.assertTrue(report.exists())

            records = [json.loads(l) for l in results.read_text().splitlines()]
            self.assertTrue(all(r["status"] == "ok" for r in records))
            # 5 questions x (no_filter, baseline, 3 proposed lambdas) = 5*5
            self.assertEqual(len(records), 25)

    def test_report_separates_main_evaluation_from_ablation_study(self):
        """The report must distinguish step 3 (RAG2 vs proposed) from step 4
        (full proposed vs the same system with its key component - recency
        weighting - removed), not just dump an undifferentiated lambda
        sweep: those are two different pipeline steps with two different
        research questions."""
        with TemporaryDirectory() as tmp:
            e2e.main([
                "--output-dir", tmp, "--n-questions", "10",
                "--ablation-lambdas", "0,1.0", "--proposed-lambda", "1.0",
            ])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())

            main_eval = report["main_evaluation"]
            self.assertEqual(main_eval["baseline"], "baseline")
            self.assertEqual(main_eval["proposed"], "proposed_lambda_1")
            self.assertEqual(main_eval["verdict"], "IMPROVES")

            ablation = report["ablation_study"]
            self.assertEqual(ablation["full_proposed"], "proposed_lambda_1")
            self.assertEqual(ablation["ablated_proposed"], "proposed_lambda_0")
            self.assertEqual(ablation["verdict"], "COMPONENT HELPS")

    def test_ablation_lambda_zero_is_always_included(self):
        """lambda=0 (the component-removed arm) is mandatory for the
        ablation study - it must be added even if the caller's
        --ablation-lambdas omits it."""
        with TemporaryDirectory() as tmp:
            e2e.main([
                "--output-dir", tmp, "--n-questions", "4",
                "--ablation-lambdas", "0.5,1.0",
            ])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            self.assertIn("proposed_lambda_0", report["metrics_by_system"])

    def test_report_contains_every_arm(self):
        with TemporaryDirectory() as tmp:
            e2e.main([
                "--output-dir", tmp, "--n-questions", "4",
                "--ablation-lambdas", "0,1.0",
            ])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            by_system = report["metrics_by_system"]
            self.assertEqual(
                set(by_system),
                {"no_filter", "baseline", "proposed_lambda_0", "proposed_lambda_1"},
            )
            for metrics in by_system.values():
                self.assertEqual(metrics["n"], 4)
                for key in (
                    "exact_match", "token_f1", "rouge_l_f1",
                    "context_precision", "context_recall", "groundedness",
                ):
                    self.assertIn(key, metrics)

    def test_recency_weighted_policy_recovers_the_current_passage(self):
        """The fixture is deliberately built so a relevance-only ranking
        (baseline, no_filter) picks the higher-reranked but STALE passage,
        while a sufficiently recency-weighted proposed policy picks the
        lower-reranked but CURRENT one. lambda=1.0 (pure recency) must
        therefore score strictly higher than the baseline on token F1 -
        this is the concrete, checkable form of "the proposed system
        improves on the baseline" (objective 3)."""
        with TemporaryDirectory() as tmp:
            e2e.main([
                "--output-dir", tmp, "--n-questions", "10",
                "--ablation-lambdas", "0,1.0",
            ])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            by_system = report["metrics_by_system"]

            baseline_f1 = by_system["baseline"]["token_f1"]
            proposed_f1 = by_system["proposed_lambda_1"]["token_f1"]
            self.assertGreater(proposed_f1, baseline_f1)
            self.assertEqual(proposed_f1, 1.0)

    def test_pure_relevance_ablation_matches_baseline_behaviour(self):
        """lambda=0 is the built-in pure-relevance ablation: with no
        recency signal at all, the proposed policy's ranking degenerates to
        the same reranker-rank ordering the baseline and no-filter arms
        use, so it should NOT outperform the baseline on this fixture."""
        with TemporaryDirectory() as tmp:
            e2e.main([
                "--output-dir", tmp, "--n-questions", "10",
                "--ablation-lambdas", "0",
            ])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            by_system = report["metrics_by_system"]
            self.assertEqual(
                by_system["proposed_lambda_0"]["token_f1"],
                by_system["baseline"]["token_f1"],
            )

    def test_report_records_which_rag2_filter_the_baseline_actually_used(self):
        """Without a trained checkpoint the baseline is a stand-in, and the
        report must say so: 'proposed IMPROVES on baseline' against an
        all-HELPFUL mock is a different claim from the same verdict against
        the paper's classifier, and a reader of metrics_report.json cannot
        be left to infer which."""
        with TemporaryDirectory() as tmp:
            e2e.main(["--output-dir", tmp, "--n-questions", "3"])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            main_eval = report["main_evaluation"]
            self.assertFalse(main_eval["baseline_is_trained_rag2"])
            self.assertIn("stand-in", main_eval["baseline_filter"])
            self.assertIn("NO trained checkpoint", main_eval["baseline_filter"])

    def test_the_real_rag2_filter_is_reachable_from_the_cli(self):
        """--rag2-checkpoint must route to the real FlanT5RAG2Filter. It
        needs torch/transformers plus a real checkpoint, so this asserts
        the wiring reaches that constructor rather than silently falling
        back to the mock - a silent fallback would produce a report
        labelled as the real baseline when it was not."""
        with self.assertRaises((ImportError, OSError, ValueError)):
            e2e.make_rag2_filter("definitely/not-a-real-checkpoint", [])

    def test_no_checkpoint_returns_the_labelled_stand_in(self):
        filt, label = e2e.make_rag2_filter(None, e2e.make_fixture_items(2))
        self.assertEqual(type(filt).__name__, "MockRAG2Filter")
        self.assertIn("stand-in", label)

    def test_real_model_defaults_to_the_contract_quantization(self):
        """A full-precision 8B load is the one configuration the documented
        target GPU cannot run, so nf4 is the default, not an opt-in."""
        gen = e2e.make_generator(True, "org/model", "abc123def456")
        self.assertEqual(gen.spec.quantization, "nf4")

    def test_quantization_can_be_disabled_explicitly(self):
        gen = e2e.make_generator(True, "org/model", "abc123def456", "none")
        self.assertIsNone(gen.spec.quantization)

    def test_real_model_flag_builds_a_valid_model_spec(self):
        """Regression test: --real-model used to construct ModelSpec with a
        'name' kwarg that does not exist on that dataclass (it takes
        model_id + a required, pinned revision), so --real-model raised
        TypeError/GeneratorError immediately. No torch/transformers needed
        for this check - only ModelSpec construction is exercised."""
        gen = e2e.make_generator(True, "org/model-name", "abc123def456")
        self.assertEqual(gen.spec.model_id, "org/model-name")
        self.assertEqual(gen.spec.revision, "abc123def456")

    def test_real_model_flag_requires_a_pinned_revision(self):
        with self.assertRaises(SystemExit):
            e2e.make_generator(True, "org/model-name", None)

    def test_real_model_flag_requires_a_model_name(self):
        with self.assertRaises(SystemExit):
            e2e.make_generator(True, None, "abc123def456")

    def test_refuses_to_overwrite_an_existing_run(self):
        """run_experiment() itself refuses to overwrite raw model output;
        this script clears its own output directory instead, so a second
        invocation with the same --output-dir must still succeed (not
        silently merge with stale results from a previous run)."""
        with TemporaryDirectory() as tmp:
            first = e2e.main(["--output-dir", tmp, "--n-questions", "3"])
            second = e2e.main(["--output-dir", tmp, "--n-questions", "3"])
            self.assertEqual(first, 0)
            self.assertEqual(second, 0)


if __name__ == "__main__":
    unittest.main()


class FittedParameterPlumbingTests(unittest.TestCase):
    """theta, the half-life and the context budget were module constants
    with no CLI override, so a real run would silently have used fixture
    placeholders (theta=0.5, H=365, budget=1) no matter what the validation
    split produced - defeating AdmissionConfig.validate()'s refusal to
    default theta. They are now arguments, and their values are recorded in
    readable form rather than only inside an opaque config hash."""

    def test_theta_half_life_and_budget_are_settable_from_the_cli(self):
        with TemporaryDirectory() as tmp:
            code = e2e.main([
                "--output-dir", tmp, "--n-questions", "3",
                "--theta", "0.25", "--half-life", "180", "--budget", "2",
            ])
            self.assertEqual(code, 0)
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            cfg = report["system_config"]
            self.assertEqual(cfg["theta"], 0.25)
            self.assertEqual(cfg["half_life_days"], 180.0)
            self.assertEqual(cfg["context_budget"], 2)

    def test_the_recorded_config_says_the_values_are_not_fitted(self):
        """A reader must not mistake a placeholder for a fitted value."""
        with TemporaryDirectory() as tmp:
            e2e.main(["--output-dir", tmp, "--n-questions", "3"])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            self.assertFalse(
                report["system_config"]["theta_and_half_life_are_fitted"])

    def test_the_budget_is_applied_to_every_arm(self):
        """The budget is a control, not a treatment: if it reached only
        some arms, an admission difference would be confounded with context
        volume. assert_budget_parity already guards this inside a run, so a
        budget that failed to plumb through would raise here."""
        with TemporaryDirectory() as tmp:
            code = e2e.main([
                "--output-dir", tmp, "--n-questions", "3", "--budget", "3",
            ])
            self.assertEqual(code, 0)
            records = [json.loads(l) for l in
                       (Path(tmp) / "results.jsonl").read_text().splitlines()]
            for record in records:
                self.assertLessEqual(len(record["admitted_evidence_ids"]), 3)

    def test_the_report_always_carries_a_temporal_subgroup_breakdown(self):
        """The key must exist on every run, not only ones a caller happened
        to populate with temporal-candidate questions - a missing key would
        silently fail the analysis a real run needs this for, rather than
        reporting zero."""
        with TemporaryDirectory() as tmp:
            e2e.main(["--output-dir", tmp, "--n-questions", "3"])
            report = json.loads((Path(tmp) / "metrics_report.json").read_text())
            sub = report["temporal_subgroup"]
            self.assertEqual(sub["n_temporal_candidate_questions"]
                             + sub["n_other_questions"], 3)
            # The built-in fixture is synthetic, not sourced from an actual
            # Cochrane republication, so it is honestly all "other".
            self.assertEqual(sub["n_temporal_candidate_questions"], 0)
            self.assertEqual(sub["temporal_candidate_questions"], {})

    def test_an_out_of_range_theta_fails_before_any_generation(self):
        """Not merely rejected, but rejected up front: a degenerate theta
        discovered part-way through a real run wastes the GPU session."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                e2e.main([
                    "--output-dir", tmp, "--n-questions", "3", "--theta", "1.5",
                ])
            self.assertFalse((Path(tmp) / "results.jsonl").exists())


class TemporalSubgroupBreakdownTests(unittest.TestCase):
    """Unit-level check of the grouping logic itself, with a genuine mix of
    temporal-candidate and non-temporal questions - the case the built-in
    fixture (all-False, being synthetic) never exercises."""

    def rows_for(self, question_id, system, token_f1):
        from experiments.evaluation import rag_metrics as rm
        return rm.MetricRow(
            question_id=question_id, system=system, exact_match=0.0,
            token_f1=token_f1, rouge_l_f1=0.0, context_precision=None,
            context_recall=None, context_f1=None, groundedness=0.0,
        )

    def test_rows_split_by_the_temporal_candidate_flag(self):
        rows = [
            self.rows_for("Q1", "baseline", 0.2),
            self.rows_for("Q1", "proposed", 0.9),   # temporal: proposed wins
            self.rows_for("Q2", "baseline", 0.5),
            self.rows_for("Q2", "proposed", 0.5),   # non-temporal: tied
        ]
        flags = {"Q1": True, "Q2": False}

        breakdown = e2e.temporal_subgroup_breakdown(rows, flags)

        self.assertEqual(breakdown["n_temporal_candidate_questions"], 1)
        self.assertEqual(breakdown["n_other_questions"], 1)
        temporal = breakdown["temporal_candidate_questions"]
        other = breakdown["other_questions"]
        self.assertAlmostEqual(
            temporal["proposed"]["token_f1"] - temporal["baseline"]["token_f1"],
            0.7,
        )
        self.assertAlmostEqual(
            other["proposed"]["token_f1"] - other["baseline"]["token_f1"], 0.0,
        )

    def test_a_question_absent_from_flags_counts_as_non_temporal(self):
        """flags.get(..., False): a question freezing never marked (an
        older manifest, or a non-EvaluationQuestion source) must not raise
        or vanish from the total - it counts as "other", not "unknown"."""
        rows = [self.rows_for("Q9", "baseline", 1.0)]
        breakdown = e2e.temporal_subgroup_breakdown(rows, flags={})
        self.assertEqual(breakdown["n_temporal_candidate_questions"], 0)
        self.assertEqual(breakdown["n_other_questions"], 1)  # scored, not dropped
        self.assertEqual(len(breakdown["other_questions"]), 1)
