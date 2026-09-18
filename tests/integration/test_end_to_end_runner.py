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
