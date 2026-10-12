"""The blinded analysis of the pre-registered test on fresh questions: the blinding, the decision rule and the files written at the end of the run (synthetic answers)."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.medchange import analyze_fresh as AF


def synthetic(n=60, gain=True):
    labs = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"]
    rng = np.random.default_rng(5)
    items, answers = {}, {}
    for k in range(n):
        gold = labs[k % 3]
        items[f"FR-{k:05d}"] = {"item_id": f"FR-{k:05d}", "kind": "unchanged", "newest": {"label": gold, "date": "2015-06-01", "row": k},
                                "previous": {"label": gold, "date": "2015-06-01", "row": k}}           # a review with a single version
        r2 = gold if rng.random() < 0.4 else "SUPPORTED"
        r2v = gold if (rng.random() < 0.6 if gain else r2 == gold) else "SUPPORTED"
        admitted = ["p1", "p2", "p3"][:k % 4]                                                              # 0, 1, 2 or 3 abstracts of one trial type
        for arm, v in (("R2", r2), ("R2C", r2), ("R2V", r2v)):
            answers[(f"FR-{k:05d}", arm)] = {"item_id": f"FR-{k:05d}", "arm": arm, "verdict": v, "text": "VERDICT", "admitted": admitted,
                                             "admitted_lower": ["2010-01-01"] * len(admitted), "admitted_upper": ["2010-12-31"] * len(admitted),
                                             "admitted_stratum": ["RCT"] * len(admitted),
                                             "valid": True, "changed": arm == "R2V" and r2v != r2, "draft_verdict": r2, "fallback": None}
    return items, answers


class BlindedAnalysisTests(unittest.TestCase):
    def test_an_incomplete_run_computes_nothing(self):
        items, answers = synthetic()
        answers.pop(("FR-00003", "R2V"))
        with self.assertRaises(AF.Blinded):
            AF.analyse(items, answers, expected=60, iterations=50)
        with self.assertRaises(AF.Blinded):
            AF.analyse(*synthetic(), expected=61, iterations=50)                     # the number of questions must be the registered one

    def test_main_prints_only_a_blinded_notice_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            items, answers = synthetic(30)
            (d / "benchmark.jsonl").write_text("\n".join(json.dumps(dict(v, split="fresh", likely_label_noise=False)) for v in items.values()),
                                               encoding="utf-8")
            (d / "rag2_answers_fresh.jsonl").write_text("\n".join(json.dumps(a) for (i, arm), a in answers.items() if i != "FR-00001"),
                                                        encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = AF.main(["--data-dir", str(d), "--results-dir", str(d / "res"), "--expected", "30"])
            self.assertEqual(code, 3)
            self.assertIn("BLINDED", out.getvalue())
            self.assertNotIn("macro", out.getvalue().lower())
            self.assertFalse((d / "res").exists())

    def test_a_complete_run_gives_the_pre_registered_outcome_and_is_reproducible(self):
        items, answers = synthetic(60)
        a = AF.analyse(items, answers, expected=60, iterations=300)
        b = AF.analyse(items, answers, expected=60, iterations=300)
        self.assertEqual(a["primary"], b["primary"])
        self.assertIn(a["primary"]["reading"], ("confirmed", "positive but not confirmed", "no evidence"))
        self.assertEqual(set(a["secondary"]), {"macro_f1_R2V_minus_R2C", "refuted_recall_R2V_minus_R2", "accuracy_R2V_minus_R2",
                                               "accuracy_R2V_minus_R2C"})
        text = AF.to_markdown(a)
        self.assertIn("Primary outcome", text)

    def test_an_unparsed_answer_counts_as_wrong_and_does_not_crash(self):
        items, answers = synthetic(60)
        answers[("FR-00004", "R2")]["verdict"] = None
        rep = AF.analyse(items, answers, expected=60, iterations=50)
        self.assertEqual(rep["arms"]["R2"]["unparsed"], 1)
        self.assertIn("Unparsed answers", AF.to_markdown(rep))

    def test_the_decision_rule(self):
        self.assertEqual(AF.read_primary({"difference": 0.04, "ci95": [0.01, 0.07]})["reading"], "confirmed")
        self.assertEqual(AF.read_primary({"difference": 0.04, "ci95": [-0.01, 0.09]})["reading"], "positive but not confirmed")
        self.assertEqual(AF.read_primary({"difference": -0.01, "ci95": [-0.04, 0.01]})["reading"], "no evidence")
        self.assertTrue(AF.read_primary({"difference": 0.0, "ci95": [-0.015, 0.015]})["gain_of_0.02_or_more_excluded"])
        self.assertFalse(AF.read_primary({"difference": 0.0, "ci95": [-0.03, 0.03]})["gain_of_0.02_or_more_excluded"])
        self.assertEqual((AF.EXPECTED, AF.ITERATIONS, AF.PRIMARY_SEED, AF.ARMS), (850, 10000, "fresh-macro-f1", ("R2", "R2C", "R2V")))

    def test_the_dementia_comparison_is_descriptive_and_checks_the_sign(self):
        cb = {"splits": {"ad": {"n": 208, "paired": {"R2V vs R2": {"macro_f1": {"difference": 0.08, "ci95": [0.02, 0.13]}}}}}}
        rep = AF.analyse(*synthetic(60), expected=60, iterations=100, class_balance=cb)
        d = rep["alzheimers_dementia_descriptive"]
        self.assertEqual(d["n"], 208)
        self.assertIn("not confirmation", d["note"])
        self.assertIsInstance(d["sign_agrees_with_fresh"], bool)


class RetrievalDescriptionTests(unittest.TestCase):
    """The description of the retrieval that was added after the run (``docs/protocol.md`` §10.9, item 10)."""

    def test_the_block_describes_the_admitted_abstracts_and_leaves_the_update_window_out(self):
        rep = AF.analyse(*synthetic(60), expected=60, iterations=50)
        r = rep["retrieval"]
        self.assertEqual(set(r["arms"]), {"R2", "R2C", "R2V"})
        self.assertEqual(set(r["arms"]["R2"]), set(AF.RETRIEVAL_FIELDS))
        self.assertNotIn("update_window_share", r["arms"]["R2"])
        self.assertEqual(r["arms"]["R2"]["n"], 60)
        self.assertEqual(r["arms"]["R2"]["mean_admitted"], 1.5)                                # 0, 1, 2 and 3 abstracts in turn
        self.assertEqual(r["arms"]["R2"]["share_without_evidence"], 0.25)
        self.assertEqual(r["arms"]["R2"]["evidence_mix"], {"SR/MA": 0.0, "RCT": 1.0, "other": 0.0})
        self.assertAlmostEqual(r["arms"]["R2"]["mean_age_years"], 4.91, places=2)              # from the midpoint of 2010 to 2015-06-01
        self.assertEqual(r["single_version"], 60)

    def test_one_row_describes_the_three_arms_when_they_read_the_same_abstracts(self):
        text = AF.to_markdown(AF.analyse(*synthetic(60), expected=60, iterations=50))
        self.assertIn("## Retrieval (descriptive)", text)
        self.assertIn("| R2, R2C, R2V | 1.5 | 25.0% | 0.0% / 100.0% / 0.0% | 4.91 |", text)
        self.assertIn("60 of the 60 questions come from reviews with a single version", text)
        self.assertIn("they are not outcomes", text)

    def test_arms_that_read_different_abstracts_get_a_row_each(self):
        items, answers = synthetic(60)
        answers[("FR-00003", "R2V")].update(admitted=["p1"], admitted_lower=["2010-01-01"], admitted_upper=["2010-12-31"], admitted_stratum=["RCT"])
        text = AF.to_markdown(AF.analyse(items, answers, expected=60, iterations=50))
        self.assertNotIn("| R2, R2C, R2V |", text)
        for arm in ("R2", "R2C", "R2V"):
            self.assertIn(f"| {arm} |", text)

    def test_questions_without_evidence_show_no_mix_and_no_age(self):
        items, answers = synthetic(12)
        for key in answers:
            answers[key].update(admitted=[], admitted_lower=[], admitted_upper=[], admitted_stratum=[])
        rep = AF.analyse(items, answers, expected=12, iterations=50)
        self.assertEqual((rep["retrieval"]["arms"]["R2"]["share_without_evidence"], rep["retrieval"]["arms"]["R2"]["evidence_mix"]), (1.0, None))
        self.assertIn("| R2, R2C, R2V | 0.0 | 100.0% | — | — |", AF.to_markdown(rep))


class CompleteRunTests(unittest.TestCase):
    """The end of the registered run: the files the analysis writes and the optional inputs it reads (exercised before the real run ends)."""

    def _run(self, tmp, n=60, with_optional=True):
        d = Path(tmp)
        items, answers = synthetic(n)
        (d / "benchmark.jsonl").write_text("\n".join(json.dumps(dict(v, split="fresh", likely_label_noise=False)) for v in items.values()),
                                           encoding="utf-8")
        (d / "rag2_answers_fresh.jsonl").write_text("\n".join(json.dumps(a) for a in answers.values()), encoding="utf-8")
        res = d / "res"
        res.mkdir()
        if with_optional:
            results = Path(AF.__file__).resolve().parent / "results"
            shutil.copy(results / "class_balance.json", res / "class_balance.json")
            shutil.copy(results / "label_audit_confirm.json", res / "label_audit_fresh.json")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = AF.main(["--data-dir", str(d), "--results-dir", str(res), "--expected", str(n)])
        return code, out.getvalue(), res

    def test_a_complete_run_writes_the_json_and_the_findings_with_every_registered_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, printed, res = self._run(tmp)
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in res.iterdir()),
                             ["RAG2_FINDINGS_FRESH.md", "class_balance.json", "label_audit_fresh.json", "rag2_analysis_fresh.json"])
            written = (res / "RAG2_FINDINGS_FRESH.md").read_text(encoding="utf-8")
            self.assertEqual(written + "\n", printed)                                  # printed with print(): one more newline
            for needle in ("## Primary outcome", "## Secondary outcomes", "## Alzheimer's disease and related dementias (descriptive)",
                           "## Label audit (sample)", "questions of the Alzheimer's disease set", "Reading by the rule of `docs/evaluation.md` §5"):
                self.assertIn(needle, written)
            self.assertNotIn("dementia questions", written)
            rep = json.loads((res / "rag2_analysis_fresh.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["n"], 60)
            self.assertIn(rep["primary"]["reading"], ("confirmed", "positive but not confirmed", "no evidence"))

    def test_the_optional_inputs_may_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, printed, res = self._run(tmp, with_optional=False)
            self.assertEqual(code, 0)
            self.assertNotIn("Label audit", printed)
            self.assertNotIn("Alzheimer's disease and related dementias (descriptive)", printed)
            self.assertIn("## Retrieval (descriptive)", printed)                       # this block needs only the answers

    def test_a_lower_end_exactly_at_zero_is_not_above_zero(self):
        self.assertEqual(AF.read_primary({"difference": 0.04, "ci95": [0.0, 0.08]})["reading"], "positive but not confirmed")
        self.assertEqual(AF.read_primary({"difference": 0.0, "ci95": [-0.03, 0.03]})["reading"], "no evidence")

    def test_a_significant_p_value_or_a_positive_interval_alone_does_not_confirm_the_accuracy_difference(self):
        base = {"a": "R2V", "b": "R2", "n": 850, "a_only_correct": 5, "b_only_correct": 1}
        interval_only = AF.accuracy_reading(dict(base, diff_a_minus_b=0.03, ci95=[0.005, 0.055], mcnemar_p=0.06))
        self.assertEqual(interval_only["reading"], "positive but not confirmed")
        negative = AF.accuracy_reading(dict(base, diff_a_minus_b=-0.03, ci95=[-0.05, -0.01], mcnemar_p=0.001))
        self.assertEqual(negative["reading"], "no evidence")

    def test_the_accuracy_difference_is_read_by_the_rule_of_the_evaluation(self):
        base = {"a": "R2V", "b": "R2", "n": 850, "a_only_correct": 1, "b_only_correct": 1}
        self.assertIsNone(AF.accuracy_reading(None))
        small = AF.accuracy_reading(dict(base, diff_a_minus_b=0.005, ci95=[-0.02, 0.03], mcnemar_p=0.7))      # no minimum size: any positive difference is positive
        self.assertEqual((small["reading"], small["confirmed"]), ("positive but not confirmed", False))
        met = AF.accuracy_reading(dict(base, diff_a_minus_b=0.02, ci95=[-0.01, 0.05], mcnemar_p=0.3))
        self.assertEqual((met["reading"], met["confirmed"]), ("positive but not confirmed", False))
        both = AF.accuracy_reading(dict(base, diff_a_minus_b=0.03, ci95=[0.005, 0.055], mcnemar_p=0.01))
        self.assertEqual((both["reading"], both["confirmed"]), ("confirmed", True))
        wide = AF.accuracy_reading(dict(base, diff_a_minus_b=0.03, ci95=[-0.001, 0.06], mcnemar_p=0.01))      # p below .05 but the interval reaches 0
        self.assertEqual(wide["reading"], "positive but not confirmed")
        for diff in (0.0, -0.01):
            self.assertEqual(AF.accuracy_reading(dict(base, diff_a_minus_b=diff, ci95=[-0.04, 0.02], mcnemar_p=0.5))["reading"], "no evidence")

    def test_only_the_comparison_with_the_baseline_carries_a_reading(self):
        rep = AF.analyse(*synthetic(60), expected=60, iterations=50)
        self.assertIn("reading", rep["secondary"]["accuracy_R2V_minus_R2"])
        self.assertNotIn("reading", rep["secondary"]["accuracy_R2V_minus_R2C"])


class PreRegisteredIntervalTests(unittest.TestCase):
    """The pre-registered wrapper around the shared bootstrap: its seeding and its stored digits."""

    def test_stored_differences_keep_six_decimals_so_the_rule_follows_the_exact_interval(self):
        self.assertEqual(AF.DIGITS, 6)
        gold = np.array([0, 1, 2])
        r = AF.paired_interval(gold, np.array([0, 1, 0]), np.array([0, 0, 0]), AF.macro_f1, "t", iterations=20)
        self.assertEqual(r["difference"], round(r["difference"], 6))
        self.assertNotEqual(r["difference"], round(r["difference"], 4))               # 1/3-type values would be cut by four decimals
        self.assertEqual(AF.read_primary({"difference": 0.00003, "ci95": [0.00002, 0.01]})["reading"], "confirmed")

    def test_the_interval_is_the_2_5_to_97_5_percentile_range_of_the_resampled_differences(self):
        calls = iter(range(10 ** 6))

        def metric(g, p):                                  # the first two calls are the point estimate; then resample k returns k - 1 and 0
            index = next(calls)
            return float(index // 2 - 1) if index >= 2 and index % 2 == 0 else 0.0
        r = AF.paired_interval(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2]), metric, "t", iterations=1000)
        self.assertEqual(r["difference"], 0.0)
        self.assertAlmostEqual(r["ci95"][0], 0.025 * 999, places=6)           # resampled differences are 0, 1, ..., 999
        self.assertAlmostEqual(r["ci95"][1], 0.975 * 999, places=6)

    def test_both_systems_are_resampled_with_the_same_questions(self):
        accuracy = lambda g, p: float((g == p).mean())
        gold, same = np.array([0, 1, 2, 0, 1, 2, 0, 1]), np.array([0, 1, 1, 0, 2, 2, 1, 1])
        r = AF.paired_interval(gold, same, same.copy(), accuracy, "t", iterations=200)
        self.assertEqual((r["difference"], r["ci95"]), (0.0, [0.0, 0.0]))     # identical systems never differ, in any resample


if __name__ == "__main__":
    unittest.main()
