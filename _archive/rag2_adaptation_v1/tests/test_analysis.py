"""The measurements on the answers of the realigned study: accuracy, unsupported-answer indicators, what the criteria arms changed, retrieval and the reading of the primary difference."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import analysis as A
from tests.fixtures import N, S, answer, item, world, write


class PointDateTests(unittest.TestCase):
    def test_the_date_of_a_passage_is_the_midpoint_of_the_bounds_on_its_first_availability(self):
        import datetime as dt
        self.assertEqual(A.point_date({"lower": "2020-01-01", "upper": "2020-01-31"}), dt.date(2020, 1, 16))
        self.assertEqual(A.point_date({"lower": "2020-03-05", "upper": "2020-03-05"}), dt.date(2020, 3, 5))
        self.assertEqual(A.point_date({"lower": "2012-01-01", "upper": "2012-12-31"}), dt.date(2012, 7, 1))


class AnalysisTests(unittest.TestCase):
    def test_a_positive_difference_has_no_minimum_size_and_zero_is_no_evidence(self):
        primary = {"diff_a_minus_b": 0.001, "ci95": [-0.02, 0.04], "mcnemar_p": 0.5, "confirmed": False}
        self.assertEqual(A.primary_reading(primary, "confirm"), "positive but not confirmed")
        self.assertEqual(A.primary_reading(dict(primary, diff_a_minus_b=0.0), "confirm"), "no evidence")
        self.assertEqual(A.primary_reading(dict(primary, diff_a_minus_b=-0.001), "confirm"), "no evidence")
        self.assertEqual(A.primary_reading(dict(primary, diff_a_minus_b=0.004, ci95=[0.001, 0.007], mcnemar_p=0.01, confirmed=True), "confirm"),
                         "confirmed")

    def test_a_significantly_worse_verifier_is_never_confirmed(self):
        items, answers = world(n=60, r2v_extra=0)
        for k, iid in enumerate(sorted(items)):                       # R2V is wrong wherever R2 is right on 25 items, and right nowhere new
            if k < 25:
                answers[(iid, "R2V")]["verdict"] = N
        primary = A.primary_result(items, answers)
        self.assertLess(primary["mcnemar_p"], A.ALPHA)
        self.assertLess(primary["diff_a_minus_b"], 0)
        self.assertFalse(primary["confirmed"])
        self.assertEqual(A.primary_reading(primary, "confirm"), "no evidence")

    def test_an_admitted_study_counts_for_the_update_window_only_if_it_surely_appeared_after_the_previous_version(self):
        it = dict(item("MC-1", split="confirm"))                      # previous version 2010-01-01, newest 2015-06-01
        rec = answer("MC-1", "R2", S, admitted=("p1", "p2", "p3"))
        rec.update(admitted_lower=["2010-01-01", "2010-01-02", "2012-01-01"], admitted_upper=["2012-01-01", "2012-01-01", "2015-06-01"],
                   admitted_stratum=["RCT", "RCT", "RCT"])
        row = A.retrieval_rows({"MC-1": it}, {("MC-1", "R2"): rec}, ["R2"], {})["R2"]
        self.assertAlmostEqual(row["update_window_share"], 2 / 3, places=5)       # the first appeared no later than the previous version

    def test_primary_reading(self):
        primary = {"diff_a_minus_b": 0.02, "ci95": [0.005, 0.04], "mcnemar_p": 0.01, "confirmed": True}
        self.assertEqual(A.primary_reading(primary, "confirm"), "confirmed")
        self.assertEqual(A.primary_reading(primary, "ad"), "confirmed")
        self.assertEqual(A.primary_reading(dict(primary, confirmed=False), "confirm"), "positive but not confirmed")
        self.assertEqual(A.primary_reading(dict(primary, diff_a_minus_b=0.009, confirmed=False), "confirm"), "positive but not confirmed")
        self.assertEqual(A.primary_reading(dict(primary, diff_a_minus_b=-0.02, confirmed=False), "confirm"), "no evidence")
        self.assertTrue(A.primary_reading(primary, "dev").startswith("development estimate only"))
        self.assertEqual(A.primary_reading(None, "confirm"), "not run")

    def test_unsupported_answer_indicators(self):
        items, answers = world(n=10)
        iid = sorted(items)[0]
        answers[(iid, "B1")]["text"] = "VERDICT: SUPPORTED. A 2019 meta-analysis shows it."
        rows = A.hallucination_rows(items, answers, ["B0", "B1"])
        self.assertAlmostEqual(rows["B1"]["anachronism_rate"], 0.1)
        self.assertIsNone(rows["B0"]["unsupported_decisive_rate"])            # B0 has no evidence
        self.assertAlmostEqual(rows["B1"]["unsupported_decisive_rate"], 0.1)  # one decisive answer cites nothing
    def test_the_ad_split_is_read_like_the_held_out_split_and_no_other_split_is_read(self):
        primary = {"diff_a_minus_b": 0.02, "ci95": [-0.01, 0.05], "mcnemar_p": 0.2, "confirmed": False}
        self.assertEqual(A.primary_reading(primary, "ad"), "positive but not confirmed")
        self.assertEqual(A.primary_reading(dict(primary, confirmed=True), "ad"), "confirmed")
        self.assertEqual(A.primary_reading(dict(primary, diff_a_minus_b=-0.01), "ad"), "no evidence")
        self.assertTrue(A.primary_reading(primary, "fresh").startswith("development estimate only"))         # the fresh test is read by analyze_fresh

    def test_report_on_a_synthetic_world(self):
        items, answers = world()
        rep = A.build_report(items, answers, "confirm")
        self.assertEqual(rep["arms"], ["B0", "B1", "R2", "R2C", "R2V"])
        p = rep["primary"]
        self.assertAlmostEqual(p["diff_a_minus_b"], 6 / 40)
        self.assertEqual((p["a_only_correct"], p["b_only_correct"]), (6, 0))
        self.assertEqual(rep["reading"], "confirmed")
        self.assertIn("R2 vs B1", rep["secondary"])
        self.assertEqual(rep["verification"]["R2V"]["changes_fixed"], 6)
        self.assertEqual(rep["verification"]["R2V"]["changes_broke"], 0)
        self.assertEqual(rep["retrieval"]["R2"]["evidence_mix"], {"SR/MA": 0.0, "RCT": 1.0, "other": 0.0})
        self.assertEqual(rep["case_study_alzheimers"]["n"], 1)

    def test_the_label_stable_ids_are_those_where_both_labelers_agree_on_the_newest_version(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            write(d / "audit.jsonl", [{"item_id": "a", "which": "newest", "gold": S, "relabel": S},
                                      {"item_id": "b", "which": "newest", "gold": S, "relabel": N},
                                      {"item_id": "a", "which": "previous", "gold": N, "relabel": N}])
            self.assertEqual(A.stable_ids_from_audit(d / "audit.jsonl"), ["a"])
            self.assertIsNone(A.stable_ids_from_audit(d / "absent.jsonl"))
            self.assertIsNone(A.stable_ids_from_audit(None))

    def test_the_answers_come_from_their_own_folder_and_only_for_usable_items_and_known_arms(self):
        with TemporaryDirectory() as tmp:
            data, results = Path(tmp) / "data", Path(tmp) / "results"
            data.mkdir()
            results.mkdir()
            items, answers = world(n=6)
            write(data / "benchmark.jsonl", list(items.values()) + [dict(item("MC-999", split="confirm"), likely_label_noise=True),
                                                                     dict(item("MC-998", split="dev"))])
            write(results / "answers_confirm.jsonl", [r for (i, a), r in answers.items() if a in ("B0", "B1")])
            write(results / "rag2_answers_confirm.jsonl", [r for (i, a), r in answers.items() if a.startswith("R2")]
                  + [answer("MC-999", "R2", S), answer("MC-000", "ZZ", S)])
            got_items, got = A.load_split(data, "confirm", results)
            self.assertEqual(set(got_items), set(items))                                  # not the noisy item, not the dev item
            self.assertEqual(set(got), set(answers))                                      # not the noisy item's answer, not the unknown arm
            self.assertEqual(A.load_split(data, "confirm")[1], {})                        # without a folder the answers are looked for in ``data``
            self.assertEqual(A.present_arms(got), ["B0", "B1", "R2", "R2C", "R2V"])

    def test_the_retrieval_rows_describe_what_each_evidence_arm_admitted(self):
        items, answers = world(n=4)
        rows = A.retrieval_rows(items, answers, A.present_arms(answers), {})
        self.assertNotIn("B0", rows)                                                     # B0 admits nothing and is not an evidence arm
        self.assertEqual(rows["R2"]["mean_admitted"], 2.0)
        self.assertEqual(rows["R2"]["share_without_evidence"], 0.0)
        self.assertEqual(rows["R2"]["evidence_mix"], {"SR/MA": 0.0, "RCT": 1.0, "other": 0.0})
        self.assertIsNone(rows["R2"]["directness_at_k"])                                 # no directness judgements were given
        self.assertEqual(rows["R2"]["jaccard_with_B1"], 1.0)                              # R2 and B1 admitted the same two studies in this world


class TruncationTests(unittest.TestCase):
    def test_the_analysis_counts_answers_written_after_the_abstracts_were_shortened(self):
        items, answers = world(n=6)
        answers[("MC-000", "R2")]["context_truncated"] = True
        rep = A.build_report(items, answers, "confirm")
        self.assertEqual(rep["context_truncated"], {"R2": 1})


if __name__ == "__main__":
    unittest.main()
