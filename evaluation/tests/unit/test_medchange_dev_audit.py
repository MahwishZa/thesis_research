"""The dev audit (synthetic data only; no benchmark, no answers file, no network)."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import dev_audit as D

S, R, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"


def item(item_id, kind, label, date="2020-01-01"):
    return {"item_id": item_id, "kind": kind, "split": "dev", "likely_label_noise": False,
            "newest": {"label": label, "date": date}}


def answer(item_id, arm, verdict, admitted=("1", "2", "3", "4", "5"), upper=None, text=None):
    return {"item_id": item_id, "arm": arm, "verdict": verdict, "admitted": list(admitted),
            "admitted_upper": list(upper if upper is not None else ["2019-01-01"] * len(admitted)),
            "text": text if text is not None else f"VERDICT: {verdict} {item_id}{arm}"}


def small_world():
    items = {
        "a": item("a", "changed", S), "b": item("b", "changed", R), "c": item("c", "changed", N),
        "d": item("d", "changed", S), "e": item("e", "unchanged", S), "f": item("f", "unchanged", N),
    }
    verdicts = {   # arm -> verdicts in the order a..f
        "B0": [S, S, S, S, S, S], "B1": [S, N, N, S, S, N], "B2": [S, S, N, N, S, S],
        "B3": [N, N, N, S, S, N], "P": [S, S, S, S, N, S], "C1": [S, S, S, N, N, S],
    }
    answers = {}
    for arm, vs in verdicts.items():
        for item_id, v in zip("abcdef", vs):
            answers[(item_id, arm)] = answer(item_id, arm, v)
    return items, answers


class BaselineAndClassTests(unittest.TestCase):
    def test_constant_baseline(self):
        items, _ = small_world()
        changed = D.constant_baseline(items, "changed")
        self.assertEqual(changed["n"], 4)
        self.assertAlmostEqual(changed["accuracy_if_always"][S], 0.5)
        self.assertAlmostEqual(changed["accuracy_if_always"][R], 0.25)
        self.assertEqual(changed["counts"][N], 1)
        self.assertEqual(D.constant_baseline(items, None)["n"], 6)

    def test_class_stats(self):
        items, answers = small_world()
        s = D.class_stats(items, answers, "B1", "changed")   # verdicts a..d: S N N S, gold S R N S
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["accuracy"], 3 / 4)
        self.assertAlmostEqual(s["recall"][S], 1.0)
        self.assertAlmostEqual(s["recall"][R], 0.0)
        self.assertAlmostEqual(s["recall"][N], 1.0)
        self.assertAlmostEqual(s["nei_share"], 2 / 4)

    def test_class_with_no_items_has_no_recall(self):
        items = {"a": item("a", "changed", S)}
        answers = {("a", "B1"): answer("a", "B1", S)}
        self.assertIsNone(D.class_stats(items, answers, "B1", "changed")["recall"][R])


class OrderSensitivityTests(unittest.TestCase):
    def test_categories(self):
        base = ["1", "2", "3", "4", "5"]
        answers = {
            ("a", "B1"): answer("a", "B1", S, base, text="same"),
            ("a", "B2"): answer("a", "B2", S, base, text="same"),                      # identical list
            ("a", "B3"): answer("a", "B3", N, ["5", "4", "3", "2", "1"]),            # same papers, other order
            ("a", "P"): answer("a", "P", S, ["1", "2", "3", "9", "8"]),               # jaccard 3/7 = 0.43
            ("a", "C1"): answer("a", "C1", S, ["7", "8", "9", "10", "11"]),           # no overlap
        }
        rep = D.order_sensitivity(answers, ["a"])
        cats = rep["categories"]
        self.assertEqual(cats["identical list"]["pairs"], 1)
        self.assertEqual(rep["identical_list_identical_text"], {"identical": 1, "pairs": 1})
        # B1-B3 and B2-B3 hold the same papers in another order (verdicts S/N and S/N: never the same)
        self.assertEqual(cats["same papers, different order"]["pairs"], 2)
        self.assertEqual(cats["same papers, different order"]["same_verdict"], 0.0)
        # B1-P, B2-P and B3-P overlap 3/7 = 0.43; P-C1 overlaps 2/8 = 0.25 (band edge is inclusive)
        self.assertEqual(cats["overlap 0.25 to 0.66"]["pairs"], 4)
        self.assertAlmostEqual(cats["overlap 0.25 to 0.66"]["same_verdict"], 3 / 4)
        # B1-C1, B2-C1 and B3-C1 share nothing
        self.assertEqual(cats["overlap below 0.25"]["pairs"], 3)
        self.assertAlmostEqual(cats["overlap below 0.25"]["same_verdict"], 2 / 3)
        self.assertEqual(sum(v["pairs"] for v in cats.values()), 10)

    def test_text_difference_is_counted(self):
        base = ["1", "2", "3", "4", "5"]
        answers = {("a", arm): answer("a", arm, S, base, text=f"text {arm}") for arm in D.EVIDENCE_ARMS}
        rep = D.order_sensitivity(answers, ["a"])
        self.assertEqual(rep["identical_list_identical_text"], {"identical": 0, "pairs": 10})


class AgeTests(unittest.TestCase):
    def test_evidence_ages_in_years(self):
        it = item("a", "changed", S, date="2020-01-01")
        ans = answer("a", "B1", S, ["1", "2"], ["2019-01-01", "2010-01-01"])
        ages = D.evidence_ages(it, ans)
        self.assertAlmostEqual(ages[0], 365 / 365.25)
        self.assertAlmostEqual(ages[1], 3652 / 365.25)

    def test_missing_dates_are_skipped_not_zero(self):
        it = item("a", "changed", S)
        self.assertIsNone(D.evidence_ages(it, answer("a", "B1", S, [], [])))
        self.assertEqual(len(D.evidence_ages(it, answer("a", "B1", S, ["1", "2"], [None, "2019-01-01"]))), 1)

    def test_age_by_gold(self):
        items = {"a": item("a", "changed", S), "b": item("b", "changed", R)}
        answers = {("a", "B1"): answer("a", "B1", S, ["1"], ["2019-01-01"]),
                   ("b", "B1"): answer("b", "B1", R, ["1"], ["2010-01-01"])}
        out = D.age_by_gold(items, answers, "B1")
        self.assertEqual(out[S]["n"], 1)
        self.assertLess(out[S]["mean_age_years"], out[R]["mean_age_years"])
        self.assertIsNone(out[N]["mean_age_years"])


class VoteAndRecalibrationTests(unittest.TestCase):
    def test_majority_vote_and_tie_rule(self):
        items = {"a": item("a", "changed", S), "b": item("b", "changed", R)}
        answers = {("a", "B1"): answer("a", "B1", S), ("a", "B2"): answer("a", "B2", N),
                   ("a", "B3"): answer("a", "B3", N),            # majority N: wrong
                   ("b", "B1"): answer("b", "B1", R), ("b", "B2"): answer("b", "B2", S),
                   ("b", "B3"): answer("b", "B3", N)}            # three-way tie: B1's R: right
        out = D.majority_vote(items, answers, ("B1", "B2", "B3"), "changed")
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["accuracy"], 0.5)

    def test_recalibration_learns_an_informative_verdict(self):
        items, answers = {}, {}
        labels = [S, R, N]
        for k in range(36):
            gold = labels[k % 3]
            iid = f"i{k:02d}"
            items[iid] = item(iid, "changed" if k % 2 == 0 else "unchanged", gold)
            for arm in D.ARMS:
                answers[(iid, arm)] = answer(iid, arm, gold)           # B1 is always right
        rep = D.recalibration_cv(items, answers, folds=3, repeats=3)
        self.assertEqual(rep["n"], 36)
        self.assertEqual(rep["raw_B1_accuracy"], 1.0)
        self.assertGreater(rep["variants"]["B1 verdict"]["accuracy"], 0.95)
        self.assertLess(rep["variants"]["constant prior"]["accuracy"], 0.6)
        self.assertEqual(set(rep["variants"]), {"constant prior", "B1 verdict", "B1 verdict + evidence age",
                                                 "evidence age only"})


class ReportTests(unittest.TestCase):
    def world36(self):
        items, answers = {}, {}
        labels = [S, S, R, N]
        for k in range(24):
            gold = labels[k % 4]
            iid = f"i{k:02d}"
            items[iid] = item(iid, "changed" if k % 3 else "unchanged", gold)
            for arm in D.ARMS:
                answers[(iid, arm)] = answer(iid, arm, gold if (k + len(arm)) % 2 else S,
                                             admitted=[str(k + j) for j in range(5)])
        return items, answers

    def test_audit_and_markdown(self):
        items, answers = self.world36()
        rep = D.audit(items, answers, folds=3, repeats=2)
        text = D.to_markdown(rep)
        for needle in ("Constant answers", "Per-class recall", "Order sensitivity", "Mean age of B1's evidence",
                       "Does refitting help", "Majority vote of B1, B2, B3"):
            self.assertIn(needle, text)
        self.assertEqual(set(rep["arms"]), set(D.ARMS))
        json.dumps(rep)                                                   # plain data

    def test_cli_writes_reports_and_refuses_confirm(self):
        items, answers = self.world36()
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            data.mkdir()
            (data / "benchmark.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in items.values()), encoding="utf-8")
            ans = tmp / "answers.jsonl"
            ans.write_text("".join(json.dumps(r) + "\n" for r in answers.values()), encoding="utf-8")
            out = tmp / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                code = D.main(["--data-dir", str(data), "--answers", str(ans), "--out-dir", str(out),
                               "--repeats", "2"])
            self.assertEqual(code, 0)
            self.assertTrue((out / "dev_audit.md").is_file())
            self.assertIn("constant", json.loads((out / "dev_audit.json").read_text(encoding="utf-8")))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                D.main(["--split", "confirm", "--data-dir", str(data), "--answers", str(ans)])

    def test_cli_reports_missing_inputs(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(D.main(["--data-dir", tmp]), 2)
            data = Path(tmp)
            (data / "benchmark.jsonl").write_text(json.dumps(item("a", "changed", S)) + "\n", encoding="utf-8")
            empty = data / "answers.jsonl"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(D.main(["--data-dir", tmp, "--answers", str(empty)]), 2)


if __name__ == "__main__":
    unittest.main()
