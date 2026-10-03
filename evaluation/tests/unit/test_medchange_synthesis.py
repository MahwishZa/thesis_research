"""Stage-2 synthesis layer and analysis: features, the fitted layer, cross-validation, gate 2,
selection, freezing/prediction and the confirmatory comparison (synthetic data; no model, no network)."""

import json
import math
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from experiments.medchange import analyze_stage2 as AN
from experiments.medchange import arms as A
from experiments.medchange import synthesis as S

LABELS = S.LABELS


def make_world(n_per_class=30, seed=5, informative=True, n_cands=10, split="dev", start=0):
    """Items, pools, stance probabilities and B1 verdicts for a synthetic benchmark."""
    rng = random.Random(seed)
    items, pools, probs, b1 = [], {}, {}, {}
    k = start
    base_probs = {"SUPPORTED": (0.6, 0.1, 0.3), "REFUTED": (0.1, 0.6, 0.3), "NOT ENOUGH INFORMATION": (0.15, 0.15, 0.7)}
    for label in LABELS:
        for _ in range(n_per_class):
            iid = f"MC-{k:04d}"
            k += 1
            items.append({"item_id": iid, "kind": "changed" if k % 3 else "unchanged", "split": split,
                          "likely_label_noise": False, "question": f"Question {k}?",
                          "newest": {"label": label, "date": "2020-01-01"},
                          "previous": {"label": "REFUTED" if label != "REFUTED" else "SUPPORTED", "date": "2010-01-01"}})
            cands = []
            for r in range(1, n_cands + 1):
                year = 2005 + rng.randrange(14)
                cands.append({"pmid": f"{k}-{r}", "rank": r, "lower": f"{year}-01-01", "upper": f"{year}-01-01",
                              "title": "T", "abstract": "A",
                              "pubtypes": rng.choice([[], [], ["Randomized Controlled Trial"], ["Meta-Analysis"]])})
            pools[iid] = {"item_id": iid, "candidates": cands}
            for c in cands:
                base = base_probs[label] if informative else (1 / 3, 1 / 3, 1 / 3)
                noisy = [max(1e-3, b + rng.uniform(-0.15, 0.15)) for b in base]
                total = sum(noisy)
                probs[(iid, c["pmid"])] = tuple(x / total for x in noisy)
            b1[iid] = label if rng.random() < 0.5 else rng.choice(LABELS)
    return items, pools, probs, b1


def write_world(directory: Path, world, split, wording="both", backend="fake:test"):
    items, pools, probs, b1 = world
    wordings = ("A", "B") if wording == "both" else (wording,)
    rows = {
        "benchmark.jsonl": items,
        f"frozen_{split}.jsonl": list(pools.values()),
        f"stance_{split}.jsonl": [{"item_id": i, "pmid": p, "wording": w, "control": False, "backend": backend,
                                   "probs": list(v), "argmax": "supports", "rank": 1}
                                  for (i, p), v in probs.items() for w in wordings],
        f"answers_{split}.jsonl": [{"item_id": i, "arm": "B1", "verdict": v, "admitted": [], "seconds": 1.0}
                                   for i, v in b1.items()],
    }
    for name, data in rows.items():
        mode = "a" if name == "benchmark.jsonl" and (directory / name).exists() else "w"
        with open(directory / name, mode, encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(r) for r in data) + "\n")


class WeightAndFeatureTests(unittest.TestCase):

    def test_study_types_prefer_reviews_then_trials(self):
        self.assertEqual(S.study_type(["Journal Article", "Systematic Review", "Randomized Controlled Trial"]), "SR/MA")
        self.assertEqual(S.study_type(["Meta-Analysis"]), "SR/MA")
        self.assertEqual(S.study_type(["Clinical Trial"]), "RCT")
        self.assertEqual(S.study_type(["Case Reports"]), "other")
        self.assertEqual(S.study_type(None), "other")
        self.assertEqual([S.design_weight(t) for t in (["Meta-Analysis"], ["Randomized Controlled Trial"], [])],
                         [3.0, 2.0, 1.0])

    def test_variant_specs(self):
        self.assertEqual(S.variant_spec("B1R"), {"verdict": True, "stance": False, "recency": False,
                                                 "design": False, "shuffled": False})
        self.assertEqual(S.variant_spec("S0"), {"verdict": False, "stance": True, "recency": False,
                                                "design": False, "shuffled": False})
        self.assertEqual(S.variant_spec("H3C"), {"verdict": True, "stance": True, "recency": True,
                                                 "design": True, "shuffled": True})
        self.assertEqual(S.variant_spec("H2")["design"], True)
        self.assertEqual(S.variant_spec("S1")["recency"], True)
        with self.assertRaises(ValueError):
            S.variant_spec("H4")

    def test_stance_features_match_a_hand_calculation(self):
        probs = [(0.8, 0.1, 0.1), (0.1, 0.7, 0.2)]
        f = S.stance_features(probs, [1.0, 3.0])
        self.assertAlmostEqual(f[0], (1.1 - 2.2) / 4, places=9)       # signed
        self.assertAlmostEqual(f[1], 0.7 / 4, places=9)                # neither share
        self.assertAlmostEqual(f[2], 2 * 1.1 / 4, places=9)            # conflict
        self.assertAlmostEqual(f[3], math.log(1 + 3.3), places=9)      # evidence mass
        self.assertEqual(S.stance_features([], []), [0.0, 1.0, 0.0, 0.0])

    def test_weights_use_the_temporal_policy_and_study_types(self):
        cands = [{"pmid": "a", "lower": "2019-01-01", "upper": "2019-01-01", "pubtypes": ["Meta-Analysis"]},
                 {"pmid": "b", "lower": "2010-01-01", "upper": "2010-01-01", "pubtypes": []}]
        dated = {c["pmid"]: c for c in cands}
        newer = A.recency(cands[0], "2020-01-01")
        older = A.recency(cands[1], "2020-01-01")
        self.assertGreater(newer, older)
        self.assertEqual(S.paper_weights(cands, dated, "2020-01-01", False, False), [1.0, 1.0])
        self.assertEqual(S.paper_weights(cands, dated, "2020-01-01", True, False), [newer, older])
        self.assertEqual(S.paper_weights(cands, dated, "2020-01-01", False, True), [3.0, 1.0])
        self.assertEqual(S.paper_weights(cands, dated, "2020-01-01", True, True), [3.0 * newer, older])

    def test_item_features_need_every_stance_and_shuffling_changes_only_recency_weights(self):
        items, pools, probs, _ = make_world(2, seed=1)
        iid = items[0]["item_id"]
        kw = dict(recency=True, design=False, top_k=8)
        a = S.item_features(iid, pools[iid], "2020-01-01", probs, shuffled=False, **kw)
        b = S.item_features(iid, pools[iid], "2020-01-01", probs, shuffled=True, **kw)
        again = S.item_features(iid, pools[iid], "2020-01-01", probs, shuffled=True, **kw)
        self.assertEqual(b, again)
        self.assertNotEqual(a, b)
        flat = dict(recency=False, design=True, top_k=8)
        self.assertEqual(S.item_features(iid, pools[iid], "2020-01-01", probs, shuffled=False, **flat),
                         S.item_features(iid, pools[iid], "2020-01-01", probs, shuffled=True, **flat))
        missing = {k: v for k, v in probs.items() if k != (iid, pools[iid]["candidates"][0]["pmid"])}
        with self.assertRaises(KeyError):
            S.item_features(iid, pools[iid], "2020-01-01", missing, shuffled=False, **kw)

    def test_load_stance_probs_skips_controls_and_maps_invalid_to_neither(self):
        rows = [{"item_id": "i", "pmid": "1", "wording": "A", "control": False, "probs": [0.5, 0.3, 0.2]},
                {"item_id": "i", "pmid": "2", "wording": "A", "control": False, "probs": None},
                {"item_id": "i", "pmid": "3", "wording": "A", "control": True, "probs": [1, 0, 0]},
                {"item_id": "i", "pmid": "4", "wording": "B", "control": False, "probs": [1, 0, 0]}]
        self.assertEqual(S.load_stance_probs(rows, "A"),
                         {("i", "1"): (0.5, 0.3, 0.2), ("i", "2"): (0.0, 0.0, 1.0)})


    def test_both_wordings_are_averaged_and_a_half_ensemble_is_left_out(self):
        rows = [{"item_id": "i", "pmid": "1", "wording": "A", "control": False, "probs": [0.8, 0.1, 0.1]},
                {"item_id": "i", "pmid": "1", "wording": "B", "control": False, "probs": [0.4, 0.3, 0.3]},
                {"item_id": "i", "pmid": "2", "wording": "A", "control": False, "probs": [1.0, 0.0, 0.0]},
                {"item_id": "i", "pmid": "3", "wording": "A", "control": False, "probs": [0.1, 0.1, 0.8]},
                {"item_id": "i", "pmid": "3", "wording": "B", "control": False, "probs": None}]
        got = S.load_stance_probs(rows, "both")
        self.assertEqual(set(got), {("i", "1"), ("i", "3")})            # paper 2 lacks wording B
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(got[("i", "1")], (0.6, 0.2, 0.2))))
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(got[("i", "3")], (0.05, 0.05, 0.9))))  # invalid = neither


class FittedLayerTests(unittest.TestCase):

    def test_an_intercept_only_model_returns_the_class_frequencies(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        model = S.fit_softmax(np.zeros((100, 0)), y)
        probs = S.predict_proba(model, np.zeros((3, 0)))
        np.testing.assert_allclose(probs[0], [0.5, 0.3, 0.2], atol=1e-4)

    def test_it_learns_a_separable_signal_and_probabilities_sum_to_one(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 40)
        X = np.c_[y + rng.normal(0, 0.15, 120), rng.normal(0, 1, 120)]
        model = S.fit_softmax(X, y)
        probs = S.predict_proba(model, X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertGreater((probs.argmax(axis=1) == y).mean(), 0.9)

    def test_a_stronger_penalty_shrinks_the_weights_and_fitting_is_deterministic(self):
        rng = np.random.default_rng(1)
        y = np.repeat([0, 1, 2], 30)
        X = np.c_[y + rng.normal(0, 0.5, 90)]
        weak, strong = S.fit_softmax(X, y, l2=0.1), S.fit_softmax(X, y, l2=500.0)
        self.assertGreater(np.abs(np.array(weak["W"])[1:]).max(), np.abs(np.array(strong["W"])[1:]).max())
        self.assertEqual(S.fit_softmax(X, y), S.fit_softmax(X, y))

    def test_the_solution_is_stationary(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 3, 80)
        X = rng.normal(0, 1, (80, 3)) + y[:, None] * 0.4
        model = S.fit_softmax(X, y)
        Xb = np.c_[np.ones(80), (X - np.array(model["mean"])) / np.array(model["std"])]
        W = np.array(model["W"])
        P = S._softmax(Xb @ W)
        pen = np.r_[S.INTERCEPT_RIDGE, np.full(3, S.L2)]
        grad = Xb.T @ (P - np.eye(3)[y])[:, 1:] + pen[:, None] * W[:, 1:]
        self.assertLess(np.abs(grad).max(), 1e-6)

    def test_cross_validation_is_deterministic_and_uses_the_same_folds_for_every_variant(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, 60)
        X = rng.normal(0, 1, (60, 2)) + y[:, None]
        a = S.repeated_cv(X, y, repeats=4)
        np.testing.assert_array_equal(a, S.repeated_cv(X, y, repeats=4))
        self.assertEqual(a.shape, (4, 60))
        self.assertGreater(S.cv_accuracy(a, y), 0.5)

    def test_accuracy_recall_and_bagging(self):
        y = np.array([0, 0, 1, 2])
        pred = np.array([[0, 1, 1, 2], [0, 0, 1, 0], [0, 0, 2, 2]])
        self.assertAlmostEqual(S.cv_accuracy(pred, y), 9 / 12)
        self.assertAlmostEqual(S.cv_accuracy(pred, y, np.array([True, True, False, False])), 5 / 6)
        recall = S.cv_recall(pred, y)
        self.assertAlmostEqual(recall["SUPPORTED"], (1 + 2 + 2) / 3 / 2)
        self.assertAlmostEqual(recall["REFUTED"], 2 / 3)
        self.assertEqual(S.bagged_verdicts(pred), [0, 0, 1, 2])

    def test_auc_cases(self):
        self.assertEqual(S.auc([1, 2, 3, 4], [False, False, True, True]), 1.0)
        self.assertEqual(S.auc([4, 3, 2, 1], [False, False, True, True]), 0.0)
        self.assertEqual(S.auc([1, 1, 1, 1], [False, True, False, True]), 0.5)
        with self.assertRaises(ValueError):
            S.auc([1, 2], [True, True])


class SelectionAndGateTests(unittest.TestCase):

    def test_h0_is_kept_unless_a_weighted_variant_beats_it_by_the_margin(self):
        base = {"H0": 0.50, "H1": 0.505, "H2": 0.509, "H3": 0.45}
        self.assertEqual(S.select_variant(base), "H0")
        self.assertEqual(S.select_variant(dict(base, H2=0.51)), "H2")
        self.assertEqual(S.select_variant(dict(base, H1=0.52, H2=0.51)), "H1")
        self.assertEqual(S.select_variant(dict(base, H1=0.51, H3=0.51)), "H1")      # tie: fewer weights

    def test_each_gate_two_condition_can_fail(self):
        cv = {"B1R": 0.52, "H0": 0.54, "H1": 0.50, "H2": 0.50, "H3": 0.50, "S0": 0.50}
        ok = S.gate2(cv, 0.465, 0.65, "H0")
        self.assertEqual(ok["gate2"], "PASS")
        self.assertEqual(S.gate2(cv, 0.465, 0.55, "H0")["gate2"], "FAIL")
        self.assertEqual(S.gate2(dict(cv, H0=0.525), 0.465, 0.65, "H0")["gate2"], "FAIL")
        self.assertEqual(S.gate2(dict(cv, S0=0.46), 0.465, 0.65, "H0")["gate2"], "FAIL")


class FitReportTests(unittest.TestCase):

    def test_an_informative_stance_passes_gate_two_and_beats_the_verdict_alone(self):
        items, pools, probs, b1 = make_world(30, seed=5)
        rep = S.fit_report(items, pools, probs, b1, repeats=3)
        v = rep["cv"]["variants"]
        self.assertGreater(v["S0"]["accuracy"], rep["cv"]["prior_accuracy"] + 0.2)
        self.assertGreater(v["H0"]["accuracy"], v["B1R"]["accuracy"])
        self.assertGreater(rep["stance_direction_auc"], 0.9)
        self.assertEqual(rep["gate2"]["gate2"], "PASS")
        self.assertIn(rep["selected"], S.HYBRIDS)
        self.assertEqual(set(rep["models"]), set(S.VARIANTS))
        self.assertEqual(len(rep["bagged"]["H0"]), len(items))
        self.assertIn("Selected hybrid", S.to_markdown(rep))

    def test_an_uninformative_stance_fails_gate_two(self):
        items, pools, probs, b1 = make_world(30, seed=6, informative=False)
        rep = S.fit_report(items, pools, probs, b1, repeats=3)
        self.assertEqual(rep["gate2"]["gate2"], "FAIL")
        self.assertLess(rep["stance_direction_auc"], 0.6)


class FreezeAndPredictTests(unittest.TestCase):

    def _fit(self, d, out):
        write_world(d, make_world(20, seed=7), "dev")
        with mock.patch("sys.stdout"):
            return S.main(["fit", "--data-dir", str(d), "--out-dir", str(out), "--repeats", "2"])

    def test_fit_freezes_a_model_and_writes_out_of_fold_dev_arms(self):
        with TemporaryDirectory() as tmp:
            d, out = Path(tmp) / "data", Path(tmp) / "results"
            d.mkdir()
            self.assertEqual(self._fit(d, out), 0)
            model = json.loads((out / "synthesis_model.json").read_text(encoding="utf-8"))
            arms = [json.loads(l) for l in (d / "synthesis_dev.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue((out / "synthesis_cv_dev.md").exists())
        self.assertEqual(model["fit_split"], "dev")
        self.assertEqual(model["stance"]["wording"], "both")
        self.assertIn(model["selected"], S.HYBRIDS)
        self.assertEqual(set(model["models"]), set(S.VARIANTS))
        self.assertEqual(len(arms), 60 * len(S.VARIANTS))
        self.assertEqual({r["arm"] for r in arms}, set(S.VARIANTS))

    def test_fit_needs_complete_inputs_and_both_wordings(self):
        with TemporaryDirectory() as tmp:
            d, out = Path(tmp) / "data", Path(tmp) / "results"
            d.mkdir()
            write_world(d, make_world(5, seed=7), "dev", wording="A")
            with mock.patch("sys.stderr"):
                self.assertEqual(S.main(["fit", "--data-dir", str(d), "--out-dir", str(out)]), 2)
                (d / "stance_dev.jsonl").write_text("", encoding="utf-8")
                self.assertEqual(S.main(["fit", "--data-dir", str(d), "--out-dir", str(out)]), 2)

    def test_predict_applies_the_frozen_model_to_the_confirmatory_split_and_refuses_a_different_one(self):
        with TemporaryDirectory() as tmp:
            d, out = Path(tmp) / "data", Path(tmp) / "results"
            d.mkdir()
            self.assertEqual(self._fit(d, out), 0)
            confirm = make_world(10, seed=8, split="confirm", start=1000)
            write_world(d, confirm, "confirm")
            model_path = out / "synthesis_model.json"
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                code = S.main(["predict", "--data-dir", str(d), "--model", str(model_path)])
                rows = [json.loads(l) for l in (d / "synthesis_confirm.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(code, 0)
                self.assertEqual(len(rows), 30 * len(S.VARIANTS))
                self.assertTrue(all(len(r["probs"]) == 3 and r["verdict"] in LABELS for r in rows))
                self.assertTrue((d / "synthesis_confirm.config.json").exists())
                self.assertEqual(S.main(["predict", "--data-dir", str(d), "--model", str(model_path)]), 0)
                other = out / "other.json"
                other.write_text(model_path.read_text(encoding="utf-8").replace('"format": 1', '"format": 1, "x": 1'),
                                 encoding="utf-8")
                self.assertEqual(S.main(["predict", "--data-dir", str(d), "--model", str(other)]), 2)
                self.assertEqual(S.main(["predict", "--data-dir", str(d), "--model", str(out / "absent.json")]), 2)


def analysis_world(n_changed=80, n_unchanged=40, b0_correct=0.4, b1_boost=30):
    """Items plus arms with a known pattern: B1 beats B0 on `b1_boost` extra items; H0 equals B1R."""
    items = {}
    answers = {}
    for k in range(n_changed + n_unchanged):
        iid = f"MC-{k:04d}"
        kind = "changed" if k < n_changed else "unchanged"
        items[iid] = {"item_id": iid, "kind": kind, "newest": {"label": "SUPPORTED", "date": "2020-01-01"},
                      "previous": {"label": "REFUTED", "date": "2010-01-01"}, "split": "dev", "likely_label_noise": False}
        right, wrong = "SUPPORTED", "REFUTED"
        b0 = right if k < int(b0_correct * 120) else wrong
        b1 = right if (k < int(b0_correct * 120) or k < int(b0_correct * 120) + b1_boost) else wrong
        for arm, verdict in (("B0", b0), ("B1", b1), ("B1R", b1), ("H0", b1), ("S0", wrong), ("H1C", b1)):
            answers[(iid, arm)] = {"item_id": iid, "arm": arm, "verdict": verdict, "seconds": 1.0, "admitted": []}
    return items, answers


class DecisionTests(unittest.TestCase):
    """The pre-declared reading: a genuine positive needs every criterion, not only RQ2."""

    @staticmethod
    def report(rq2=True, vs_b1=True, f1_h=0.5, f1_r=0.45, primary="H0", control_ok=True, rq1=True,
               rec_b1=(0.6, 0.2), rec_b0=(0.8, 0.0)):
        def arm(f1, rec):
            return {"macro_f1": f1, "recall": {"SUPPORTED": rec[0], "REFUTED": rec[1], "NOT ENOUGH INFORMATION": 0.5}}
        sec = {f"{primary} vs B1": {"confirmed": vs_b1}}
        if primary in ("H1", "H3"):
            sec[f"{primary} vs {primary}C"] = {"confirmed": control_ok}
        return {"primary_arm": primary,
                "primary_family": {"RQ1 B1 vs B0": {"confirmed": rq1}, f"RQ2 {primary} vs B1R": {"confirmed": rq2}},
                "secondary_family": sec,
                "per_arm": {primary: arm(f1_h, (0.7, 0.3)), "B1R": arm(f1_r, (0.7, 0.2)),
                            "B1": arm(0.4, rec_b1), "B0": arm(0.3, rec_b0)}}

    def test_all_criteria_met_is_a_genuine_positive(self):
        out = AN.decide(self.report(), {"n": 100, "diff_a_minus_b": 0.02})
        self.assertEqual(out["RQ2_tier"], "genuine positive")

    def test_each_missing_criterion_downgrades_to_fragile(self):
        stable = {"n": 100, "diff_a_minus_b": 0.02}
        for kw in ({"vs_b1": False}, {"f1_h": 0.40}, {"primary": "H1", "control_ok": False}):
            self.assertEqual(AN.decide(self.report(**kw), stable)["RQ2_tier"], "fragile positive", kw)
        self.assertEqual(AN.decide(self.report(), {"n": 100, "diff_a_minus_b": -0.01})["RQ2_tier"], "fragile positive")
        self.assertEqual(AN.decide(self.report())["RQ2_tier"], "fragile positive")      # label audit not supplied

    def test_unconfirmed_rq2_is_not_confirmed_whatever_else_holds(self):
        self.assertEqual(AN.decide(self.report(rq2=False), {"n": 9, "diff_a_minus_b": 0.1})["RQ2_tier"], "not confirmed")

    def test_the_retrieval_gain_is_labelled_abstention_when_decisive_recall_did_not_rise(self):
        self.assertIn("abstention", AN.decide(self.report())["RQ1_note"])
        self.assertIn("higher", AN.decide(self.report(rec_b1=(0.9, 0.4), rec_b0=(0.8, 0.0)))["RQ1_note"])
        self.assertIsNone(AN.decide(self.report(rq1=False))["RQ1_note"])

    def test_a_missing_synthesis_arm_is_reported_as_not_run(self):
        rep = self.report()
        del rep["per_arm"]["H0"]
        out = AN.decide(rep)
        self.assertEqual((out["RQ2_tier"], out["RQ1"]), ("not run", "confirmed"))

    def test_stable_difference_is_computed_on_the_stable_items_only(self):
        items = {"a": {"kind": "changed", "newest": {"label": "SUPPORTED"}},
                 "b": {"kind": "changed", "newest": {"label": "SUPPORTED"}}}
        ans = {("a", "H0"): {"verdict": "SUPPORTED"}, ("a", "B1R"): {"verdict": "REFUTED"},
               ("b", "H0"): {"verdict": "REFUTED"}, ("b", "B1R"): {"verdict": "SUPPORTED"}}
        self.assertEqual(AN.stable_difference(items, ans, "H0", "B1R", ["a"]), {"n": 1, "diff_a_minus_b": 1.0})
        self.assertIsNone(AN.stable_difference(items, ans, "H0", "B1R", []))


class AnalysisTests(unittest.TestCase):

    def test_wilson_interval(self):
        lo, hi = AN.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertTrue(math.isnan(AN.wilson(0, 0)[0]))

    def test_a_clear_retrieval_gain_is_confirmed_and_an_identical_layer_is_not(self):
        items, answers = analysis_world()
        rep = AN.stage2_report(items, answers, "H0", "confirm")
        rq1 = rep["primary_family"]["RQ1 B1 vs B0"]
        rq2 = rep["primary_family"]["RQ2 H0 vs B1R"]
        self.assertEqual((rq1["a_only_correct"], rq1["b_only_correct"]), (30, 0))
        self.assertTrue(rq1["confirmed"])
        self.assertLess(rq1["holm_p"], 0.001)
        self.assertEqual((rq2["a_only_correct"], rq2["b_only_correct"]), (0, 0))
        self.assertFalse(rq2["confirmed"])
        self.assertEqual(rep["status"], "confirmatory")
        self.assertEqual(rep["per_arm"]["B1"]["all"]["n"], 120)
        self.assertEqual(rep["per_arm"]["B1"]["changed"]["n"], 80)

    def test_secondary_comparisons_include_the_control_only_for_a_recency_variant(self):
        items, answers = analysis_world()
        self.assertNotIn("H0 vs H1C", AN.stage2_report(items, answers, "H0", "dev")["secondary_family"])
        for iid in items:
            answers[(iid, "H1")] = dict(answers[(iid, "H1C")], arm="H1")
        sec = AN.stage2_report(items, answers, "H1", "dev")["secondary_family"]
        self.assertIn("H1 vs H1C", sec)
        self.assertIn("H1 vs H0", sec)
        self.assertEqual(AN.stage2_report(items, answers, "H1", "dev")["status"].split()[0], "exploratory")

    def test_class_statistics(self):
        items, answers = analysis_world(2, 0)
        answers[("MC-0000", "B1")]["verdict"] = "SUPPORTED"
        answers[("MC-0001", "B1")]["verdict"] = "REFUTED"
        stats = AN.class_stats(items, answers, "B1", ("changed",))
        self.assertEqual(stats["recall"]["SUPPORTED"], 0.5)
        self.assertIsNone(stats["recall"]["REFUTED"])
        self.assertEqual(stats["predicted_share"]["REFUTED"], 0.5)

    def test_cli_writes_the_report_and_refuses_incomplete_confirmatory_arms(self):
        items, answers = analysis_world(20, 10, b1_boost=8)
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "benchmark.jsonl").write_text("\n".join(json.dumps(i) for i in items.values()) + "\n", encoding="utf-8")
            by_file = {"answers_dev.jsonl": ("B0", "B1"), "synthesis_dev.jsonl": ("B1R", "H0")}
            for name, arms in by_file.items():
                rows = [r for (i, a), r in answers.items() if a in arms]
                (d / name).write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(AN.main(["--split", "dev", "--data-dir", str(d), "--primary", "H0",
                                          "--out-dir", str(d / "out")]), 0)
                self.assertTrue((d / "out" / "stage2_analysis_dev.md").exists())
                self.assertEqual(AN.main(["--split", "dev", "--data-dir", str(d), "--primary", "H3"]), 2)
                self.assertEqual(AN.main(["--split", "dev", "--data-dir", str(d)]), 2)       # no frozen model
                confirm_items = {k: dict(v, split="confirm") for k, v in items.items()}
                (d / "benchmark.jsonl").write_text("\n".join(json.dumps(i) for i in confirm_items.values()) + "\n",
                                                   encoding="utf-8")
                for name, arms in by_file.items():
                    rows = [r for (i, a), r in answers.items() if a in arms][:-1]       # one answer missing
                    (d / name.replace("dev", "confirm")).write_text("\n".join(map(json.dumps, rows)) + "\n",
                                                                    encoding="utf-8")
                self.assertEqual(AN.main(["--split", "confirm", "--data-dir", str(d), "--primary", "H0"]), 2)


if __name__ == "__main__":
    unittest.main()
