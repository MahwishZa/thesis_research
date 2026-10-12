"""The fresh split: selection, the blinded analysis, the freeze guards and the pipeline plan (no model, no network)."""

import contextlib
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.medchange import analyze_fresh as AF
from experiments.medchange import fresh_benchmark as FB
from experiments.medchange import label_audit as LA
from experiments.medchange import rag2_pipeline as P
from experiments.medchange.benchmark import load_groups


def row(i, label, date, q):
    return {"": str(i), "Label": label, "Question": q, "PMID": f"/{1000 + i}/", "objectives": "", "conclusions": "",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + i}. doi: x"}


def medrev(n=60):
    labs = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
    rows = {i: row(i, labs[i % 3], f"{2004 + i % 18} Mar 5", f"Does intervention {i} help condition {i}?") for i in range(n)}
    rows[7] = row(7, "SUPPORTED", "2018 Mar 5", "Does X help in dementia?")               # dementia wording: never selected
    return rows


class SelectionTests(unittest.TestCase):
    def build(self, n):
        mr = medrev()
        return FB.build(mr, load_groups([]), [], n)

    def test_the_selection_is_deterministic_dated_from_2010_and_without_dementia_wording(self):
        a, meta = self.build(10)
        b, _ = self.build(10)
        self.assertEqual([i["item_id"] for i in a], [i["item_id"] for i in b])
        self.assertTrue(all(i["newest"]["date"] >= FB.EARLIEST for i in a))
        self.assertNotIn("FR-00007", [i["item_id"] for i in a])
        self.assertGreater(meta["qualifying"], 10)

    def test_the_order_is_the_sha256_of_seed_and_row_and_ignores_labels(self):
        items, _ = self.build(8)
        rows = [i["newest"]["row"] for i in items]
        self.assertEqual(rows, sorted(rows, key=FB.order_key))
        self.assertEqual(FB.order_key(3), FB.order_key(3))

    def test_items_are_fresh_unchanged_and_not_alzheimers(self):
        it = self.build(3)[0][0]
        self.assertEqual((it["split"], it["kind"], it["ad_related"], it["likely_label_noise"]), ("fresh", "unchanged", False, False))
        self.assertTrue(it["item_id"].startswith("FR-"))
        self.assertEqual(it["previous"]["row"], it["newest"]["row"])

    def test_main_writes_a_manifest_with_the_rows_and_refuses_a_different_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "Datasets").mkdir()
            mr = medrev()
            with open(d / "Datasets" / "MedRevQA.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["", "Label", "Question", "PMID", "DOI_Date", "objectives", "conclusions"])
                w.writeheader()
                w.writerows(mr.values())
            (d / "Datasets" / "AllStudyGroups.csv").write_text("Group_ID,Study_ID\n", encoding="utf-8")
            data = d / "data"
            data.mkdir()
            (data / "benchmark.jsonl").write_text(json.dumps({"item_id": "X", "group_id": None, "split": "dev", "question": "Other?",
                                                              "newest": {"cochrane_id": "CD000001"}, "previous": {"cochrane_id": "CD000001"}}) + "\n",
                                                  encoding="utf-8")
            man = d / "manifest_fresh.json"
            args = ["--medchange-dir", str(d), "--data-dir", str(data), "--manifest", str(man), "--n", "10"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(FB.main(args), 0)
            rep = json.loads(man.read_text(encoding="utf-8"))
            self.assertEqual((rep["items"], len(rep["rows"])), (10, 10))
            self.assertNotIn("Question", json.dumps(rep))
            lines = [json.loads(l) for l in (data / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(l["split"] == "fresh" for l in lines), 10)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(FB.main(args), 0)                                    # same selection: allowed
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(FB.main(args[:-1] + ["8"]), 2)                       # a different selection: refused
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(FB.main(args[:-1] + ["999"]), 3)                     # fewer qualifying questions than N


class FreezeTests(unittest.TestCase):
    def test_freeze_replaces_the_draft_status_and_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "protocol.md"
            p.write_text("# T\n\n## 10. Pre-registration of a confirmatory test on fresh questions (DRAFT: not in force)\n\n"
                         "**Status.** This section is a draft written on 2026-10-09. It binds nothing\nuntil the freeze.\n\n### 10.1 Why\n", encoding="utf-8")
            man = {"items": 850, "item_ids_sha256": "ab" * 32}
            self.assertTrue(FB.freeze(p, man, "2026-11-02"))
            text = p.read_text(encoding="utf-8")
            self.assertIsNotNone(P.IN_FORCE.search(text))
            self.assertIn("### 10.1 Why", text)
            self.assertIn("## 10. Pre-registration of a confirmatory test on fresh questions (in force since 2026-11-02)", text)
            self.assertNotIn("DRAFT", text)
            self.assertFalse(FB.freeze(p, man, "2026-11-03"), "a frozen section is not rewritten")

    def test_the_committed_protocol_has_a_draft_status_that_freeze_can_replace_or_is_frozen(self):
        text = (Path(P.HERE).parents[1] / "docs" / "protocol.md").read_text(encoding="utf-8")
        self.assertTrue(FB.DRAFT_STATUS.search(text) or P.IN_FORCE.search(text))


def synthetic(n=60, gain=True):
    labs = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"]
    rng = np.random.default_rng(5)
    items, answers = {}, {}
    for k in range(n):
        gold = labs[k % 3]
        items[f"FR-{k:05d}"] = {"item_id": f"FR-{k:05d}", "kind": "unchanged", "newest": {"label": gold}}
        r2 = gold if rng.random() < 0.4 else "SUPPORTED"
        r2v = gold if (rng.random() < 0.6 if gain else r2 == gold) else "SUPPORTED"
        for arm, v in (("R2", r2), ("R2C", r2), ("R2V", r2v)):
            answers[(f"FR-{k:05d}", arm)] = {"item_id": f"FR-{k:05d}", "arm": arm, "verdict": v, "text": "VERDICT", "admitted": [],
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
            self.assertNotIn("descriptive)", printed)

    def test_stored_differences_keep_six_decimals_so_the_rule_follows_the_exact_interval(self):
        self.assertEqual(AF.DIGITS, 6)
        gold = np.array([0, 1, 2])
        r = AF.paired_interval(gold, np.array([0, 1, 0]), np.array([0, 0, 0]), AF.macro_f1, "t", iterations=20)
        self.assertEqual(r["difference"], round(r["difference"], 6))
        self.assertNotEqual(r["difference"], round(r["difference"], 4))               # 1/3-type values would be cut by four decimals
        self.assertEqual(AF.read_primary({"difference": 0.00003, "ci95": [0.00002, 0.01]})["reading"], "confirmed")

    def test_a_lower_end_exactly_at_zero_is_not_above_zero(self):
        self.assertEqual(AF.read_primary({"difference": 0.04, "ci95": [0.0, 0.08]})["reading"], "positive but not confirmed")
        self.assertEqual(AF.read_primary({"difference": 0.0, "ci95": [-0.03, 0.03]})["reading"], "no evidence")

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

    def test_the_four_places_that_hold_the_number_of_questions_agree_with_the_manifest(self):
        manifest = json.loads((Path(AF.__file__).resolve().parent / "manifest_fresh.json").read_text(encoding="utf-8"))
        self.assertEqual({AF.EXPECTED, FB.N_DEFAULT, P.EXPECTED_ITEMS["fresh"]}, {manifest["items"]})
        self.assertEqual(len(manifest["rows"]), manifest["items"])
        self.assertEqual(sum(manifest["labels"].values()), manifest["items"])


class FreezeAndPlanTests(unittest.TestCase):
    def test_the_status_line_that_puts_the_preregistration_in_force(self):
        self.assertTrue(P.IN_FORCE.search("**Status.** This section is IN FORCE since 2026-11-01 (commit abc)."))
        self.assertIsNone(P.IN_FORCE.search("**Status.** This section is a draft written on 2026-10-09."))
        text = (Path(P.HERE).parents[1] / "docs" / "protocol.md").read_text(encoding="utf-8")
        draft = "**Status.** This section is a draft" in text
        self.assertTrue(draft or P.IN_FORCE.search(text), "§10 must read either as a draft or as in force, never neither")
        self.assertFalse(draft and P.IN_FORCE.search(text))

    def test_without_the_freeze_the_fresh_run_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "protocol.md").write_text("**Status.** draft", encoding="utf-8")
            problems = P.freeze_problems(root / "data", root)
            self.assertTrue(any("IN FORCE" in p for p in problems))
            self.assertTrue(any("manifest_fresh.json" in p for p in problems))

    def test_the_fresh_plan_runs_r2_r2c_r2v_and_nothing_else(self):
        a = P.argparse.Namespace(model_path="m.gguf", judge_path="j.gguf", n_threads="6", commit=False, ablations=False, no_temporal_ablation=False)
        steps = P.fresh_plan(a, Path("data"), Path("results"))
        names = " | ".join(n for n, _ in steps)
        argv = " ".join(" ".join(map(str, act)) for _, act in steps if isinstance(act, list))
        self.assertIn("--arms R2 R2C R2V", argv)
        self.assertNotIn("R2V-ND", argv)
        self.assertNotIn("generate_answers", argv)                                   # no B0 / B1
        self.assertNotIn("judge", names.lower().replace("no directness judge", ""))
        self.assertIn("--abstracts-only", argv)
        self.assertIn("analyze_fresh", argv)
        self.assertEqual(P.EXPECTED_ITEMS["fresh"], 850)

    def test_preflight_of_the_fresh_split_does_not_ask_for_b0_and_b1(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "benchmark.jsonl").write_text("", encoding="utf-8")
            names = [n for n, _, _ in P.preflight(d, "fresh")]
            self.assertFalse(any("B0 and B1" in n for n in names))
            self.assertTrue(any("B0 and B1" in n for n in [n for n, _, _ in P.preflight(d, "ad")]))

    def test_the_items_in_the_benchmark_file_must_match_the_manifest_when_the_list_is_pushed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "protocol.md").write_text("**Status.** This section is IN FORCE since 2026-10-10 (list `abc`).", encoding="utf-8")
            ids = ["FR-00001", "FR-00002"]
            manifest = root / "experiments" / "medchange" / "manifest_fresh.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"items": 2, "item_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}), encoding="utf-8")
            data = root / "data"
            data.mkdir()

            def write(listed):
                (data / "benchmark.jsonl").write_text("\n".join(json.dumps({"item_id": i, "split": "fresh"}) for i in listed) + "\n", encoding="utf-8")
            with mock.patch.object(P, "frozen_is_pushed", return_value=(True, "file is on origin/main")):
                write(ids)
                self.assertEqual(P.freeze_problems(data, root), [])
                write(ids[:1])
                self.assertEqual(len(P.freeze_problems(data, root)), 1)
                write(ids[::-1])
                self.assertTrue(any("differ" in p for p in P.freeze_problems(data, root)))               # the order is part of the list
            with mock.patch.object(P, "frozen_is_pushed", return_value=(False, "differs from origin/main")):
                write(ids)
                self.assertEqual(P.freeze_problems(data, root), ["differs from origin/main"])

    def test_the_registered_audit_sample_of_the_committed_question_list_is_these_300_questions(self):
        """The rule of protocol §10.8 (SHA-256 order of ``fresh-audit-v1|<item id>``) applied to the committed manifest; a change of the rule or of the
        list changes the sample and fails here."""
        manifest = json.loads((Path(AF.__file__).resolve().parent / "manifest_fresh.json").read_text(encoding="utf-8"))
        items = [{"item_id": f"FR-{row:05d}"} for row in manifest["rows"]]
        sample = [x["item_id"] for x in LA.audit_sample(items, 300)]
        self.assertEqual(sample[:5], ["FR-00725", "FR-03135", "FR-07991", "FR-03991", "FR-02782"])
        self.assertEqual(hashlib.sha256(json.dumps(sorted(sample)).encode()).hexdigest()[:16], "0dd10528022c2eae")
        self.assertEqual(len(set(sample)), 300)

    def test_the_audit_sample_is_the_same_items_every_time(self):
        items = [{"item_id": f"FR-{i:05d}"} for i in range(100)]
        a, b = LA.audit_sample(items, 20), LA.audit_sample(list(reversed(items)), 20)
        self.assertEqual([x["item_id"] for x in a], [x["item_id"] for x in b])
        self.assertEqual(len(a), 20)


if __name__ == "__main__":
    unittest.main()
