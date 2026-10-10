"""The fresh split: selection, the blinded analysis, the freeze guards and the pipeline plan (no model, no network)."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

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
            p.write_text("# T\n\n**Status.** This section is a draft written on 2026-10-09. It binds nothing\nuntil the freeze.\n\n### 10.1 Why\n", encoding="utf-8")
            man = {"items": 1000, "item_ids_sha256": "ab" * 32}
            self.assertTrue(FB.freeze(p, man, "2026-11-02"))
            text = p.read_text(encoding="utf-8")
            self.assertIsNotNone(P.IN_FORCE.search(text))
            self.assertIn("### 10.1 Why", text)
            self.assertFalse(FB.freeze(p, man, "2026-11-03"), "a frozen section is not rewritten")

    def test_the_committed_draft_has_a_status_paragraph_that_freeze_can_replace(self):
        text = (Path(P.HERE).parents[1] / "docs" / "protocol.md").read_text(encoding="utf-8")
        self.assertIsNotNone(FB.DRAFT_STATUS.search(text))


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
        self.assertEqual((AF.EXPECTED, AF.ITERATIONS, AF.PRIMARY_SEED, AF.ARMS), (1000, 10000, "fresh-macro-f1", ("R2", "R2C", "R2V")))

    def test_the_dementia_comparison_is_descriptive_and_checks_the_sign(self):
        cb = {"splits": {"ad": {"n": 208, "paired": {"R2V vs R2": {"macro_f1": {"difference": 0.08, "ci95": [0.02, 0.13]}}}}}}
        rep = AF.analyse(*synthetic(60), expected=60, iterations=100, class_balance=cb)
        d = rep["alzheimers_dementia_descriptive"]
        self.assertEqual(d["n"], 208)
        self.assertIn("not confirmation", d["note"])
        self.assertIsInstance(d["sign_agrees_with_fresh"], bool)


class FreezeAndPlanTests(unittest.TestCase):
    def test_the_status_line_that_puts_the_preregistration_in_force(self):
        self.assertTrue(P.IN_FORCE.search("**Status.** This section is IN FORCE since 2026-11-01 (commit abc)."))
        self.assertIsNone(P.IN_FORCE.search("**Status.** This section is a draft written on 2026-10-09."))
        text = (Path(P.HERE).parents[1] / "docs" / "protocol.md").read_text(encoding="utf-8")
        self.assertIsNone(P.IN_FORCE.search(text), "the committed protocol must still read as a draft until the researcher freezes it")

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
        self.assertEqual(P.EXPECTED_ITEMS["fresh"], 1000)

    def test_preflight_of_the_fresh_split_does_not_ask_for_b0_and_b1(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "benchmark.jsonl").write_text("", encoding="utf-8")
            names = [n for n, _, _ in P.preflight(d, "fresh")]
            self.assertFalse(any("B0 and B1" in n for n in names))
            self.assertTrue(any("B0 and B1" in n for n in [n for n, _, _ in P.preflight(d, "ad")]))

    def test_the_audit_sample_is_the_same_items_every_time(self):
        items = [{"item_id": f"FR-{i:05d}"} for i in range(100)]
        a, b = LA.audit_sample(items, 20), LA.audit_sample(list(reversed(items)), 20)
        self.assertEqual([x["item_id"] for x in a], [x["item_id"] for x in b])
        self.assertEqual(len(a), 20)


if __name__ == "__main__":
    unittest.main()
