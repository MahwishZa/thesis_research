"""The Alzheimer's-named subgroup of the dementia run: its code, its committed file and the documents that quote it."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import subgroup_ad as G

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "experiments" / "medchange" / "results"


class SubgroupCodeTests(unittest.TestCase):
    def test_only_questions_naming_alzheimers_disease_are_used(self):
        self.assertTrue(G.NAMES_ALZHEIMER.search("Does X help in ALZHEIMER's disease?"))
        self.assertFalse(G.NAMES_ALZHEIMER.search("Does X help in dementia?"))

    def test_small_synthetic_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            items, answers = [], []
            for k in range(10):
                lab = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")[k % 3]
                items.append({"item_id": f"AD-{k}", "split": "ad", "likely_label_noise": False, "kind": "unchanged",
                              "question": "Does X help in Alzheimer's disease?" if k < 8 else "Does X help in dementia?",
                              "newest": {"label": lab, "date": "2023-01-01"}, "previous": {"label": lab}})
                for arm in ("B0", "B1", "R2", "R2C", "R2V"):
                    answers.append({"item_id": f"AD-{k}", "arm": arm, "verdict": lab if arm == "R2V" else "REFUTED",
                                    "text": "VERDICT", "admitted": []})
            (d / "benchmark.jsonl").write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
            (d / "answers_ad.jsonl").write_text("\n".join(json.dumps(a) for a in answers if a["arm"] in ("B0", "B1")), encoding="utf-8")
            (d / "rag2_answers_ad.jsonl").write_text("\n".join(json.dumps(a) for a in answers if a["arm"] not in ("B0", "B1")),
                                                     encoding="utf-8")
            rep = G.subgroup(d, d)
            self.assertEqual(rep["n"], 8)
            self.assertEqual(rep["accuracy"]["R2V"]["accuracy"], 1.0)
            self.assertIn("R2V vs R2", rep["comparisons"])


class CommittedFileTests(unittest.TestCase):
    rep = json.loads((RES / "ad_subgroup_alzheimer.json").read_text(encoding="utf-8"))

    def test_the_subgroup_has_the_48_questions_the_data_description_names(self):
        self.assertEqual((self.rep["n"], self.rep["dementia_set_items"]), (48, 208))
        self.assertIn("Alzheimer", (ROOT / "docs" / "data.md").read_text(encoding="utf-8"))
        self.assertIn("48 name Alzheimer's disease", (ROOT / "docs" / "data.md").read_text(encoding="utf-8"))

    def test_evaluation_md_quotes_the_committed_numbers(self):
        text = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
        sec = text[text.index("### 6.7"):text.index("### 6.8")]
        for arm, v in self.rep["accuracy"].items():
            self.assertIn(f"| {arm} | {100 * v['accuracy']:.1f}% ({100 * v['wilson95'][0]:.1f}–{100 * v['wilson95'][1]:.1f}) |", sec)
        r = self.rep["comparisons"]["R2V vs R2"]
        self.assertIn(f"{100 * r['diff_a_minus_b']:+.1f}".replace("-", "−") + " pp", sec)
        self.assertIn(f"{100 * self.rep['constant_answer']['accuracy']:.1f}%", sec)

    def test_the_per_verdict_view_quoted_in_evaluation_md_is_the_committed_one(self):
        text = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
        sec = text[text.index("### 6.7"):text.index("### 6.8")]
        m = self.rep["macro_f1"]
        self.assertIn(f"macro-F1 R2 {m['R2']:.3f}, R2V {m['R2V']:.3f}", sec)
        d = self.rep["macro_f1_R2V_minus_R2"]
        self.assertIn(f"{d['difference']:+.3f}, 95% interval {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}", sec)

    def test_no_system_beats_the_constant_answer_as_the_text_says(self):
        c = self.rep["constant_answer"]["accuracy"]
        self.assertTrue(all(v["accuracy"] < c for v in self.rep["accuracy"].values()))


if __name__ == "__main__":
    unittest.main()
