"""Automatic label audit and consistency check (synthetic data; a fake in place of the model)."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import consistency_auto as CA
from experiments.medchange import label_audit as LA

S, R, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"


def item(i, kind, new, prev, row_new, row_prev):
    return {"item_id": i, "kind": kind, "split": "dev", "likely_label_noise": False, "question": f"Q {i}?",
            "newest": {"label": new, "row": row_new, "date": "2020-01-01"},
            "previous": {"label": prev, "row": row_prev, "date": "2010-01-01"}}


MEDREV = {1: {"objectives": "obj one", "conclusions": "conc one"}, 2: {"objectives": "o2", "conclusions": "c2"},
          3: {"objectives": "o3", "conclusions": "c3"}, 4: {"objectives": "o4", "conclusions": "c4"}}


class LabelAuditTests(unittest.TestCase):

    def test_prompt_carries_the_authors_rubric_and_the_text(self):
        p = LA.build_prompt("Does X help?", "obj", "conc")
        for needle in ("only select the third label if not enough studies were found", "at least partially supported",
                       "Does X help?", "obj", "conc", "LABEL:"):
            self.assertIn(needle, p)

    def test_parse_label(self):
        self.assertEqual(LA.parse_label("LABEL: not  enough information"), N)
        self.assertEqual(LA.parse_label("label: **Refuted**"), R)
        self.assertIsNone(LA.parse_label("I think it works"))

    def test_kappa(self):
        self.assertEqual(LA.cohens_kappa([(S, S), (R, R), (N, N)]), 1.0)
        self.assertEqual(LA.cohens_kappa([(S, R), (R, S)]), -1.0)
        self.assertIsNone(LA.cohens_kappa([]))

    def test_run_is_resumable_and_summary_finds_stable_items_and_reproduced_changes(self):
        items = [item("a", "changed", S, N, 1, 2), item("b", "unchanged", R, R, 3, 4)]
        answers = {"obj one": "LABEL: SUPPORTED", "o2": "LABEL: NOT ENOUGH INFORMATION", "o3": "LABEL: SUPPORTED"}
        calls = []

        def fake(system, user):
            calls.append(user)
            for k, v in answers.items():
                if f"OBJECTIVES: {k}" in user:
                    return v
            return "no idea"

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit.jsonl"
            self.assertEqual(LA.run(items, MEDREV, fake, out), 3)          # a newest, a previous, b newest
            self.assertEqual(LA.run(items, MEDREV, fake, out), 0)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        rep = LA.summarize(rows, {i["item_id"]: i for i in items})
        self.assertEqual(rep["n_items"], 2)
        self.assertAlmostEqual(rep["agreement"], 0.5)                       # a agrees, b (REFUTED vs SUPPORTED) not
        self.assertEqual(rep["stable_item_ids"], ["a"])
        self.assertEqual(rep["label_change_reproduced"], 1.0)
        self.assertEqual(rep["per_gold_class"][R]["agreement"], 0.0)
        self.assertEqual(len(calls), 3)

    def test_unparsed_relabels_are_counted_not_guessed(self):
        items = [item("a", "unchanged", S, S, 1, 2)]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.jsonl"
            LA.run(items, MEDREV, lambda s, u: "hmm", out)
            rep = LA.summarize([json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()], {"a": items[0]})
        self.assertEqual((rep["n_unparsed"], rep["agreement"], rep["stable_item_ids"]), (1, None, []))

    def test_cli_refuses_the_confirmatory_split_before_the_model_is_frozen(self):
        with TemporaryDirectory() as tmp, mock.patch("sys.stderr"):
            code = LA.main(["--split", "confirm", "--medchange-dir", tmp, "--model-path", str(Path(tmp) / "m.gguf"),
                            "--frozen-model", str(Path(tmp) / "absent.json")])
        self.assertEqual(code, 2)

    def test_cli_writes_reports_with_a_fake_model(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, res, ds = tmp / "data", tmp / "results", tmp / "mc" / "Datasets"
            data.mkdir(); ds.mkdir(parents=True)
            (data / "benchmark.jsonl").write_text(json.dumps(item("a", "unchanged", S, S, 1, 2)) + "\n", encoding="utf-8")
            with open(ds / "MedRevQA.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["", "objectives", "conclusions"])
                w.writeheader()
                for k, v in MEDREV.items():
                    w.writerow({"": k, **v})
            model = tmp / "m.gguf"
            model.write_bytes(b"x")
            with mock.patch.object(LA, "llama_generator", lambda *a, **k: (lambda s, u: "LABEL: SUPPORTED")), \
                    mock.patch("sys.stdout"):
                code = LA.main(["--split", "dev", "--medchange-dir", str(tmp / "mc"), "--model-path", str(model),
                                "--data-dir", str(data), "--out-dir", str(res)])
            rep = json.loads((res / "label_audit_dev.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(rep["stable_item_ids"], ["a"])


def answer(i, arm, verdict, text):
    return {"item_id": i, "arm": arm, "verdict": verdict, "text": text}


class ConsistencyAutoTests(unittest.TestCase):

    def test_the_explanation_drops_only_the_verdict_line(self):
        self.assertEqual(CA.explanation("VERDICT: SUPPORTED\n\nIt works [1]."), "It works [1].")
        self.assertEqual(CA.explanation("no verdict here"), "no verdict here")

    def test_sample_is_arm_balanced_seeded_and_skips_unparsed(self):
        answers = [answer(f"i{k}", arm, "SUPPORTED", "x") for k in range(10) for arm in ("B0", "B1")]
        answers.append(answer("bad", "B0", None, "x"))
        s1, s2 = CA.sample_answers(answers, 6, 3), CA.sample_answers(answers, 6, 3)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 6)
        self.assertEqual(sum(a["arm"] == "B0" for a in s1), 3)
        self.assertNotIn("bad", {a["item_id"] for a in s1})

    def test_run_and_summary(self):
        answers = [answer("a", "B0", S, "VERDICT: SUPPORTED\nIt helps."), answer("b", "B0", R, "VERDICT: REFUTED\nIt helps."),
                   answer("c", "B1", None, "no verdict")]
        seen = []

        def fake(system, user):
            seen.append(user)
            return "LABEL: SUPPORTED"

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "c.jsonl"
            CA.run(answers[:2], {"a": "Qa?", "b": "Qb?"}, fake, out)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all("VERDICT" not in u.split("EXPLANATION:")[1] for u in seen))   # the stated verdict is hidden
        rep = CA.summarize(rows, answers)
        self.assertAlmostEqual(rep["parse_rate"], 2 / 3, places=3)
        self.assertEqual(rep["parse_gate"], "FAIL")
        self.assertEqual(rep["agreement"], 0.5)
        self.assertEqual(rep["disagreements"][f"{R}->{S}"], 1)


if __name__ == "__main__":
    unittest.main()
