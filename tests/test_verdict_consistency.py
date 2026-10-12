"""Consistency of the stated verdict with the explanation (synthetic answers; a fake in place of the model)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import verdict_consistency as CA

S, R, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"


def answer(i, arm, verdict, text):
    return {"item_id": i, "arm": arm, "verdict": verdict, "text": text}


class VerdictConsistencyTests(unittest.TestCase):

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
