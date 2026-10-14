"""Consistency of the stated verdict with the explanation (synthetic answers; a fake in place of the model)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

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



class CommandTests(unittest.TestCase):
    """The command with a fake judge: the files it writes, the resumption and the refusals."""

    def _world(self, tmp, answers=True):
        d = Path(tmp)
        (d / "data").mkdir()
        (d / "data" / "benchmark.jsonl").write_text("\n".join(json.dumps({"item_id": f"i{k}", "question": f"Does X{k} help?"}) for k in range(6)) + "\n",
                                                    encoding="utf-8")
        if answers:
            rows = [answer(f"i{k}", arm, (S, R, N)[k % 3], f"VERDICT: {(S, R, N)[k % 3]}\nBecause of [1].") for k in range(6) for arm in ("B0", "B1")]
            (d / "data" / "answers_dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        model = d / "judge.gguf"
        model.write_bytes(b"not a model")
        return d, model

    def _main(self, d, model, *extra, reply="LABEL: SUPPORTED"):
        loaded = []

        def fake_generator(*args):
            loaded.append(args)
            return lambda system, user: reply
        argv = ["--split", "dev", "--model-path", str(model), "--data-dir", str(d / "data"), "--out-dir", str(d / "out"), "--n", "6", *extra]
        with mock.patch.object(CA, "llama_generator", fake_generator), mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return CA.main(argv), loaded

    def test_the_command_writes_the_summary_and_the_table_and_a_rerun_asks_the_judge_nothing_new(self):
        with TemporaryDirectory() as tmp:
            d, model = self._world(tmp)
            code, loaded = self._main(d, model)
            self.assertEqual((code, len(loaded)), (0, 1))
            rep = json.loads((d / "out" / "verdict_consistency_dev.json").read_text(encoding="utf-8"))
            self.assertEqual((rep["split"], rep["judge"], rep["parse_gate"], rep["n_judged"]), ("dev", "judge.gguf", "PASS", 6))
            self.assertEqual(rep["agreement"], round(2 / 6, 4))                                  # the fake always says SUPPORTED: right for the two SUPPORTED stated verdicts
            self.assertIn("# Automatic consistency check, dev split", (d / "out" / "verdict_consistency_dev.md").read_text(encoding="utf-8"))
            before = (d / "data" / "verdict_consistency_dev.jsonl").read_bytes()
            code, _ = self._main(d, model, reply="LABEL: REFUTED")                     # a different reply would show if anything were asked again
            self.assertEqual(code, 0)
            self.assertEqual((d / "data" / "verdict_consistency_dev.jsonl").read_bytes(), before)

    def test_a_changed_setting_a_missing_model_or_missing_answers_are_refused(self):
        with TemporaryDirectory() as tmp:
            d, model = self._world(tmp)
            self.assertEqual(self._main(d, model)[0], 0)
            code, loaded = self._main(d, model, "--seed", "99")
            self.assertEqual((code, loaded), (2, []))                                  # the sample would differ: refused before the model loads
            self.assertEqual(self._main(d, d / "absent.gguf")[0], 2)
        with TemporaryDirectory() as tmp:
            d, model = self._world(tmp, answers=False)
            self.assertEqual(self._main(d, model), (2, []))


if __name__ == "__main__":
    unittest.main()
