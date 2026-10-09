"""Reading of model replies and the re-scoring of the verifier qualification. No model."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.adkqa import qualify as Q
from experiments.adkqa import spec


class ReadingTests(unittest.TestCase):
    def test_replies_seen_in_the_first_qualification_run_are_read(self):
        cases = {"SUPPORTED: NIPPV likely reduces the": "SUPPORTED", "NOT ENOUGH INFORMATION\n\nLABEL": "NOT ENOUGH INFORMATION",
                 "REFUTED; the evidence suggests treatments may not be": "REFUTED", "REFUTED\n\nThe author's conclusions indicate": "REFUTED",
                 "LABEL: SUPPORTED": "SUPPORTED", "**LABEL:** not enough information": "NOT ENOUGH INFORMATION",
                 "\n  Refuted. The review found": "REFUTED"}
        for reply, want in cases.items():
            self.assertEqual(spec.parse_verdict(reply), want, reply)

    def test_nothing_is_guessed_from_the_middle_of_a_reply(self):
        for reply in ("", None, "Based on the conclusions the answer is SUPPORTED", "Unsupported by the evidence",
                      "Not supported", "SUPPORTEDLY so", "The label is REFUTED"):
            self.assertIsNone(spec.parse_verdict(reply), reply)


class RescoreTests(unittest.TestCase):
    def rows(self, n_ok, n_wrong, n_unparsed):
        rows = [{"which": "newest", "gold": "SUPPORTED", "raw": "SUPPORTED: x"}] * n_ok
        rows += [{"which": "newest", "gold": "SUPPORTED", "raw": "REFUTED"}] * n_wrong
        rows += [{"which": "newest", "gold": "SUPPORTED", "raw": "I think"}] * n_unparsed
        rows += [{"which": "previous", "gold": "SUPPORTED", "raw": "I think"}] * 5      # older versions are not scored
        return rows

    def test_thresholds_are_the_declared_ones(self):
        self.assertEqual((Q.MIN_AGREEMENT, Q.MIN_KAPPA, Q.MAX_UNPARSED), (0.75, 0.60, 0.02))

    def test_each_criterion_can_fail(self):
        good = Q.rescore(self.rows(60, 5, 0) + [{"which": "newest", "gold": "REFUTED", "raw": "REFUTED"}] * 35)
        self.assertTrue(good["qualified"], good)
        many_unparsed = Q.rescore(self.rows(60, 5, 20) + [{"which": "newest", "gold": "REFUTED", "raw": "REFUTED"}] * 35)
        self.assertFalse(many_unparsed["checks"]["unparsed_at_most_2pct"])
        self.assertFalse(many_unparsed["qualified"])
        low = Q.rescore(self.rows(10, 60, 0) + [{"which": "newest", "gold": "REFUTED", "raw": "REFUTED"}] * 10)
        self.assertFalse(low["checks"]["agreement_at_least_75pct"])
        self.assertEqual(good["n_items"], 100)

    def test_main_writes_counts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit, out = Path(tmp) / "a.jsonl", Path(tmp) / "q.json"
            audit.write_text("\n".join(json.dumps(r) for r in self.rows(10, 0, 0)), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                Q.main(["--audit", str(audit), "--out", str(out), "--model-name", "m.gguf"])
            rep = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(rep["verifier"], "m.gguf")
            self.assertNotIn("raw", json.dumps(rep))


if __name__ == "__main__":
    unittest.main()
