"""Per-verdict behaviour and macro-F1 (no model): the analysis on synthetic splits, the committed file and the documents that quote it."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import class_balance as C
from tests.fixtures import F_, N, S, answer, item, write

ROOT = Path(__file__).resolve().parents[1]
REP = json.loads((ROOT / "experiments" / "medchange" / "results" / "class_balance.json").read_text(encoding="utf-8"))


class SyntheticRunTests(unittest.TestCase):
    """The analysis and its tables on small synthetic splits (no committed file is read)."""

    def _write(self, d):
        rows, base, r2 = [], [], []
        for split, prefix in (("confirm", "MC"), ("ad", "AD")):
            for k in range(30):
                gold = (S, F_, N)[k % 3]
                iid = f"{prefix}-{k:03d}"
                rows.append(item(iid, kind="unchanged", gold=gold, split=split))
                base += [answer(iid, "B0", N), answer(iid, "B1", gold if k % 2 else N)]
                r2 += [answer(iid, "R2", gold if k % 3 else N), answer(iid, "R2C", gold if k % 3 else N),
                       answer(iid, "R2V", gold if k % 4 else None)]           # R2V leaves a quarter of the answers without a verdict
            write(d / f"answers_{split}.jsonl", [r for r in base if r["item_id"].startswith(prefix)])
            write(d / f"rag2_answers_{split}.jsonl", [r for r in r2 if r["item_id"].startswith(prefix)])
        write(d / "benchmark.jsonl", rows)

    def test_the_loader_reads_the_benchmark_and_the_verdicts_of_the_known_arms(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d)
            items, answers = C.load(d, d, "ad")
            self.assertEqual(len(items), 30)
            self.assertEqual(sorted({a for _, a in answers}), ["B0", "B1", "R2", "R2C", "R2V"])
            self.assertIsNone(answers[("AD-000", "R2V")])                                 # an unparsed answer stays None

    def test_the_recorded_number_of_resamples_is_10000(self):
        self.assertEqual(C.ITERATIONS, 10000)

    def test_the_analysis_has_the_shares_the_recalls_and_the_three_paired_differences_of_each_split(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d)
            with mock.patch.object(C, "ITERATIONS", 100):                      # the recorded value is 10,000; the tables do not depend on it
                rep = C.analyse(d, d)
        self.assertEqual((rep["iterations"], rep["seed"]), (100, "class-balance-v1"))
        for split in ("confirm", "ad"):
            r = rep["splits"][split]
            self.assertEqual(r["n"], 30)
            self.assertAlmostEqual(sum(r["gold_share"].values()), 1.0, places=5)
            self.assertEqual(set(r["arms"]), {"B0", "B1", "R2", "R2C", "R2V"})
            self.assertEqual(set(r["paired"]), {"R2V vs R2", "R2C vs R2", "R2V vs R2C"})
            self.assertAlmostEqual(sum(r["arms"]["R2V"]["predicted_share"].values()), 1.0 - 8 / 30, places=5)       # the unparsed answers are no verdict
            self.assertEqual(r["paired"]["R2C vs R2"]["macro_f1"]["difference"], 0.0)            # the same answers
            self.assertGreater(r["paired"]["R2V vs R2"]["macro_f1"]["difference"], 0)             # right on 22 of 30 against 20 of 30

    def test_the_command_writes_the_json_and_the_markdown_and_prints_the_tables(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(C, "ITERATIONS", 100):
                self.assertEqual(C.main(["--data-dir", tmp, "--results-dir", tmp]), 0)
            written = (d / "class_balance.md").read_text(encoding="utf-8")
            self.assertEqual(written + "\n", out.getvalue())
            self.assertEqual(json.loads((d / "class_balance.json").read_text(encoding="utf-8"))["splits"]["ad"]["n"], 30)
        for needle in ("# Per-verdict behaviour (exploratory)", "## confirm split (n = 30)", "## ad split (n = 30)", "| R2V vs R2 |", "Gold shares:"):
            self.assertIn(needle, written)


class CommittedFileTests(unittest.TestCase):
    def test_sizes_and_the_pairs_are_those_of_the_two_splits(self):
        self.assertEqual((REP["splits"]["confirm"]["n"], REP["splits"]["ad"]["n"]), (528, 208))
        for sp in REP["splits"].values():
            self.assertEqual(set(sp["paired"]), {"R2V vs R2", "R2C vs R2", "R2V vs R2C"})

    def test_the_gains_quoted_in_the_protocol_are_those_of_the_file(self):
        text = (ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")
        for sp in REP["splits"].values():
            d = sp["paired"]["R2V vs R2"]["macro_f1"]
            self.assertIn(f"+{d['difference']:.3f}", text)

    def test_every_verdict_share_sums_to_one(self):
        for sp in REP["splits"].values():
            for v in sp["arms"].values():
                self.assertAlmostEqual(sum(v["predicted_share"].values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
