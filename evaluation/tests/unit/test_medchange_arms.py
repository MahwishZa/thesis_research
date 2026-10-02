"""Arms, prompts, resumable generation and the dev gates (no model, no network)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import arms as A
from experiments.medchange.analyze import correctness, gates, paired, summarize
from experiments.medchange.generate_answers import load_jsonl, run
from experiments.medchange.prompts import build_prompt, parse_verdict

CUTOFF = "2020-01-01"


def cand(pmid, rank, year, helpful=None, month="01"):
    c = {"pmid": pmid, "rank": rank, "lower": f"{year}-{month}-01", "upper": f"{year}-{month}-01",
         "title": f"T{pmid}", "abstract": f"abstract {pmid}", "pubtypes": [], "journal": "J"}
    if helpful is not None:
        c["helpful"] = helpful
    return c


def pool(n=10, with_helpful=True):
    # rank 1 is the OLDEST; helpfulness is highest for rank 3
    out = []
    for r in range(1, n + 1):
        out.append(cand(f"{r:03d}", r, 2019 - 2 * (n - r) if False else 2000 + r * 2,
                        helpful=(0.9 if r == 3 else 0.1 + 0.01 * (n - r)) if with_helpful else None))
    return out


class ArmTests(unittest.TestCase):

    def test_b0_admits_nothing_b1_follows_rerank_rank(self):
        self.assertEqual(A.admit("B0", pool(), CUTOFF), [])
        self.assertEqual([c["rank"] for c in A.admit("B1", pool(), CUTOFF)], [1, 2, 3, 4, 5])

    def test_b2_follows_helpfulness_not_rank(self):
        got = [c["rank"] for c in A.admit("B2", pool(), CUTOFF)]
        self.assertEqual(got[0], 3)
        self.assertEqual(sorted(got), sorted(got))      # budget respected
        self.assertEqual(len(got), 5)

    def test_recency_arms_prefer_newer_passages_than_their_relevance_only_twins(self):
        newest = lambda arm: max(c["upper"] for c in A.admit(arm, pool(), CUTOFF))
        mean_year = lambda arm: sum(int(c["upper"][:4]) for c in A.admit(arm, pool(), CUTOFF)) / 5
        self.assertGreater(mean_year("B3"), mean_year("B1"))
        self.assertGreater(mean_year("P"), mean_year("B2"))

    def test_shuffled_dates_keep_relevance_but_break_the_date_signal(self):
        p = pool()
        c1 = A.admit("C1", p, CUTOFF, item_id="x")
        pp = A.admit("P", p, CUTOFF, item_id="x")
        self.assertEqual(A.admit("C1", p, CUTOFF, item_id="x"), c1)          # deterministic
        self.assertNotEqual([c["pmid"] for c in c1], [c["pmid"] for c in pp])
        # the passages' own (true) dates are what is returned, only selection changed
        self.assertTrue(all(c in p for c in c1))

    def test_rank_normalisation(self):
        self.assertEqual(A.rank_normalise([3, 1, 2]), [1.0, 0.0, 0.5])
        self.assertEqual(A.rank_normalise([5]), [1.0])
        self.assertEqual(A.rank_normalise([2, 2, 1]), [1.0, 1.0, 0.0])

    def test_recency_halves_every_half_life_and_clamps_future(self):
        c = cand("1", 1, 2017, month="01")
        self.assertAlmostEqual(A.recency(c, "2020-01-01", half_life_days=1096), 0.5, places=2)
        self.assertEqual(A.recency(cand("2", 1, 2021), CUTOFF), 1.0)

    def test_helpfulness_arms_refuse_unscored_pools_and_unknown_arm(self):
        with self.assertRaises(ValueError):
            A.admit("P", pool(with_helpful=False), CUTOFF)
        with self.assertRaises(ValueError):
            A.admit("Z", pool(), CUTOFF)

    def test_empty_pool_and_small_pool(self):
        self.assertEqual(A.admit("B1", [], CUTOFF), [])
        self.assertEqual(len(A.admit("B3", pool(3), CUTOFF)), 3)

    def test_settings_hash_is_stable(self):
        self.assertEqual(A.settings_hash(), A.settings_hash())


class PromptTests(unittest.TestCase):

    def test_prompt_has_evidence_only_when_admitted_and_never_dates(self):
        p = build_prompt("Does X help Y?", [cand("1", 1, 2001)])
        self.assertIn("[1] T1. abstract 1", p)
        self.assertNotIn("2001", p)
        self.assertNotIn("Evidence:", build_prompt("Does X help Y?", []))

    def test_verdict_parsing(self):
        self.assertEqual(parse_verdict("VERDICT: SUPPORTED\nbecause"), "SUPPORTED")
        self.assertEqual(parse_verdict("verdict: **Not  enough information**"), "NOT ENOUGH INFORMATION")
        self.assertEqual(parse_verdict("VERDICT: REFUTED"), "REFUTED")
        self.assertIsNone(parse_verdict("The evidence supports it."))   # never guessed
        self.assertIsNone(parse_verdict(None))


def item(iid, kind, newest, previous="REFUTED"):
    return {"item_id": iid, "kind": kind, "newest": {"label": newest, "date": CUTOFF},
            "previous": {"label": previous}, "question": f"Q {iid}"}


class RunTests(unittest.TestCase):

    def setUp(self):
        self.items = [item("a", "changed", "SUPPORTED"), item("b", "changed", "REFUTED")]
        self.pools = {i["item_id"]: {"candidates": pool()} for i in self.items}
        self.helpful = {}

    def test_resume_skips_finished_pairs_and_tolerates_truncated_last_line(self):
        calls = []
        def gen(system, user):
            calls.append(user)
            return "VERDICT: SUPPORTED\nok"
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ans.jsonl"
            n1 = run(self.items, self.pools, self.helpful, ["B0", "B1"], out, gen)
            self.assertEqual((n1, len(calls)), (4, 4))
            with open(out, "a") as f:
                f.write('{"item_id": "zz", "arm": "B0", "text": "trunc')    # killed mid-write
            n2 = run(self.items, self.pools, self.helpful, ["B0", "B1"], out, gen)
            self.assertEqual(n2, 0)
            self.assertEqual(len(calls), 4)
            rows = load_jsonl(out)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["arm"], "B0")           # items interleaved across arms
        self.assertEqual(rows[1]["arm"], "B1")
        self.assertEqual(rows[0]["admitted"], [])
        self.assertEqual(len(rows[1]["admitted"]), 5)

    def test_prompt_hash_differs_between_arms_that_admit_different_evidence(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.jsonl"
            run(self.items[:1], self.pools, self.helpful, ["B0", "B1"], out, lambda s, u: "x")
            r = load_jsonl(out)
        self.assertNotEqual(r[0]["prompt_sha256"], r[1]["prompt_sha256"])


def answers(spec):
    """spec: {(item, arm): verdict}"""
    return {k: {"verdict": v, "seconds": 1.0} for k, v in spec.items()}


class GateTests(unittest.TestCase):

    def setUp(self):
        self.items = {f"c{i}": item(f"c{i}", "changed", "SUPPORTED") for i in range(20)}

    def test_correctness_and_summary(self):
        spec = {(i, "B1"): ("SUPPORTED" if n < 12 else "REFUTED") for n, i in enumerate(self.items)}
        s = summarize(self.items, answers(spec), ["B1"])
        self.assertEqual(s["B1"]["changed"]["accuracy"], 0.6)
        self.assertEqual(s["B1"]["changed"]["outdated_match"], 0.4)   # previous label REFUTED

    def test_unparsed_counts_wrong(self):
        spec = {("c0", "B1"): None}
        self.assertEqual(correctness(self.items, answers(spec), "B1", "changed"), {"c0": 0})

    def test_g2_threshold(self):
        spec = {}
        for n, i in enumerate(self.items):
            spec[(i, "B0")] = "NOT ENOUGH INFORMATION"
            spec[(i, "B1")] = "SUPPORTED" if n < 4 else "NOT ENOUGH INFORMATION"   # 20% change
        self.assertEqual(gates(self.items, answers(spec))["G2"], "PASS")
        spec2 = {k: v for k, v in spec.items()}
        for n, i in enumerate(self.items):
            spec2[(i, "B1")] = "SUPPORTED" if n < 3 else "NOT ENOUGH INFORMATION"   # 15%
        self.assertEqual(gates(self.items, answers(spec2))["G2"], "FAIL")

    def test_g3_requires_gain_over_b2_and_over_the_shuffled_control(self):
        spec = {}
        for n, i in enumerate(self.items):
            spec[(i, "B2")] = "REFUTED"                                   # 0% correct
            spec[(i, "P")] = "SUPPORTED" if n < 4 else "REFUTED"           # 20% correct
            spec[(i, "C1")] = "SUPPORTED" if n < 4 else "REFUTED"          # reproduces it
        g = gates(self.items, answers(spec))
        self.assertEqual(g["G3"], "FAIL")
        for n, i in enumerate(self.items):
            spec[(i, "C1")] = "REFUTED"
        self.assertEqual(gates(self.items, answers(spec))["G3"], "PASS")

    def test_paired_needs_minimum_pairs(self):
        spec = {("c0", "P"): "SUPPORTED", ("c0", "B2"): "REFUTED"}
        self.assertIsNone(paired(self.items, answers(spec), "P", "B2"))


from experiments.medchange.consistency import sample_rows, score_sheet  # noqa: E402


class ConsistencyTests(unittest.TestCase):

    def test_sample_is_seeded_and_capped(self):
        rows = [{"item_id": f"i{k}", "arm": "B1"} for k in range(100)]
        a, b = sample_rows(rows, 50, 7), sample_rows(rows, 50, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)
        self.assertNotEqual(a, sample_rows(rows, 50, 8))

    def test_score_requires_every_row_marked_and_applies_thresholds(self):
        sheet = [{"consistent": "Y"}] * 9 + [{"consistent": "N"}]
        self.assertEqual(score_sheet(sheet, 0.97)["G1"], "PASS")
        self.assertEqual(score_sheet(sheet, 0.90)["G1"], "FAIL")           # parse rate too low
        self.assertEqual(score_sheet([{"consistent": "Y"}] * 8 + [{"consistent": "N"}] * 2, 1.0)["G1"], "FAIL")
        with self.assertRaises(ValueError):
            score_sheet([{"consistent": "Y"}, {"consistent": ""}], 1.0)


if __name__ == "__main__":
    unittest.main()
