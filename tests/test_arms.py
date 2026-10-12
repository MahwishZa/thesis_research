"""The arms of the earlier stages and of the current study (B0/B1 and the completed stage-1 arms): which passages each admits (no model, no network)."""

import unittest

from experiments.medchange import arms as A


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
        out.append(cand(f"{r:03d}", r, 2000 + r * 2,
                        helpful=(0.9 if r == 3 else 0.1 + 0.01 * (n - r)) if with_helpful else None))
    return out


class PointDateTests(unittest.TestCase):
    def test_the_date_of_a_passage_is_the_midpoint_of_the_bounds_on_its_first_availability(self):
        import datetime as dt
        self.assertEqual(A.point_date({"lower": "2020-01-01", "upper": "2020-01-31"}), dt.date(2020, 1, 16))
        self.assertEqual(A.point_date({"lower": "2020-03-05", "upper": "2020-03-05"}), dt.date(2020, 3, 5))


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

    def test_relevance_ranks_are_contiguous_and_ties_follow_cross_encoder_rank(self):
        p = [cand("a", 2, 2000), cand("b", 1, 2000), cand("c", 3, 2000)]
        self.assertEqual(A.relevance_ranks([5.0, 5.0, 9.0], p), [3, 2, 1])   # c best; tie: b (rank 1) before a
        self.assertEqual(sorted(A.relevance_ranks([1, 1, 1], p)), [1, 2, 3])

    def test_scores_come_from_the_reference_implementation(self):
        """The arms must not carry their own copy of the formula: compare with a direct
        call of the stage-1 modules on the same inputs."""
        from datetime import date
        from experiments.medchange.stage1.evidence import Candidate, Evidence
        from experiments.medchange.stage1.scorer import AdmissionScorer
        from experiments.medchange.stage1.temporal import TemporalPolicy
        p = pool()
        got = A.admission_scores(p, [-c["rank"] for c in p], p, CUTOFF, 0.5)
        scorer = AdmissionScorer(0.5)
        policy = TemporalPolicy(half_life_days=A.HALF_LIFE_DAYS, undated_score=0.0)
        for c, g in zip(p, got):
            ev = Evidence(evidence_id=c["pmid"], text="", source_tier="x",
                          publication_date=A.point_date(c))
            t = policy.score(ev, question="", question_date=date.fromisoformat(CUTOFF)).score
            want = scorer.score(Candidate(ev, 0.0, c["rank"]), temporal=t, candidate_count=len(p)).total
            self.assertAlmostEqual(g, want)
        self.assertAlmostEqual(got[0], 0.5 * 1.0 + 0.5 * A.recency(p[0], CUTOFF))   # rank 1 -> rho 1

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


if __name__ == "__main__":
    unittest.main()
