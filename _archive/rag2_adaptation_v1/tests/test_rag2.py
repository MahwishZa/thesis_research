"""The adapted RAG² (R2) and the evidence-criteria verification (R2V): retrieval, filter and criteria prompts (synthetic data and fakes only)."""

import unittest

from experiments.medchange import rag2 as R
from tests.fixtures import F_, N, S, TYPES, HashingEncoder, LexicalOverlapReranker, abstracts, cand, item, lists_for, records, scores_for


class RetrievalTests(unittest.TestCase):
    def test_balanced_takes_the_quota_of_each_type_in_order(self):
        strata = ["other"] * 5 + ["RCT"] * 3 + ["SR/MA"]
        order = list(range(9))
        self.assertEqual(R.balanced(order, strata, quota=2), [0, 1, 5, 6, 8])

    def _build(self, recs, rationale="Population adults; intervention drug; outcome pain.", date="2015-06-01"):
        return R.build_lists(item("MC-1", date=date), recs, abstracts(recs), rationale,
                             query_encoder=HashingEncoder(), article_encoder=HashingEncoder(),
                             reranker=LexicalOverlapReranker())

    def test_lists_are_capped_deterministic_and_typed(self):
        recs = records()
        a, b = self._build(recs), self._build(recs)
        self.assertEqual(a["lists_hash"], b["lists_hash"])
        self.assertEqual(set(a["lists"]), set(R.LISTS))
        for variant, cands in a["lists"].items():
            self.assertEqual(len(cands), R.RERANK_KEEP)
            self.assertEqual([c["rank"] for c in cands], list(range(1, R.RERANK_KEEP + 1)))
            scores = [c["rerank_score"] for c in cands]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(c["stratum"] in R.STRATA for c in cands))
        self.assertTrue(a["rationale_used"])

    def test_records_are_ranked_from_the_highest_score_and_ties_go_to_the_smaller_pmid(self):
        recs = [{"pmid": "b"}, {"pmid": "a"}, {"pmid": "c"}, {"pmid": "d"}]
        self.assertEqual(R._ranked([0.1, 0.9, 0.5, 0.9], recs), [1, 3, 2, 0])         # 0.9 (a), 0.9 (d), 0.5 (c), 0.1 (b)

    def test_a_given_rationale_is_the_dense_query_of_r2_and_the_question_is_that_of_the_ablation(self):
        rec = self._build(records(n_per_type=3))
        for cands in rec["lists"].values():
            self.assertTrue(all(c["dense_rationale"] != c["dense_question"] for c in cands))
        empty = self._build(records(n_per_type=3), rationale="")
        for cands in empty["lists"].values():
            self.assertTrue(all(c["dense_rationale"] == c["dense_question"] for c in cands))

    def test_the_unbalanced_ablation_re_ranks_as_many_records_as_r2_does(self):
        recs = [r for r in records(n_per_type=12) if r["pmid"].startswith("other") or r["pmid"] in ("RCT-0", "SR/MA-0")]      # 12 + 1 + 1 records
        rec = R.build_lists(item("MC-1"), recs, abstracts(recs), "Population adults; intervention drug; outcome pain.",
                            query_encoder=HashingEncoder(), article_encoder=HashingEncoder(), reranker=LexicalOverlapReranker(), keep=20)
        self.assertEqual(len(rec["lists"]["R2"]), R.QUOTA + 1 + 1)               # 8 of the common type and the two single records
        self.assertEqual(len(rec["lists"]["R2-BR"]), len(rec["lists"]["R2"]))     # the same number, taken without balancing

    def test_records_after_the_cutoff_or_ineligible_are_never_listed(self):
        recs = records(n_per_type=2)
        recs.append({"pmid": "late", "lower": "2016-01-01", "upper": "2016-12-31", "pubtypes": TYPES["RCT"], "journal": "J"})
        recs.append({"pmid": "erratum", "lower": "2010-01-01", "upper": "2010-12-31",
                     "pubtypes": ["Published Erratum"], "journal": "J"})
        rec = self._build(recs)
        listed = {c["pmid"] for cands in rec["lists"].values() for c in cands}
        self.assertNotIn("late", listed)
        self.assertNotIn("erratum", listed)
        self.assertEqual(rec["n_eligible"], 6)

    def test_an_empty_rationale_falls_back_to_the_question(self):
        rec = self._build(records(n_per_type=3), rationale="  ")
        self.assertFalse(rec["rationale_used"])
        self.assertEqual(rec["lists"]["R2"], rec["lists"]["R2-RQ"])

    def test_no_eligible_record_gives_empty_lists(self):
        rec = self._build(records(n_per_type=2, upper="2020-01-01", lower="2019-01-01"))
        self.assertEqual(rec["n_eligible"], 0)
        self.assertTrue(all(v == [] for v in rec["lists"].values()))

    def test_strip_text_removes_publisher_text_only(self):
        rec = self._build(records(n_per_type=2))
        stripped = R.strip_text(rec)
        for cands in stripped["lists"].values():
            for c in cands:
                self.assertNotIn("title", c)
                self.assertNotIn("abstract", c)
                self.assertIn("pmid", c)
        self.assertIn("title", rec["lists"]["R2"][0])            # the original is untouched


class FilterTests(unittest.TestCase):
    def test_yes_probability_from_first_token(self):
        p, mass = R.yes_probability([{"token": "Yes", "logprob": -0.1}, {"token": " no", "logprob": -2.5},
                                     {"token": "Maybe", "logprob": -3.0}])
        self.assertAlmostEqual(mass, 0.9048 + 0.0821, places=3)
        self.assertGreater(p, 0.9)
        self.assertEqual(R.yes_probability([{"token": "A", "logprob": -0.1}]), (None, 0.0))

    def test_the_probability_of_yes_is_renormalised_and_takes_the_most_likely_spelling_of_each_word(self):
        import math
        p, mass = R.yes_probability([{"token": "Yes", "logprob": math.log(0.5)}, {"token": " yes", "logprob": math.log(0.2)},
                                     {"token": "No", "logprob": math.log(0.1)}, {"token": "no", "logprob": math.log(0.05)}])
        self.assertAlmostEqual(mass, 0.6)                           # max over spellings: 0.5 + 0.1 (the spellings are not summed)
        self.assertAlmostEqual(p, 0.5 / 0.6)

    def test_admission_filters_keeps_order_and_budget(self):
        lists = lists_for()
        scores = scores_for(lists)
        scores["R2-0"]["p_yes"] = 0.2                      # filtered out
        scores["R2-1"]["valid"] = False                    # invalid: kept
        scores["R2-1"]["p_yes"] = 0.0
        admitted = R.admit("R2", lists, scores)
        self.assertEqual([c["pmid"] for c in admitted], ["R2-1", "R2-2", "R2-3", "R2-4", "R2-5"])
        self.assertEqual([c["pmid"] for c in R.admit("R2-NF", lists, {})], [f"R2-{i}" for i in range(5)])
        self.assertEqual(R.admit("R2V", lists, scores), admitted)          # the criteria arms read R2's evidence

    def test_a_paper_passes_the_filter_at_exactly_the_threshold(self):
        lists = lists_for()
        scores = scores_for(lists, p=R.FILTER_THRESHOLD)                    # "at least 0.5" passes
        self.assertEqual(len(R.admit("R2", lists, scores)), R.BUDGET)
        below = scores_for(lists, p=R.FILTER_THRESHOLD - 1e-9)
        self.assertEqual(R.admit("R2", lists, below), [])

    def test_everything_filtered_admits_nothing_and_a_missing_judgement_is_an_error(self):
        lists = lists_for()
        self.assertEqual(R.admit("R2", lists, scores_for(lists, p=0.1)), [])
        with self.assertRaises(KeyError):
            R.admit("R2", lists, {})


class CriteriaTests(unittest.TestCase):
    def test_dated_and_undated_prompts(self):
        passages = [cand("1", 1, "RCT", upper="2009-05-31"), cand("2", 2, "SR/MA", upper="2011-03-31")]
        _, dated = R.build_criteria_prompt("Q?", passages, draft="VERDICT: SUPPORTED", dated=True)
        _, undated = R.build_criteria_prompt("Q?", passages, draft="VERDICT: SUPPORTED", dated=False)
        self.assertIn("(randomized or controlled trial, 2009)", dated)
        self.assertIn("(systematic review or meta-analysis, 2011)", dated)
        self.assertIn("newer evidence take precedence", dated)
        self.assertNotIn("2009", undated)
        self.assertNotIn("newer evidence", undated)
        self.assertIn("direct randomized evidence.", undated)

    def test_criteria_control_and_verification_share_everything_before_the_task(self):
        passages = [cand("1", 1)]
        _, crit = R.build_criteria_prompt("Q?", passages)
        _, verify = R.build_criteria_prompt("Q?", passages, draft="VERDICT: SUPPORTED\nBecause [1].")
        shared = crit.split("Answer the question by applying")[0]
        self.assertTrue(verify.startswith(shared))
        self.assertIn("Draft answer:\nVERDICT: SUPPORTED", verify)
        self.assertNotIn("Draft answer", crit)

    def test_parse_final_verdict(self):
        self.assertEqual(R.parse_final_verdict("DIRECT STUDIES: 1\nFINDINGS: x.\nFINAL VERDICT: REFUTED"), F_)
        self.assertEqual(R.parse_final_verdict("**FINAL VERDICT:** not enough  information"), N)
        self.assertEqual(R.parse_final_verdict("FINAL VERDICT: SUPPORTED\nFINAL VERDICT: REFUTED"), F_)
        self.assertIsNone(R.parse_final_verdict("VERDICT: SUPPORTED"))
        self.assertIsNone(R.parse_final_verdict(R.FORMAT))                  # the format line copied verbatim

    def test_cited_and_direct_numbers(self):
        self.assertEqual(R.cited_numbers("See [1] and [2, 4] and [3-5]."), [1, 2, 3, 4, 5])
        self.assertEqual(R.direct_numbers("DIRECT STUDIES: 1, 3\nFINAL VERDICT: SUPPORTED"), [1, 3])
        self.assertEqual(R.direct_numbers("DIRECT STUDIES: NONE"), [])
        self.assertIsNone(R.direct_numbers("VERDICT: SUPPORTED"))

    def test_unsupported_decisive(self):
        self.assertIsNone(R.unsupported_decisive(S, "VERDICT: SUPPORTED", 0))
        self.assertFalse(R.unsupported_decisive(N, "VERDICT: NOT ENOUGH INFORMATION", 5))
        self.assertFalse(R.unsupported_decisive(S, "VERDICT: SUPPORTED because [2] shows it", 5))
        self.assertTrue(R.unsupported_decisive(S, "VERDICT: SUPPORTED, as is well known", 5))
        self.assertTrue(R.unsupported_decisive(F_, "VERDICT: REFUTED per [7]", 5))
        self.assertFalse(R.unsupported_decisive(F_, "DIRECT STUDIES: 2\nFINAL VERDICT: REFUTED", 5))
        self.assertTrue(R.unsupported_decisive(F_, "DIRECT STUDIES: NONE\nFINAL VERDICT: REFUTED", 5))

    def test_anachronistic_years(self):
        text = "A 2019 meta-analysis and a 2010 trial; see also 2015."
        self.assertEqual(R.anachronistic_years(text, "2015-06-01"), [2019])
        self.assertEqual(R.anachronistic_years("no years here", "2015-06-01"), [])

    def test_design_record_is_stable_and_complete(self):
        a, b = R.design_record("abc"), R.design_record("abc")
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"settings", "settings_sha256", "prompts", "generator_model_sha256", "encoders"})
        self.assertEqual(len(a["settings_sha256"]), 12)
        self.assertIn("verify_task", a["prompts"])


if __name__ == "__main__":
    unittest.main()
