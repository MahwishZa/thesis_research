"""Which questions of MedRevQA no split uses yet, and which of them name dementia or Alzheimer's disease (synthetic rows; no network)."""

import unittest

from experiments.medchange.benchmark import load_groups
from experiments.medchange.selection import (
    AD_DOMAIN_TEXT, AD_QUESTION, NEURO_TEXT, candidates, fresh_items, used_review_ids,
)


def row(i, label, date, q):
    return {"": str(i), "Label": label, "Question": q, "PMID": f"/{1000 + i}/", "objectives": "", "conclusions": "",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + i}. doi: x"}


MEDREV = {0: row(0, "SUPPORTED", "2012 Mar 5", "Does A help B?"), 1: row(1, "NOT ENOUGH INFORMATION", "2008 Jan 2", "Does A help B?"),
          2: row(2, "REFUTED", "2019 May 1", "Does C help D?"), 3: row(3, "REFUTED", "2003", "Does E help F?"),
          4: row(4, "NOT ENOUGH INFORMATION", "2016 Jun 1", "Does G help dementia?"), 5: row(5, "SUPPORTED", "2021 Jun 1", "Does H help I?"),
          6: row(6, "SUPPORTED", "2024 Jun 1", "Does J help K?")}


GROUPS = load_groups([{"Group_ID": "1", "Study_ID": "0"}, {"Group_ID": "", "Study_ID": "1"},
                      {"Group_ID": "2", "Study_ID": "2"}])


class FreshItemsTests(unittest.TestCase):
    def items(self, used_groups=frozenset(), used_reviews=frozenset(), used_questions=frozenset()):
        return fresh_items(MEDREV, GROUPS, set(used_groups), frozenset(used_reviews), frozenset(used_questions))

    def test_grouped_and_ungrouped_questions_are_candidates_with_their_kind(self):
        got = {i["row"]: i for i in self.items()}
        self.assertEqual(set(got), {0, 2, 3, 4, 5, 6})
        self.assertEqual(got[0]["kind"], "changed")                       # newer SUPPORTED, older NEI
        self.assertFalse(got[0]["single_version"])
        self.assertTrue(got[3]["single_version"])

    def test_used_groups_reviews_and_wording_are_excluded(self):
        self.assertNotIn(0, {i["row"] for i in self.items(used_groups={1})})
        self.assertNotIn(5, {i["row"] for i in self.items(used_reviews={"CD100005"})})
        self.assertNotIn(2, {i["row"] for i in self.items(used_questions={"does c help d?"})})

    def test_candidates_are_every_group_or_ungrouped_row_outside_the_used_groups_in_row_order(self):
        got = candidates(MEDREV, GROUPS, set())
        self.assertEqual([c[:3] for c in got], [(0, 1, "changed"), (2, 2, "unchanged"), (3, 3, "unchanged"), (4, 4, "unchanged"),
                                                (5, 5, "unchanged"), (6, 6, "unchanged")])
        self.assertEqual([c[3] for c in got], [1, 2, None, None, None, None])           # the group of a grouped row, None for an ungrouped one
        self.assertNotIn(2, [c[0] for c in candidates(MEDREV, GROUPS, {2})])

    def test_a_group_of_one_row_is_unchanged_and_two_versions_with_one_label_are_unchanged(self):
        medrev = {0: row(0, "SUPPORTED", "2012 Mar 5", "Q0?"), 1: row(1, "SUPPORTED", "2008 Jan 2", "Q0?"), 2: row(2, "REFUTED", "2012 Mar 5", "Q2?")}
        groups = load_groups([{"Group_ID": "1", "Study_ID": "0"}, {"Group_ID": "", "Study_ID": "1"}, {"Group_ID": "2", "Study_ID": "2"}])
        self.assertEqual([c[:3] for c in candidates(medrev, groups, set())], [(0, 1, "unchanged"), (2, 2, "unchanged")])

    def test_a_citation_without_a_date_cannot_be_asked_as_of_a_date_and_is_left_out(self):
        medrev = dict(MEDREV)
        medrev[6] = dict(MEDREV[6], DOI_Date="Cochrane Database Syst Rev. no date given;1:CD100006. doi: x")
        self.assertNotIn(6, {i["row"] for i in fresh_items(medrev, GROUPS, set(), frozenset(), frozenset())})

    def test_the_same_wording_is_kept_once(self):
        medrev = dict(MEDREV)
        medrev[6] = dict(MEDREV[6], Question="Does C help D?")                              # the wording of row 2
        rows = {i["row"] for i in fresh_items(medrev, GROUPS, set(), frozenset(), frozenset())}
        self.assertIn(2, rows)
        self.assertNotIn(6, rows)                                                           # the later row repeats a question already kept


class UsedReviewTests(unittest.TestCase):
    def test_the_cochrane_ids_of_both_versions_of_every_main_item_are_used(self):
        main = [{"newest": {"cochrane_id": "CD1"}, "previous": {"cochrane_id": "CD2"}},
                {"newest": {"cochrane_id": "CD3"}, "previous": {"cochrane_id": None}},
                {"newest": {"cochrane_id": "CD3"}, "previous": {}}]
        self.assertEqual(used_review_ids(main), frozenset({"CD1", "CD2", "CD3"}))
        self.assertEqual(used_review_ids([]), frozenset())

    def test_the_wording_of_the_alzheimers_disease_set_and_of_the_review_text(self):
        for yes in ("Does X help in dementia?", "Is X effective for ALZHEIMER's disease?", "Does X slow cognitive decline?",
                    "X for mild cognitive impairment", "X in cognitively impaired adults", "X for vascular cognitive impairment"):
            self.assertTrue(AD_QUESTION.search(yes), yes)
        self.assertFalse(AD_QUESTION.search("Does X help in Parkinson's disease?"))
        self.assertTrue(AD_DOMAIN_TEXT.search("people with Alzheimer's disease"))
        self.assertTrue(NEURO_TEXT.search("Lewy bodies and Huntington's disease"))
        self.assertFalse(NEURO_TEXT.search("dementia"))                                     # dementia itself is not in the neurodegenerative count


if __name__ == "__main__":
    unittest.main()
