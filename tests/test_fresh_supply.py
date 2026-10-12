"""Counting the unused questions of MedRevQA (no network, no model)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import fresh_supply as F
from experiments.medchange.benchmark import load_groups


def row(i, label, date, q):
    return {"": str(i), "Label": label, "Question": q, "PMID": f"/{1000 + i}/", "objectives": "", "conclusions": "",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + i}. doi: x"}


MEDREV = {0: row(0, "SUPPORTED", "2012 Mar 5", "Does A help B?"), 1: row(1, "NOT ENOUGH INFORMATION", "2008 Jan 2", "Does A help B?"),
          2: row(2, "REFUTED", "2019 May 1", "Does C help D?"), 3: row(3, "REFUTED", "2003", "Does E help F?"),
          4: row(4, "NOT ENOUGH INFORMATION", "2016 Jun 1", "Does G help dementia?"), 5: row(5, "SUPPORTED", "2021 Jun 1", "Does H help I?"),
          6: row(6, "SUPPORTED", "2024 Jun 1", "Does J help K?")}
GROUPS = load_groups([{"Group_ID": "1", "Study_ID": "0"}, {"Group_ID": "", "Study_ID": "1"},
                      {"Group_ID": "2", "Study_ID": "2"}])


class SupplyTests(unittest.TestCase):
    def items(self, used_groups=frozenset(), used_reviews=frozenset(), used_questions=frozenset()):
        return F.fresh_items(MEDREV, GROUPS, set(used_groups), frozenset(used_reviews), frozenset(used_questions))

    def test_grouped_and_ungrouped_questions_are_candidates_with_their_kind(self):
        got = {i["row"]: i for i in self.items()}
        self.assertEqual(set(got), {0, 2, 3, 4, 5, 6})
        self.assertEqual(got[0]["kind"], "changed")                       # newer SUPPORTED, older NEI
        self.assertFalse(got[0]["single_version"])
        self.assertTrue(got[3]["single_version"])

    def test_a_review_about_dementia_is_counted_even_when_the_question_does_not_say_so(self):
        medrev = dict(MEDREV)
        medrev[5] = dict(MEDREV[5], objectives="To assess H in people with Alzheimer's disease.")
        items = F.fresh_items(medrev, GROUPS, set(), frozenset(), frozenset())
        by = {i["row"]: i for i in items}
        self.assertTrue(by[5]["dementia_in_review_text"])
        self.assertFalse(by[5]["dementia_wording"])
        self.assertTrue(by[4]["dementia_in_review_text"])                 # the question names dementia
        rep = F.report(items)["by_earliest_date"]["2010-01-01"]["dementia_in_review_text"]
        self.assertEqual((rep["questions"], rep["labels"]["SUPPORTED"], rep["labels"]["NOT ENOUGH INFORMATION"]), (2, 1, 1))

    def test_neurodegenerative_wording_is_counted_separately(self):
        medrev = dict(MEDREV)
        medrev[5] = dict(MEDREV[5], Question="Does H help Parkinson's disease?")
        medrev[6] = dict(MEDREV[6], objectives="People with Huntington's disease.")
        rep = F.report(F.fresh_items(medrev, GROUPS, set(), frozenset(), frozenset()))["by_earliest_date"]["2010-01-01"]
        self.assertEqual(rep["neurodegenerative_wording"]["questions"], 1)
        self.assertEqual(rep["neurodegenerative_in_review_text"]["questions"], 2)

    def test_used_groups_reviews_and_wording_are_excluded(self):
        self.assertNotIn(0, {i["row"] for i in self.items(used_groups={1})})
        self.assertNotIn(5, {i["row"] for i in self.items(used_reviews={"CD100005"})})
        self.assertNotIn(2, {i["row"] for i in self.items(used_questions={"does c help d?"})})

    def test_the_report_counts_by_earliest_date_without_selecting_anything(self):
        rep = F.report(self.items())
        self.assertEqual(rep["by_earliest_date"]["2010-01-01"]["questions"], 5)    # rows 0, 2, 4, 5, 6
        self.assertEqual(rep["by_earliest_date"]["2005-01-01"]["questions"], 5)
        self.assertEqual(rep["by_earliest_date"]["2023-04-01"]["questions"], 1)
        self.assertEqual(rep["by_earliest_date"]["2010-01-01"]["dementia_wording"], 1)
        self.assertEqual(rep["by_earliest_date"]["2010-01-01"]["labels"]["SUPPORTED"], 3)
        self.assertEqual(rep["by_year"]["2003"], 1)

    def test_main_writes_counts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "Datasets").mkdir()
            import csv
            with open(d / "Datasets" / "MedRevQA.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["", "Label", "Question", "PMID", "DOI_Date", "objectives", "conclusions"])
                w.writeheader()
                w.writerows(MEDREV.values())
            with open(d / "Datasets" / "AllStudyGroups.csv", "w", encoding="utf-8", newline="") as f:
                f.write("Group_ID,Study_ID\n1,0\n,1\n2,2\n")
            data = d / "data"
            data.mkdir()
            (data / "benchmark.jsonl").write_text(json.dumps({"item_id": "X", "group_id": 2, "split": "dev", "question": "Does C help D?",
                                                              "newest": {"cochrane_id": "CD100002"}, "previous": {"cochrane_id": "CD100002"}}) + "\n",
                                                  encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(F.main(["--medchange-dir", str(d), "--data-dir", str(data), "--out-dir", str(d / "out")]), 0)
            rep = json.loads((d / "out" / "fresh_supply.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["fresh_questions_all_dates"], 5)           # row 2's group and wording are used
            self.assertNotIn("Question", json.dumps(rep))


if __name__ == "__main__":
    unittest.main()
