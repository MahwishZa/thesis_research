"""The Alzheimer's disease set: selection from MedRevQA outside the other splits, and the command that appends it to the benchmark file."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import ad_benchmark as AD
from experiments.medchange.selection import used_review_ids
from tests.fixtures import F_, N, S, item, write


def medrev_row(k, question, label=S, date="2015 Mar 2"):
    return {"": str(k), "Question": question, "objectives": "", "Label": label, "PMID": f"https://pubmed/{9000 + k}",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + k:06d}. doi: x"}


class AlzheimersBenchmarkTests(unittest.TestCase):
    def _world(self):
        rows = {1: medrev_row(1, "Does donepezil help people with Alzheimer's disease?"),
                2: medrev_row(2, "Does donepezil help people with Alzheimer's disease?", F_, "2010 Jan 1"),
                3: medrev_row(3, "Does exercise slow cognitive decline in older adults?", N),
                4: medrev_row(4, "Does exercise slow cognitive decline in older adults?", N),   # duplicate question
                5: medrev_row(5, "Do acetylcholinesterase inhibitors improve autism?"),           # off topic
                6: medrev_row(6, "Is music therapy effective for dementia?", F_),
                7: medrev_row(7, "Is music therapy effective for dementia?", F_, "2009 May 5")}
        groups = {10: [1, 2], 11: [6, 7], 12: [8]}
        rows[8] = medrev_row(8, "Does aspirin prevent dementia?")
        return rows, groups

    def test_selection(self):
        rows, groups = self._world()
        items = AD.build_ad_items(rows, groups, used_groups={12})
        by_id = {i["item_id"]: i for i in items}
        self.assertEqual(set(by_id), {"AD-00001", "AD-00003", "AD-00006"})
        self.assertEqual(by_id["AD-00001"]["kind"], "changed")           # unused multi-version review, label changed
        self.assertEqual(by_id["AD-00006"]["kind"], "unchanged")
        single = by_id["AD-00003"]
        self.assertEqual(single["previous"], single["newest"])
        self.assertTrue(single["notes"])
        self.assertTrue(all(i["split"] == "ad" and i["ad_related"] for i in items))

    def test_an_older_version_of_a_review_already_in_dev_or_confirm_is_excluded(self):
        """MedRevQA can hold an old version of a review as an ungrouped row of its own, so excluding by study
        group alone lets a review through that dev or confirm already holds (four such questions were in the
        first build of the set). The Cochrane ID closes that gap; a missing ID cannot exclude anything."""
        rows, groups = self._world()
        rows[9] = medrev_row(9, "Does donepezil improve well-being in mild Alzheimer's disease?", S, "2001 Jan 1")
        rows[9]["DOI_Date"] = rows[9]["DOI_Date"].replace("CD100009", "CD100008")   # same review as row 8
        rows[10] = medrev_row(10, "Does memantine help in dementia?", S, "2002 Jan 1")
        rows[10]["DOI_Date"] = "Cochrane Database Syst Rev. 2002 Jan 1;1:no-id. doi: x"      # a date, no CD id
        before = {i["item_id"] for i in AD.build_ad_items(rows, groups, used_groups={12})}
        self.assertIn("AD-00009", before)                                   # the group check alone keeps it
        main = [dict(item("MC-1"), group_id=12,
                     newest={"cochrane_id": "CD100008", "row": 8}, previous={"cochrane_id": "CD100008", "row": 8})]
        used = used_review_ids(main)
        self.assertEqual(used, frozenset({"CD100008"}))
        after = {i["item_id"] for i in AD.build_ad_items(rows, groups, used_groups={12}, used_reviews=used)}
        self.assertNotIn("AD-00009", after)
        self.assertIn("AD-00010", after)                                    # no ID: nothing to match, kept
        self.assertEqual(after, before - {"AD-00009"})

    def test_cli_appends_once_and_keeps_the_main_benchmark(self):
        rows, groups = self._world()
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "Datasets").mkdir()
            import csv
            with open(d / "Datasets" / "MedRevQA.csv", "w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["", "Question", "objectives", "Label", "PMID", "DOI_Date"])
                w.writeheader()
                for k in sorted(rows):
                    w.writerow(rows[k])
            with open(d / "Datasets" / "AllStudyGroups.csv", "w", newline="", encoding="utf-8") as h:
                w = csv.writer(h)
                w.writerow(["Group_ID", "Study_ID"])
                for gid, keys in groups.items():
                    for j, k in enumerate(keys):
                        w.writerow([gid if j == 0 else "", k])
            write(d / "benchmark.jsonl", [dict(item("MC-1"), group_id=12)])
            args = ["--medchange-dir", tmp, "--data-dir", tmp, "--manifest", str(d / "m.json")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(AD.main(args), 0)
                self.assertEqual(AD.main(args), 0)                        # a rerun replaces, never duplicates
            bench = [json.loads(l) for l in (d / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["split"] for r in bench], ["dev", "ad", "ad", "ad"])
            self.assertEqual(json.loads((d / "m.json").read_text(encoding="utf-8"))["items"], 3)


if __name__ == "__main__":
    unittest.main()
