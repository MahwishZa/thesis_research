"""The fresh split: the registered selection and the freeze of the pre-registration (synthetic MedRevQA rows)."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import fresh_benchmark as FB
from experiments.medchange import rag2_pipeline as P
from experiments.medchange.benchmark import load_groups


def row(i, label, date, q):
    return {"": str(i), "Label": label, "Question": q, "PMID": f"/{1000 + i}/", "objectives": "", "conclusions": "",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + i}. doi: x"}


def medrev(n=60):
    labs = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
    rows = {i: row(i, labs[i % 3], f"{2004 + i % 18} Mar 5", f"Does intervention {i} help condition {i}?") for i in range(n)}
    rows[7] = row(7, "SUPPORTED", "2018 Mar 5", "Does X help in dementia?")               # dementia wording: never selected
    return rows


class SelectionTests(unittest.TestCase):
    def build(self, n):
        mr = medrev()
        return FB.build(mr, load_groups([]), [], n)

    def test_the_selection_is_deterministic_dated_from_2010_and_without_dementia_wording(self):
        a, meta = self.build(10)
        b, _ = self.build(10)
        self.assertEqual([i["item_id"] for i in a], [i["item_id"] for i in b])
        self.assertTrue(all(i["newest"]["date"] >= FB.EARLIEST for i in a))
        self.assertNotIn("FR-00007", [i["item_id"] for i in a])
        self.assertGreater(meta["qualifying"], 10)

    def test_the_order_is_the_sha256_of_seed_and_row_and_ignores_labels(self):
        items, _ = self.build(8)
        rows = [i["newest"]["row"] for i in items]
        self.assertEqual(rows, sorted(rows, key=FB.order_key))
        self.assertEqual(FB.order_key(3), FB.order_key(3))

    def test_items_are_fresh_unchanged_and_not_alzheimers(self):
        it = self.build(3)[0][0]
        self.assertEqual((it["split"], it["kind"], it["ad_related"], it["likely_label_noise"]), ("fresh", "unchanged", False, False))
        self.assertTrue(it["item_id"].startswith("FR-"))
        self.assertEqual(it["previous"]["row"], it["newest"]["row"])

    def test_main_writes_a_manifest_with_the_rows_and_refuses_a_different_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "Datasets").mkdir()
            mr = medrev()
            with open(d / "Datasets" / "MedRevQA.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["", "Label", "Question", "PMID", "DOI_Date", "objectives", "conclusions"])
                w.writeheader()
                w.writerows(mr.values())
            (d / "Datasets" / "AllStudyGroups.csv").write_text("Group_ID,Study_ID\n", encoding="utf-8")
            data = d / "data"
            data.mkdir()
            (data / "benchmark.jsonl").write_text(json.dumps({"item_id": "X", "group_id": None, "split": "dev", "question": "Other?",
                                                              "newest": {"cochrane_id": "CD000001"}, "previous": {"cochrane_id": "CD000001"}}) + "\n",
                                                  encoding="utf-8")
            man = d / "manifest_fresh.json"
            args = ["--medchange-dir", str(d), "--data-dir", str(data), "--manifest", str(man), "--n", "10"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(FB.main(args), 0)
            rep = json.loads(man.read_text(encoding="utf-8"))
            self.assertEqual((rep["items"], len(rep["rows"])), (10, 10))
            self.assertNotIn("Question", json.dumps(rep))
            lines = [json.loads(l) for l in (data / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(l["split"] == "fresh" for l in lines), 10)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(FB.main(args), 0)                                    # same selection: allowed
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(FB.main(args[:-1] + ["8"]), 2)                       # a different selection: refused
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(FB.main(args[:-1] + ["999"]), 3)                     # fewer qualifying questions than N


class FreezeTests(unittest.TestCase):
    def test_freeze_replaces_the_draft_status_and_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "protocol.md"
            p.write_text("# T\n\n## 10. Pre-registration of a confirmatory test on fresh questions (DRAFT: not in force)\n\n"
                         "**Status.** This section is a draft written on 2026-10-09. It binds nothing\nuntil the freeze.\n\n### 10.1 Why\n", encoding="utf-8")
            man = {"items": 850, "item_ids_sha256": "ab" * 32}
            self.assertTrue(FB.freeze(p, man, "2026-11-02"))
            text = p.read_text(encoding="utf-8")
            self.assertIsNotNone(P.IN_FORCE.search(text))
            self.assertIn("### 10.1 Why", text)
            self.assertIn("## 10. Pre-registration of a confirmatory test on fresh questions (in force since 2026-11-02)", text)
            self.assertNotIn("DRAFT", text)
            self.assertFalse(FB.freeze(p, man, "2026-11-03"), "a frozen section is not rewritten")

    def test_the_committed_protocol_has_a_draft_status_that_freeze_can_replace_or_is_frozen(self):
        text = (Path(P.HERE).parents[1] / "docs" / "protocol.md").read_text(encoding="utf-8")
        self.assertTrue(FB.DRAFT_STATUS.search(text) or P.IN_FORCE.search(text))


if __name__ == "__main__":
    unittest.main()
