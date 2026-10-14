"""The two standard arms: which passages each admits, and the settings hash that the committed B0 and B1 answers carry (no model, no network)."""

import json
import unittest
from pathlib import Path

from experiments.medchange import arms as A
from experiments.medchange.files import load_jsonl

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "medchange" / "results"


def cand(pmid, rank):
    return {"pmid": pmid, "rank": rank, "lower": "2010-01-01", "upper": "2010-01-31", "title": f"T{pmid}", "abstract": f"abstract {pmid}",
            "pubtypes": [], "journal": "J"}


def pool(n=10):
    return [cand(f"{r:03d}", r) for r in range(1, n + 1)]


class ArmTests(unittest.TestCase):

    def test_b0_admits_nothing_b1_follows_rerank_rank(self):
        self.assertEqual(A.admit("B0", pool()), [])
        self.assertEqual([c["rank"] for c in A.admit("B1", pool())], [1, 2, 3, 4, 5])

    def test_b1_does_not_depend_on_the_order_of_the_pool_and_breaks_ties_by_pmid(self):
        shuffled = [cand("b", 2), cand("z", 1), cand("a", 2), cand("c", 3)]
        self.assertEqual([c["pmid"] for c in A.admit("B1", shuffled, budget=3)], ["z", "a", "b"])      # rank 1; then the tie of rank 2 by PMID

    def test_the_budget_is_five_and_a_small_or_empty_pool_is_admitted_whole(self):
        self.assertEqual(A.BUDGET, 5)
        self.assertEqual(len(A.admit("B1", pool(3))), 3)
        self.assertEqual(A.admit("B1", []), [])
        self.assertEqual(len(A.admit("B1", pool(20), budget=2)), 2)

    def test_the_arms_are_b0_and_b1_and_the_removed_recency_arms_are_refused(self):
        self.assertEqual(A.ARMS, ("B0", "B1"))
        for arm in ("B2", "B3", "P", "C1", "Z"):
            with self.assertRaises(ValueError):
                A.admit(arm, pool())


class RecordedSettingsTests(unittest.TestCase):
    """``SETTINGS`` still lists the parameters of the removed stage-1 arms because its hash is in every committed B0 and B1 record; this ties the code to them."""

    def test_the_settings_hash_is_stable_and_twelve_hex_digits(self):
        self.assertEqual(A.settings_hash(), A.settings_hash())
        self.assertRegex(A.settings_hash(), r"^[0-9a-f]{12}$")

    def test_the_hash_of_the_settings_is_the_one_in_every_committed_record_and_configuration(self):
        for split in ("dev", "confirm", "ad"):
            config = json.loads((RESULTS / f"answers_{split}.config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["arm_settings"], A.settings_hash(), split)
            self.assertEqual({r["settings"] for r in load_jsonl(RESULTS / f"answers_{split}.jsonl")}, {A.settings_hash()}, split)


if __name__ == "__main__":
    unittest.main()
