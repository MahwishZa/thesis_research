"""Checks of src/balance_test.py: the balanced test split, on made-up cases and on the committed table."""

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import balance_test as bt  # noqa: E402
from common import ALL_CASES, read_csv  # noqa: E402

TABLE = ROOT / "data" / ALL_CASES


def cases(spec):
    """spec: [(label, split, group size), ...] -> one case per image, group ids g0, g1, ..."""
    out = []
    for k, (label, split, size) in enumerate(spec):
        out += [{"isic_id": f"ISIC_{k:03d}_{j}", "group_id": f"g{k}", "label": label, "split": split} for j in range(size)]
    return out


def counts(cs, new):
    return Counter((c["label"], new[c["isic_id"]]) for c in cs)


class RuleTests(unittest.TestCase):
    def test_every_class_gets_exactly_the_target_and_validation_is_untouched(self):
        cs = cases([("MEL", "test", 1)] * 6 + [("MEL", "train", 1)] * 4 + [("MEL", "val", 1)] * 2 +
                   [("DF", "test", 1)] * 1 + [("DF", "train", 2)] * 3 + [("DF", "train", 1)] * 2 + [("DF", "val", 1)] * 2)
        new = bt.balanced_splits(cs, per_class=3, seed=42)
        c = counts(cs, new)
        self.assertEqual((c[("MEL", "test")], c[("DF", "test")]), (3, 3))
        self.assertEqual((c[("MEL", "val")], c[("DF", "val")]), (2, 2))
        self.assertTrue(all(new[x["isic_id"]] == "val" for x in cs if x["split"] == "val"))
        self.assertEqual((c[("MEL", "train")], c[("DF", "train")]), (7, 6))       # nothing lost: 12 and 11 images in all

    def test_whole_lesion_groups_move_and_a_group_that_would_go_over_is_skipped(self):
        cs = cases([("AK", "test", 1), ("AK", "train", 3), ("AK", "train", 2), ("AK", "train", 1)])
        new = bt.balanced_splits(cs, per_class=3, seed=42)
        by_group = {}
        for x in cs:
            by_group.setdefault(x["group_id"], set()).add(new[x["isic_id"]])
        self.assertTrue(all(len(s) == 1 for s in by_group.values()))
        self.assertEqual(counts(cs, new)[("AK", "test")], 3)
        self.assertEqual(by_group["g1"], {"train"})                                  # 1 + 3 would be 4: skipped

    def test_test_cases_are_kept_first_and_only_what_is_needed_moves(self):
        cs = cases([("NV", "test", 1)] * 2 + [("NV", "train", 1)] * 5)
        new = bt.balanced_splits(cs, per_class=3, seed=42)
        self.assertTrue(all(new[x["isic_id"]] == "test" for x in cs if x["split"] == "test"))
        self.assertEqual(sum(x["split"] == "train" and new[x["isic_id"]] == "test" for x in cs), 1)

    def test_the_choice_is_fixed_by_the_seed_and_not_by_the_order_of_the_table(self):
        cs = cases([("BCC", "test", 1)] * 8 + [("BCC", "train", 1)] * 8)
        a = bt.balanced_splits(cs, per_class=4, seed=42)
        self.assertEqual(a, bt.balanced_splits(list(reversed(cs)), per_class=4, seed=42))
        self.assertNotEqual(a, bt.balanced_splits(cs, per_class=4, seed=7))
        kept = sorted(x["group_id"] for x in cs if a[x["isic_id"]] == "test")
        self.assertEqual(kept, sorted(sorted({x["group_id"] for x in cs if x["split"] == "test"},
                                             key=lambda g: bt.group_order(42, g))[:4]))

    def test_it_stops_when_a_class_cannot_reach_the_target_or_a_group_is_mixed(self):
        with self.assertRaisesRegex(ValueError, "VASC: only 2 test images"):
            bt.balanced_splits(cases([("VASC", "test", 1), ("VASC", "train", 1), ("VASC", "val", 5)]), per_class=3)
        with self.assertRaisesRegex(ValueError, "only 2 test images"):
            bt.balanced_splits(cases([("SCC", "test", 2), ("SCC", "train", 2)]), per_class=3)    # 2 + 2 > 3, cannot be 3
        mixed = cases([("MEL", "test", 2)])
        mixed[1]["label"] = "NV"
        with self.assertRaisesRegex(ValueError, "more than one class or split"):
            bt.balanced_splits(mixed, per_class=1)


@unittest.skipUnless(TABLE.is_file(), "data/processed_cases/all_cases.csv is not there")
class CommittedTableTests(unittest.TestCase):
    def test_the_committed_cases_give_30_test_images_of_every_class(self):
        cs = read_csv(TABLE)
        new = bt.balanced_splits(cs)
        c = counts(cs, new)
        labels = sorted({x["label"] for x in cs})
        self.assertEqual({lab: c[(lab, "test")] for lab in labels}, {lab: 30 for lab in labels})
        self.assertTrue(all(new[x["isic_id"]] == "val" for x in cs if x["split"] == "val"))
        self.assertEqual(sum(v == "val" for v in new.values()), sum(x["split"] == "val" for x in cs))
        groups = {}
        for x in cs:
            groups.setdefault(x["group_id"], set()).add(new[x["isic_id"]])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        self.assertEqual(bt.balanced_splits(cs), new)                                 # the same again


if __name__ == "__main__":
    unittest.main()
