"""The as-of benchmark builder: dates, groups, the rebuilt MedChangeQA, noise flags and the seeded split (synthetic MedRevQA rows)."""

import unittest

from experiments.medchange.benchmark import (
    BenchmarkError, assign_splits, build_items, parse_cochrane_date,
    rebuild_medchangeqa, verify_against_release,
)
from tests.fixtures import fixture


class BenchmarkTests(unittest.TestCase):

    def test_dates_and_precision(self):
        self.assertEqual(parse_cochrane_date("Cochrane Database Syst Rev. 2024 Jan 18;1(1):CD011039."),
                         ("2024-01-18", "day"))
        self.assertEqual(parse_cochrane_date("Cochrane Database Syst Rev. 2001;(3):CD002120."),
                         ("2001-01-01", "year"))
        with self.assertRaises(BenchmarkError):
            parse_cochrane_date("no date here")

    def test_blank_group_id_continues_previous_group(self):
        _, groups = fixture()
        self.assertEqual(groups, {1: [0, 1], 2: [2, 3], 3: [4, 5]})

    def test_rebuild_uses_authors_rule_and_verification_catches_mismatch(self):
        medrev, groups = fixture()
        rebuilt = rebuild_medchangeqa(medrev, groups)
        self.assertEqual(rebuilt, [(1, 0, 1), (3, 4, 5)])
        good = [{"Newest Label": "SUPPORTED", "Outdated Label": "NOT ENOUGH INFORMATION"},
                {"Newest Label": "SUPPORTED", "Outdated Label": "NOT ENOUGH INFORMATION"}]
        verify_against_release(rebuilt, medrev, good)
        bad = [dict(good[0]), {"Newest Label": "REFUTED", "Outdated Label": "SUPPORTED"}]
        with self.assertRaises(BenchmarkError):
            verify_against_release(rebuilt, medrev, bad)

    def test_items_flag_noise_and_sample_only_unchanged_groups(self):
        medrev, groups = fixture()
        items = build_items(medrev, groups, rebuild_medchangeqa(medrev, groups),
                            n_unchanged=5, seed=1)
        by = {i.item_id: i for i in items}
        self.assertFalse(by["MC-00001"].likely_label_noise)
        self.assertTrue(by["MC-00003"].likely_label_noise)
        self.assertEqual(by["MC-00002"].kind, "unchanged")
        self.assertEqual(by["MC-00001"].previous.date, "2010-01-02")
        self.assertEqual(by["MC-00001"].change_type, "NOT ENOUGH INFORMATION -> SUPPORTED")

    def test_splits_are_deterministic(self):
        medrev, groups = fixture()
        def run():
            items = build_items(medrev, groups, rebuild_medchangeqa(medrev, groups),
                                n_unchanged=5, seed=1)
            assign_splits(items, dev_fraction=0.5, seed=7)
            return [(i.item_id, i.split) for i in items]
        self.assertEqual(run(), run())
        self.assertTrue(all(s in ("dev", "confirm") for _, s in run()))


class NoiseAndSplitTests(unittest.TestCase):
    def _medrev(self, conclusion_new, conclusion_old):
        def row(i, label, conclusion, date):
            return {"": str(i), "Label": label, "Question": f"Does X help Y {i}?", "PMID": f"/{i}/", "objectives": "", "conclusions": conclusion,
                    "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + i}. doi: x"}
        return {0: row(0, "SUPPORTED", conclusion_new, "2020 Mar 5"), 1: row(1, "REFUTED", conclusion_old, "2012 Mar 5")}

    def test_a_label_change_between_near_identical_conclusions_is_flagged_as_noise_from_a_similarity_of_0_85(self):
        from experiments.medchange.benchmark import NOISE_SIMILARITY, _make_item
        self.assertEqual(NOISE_SIMILARITY, 0.85)
        same = _make_item(self._medrev("a" * 100, "a" * 90 + "b" * 10), 7, 0, 1, "changed")        # ratio 0.90
        different = _make_item(self._medrev("a" * 100, "a" * 70 + "b" * 30), 7, 0, 1, "changed")   # ratio 0.70
        self.assertEqual((same.conclusion_similarity, same.likely_label_noise), (0.9, True))
        self.assertEqual((different.conclusion_similarity, different.likely_label_noise), (0.7, False))
        unchanged = _make_item(self._medrev("a" * 100, "a" * 100), 7, 0, 1, "unchanged")
        self.assertFalse(unchanged.likely_label_noise)                 # only changed items can be noise

    def test_the_development_share_of_a_stratum_is_rounded_to_nearest_not_down(self):
        from experiments.medchange.benchmark import Item, Version, assign_splits
        v = Version(row=1, pmid="1", cochrane_id="CD1", date="2020-01-01", date_precision="day", label="SUPPORTED")
        items = [Item(item_id=f"MC-{k:05d}", group_id=k, kind="unchanged", question="Q?", newest=v, previous=v, change_type=None,
                      decisive_flip=False, conclusion_similarity=0.0, likely_label_noise=False, ad_related=False) for k in range(5)]
        assign_splits(items, dev_fraction=0.3, seed=1)                  # 5 x 0.3 = 1.5 -> 2 development items
        self.assertEqual(sorted(i.split for i in items), ["confirm"] * 3 + ["dev"] * 2)


if __name__ == "__main__":
    unittest.main()
