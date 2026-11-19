"""Step split (src/build/clinic_patients/split.py) on the invented cohort of project.py."""

import csv
import io
import random
import unittest
from collections import Counter

from src.build.clinic_patients import clean, dedupe, inventory, split
from tests.build.clinic_patients.project import CLASSES, Cohort


class AssignTests(unittest.TestCase):
    def test_a_single_group_goes_to_train_and_the_result_does_not_depend_on_the_order_given(self):
        self.assertEqual(split.assign({"g": Counter({"A": 3})}), {"g": "train"})
        groups = {f"g{i}": Counter({"A": 1 + i % 3, "B": 1}) for i in range(12)}
        reversed_order = dict(reversed(list(groups.items())))
        self.assertEqual(split.assign(groups), split.assign(reversed_order))

    def test_the_group_with_most_of_a_rare_class_is_placed_first_and_every_group_is_placed_once(self):
        groups = {"big": Counter({"RARE": 9, "A": 1}), **{f"s{i}": Counter({"A": 5}) for i in range(8)}}
        result = split.assign(groups)
        self.assertEqual(sorted(result), sorted(groups))
        self.assertEqual(result["big"], "train")  # nothing is placed yet when it comes: the largest target takes it
        self.assertEqual(set(result.values()), {"train", "val", "test"})

    def test_the_rare_class_goes_first_and_a_tie_goes_to_the_first_split_by_hand(self):
        # R has 10 photos: X holds 8, Y and Z one each. X comes first (share 0.8) and takes train (cost -0.32; val +0.48, test +0.16).
        # Then Y: train +0.05, val -0.01, test -0.05, so test; Z: train +0.05, val -0.01, test -0.03, so test again.
        groups = {"X": Counter({"R": 8}), "Y": Counter({"R": 1}), "Z": Counter({"R": 1})}
        self.assertEqual(split.assign(groups), {"X": "train", "Y": "test", "Z": "test"})
        # placed the other way round, Y and Z would both go to train; so X would follow into train too
        self.assertEqual(split.assign({"g": Counter({"A": 10})}, fractions=(("a", 0.5), ("b", 0.5))), {"g": "a"})  # equal costs: the first

    def test_patients_in_two_splits_are_found_and_stop_the_step(self):
        rows = [{"patient_id": "P1", "split": "train"}, {"patient_id": "P1", "split": "test"}, {"patient_id": "P2", "split": "val"},
                {"patient_id": "P3", "split": "val"}, {"patient_id": "P3", "split": "val"}]
        self.assertEqual(split.patients_in_two_splits(rows), ["P1"])
        self.assertEqual(split.patients_in_two_splits([]), [])

    def test_class_shares_come_near_the_targets(self):
        rng = random.Random(1)
        groups = {f"g{i}": Counter({rng.choice("ABC"): rng.randint(1, 4)}) for i in range(150)}
        result = split.assign(groups)
        total, got = Counter(), {s: Counter() for s in split.SPLITS}
        for g, counts in groups.items():
            total.update(counts)
            got[result[g]].update(counts)
        for label in "ABC":
            for name, share in split.FRACTIONS:
                self.assertAlmostEqual(got[name][label] / total[label], share, delta=0.04, msg=(label, name))


class SplitTests(Cohort):
    def prepare(self):
        inventory.inventory(self.data)
        clean.clean(self.data)
        dedupe.dedupe(self.data)

    def test_the_table_of_questions_holds_every_kept_photo_with_its_split_label_file_and_record(self):
        self.prepare()
        done = split.split(self.data)
        rows = self.table("split.csv")
        self.assertEqual((len(rows), done["photos"], done["excluded"]), (121, 121, 4))
        self.assertEqual(list(rows[0])[:7], ["split", "label", "img_id", "patient_id", "lesion_id", "split_group", "biopsed"])
        self.assertEqual(list(rows[0])[7:], ["smoke", "background_father", "age", "gender", "region", "diameter_1", "itch", "grew"])
        by_id = {r["img_id"]: r for r in rows}
        self.assertEqual(len(by_id), 121)
        for left_out in ("P900_1_b.png", "P901_1_a.png", "P901_2_a.png", "P902_1_a.png"):
            self.assertNotIn(left_out, by_id)
        first = by_id["P000_1_a.png"]
        self.assertEqual((first["label"], first["biopsed"], first["gender"]), ("ACK", "FALSE", "not recorded"))  # an early benign record
        self.assertFalse((self.data / "dataset.csv").exists())  # the final table is made by step images
        self.assertEqual({r["reason"] for r in self.table("excluded.csv")}, {"identical_copy", "conflicting_labels", "unreadable"})
        self.assertEqual(sorted(Counter(r["split"] for r in rows)), ["test", "train", "val"])

    def test_no_patient_is_in_two_splits_and_every_class_is_in_every_split_near_the_targets(self):
        self.prepare()
        split.split(self.data)
        rows = self.table("split.csv")
        splits_of = {}
        for r in rows:
            splits_of.setdefault(r["patient_id"], set()).add(r["split"])
        self.assertEqual({len(v) for v in splits_of.values()}, {1})
        for label in CLASSES:
            counts = Counter(r["split"] for r in rows if r["label"] == label)
            self.assertTrue(all(counts[s] > 0 for s in split.SPLITS), (label, counts))
            total = sum(counts.values())
            self.assertAlmostEqual(counts["test"] / total, 0.3, delta=0.1, msg=label)
            self.assertAlmostEqual(counts["train"] / total, 0.6, delta=0.1, msg=label)
        report = self.report()
        self.assertIn("Targets: train 60%, val 10%, test 30%, class by class.", report)
        self.assertIn("- No patient is in two splits (checked).", report)
        self.assertIn(f"- Seed: {split.SEED}", report)

    def test_patients_joined_by_a_near_identical_photo_stay_in_one_split_group(self):
        self.prepare()
        pairs = self.table("near_pairs.csv")
        extra = {"img_a": "P000_1_a.png", "img_b": "P100_1_a.png", "bits": "5", "same_patient": "no", "same_lesion": "no",
                 "label_a": "ACK", "label_b": "BCC", "same_cluster": "no"}
        ghost = {**extra, "img_a": "P901_1_a.png", "img_b": "P200_1_a.png", "label_a": "NEV", "label_b": "MEL"}  # P901 has no photo that stays
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(extra), lineterminator="\n")
        writer.writeheader()
        writer.writerows([*pairs, extra, ghost])
        (self.data / "build" / "near_pairs.csv").write_text(out.getvalue(), encoding="utf-8")
        split.split(self.data)
        rows = self.table("split.csv")
        group = {r["patient_id"]: (r["split"], r["split_group"]) for r in rows}
        self.assertEqual(group["P000"], group["P100"])
        self.assertEqual(group["P000"][1], "G_P000")
        self.assertNotEqual(group["P200"][1], "G_P901")  # a patient with no photo that stays joins nothing
        self.assertIn("patients joined because of a near-identical photo of another patient: 1 links", self.report())

    def test_a_class_missing_from_a_split_stops_the_step_without_writing_the_table(self):
        self.rows = [r for r in self.rows if not (r["diagnostic"] == "MEL" and r["patient_id"] != "P200")]
        self.photos = {r["img_id"]: self.photos[r["img_id"]] for r in self.rows}
        for p in (self.raw / "images" / "part_1").glob("P2*_1_*.png"):
            p.unlink()
        self.write()
        self.prepare()
        with self.assertRaisesRegex(ValueError, "a class has no photo in a split: MEL in val, MEL in test"):
            split.split(self.data)
        self.assertFalse((self.data / "build" / "split.csv").exists())

    def test_the_check_that_no_patient_is_in_two_splits_is_made_by_the_step(self):
        from unittest import mock
        self.prepare()
        with mock.patch.object(split, "patients_in_two_splits", return_value=["P000"]):
            with self.assertRaisesRegex(ValueError, "patients in more than one split: `P000`"):
                split.split(self.data)

    def test_the_steps_before_it_must_have_run(self):
        with self.assertRaisesRegex(FileNotFoundError, "run the clean step first"):
            split.split(self.data)

    def test_the_split_is_reproducible_to_the_byte(self):
        self.prepare()
        split.split(self.data)
        first = (self.data / "build" / "split.csv").read_bytes(), (self.data / "build" / "excluded.csv").read_bytes()
        split.split(self.data)
        self.assertEqual((self.data / "build" / "split.csv").read_bytes(), first[0])
        self.assertEqual((self.data / "build" / "excluded.csv").read_bytes(), first[1])


if __name__ == "__main__":
    unittest.main()
