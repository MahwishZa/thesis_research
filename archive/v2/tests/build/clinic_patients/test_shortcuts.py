"""Step shortcuts (src/build/clinic_patients/shortcuts.py): the encoding, the vote and the folds by hand, then the step on the invented
cohort of project.py."""

import csv
import io
import unittest
from unittest import mock

import numpy as np

from src.build.clinic_patients import clean, dedupe, images, inventory, shortcuts, split, views
from tests.build.clinic_patients.project import Cohort

NR = shortcuts.NOT_RECORDED


class EncodeTests(unittest.TestCase):
    def test_numbers_stay_numbers_with_a_marker_for_a_gap_and_words_become_one_column_per_value(self):
        rows = [{"age": "40", "region": "FACE"}, {"age": NR, "region": "ARM"}, {"age": "50", "region": "FACE"}]
        # columns: age (a gap is 0), age gap (1 where recorded as a gap), region ARM, region FACE
        self.assertEqual(shortcuts.encode(rows, ["age", "region"]).tolist(), [[40, 0, 0, 1], [0, 1, 1, 0], [50, 0, 0, 1]])
        self.assertEqual(shortcuts.encode([{"x": "1"}, {"x": "2"}], ["x"]).tolist(), [[1], [2]])  # no gap, so no gap column
        self.assertEqual(shortcuts.encode(rows, []).shape, (3, 0))

    def test_a_gap_indicator_is_one_where_the_value_is_not_recorded(self):
        rows = [{"a": NR, "b": "1"}, {"a": "x", "b": NR}]
        self.assertEqual(shortcuts.gaps(rows, ["a", "b"]).tolist(), [[1, 0], [0, 1]])


    def test_the_properties_of_a_photo_by_hand(self):
        self.assertEqual(shortcuts.photo_properties({"width": "100", "height": "50", "bytes": "10000"}), [100.0, 50.0, 10000.0, 2.0])


class VoteTests(unittest.TestCase):
    def test_the_nearest_neighbours_decide(self):
        train_x = np.array([[0.0], [0.1], [10.0], [10.1]])
        train_y = np.array(["A", "A", "B", "B"])
        self.assertEqual(shortcuts.vote(train_x, train_y, np.array([[0.05], [10.05], [9.0]]), ["A", "B"], k=1), ["A", "B", "B"])
        self.assertEqual(shortcuts.vote(train_x, train_y, np.array([[4.0]]), ["A", "B"], k=3), ["A"])  # A, A and the nearer B: two votes

    def test_a_tie_goes_to_the_class_that_comes_first_and_k_is_capped_by_the_training_rows(self):
        train_x, train_y = np.array([[0.0], [1.0]]), np.array(["B", "A"])
        self.assertEqual(shortcuts.vote(train_x, train_y, np.array([[0.4]]), ["A", "B"], k=2), ["A"])
        self.assertEqual(shortcuts.vote(train_x, train_y, np.array([[0.4]]), ["B", "A"], k=50), ["B"])

    def test_a_feature_that_never_changes_does_not_break_the_standardising(self):
        train_x = np.array([[1.0, 0.0], [1.0, 5.0]])
        self.assertEqual(shortcuts.vote(train_x, np.array(["A", "B"]), np.array([[1.0, 4.9]]), ["A", "B"], k=1), ["B"])


class FoldTests(unittest.TestCase):
    def test_a_group_is_always_in_one_fold_and_the_folds_are_fixed(self):
        groups = [f"g{i % 40}" for i in range(400)]
        folds = shortcuts.folds(groups)
        self.assertEqual({f for f in folds}, set(range(shortcuts.FOLDS)))
        seen = {}
        for g, f in zip(groups, folds):
            self.assertEqual(seen.setdefault(g, f), f)
        self.assertEqual(folds, shortcuts.folds(groups))

    def test_every_row_gets_a_prediction_made_without_its_own_fold(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(6, 1, (60, 2))])
        y = ["A"] * 60 + ["B"] * 60
        groups = [f"g{i // 2}" for i in range(120)]
        predicted = shortcuts.cross_validate(x, y, groups, ["A", "B"])
        self.assertNotIn("", predicted)
        self.assertGreater(np.mean([p == t for p, t in zip(predicted, y)]), 0.95)  # two far-apart clouds


class ShortcutStepTests(Cohort):
    def prepare(self):
        inventory.inventory(self.data)
        clean.clean(self.data)
        dedupe.dedupe(self.data)
        split.split(self.data)
        images.images(self.data)
        views.views(self.data)

    def result(self, features, task):
        return next(r for r in self.table("shortcuts.csv") if r["features"] == features and r["task"] == task)

    def test_the_empty_fields_of_the_early_records_give_away_a_cancer_and_the_photo_properties_do_not(self):
        self.prepare()
        done = shortcuts.shortcuts(self.data)
        rows = self.table("shortcuts.csv")
        self.assertEqual([(r["features"], r["task"]) for r in rows],
                         [(f, t) for f in ("gaps", "photo properties (originals)", "photo properties (standardised)", "core record", "full record")
                          for t in ("6 classes", "cancer or not")])
        self.assertEqual({(r["task"], r["chance"]) for r in rows}, {("6 classes", "0.1667"), ("cancer or not", "0.5")})
        # in this cohort the cancers never have the block of empty fields and the early benign records always do
        self.assertGreater(float(self.result("gaps", "cancer or not")["balanced_accuracy"]), 0.7)
        for name in ("photo properties (originals)", "photo properties (standardised)"):
            self.assertLess(float(self.result(name, "cancer or not")["balanced_accuracy"]), 0.7, name)  # 40 x 40 noise: nothing to find
            self.assertGreater(float(self.result("gaps", "cancer or not")["balanced_accuracy"]),
                               float(self.result(name, "cancer or not")["balanced_accuracy"]), name)
        for r in rows:
            self.assertTrue(float(r["ci_low"]) <= float(r["balanced_accuracy"]) <= float(r["ci_high"]), r)
        self.assertIn("train split only", done["summary"])
        report = self.report()
        self.assertIn("| Features | 6 classes (chance 16.7%) | Cancer or not (chance 50%) | Recall by class (6 classes) |", report)
        self.assertIn("The validation and test splits are not used.", report)

    def rewrite(self, name, change):
        """build/<name> with ``change(row)`` applied to every row."""
        with (self.data / "build" / name).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            change(r)
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        (self.data / "build" / name).write_text(out.getvalue(), encoding="utf-8")

    def test_only_the_train_split_is_used(self):
        self.prepare()
        shortcuts.shortcuts(self.data)
        first = (self.data / "build" / "shortcuts.csv").read_bytes()
        elsewhere = {r["img_id"] for r in self.table("dataset.csv") if r["split"] != "train"}

        def absurd(r):
            if r["img_id"] in elsewhere:
                r["bytes"], r["width"], r["height"] = "999999999", "5", "5"  # absurd values for photos of validation and test
        self.rewrite("raw_images.csv", absurd)
        self.rewrite("files.csv", absurd)
        shortcuts.shortcuts(self.data)
        self.assertEqual((self.data / "build" / "shortcuts.csv").read_bytes(), first)

    def test_a_hint_planted_in_the_train_photos_is_found_in_the_table_it_was_planted_in_only(self):
        self.prepare()
        shortcuts.shortcuts(self.data)
        plain = float(self.result("photo properties (standardised)", "6 classes")["balanced_accuracy"])
        rows = {r["img_id"]: r for r in self.table("dataset.csv")}
        order = sorted({r["label"] for r in rows.values()})

        def plant(r):
            row = rows.get(r["img_id"])
            if row and row["split"] == "train":
                r["bytes"] = str(1000 * (1 + order.index(row["label"])))  # file size = the class
        self.rewrite("files.csv", plant)
        with mock.patch.object(shortcuts, "K", 3):  # about 10 photos per class in this small cohort: fewer neighbours than K = 15
            shortcuts.shortcuts(self.data)
        found = float(self.result("photo properties (standardised)", "6 classes")["balanced_accuracy"])
        self.assertGreater(found, 0.95)  # the hint is in the photos that are used, so it is found
        self.assertGreater(found, plain + 0.4)
        self.assertLess(float(self.result("photo properties (originals)", "6 classes")["balanced_accuracy"]), 0.5)  # and not in the other table

    def test_the_views_must_have_been_made_first(self):
        with self.assertRaisesRegex(FileNotFoundError, "run the views step first"):
            shortcuts.shortcuts(self.data)

    def test_the_run_is_repeatable(self):
        self.prepare()
        shortcuts.shortcuts(self.data)
        first = (self.data / "build" / "shortcuts.csv").read_bytes()
        shortcuts.shortcuts(self.data)
        self.assertEqual((self.data / "build" / "shortcuts.csv").read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
