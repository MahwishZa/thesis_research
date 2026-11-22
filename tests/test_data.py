"""Checks of the dataset tables and question files in data/ (no images needed). They skip until the tables are in data/.
Set THESIS_DATA to check another copy of the folder."""

import csv
import json
import os
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("THESIS_DATA", ROOT / "data"))
SPLITS = ("train", "val", "test")
VIEWS = ("image_only", "text_only", "multimodal")
CLASS_WORDS = ("melanoma", "nevus", "carcinoma", "keratosis", "dermatofibroma", "vascular")


def rows(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@unittest.skipUnless((DATA / "master_clean.csv").is_file(), "the dataset tables are not in data/ yet")
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = {r["isic_id"]: r for r in rows("master_clean.csv")}
        cls.labels = json.loads((DATA / "label_map.json").read_text(encoding="utf-8"))

    def test_every_case_is_once_in_one_split_with_a_known_label(self):
        ids = [r["isic_id"] for r in rows("master_clean.csv")]
        self.assertEqual(len(ids), len(set(ids)), "a case is listed twice")
        for r in self.master.values():
            self.assertIn(r["split"], SPLITS)
            self.assertIn(r["label"], self.labels)
            self.assertEqual(r["cv_fold"] == "-1", r["split"] == "test", f"{r['isic_id']}: the test split must be outside the folds")

    def test_no_lesion_group_is_in_two_splits(self):
        splits = {}
        for r in self.master.values():
            splits.setdefault(r["group_id"], set()).add(r["split"])
        self.assertEqual([g for g, s in splits.items() if len(s) > 1], [])

    def test_the_three_views_hold_the_same_cases_splits_and_labels(self):
        for view in VIEWS:
            with self.subTest(view):
                table = {r["isic_id"]: r for r in rows(f"dataset_{view}.csv")}
                self.assertEqual(set(table), set(self.master))
                for i, r in table.items():
                    self.assertEqual((r["split"], r["label"]), (self.master[i]["split"], self.master[i]["label"]), i)

    def test_every_question_is_a_case_of_its_split_and_its_answer_is_the_label(self):
        for view in VIEWS:
            for split in SPLITS:
                with self.subTest(f"{view} {split}"):
                    items = jsonl(f"qa/qa_{view}_{split}.jsonl")
                    ids = [q["id"] for q in items]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {i for i, r in self.master.items() if r["split"] == split})
                    for q in items:
                        self.assertEqual(q["options"][q["answer"]], q["answer_text"], q["id"])
                        self.assertEqual(q["gold_label"], self.master[q["id"]]["label"], q["id"])
                        self.assertFalse(any(w in q["question"].lower() for w in CLASS_WORDS), f"{q['id']}: the question names a class")
                        self.assertEqual("image_path" in q, view != "text_only", f"{q['id']}: image path and view disagree")
                        if view == "text_only":
                            self.assertNotIn("image", q["question"].lower(), f"{q['id']}: a text-only question mentions an image")

    def test_the_balanced_lists_use_training_cases_only_with_equal_classes(self):
        train = {i for i, r in self.master.items() if r["split"] == "train"}
        for name in ("train_oversampled_index.csv", "train_undersampled_index.csv"):
            with self.subTest(name):
                table = rows(name)
                self.assertTrue({r["isic_id"] for r in table} <= train)
                self.assertTrue(all(r["label"] == self.master[r["isic_id"]]["label"] for r in table))
                self.assertEqual(len(set(Counter(r["label"] for r in table).values())), 1, "classes are not equal")

    def test_the_class_weights_are_from_the_training_counts(self):
        counts = Counter(r["label"] for r in self.master.values() if r["split"] == "train")
        balance = json.loads((DATA / "class_balance.json").read_text(encoding="utf-8"))
        self.assertEqual(balance["class_counts_train"], dict(counts))

    def test_every_original_file_in_the_audit_is_a_kept_case_or_a_marked_copy_of_one(self):
        for r in rows("original_files_audit.csv"):
            if r["isic_id"] not in self.master:
                self.assertTrue(r["split"].startswith("removed: same picture as "), r["isic_id"])
                self.assertIn(r["split"].removeprefix("removed: same picture as "), self.master, r["isic_id"])

    def test_the_audit_lists_all_2357_files_of_the_kaggle_download(self):
        self.assertEqual(len(rows("original_files_audit.csv")), 2357)

    def test_the_answer_options_are_not_in_one_fixed_order(self):
        orders = {json.dumps(q["options"]) for q in jsonl("qa/qa_image_only_test.jsonl")}
        self.assertGreater(len(orders), 1)

    def test_every_image_path_is_images_slash_the_case_id(self):
        expected = {i: f"images/{i}.jpg" for i in self.master}
        for name in ("master_clean.csv", "dataset_image_only.csv", "dataset_multimodal.csv"):
            with self.subTest(name):
                wrong = [r["isic_id"] for r in rows(name) if r["image_path"] != expected[r["isic_id"]]]
                self.assertEqual(wrong[:5], [], f"{len(wrong)} image paths are not images/<id>.jpg")
        for view in ("image_only", "multimodal"):
            for split in SPLITS:
                with self.subTest(f"{view} {split}"):
                    wrong = [q["id"] for q in jsonl(f"qa/qa_{view}_{split}.jsonl") if q["image_path"] != expected[q["id"]]]
                    self.assertEqual(wrong[:5], [], f"{len(wrong)} image paths are not images/<id>.jpg")

    def test_the_text_only_table_does_not_mention_an_image(self):
        self.assertFalse(any("image" in r["clinical_text"].lower() for r in rows("dataset_text_only.csv")))


if __name__ == "__main__":
    unittest.main()
