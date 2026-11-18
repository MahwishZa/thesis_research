"""The prepared datasets (src/datasets.py): reading a table and checking every file against it."""

import hashlib
import unittest
from pathlib import Path

from PIL import Image

from src import datasets

from tests.fixtures import Project, write_table

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------------------------------
# The prepared dataset


class DatasetTests(Project):
    header = "split,label,path,sha256\n"

    def _table(self, *rows: str) -> None:
        self.table.write_text(self.header + "".join(r + "\n" for r in rows), encoding="utf-8")

    def _row(self, path: str) -> str:
        split, label, _ = path.split("/")
        return f"{split},{label},{path},{hashlib.sha256((self.image_dir / path).read_bytes()).hexdigest()}"

    def test_a_consistent_dataset_loads_with_its_classes_from_the_training_split(self):
        items = datasets.load_dataset()
        self.assertEqual(len(items), 14)
        self.assertEqual(sorted({i.split for i in items}), ["test", "train", "val"])
        self.assertEqual(datasets.classes(items), ["melanoma", "nevus"])

    def test_a_missing_table_says_the_dataset_is_not_prepared_yet(self):
        self.table.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "does not exist yet"):
            datasets.load_dataset()

    def test_changed_missing_and_extra_images_stop_with_a_count_of_each(self):
        Image.new("RGB", (60, 45), (1, 2, 3)).save(self.image_dir / "test" / "nevus" / "ISIC_0000013.jpg")
        (self.image_dir / "val" / "nevus" / "ISIC_0000009.jpg").unlink()
        (self.image_dir / "train" / "nevus" / "stray.jpg").write_bytes(b"x")
        with self.assertRaises(ValueError) as raised:
            datasets.load_dataset()
        for expected in ("1 changed", "1 missing", "1 not in the table", "changed: test/nevus/ISIC_0000013.jpg"):
            self.assertIn(expected, str(raised.exception))

    def test_files_that_windows_and_macos_put_in_folders_are_not_extra_images(self):
        for name in ("Thumbs.db", "desktop.ini", ".DS_Store"):
            (self.image_dir / "test" / "nevus" / name).write_bytes(b"x")
        self.assertEqual(len(datasets.load_dataset()), 14)

    def test_a_table_that_contradicts_itself_is_refused(self):
        good = "train/nevus/ISIC_0000004.jpg"
        sha = hashlib.sha256((self.image_dir / good).read_bytes()).hexdigest()
        cases = {
            "lacks the column": ("split,label,path\n", ["train,nevus," + good]),
            "is not one of": (self.header, [f"holdout,nevus,{good},{sha}"]),
            "must be <split>/<label>/<file>": (self.header, [f"train,melanoma,{good},{sha}"]),  # the folder says nevus
            "not a JPEG or PNG": (self.header, [f"train,nevus,train/nevus/a.gif,{sha}"]),
            "64 lowercase": (self.header, [f"train,nevus,{good},{sha.upper()}"]),
            "same image is also on line 2": (self.header, [self._row(good), f"test,nevus,test/nevus/copy.jpg,{sha}"]),
            "same path is also on line 2": (self.header, [self._row(good), self._row(good)]),
            "lists no file": (self.header, []),
        }
        for message, (header, rows) in cases.items():
            with self.subTest(message):
                self.table.write_text(header + "".join(r + "\n" for r in rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    datasets.read_table()
        for bad in ("train/../ISIC_0000004.jpg", "train/nevus/../x.jpg"):  # a path cannot point outside its class folder
            with self.subTest(bad):
                self._table(f"train,nevus,{bad},{sha}")
                with self.assertRaises(ValueError):
                    datasets.read_table()

    def test_a_validation_or_test_class_without_training_images_is_refused(self):
        folder = self.image_dir / "test" / "vascular lesion"
        folder.mkdir()
        Image.new("RGB", (60, 45), (200, 0, 0)).save(folder / "ISIC_0000099.jpg")
        write_table(self.base)
        with self.assertRaisesRegex(ValueError, "vascular lesion"):
            datasets.classes(datasets.load_dataset())

    def test_the_image_is_sent_exactly_as_stored(self):
        item = next(i for i in datasets.load_dataset() if i.split == "test")
        content, mime, sent = datasets.read_image(item)
        self.assertEqual(content, (self.image_dir / item.path).read_bytes())
        self.assertEqual((mime, sent["sha256"], sent["width"], sent["height"]), ("image/jpeg", item.sha256, 60, 45))

    def test_an_unreadable_image_is_reported_not_skipped(self):
        (self.image_dir / "test" / "nevus" / "ISIC_0000013.jpg").write_bytes(b"not a jpeg")
        write_table(self.base)
        item = next(i for i in datasets.load_dataset() if i.path == "test/nevus/ISIC_0000013.jpg")
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            datasets.read_image(item)

    def test_the_hash_of_a_list_of_images_changes_with_any_file_label_or_split(self):
        items = datasets.load_dataset()
        base = datasets.items_hash(items)
        self.assertEqual(base, datasets.items_hash(list(reversed(items))))  # the order does not matter
        first = items[0]
        for change in ({"sha256": "0" * 64}, {"label": "other"}, {"split": "val"}):
            with self.subTest(change):
                self.assertNotEqual(datasets.items_hash([datasets.Item(**{**first.__dict__, **change}), *items[1:]]), base)


if __name__ == "__main__":
    unittest.main()
