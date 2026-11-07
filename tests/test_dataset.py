import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src import dataset
from tests.support import make_dataset

SPLITS = {"train": "Train", "test": "Test"}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        make_dataset(self.root, {"Train": {"melanoma": 4, "nevus": 3}, "Test": {"melanoma": 2, "nevus": 1}})
        (self.root / "Train" / "nevus" / "notes.txt").write_text("x", encoding="utf-8")
        (self.root / "Test" / "nevus" / "broken.jpg").write_bytes(b"not a jpeg")
        shutil.copy(self.root / "Train" / "melanoma" / "ISIC_0000001.jpg", self.root / "Test" / "melanoma" / "copy.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_records_and_problems(self):
        records, problems = dataset.scan(self.root, SPLITS)
        self.assertEqual(len(records), 4 + 3 + 3 + 1)
        self.assertEqual(dataset.labels_of(records), ["melanoma", "nevus"])
        self.assertEqual({p for p, _ in problems}, {"Train/nevus/notes.txt", "Test/nevus/broken.jpg"})
        first = next(r for r in records if r.path == "Train/melanoma/ISIC_0000001.jpg")
        self.assertEqual((first.split, first.label, first.width, first.height, first.isic_id), ("train", "melanoma", 60, 45, "ISIC_0000001"))

    def test_missing_split_folder_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            dataset.scan(self.root, {"train": "Train", "test": "Nope"})

    def test_dev_subset_is_seeded_stratified_and_training_only(self):
        records, _ = dataset.scan(self.root, SPLITS)
        a = dataset.dev_subset(records, 2, seed=0)
        self.assertEqual(a, dataset.dev_subset(records, 2, seed=0))
        self.assertEqual(sorted(r.label for r in a), ["melanoma", "melanoma", "nevus", "nevus"])
        self.assertTrue(all(r.split == "train" for r in a))

    def test_items_hash_changes_with_content(self):
        records, _ = dataset.scan(self.root, SPLITS)
        before = dataset.items_hash(records)
        (self.root / "Train" / "nevus" / "ISIC_0000005.jpg").write_bytes((self.root / "Train" / "melanoma" / "ISIC_0000002.jpg").read_bytes())
        self.assertNotEqual(before, dataset.items_hash(dataset.scan(self.root, SPLITS)[0]))

    def test_small_image_is_sent_unchanged_and_large_one_scaled_down(self):
        small = self.root / "Train" / "melanoma" / "ISIC_0000001.jpg"
        data, mime, info = dataset.prepare_image(small, 2048)
        self.assertEqual((data, mime, info["resized"]), (small.read_bytes(), "image/jpeg", False))
        big = Path(self.tmp.name) / "big.png"
        Image.new("RGB", (3000, 1000), (10, 20, 30)).save(big)
        data, mime, info = dataset.prepare_image(big, 2048)
        self.assertEqual((mime, info["resized"], info["width"], info["height"]), ("image/jpeg", True, 2048, 683))
        with Image.open(io.BytesIO(data)) as im:
            self.assertEqual(im.size, (2048, 683))

    def test_an_image_exactly_at_the_size_limit_is_sent_unchanged_and_one_pixel_more_is_scaled(self):
        for size, resized in (((2048, 100), False), ((100, 2048), False), ((2049, 100), True), ((100, 2049), True)):
            path = Path(self.tmp.name) / f"edge{size[0]}x{size[1]}.jpg"
            Image.new("RGB", size, (10, 20, 30)).save(path)
            data, mime, info = dataset.prepare_image(path, 2048)
            self.assertEqual(info["resized"], resized, size)
            self.assertEqual(data == path.read_bytes(), not resized, size)

    def test_profile_reports_cross_split_duplicate(self):
        out = Path(self.tmp.name) / "profile"
        report = dataset.profile(self.root, SPLITS, out, Path(self.tmp.name) / "sheets")
        text = report.read_text(encoding="utf-8")
        self.assertIn("(ACROSS SPLITS)", text)
        self.assertIn("| melanoma | 4 | 3 |", text)
        self.assertTrue((out / "manifest.csv").exists())
        self.assertEqual(len(list((Path(self.tmp.name) / "sheets").glob("*.jpg"))), 4)


if __name__ == "__main__":
    unittest.main()
