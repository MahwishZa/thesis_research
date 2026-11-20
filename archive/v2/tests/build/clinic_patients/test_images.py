"""Step images (src/build/clinic_patients/images.py) on the invented cohort of project.py."""

import io
import random
from unittest import mock

import numpy as np
from PIL import Image, ImageCms

from src.build.clinic_patients import clean, dedupe, images, inventory, split
from tests.build.clinic_patients.project import Cohort, noise_png


def png(image: Image.Image, **options) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", **options)
    return out.getvalue()


def bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(dedupe.POPCOUNT[a ^ b].sum())


def good(**change):
    row = {"img_id": "a", "psnr_db": 40.0, "hash_bits": 5, "nearest_other_bits": 120, "bytes": 1000}
    return {**row, **change}


class PsnrAndPrepareTests(Cohort):
    def test_psnr_by_hand(self):
        zeros, full = np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 255, dtype=np.uint8)
        self.assertEqual(images.psnr(zeros, zeros), 99.0)  # equal pictures
        self.assertAlmostEqual(images.psnr(zeros, full), 0.0)  # mean squared error 255 squared
        self.assertAlmostEqual(images.psnr(zeros, np.ones((2, 2, 3), dtype=np.uint8)), 10 * np.log10(255 ** 2), places=6)  # 48.13 dB

    def test_a_photo_is_turned_upright_by_its_exif_orientation(self):
        picture = Image.frombytes("RGB", (40, 20), bytes(range(256)) * 9 + bytes(240))
        exif = Image.Exif()
        exif[images.ORIENTATION] = 6  # to be turned a quarter turn to look upright
        saved, quality, changes, raw_size = images.prepare_image(png(picture, exif=exif))
        self.assertEqual(raw_size, (40, 20))
        self.assertEqual(Image.open(io.BytesIO(saved)).size, (20, 40))
        self.assertEqual(changes, ["turned upright as its EXIF orientation tag says"])

    def test_colours_are_converted_from_the_embedded_profile_and_other_modes_to_rgb(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        base = Image.frombytes("RGB", (30, 30), random.Random(3).randbytes(30 * 30 * 3))
        saved, _, changes, _ = images.prepare_image(png(base, icc_profile=profile))
        self.assertEqual(len(changes), 1)
        self.assertTrue(changes[0].startswith("colours converted from its embedded colour profile (") and changes[0].endswith(") to sRGB"))
        self.assertEqual(Image.open(io.BytesIO(saved)).mode, "RGB")
        self.assertEqual(images.prepare_image(png(base.convert("L")))[2], ["converted from L to RGB"])
        opaque = base.convert("RGBA")
        self.assertEqual(images.prepare_image(png(opaque))[2], ["converted from RGBA to RGB"])
        see_through = opaque.copy()
        see_through.putpixel((3, 3), (1, 2, 3, 0))
        with self.assertRaisesRegex(ValueError, "it has transparent pixels"):
            images.prepare_image(png(see_through))

    def test_only_a_photo_longer_than_the_limit_is_shrunk_and_never_stretched_or_enlarged(self):
        with mock.patch.object(images, "MAX_SIDE", 30):
            saved, _, changes, _ = images.prepare_image(noise_png(1, 60))
            self.assertEqual((Image.open(io.BytesIO(saved)).size, changes), ((30, 30), ["shrunk from 60x60 to 30x30"]))
            wide = png(Image.frombytes("RGB", (90, 40), bytes(range(256)) * 42 + bytes(48)))
            self.assertEqual(Image.open(io.BytesIO(images.prepare_image(wide)[0])).size, (30, 13))  # 40 x 30 / 90 = 13.3: proportions kept
            exact = images.prepare_image(noise_png(2, 30))  # a longest side equal to the limit is not shrunk
            self.assertEqual(exact[2], ["saved again only"])
            small = png(Image.frombytes("RGB", (20, 10), bytes(range(200)) * 3))
            saved, _, changes, _ = images.prepare_image(small)
            self.assertEqual((Image.open(io.BytesIO(saved)).size, changes), ((20, 10), ["saved again only"]))  # not enlarged

    def test_the_saved_file_holds_no_metadata_and_is_a_jpeg_of_the_one_setting(self):
        picture = Image.frombytes("RGB", (40, 40), bytes(range(256)) * 18 + bytes(192))
        exif = Image.Exif()
        exif[0x010F] = "a phone"
        saved = images.prepare_image(png(picture, exif=exif))[0]
        with Image.open(io.BytesIO(saved)) as jpeg:
            self.assertEqual(jpeg.format, "JPEG")
            self.assertEqual(len(jpeg.getexif()), 0)
            self.assertNotIn("icc_profile", jpeg.info)
            self.assertEqual(jpeg.layer[0][1:3], (1, 1))  # no chroma subsampling (4:4:4): every colour component at full size


class CheckTests(Cohort):
    def test_each_bar_names_the_photos_that_miss_it(self):
        images._check([good()])
        with self.assertRaisesRegex(ValueError, "1 photo\\(s\\) with fidelity below 30.0 dB: `x`"):
            images._check([good(), good(img_id="x", psnr_db=29.9)])
        with self.assertRaisesRegex(ValueError, "not nearer to its own original than to another photo's original: `y`"):
            images._check([good(img_id="y", hash_bits=40, nearest_other_bits=40)])  # a tie is not enough
        images._check([good(hash_bits=39, nearest_other_bits=40)])
        with self.assertRaisesRegex(ValueError, "file larger than 4000000 bytes: `z`"):
            images._check([good(img_id="z", bytes=4_000_001)])
        images._check([good(bytes=4_000_000)])


class ImagesStepTests(Cohort):
    def prepare(self):
        inventory.inventory(self.data)
        clean.clean(self.data)
        dedupe.dedupe(self.data)
        split.split(self.data)

    def test_every_kept_photo_is_made_listed_and_joined_to_its_question(self):
        self.prepare()
        done = images.images(self.data)
        self.assertEqual((done["photos"], done["written"], done["removed"]), (121, 121, 0))
        files = self.table("files.csv")
        self.assertEqual(len(files), 121)
        for r in files[:5]:
            saved = (self.data / "files" / "photos" / r["path"]).read_bytes()
            self.assertEqual(r["sha256"], images.hashlib.sha256(saved).hexdigest())
            self.assertEqual((r["changes"], int(r["bytes"])), ("saved again only", len(saved)))
            self.assertGreaterEqual(float(r["psnr_db"]), 35.0)
            self.assertLess(int(r["hash_bits"]), int(r["nearest_other_bits"]))
        # the two measures are the real distances: its own original, and the nearest hash of any other photo's original
        by_id = {r["img_id"]: r for r in files}
        originals = {i: dedupe.dhash(self.raw / r["raw_path"]) for i, r in by_id.items()}
        for r in files[:5]:
            saved = dedupe.dhash(self.data / "files" / "photos" / r["path"])
            self.assertEqual(int(r["hash_bits"]), bits(saved, originals[r["img_id"]]))
            self.assertEqual(int(r["nearest_other_bits"]), min(bits(saved, h) for i, h in originals.items() if i != r["img_id"]))
        raw = {r["img_id"]: r for r in self.table("raw_images.csv")}
        questions = {r["img_id"]: r for r in self.table("split.csv")}
        self.assertEqual(sorted(by_id), sorted(questions))  # one photo for every question, and for no other
        first = files[0]
        self.assertEqual(first["path"], f"{questions[first['img_id']]['split']}/{questions[first['img_id']]['label']}/"
                                        f"{first['img_id'][:-4]}.jpg")
        self.assertEqual((first["raw_path"], first["raw_sha256"]), (raw[first["img_id"]]["file"], raw[first["img_id"]]["sha256"]))
        photos = self.data / "files" / "photos"
        self.assertEqual(sorted(p.relative_to(photos).as_posix() for p in photos.rglob("*.jpg")), sorted(r["path"] for r in files))
        self.assertEqual([p.name for p in (self.data / "files").iterdir()], ["photos"])  # the records are made by step views, elsewhere
        self.assertFalse((self.data / "dataset.csv").exists())  # the table of questions is made by step views

    def test_the_originals_are_never_changed(self):
        self.prepare()
        before = {p.name: p.read_bytes() for p in (self.raw / "images" / "part_1").iterdir()} | {"m": (self.raw / "metadata.csv").read_bytes()}
        images.images(self.data)
        after = {p.name: p.read_bytes() for p in (self.raw / "images" / "part_1").iterdir()} | {"m": (self.raw / "metadata.csv").read_bytes()}
        self.assertEqual(before, after)

    def test_a_replaced_original_stops_the_step(self):
        self.prepare()
        (self.raw / "images" / "part_1" / "P000_1_a.png").write_bytes(noise_png(777))
        with self.assertRaisesRegex(ValueError, "differs from build/raw_images.csv"):
            images.images(self.data)
        (self.raw / "images" / "part_1" / "P000_1_a.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "P000_1_a.png not found"):
            images.images(self.data)

    def test_a_photo_with_transparent_pixels_stops_the_step_and_names_it(self):
        clear = Image.new("RGBA", (40, 40), (10, 20, 30, 255))
        clear.putpixel((0, 0), (1, 1, 1, 0))
        self.photos["P000_1_a.png"] = png(clear)
        self.write()
        self.prepare()
        with self.assertRaisesRegex(ValueError, "P000_1_a.png: it has transparent pixels"):
            images.images(self.data)

    def test_a_photo_that_misses_the_fidelity_bar_stops_the_step_before_the_table_is_written(self):
        self.prepare()
        with mock.patch.object(images, "MIN_PSNR", 99.0):
            with self.assertRaisesRegex(ValueError, "photo\\(s\\) with fidelity below 99.0 dB"):
                images.images(self.data)
        self.assertFalse((self.data / "dataset.csv").exists())
        self.assertFalse((self.data / "build" / "files.csv").exists())

    def test_shrinking_is_reported_and_a_second_run_writes_nothing_and_removes_what_is_no_longer_wanted(self):
        self.photos["P000_1_a.png"] = noise_png(5, 60)
        self.write()
        self.prepare()
        with mock.patch.object(images, "MAX_SIDE", 50):  # only the 60 x 60 photo is longer
            images.images(self.data)
            row = next(r for r in self.table("files.csv") if r["img_id"] == "P000_1_a.png")
            self.assertEqual((row["raw_width"], row["width"], row["changes"]), ("60", "50", "shrunk from 60x60 to 50x50"))
            (self.data / "files" / "photos" / "train" / "actinic keratosis").mkdir(parents=True, exist_ok=True)
            (self.data / "files" / "photos" / "train" / "actinic keratosis" / "stray.jpg").write_bytes(b"x")
            (self.data / "files" / "photos" / "empty" / "folder").mkdir(parents=True)
            (self.data / "files" / "records").mkdir(parents=True)  # another step's folder is not touched
            (self.data / "files" / "records" / "keep.txt").write_bytes(b"x")
            (self.data / "files" / "photos" / "notes.txt").write_bytes(b"x")  # only photos are removed, never another kind of file
            done = images.images(self.data)
        self.assertEqual((done["written"], done["removed"]), (0, 1))
        self.assertFalse((self.data / "files" / "photos" / "train" / "actinic keratosis" / "stray.jpg").exists())
        self.assertFalse((self.data / "files" / "photos" / "empty").exists())
        self.assertTrue((self.data / "files" / "records" / "keep.txt").exists())
        self.assertTrue((self.data / "files" / "photos" / "notes.txt").exists())
        report = self.report()
        self.assertEqual(images.MAX_SIDE, 1024)  # the value the data sheet states and the models' photos are made with
        for expected in ("- Photos made: 121", "- saved again only: 120; shrunk: 1",
                         "| Longest side (pixels): smallest, median, largest | 40, 40, 60 | 40, 40, 50 |",
                         "| Files over 5 MB | 0 | 0 |", "| Fidelity: PSNR", "| Same picture: bits by which", "nearest *other* photo's original"):
            self.assertIn(expected, report)

    def test_the_steps_before_it_must_have_run(self):
        with self.assertRaisesRegex(FileNotFoundError, "run the split step first"):
            images.images(self.data)

    def test_the_run_is_repeatable_to_the_byte(self):
        self.prepare()
        images.images(self.data)
        first = (self.data / "build" / "files.csv").read_bytes()
        images.images(self.data)
        self.assertEqual((self.data / "build" / "files.csv").read_bytes(), first)


if __name__ == "__main__":
    import unittest
    unittest.main()
