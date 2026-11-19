"""The PAD-UFES-20 inventory (src/build/clinic_patients/prepare.py) on a tiny invented download laid out like the real one: no network, no real
data. Patients P1 (two lesions) and P2; lesion L1 has two photos."""

import csv
import hashlib
import io
import random
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.build.clinic_patients import prepare

COLUMNS = ["patient_id", "lesion_id", "smoke", "age", "gender", "region", "diameter_1", "itch", "diagnostic", "img_id", "biopsed"]
ROWS = [
    ["P1", "L1", "False", "55", "FEMALE", "FACE", "8.0", "True", "BCC", "P1_L1_a.png", "True"],
    ["P1", "L1", "False", "55", "FEMALE", "FACE", "9.0", "UNK", "BCC", "P1_L1_b.png", "True"],
    ["P1", "L2", "False", "55", "FEMALE", "BACK", "", "False", "NEV", "P1_L2_a.png", "False"],
    ["P2", "L3", "", "71", "MALE", "ARM", "12.0", "True", "ACK", "P2_L3_a.png", "False"],
]


def png(colour, size=(30, 20)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="PNG")
    return out.getvalue()


def noise(side: int, seed: int) -> bytes:
    """A picture of random pixels: it does not compress, so its file size grows with its side."""
    out = io.BytesIO()
    Image.frombytes("RGB", (side, side), random.Random(seed).randbytes(side * side * 3)).save(out, format="PNG")
    return out.getvalue()


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        self.data = self.base / "data" / "clinic_patients"
        self.raw = self.data / "raw"
        (self.raw / "images" / "imgs_part_1").mkdir(parents=True)
        (self.raw / "images" / "imgs_part_2").mkdir(parents=True)
        self.write_metadata(COLUMNS, ROWS)
        colours = {"P1_L1_a.png": (200, 0, 0), "P1_L1_b.png": (0, 200, 0), "P1_L2_a.png": (0, 0, 200), "P2_L3_a.png": (9, 9, 9)}
        for i, (name, colour) in enumerate(colours.items()):
            (self.raw / "images" / f"imgs_part_{1 + i % 2}" / name).write_bytes(png(colour))
        (self.data / "downloads").mkdir()
        (self.data / "downloads" / "zr7vgbcyr2-1.zip").write_bytes(b"the zip as downloaded")

    def write_metadata(self, columns, rows, bom=False):
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
        (self.raw / "metadata.csv").write_text(("﻿" if bom else "") + out.getvalue(), encoding="utf-8")

    def table(self, name):
        with (self.data / "build" / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_counts_classes_and_biopsy_column_as_written(self):
        done = prepare.inventory(self.data)
        self.assertEqual((done["rows"], done["photos"], done["lesions"], done["patients"], done["columns"]), (4, 4, 3, 2, 11))
        self.assertEqual(done["classes"], {"BCC": 2, "NEV": 1, "ACK": 1})
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("| Class | Photos | Lesions | Patients | `biopsed` = 'False' | `biopsed` = 'True' |", report)
        self.assertIn("| BCC | 2 | 1 | 1 | 0 | 2 |", report)
        self.assertIn("| ACK | 1 | 1 | 1 | 1 | 0 |", report)
        self.assertIn("- Photos per lesion: 1: 2 lesions, 2: 1 lesions", report)
        self.assertIn("- Lesions per patient: 1: 1 patients, 2: 1 patients", report)
        self.assertIn("- Photos named in the table but not found in raw/: 0", report)
        self.assertIn("- Formats: PNG 4", report)

    def test_every_photo_is_fingerprinted_where_it_lies(self):
        prepare.inventory(self.data)
        images = {r["img_id"]: r for r in self.table("raw_images.csv")}
        self.assertEqual(sorted(images), ["P1_L1_a.png", "P1_L1_b.png", "P1_L2_a.png", "P2_L3_a.png"])
        self.assertEqual(images["P1_L1_b.png"]["file"], "images/imgs_part_2/P1_L1_b.png")
        self.assertEqual((images["P2_L3_a.png"]["width"], images["P2_L3_a.png"]["height"]), ("30", "20"))
        self.assertEqual(images["P1_L1_a.png"]["sha256"], hashlib.sha256(png((200, 0, 0))).hexdigest())
        downloads = self.table("downloads.csv")
        self.assertEqual([(d["file"], d["bytes"]) for d in downloads], [("zr7vgbcyr2-1.zip", "21")])

    def test_each_column_with_its_gaps_values_and_level(self):
        prepare.inventory(self.data)
        columns = {r["column"]: r for r in self.table("columns.csv")}
        self.assertEqual(list(columns), COLUMNS)
        self.assertEqual((columns["itch"]["filled"], columns["itch"]["empty"]), ("3", "1"))  # UNK is a gap
        self.assertEqual(columns["itch"]["values"], "'True'=2; 'False'=1; 'UNK'=1")
        self.assertEqual((columns["smoke"]["filled"], columns["smoke"]["empty"]), ("3", "1"))  # blank is a gap
        self.assertEqual(columns["gender"]["level"], "patient")
        self.assertEqual(columns["region"]["level"], "lesion")
        self.assertEqual(columns["diameter_1"]["level"], "photo")  # two photos of L1 record different sizes
        self.assertEqual(columns["age"]["level"], "patient")

    def test_a_column_of_many_numbers_is_summarised_not_listed(self):
        rows = [[f"P{i}", f"L{i}", "False", str(20 + i), "MALE", "ARM", "1.0", "True", "NEV", f"x{i}.png", "False"]
                for i in range(prepare.LISTED_VALUES + 5)]
        rows[0][3] = ""
        rows[-1][3] = "90"  # skewed, so the median (32.5) differs from the mean
        self.write_metadata(COLUMNS, rows)
        for row in rows:
            (self.raw / row[9]).write_bytes(png((1, 2, 3)))
        prepare.inventory(self.data)
        age = {r["column"]: r for r in self.table("columns.csv")}["age"]
        self.assertEqual(age["values"], "numbers: lowest 21, median 32.5, highest 90; gaps: ''=1")
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("- Groups of photos with identical content: 1", report)  # every photo here is the same picture

    def test_problems_are_reported_not_hidden(self):
        (self.raw / "images" / "imgs_part_1" / "P1_L1_a.png").unlink()  # named but missing
        (self.raw / "images" / "imgs_part_1" / "stray.png").write_bytes(png((5, 5, 5)))  # on disk but not named
        (self.raw / "images" / "imgs_part_1" / "P2_L3_a.png").write_bytes(png((9, 9, 9)))  # the same name in two folders
        (self.raw / "images" / "imgs_part_1" / "P1_L2_a.png").write_bytes(b"not a picture")
        rows = [list(r) for r in ROWS] + [["P3", "L1", "", "40", "MALE", "LEG", "", "", "MEL", "P1_L1_b.png", "True"]]
        self.write_metadata(COLUMNS, rows)
        done = prepare.inventory(self.data)
        self.assertEqual((done["missing"], done["unnamed"]), (1, 1))
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for line in ("- Rows naming the same photo twice: 1 (`P1_L1_b.png`)",
                     "- Photos named in the table but not found in raw/: 1 (`P1_L1_a.png`)",
                     "- Photo files in raw/ not named in the table: 1 (`stray.png`)",
                     "- File names found in more than one folder of raw/ (the first path is used): 1 (`P2_L3_a.png`)",
                     "- Photos that cannot be opened: 1 (`P1_L2_a.png`)",
                     "- Lesions (patient + `lesion_id`) with more than one diagnosis: 0",
                     "- `lesion_id` values used by more than one patient: 1 (`L1`)"):
            self.assertIn(line, report)

    def test_a_pair_of_identical_photos_is_reported(self):
        (self.raw / "images" / "imgs_part_2" / "P1_L1_b.png").write_bytes(png((200, 0, 0)))  # the same picture as P1_L1_a
        prepare.inventory(self.data)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("- Groups of photos with identical content: 1 (`P1_L1_a.png` = `P1_L1_b.png`)", report)

    def test_clinical_fields_filled_by_class_leave_out_ids_label_and_biopsy(self):
        prepare.inventory(self.data)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("The 6 columns other than `img_id`, `patient_id`, `lesion_id`, `diagnostic`, `biopsed`.", report)
        self.assertIn("| BCC | 2 | 5.5 | 5 | 6 |", report)  # smoke, age, gender, region, diameter_1, itch: L1b has itch UNK
        self.assertIn("| ACK | 1 | 5.0 | 5 | 5 |", report)  # smoke blank

    def test_the_sets_of_empty_fields_are_listed_with_their_classes(self):
        # by hand: P1_L1_a has no empty field; P1_L1_b has itch UNK; P1_L2_a has diameter_1 blank; P2_L3_a has smoke blank
        rows = [list(r) for r in ROWS] + [["P5", "L5", "", "30", "MALE", "ARM", "", "True", "SEK", "extra.png", "False"],
                                          ["P5", "L5", "", "30", "MALE", "ARM", "", "True", "SEK", "extra2.png", "False"]]
        self.write_metadata(COLUMNS, rows)
        for name in ("extra.png", "extra2.png"):
            (self.raw / "images" / "imgs_part_1" / name).write_bytes(png((7, 7, 7)))
        prepare.inventory(self.data)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("| Empty fields | Photos | Patients | ACK | BCC | NEV | SEK |", report)
        self.assertIn("| (none) | 1 | 1 | 0 | 1 | 0 | 0 |", report)
        self.assertIn("| `smoke` | 1 | 1 | 1 | 0 | 0 | 0 |", report)
        self.assertIn("| `itch` | 1 | 1 | 0 | 1 | 0 | 0 |", report)
        self.assertIn("| `smoke`, `diameter_1` | 2 | 1 | 0 | 0 | 0 | 2 |", report)  # 2 photos of 1 patient; columns in the order of the table
        patterns = [line for line in report.splitlines() if line.startswith("| (none)") or line.startswith("| `")]
        self.assertTrue(patterns[0].startswith("| `smoke`, `diameter_1` | 2 |"), "the most common set comes first")

    def test_photo_size_by_class_and_the_checks_on_lesion_numbers_and_identical_photos(self):
        (self.raw / "images" / "imgs_part_1" / "P2_L3_a.png").write_bytes(png((9, 9, 9), size=(60, 20)))
        rows = [list(r) for r in ROWS] + [["P3", "L1", "", "40", "MALE", "LEG", "", "", "MEL", "P3_L1_a.png", "True"]]
        self.write_metadata(COLUMNS, rows)
        (self.raw / "images" / "imgs_part_1" / "P3_L1_a.png").write_bytes(png((200, 0, 0)))  # the same picture as P1_L1_a
        prepare.inventory(self.data)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("| Class | Photos | Width: smallest | median | largest | File size (KB): median |", report)
        self.assertIn("| ACK | 1 | 60 | 60 | 60 |", report)
        self.assertIn("| BCC | 2 | 30 | 30 | 30 |", report)
        self.assertIn("- Distinct patient + lesion pairs: 4 (distinct `lesion_id` values: 3)", report)
        self.assertIn("  - `lesion_id` `L1` is under `P1` (BCC), `P3` (MEL)", report)
        self.assertIn("- Groups of photos with identical content: 1 (`P1_L1_a.png` = `P3_L1_a.png`)", report)
        self.assertIn("  - under one patient: 0 of 1; under one lesion number: 1 of 1; with the same diagnosis: 0 of 1", report)

    def test_the_median_file_size_of_a_class_is_not_its_mean(self):
        contents = [noise(100, 1), noise(200, 2), noise(300, 3)]
        rows = [["P%d" % i, "L%d" % i, "False", "50", "MALE", "ARM", "5.0", "True", "NEV", "n%d.png" % i, "False"] for i in range(3)]
        self.write_metadata(COLUMNS, rows)
        for i, content in enumerate(contents):
            (self.raw / "images" / "imgs_part_1" / f"n{i}.png").write_bytes(content)
        for name in ("P1_L1_a.png", "P1_L1_b.png", "P1_L2_a.png", "P2_L3_a.png"):
            (self.raw / "images" / "imgs_part_1" / name).unlink(missing_ok=True)
            (self.raw / "images" / "imgs_part_2" / name).unlink(missing_ok=True)
        prepare.inventory(self.data)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        sizes = [len(c) for c in contents]
        median, mean = f"{statistics.median(sizes) / 1024:.0f}", f"{statistics.mean(sizes) / 1024:.0f}"
        self.assertNotEqual(median, mean)
        self.assertIn(f"| NEV | 3 | 100 | 200 | 300 | {median} |", report)

    def test_a_lesion_with_two_diagnoses_under_one_patient_is_not_listed_as_shared_between_patients(self):
        rows = [list(r) for r in ROWS] + [["P7", "L9", "", "40", "MALE", "LEG", "", "", "NEV", "q1.png", "False"],
                                          ["P7", "L9", "", "40", "MALE", "LEG", "", "", "SEK", "q2.png", "False"]]
        self.write_metadata(COLUMNS, rows)
        for name, colour in (("q1.png", (1, 1, 1)), ("q2.png", (2, 2, 2))):
            (self.raw / "images" / "imgs_part_1" / name).write_bytes(png(colour))
        prepare.inventory(self.data)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("- Lesions (patient + `lesion_id`) with more than one diagnosis: 1 (`P7/L9`)", report)
        self.assertIn("- `lesion_id` values used by more than one patient: 0", report)
        self.assertNotIn("is under", report)

    def test_a_lesion_is_a_patient_and_a_lesion_id_together(self):
        # lesion_id L1 is used by two patients: P1 and P3 have each a lesion L1, and P1 has another lesion L2 elsewhere on the body
        rows = [["P1", "L1", "False", "55", "FEMALE", "FACE", "8.0", "True", "BCC", "P1_L1_a.png", "True"],
                ["P1", "L1", "False", "55", "FEMALE", "FACE", "9.0", "True", "BCC", "P1_L1_b.png", "True"],
                ["P1", "L2", "False", "55", "FEMALE", "BACK", "5.0", "True", "BCC", "P1_L2_a.png", "True"],
                ["P3", "L1", "False", "40", "MALE", "LEG", "5.0", "True", "NEV", "P3_L1_a.png", "False"],
                ["P4", "L1", "False", "61", "MALE", "NOSE", "7.0", "True", "BCC", "P4_L1_a.png", "True"]]
        self.write_metadata(COLUMNS, rows)
        (self.raw / "images" / "imgs_part_1" / "P3_L1_a.png").write_bytes(png((4, 4, 4)))
        (self.raw / "images" / "imgs_part_1" / "P4_L1_a.png").write_bytes(png((6, 6, 6)))
        for part in ("imgs_part_1", "imgs_part_2"):
            (self.raw / "images" / part / "P2_L3_a.png").unlink(missing_ok=True)
        done = prepare.inventory(self.data)
        self.assertEqual((done["lesions"], done["lesion_ids"], done["patients"]), (4, 2, 3))
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("lesions (patient + `lesion_id`): 4; patients: 3", report)
        self.assertIn("| BCC | 4 | 3 | 2 |", report)
        self.assertIn("| NEV | 1 | 1 | 1 |", report)
        self.assertIn("- Photos per lesion: 1: 3 lesions, 2: 1 lesions", report)
        self.assertIn("- Distinct patient + lesion pairs: 4 (distinct `lesion_id` values: 2)", report)
        columns = {r["column"]: r for r in self.table("columns.csv")}
        self.assertEqual(columns["region"]["level"], "lesion")  # fixed within each lesion; lesion_id alone would say 'photo'
        self.assertEqual(columns["gender"]["level"], "patient")
        self.assertEqual(columns["diameter_1"]["level"], "photo")

    def test_a_byte_order_mark_is_ignored_and_the_run_is_repeatable(self):
        self.write_metadata(COLUMNS, ROWS, bom=True)
        prepare.inventory(self.data)
        first = {p.name: p.read_bytes() for p in (self.data / "build").iterdir()} | {"r": (self.data / prepare.REPORT).read_bytes()}
        prepare.inventory(self.data)
        second = {p.name: p.read_bytes() for p in (self.data / "build").iterdir()} | {"r": (self.data / prepare.REPORT).read_bytes()}
        self.assertEqual(first, second)
        self.assertEqual(self.table("columns.csv")[0]["column"], "patient_id")

    def test_wrong_downloads_stop_with_a_clear_message_before_anything_is_written(self):
        self.write_metadata([c for c in COLUMNS if c != "biopsed"], [r[:-1] for r in ROWS])
        with self.assertRaisesRegex(ValueError, "has no column biopsed; its columns are: patient_id, lesion_id"):
            prepare.inventory(self.data)
        self.assertFalse((self.data / "build").exists())
        self.write_metadata(COLUMNS, ROWS)
        (self.raw / "images" / "metadata.csv").write_text("x\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one metadata.csv; found 2"):
            prepare.inventory(self.data)
        shutil.rmtree(self.raw)
        with self.assertRaisesRegex(FileNotFoundError, "unpack the PAD-UFES-20 download there"):
            prepare.inventory(self.data)

    def test_a_download_without_photos_stops(self):
        for path in list(self.raw.rglob("*.png")):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "no photo"):
            prepare.inventory(self.data)

    def test_gaps_are_recognised_in_any_case(self):
        for value in ("", " ", "nan", "NaN", "UNK", "unknown", None):
            self.assertTrue(prepare.is_empty(value), repr(value))
        for value in ("False", "0", "FACE", "8.0"):
            self.assertFalse(prepare.is_empty(value), value)


if __name__ == "__main__":
    unittest.main()
