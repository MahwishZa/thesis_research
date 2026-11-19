"""A small invented PAD-UFES-20 download for the tests of the preparation steps (src/build/clinic_patients): no network, no real data.

Six classes with 10 patients each (one lesion and two photos per patient; every patient's lesion_id is "1", as the real data reuse
lesion numbers), written as the real table is: the early benign records (patients 0 to 5 of ACK, NEV and SEK) have a block of empty
fields, the cancers never do. Three special patients: P900 has two identical photos of one lesion (BCC); P901 has one picture under two
lesions with two diagnoses (NEV and ACK); P902 has a photo that cannot be opened (SEK).
"""

from __future__ import annotations

import csv
import io
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

COLUMNS = ["patient_id", "lesion_id", "smoke", "background_father", "age", "gender", "region", "diameter_1", "itch", "grew",
           "diagnostic", "img_id", "biopsed"]
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
CANCER = ("BCC", "MEL", "SCC")
REGIONS = ("FACE", "ARM", "BACK", "CHEST", "NOSE")


def noise_png(seed: int, side: int = 40) -> bytes:
    """A picture of random pixels: every seed gives a different picture, about half of whose 256 hash bits differ from any other."""
    out = io.BytesIO()
    Image.frombytes("RGB", (side, side), random.Random(seed).randbytes(side * side * 3)).save(out, format="PNG")
    return out.getvalue()


class Cohort(unittest.TestCase):
    """The invented download in a temporary folder: self.rows (the rows of metadata.csv as dicts), self.photos (img_id -> bytes)."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        self.data = self.base / "data" / "clinic_patients"
        self.raw = self.data / "raw"
        (self.raw / "images" / "part_1").mkdir(parents=True)
        self.rows, self.photos = [], {}
        for c, label in enumerate(CLASSES):
            for n in range(10):
                early = label not in CANCER and n < 6  # the block of empty fields
                for k, suffix in enumerate("ab"):
                    patient, img = f"P{c}{n:02d}", f"P{c}{n:02d}_1_{suffix}.png"
                    self.add(patient, "1", label, img, noise_png(c * 1000 + n * 10 + k), {
                        "smoke": "" if early else "FALSE", "background_father": "" if early else "ITALY",
                        "gender": "" if early else ("MALE" if n % 2 else "FEMALE"), "diameter_1": "" if early else str(5 + n),
                        "age": str(30 + 3 * n + c), "region": REGIONS[(c + n) % len(REGIONS)], "itch": "TRUE" if n % 2 else "FALSE",
                        "grew": "UNK" if label in CANCER and n % 3 == 0 else "FALSE", "biopsed": "TRUE" if label in CANCER else "FALSE"})
        same = noise_png(9000)  # P900: the same picture twice, one lesion, one diagnosis
        self.add("P900", "1", "BCC", "P900_1_a.png", same, {"age": "60", "region": "FACE", "biopsed": "TRUE"})
        self.add("P900", "1", "BCC", "P900_1_b.png", same, {"age": "60", "region": "FACE", "biopsed": "TRUE"})
        shared = noise_png(9001)  # P901: one picture, two lesions, two diagnoses
        self.add("P901", "1", "NEV", "P901_1_a.png", shared, {"age": "50", "region": "BACK", "biopsed": "FALSE"})
        self.add("P901", "2", "ACK", "P901_2_a.png", shared, {"age": "50", "region": "BACK", "biopsed": "FALSE"})
        self.add("P902", "1", "SEK", "P902_1_a.png", b"this is not a picture", {"age": "70", "region": "ARM", "biopsed": "FALSE"})
        self.write()
        (self.data / "downloads").mkdir()

    def add(self, patient, lesion, label, img, content, fields=None):
        row = {c: "" for c in COLUMNS} | {"patient_id": patient, "lesion_id": lesion, "diagnostic": label, "img_id": img,
                                           "gender": "FEMALE", "smoke": "FALSE", "background_father": "ITALY", "diameter_1": "8",
                                           "itch": "FALSE", "grew": "FALSE", "biopsed": "FALSE"} | (fields or {})
        self.rows.append(row)
        self.photos[img] = content

    def write(self):
        """Writes metadata.csv and the photos from self.rows and self.photos."""
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        (self.raw / "metadata.csv").write_text(out.getvalue(), encoding="utf-8")
        for img, content in self.photos.items():
            (self.raw / "images" / "part_1" / img).write_bytes(content)

    def table(self, name):
        path = self.data / name if name == "dataset.csv" else self.data / "build" / name
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def report(self):
        return (self.data / "dataset_report.md").read_text(encoding="utf-8")
