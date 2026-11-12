"""The dataset preparation (src/prepare.py) on a tiny synthetic data/raw holding every kind of problem it must find, with local
stand-ins for the ISIC Archive and Harvard Dataverse (no real data, no network beyond this computer), and the files the steps
share: raw_photos.csv, similar_photo_pairs.csv and dataset_report.md, of which each step rewrites only its own part."""

import contextlib
import csv
import hashlib
import io
import json
import os
import random
import re
import shlex
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import PIL
from PIL import Image, ImageChops, ImageCms, ImageStat, JpegImagePlugin

from src import evaluate, prepare


def _jpeg(colour, mark: int, mode: str = "RGB", orientation: int | None = None) -> bytes:
    image = Image.new(mode, (40, 30), colour)
    image.paste((255,) * len(mode), (0, 0, 2 + mark, 2 + mark))  # every photo differs from every other
    buffer = io.BytesIO()
    exif = Image.Exif()
    if orientation:
        exif[prepare.ORIENTATION] = orientation
    image.save(buffer, "JPEG", quality=95, exif=exif)
    return buffer.getvalue()


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (0, 90, 0)).save(buffer, "PNG")
    return buffer.getvalue()


class RawProject(unittest.TestCase):
    """A folder laid out like the repository, with a data/raw holding every kind of problem; each test runs inside it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        previous = os.getcwd()
        os.chdir(self.base)  # the default paths data/ and data/raw point into this folder
        self.addCleanup(os.chdir, previous)
        self.raw, self.data = self.base / "data" / "raw", self.base / "data"
        one, two, four = _jpeg((120, 40, 30), 1), _jpeg((120, 40, 30), 2), _jpeg((120, 40, 30), 4)
        files = {
            "Train/melanoma/ISIC_0000001.jpg": one,
            "Train/nevus/ISIC_0000001.jpg": one,                  # the same photo under a second class folder
            "Train/melanoma/ISIC_0000002.jpg": two,
            "Train/melanoma/ISIC_0000005.jpg": two,               # the same photo under a second ISIC number
            "Train/melanoma/ISIC_0000004.jpg": four,
            "Test/melanoma/ISIC_0000004.jpg": four,               # the same photo in Train and in Test
            "Train/nevus/ISIC_0000003.jpg": _jpeg((200, 150, 120), 3),
            "Train/nevus/broken_ISIC_0000006.jpg": b"\xff\xd8 not a picture",  # damaged
            "Train/nevus/ISIC_0000007.jpg": _png(),               # a PNG with a JPEG name
            "Train/nevus/no_number.jpg": _jpeg((10, 20, 30, 40), 8, mode="CMYK", orientation=6),
            "Train/nevus/Thumbs.db": b"windows",                 # written by Explorer: ignored
            "Train/readme.txt": b"not in a class folder",
            "dataset.zip": b"the downloaded archive",
        }
        for rel, content in files.items():
            (self.raw / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.raw / rel).write_bytes(content)


class InventoryTests(RawProject):
    def _photos(self):
        return prepare._read_photos(self.data)

    def _rows(self):  # every original file, rebuilt from raw_photos.csv
        return {r["path"]: r for r in prepare._files(self._photos())}

    def _raw_state(self):
        return {p.relative_to(self.raw).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.raw.rglob("*") if p.is_file()}

    def test_every_image_file_is_listed_opened_fingerprinted_and_grouped_into_photos(self):
        done = prepare.inventory()
        self.assertEqual((done["files"], done["unreadable"], done["other_problems"], done["other_files"], done["unique"],
                          done["duplicate_groups"]), (10, 1, 1, 2, 7, 3))
        with (self.data / prepare.PHOTOS).open(encoding="utf-8", newline="") as f:
            self.assertEqual(next(csv.reader(f)), prepare.PHOTO_COLUMNS)
        photos = self._photos()
        no_number = next(p for p in photos if p.startswith("SHA256_"))
        self.assertEqual(list(photos), ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003", "ISIC_0000004", "ISIC_0000006", "ISIC_0000007",
                                        no_number])  # one row per photo, sorted
        one, two, four = photos["ISIC_0000001"], photos["ISIC_0000002"], photos["ISIC_0000004"]
        self.assertEqual((one["files"], one["kaggle_labels"], one["kaggle_test"], one["label"]),
                         ("Train/melanoma/ISIC_0000001.jpg | Train/nevus/ISIC_0000001.jpg", "melanoma | nevus", "no", ""))
        self.assertEqual((two["isic_ids"], two["files"]), ("ISIC_0000002 ISIC_0000005",
                                                           "Train/melanoma/ISIC_0000002.jpg | Train/melanoma/ISIC_0000005.jpg"))
        self.assertEqual((four["files"], four["kaggle_test"]), ("Test/melanoma/ISIC_0000004.jpg | Train/melanoma/ISIC_0000004.jpg", "yes"))
        rows = self._rows()
        self.assertEqual(len(rows), 10)
        first = rows["Train/melanoma/ISIC_0000001.jpg"]
        self.assertEqual((first["kaggle_split"], first["kaggle_label"], first["isic_id"], first["copies"], first["format"], first["mode"],
                          first["width"], first["height"], first["exif_orientation"], first["icc_profile"], first["problem"]),
                         ("Train", "melanoma", "ISIC_0000001", "2", "JPEG", "RGB", "40", "30", "", "no", ""))
        self.assertEqual(first["sha256"], hashlib.sha256((self.raw / first["path"]).read_bytes()).hexdigest())
        self.assertTrue(rows["Train/nevus/broken_ISIC_0000006.jpg"]["problem"].startswith("unreadable"))
        self.assertEqual(rows["Train/nevus/broken_ISIC_0000006.jpg"]["isic_id"], "ISIC_0000006")
        self.assertEqual(rows["Train/nevus/ISIC_0000007.jpg"]["problem"], "the file is PNG but its name ends in .jpg")
        odd = rows["Train/nevus/no_number.jpg"]
        self.assertEqual((odd["isic_id"], odd["mode"], odd["exif_orientation"]), ("", "CMYK", "6"))
        self.assertNotIn("Train/nevus/Thumbs.db", rows)

    def test_the_summary_gives_every_count_and_every_group_of_identical_files(self):
        prepare.inventory()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("| Image files in `<split>/<class>/` folders | 10 |", "| Opened completely without error | 9 |",
                         "| Unreadable (damaged or not an image) | 1 |", "| Readable, but the file type does not match its name | 1 |",
                         "| Other files in `data/raw` (not used) | 2 |", "| Files whose name has no ISIC number | 1 |",
                         "| melanoma | 1 | 4 | 5 | 3 |", "| nevus | 0 | 5 | 5 | 5 |", "| **Total** | 1 | 9 | 10 | 7 |",
                         "Unique photos: 7 (from 10 files)", "Groups of identical files: 3, holding 3 extra copies",
                         "Groups filed under two or more class folders: 1", "Groups with copies in two or more split folders: 1",
                         "Groups whose copies carry different ISIC numbers: 1", "ISIC numbers that name two or more different photos: 0",
                         "Colour mode: RGB: 8, CMYK: 1", "EXIF orientation tag", "6: 1",
                         "`Train/nevus/broken_ISIC_0000006.jpg`: unreadable: cannot identify image file\n",
                         "Smallest image (width x height): 40x30 (Test/melanoma/ISIC_0000004.jpg)",
                         "Train/melanoma/ISIC_0000001.jpg<br>Train/nevus/ISIC_0000001.jpg | melanoma, nevus | Train |",
                         "Test/melanoma/ISIC_0000004.jpg<br>Train/melanoma/ISIC_0000004.jpg | melanoma | Test, Train |",
                         "`dataset.zip`: not an image", "`Train/nevus/broken_ISIC_0000006.jpg`: unreadable", "Not checked in this step"):
            self.assertIn(expected, text)
        self.assertRegex(text, r"Fingerprint of `data/raw` .*: `[0-9a-f]{64}`")

    def test_nothing_in_data_raw_changes_and_a_second_run_writes_identical_files(self):
        before = self._raw_state()
        prepare.inventory()
        first = {name: (self.data / name).read_bytes() for name in (prepare.PHOTOS, prepare.REPORT)}
        prepare.inventory()  # reproducible: the same data gives byte-identical files (no dates, a fixed order)
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)
        self.assertEqual(self._raw_state(), before)

    def test_a_changed_missing_or_new_raw_file_stops_the_step_and_keeps_the_committed_table(self):
        prepare.inventory()
        committed = (self.data / prepare.PHOTOS).read_bytes()
        (self.raw / "Train/nevus/ISIC_0000003.jpg").write_bytes(_jpeg((1, 2, 3), 9))
        with self.assertRaisesRegex(ValueError, "1 changed, 0 missing, 0 new .*changed: Train/nevus/ISIC_0000003.jpg"):
            prepare.inventory()
        (self.raw / "Train/nevus/ISIC_0000003.jpg").unlink()
        (self.raw / "Test/melanoma/ISIC_0000099.jpg").write_bytes(_jpeg((5, 5, 5), 7))
        with self.assertRaisesRegex(ValueError, "0 changed, 1 missing, 1 new"):
            prepare.inventory()
        self.assertEqual((self.data / prepare.PHOTOS).read_bytes(), committed)
        (self.data / prepare.PHOTOS).unlink()  # a deliberate change: the table is written anew
        self.assertEqual(prepare.inventory()["files"], 10)

    def test_one_isic_number_on_two_different_photos_keeps_both_apart(self):
        (self.raw / "Test/nevus").mkdir(parents=True)
        (self.raw / "Test/nevus/ISIC_0000003.jpg").write_bytes(_jpeg((200, 150, 120), 11))  # another picture under the same number
        prepare.inventory()
        photos = self._photos()
        twin = next(p for p in photos if p.startswith("ISIC_0000003_SHA256_"))
        self.assertEqual((photos["ISIC_0000003"]["files"], photos[twin]["files"], photos[twin]["isic_ids"]),
                         ("Test/nevus/ISIC_0000003.jpg", "Train/nevus/ISIC_0000003.jpg", "ISIC_0000003"))
        self.assertEqual(len(self._rows()), 11)
        self.assertIn("ISIC numbers that name two or more different photos: 1", (self.data / prepare.REPORT).read_text(encoding="utf-8"))

    def test_a_missing_or_empty_raw_folder_is_explained(self):
        for rel in [p for p in self.raw.rglob("*") if p.is_file() and p.suffix == ".jpg"]:
            rel.unlink()
        with self.assertRaisesRegex(ValueError, "no image found"):
            prepare.inventory()
        os.chdir(self.base / "data")  # here there is no data/raw at all
        with self.assertRaisesRegex(FileNotFoundError, "put the original dataset there"):
            prepare.inventory()

    def test_the_command_line_prints_the_counts_and_turns_problems_into_messages(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["inventory"])
        for expected in ("10 image files, 1 unreadable, 1 with another problem, 2 other files", "7 unique photos; 3 groups",
                         "Nothing in data/raw was changed"):
            self.assertIn(expected, shown.getvalue())
        (self.raw / "Train/nevus/ISIC_0000003.jpg").unlink()
        with self.assertRaises(SystemExit) as raised:
            prepare.main(["inventory"])
        self.assertTrue(str(raised.exception).startswith("ERROR: data/raw differs"))

    def test_every_command_in_the_usage_documentation_is_accepted(self):
        usage = re.findall(r"^\s+python -m src\.prepare (.+?)$", prepare.__doc__, re.M)
        self.assertGreaterEqual(len(usage), 2)
        for line in usage:
            for form in (re.sub(r"\[[^\]]*\]", "", line), re.sub(r"\[([^\]]*)\]", r"\1", line)):  # without / with the optional [parts]
                with self.subTest(form):
                    prepare.build_parser().parse_args(shlex.split(form))


def isic_record(number: str, diagnosis: tuple, confirmed="histopathology", lesion=None, picture="") -> dict:
    """A record shaped like the ISIC Archive's (API v2), with a legacy diagnosis field and a signed picture link, as real ones have."""
    clinical = {f"diagnosis_{n}": d for n, d in enumerate(diagnosis, start=1)} | {"diagnosis_confirm_type": confirmed,
                                                                                 "benign_malignant": "malignant", "diagnosis": "legacy",
                                                                                 "concomitant_biopsy": confirmed == "histopathology"}
    if lesion:
        clinical["lesion_id"] = lesion
    return {"isic_id": number, "copyright_license": "CC-BY-NC", "attribution": "A test clinic", "public": True,
            "metadata": {"acquisition": {"image_type": "dermoscopic", "pixels_x": 40}, "clinical": clinical | {"age_approx": 55}},
            "files": {"thumbnail_256": {"url": picture, "size": 100}}}


MELANOMA = ("Malignant", "Malignant melanocytic proliferations (Melanoma)", "Melanoma Invasive")
NEVUS = ("Benign", "Benign melanocytic proliferations", "Nevus", "Nevus, Atypical, Dysplastic, or Clark")


def isic_records(pictures: str) -> dict:
    """The stand-in's records; each names its preview picture at ``pictures``/<number>.jpg with a link token that can expire."""
    def record(number, *args, **kwargs):
        return isic_record(number, *args, picture=f"{pictures}/{number}.jpg?token=old", **kwargs)

    return {"ISIC_0000001": record("ISIC_0000001", MELANOMA),
            "ISIC_0000002": record("ISIC_0000002", MELANOMA, lesion="IL_1"),
            "ISIC_0000003": record("ISIC_0000003", NEVUS, confirmed="single image expert consensus", lesion="IL_2"),
            "ISIC_0000004": record("ISIC_0000004", MELANOMA, lesion="IL_2"),  # the same lesion as a Train photo, in Test
            "ISIC_0000005": record("ISIC_0000005", MELANOMA, lesion="IL_1"),
            "ISIC_0000007": record("ISIC_0000007", ("Benign",), confirmed=None)}


class FakeIsic(BaseHTTPRequestHandler):
    """The ISIC Archive: records (a number it does not know, a busy moment, a refusal) and preview pictures (a link that has
    expired until the record is asked for again, and a picture that arrives garbled the first time)."""

    def _send(self, status: int, data: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        server = self.server
        if path.startswith("/pictures/"):
            number = path.split("/")[-1].removesuffix(".jpg")
            server.pictures_asked.append(number)
            if number in server.expired and query == "token=old":
                return self._send(403, b'{"detail": "link expired"}', "application/json")
            if server.pictures_asked.count(number) <= server.garbled.get(number, 0):
                return self._send(200, server.pictures[number][:40], "image/jpeg")  # cut off
            return self._send(200, server.pictures[number], "image/jpeg")
        number = path.rstrip("/").split("/")[-1]
        server.asked.append(number)
        if number in server.busy and server.asked.count(number) <= server.busy[number]:
            status, body = 503, {"detail": "busy"}
        elif number in server.refuse:
            status, body = 403, {"detail": "forbidden"}
        elif number in server.records:
            status, body = 200, server.records[number]
            if number in server.expired and server.asked.count(number) > 1:  # asked again: a fresh link
                body = json.loads(json.dumps(body).replace("token=old", "token=new"))
        else:
            status, body = 404, {"detail": "Not found."}
        self._send(status, json.dumps(body).encode(), "application/json")

    def log_message(self, *args):
        pass


def _preview(content: bytes) -> bytes:
    """A preview picture as ISIC serves one: the same photo, smaller, saved again as JPEG."""
    buffer = io.BytesIO()
    with Image.open(io.BytesIO(content)) as image:
        image.convert("RGB").resize((32, 24), Image.Resampling.LANCZOS).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


class IsicServer(RawProject):
    """The project after the inventory, with a local stand-in for the ISIC Archive serving its records and pictures."""

    def setUp(self):
        super().setUp()
        prepare.inventory()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeIsic)
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.server.records = isic_records(f"{base}/pictures")
        self.server.asked, self.server.busy, self.server.refuse = [], {}, set()
        self.server.pictures_asked, self.server.expired, self.server.garbled = [], set(), {}
        self.server.pictures = {number: _preview((self.raw / path).read_bytes()) for number, path in (
            ("ISIC_0000001", "Train/melanoma/ISIC_0000001.jpg"), ("ISIC_0000002", "Train/melanoma/ISIC_0000002.jpg"),
            ("ISIC_0000004", "Test/melanoma/ISIC_0000004.jpg"), ("ISIC_0000005", "Train/melanoma/ISIC_0000005.jpg"),
            ("ISIC_0000007", "Train/nevus/ISIC_0000007.jpg"))}
        self.server.pictures["ISIC_0000003"] = _preview(_jpeg((10, 160, 220), 30))  # ISIC's picture is another photo
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for patcher in (mock.patch.object(prepare, "ISIC_API", f"{base}/api/v2/images/{{}}/"), mock.patch("src.prepare.time.sleep")):
            self.sleep = patcher.start()
            self.addCleanup(patcher.stop)
        self.cache = self.base / "data" / "downloads" / "isic"


class IsicTests(IsicServer):

    def _photos(self):
        return prepare._read_photos(self.data)

    def test_a_sample_run_downloads_only_a_few_records_and_writes_no_table_yet(self):
        done = prepare.isic(limit=2)
        self.assertEqual((done["have"], done["total"], len(done["new"]), done["written"]), (2, 7, 2, False))
        self.assertEqual(self.server.asked, ["ISIC_0000001", "ISIC_0000002"])
        self.assertFalse(any(p["isic_record"] for p in self._photos().values()))
        self.assertEqual(done["new"][0]["diagnosis_3"], "Melanoma Invasive")
        done = prepare.isic()  # the full run continues: what is already downloaded is never asked for again
        self.assertTrue(done["written"])
        self.assertEqual(sorted(self.server.asked), ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003", "ISIC_0000004", "ISIC_0000005",
                                                     "ISIC_0000006", "ISIC_0000007"])

    def test_each_photo_gets_its_record_by_field_name_and_the_numbers_isic_does_not_know_are_said(self):
        done = prepare.isic()
        self.assertEqual((done["total"], done["found"]), (7, 6))
        photos = self._photos()
        nevus = photos["ISIC_0000003"]
        self.assertEqual((nevus["isic_record"], nevus["isic_diagnosis"], nevus["diagnosis_confirm_type"], nevus["image_type"],
                          nevus["lesion_id"], nevus["patient_id"], nevus["licence"], nevus["attribution"], nevus["benign_malignant"],
                          nevus["other_diagnosis_fields"]),
                         ("found", " > ".join(NEVUS), "single image expert consensus", "dermoscopic", "IL_2", "", "CC-BY-NC",
                          "A test clinic", "malignant", "metadata.clinical.diagnosis=legacy"))  # an unexpected diagnosis field is kept
        self.assertRegex(nevus["isic_retrieved_utc"], r"^\d{4}-\d{2}-\d{2}T")
        def fingerprint(number):
            return hashlib.sha256(json.dumps(self.server.records[number], sort_keys=True).encode()).hexdigest()
        self.assertEqual(nevus["isic_record_sha256"], fingerprint("ISIC_0000003"))
        two = photos["ISIC_0000002"]  # one photo, two ISIC numbers: the first describes it, both are fingerprinted
        self.assertEqual((two["lesion_id"], two["isic_record_sha256"]), ("IL_1", f"{fingerprint('ISIC_0000002')} {fingerprint('ISIC_0000005')}"))
        gone = photos["ISIC_0000006"]
        self.assertEqual((gone["isic_record"], gone["isic_diagnosis"], gone["isic_record_sha256"]), ("not found", "", "-"))
        self.assertEqual(photos["ISIC_0000007"]["diagnosis_confirm_type"], "")  # not recorded stays empty: nothing is guessed
        self.assertEqual((photos["ISIC_0000001"]["concomitant_biopsy"], nevus["concomitant_biopsy"]), ("True", "False"))
        self.assertEqual(next(p for i, p in photos.items() if i.startswith("SHA256_"))["isic_record"], "no ISIC number")
        self.assertEqual(photos["ISIC_0000001"]["files"], "Train/melanoma/ISIC_0000001.jpg | Train/nevus/ISIC_0000001.jpg")  # kept
        self.assertNotIn("age_approx", (self.data / prepare.PHOTOS).read_text(encoding="utf-8"))  # not part of this phase
        cached = json.loads((self.cache / "ISIC_0000003.json").read_text(encoding="utf-8"))
        self.assertEqual(cached["record"], self.server.records["ISIC_0000003"])  # the whole record is kept, unchanged
        self.assertEqual(cached["record"]["metadata"]["clinical"]["age_approx"], 55)

    def test_the_summary_shows_what_isic_records_for_each_kaggle_folder(self):
        prepare.isic()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("ISIC numbers: 7; found in the ISIC Archive: 6; not found: 1 (ISIC_0000006)",
                         "| melanoma | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive | 4 |",
                         "| nevus | (not in the ISIC Archive) | 1 |",
                         "| nevus | Benign > Benign melanocytic proliferations > Nevus | 1 |",
                         "| nevus | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive | 1 |",
                         "| Kaggle folder | histopathology | single image expert consensus | (not recorded) |",
                         "| melanoma | 4 |  |  |", "| nevus | 1 | 1 | 2 |",
                         "## Biopsy taken when the photo was made (concomitant_biopsy)", "| Kaggle folder | False | True | (not recorded) |",
                         "Lesions with two or more ISIC numbers: 2",
                         "lesions whose photos sit in both Kaggle splits (Train and Test): 1",
                         "| IL_2 | ISIC_0000003 (Train; nevus), ISIC_0000004 (Test and Train; melanoma) |", "No label is changed"):
            self.assertIn(expected, text)

    def test_a_busy_service_is_asked_again_and_a_lasting_failure_stops_with_the_records_kept(self):
        self.server.busy = {"ISIC_0000002": 2}  # busy twice, then the record
        prepare.isic(limit=2)
        self.assertEqual(self.server.asked.count("ISIC_0000002"), 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list][1:4], [prepare.ISIC_DELAY, 4, 8])  # waits grow
        self.server.busy = {"ISIC_0000003": 99}
        with self.assertRaisesRegex(ValueError, "after 4 attempts \\(HTTP 503\\).*run the same command again"):
            prepare.isic()
        self.assertTrue((self.cache / "ISIC_0000002.json").exists())  # what was downloaded is kept
        self.assertFalse((self.cache / "ISIC_0000003.json").exists())
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_a_refusal_stops_at_once(self):
        self.server.refuse = {"ISIC_0000001"}
        with self.assertRaisesRegex(ValueError, "refused .*HTTP 403"):
            prepare.isic()
        self.assertEqual(self.server.asked, ["ISIC_0000001"])  # not asked again: retrying cannot help

    def test_the_table_is_reproducible_from_the_saved_records_without_asking_again(self):
        prepare.isic()
        first = {name: (self.data / name).read_bytes() for name in (prepare.PHOTOS, prepare.REPORT)}
        asked = len(self.server.asked)
        prepare.isic()
        self.assertEqual(len(self.server.asked), asked)
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)

    def test_the_inventory_comes_first(self):
        (self.data / prepare.PHOTOS).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare inventory first"):
            prepare.isic()

    def test_the_command_line_shows_a_sample_and_then_writes_the_table(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["isic", "--limit", "1"])
        for expected in ("ISIC_0000001: found; Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive; "
                         "confirmed by histopathology; image type dermoscopic; lesion -; patient -; licence CC-BY-NC; credit A test clinic",
                         "Field names in the first record: attribution, copyright_license, files.thumbnail_256.size, "
                         "files.thumbnail_256.url, isic_id, metadata.acquisition.image_type, metadata.acquisition.pixels_x, "
                         "metadata.clinical.age_approx, metadata.clinical.benign_malignant, metadata.clinical.concomitant_biopsy, "
                         "metadata.clinical.diagnosis,",
                         "ISIC records: 1 of 7 downloaded (1 just now)", "run the same command without --limit"):
            self.assertIn(expected, shown.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["isic"])
        self.assertIn("6 found in the ISIC Archive, 1 not found", shown.getvalue())
        self.assertNotIn("ISIC_0000002: found", shown.getvalue())  # the full run does not list every record
        with self.assertRaisesRegex(SystemExit, "--limit must be at least 1"):
            prepare.main(["isic", "--limit", "0"])


class IdentityTests(IsicServer):
    def setUp(self):
        super().setUp()
        prepare.isic()
        self.pictures = self.base / "data" / "downloads" / "isic_pictures"
        self.review = self.base / "data" / "downloads" / "eye_check" / "identity"

    def _photos(self):
        return prepare._read_photos(self.data)

    def test_every_photo_is_compared_with_the_picture_isic_serves_for_its_number(self):
        done = prepare.identity()
        self.assertTrue(done["written"])
        self.assertEqual({row["isic_id"]: row["verdict"] for row in done["rows"]},
                         {"ISIC_0000001": "same picture", "ISIC_0000002": "same picture", "ISIC_0000003": "check by eye",
                          "ISIC_0000004": "same picture", "ISIC_0000005": "same picture", "ISIC_0000006": "no ISIC picture",
                          "ISIC_0000007": "same picture"})
        photos = self._photos()
        no_number = next(p for p in photos if p.startswith("SHA256_"))
        self.assertEqual({p: row["identity"] for p, row in photos.items()},  # ISIC_0000002 holds two numbers, both the same picture
                         {"ISIC_0000001": "same picture", "ISIC_0000002": "same picture", "ISIC_0000003": "check by eye",
                          "ISIC_0000004": "same picture", "ISIC_0000006": "no ISIC picture", "ISIC_0000007": "same picture",
                          no_number: "no ISIC number"})
        same = photos["ISIC_0000001"]
        self.assertEqual((same["width"], same["height"], same["isic_picture_width"], same["isic_picture_height"],
                          same["identity_shape_difference_pct"]), ("40", "30", "32", "24", "0.0"))
        self.assertLessEqual(int(same["identity_hash_distance"]), 3)
        self.assertLessEqual(float(same["identity_mean_difference"]), 5)
        other = photos["ISIC_0000003"]
        self.assertGreater(float(other["identity_mean_difference"]), prepare.SAME["mean_difference"])
        self.assertFalse(self.review.exists())  # the eye check is finished: no side-by-side pictures are made any more
        self.assertEqual(sorted(p.name for p in self.pictures.iterdir()),
                         [f"ISIC_000000{n}.jpg" for n in (1, 2, 3, 4, 5, 7)])  # ISIC has no record of ISIC_0000006

    def test_the_summary_counts_the_verdicts_and_lists_what_to_look_at(self):
        prepare.identity()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("ISIC numbers: 7; check by eye: 1; no ISIC picture: 1; same picture: 5", "| over 5 to 10 | 0 |", "| over 10 | 1 |",
                         "| ISIC_0000003 | Train/nevus/ISIC_0000003.jpg | check by eye |", "Nothing is changed"):
            self.assertIn(expected, text)

    def test_a_sample_run_compares_a_few_and_the_full_run_continues_without_downloading_again(self):
        done = prepare.identity(limit=2)
        self.assertEqual((done["compared"], done["new"], done["written"]), (2, 2, False))
        self.assertFalse(any(p["identity"] for p in self._photos().values()))
        done = prepare.identity()
        self.assertEqual((done["compared"], done["new"], done["written"]), (7, 4, True))
        self.assertEqual(sorted(self.server.pictures_asked), [f"ISIC_000000{n}" for n in (1, 2, 3, 4, 5, 7)])  # each asked once
        first = {name: (self.data / name).read_bytes() for name in (prepare.PHOTOS, prepare.REPORT)}
        prepare.identity()  # reproducible: the same pictures give byte-identical files, and nothing is downloaded again
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)
        self.assertEqual(len(self.server.pictures_asked), 6)

    def test_an_expired_link_is_renewed_and_the_saved_record_stays_as_downloaded(self):
        self.server.expired = {"ISIC_0000004"}
        saved = (self.cache / "ISIC_0000004.json").read_bytes()
        prepare.identity()
        self.assertEqual(self._photos()["ISIC_0000004"]["identity"], "same picture")
        self.assertEqual(self.server.asked.count("ISIC_0000004"), 2)  # the record was asked for again, for a fresh link
        self.assertEqual((self.cache / "ISIC_0000004.json").read_bytes(), saved)

    def test_a_garbled_picture_is_fetched_again(self):
        self.server.garbled = {"ISIC_0000002": 1}
        prepare.identity()
        self.assertEqual(self.server.pictures_asked.count("ISIC_0000002"), 2)
        self.assertEqual(self._photos()["ISIC_0000002"]["identity"], "same picture")

    def test_a_changed_photo_or_a_missing_record_stops_the_step(self):
        (self.raw / "Train/melanoma/ISIC_0000002.jpg").write_bytes(_jpeg((1, 2, 3), 9))
        with self.assertRaisesRegex(ValueError, "Train/melanoma/ISIC_0000002.jpg differs from .*must stay as downloaded"):
            prepare.identity()
        (self.cache / "ISIC_0000005.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare isic first"):
            prepare.identity()

    def test_the_command_line_shows_a_sample_and_then_writes_the_table(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["identity", "--limit", "1"])
        for expected in ("ISIC_0000001: same picture; photo 40x30, ISIC picture 32x24; shape difference 0.0%",
                         "Compared: 1 of 7 ISIC numbers (1 pictures downloaded just now)", "run the same command without --limit"):
            self.assertIn(expected, shown.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["identity"])
        for expected in ("check by eye: 1; no ISIC picture: 1; same picture: 5", "Commit both files", "Photos to check by eye are listed in data/dataset_report.md"):
            self.assertIn(expected, shown.getvalue())
        with self.assertRaisesRegex(SystemExit, "--limit must be at least 1"):
            prepare.main(["identity", "--limit", "0"])


HAM_CSV = b"""lesion_id,image_id,dx,dx_type,age,sex,localization
HAM_1,ISIC_0000001,mel,histo,55.0,male,back
HAM_2,ISIC_0000002,mel,histo,60.0,female,trunk
HAM_2,ISIC_0000005,mel,histo,60.0,female,trunk
HAM_3,ISIC_0000003,nv,consensus,40.0,male,face
HAM_4,ISIC_0000004,bkl,histo,70.0,male,back
HAM_7,ISIC_0000007,nv,follow_up,30.0,female,lower extremity
HAM_9,ISIC_0099999,df,histo,50.0,male,lower extremity
"""


class FakeDataverse(BaseHTTPRequestHandler):
    """Harvard Dataverse: the description of the HAM10000 dataset, and its files."""

    def do_GET(self):
        self.server.asked.append(self.path)
        if self.path.startswith("/api/datasets/:persistentId/"):
            status, data = 200, json.dumps(self.server.listing).encode()
        elif self.path in self.server.files:
            status, data = 200, self.server.files[self.path]
        else:
            status, data = 404, b'{"status": "ERROR"}'
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def dataverse_listing(files: list[dict]) -> dict:
    return {"status": "OK", "data": {"latestVersion": {"versionNumber": 4, "versionMinorNumber": 0, "files": files}}}


class HamServer(IsicServer):
    """The project after the ISIC step, with a local stand-in for Harvard Dataverse serving HAM10000's metadata."""

    def setUp(self):
        super().setUp()
        prepare.isic()
        self.dataverse = ThreadingHTTPServer(("127.0.0.1", 0), FakeDataverse)
        self.dataverse.asked = []
        self.dataverse.listing = dataverse_listing([
            {"label": "HAM10000_images_part_1.zip", "dataFile": {"id": 1, "filename": "HAM10000_images_part_1.zip"}},
            {"label": "HAM10000_metadata.tab", "dataFile": {"id": 42, "filename": "HAM10000_metadata.tab",
                                                            "originalFileName": "HAM10000_metadata.csv",
                                                            "checksum": {"type": "MD5", "value": hashlib.md5(HAM_CSV).hexdigest()}}}])
        self.dataverse.files = {"/api/access/datafile/42?format=original": HAM_CSV}
        threading.Thread(target=self.dataverse.serve_forever, daemon=True).start()
        self.addCleanup(self.dataverse.server_close)
        self.addCleanup(self.dataverse.shutdown)
        patcher = mock.patch.object(prepare, "DATAVERSE", f"http://127.0.0.1:{self.dataverse.server_port}")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = self.base / "data" / "downloads" / "ham10000"


class Ham10000Tests(HamServer):
    def test_each_photo_in_ham10000_is_compared_on_diagnosis_confirmation_and_lesion(self):
        done = prepare.ham10000()
        self.assertEqual((done["total"], done["in_ham"]), (7, 6))  # ISIC_0000006 is not in HAM10000; ISIC_0099999 is not in this dataset
        rows = {row["isic_id"]: row for row in done["rows"]}
        self.assertEqual({n: (r["diagnosis"], r["confirmation"], r["lesion_group"]) for n, r in rows.items()},
                         {"ISIC_0000001": ("agree", "agree", ""),          # ISIC has no lesion number here: nothing to compare
                          "ISIC_0000002": ("agree", "agree", "agree"),     # both sources group 2 and 5
                          "ISIC_0000003": ("agree", "agree", "differ"),    # ISIC groups 3 with 4, HAM10000 does not
                          "ISIC_0000004": ("differ", "agree", "differ"),   # melanoma in ISIC, a benign keratosis in HAM10000
                          "ISIC_0000005": ("agree", "agree", "agree"),
                          "ISIC_0000007": ("", "", "")})                   # ISIC says only "Benign" and not how it was confirmed
        self.assertEqual((rows["ISIC_0000004"]["isic_diagnosis"], rows["ISIC_0000004"]["ham_dx"]),
                         ("Malignant melanocytic proliferations (Melanoma)", "bkl"))
        self.assertEqual((rows["ISIC_0000003"]["isic_diagnosis"], rows["ISIC_0000003"]["ham_dx_type"]), ("Nevus", "consensus"))
        photos = prepare._read_photos(self.data)
        self.assertEqual({p: row["second_source"] for p, row in photos.items() if not p.startswith("SHA256_")},
                         {"ISIC_0000001": "HAM10000 agrees", "ISIC_0000002": "HAM10000 agrees", "ISIC_0000003": "HAM10000 agrees",
                          "ISIC_0000004": "HAM10000 differs", "ISIC_0000006": "not in HAM10000", "ISIC_0000007": "HAM10000 cannot compare"})
        two = photos["ISIC_0000002"]
        self.assertEqual((two["ham_dx"], two["ham_dx_type"], two["ham_confirmation"], two["ham_lesion_id"], two["ham_lesion_group"]),
                         ("mel", "histo", "agree", "HAM_2", "agree"))
        self.assertEqual((self.folder / "HAM10000_metadata.csv").read_bytes(), HAM_CSV)  # kept unchanged

    def test_the_summary_names_the_source_and_every_difference(self):
        prepare.ham10000()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("Harvard Dataverse doi:10.7910/DVN/DBW86T, version 4.0, file `HAM10000_metadata.csv`", "(they match)",
                         "ISIC numbers in this dataset: 7; also in HAM10000: 6",
                         "Diagnosis group: agree: 4; differ: 1; cannot be compared: 1",
                         "Lesion groups: agree: 2; differ: 2; cannot be compared: 2",
                         "| Malignant melanocytic proliferations (Melanoma) | 1 | 3 |  |",
                         "| ISIC_0000004 | Malignant melanocytic proliferations (Melanoma) | bkl | histopathology | histo | IL_2 | HAM_4 | "
                         "diagnosis, lesion_group |",
                         "Photos with a HAM10000 lesion number but no ISIC lesion number: 2 (ISIC_0000001, ISIC_0000007)",
                         "- `akiec`: actinic keratoses and intraepithelial carcinoma (Bowen's disease)", "No label is changed"):
            self.assertIn(expected, text)

    def test_the_file_is_downloaded_once_and_the_result_is_reproducible(self):
        prepare.ham10000()
        first = {name: (self.data / name).read_bytes() for name in (prepare.PHOTOS, prepare.REPORT)}
        asked = len(self.dataverse.asked)
        self.assertEqual(asked, 2)  # the dataset's description, then the file in its original form
        prepare.ham10000()
        self.assertEqual(len(self.dataverse.asked), asked)
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)

    def test_a_tab_separated_file_without_an_original_is_read_too(self):
        tab = HAM_CSV.decode().replace(",", "\t")
        tab = "\n".join("\t".join(f'"{cell}"' for cell in line.split("\t")) for line in tab.strip().split("\n")) + "\n"
        self.dataverse.listing = dataverse_listing([{"label": "HAM10000_metadata.tab",
                                                     "dataFile": {"id": 7, "filename": "HAM10000_metadata.tab"}}])
        self.dataverse.files = {"/api/access/datafile/7": tab.encode()}
        self.assertEqual(prepare.ham10000()["in_ham"], 6)
        self.assertIn("(they do not match", (self.data / prepare.REPORT).read_text(encoding="utf-8"))  # no checksum published

    def test_problems_are_explained(self):
        self.dataverse.listing = dataverse_listing([{"label": "readme.txt", "dataFile": {"id": 3, "filename": "readme.txt"}}])
        with self.assertRaisesRegex(ValueError, "expected one HAM10000_metadata file"):
            prepare.ham10000()
        (self.folder / "dataset.json").unlink()
        self.dataverse.listing = dataverse_listing([{"label": "HAM10000_metadata.csv", "dataFile": {"id": 5, "filename": "HAM10000_metadata.csv"}}])
        self.dataverse.files = {"/api/access/datafile/5": b"name,size\na,1\n"}
        with self.assertRaisesRegex(ValueError, "is not the HAM10000 metadata file"):
            prepare.ham10000()
        photos = prepare._read_photos(self.data)
        kept = photos["ISIC_0000002"]["isic_record_sha256"]
        photos["ISIC_0000002"]["isic_record_sha256"] = kept.split()[0]  # two ISIC numbers, but one fingerprint
        prepare._write_photos(self.data, photos)
        with self.assertRaisesRegex(FileNotFoundError, "lacks the record fingerprints of ISIC_0000002: run python -m src.prepare isic"):
            prepare.ham10000()
        photos["ISIC_0000002"]["isic_record_sha256"] = kept
        prepare._write_photos(self.data, photos)
        record = self.cache / "ISIC_0000005.json"
        entry = json.loads(record.read_text(encoding="utf-8"))
        entry["record"]["attribution"] = "someone else"  # a downloaded record that is no longer the one raw_photos.csv was made from
        record.write_text(json.dumps(entry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ISIC_0000005.json is not the record .*raw_photos.csv was made from"):
            prepare.ham10000()
        record.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "ISIC_0000005.json not found: run python -m src.prepare isic first"):
            prepare.ham10000()
        prepare._update_photos(self.data, "isic", {})  # as if the isic step had never run
        with self.assertRaisesRegex(FileNotFoundError, "lacks what the isic step adds: run python -m src.prepare isic first"):
            prepare.ham10000()

    def test_the_command_line_prints_the_agreement(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["ham10000"])
        for expected in ("ISIC numbers: 7; also in HAM10000: 6", "Diagnosis group: agree: 4; differ: 1; cannot be compared: 1",
                         "How confirmed: agree: 5; cannot be compared: 1", "No label was changed. Commit both files."):
            self.assertIn(expected, shown.getvalue())


RULE = """level,isic_wording,class,reason
diagnosis_2,Malignant melanocytic proliferations (Melanoma),melanoma,
diagnosis_3,Nevus,nevus,all subtypes alike
diagnosis_1,Benign,,too general: no class
"""


class LabelsTests(HamServer):
    def setUp(self):
        super().setUp()
        prepare.ham10000()
        self.rule = self.data / prepare.NAMING_RULE
        self.rule.write_text(RULE, encoding="utf-8")

    def _photos(self):
        return prepare._read_photos(self.data)

    def test_each_unique_photo_is_labelled_from_its_isic_record_never_from_its_folder(self):
        done = prepare.labels()
        self.assertEqual((done["photos"], done["labelled"]), (7, 4))
        with (self.data / prepare.PHOTOS).open(encoding="utf-8", newline="") as f:
            self.assertEqual(next(csv.reader(f)), prepare.PHOTO_COLUMNS)
        photos = self._photos()
        no_number = next(p for p in photos if p.startswith("SHA256_"))
        self.assertEqual({p: (r["label"], r["status"], r["label_vs_folder"]) for p, r in photos.items()},
                         {"ISIC_0000001": ("melanoma", "labelled", "one of two folders"),  # filed as melanoma and nevus
                          "ISIC_0000002": ("melanoma", "labelled", "same as folder"),
                          "ISIC_0000003": ("nevus", "labelled", "same as folder"),
                          "ISIC_0000004": ("melanoma", "labelled", "same as folder"),
                          "ISIC_0000006": ("", "unreadable file", "no class"),
                          "ISIC_0000007": ("", "too general: no class", "no class"),  # the rule's reason, never a guessed class
                          no_number: ("", "no ISIC number", "no class")})
        two = photos["ISIC_0000002"]  # one photo with two ISIC numbers is one row
        self.assertEqual((two["isic_ids"], two["files"], two["lesion_id"]),
                         ("ISIC_0000002 ISIC_0000005", "Train/melanoma/ISIC_0000002.jpg | Train/melanoma/ISIC_0000005.jpg", "IL_1"))
        first = photos["ISIC_0000001"]
        self.assertEqual((first["kaggle_labels"], first["kaggle_test"], first["diagnosis_confirm_type"], first["second_source"], first["width"],
                          first["licence"], first["attribution"]),
                         ("melanoma | nevus", "no", "histopathology", "HAM10000 agrees", "40", "CC-BY-NC", "A test clinic"))
        self.assertEqual((photos["ISIC_0000004"]["kaggle_test"], photos["ISIC_0000004"]["second_source"]), ("yes", "HAM10000 differs"))
        self.assertEqual(photos["ISIC_0000003"]["diagnosis_confirm_type"], "single image expert consensus")
        self.assertEqual(photos["ISIC_0000007"]["second_source"], "HAM10000 cannot compare")
        self.assertEqual(photos["ISIC_0000003"]["isic_diagnosis"], " > ".join(NEVUS))

    def test_the_summary_shows_the_rule_the_classes_and_every_change(self):
        prepare.labels()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("Photos: 7; labelled: 4; without a class: 3", "Classes: 2 (melanoma, nevus)",
                         "| diagnosis_2 | Malignant melanocytic proliferations (Melanoma) | melanoma |  | 4 |",
                         "| diagnosis_1 | Benign | (no class) | too general: no class | 1 |",
                         "| melanoma | 3 | 1 | 3 | 2 |", "| nevus | 1 | 0 | 0 | 1 |", "| **Total** | 4 | 1 | 3 | 3 |",
                         "| melanoma | 3 |  |  |", "| nevus | 1 | 1 | 3 |",
                         "Filed under melanoma | nevus and labelled **melanoma**: 1",
                         "| ISIC_0000007 | nevus | Benign | (no class) | too general: no class |", "No image is changed"):
            self.assertIn(expected, text)

    def test_the_most_specific_rule_decides(self):
        self.rule.write_text(RULE + "diagnosis_1,Malignant,,a level-1 rule loses to the level-2 melanoma rule\n", encoding="utf-8")
        self.assertEqual(self._run_labels()["ISIC_0000004"]["label"], "melanoma")

    def _run_labels(self):
        prepare.labels()
        return self._photos()

    def test_a_wording_without_a_rule_stops_instead_of_guessing(self):
        self.rule.write_text(RULE.replace("diagnosis_3,Nevus,nevus,all subtypes alike\n", "").replace("diagnosis_1,Benign,,too general: no class\n", ""),
                             encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no rule for 2 ISIC records.*ISIC_0000003"):
            prepare.labels()
        self.assertFalse(any(p["status"] for p in self._photos().values()))  # nothing labelled

    def test_a_rule_file_that_cannot_be_applied_is_refused(self):
        cases = {"is not one of the dataset's class names": RULE + "diagnosis_3,Seborrheic keratosis,seborrheic keratosis,\n",
                 "needs a reason": RULE + "diagnosis_3,Something,,\n",
                 "has a rule already": RULE + "diagnosis_3,Nevus,melanoma,\n",
                 "level must be one of": RULE + "diagnosis_9,Nevus,nevus,\n",
                 "must have the columns": "level,wording,class\n"}
        for message, text in cases.items():
            with self.subTest(message):
                self.rule.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    prepare.labels()

    def test_isic_numbers_of_one_photo_that_name_different_classes_stop_the_step(self):
        record = self.cache / "ISIC_0000005.json"
        entry = json.loads(record.read_text(encoding="utf-8"))
        entry["record"]["metadata"]["clinical"].update(diagnosis_2="Benign melanocytic proliferations", diagnosis_3="Nevus")
        record.write_text(json.dumps(entry), encoding="utf-8")
        prepare.isic()  # the changed record is taken in (from the saved records: nothing is downloaded)
        with self.assertRaisesRegex(ValueError, r"ISIC_0000002', 'ISIC_0000005'\] of one photo .* name different classes"):
            prepare.labels()

    def test_reproducible_and_dependent_on_the_earlier_steps(self):
        prepare.labels()
        first = {name: (self.data / name).read_bytes() for name in (prepare.PHOTOS, prepare.REPORT)}
        prepare.labels()
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)
        prepare._update_photos(self.data, "ham10000", {})  # as if the ham10000 step had never run
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare ham10000 first"):
            prepare.labels()
        self.rule.unlink()
        prepare.ham10000()
        with self.assertRaisesRegex(FileNotFoundError, "the naming rule is part of the repository"):
            prepare.labels()

    def test_the_command_line_prints_the_classes_and_comes_after_the_earlier_steps(self):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.main(["labels"])
        prepare._update_photos(self.data, "labels", {})  # as if the labels step had never run
        with self.assertRaisesRegex(SystemExit, "run python -m src.prepare labels first"):
            prepare.main(["duplicates"])

    def test_the_command_line_prints_the_classes(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["labels"])
        for expected in ("Photos: 7; labelled: 4; without a class: 3", "Per class: melanoma: 3; nevus: 1",
                         "Against the Kaggle folders: no class: 3; one of two folders: 1; same as folder: 3", "No image was changed"):
            self.assertIn(expected, shown.getvalue())


def _lesion(seed: int, background=(200, 150, 140), size=(200, 150)) -> Image.Image:
    """A dermoscopy-like picture: an irregular brown blob with dots on skin, blurred (alike enough to make the comparison work hard)."""
    from PIL import ImageDraw, ImageFilter

    rng = __import__("random").Random(seed)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    cx, cy = size[0] // 2 + rng.randint(-20, 20), size[1] // 2 + rng.randint(-15, 15)
    for _ in range(rng.randint(3, 7)):
        rx, ry, dx, dy = rng.randint(20, 55), rng.randint(15, 45), rng.randint(-20, 20), rng.randint(-15, 15)
        draw.ellipse((cx + dx - rx, cy + dy - ry, cx + dx + rx, cy + dy + ry),
                     fill=(rng.randint(60, 140), rng.randint(30, 90), rng.randint(20, 70)))
    for _ in range(120):
        draw.point((rng.randrange(size[0]), rng.randrange(size[1])), fill=(rng.randint(0, 80),) * 3)
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(1, 2)))


def _saved(image: Image.Image, quality=95) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


class DuplicatesTests(unittest.TestCase):
    """A small raw_photos.csv over realistic synthetic photos: one picture saved twice, the same picture under a second label, two
    photos of one known lesion (the second slightly cropped), and a different lesion that must not be paired with anything."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        previous = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous)
        self.raw, self.data = self.base / "data" / "raw", self.base / "data"
        one, two, other = _lesion(1), _lesion(2), _lesion(3, background=(150, 170, 210))
        cropped = two.crop((6, 4, 194, 146)).resize((200, 150), Image.Resampling.LANCZOS)
        files = {"ISIC_1000001": ("Train/melanoma/ISIC_1000001.jpg", _saved(one), "melanoma", ""),
                 "ISIC_1000002": ("Test/melanoma/ISIC_1000002.jpg", _saved(one.resize((160, 120), Image.Resampling.LANCZOS), 80),
                                  "melanoma", ""),  # the same picture, smaller and re-compressed
                 "ISIC_1000003": ("Train/nevus/ISIC_1000003.jpg", _saved(two), "nevus", "IL_A"),
                 "ISIC_1000004": ("Train/nevus/ISIC_1000004.jpg", _saved(cropped, 90), "nevus", "IL_A"),  # ISIC: the same lesion
                 "ISIC_1000005": ("Train/nevus/ISIC_1000005.jpg", _saved(other), "nevus", "IL_B"),  # a different lesion
                 "ISIC_1000006": ("Train/nevus/ISIC_1000006.jpg", _saved(one, 85), "nevus", ""),  # picture one under another label
                 "ISIC_1000007": ("Train/nevus/broken.jpg", b"not a picture", "", "")}
        rows = []
        for number, (path, content, label, lesion) in files.items():
            (self.raw / path).parent.mkdir(parents=True, exist_ok=True)
            (self.raw / path).write_bytes(content)
            rows.append({c: "" for c in prepare.PHOTO_COLUMNS} | {
                "photo_id": number, "isic_ids": number, "sha256": hashlib.sha256(content).hexdigest(), "files": path, "label": label,
                "status": "labelled" if label else "unreadable file", "lesion_id": lesion})
        self.data.mkdir(parents=True, exist_ok=True)
        with (self.data / prepare.PHOTOS).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=prepare.PHOTO_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _pairs(self):
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            return {(r["photo_a"], r["photo_b"]): r for r in csv.DictReader(f) if r["found_by"] == "picture comparison"}

    def test_one_picture_saved_differently_and_a_second_photo_of_a_known_lesion_are_flagged(self):
        done = prepare.duplicates()
        self.assertEqual((done["photos"], done["pairs"], done["known_same"], done["found_same"]), (6, 15, 1, 1))  # the broken file is skipped
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            self.assertEqual(next(csv.reader(f)), prepare.PAIR_COLUMNS)
        pairs = self._pairs()
        for pair in (("ISIC_1000001", "ISIC_1000002"), ("ISIC_1000001", "ISIC_1000006"), ("ISIC_1000002", "ISIC_1000006")):
            self.assertEqual(pairs[pair]["tier"], "near-identical", pair)
        self.assertIn(pairs[("ISIC_1000003", "ISIC_1000004")]["tier"], ("near-identical", "similar"))
        self.assertEqual(pairs[("ISIC_1000003", "ISIC_1000004")]["isic_lesion"], "same lesion")
        self.assertEqual(pairs[("ISIC_1000001", "ISIC_1000006")][("label_a")], "melanoma")
        self.assertFalse([pair for pair in pairs if "ISIC_1000005" in pair])  # the different lesion is paired with nothing
        self.assertFalse((self.base / "data" / "downloads" / "eye_check").exists())  # no side-by-side pictures are made any more

    def test_the_summary_measures_the_method_and_lists_pairs_with_two_labels(self):
        prepare.duplicates()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("Photos compared: 6; pairs: 15", "Photos with an ISIC lesion number: 3. Pairs of them that show the same lesion: 1; "
                         "pairs that show different lesions: 2.", "Share of same-lesion pairs found: 100.0%",
                         "| Different lesions (should not be flagged) | 0 | 0 | 2 |",
                         "| ISIC_1000001 | melanoma | ISIC_1000006 | nevus | near-identical |",
                         # near-identical pairs make two small groups (picture one three times; the known lesion's two photos) ...
                         "- near-identical: 2 groups holding 5 photos; sizes: 1 of 2, 1 of 3",
                         # ... while the looser tier links two different look-alike lesions into one group: the risk the summary names
                         "- near-identical or similar: 1 groups holding 5 photos", "A flagged pair decides nothing by itself",
                         "No image or label is changed. The eye check's verdicts go into `eye_check_verdicts.csv`."):
            self.assertIn(expected, text)
        self.assertNotIn("ISIC_1000005", text)  # the clearly different lesion is never linked

    def test_the_mean_difference_is_the_one_the_identity_check_computes(self):
        one, other = prepare._signature(_saved(_lesion(1))), prepare._signature(_saved(_lesion(3, background=(150, 170, 210))))
        fast = sum(ImageStat.Stat(ImageChops.difference(*(Image.frombytes("RGB", (32, 32), s["pixels"]) for s in (one, other)))).mean) / 3
        self.assertAlmostEqual(round(fast, 1), prepare._compare(one, other)["mean_difference"])

    def test_reproducible_and_a_changed_photo_stops_the_step(self):
        prepare.duplicates()
        first = {name: (self.data / name).read_bytes() for name in (prepare.PAIRS, prepare.REPORT)}
        prepare.duplicates()
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)
        (self.raw / "Train/nevus/ISIC_1000003.jpg").write_bytes(_saved(_lesion(9)))
        with self.assertRaisesRegex(ValueError, "ISIC_1000003.jpg differs from .*must stay as downloaded"):
            prepare.duplicates()

    def test_linked_pairs_make_one_group(self):
        self.assertEqual(prepare._components([("a", "b"), ("c", "d"), ("b", "c"), ("x", "y")]), [{"a", "b", "c", "d"}, {"x", "y"}])

    def test_the_command_line_prints_the_counts(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["duplicates"])
        for expected in ("Photos compared: 6 (15 pairs)", "Known same-lesion pairs (ISIC lesion numbers) found: 1 of 1",
                         "Commit both files"):
            self.assertIn(expected, shown.getvalue())


def _skin(seed: int, size=(800, 600)) -> Image.Image:
    """A textured skin-like picture with dark spots: enough distinct points for skin-pattern matching."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter

    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size[1] // 8, size[0] // 8, 3))
    image = Image.fromarray(np.uint8(np.clip(128 + 40 * base, 0, 255))).resize(size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)
    for _ in range(60 * size[0] * size[1] // 480000):
        x, y, r = rng.integers(0, size[0]), rng.integers(0, size[1]), rng.integers(3, 25)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in rng.integers(30, 160, 3)))
    return image.filter(ImageFilter.GaussianBlur(1.2))


class LesionsTests(unittest.TestCase):
    """Skin-pattern matching: a known lesion photographed again (turned and zoomed), a melanoma photographed again without a
    lesion number, different lesions that must not be flagged, a copy left out by the eye check, and a photo without a class."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        previous = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous)
        self.raw, self.data = self.base / "data" / "raw", self.base / "data"
        self.review = self.base / "data" / "downloads" / "eye_check" / "lesions"
        one, four = _skin(1), _skin(4, (1600, 1200))
        files = {"ISIC_2000001": (one.resize((600, 450), Image.Resampling.LANCZOS), "basal cell carcinoma", "IL_1"),
                 "ISIC_2000002": (one.rotate(20, Image.Resampling.BICUBIC).crop((100, 75, 700, 525)), "basal cell carcinoma", "IL_1"),
                 "ISIC_2000003": (_skin(2).resize((600, 450), Image.Resampling.LANCZOS), "basal cell carcinoma", "IL_2"),
                 "ISIC_2000004": (four, "melanoma", ""),  # a large photo: compared at 600 pixels like the rest
                 "ISIC_2000005": (four.crop((200, 150, 1400, 1050)), "melanoma", ""),  # the same melanoma, closer: no lesion number
                 "ISIC_2000006": (_skin(6).resize((600, 450), Image.Resampling.LANCZOS), "melanoma", ""),
                 "ISIC_2000007": (four.resize((1024, 768), Image.Resampling.LANCZOS), "melanoma", ""),  # a copy the eye check leaves out
                 "ISIC_2000008": (_skin(8).resize((600, 450), Image.Resampling.LANCZOS), "", "")}  # a collision: no class
        rows = []
        for number, (image, label, lesion) in files.items():
            path = f"Train/{label or 'nevus'}/{number}.jpg"
            content = _saved(image, 92)
            (self.raw / path).parent.mkdir(parents=True, exist_ok=True)
            (self.raw / path).write_bytes(content)
            rows.append({c: "" for c in prepare.PHOTO_COLUMNS} | {
                "photo_id": number, "isic_ids": number, "sha256": hashlib.sha256(content).hexdigest(), "files": path, "label": label,
                "status": "labelled" if label else "collision: two lesions in one photo", "lesion_id": lesion})
        self.data.mkdir(parents=True, exist_ok=True)
        with (self.data / prepare.PHOTOS).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=prepare.PHOTO_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (self.data / prepare.EYE_CHECK).write_text("photo_a,photo_b,verdict,keep,note\nISIC_2000004,ISIC_2000007,same photo,ISIC_2000004,\n",
                                                   encoding="utf-8")

    def _pairs(self):
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            return {(r["photo_a"], r["photo_b"]): r for r in csv.DictReader(f) if r["found_by"] == "skin pattern"}

    def test_second_photos_are_found_and_different_lesions_are_not(self):
        done = prepare.lesions()
        self.assertEqual((done["photos"], done["pairs"], done["known_same"], done["found_same"]), (6, 6, 1, 1))  # 3 per class
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            self.assertEqual(next(csv.reader(f)), prepare.PAIR_COLUMNS)
        pairs = self._pairs()
        self.assertEqual(set(pairs), {("ISIC_2000001", "ISIC_2000002"), ("ISIC_2000004", "ISIC_2000005")})
        known, melanoma = pairs[("ISIC_2000001", "ISIC_2000002")], pairs[("ISIC_2000004", "ISIC_2000005")]
        self.assertEqual((known["isic_lesion"], known["label_a"], known["label_b"], melanoma["isic_lesion"]),
                         ("same lesion", "basal cell carcinoma", "basal cell carcinoma", "unknown"))
        self.assertGreaterEqual(int(known["inliers"]), 50)
        self.assertAlmostEqual(abs(int(known["turn_degrees"])), 20, delta=2)  # the turn and zoom of the second photo are measured
        self.assertAlmostEqual(float(known["zoom"]), 1.33, delta=0.05)
        self.assertAlmostEqual(float(melanoma["zoom"]), 1.33, delta=0.05)
        self.assertFalse(self.review.exists())  # no side-by-side pictures are made any more

    def test_the_summary_measures_the_method_per_threshold_and_class(self):
        prepare.lesions()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("Photos compared: 6; pairs (within each class): 6", "basal cell carcinoma: 3 photos, 3 pairs",
                         "Pairs that ISIC names as one lesion: 1; as different lesions (same class): 2.", "| 10 | 1 of 1 (100.0%) | 0 of 2 | 1 |",
                         "| melanoma | 1 | 1 | 1 | 1 |", "| basal cell carcinoma | 0 | 0 | 0 | 0 | 0 | 0 |", "Copies left out by the eye check",
                         "OpenCV 4.", "No image or label is changed"):
            self.assertIn(expected, text)

    def test_reproducible_and_a_changed_photo_stops_the_step(self):
        prepare.lesions()
        first = {name: (self.data / name).read_bytes() for name in (prepare.PAIRS, prepare.REPORT)}
        prepare.lesions()
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)
        (self.raw / "Train/melanoma/ISIC_2000006.jpg").write_bytes(_saved(_skin(9)))
        with self.assertRaisesRegex(ValueError, "ISIC_2000006.jpg differs from .*must stay as downloaded"):
            prepare.lesions()

    def test_a_trial_run_times_the_search_and_writes_nothing(self):
        done = prepare.lesions(limit=2)
        self.assertEqual((done["photos"], done["pairs"], done["all_pairs"]), (4, 2, 6))
        # the estimate scales the time per photo by all photos (6) and the time per pair by all pairs (6), not everything by pairs
        self.assertLess(done["estimate_seconds"], done["seconds"] / done["pairs"] * done["all_pairs"])
        self.assertFalse((self.data / prepare.PAIRS).exists())
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["lesions", "--limit", "2"])
        for expected in ("basal cell carcinoma: 2 photos, 1 pairs", "The full search compares 6 pairs: about 1 minute on", "Nothing was written"):
            self.assertIn(expected, shown.getvalue())
        with self.assertRaisesRegex(SystemExit, "--limit must be at least 2"):
            prepare.main(["lesions", "--limit", "1"])

    def test_the_command_line_prints_the_counts(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["lesions"])
        for expected in ("Photos compared: 6 (6 pairs within classes)", "Known same-lesion pairs (ISIC lesion numbers) with at least 10 "
                         "inliers: 1 of 1", "Commit both files"):
            self.assertIn(expected, shown.getvalue())

    def test_a_matching_photo_with_another_lesion_number_counts_as_a_false_alarm(self):
        content = _saved(_skin(1).rotate(-10, Image.Resampling.BICUBIC).crop((50, 40, 750, 565)), 92)  # lesion one again ...
        (self.raw / "Train/basal cell carcinoma/ISIC_2000009.jpg").write_bytes(content)
        with (self.data / prepare.PHOTOS).open("a", newline="", encoding="utf-8") as f:  # ... under another lesion number
            csv.DictWriter(f, fieldnames=prepare.PHOTO_COLUMNS, lineterminator="\n").writerow({c: "" for c in prepare.PHOTO_COLUMNS} | {
                "photo_id": "ISIC_2000009", "isic_ids": "ISIC_2000009", "sha256": hashlib.sha256(content).hexdigest(),
                "files": "Train/basal cell carcinoma/ISIC_2000009.jpg", "label": "basal cell carcinoma", "status": "labelled",
                "lesion_id": "IL_3"})
        prepare.lesions()
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("| 10 | 1 of 1 (100.0%) | 2 of 5 | 1 |", text)  # (1, 9) and (2, 9) match although ISIC names two lesions
        self.assertEqual(self._pairs()[("ISIC_2000001", "ISIC_2000009")]["isic_lesion"], "different lesions")
        (self.data / prepare.PHOTOS).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare inventory first"):
            prepare.lesions()

    def test_many_points_paired_with_one_and_an_impossible_zoom_are_no_match(self):
        import cv2
        import numpy as np

        rng, matcher = np.random.default_rng(0), cv2.BFMatcher(cv2.NORM_L2)
        points = np.float32(rng.uniform(0, 100, (30, 2)))
        descriptors = np.float32(rng.uniform(0, 100, (30, 128)))
        # 30 points of a all closest to the one point of b with the same pattern: one pair, not 30 (once this "fitted" with zoom 0)
        one = (np.float32([[50, 50]] + [[0, 0]] * 29), np.vstack([descriptors[:1], np.float32(rng.uniform(500, 600, (29, 128)))]))
        crowd = (points, np.float32(descriptors[:1] + rng.normal(0, 0.5, (30, 128))))
        self.assertEqual(prepare._match(crowd, one, matcher)["good_matches"], 1)
        self.assertEqual(prepare._match(crowd, one, matcher)["inliers"], 0)
        for zoom, expected in ((2.0, 30), (6.0, 0)):  # the same 30 points twice as close is a match; six times as close is not
            result = prepare._match((points, descriptors), (points * zoom + 40, descriptors), matcher)
            self.assertEqual((result["inliers"], result["zoom"]), (expected, zoom), zoom)

    def test_both_searches_share_pairs_csv_and_each_rewrites_only_its_own_pairs(self):
        prepare.lesions()
        prepare.duplicates()  # the copy left out by the eye check is still one picture with ISIC_2000004 for the picture comparison
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        found_by = [r["found_by"] for r in rows]
        self.assertEqual(found_by, sorted(found_by, key=["picture comparison", "skin pattern"].index))  # duplicates' pairs first
        self.assertIn(("ISIC_2000004", "ISIC_2000007", "picture comparison"), [(r["photo_a"], r["photo_b"], r["found_by"]) for r in rows])
        before = [r for r in rows if r["found_by"] == "skin pattern"]
        prepare.duplicates()
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            self.assertEqual([r for r in csv.DictReader(f) if r["found_by"] == "skin pattern"], before)
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertLess(text.index("<!-- step: duplicates -->"), text.index("<!-- step: lesions -->"))  # in the order the steps run

    def test_a_photo_without_points_and_a_missing_opencv_are_handled(self):
        import cv2

        sift, matcher = cv2.SIFT_create(nfeatures=prepare.FEATURES), cv2.BFMatcher(cv2.NORM_L2)
        blank = prepare._features(_saved(Image.new("RGB", (600, 450), (180, 120, 110))), sift)
        textured = prepare._features(_saved(_skin(1)), sift)
        self.assertEqual(prepare._match(blank, textured, matcher), {"good_matches": 0, "inliers": 0, "zoom": "", "turn_degrees": ""})
        with mock.patch.dict("sys.modules", {"cv2": None}), self.assertRaisesRegex(FileNotFoundError, "pip install -r requirements.txt"):
            prepare.lesions()


class SplitsTests(unittest.TestCase):
    """A labelled raw_photos.csv: melanoma with every kind of lesion link (ISIC and HAM10000 lesion numbers, an eye-check copy and an
    eye-check second photo) and an original test photo with a lesion-mate; nevus with single photos; and a collision."""

    SMALL = {"test": 4, "val": 2, "min_train": 3, "draws": 50, "patience": 300}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data"
        blank = {c: "" for c in prepare.PHOTO_COLUMNS}
        photos = {}
        for n in range(1, 17):
            photos[f"M{n:02}"] = blank | {"label": "melanoma", "isic_diagnosis": "in situ" if n % 2 else "invasive"}
        for n in range(1, 15):
            photos[f"N{n:02}"] = blank | {"label": "nevus", "isic_diagnosis": "dysplastic" if n <= 3 else "common",
                                          "diagnosis_confirm_type": "histopathology" if n % 2 else ""}
        photos["X01"] = blank | {"status": "collision: two lesions in one photo"}
        for i, p in photos.items():
            p |= {"photo_id": i, "width": "600", "height": "450", "attribution": "A", "kaggle_test": "no", "sha256": i}
            p["status"] = p["status"] or "labelled"
        photos["M01"] |= {"kaggle_test": "yes", "lesion_id": "IL_1"}
        photos["M02"]["lesion_id"] = "IL_1"  # the original test photo's lesion-mate
        for i in ("M03", "M04", "M05"):
            photos[i]["lesion_id"] = "IL_2"
        for i in ("M06", "M07"):
            photos[i]["ham_lesion_id"] = "HAM_1"  # one lesion only by HAM10000's number
        photos["N01"]["kaggle_test"] = "yes"
        prepare._write_photos(self.data, photos)
        (self.data / prepare.EYE_CHECK).write_text("photo_a,photo_b,verdict,keep,note\nM08,M09,same photo,M08,\n"
                                                    "M10,M11,\"same lesion, new photo\",,\n", encoding="utf-8")

    def _photos(self):
        return prepare._read_photos(self.data)

    def test_every_photo_gets_one_split_the_counts_are_exact_and_every_lesion_stays_together(self):
        prepare.splits(self.data, pin=True, **self.SMALL)  # with the original test photos fixed in test (an option, not the default)
        photos = self._photos()
        self.assertEqual(Counter((p["label"], p["split"]) for p in photos.values()),
                         Counter({("melanoma", "test"): 4, ("melanoma", "val"): 2, ("melanoma", "train"): 9, ("nevus", "test"): 4,
                                  ("nevus", "val"): 2, ("nevus", "train"): 8, ("melanoma", "none"): 1, ("", "none"): 1}))
        for group in (("M01", "M02"), ("M03", "M04", "M05"), ("M06", "M07"), ("M10", "M11")):
            self.assertEqual(len({photos[i]["split"] for i in group}), 1, group)
            self.assertEqual(len({photos[i]["lesion_group"] for i in group}), 1, group)
        self.assertEqual([(photos[i]["split"], photos[i]["split_note"]) for i in ("M01", "M02", "N01")],
                         [("test", "original test photo"), ("test", "same lesion as original test photo M01"), ("test", "original test photo")])
        self.assertEqual((photos["M09"]["split"], photos["M09"]["split_note"], photos["M09"]["lesion_group"]),
                         ("none", "same photo as M08 (eye check): left out", photos["M08"]["lesion_group"]))
        self.assertEqual((photos["X01"]["split"], photos["X01"]["split_note"]), ("none", "no class: collision: two lesions in one photo"))

    def test_the_balance_is_reached_by_swaps_and_compared_with_simple_random_splits(self):
        done = prepare.splits(self.data, **self.SMALL)
        self.assertEqual(len(done["random"]), 50)
        self.assertLess(done["imbalance"], done["start"])  # the swaps improve on the random fill they start from ...
        self.assertLess(done["imbalance"], done["random"][len(done["random"]) // 2])  # ... and beat a typical random split
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## Splits: train, validation and test", "seed 2026, fixed in advance", "exactly 4 test and 2 validation photos",
                         "the splits are made anew, and the original test photos are split like every other photo",
                         "| Class | Test | Validation | Train | Photos | Original test photos (test / val / train) |",
                         "| melanoma | 4 | 2 | 9 | 15 | ", "| nevus | 4 | 2 | 8 | 14 | ", "| **Total** | 8 | 4 | 17 | 29 | ",
                         "groups found in two splits: 0", "In no split: 2 (no class: collision: two lesions in one photo: 1; same photo "
                         "as another (eye check): 1)", "50 simple random splits under the same rules: median",
                         "| nevus | subtype | common | 79% |", "No image is changed"):
            self.assertIn(expected, text)

    def test_reproducible_and_the_original_test_photos_are_fixed_only_when_asked(self):
        done = prepare.splits(self.data, **self.SMALL)  # the default: made anew, the original test photos like any other
        first = {name: (self.data / name).read_bytes() for name in (prepare.PHOTOS, prepare.REPORT)}
        photos = self._photos()
        self.assertEqual((photos["M01"]["split_note"], photos["M02"]["split_note"]), ("", ""))
        self.assertEqual(photos["M01"]["split"], photos["M02"]["split"])  # still one lesion
        self.assertEqual(sum(done["counts"][(label, "test")] for label in ("melanoma", "nevus")), 8)
        self.assertNotIn("Fixed in test", (self.data / prepare.REPORT).read_text(encoding="utf-8"))
        prepare.splits(self.data, **self.SMALL)
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)
        prepare.splits(self.data, pin=True, **self.SMALL)
        self.assertIn("- Fixed in test: 2 original test photos and 1 photos of the same lesions",
                      (self.data / prepare.REPORT).read_text(encoding="utf-8"))

    def test_what_cannot_be_split_as_asked_stops_with_a_message(self):
        with self.assertRaisesRegex(ValueError, "melanoma: 15 photos leave 9 for training .* fewer than 12"):
            prepare.splits(self.data, **self.SMALL | {"min_train": 12})
        photos = self._photos()
        for n in range(1, 15, 2):  # every nevus photo in a pair: with one pair fixed in test, three test places cannot be filled
            photos[f"N{n:02}"]["lesion_id"] = photos[f"N{n + 1:02}"]["lesion_id"] = f"IL_N{n}"
        prepare._write_photos(self.data, photos)
        with self.assertRaisesRegex(ValueError, "nevus: the lesion groups cannot fill exactly 3 test"):
            prepare.splits(self.data, **self.SMALL | {"test": 3})
        photos["N13"]["lesion_id"] = "IL_2"  # a nevus photo of a melanoma's lesion
        prepare._write_photos(self.data, photos)
        with self.assertRaisesRegex(ValueError, "show one lesion but have different labels"):
            prepare.splits(self.data, **self.SMALL)
        prepare._update_photos(self.data, "labels", {})
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare labels first"):
            prepare.splits(self.data, **self.SMALL)

    def test_the_command_line_prints_the_counts_and_the_balance(self):
        real = prepare.splits
        with mock.patch.object(prepare, "splits", lambda: real(self.data, **self.SMALL)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["splits"])
        for expected in ("melanoma: test 4, val 2, train 9", "nevus: test 4, val 2, train 8", "simple random splits: median",
                         "Commit both files"):
            self.assertIn(expected, shown.getvalue())


CORNERS = ((255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0))  # top left, top right, bottom left, bottom right


def _cornered(size: tuple[int, int], mark: int) -> Image.Image:
    """A photo with a coloured block in each corner, so a test can see that nothing was cut and which way it was turned, and a
    mark in the middle that makes it differ from every other photo."""
    image = Image.new("RGB", size, (150, 110, 100))
    width, height = size
    image.paste((40, 30, 20), (width // 3, height // 3, width // 3 + 20 + 10 * mark, height // 3 + 20))
    w, h = width // 6, height // 6
    for colour, (x, y) in zip(CORNERS, ((0, 0), (width - w, 0), (0, height - h), (width - w, height - h))):
        image.paste(colour, (x, y, x + w, y + h))
    return image


def _corner_colours(image: Image.Image) -> list[tuple]:
    """The colour near each corner (top left, top right, bottom left, bottom right), as the closest of CORNERS."""
    width, height = image.size
    found = []
    for x, y in ((3, 3), (width - 4, 3), (3, height - 4), (width - 4, height - 4)):
        pixel = image.getpixel((x, y))
        found.append(min(CORNERS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, pixel))))
        assert sum((a - b) ** 2 for a, b in zip(found[-1], pixel)) < 40 ** 2, (pixel, found[-1])
    return found


class ImagesTests(unittest.TestCase):
    """Photos of every shape the dataset has (4:3 and larger, exactly 600 x 450, 4:3 up to rounding, portrait, square, smaller,
    3:2, with a colour profile and an EXIF orientation tag) in train, val and test, plus a copy left out and a photo without a class."""

    PHOTOS = {  # photo: label, split, original size, how it is saved
        "ISIC_0000001": ("melanoma", "test", (800, 600), {}),
        "ISIC_0000002": ("melanoma", "train", (600, 450), {}),
        "ISIC_0000003": ("nevus", "test", (576, 767), {}),  # portrait; 767 x 576 fits as 599.2 x 450
        "ISIC_0000004": ("nevus", "train", (1000, 1000), {}),
        "ISIC_0000005": ("nevus", "val", (500, 400), {}),  # smaller than 600 x 450: never enlarged
        "ISIC_0000006": ("melanoma", "val", (800, 600), {"profile": True, "orientation": 6}),
        "ISIC_0000007": ("melanoma", "none", (800, 600), {}),  # a copy left out by the eye check
        "ISIC_0000008": ("", "none", (800, 600), {}),  # no class
        "ISIC_0000010": ("melanoma", "train", (900, 600), {}),  # 3:2
    }

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data"
        blank = {c: "" for c in prepare.PHOTO_COLUMNS}
        photos = {}
        for n, (i, (label, split, size, how)) in enumerate(self.PHOTOS.items()):
            exif = Image.Exif()
            if how.get("orientation"):
                exif[prepare.ORIENTATION] = how["orientation"]
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes() if how.get("profile") else None
            buffer = io.BytesIO()
            _cornered(size, n).save(buffer, "JPEG", quality=90, exif=exif, icc_profile=profile)
            files = [f"Train/{label or 'melanoma'}/{i}.jpg"] + (["Test/melanoma/ISIC_0000001.jpg"] if i == "ISIC_0000001" else [])
            for rel in files:
                (self.data / "raw" / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.data / "raw" / rel).write_bytes(buffer.getvalue())
            photos[i] = blank | {"photo_id": i, "label": label, "status": "labelled" if label else "no class", "split": split,
                                 "lesion_group": "ISIC_0000001" if i == "ISIC_0000006" else i, "files": " | ".join(files),
                                 "kaggle_test": "yes" if i == "ISIC_0000001" else "no", "sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
        prepare._write_photos(self.data, photos)

    def _table(self) -> dict[str, dict]:
        with (self.data / prepare.IMAGE_TABLE).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, prepare.IMAGE_COLUMNS)
            return {row["photo_id"]: row for row in reader}

    def _files(self) -> dict[str, bytes]:
        root = self.data / prepare.IMAGE_FOLDER
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    def test_every_photo_is_fitted_whole_within_600_x_450_and_saved_the_same_way(self):
        prepare.images(self.data)
        table = self._table()
        self.assertEqual({i: (int(r["width"]), int(r["height"])) for i, r in table.items()},
                         {"ISIC_0000001": (600, 450), "ISIC_0000002": (600, 450), "ISIC_0000003": (600, 450), "ISIC_0000004": (450, 450),
                          "ISIC_0000005": (500, 400), "ISIC_0000006": (600, 450), "ISIC_0000010": (600, 400)})
        self.assertEqual({i: r["changes"] for i, r in table.items() if i != "ISIC_0000006"}, {
            "ISIC_0000001": "shrunk from 800x600", "ISIC_0000002": "saved again only",
            "ISIC_0000003": "turned a quarter turn to the left (portrait to landscape); shrunk from 767x576 (stretched by less than "
                            "one pixel to make it exactly 600x450)",
            "ISIC_0000004": "shrunk from 1000x1000", "ISIC_0000005": "saved again only", "ISIC_0000010": "shrunk from 900x600"})
        self.assertRegex(table["ISIC_0000006"]["changes"], r"^turned upright as its EXIF orientation tag says; colours converted from "
                                                           r"its embedded colour profile \(sRGB.*\) to sRGB; turned a quarter turn to "
                                                           r"the left \(portrait to landscape\); shrunk from 800x600$")
        tables = set()
        for path, content in self._files().items():
            with Image.open(io.BytesIO(content)) as image:
                self.assertEqual((image.format, image.mode), ("JPEG", "RGB"), path)
                self.assertNotIn("icc_profile", image.info, path)  # no metadata of any kind
                self.assertNotIn("exif", image.info, path)
                self.assertEqual(JpegImagePlugin.get_sampling(image), 0, path)  # full colour resolution (4:4:4)
                tables.add(repr(image.quantization))
                corners = _corner_colours(image.convert("RGB"))
            self.assertEqual(sorted(corners), sorted(CORNERS), path)  # every corner is still there: nothing was cut
            if "ISIC_0000003" in path:  # a quarter turn to the left: the top right corner is now the top left one
                self.assertEqual(corners, [CORNERS[1], CORNERS[3], CORNERS[0], CORNERS[2]])
            elif "ISIC_0000006" not in path:  # the others are not turned (6 is turned upright by its tag, then to landscape)
                self.assertEqual(corners, list(CORNERS), path)
        self.assertEqual(len(tables), 1)  # one JPEG setting for every image
        with Image.open(self.data / "raw" / "Train/melanoma/ISIC_0000001.jpg") as original, \
                Image.open(self.data / prepare.IMAGE_FOLDER / "test/melanoma/ISIC_0000001.jpg") as prepared:
            expected = original.resize((600, 450), Image.Resampling.LANCZOS)
            self.assertLess(max(ImageStat.Stat(ImageChops.difference(expected, prepared.convert("RGB"))).mean), 2)
        grey = io.BytesIO()
        Image.new("L", (40, 30), 128).save(grey, "JPEG")
        content, changes = prepare._prepare_image(grey.getvalue())  # no photo in the dataset needs this, but one could
        self.assertEqual(changes, ["converted from L to RGB"])
        with Image.open(io.BytesIO(content)) as image:
            self.assertEqual(image.mode, "RGB")

    def test_images_csv_is_what_the_evaluation_reads_and_photos_in_no_split_get_no_image(self):
        self.assertEqual((prepare.DATA / prepare.IMAGE_FOLDER, prepare.DATA / prepare.IMAGE_TABLE), (evaluate.IMAGES, evaluate.TABLE))
        prepare.images(self.data)
        items = evaluate.load_dataset(self.data / prepare.IMAGE_TABLE, self.data / prepare.IMAGE_FOLDER)
        self.assertEqual(sorted((item.split, item.label, item.path, item.lesion_group) for item in items), [
            ("test", "melanoma", "test/melanoma/ISIC_0000001.jpg", "ISIC_0000001"),
            ("test", "nevus", "test/nevus/ISIC_0000003.jpg", "ISIC_0000003"),
            ("train", "melanoma", "train/melanoma/ISIC_0000002.jpg", "ISIC_0000002"),
            ("train", "melanoma", "train/melanoma/ISIC_0000010.jpg", "ISIC_0000010"),
            ("train", "nevus", "train/nevus/ISIC_0000004.jpg", "ISIC_0000004"),
            ("val", "melanoma", "val/melanoma/ISIC_0000006.jpg", "ISIC_0000001"),
            ("val", "nevus", "val/nevus/ISIC_0000005.jpg", "ISIC_0000005")])
        self.assertEqual(sorted(self._files()), sorted(item.path for item in items))
        self.assertEqual({i: r["kaggle_test"] for i, r in self._table().items() if r["kaggle_test"] == "yes"}, {"ISIC_0000001": "yes"})
        self.assertEqual(sorted(item.sha256 for item in items), sorted(hashlib.sha256(c).hexdigest() for c in self._files().values()))

    def test_running_again_changes_nothing_and_a_photo_that_leaves_its_split_leaves_no_image_behind(self):
        prepare.images(self.data)
        first = self._files(), *((self.data / name).read_bytes() for name in (prepare.IMAGE_TABLE, prepare.REPORT))
        done = prepare.images(self.data)
        self.assertEqual((done["written"], done["removed"]), (0, 0))
        self.assertEqual((self._files(), *((self.data / name).read_bytes() for name in (prepare.IMAGE_TABLE, prepare.REPORT))), first)
        photos = prepare._read_photos(self.data)
        photos["ISIC_0000005"]["split"] = "train"  # the splits made again: it moves ...
        photos["ISIC_0000010"]["split"] = "none"  # ... and this one leaves every split
        prepare._write_photos(self.data, photos)
        done = prepare.images(self.data)
        self.assertEqual((done["written"], done["removed"]), (1, 2))
        self.assertNotIn("val/nevus/ISIC_0000005.jpg", self._files())
        self.assertNotIn("train/melanoma/ISIC_0000010.jpg", self._files())
        self.assertEqual(self._files()["train/nevus/ISIC_0000005.jpg"], first[0]["val/nevus/ISIC_0000005.jpg"])
        self.assertFalse((self.data / prepare.IMAGE_FOLDER / "val" / "nevus").exists())  # an emptied folder goes too
        (self.data / prepare.IMAGE_FOLDER / "test" / "notes.txt").write_text("not an image")
        with self.assertRaisesRegex(ValueError, "1 not in the table .not in the table: test/notes.txt"):
            prepare.images(self.data)  # anything else in data/dataset stops the step, as it would stop a run

    def test_a_changed_or_missing_original_stops_the_step(self):
        source = self.data / "raw" / "Train/melanoma/ISIC_0000002.jpg"
        original = source.read_bytes()
        source.write_bytes(original + b"\0")
        with self.assertRaisesRegex(ValueError, "Train/melanoma/ISIC_0000002.jpg differs from raw_photos.csv: the original dataset must stay"):
            prepare.images(self.data)
        source.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "ISIC_0000002.jpg not found: put the original dataset in data/raw"):
            prepare.images(self.data)
        source.write_bytes(original)
        prepare._update_photos(self.data, "splits", {})
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare splits first"):
            prepare.images(self.data)

    def test_the_report_says_how_every_image_was_made_and_gives_the_fingerprints(self):
        prepare.images(self.data)
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        items = evaluate.read_table(self.data / prepare.IMAGE_TABLE)
        for expected in ("## Images: the prepared dataset", "Fitted whole within 600 x 450 pixels", "Nothing is cut and nothing is added",
                         "quality 95 with full colour resolution (4:4:4 chroma), without metadata", f"Made with Pillow {PIL.__version__}",
                         "| melanoma | 1 | 1 | 2 | 4 |", "| nevus | 1 | 1 | 1 | 3 |", "| **Total** | 2 | 2 | 3 | 7 |",
                         "- Exactly 600 x 450: 4 of 7 images (57.1%)", "- Another size: 3.",
                         "| melanoma | 4 | 3 | 1 | 600x400 (1) |", "| nevus | 3 | 1 | 2 | 450x450 (1), 500x400 (1) |",
                         "- Shrunk: 5 (of which made exactly 600 x 450 by a stretch of less than one pixel: 1)",
                         "(saved again only, never enlarged): 2", "- Turned upright by an EXIF orientation tag: 1",
                         "- Turned from portrait to landscape: 2 (ISIC_0000003, ISIC_0000006)",
                         "- Colours converted from an embedded colour profile: 1 (ISIC_0000006: sRGB",
                         "- Converted to RGB from another colour mode: 0", "marks the 1 photos from the Kaggle dataset's Test folder: "
                         "test 1, val 0, train 0", *(f"- {part}: `{evaluate.items_hash([i for i in items if i.split == part])}`"
                                                     for part in ("test", "val", "train")),
                         "- Photos in no split (2) get no image."):
            self.assertIn(expected, text)

    def test_the_command_line_prints_the_counts_and_what_to_commit(self):
        real = prepare.images
        with mock.patch.object(prepare, "images", lambda show=None: real(self.data, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["images"])
        for expected in ("7 of 7 images made", "Images: 7 (test 2, val 2, train 3); exactly 600x450: 4; another size: 3",
                         "(7 images new or changed, 0 removed)", "Nothing in data/raw was changed", "the images stay on this computer"):
            self.assertIn(expected, shown.getvalue())


class ShortcutsTests(unittest.TestCase):
    """Prepared images of two classes from two sources that differ in one planted clue only: every melanoma photo has dark corners
    (as from a dermoscope's round field), no nevus photo has; the skin texture, size and colour are alike."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data"
        blank = {c: "" for c in prepare.PHOTO_COLUMNS}
        photos = {}
        for label, source, dark in (("melanoma", "Anonymous", True), ("nevus", "MILK study team", False)):
            for n in range(20):
                i = f"ISIC_{len(photos):07}"
                rng = random.Random(i)
                skin = Image.frombytes("L", (120, 90), rng.randbytes(120 * 90)).point(lambda v: 120 + v // 8)
                image = Image.merge("RGB", (skin, skin.point(lambda v: v - 20), skin.point(lambda v: v - 30)))
                if dark:
                    for box in ((0, 0, 14, 14), (106, 0, 120, 14), (0, 76, 14, 90), (106, 76, 120, 90)):
                        image.paste((5, 5, 5), box)
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=95)
                rel = f"Train/{label}/{i}.jpg"
                (self.data / "raw" / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.data / "raw" / rel).write_bytes(buffer.getvalue())
                split = "test" if n < 4 else "val" if n < 8 else "train"
                photos[i] = blank | {"photo_id": i, "label": label, "status": "labelled", "split": split, "lesion_group": i,
                                     "files": rel, "kaggle_test": "no", "attribution": source,
                                     "sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
        prepare._write_photos(self.data, photos)
        prepare.images(self.data)

    def test_a_planted_clue_is_found_and_a_property_that_does_not_differ_tells_nothing(self):
        done = prepare.shortcuts(self.data)
        results = done["results"]
        self.assertEqual(list(results), ["image size", "file size", "dark corners", "edge colour", "sharpness", "all five together"])
        for target in ("label", "source"):  # here every class has its own source, so both are revealed alike
            self.assertEqual(results["dark corners"][target]["balanced_accuracy"], 1.0)
            self.assertEqual(results["image size"][target]["balanced_accuracy"], 0.5)  # every image is 120 x 90: chance
            self.assertTrue(all(result[target]["k"] in (1, 5, 15) for result in results.values()))  # never more than the 24 in train
        self.assertEqual((done["labels"], done["sources"]), (["melanoma", "nevus"], ["Anonymous", "MILK study team"]))

    def test_the_report_shows_the_sources_the_properties_and_what_they_reveal_and_is_the_same_every_time(self):
        prepare.shortcuts(self.data)
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## Shortcut check: what the images reveal besides the lesion", "it changes nothing",
                         "| Class | Images | Anonymous | MILK study team |", "| melanoma | 20 | 20 (100%) | 0 (0%) |",
                         "| nevus | 20 | 0 (0%) | 20 (100%) |",
                         "| Class | Not 600 x 450 | File size (kB) | Dark corners (0–255) | Edge brightness (0–255) | Sharpness |",
                         "Chance is 50.0% for the 2 classes and 50% for the 2 sources",
                         "| dark corners | 100.0% (100.0–100.0%) | ", "| image size | 50.0% (50.0–50.0%) | 1 | 50.0% (50.0–50.0%) | 1 |",
                         "| melanoma | 100% |", "No image, label or split is changed"):
            self.assertIn(expected, text)
        self.assertRegex(text, r"\| melanoma \| 20 \(100%\) \| \d+ \| [0-9] \| \d+ \| [0-9.]+ \|")  # dark corners near 0
        self.assertRegex(text, r"\| nevus \| 20 \(100%\) \| \d+ \| 1[0-9][0-9] \| 1[0-9][0-9] \| [0-9.]+ \|")
        prepare.shortcuts(self.data)
        self.assertEqual((self.data / prepare.REPORT).read_text(encoding="utf-8"), text)

    def test_each_training_class_counts_equally_however_many_images_it_has(self):
        photos = prepare._read_photos(self.data)
        for i in [i for i, p in photos.items() if p["label"] == "melanoma" and p["split"] == "train"][:6]:
            photos[i]["split"] = "none"  # melanoma now has 6 training images, nevus 12
        prepare._write_photos(self.data, photos)
        prepare.images(self.data)
        recall = prepare.shortcuts(self.data)["results"]["image size"]["label"]["recall"]
        self.assertEqual(recall, {"melanoma": 1.0, "nevus": 0.0})  # a property that tells nothing does not pull towards the bigger class

    def test_the_measures_and_the_vote(self):
        grey = Image.new("RGB", (60, 40), (100, 100, 100))
        self.assertEqual(prepare._measure(_saved(grey, quality=100)), {"corners": 100, "edge_r": 100, "edge_g": 100, "edge_b": 100,
                                                                         "sharpness": 0})
        board = Image.new("RGB", (60, 40))
        board.putdata([(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(40) for x in range(60)])
        lossless = io.BytesIO()
        board.save(lossless, "PNG")
        self.assertEqual(prepare._measure(lossless.getvalue())["sharpness"], 255)
        weight = {"a": 1.0, "b": 1.0}
        ranked = [(0.0, 0), (0.0, 1), (0.0, 2), (1.0, 3)]
        self.assertEqual(prepare._vote(ranked, ["a", "b", "b", "a"], weight, 1), "b")  # all three at the nearest distance vote
        self.assertEqual(prepare._vote(ranked, ["a", "b", "b", "a"], {"a": 3.0, "b": 1.0}, 1), "a")  # each class counts equally
        self.assertEqual(prepare._vote([(0.0, 0), (0.5, 1)], ["b", "a"], weight, 2), "b")  # a tied vote: the nearest row's class
        self.assertEqual(prepare._vote([(0.0, 0), (0.0, 1)], ["b", "a"], weight, 1), "a")  # still tied: the first class name

    def test_without_the_prepared_images_or_with_a_changed_one_the_step_stops(self):
        image = next((self.data / prepare.IMAGE_FOLDER / "test").rglob("*.jpg"))
        image.write_bytes(image.read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "1 changed"):
            prepare.shortcuts(self.data)
        photos = prepare._read_photos(self.data)
        for p in photos.values():
            if p["split"] == "val":
                p["split"] = "none"
        prepare._write_photos(self.data, photos)
        prepare.images(self.data)
        with self.assertRaisesRegex(ValueError, "needs images in train, val and test"):
            prepare.shortcuts(self.data)
        (self.data / prepare.IMAGE_TABLE).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.prepare images first"):
            prepare.shortcuts(self.data)

    def test_the_command_line_prints_what_each_property_reveals(self):
        real = prepare.shortcuts
        with mock.patch.object(prepare, "shortcuts", lambda: real(self.data)), contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["shortcuts"])
        for expected in ("chance: classes 50.0%, sources 50%", "  dark corners: classes 100.0%, sources 100.0%",
                         "  image size: classes 50.0%, sources 50.0%", "Nothing was changed. Commit it."):
            self.assertIn(expected, shown.getvalue())


class SharedFilesTests(unittest.TestCase):
    """raw_photos.csv, similar_photo_pairs.csv and dataset_report.md are shared by the steps: each step rewrites only its own part of them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data"

    def test_every_column_of_photos_csv_belongs_to_exactly_one_step(self):
        owned = [column for columns in prepare.STEP_COLUMNS.values() for column in columns]
        self.assertEqual(sorted(owned), sorted(prepare.PHOTO_COLUMNS))
        self.assertEqual(len(owned), len(set(owned)))
        self.assertTrue(all(prepare.DONE[step] in prepare.STEP_COLUMNS[step] for step in prepare.DONE))
        self.assertEqual(list(prepare.FOUND_BY), [step for step in prepare.STEPS if step in prepare.FOUND_BY])

    def test_a_step_rewrites_only_its_own_columns(self):
        blank = {c: "" for c in prepare.PHOTO_COLUMNS}
        prepare._write_photos(self.data, {"B": blank | {"photo_id": "B", "label": "nevus"}, "A": blank | {"photo_id": "A", "label": "melanoma"}})
        photos = prepare._update_photos(self.data, "isic", {"A": {"lesion_id": "IL_1", "label": "not the isic step's column"}})
        self.assertEqual([(i, p["lesion_id"], p["label"]) for i, p in photos.items()], [("A", "IL_1", "melanoma"), ("B", "", "nevus")])
        self.assertEqual(list(prepare._read_photos(self.data)), ["A", "B"])  # sorted by photo_id
        with self.assertRaisesRegex(FileNotFoundError, "lacks what the labels step adds: run python -m src.prepare labels first"):
            prepare._read_photos(self.data, "labels")  # a label without the labels step's status

    def test_a_step_rewrites_only_its_own_section_and_the_sections_keep_the_step_order(self):
        prepare._write_section(self.data, "labels", ["# Labels", "", "first text", ""])
        prepare._write_section(self.data, "inventory", ["# Inventory", "", "## Files", "files text"])
        prepare._write_section(self.data, "labels", ["# Labels", "", "second text", "", ""])
        expected = ("\n".join(prepare.REPORT_HEAD) + "\n<!-- step: inventory -->\n## Inventory\n\n### Files\nfiles text\n\n"
                    "<!-- step: labels -->\n## Labels\n\nsecond text\n")
        self.assertEqual((self.data / prepare.REPORT).read_text(encoding="utf-8"), expected)
        prepare._write_section(self.data, "inventory", ["# Inventory", "", "## Files", "files text"])  # the same again: no change
        self.assertEqual((self.data / prepare.REPORT).read_text(encoding="utf-8"), expected)

    def test_a_step_rewrites_only_the_pairs_it_found(self):
        prepare._update_pairs(self.data, "lesions", [{"photo_a": "A", "photo_b": "B", "inliers": 30}])
        prepare._update_pairs(self.data, "duplicates", [{"photo_a": "C", "photo_b": "D", "tier": "similar"}])
        prepare._update_pairs(self.data, "lesions", [{"photo_a": "A", "photo_b": "E", "inliers": 12}])
        with (self.data / prepare.PAIRS).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["found_by"], r["photo_a"], r["photo_b"], r["tier"], r["inliers"]) for r in rows],
                         [("picture comparison", "C", "D", "similar", ""), ("skin pattern", "A", "E", "", "12")])


class EyeCheckTests(unittest.TestCase):
    """eye_check_verdicts.csv is written by people: it must name real photos, use the known verdicts and say which copy of a photo stays."""

    PHOTOS = {i: {"photo_id": i, "label": label} for i, label in (("A", "melanoma"), ("B", "melanoma"), ("C", "melanoma"), ("D", "nevus"),
                                                                    ("E", "nevus"), ("X", ""))}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / prepare.EYE_CHECK

    def _read(self, *rows, columns=None):
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(columns or prepare.EYE_CHECK_COLUMNS)
            writer.writerows(rows)
        return prepare._read_eye_check(self.path, self.PHOTOS)

    def test_copies_are_left_out_and_second_photos_of_one_lesion_kept_together(self):
        done = self._read(("A", "B", "same photo", "A", "colour varies"), ("C", "A", "same photo", "A", ""),
                          ("D", "E", "same lesion, new photo", "", ""), ("B", "D", "different lesions", "", ""))
        self.assertEqual(done, {"left_out": {"B": "A", "C": "A"}, "same_lesion": [("D", "E")]})
        self.assertEqual(self._read(), {"left_out": {}, "same_lesion": []})  # no verdicts yet: nothing changes

    def test_anything_that_would_need_a_guess_stops_with_the_line(self):
        for rows, message in (
                ([("A", "Q", "same photo", "A", "")], "line 2: photo_a and photo_b must be two different photo_id values"),
                ([("A", "A", "same photo", "A", "")], "two different photo_id values"),
                ([("A", "B", "duplicate", "A", "")], "verdict must be one of"),
                ([("A", "B", "same photo", "", "")], "a same photo needs keep: A or B"),
                ([("A", "B", "same photo", "C", "")], "a same photo needs keep"),
                ([("A", "D", "same photo", "A", "")], "one photo cannot have two labels .A: melanoma, D: nevus."),
                ([("A", "X", "same photo", "A", "")], "X: no class"),
                ([("D", "E", "same lesion, new photo", "D", "")], "keep is only for a same photo"),
                ([("A", "B", "different lesions", "", ""), ("B", "A", "same photo", "A", "")], "line 3: B and A have a verdict already"),
                ([("A", "B", "same photo", "A", ""), ("B", "C", "same photo", "C", "")], "A, B, C are one photo, so every row about them")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self._read(*rows)
        with self.assertRaisesRegex(ValueError, "must have the columns"):
            self._read(columns=["photo_a", "photo_b", "verdict"])
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "the eye check is part of the repository"):
            prepare._read_eye_check(self.path, self.PHOTOS)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data" / prepare.PHOTOS).is_file(), "raw_photos.csv is not written yet")
    def test_the_committed_eye_check_fits_the_committed_photos(self):
        root = Path(__file__).resolve().parents[1] / "data"
        with (root / prepare.PHOTOS).open(encoding="utf-8", newline="") as f:
            photos = {p["photo_id"]: p for p in csv.DictReader(f)}
        done = prepare._read_eye_check(root / prepare.EYE_CHECK, photos)
        self.assertLessEqual({"ISIC_0001124": "ISIC_0000399", "ISIC_0010302": "ISIC_0000520", "ISIC_0010990": "ISIC_0000521",
                              "ISIC_0030387": "ISIC_0024522"}.items(), done["left_out"].items())  # the first eye check stays as decided
        self.assertTrue(all(photos[kept]["kaggle_test"] == "yes" or photos[gone]["kaggle_test"] == "no" for gone, kept in done["left_out"].items()),
                        "an original test photo is never the copy left out")


if __name__ == "__main__":
    unittest.main()
