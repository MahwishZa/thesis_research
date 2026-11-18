"""The check of the saved pictures (src/multimodal/figures.py, step "check"), with the image model and the reader replaced."""

import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.multimodal import figures


def noise(seed: int, size=(300, 200)) -> bytes:
    rng = random.Random(seed)
    picture = Image.new("RGB", size)
    picture.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size[0] * size[1])])
    out = io.BytesIO()
    picture.save(out, format="PNG")
    return out.getvalue()


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{c: r.get(c, "") for c in columns} for r in rows])


class WordTests(unittest.TestCase):
    def test_names_and_short_forms_in_writing(self):
        names, forms = ["melanoma", "basal cell carcinoma", "nevus"], ["BCC", "SCC"]
        self.assertEqual(figures.diagnosis_words(["A", "B", "1 cm"], names, forms), [])
        self.assertEqual(figures.diagnosis_words(["Melanomas", "in situ"], names, forms), ["melanoma"])
        self.assertEqual(figures.diagnosis_words(["Basal cell", "carcinoma"], names, forms), ["basal cell carcinoma"])
        self.assertEqual(figures.diagnosis_words(["naevi"], names, forms), ["nevus"])  # either spelling, singular or plural
        self.assertEqual(figures.diagnosis_words(["BCCs", "scc"], names, forms), ["BCC"])  # a short form only in capitals
        self.assertEqual(figures.diagnosis_words(["ABCC1"], names, forms), [])  # not inside a longer word

    def test_kappa_by_hand(self):
        self.assertAlmostEqual(figures.kappa([(True, True), (False, False), (True, False), (False, False)]), 0.5)
        self.assertEqual(figures.kappa([(True, True), (False, False)]), 1.0)
        self.assertIsNone(figures.kappa([]))
        self.assertIsNone(figures.kappa([(True, True), (True, True)]))  # no variation: kappa is not defined


class CheckTests(unittest.TestCase):
    # patient_uid, split, class, figure, picture seed, size, model kind, writing
    PICTURES = [("c1", "test", "melanoma", "f1", 1, (300, 200), "clinical photo", ["A", "B"]),
                ("c1", "test", "melanoma", "f2", 1, (300, 200), "clinical photo", []),  # the same picture again in the same case
                ("c2", "test", "nevus", "f1", 2, (100, 80), "dermoscopy", []),  # too small
                ("c3", "test", "nevus", "f1", 3, (300, 200), "dermoscopy", ["Melanomas in situ"]),
                ("c4", "val", "nevus", "f1", 4, (300, 200), "microscopy", []),
                ("c5", "val", "melanoma", "f1", 5, (300, 200), "clinical photo", []),
                ("c6", "train", "melanoma", "f1", 5, (300, 200), "clinical photo", []),  # the same picture as c5's
                ("c7", "train", "nevus", "f1", 7, (300, 200), "dermoscopy", []),  # the same picture as one of the image dataset
                ("c8", "train", "nevus", "f1", 8, (300, 200), "dermoscopy", ["x BCC"])]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.data, self.text, self.image = base / "multimodal", base / "text", base / "image"
        files, captions, self.kinds, self.writing = [], [], {}, {}
        for uid, split, label, fig, seed, size, kind, writing in self.PICTURES:
            content = noise(seed, size)
            path = Path("figures") / "images" / uid / f"{fig}.png"
            (self.data / path).parent.mkdir(parents=True, exist_ok=True)
            (self.data / path).write_bytes(content)
            files.append({"patient_uid": uid, "split": split, "class": label, "pmcid": f"PMC{uid}", "figure": fig, "file": path.as_posix(),
                          "sha256": hashlib.sha256(content).hexdigest(), "width": size[0], "height": size[1], "status": figures.SAVED})
            captions.append({"patient_uid": uid, "figure": fig, "dermoscopy": kind == "dermoscopy", "clinical": kind != "dermoscopy"})
            self.kinds[(self.data / path).as_posix()], self.writing[(self.data / path).as_posix()] = kind, writing
        files.append({**files[0], "figure": "f9", "file": "", "sha256": "", "status": figures.NOT_OPEN})  # not saved: not checked
        write_csv(self.data / figures.FILES_TABLE, figures.FILE_COLUMNS, files)
        (self.data / figures.CAPTIONS).write_text("".join(json.dumps(c) + "\n" for c in captions), encoding="utf-8")
        cases = [{"patient_uid": u, "split": s, "class": c, "counts": "yes"} for u, s, c, *_ in self.PICTURES]
        cases = list({c["patient_uid"]: c for c in cases}.values()) + [{"patient_uid": "c9", "split": "test", "class": "nevus", "counts": "no"}]
        write_csv(self.data / figures.COUNT_TABLE, figures.CASE_COLUMNS, cases)
        write_csv(self.text / "class_terms.csv", ["class", "term", "counts", "source", "note"],
                  [{"class": "melanoma", "term": "melanoma", "counts": "yes"}, {"class": "nevus", "term": "spider nevus", "counts": "no"}])
        (self.image / "dataset" / "train" / "nevus").mkdir(parents=True)
        (self.image / "dataset" / "train" / "nevus" / "a.png").write_bytes(noise(7))
        (self.image / "dataset" / "train" / "nevus" / "b.png").write_bytes(noise(70))
        write_csv(self.image / "dataset.csv", ["split", "label", "path"], [{"split": "train", "label": "nevus", "path": "train/nevus/a.png"},
                                                                            {"split": "train", "label": "nevus", "path": "train/nevus/b.png"}])
        self.calls = 0

    def classify(self, path):
        self.calls += 1
        kind = self.kinds[Path(path).as_posix()]
        return {k: (0.9 if k == kind else 0.025) for k in figures.KINDS}

    def read(self, path):
        return self.writing[Path(path).as_posix()]

    def run_check(self, **limits):
        with mock.patch.multiple(figures, **limits) if limits else mock.patch.object(figures, "MIN_SIDE", figures.MIN_SIDE):
            return figures.check(self.data, self.text, self.image, classify=self.classify, read=self.read)

    def test_every_picture_is_kept_or_gets_its_reasons(self):
        done = self.run_check()
        found = {(r["patient_uid"], r["figure"]): (r["kept"], r["reason"]) for r in done["rows"]}
        self.assertEqual(found, {("c1", "f1"): ("yes", ""), ("c1", "f2"): ("no", "the same picture twice in this case"),
                                 ("c2", "f1"): ("no", "smaller than the least size"),
                                 ("c3", "f1"): ("no", "writing in the picture names a diagnosis"),
                                 ("c4", "f1"): ("no", "the image model sees no photo of the skin"),
                                 ("c5", "f1"): ("no", "the same picture is in another case"),
                                 ("c6", "f1"): ("no", "the same picture is in another case"),
                                 ("c7", "f1"): ("no", "the same picture is in the image dataset"),
                                 ("c8", "f1"): ("no", "writing in the picture names a diagnosis")})
        rows = {(r["patient_uid"], r["figure"]): r for r in done["rows"]}
        self.assertEqual((rows[("c3", "f1")]["diagnosis_words"], rows[("c8", "f1")]["diagnosis_words"]), ("melanoma", "BCC"))
        self.assertEqual((rows[("c1", "f1")]["caption_type"], rows[("c1", "f1")]["model_kind"], rows[("c1", "f1")]["writing"]),
                         ("clinical", "clinical photo", "yes"))
        self.assertEqual(done["decision"]["test_cases"], 1)

    def test_the_tables_and_the_rule_after_the_checks(self):
        self.run_check(GO_TEST_CASES=1, GO_CLASSES=1, GO_PER_CLASS=1)
        with (self.data / figures.CHECKS_TABLE).open(encoding="utf-8", newline="") as f:
            self.assertEqual(list(next(csv.reader(f))), figures.CHECK_COLUMNS)
        report = (self.data / figures.CHECKS_REPORT).read_text(encoding="utf-8")
        for expected in ("**After the checks: 1 test cases keep a picture; 1 classes have at least 1 (melanoma). The rule for building the "
                         "dataset (at least 1 test cases and 1 classes) is met.**", "| melanoma | 1 | 0 | 0 |", "| all | 1 | 0 | 0 |",
                         "Pictures checked: 9; kept: 1.", "| the same picture is in another case | 2 |",
                         "agree on whether it is a dermoscopy image for 8; Cohen's kappa 1.00",
                         "with the name of a diagnosis: 2"):
            self.assertIn(expected, report)

    def test_a_second_run_asks_the_models_nothing(self):
        first = self.run_check()["rows"]
        self.calls = 0
        again = figures.check(self.data, self.text, self.image, classify=lambda p: 1 / 0, read=lambda p: 1 / 0)["rows"]
        self.assertEqual(again, first)

    def test_a_changed_picture_stops_the_check(self):
        (self.data / "figures" / "images" / "c1" / "f1.png").write_bytes(noise(99))
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            self.run_check()

    def test_the_least_size_and_the_kinds_are_fixed(self):
        self.assertEqual(figures.MIN_SIDE, 128)
        self.assertEqual(figures.SKIN_KINDS, ("clinical photo", "dermoscopy"))
        self.assertEqual(list(figures.KINDS), ["clinical photo", "dermoscopy", "microscopy", "scan", "drawing"])


if __name__ == "__main__":
    unittest.main()


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.data, self.text = base / "multimodal", base / "text"
        checks = [{"patient_uid": "c1", "split": "test", "class": "melanoma", "figure": "f10", "file": "figures/images/P1/f10-a.jpg",
                   "sha256": "a" * 64, "kept": "yes"},
                  {"patient_uid": "c1", "split": "test", "class": "melanoma", "figure": "f2", "file": "figures/images/P1/f2-b.jpg",
                   "sha256": "b" * 64, "kept": "yes"},
                  {"patient_uid": "c1", "split": "test", "class": "melanoma", "figure": "f3", "file": "figures/images/P1/f3-c.jpg",
                   "sha256": "c" * 64, "kept": "no"},
                  {"patient_uid": "c2", "split": "train", "class": "nevus", "figure": "F1", "file": "figures/images/P2/F1-d.jpg",
                   "sha256": "d" * 64, "kept": "yes"},
                  {"patient_uid": "c3", "split": "val", "class": "nevus", "figure": "F1", "file": "figures/images/P3/F1-e.jpg",
                   "sha256": "e" * 64, "kept": "no"}]
        write_csv(self.data / figures.CHECKS_TABLE, figures.CHECK_COLUMNS, checks)
        write_csv(self.data / figures.FILES_TABLE, figures.FILE_COLUMNS,
                  [{"patient_uid": c["patient_uid"], "figure": c["figure"], "label": f"Figure {c['figure'][1:]}"} for c in checks])
        self.questions = [{"split": "test", "label": "melanoma", "path": "test/melanoma/c1.txt", "sha256": "1" * 64, "patient_uid": "c1", "pmid": "11"},
                          {"split": "train", "label": "nevus", "path": "train/nevus/c2.txt", "sha256": "2" * 64, "patient_uid": "c2", "pmid": "12"},
                          {"split": "val", "label": "nevus", "path": "val/nevus/c3.txt", "sha256": "3" * 64, "patient_uid": "c3", "pmid": "13"},
                          {"split": "test", "label": "nevus", "path": "test/nevus/c4.txt", "sha256": "4" * 64, "patient_uid": "c4", "pmid": "14"}]
        write_csv(self.text / "dataset.csv", ["split", "label", "path", "sha256", "patient_uid", "pmid"], self.questions)

    def test_one_question_per_case_that_keeps_a_picture_with_its_pictures_in_figure_order(self):
        done = figures.build(self.data, self.text)
        self.assertEqual(done["rows"], [
            {"split": "test", "label": "melanoma", "path": "test/melanoma/c1.txt", "sha256": "1" * 64, "pmid": "11", "patient_uid": "c1",
             "pictures": f"figures/images/P1/f2-b.jpg|{'b' * 64};figures/images/P1/f10-a.jpg|{'a' * 64}", "n_pictures": 2},
            {"split": "train", "label": "nevus", "path": "train/nevus/c2.txt", "sha256": "2" * 64, "pmid": "12", "patient_uid": "c2",
             "pictures": f"figures/images/P2/F1-d.jpg|{'d' * 64}", "n_pictures": 1}])
        report = (self.data / figures.CASES_REPORT).read_text(encoding="utf-8")
        self.assertIn("| all | 1 | 0 | 1 |", report)
        self.assertIn("Questions: 2; pictures: 3. Pictures per question: 1: 1, 2: 1.", report)
        with (self.data / figures.CASES_TABLE).open(encoding="utf-8", newline="") as f:
            self.assertEqual(next(csv.reader(f)), figures.CASES_COLUMNS)

    def test_a_picture_whose_case_is_elsewhere_in_the_text_dataset_stops_the_build(self):
        self.questions[1]["split"] = "test"
        write_csv(self.text / "dataset.csv", ["split", "label", "path", "sha256", "patient_uid", "pmid"], self.questions)
        with self.assertRaisesRegex(ValueError, "but the text dataset has the case in test/nevus"):
            figures.build(self.data, self.text)
        write_csv(self.text / "dataset.csv", ["split", "label", "path", "sha256", "patient_uid", "pmid"], self.questions[2:])
        with self.assertRaisesRegex(ValueError, "is no case of the text dataset"):
            figures.build(self.data, self.text)
