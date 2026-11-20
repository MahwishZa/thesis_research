"""Step views (src/build/clinic_patients/views.py): the record text by hand, then the step on the invented cohort of project.py."""

import csv
import hashlib
import io
import unittest

from src.build.clinic_patients import clean, dedupe, images, inventory, split, views
from src.build.clinic_patients.table import CLASS_NAMES, NOT_RECORDED
from tests.build.clinic_patients.project import Cohort

ROW = {"img_id": "x.png", "age": "55", "region": "FACE", "itch": "TRUE", "hurt": "FALSE", "bleed": NOT_RECORDED, "elevation": "TRUE"}


class RecordTextTests(unittest.TestCase):
    def test_the_record_is_six_numbered_lines_in_fixed_wording_by_hand(self):
        self.assertEqual(views.record_text(ROW),
                         "1. Age in years: 55\n2. Body site: face\n3. The lesion itches: yes\n4. The lesion hurts: no\n"
                         "5. The lesion has bled: unknown\n6. The lesion has an elevation: yes")

    def test_the_wording_and_the_template_are_pinned(self):
        # a change of wording is a new template version: change TEMPLATE_VERSION and these two lines together, on purpose
        self.assertEqual(views.TEMPLATE_VERSION, "record-v1")
        self.assertEqual(views.TEMPLATE, "1. Age in years: <age>\n2. Body site: <region>\n3. The lesion itches: <itch>\n"
                                         "4. The lesion hurts: <hurt>\n5. The lesion has bled: <bleed>\n"
                                         "6. The lesion has an elevation: <elevation>")
        self.assertEqual(views.CORE, ("age", "region", "itch", "hurt", "bleed", "elevation"))
        self.assertEqual(views.FINGERPRINT, hashlib.sha256(f"record-v1\n{views.TEMPLATE}\nnot recorded: not recorded".encode()).hexdigest())

    def test_every_value_of_every_kind_by_hand(self):
        self.assertEqual(views.value_of(views.FIELDS[0], {**ROW, "age": "07"}), "7")  # a whole number, written plainly
        self.assertEqual(views.value_of(views.FIELDS[0], {**ROW, "age": "0"}), "0")
        self.assertEqual(views.value_of(views.FIELDS[0], {**ROW, "age": "120"}), "120")
        self.assertEqual(views.value_of(views.FIELDS[1], {**ROW, "region": "FOREARM"}), "forearm")
        self.assertEqual(views.value_of(views.FIELDS[2], {**ROW, "itch": "FALSE"}), "no")
        self.assertEqual(views.value_of(views.FIELDS[2], {**ROW, "itch": "TRUE"}), "yes")
        self.assertEqual(views.value_of(views.FIELDS[3], {**ROW, "hurt": NOT_RECORDED}), "unknown")  # the patient did not know (UNK)
        self.assertEqual(views.UNKNOWN, "unknown")
        self.assertEqual(views.SYMPTOMS, ("itch", "hurt", "bleed", "elevation"))  # the only fields that can be a gap

    def test_a_gap_in_the_age_or_the_body_site_stops_the_step_because_the_authors_say_they_are_always_present(self):
        for column, field in (("age", views.FIELDS[0]), ("region", views.FIELDS[1])):
            with self.assertRaisesRegex(ValueError, f"x.png: {column} is not recorded, but the authors say it is always present"):
                views.value_of(field, {**ROW, column: NOT_RECORDED})

    def test_a_value_a_line_cannot_print_stops_with_the_photo_named(self):
        for age in ("121", "-3", "5.5", "abc", "", "\u00b2"):  # the last is a superscript two: a "digit" that is not a number
            with self.assertRaisesRegex(ValueError, "x.png: age = .* is not a whole number from 0 to 120"):
                views.record_text({**ROW, "age": age})
        with self.assertRaisesRegex(ValueError, "x.png: bleed = 'MAYBE' is neither yes, no nor a gap"):
            views.record_text({**ROW, "bleed": "MAYBE"})

    def test_the_template_has_one_line_for_every_field_of_the_record(self):
        self.assertEqual(len(views.FIELDS), 6)
        self.assertEqual(views.TEMPLATE.count("\n"), 5)
        self.assertEqual(set(views.LEFT_OUT) & set(views.CORE), set())  # what is left out is never also shown


class ViewsStepTests(Cohort):
    def prepare(self):
        inventory.inventory(self.data)
        clean.clean(self.data)
        dedupe.dedupe(self.data)
        split.split(self.data)
        images.images(self.data)

    def test_every_question_is_in_the_photo_table_the_record_table_and_the_record_files(self):
        self.prepare()
        done = views.views(self.data)
        self.assertEqual((done["questions"], done["written"], done["removed"]), (121, 121, 0))
        photos, records = self.table("dataset.csv"), self.table("record.csv")
        self.assertEqual([r["img_id"] for r in photos], [r["img_id"] for r in records])
        self.assertEqual(len(photos), 121)
        self.assertEqual(list(photos[0])[:12], list(views.DATASET_FIRST))
        self.assertEqual(list(photos[0])[12:], ["smoke", "background_father", "age", "gender", "region", "diameter_1", "itch", "grew", "hurt",
                                                "bleed", "elevation"])  # the whole clinical record stays, for the secondary analysis
        self.assertEqual(list(records[0]), views.RECORD_COLUMNS)
        questions, files, raw = ({r["img_id"]: r for r in self.table(n)} for n in ("split.csv", "files.csv", "raw_images.csv"))
        for p, r in zip(photos, records):
            q = questions[p["img_id"]]
            for column in ("split", "label", "patient_id", "lesion_id", "split_group", "biopsed"):
                self.assertEqual((p[column], r[column]), (q[column], q[column]))  # the split, the class and the group are those of step split
            self.assertEqual((p["path"], p["sha256"]), (files[p["img_id"]]["path"], files[p["img_id"]]["sha256"]))  # the photo of step images
            self.assertEqual((p["raw_path"], p["raw_sha256"]), (raw[p["img_id"]]["file"], raw[p["img_id"]]["sha256"]))
            self.assertEqual(p["path"], f"{p['split']}/{p['label']}/{p['img_id'][:-4]}.jpg")
            self.assertEqual(r["path"], f"{p['split']}/{p['label']}/{p['img_id'][:-4]}.txt")
            self.assertEqual((r["photo_path"], r["photo_sha256"]), (p["path"], p["sha256"]))
            self.assertEqual(p["facts"], views.record_text(q))
            content = (self.data / "files" / "records" / r["path"]).read_bytes()
            self.assertEqual(content, p["facts"].encode("utf-8"))  # the record view and the photo + record view show the same words, to the byte
            self.assertEqual(r["sha256"], hashlib.sha256(content).hexdigest())
            self.assertFalse(content.endswith(b"\n"))
        root = self.data / "files" / "records"
        self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob("*.txt")), sorted(r["path"] for r in records))

    def test_the_record_of_a_known_photo_is_what_the_cleaned_table_says_by_hand(self):
        self.prepare()
        views.views(self.data)
        # P000 (actinic keratosis, n = 0, c = 0): age 30, region REGIONS[0] = FACE, itch FALSE, hurt FALSE (n % 3 != 1), bleed FALSE,
        # elevation FALSE (not a cancer)
        row = next(r for r in self.table("dataset.csv") if r["img_id"] == "P000_1_a.png")
        self.assertEqual(row["facts"], "1. Age in years: 30\n2. Body site: face\n3. The lesion itches: no\n4. The lesion hurts: no\n"
                                       "5. The lesion has bled: no\n6. The lesion has an elevation: no")
        # P301 (nevus: c = 3, n = 1): age 30 + 3 + 3 = 36, region REGIONS[4] = NOSE, itch TRUE (n odd), hurt TRUE (n % 3 == 1)
        row = next(r for r in self.table("dataset.csv") if r["img_id"] == "P301_1_a.png")
        self.assertEqual(row["facts"].splitlines()[:4], ["1. Age in years: 36", "2. Body site: nose", "3. The lesion itches: yes",
                                                         "4. The lesion hurts: yes"])
        self.assertEqual(row["label"], "nevus")

    def test_a_gap_in_a_symptom_line_is_written_unknown_and_counted(self):
        for row in self.rows:
            if row["img_id"] == "P500_1_a.png":
                row["bleed"] = "UNK"
        self.write()
        self.prepare()
        views.views(self.data)
        row = next(r for r in self.table("dataset.csv") if r["img_id"] == "P500_1_a.png")
        self.assertIn("5. The lesion has bled: unknown", row["facts"])
        self.assertNotIn("not recorded", row["facts"])
        self.assertEqual(row["bleed"], NOT_RECORDED)  # the cleaned table's own wording stays in the clinical columns
        report = self.report()
        self.assertIn("| Class | Questions | No `unknown` in the record | `itch` unknown | `hurt` unknown | `bleed` unknown | `elevation` unknown |", report)
        self.assertIn("| seborrheic keratosis | 20 | 19 | 0 | 0 | 1 | 0 |", report)

    def test_a_gap_in_the_age_stops_the_step_before_anything_is_written(self):
        self.rows[0]["age"] = ""
        self.write()
        self.prepare()
        with self.assertRaisesRegex(ValueError, "P000_1_a.png: age is not recorded, but the authors say it is always present"):
            views.views(self.data)
        self.assertFalse((self.data / "files" / "records").exists())

    def test_a_value_a_line_cannot_print_stops_the_step_before_anything_is_written(self):
        self.rows[0]["age"] = "200"
        self.write()
        self.prepare()
        with self.assertRaisesRegex(ValueError, "P000_1_a.png: age = '200' is not a whole number from 0 to 120"):
            views.views(self.data)
        self.assertFalse((self.data / "dataset.csv").exists())
        self.assertFalse((self.data / "record.csv").exists())
        self.assertFalse((self.data / "files" / "records").exists())

    def test_a_record_file_that_is_no_longer_wanted_is_removed_and_the_photos_are_not_touched(self):
        self.prepare()
        views.views(self.data)
        (self.data / "files" / "records" / "train" / "nevus").mkdir(parents=True, exist_ok=True)
        (self.data / "files" / "records" / "train" / "nevus" / "stray.txt").write_text("x", encoding="utf-8")
        (self.data / "files" / "records" / "empty" / "folder").mkdir(parents=True)
        before = {p: p.read_bytes() for p in (self.data / "files" / "photos").rglob("*.jpg")}
        done = views.views(self.data)
        self.assertEqual((done["written"], done["removed"]), (0, 1))
        self.assertFalse((self.data / "files" / "records" / "train" / "nevus" / "stray.txt").exists())
        self.assertFalse((self.data / "files" / "records" / "empty").exists())
        self.assertEqual({p: p.read_bytes() for p in (self.data / "files" / "photos").rglob("*.jpg")}, before)

    def test_a_record_file_that_was_changed_is_written_again(self):
        self.prepare()
        views.views(self.data)
        file = next((self.data / "files" / "records").rglob("*.txt"))
        right = file.read_bytes()
        file.write_bytes(right + b" and more")
        done = views.views(self.data)
        self.assertEqual((done["written"], done["removed"]), (1, 0))
        self.assertEqual(file.read_bytes(), right)

    def test_the_steps_before_it_must_have_run(self):
        with self.assertRaisesRegex(FileNotFoundError, "run the split step first"):
            views.views(self.data)
        inventory.inventory(self.data)
        clean.clean(self.data)
        dedupe.dedupe(self.data)
        split.split(self.data)
        with self.assertRaisesRegex(FileNotFoundError, "run the images step first"):
            views.views(self.data)

    def test_a_record_column_missing_from_the_table_stops_the_step(self):
        self.prepare()
        text = (self.data / "build" / "split.csv").read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=[c for c in rows[0] if c != "bleed"], lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items() if k != "bleed"} for r in rows])
        (self.data / "build" / "split.csv").write_text(out.getvalue(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "have no column bleed"):
            views.views(self.data)

    def test_a_question_without_its_photo_stops_the_step_and_names_it(self):
        self.prepare()
        rows = self.table("files.csv")
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[1:])
        (self.data / "build" / "files.csv").write_text(out.getvalue(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no standardised photo for `" + rows[0]["img_id"] + "`"):
            views.views(self.data)

    def test_the_report_names_the_template_the_gaps_the_identical_records_and_an_example_of_each_class(self):
        self.prepare()
        views.views(self.data)
        report = self.report()
        self.assertIn(f"Record template `record-v1` (fingerprint `{views.FINGERPRINT[:12]}`)", report)
        self.assertIn("- Questions: 121 (train ", report)
        self.assertIn("| Class | Questions | No `unknown` in the record | `itch` unknown |", report)
        self.assertIn("| all | 121 | 121 | 0 | 0 | 0 | 0 |", report)
        self.assertIn("A gap in a symptom line is written `unknown`: the patient did not know the answer.", report)
        self.assertIn("- Not in any view: `grew`, `changed`", report)
        for name in CLASS_NAMES.values():
            self.assertIn(f"**{name}** (`", report)
        self.assertIn(views.TEMPLATE, report)

    def test_identical_records_are_counted_by_what_they_share_by_hand(self):
        rows = [{"img_id": "a", "patient_id": "P1", "label": "nevus", "facts": "same"},
                {"img_id": "b", "patient_id": "P1", "label": "nevus", "facts": "same"},     # a second photo of P1's visit
                {"img_id": "c", "patient_id": "P2", "label": "melanoma", "facts": "same"},  # another patient, another class
                {"img_id": "d", "patient_id": "P3", "label": "nevus", "facts": "other"},
                {"img_id": "e", "patient_id": "P4", "label": "nevus", "facts": "twin"},     # two patients, one class, one record
                {"img_id": "f", "patient_id": "P5", "label": "nevus", "facts": "twin"},
                {"img_id": "g", "patient_id": "P6", "label": "nevus", "facts": "visit"},    # two photos of one visit, nothing else
                {"img_id": "h", "patient_id": "P6", "label": "nevus", "facts": "visit"}]
        self.assertEqual([[r["img_id"] for r in g] for g in views._identical(rows)], [["a", "b", "c"], ["e", "f"], ["g", "h"]])
        lines = views._report([{**r, "split": "train", **{c: "1" for c in views.CORE}} for r in rows])
        text = "\n".join(lines)
        self.assertIn("Distinct records: 4 for 8 questions. Questions whose record is the same as another question's: 7 (in 3 groups).", text)
        self.assertIn("  - with another photo of the same patient: 4 (several photos of one visit share one record);", text)  # a, b, g, h
        self.assertIn("  - with a photo of a different patient: 5;", text)  # a, b, c, e and f; g and h share with their own patient only
        self.assertIn("  - in a group of more than one class: 3 (", text)  # a, b and c; e and f are one class

    def test_the_run_is_repeatable_to_the_byte_and_a_second_run_writes_nothing(self):
        self.prepare()
        views.views(self.data)
        first = [(self.data / n).read_bytes() for n in ("dataset.csv", "record.csv", "dataset_report.md")]
        done = views.views(self.data)
        self.assertEqual(done["written"], 0)
        self.assertEqual([(self.data / n).read_bytes() for n in ("dataset.csv", "record.csv", "dataset_report.md")], first)


if __name__ == "__main__":
    unittest.main()
