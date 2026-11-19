"""Step clean (src/build/clinic_patients/clean.py) on the invented cohort of project.py."""

import unittest

from src.build.clinic_patients import clean
from tests.build.clinic_patients.project import Cohort


class CleanTests(Cohort):
    def row(self, img):
        return next(r for r in self.rows if r["img_id"] == img)

    def test_a_clean_download_changes_nothing_but_the_way_gaps_are_written(self):
        before = (self.raw / "metadata.csv").read_bytes()
        done = clean.clean(self.data)
        self.assertEqual((done["rows"], done["normalised"], done["flags"]), (125, 0, 0))
        # by hand: the early benign records (ACK, NEV, SEK; patients 0 to 5; 2 photos each = 36 photos) lack 4 fields; 24 cancer photos have grew = UNK
        self.assertEqual(done["gaps"], {"smoke": 36, "background_father": 36, "gender": 36, "diameter_1": 36, "grew": 24})
        cleaned = self.table("cleaned.csv")
        self.assertEqual([r["img_id"] for r in cleaned], [r["img_id"] for r in self.rows])  # no row left out, same order
        for original, new in zip(self.rows, cleaned):
            for column in ("smoke", "background_father", "gender", "diameter_1", "grew"):
                self.assertEqual(new[column] == clean.NOT_RECORDED, original[column] in ("", "UNK"), (original["img_id"], column))
            self.assertEqual((new["age"], new["region"], new["diagnostic"]), (original["age"], original["region"], original["diagnostic"]))
        self.assertEqual((self.raw / "metadata.csv").read_bytes(), before)  # the original is never changed
        self.assertEqual(self.table("cleaning.csv"), [])

    def test_the_pattern_of_gaps_is_kept_by_class(self):
        clean.clean(self.data)
        gap = {}
        for r in self.table("cleaned.csv"):
            gap.setdefault(r["diagnostic"], []).append(r["gender"] == clean.NOT_RECORDED)
        self.assertEqual({k: sum(v) for k, v in gap.items()}, {"ACK": 12, "BCC": 0, "MEL": 0, "NEV": 12, "SCC": 0, "SEK": 12 + 0})

    def test_case_spaces_and_the_spelling_variant_are_normalised_and_logged(self):
        self.row("P100_1_a.png")["itch"] = "true"
        self.row("P100_1_b.png")["region"] = " FACE "
        self.row("P101_1_a.png")["background_father"] = "Brasil"
        self.row("P101_1_b.png")["background_father"] = "poland"
        self.write()
        done = clean.clean(self.data)
        self.assertEqual(done["normalised"], 4)
        log = {(r["img_id"], r["column"]): r for r in self.table("cleaning.csv")}
        self.assertEqual((log["P100_1_a.png", "itch"]["old"], log["P100_1_a.png", "itch"]["new"], log["P100_1_a.png", "itch"]["reason"]),
                         ("true", "TRUE", "case or spaces standardised"))
        self.assertEqual(log["P100_1_b.png", "region"]["new"], "FACE")
        self.assertEqual((log["P101_1_a.png", "background_father"]["new"], log["P101_1_a.png", "background_father"]["reason"]),
                         ("BRAZIL", "spelling variant unified"))
        self.assertEqual(log["P101_1_b.png", "background_father"]["new"], "POLAND")  # an unknown country is only put in capitals
        cleaned = {r["img_id"]: r for r in self.table("cleaned.csv")}
        self.assertEqual(cleaned["P101_1_b.png"]["background_father"], "POLAND")

    def test_implausible_values_and_disagreements_are_flagged_and_kept(self):
        self.row("P100_1_a.png")["diameter_1"] = "0"
        self.row("P100_1_b.png")["age"] = "130"
        self.row("P101_1_a.png")["gender"] = "MALE"  # its other photo says FEMALE
        self.row("P101_1_b.png")["gender"] = "FEMALE"
        self.row("P102_1_b.png")["region"] = "NOSE"  # its other photo: another region, same lesion
        self.row("P102_1_a.png")["region"] = "ARM"
        self.row("P103_1_a.png")["age"] = "forty"
        self.write()
        done = clean.clean(self.data)
        flags = {(r["img_id"], r["patient_id"], r["column"], r["reason"]) for r in self.table("cleaning.csv")}
        self.assertEqual(done["flags"], 5)
        self.assertEqual(flags, {("P100_1_a.png", "P100", "diameter_1", "diameter is 0: implausible for a lesion"),
                                 ("P100_1_b.png", "P100", "age", "age outside 0 to 120"),
                                 ("P103_1_a.png", "P103", "age", "age is not a whole number"),
                                 ("", "P101", "gender", "differs between photos of one patient"),
                                 ("", "P102", "region", "lesion 1 is on different body regions")})
        cleaned = {r["img_id"]: r for r in self.table("cleaned.csv")}
        self.assertEqual((cleaned["P100_1_a.png"]["diameter_1"], cleaned["P100_1_b.png"]["age"], cleaned["P103_1_a.png"]["age"]),
                         ("0", "130", "forty"))  # kept as recorded
        report = self.report()
        self.assertIn("| diameter is 0: implausible for a lesion | 1 | BCC 1 |", report)
        self.assertIn("| differs between photos of one patient (a patient-level field) | 1 | `P101: gender` |", report)

    def test_a_value_outside_the_vocabulary_stops_the_step_before_anything_is_written(self):
        self.row("P100_1_a.png")["region"] = "KNEE"
        self.row("P101_1_a.png")["diagnostic"] = "XYZ"
        self.row("P102_1_a.png")["biopsed"] = ""
        self.row("P103_1_a.png")["gender"] = "OTHER"
        self.write()
        with self.assertRaises(ValueError) as raised:
            clean.clean(self.data)
        message = str(raised.exception)
        self.assertIn("4 value(s) outside the known vocabulary", message)
        for expected in ("P100_1_a.png: region = 'KNEE'", "P101_1_a.png: diagnostic = 'XYZ'", "P102_1_a.png: biopsed = ''",
                         "P103_1_a.png: gender = 'OTHER'"):
            self.assertIn(expected, message)
        self.assertFalse((self.data / "build" / "cleaned.csv").exists())

    def test_a_blank_diagnosis_or_biopsy_value_is_not_a_gap_to_be_written_but_stops_the_step(self):
        self.row("P100_1_a.png")["diagnostic"] = ""
        self.write()
        with self.assertRaisesRegex(ValueError, "P100_1_a.png: diagnostic = ''"):
            clean.clean(self.data)

    def test_the_limits_of_the_age_range_are_allowed(self):
        self.row("P100_1_a.png")["age"] = "0"
        self.row("P100_1_b.png")["age"] = "120"
        self.row("P101_1_a.png")["age"] = "121"
        self.row("P101_1_b.png")["age"] = "-1"
        self.write()
        clean.clean(self.data)
        flagged = {r["img_id"] for r in self.table("cleaning.csv") if r["column"] == "age"}
        self.assertEqual(flagged, {"P101_1_a.png", "P101_1_b.png"})

    def test_the_run_is_repeatable(self):
        clean.clean(self.data)
        first = {p.name: p.read_bytes() for p in (self.data / "build").iterdir()} | {"r": (self.data / "dataset_report.md").read_bytes()}
        clean.clean(self.data)
        second = {p.name: p.read_bytes() for p in (self.data / "build").iterdir()} | {"r": (self.data / "dataset_report.md").read_bytes()}
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
