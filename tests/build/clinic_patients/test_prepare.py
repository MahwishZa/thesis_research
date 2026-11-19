"""The command line of the clinic patients cohort (src/build/clinic_patients/prepare.py): the five steps in order, on the invented cohort."""

import contextlib
import io
import os

from src.build.clinic_patients import prepare
from tests.build.clinic_patients.project import Cohort


class PrepareTests(Cohort):
    def run_steps(self, *args):
        out = io.StringIO()
        here = os.getcwd()
        os.chdir(self.base)  # the steps write to data/clinic_patients, relative to where they run
        try:
            with contextlib.redirect_stdout(out):
                prepare.main(list(args))
        finally:
            os.chdir(here)
        return out.getvalue()

    def files(self):
        return {p.relative_to(self.data).as_posix(): p.read_bytes() for p in self.data.rglob("*")
                if p.is_file() and "raw" not in p.relative_to(self.data).parts and "downloads" not in p.relative_to(self.data).parts}

    def test_all_runs_the_five_steps_in_order_and_writes_every_table_and_one_report_section_each(self):
        text = self.run_steps("all")
        steps = [line.split("]")[0] for line in text.splitlines() if line.startswith("[")]
        self.assertEqual(steps, ["[inventory", "[clean", "[dedupe", "[split", "[shortcuts"])
        self.assertIn("[split] 121 photos in 61 split groups:", text)
        self.assertIn("Commit them all: git add data/clinic_patients", text)
        written = set(self.files())
        for expected in ("dataset.csv", "dataset_report.md", "build/raw_images.csv", "build/columns.csv", "build/cleaned.csv",
                         "build/cleaning.csv", "build/photo_hashes.csv", "build/duplicates.csv", "build/near_pairs.csv",
                         "build/excluded.csv", "build/shortcuts.csv"):
            self.assertIn(expected, written)
        report = self.report()
        sections = [line for line in report.splitlines() if line.startswith("<!-- step:")]
        self.assertEqual(sections, [f"<!-- step: {s} -->" for s in ("inventory", "clean", "dedupe", "split", "shortcuts")])

    def test_a_second_run_gives_the_same_files_to_the_byte(self):
        self.run_steps("all")
        first = self.files()
        self.run_steps("all")
        self.assertEqual(self.files(), first)

    def test_one_step_can_be_run_alone_and_a_missing_earlier_step_is_named(self):
        self.run_steps("inventory")
        with self.assertRaises(SystemExit) as raised:
            self.run_steps("split")
        self.assertIn("ERROR in step split:", str(raised.exception))
        self.assertIn("run the clean step first", str(raised.exception))
        with self.assertRaises(SystemExit) as raised:
            self.run_steps("unknown")
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    import unittest
    unittest.main()
