"""Checks of the scripts in src/ that need no images: the patient text, the options of a question, the raw folder layout and the
audit of the original files. They skip when the scripts' packages are not installed (pip install -r requirements.txt)."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAVE = all(importlib.util.find_spec(p) for p in ("pandas", "numpy", "cv2", "PIL", "sklearn"))
SENTENCE = " A dermoscopic image of the lesion is available."


@unittest.skipUnless(HAVE, "the packages of src/preprocess.py are not installed")
class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import preprocess
        cls.pd, cls.p = pd, preprocess

    def test_the_patient_text_says_only_the_recorded_facts_and_nothing_about_an_image(self):
        nan = float("nan")
        self.assertEqual(self.p.clinical_text(55.0, "female", "head/neck", "face"), "Patient: 55-year-old female. Skin lesion on the head/neck (face).")
        self.assertEqual(self.p.clinical_text(nan, nan, nan, nan), "Patient: age not recorded, sex not recorded. Lesion site not recorded.")
        self.assertEqual(self.p.clinical_text(70.0, "male", "trunk", "trunk"), "Patient: 70-year-old male. Skin lesion on the trunk.")  # no repeat

    def test_the_new_text_is_the_old_text_without_the_image_sentence_for_every_committed_case(self):
        path = ROOT / "data" / "master_clean.csv"
        if not path.is_file():
            self.skipTest("data/master_clean.csv is not in data/")
        m = self.pd.read_csv(path)
        old = [t.removesuffix(SENTENCE) for t in m.clinical_text]
        new = [self.p.clinical_text(*x) for x in zip(m.age, m.sex, m.site, m.site_detail)]
        self.assertEqual(new, old)

    def test_an_image_path_uses_a_forward_slash_on_every_system(self):
        self.assertEqual(self.p.image_rel_path("ISIC_0000000"), "images/ISIC_0000000.jpg")

    def test_both_layouts_of_the_kaggle_download_are_read(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.p.source_dirs(d), self.p.SOURCE_DIRS)  # the zip parts
            os.makedirs(os.path.join(d, "Train"))
            self.assertEqual(self.p.source_dirs(d), [("Train", "Train"), ("Test", "Test")])  # the unpacked dataset

    def test_the_audit_keeps_every_original_file_and_marks_the_removed_copies(self):
        pd = self.pd
        raw = pd.DataFrame({"isic_id": ["A", "A", "B", "C"], "path": ["p1", "p2", "p3", "p4"], "md5": ["x", "x", "y", "z"]})
        kept = pd.DataFrame({"isic_id": ["A", "B"], "label": ["MEL", "NV"], "split": ["train", "test"]})
        audit = self.p.audit_table(raw, kept, {"C": "A"}, pd.Series({"A": "MEL", "B": "NV", "C": "MEL"}))
        self.assertEqual(len(audit), 4)  # one row per original file
        self.assertEqual(audit[audit.isic_id == "C"][["label", "split"]].values.tolist(), [["MEL", "removed: same picture as A"]])
        self.assertNotIn("path", audit.columns)
        with self.assertRaises(AssertionError):  # a file that is neither kept nor a known copy
            self.p.audit_table(raw, kept, {}, pd.Series({"A": "MEL", "B": "NV"}))


class OptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_qa
        cls.q = build_qa

    def test_the_options_are_the_eight_classes_in_a_fixed_order_for_each_case(self):
        options, letter = self.q.options_for("ISIC_0024948")
        self.assertEqual(sorted(options.values()), sorted(self.q.OPTION_TEXT.values()))
        self.assertEqual(list(options), list("ABCDEFGH"))
        for code, text in self.q.OPTION_TEXT.items():
            self.assertEqual(options[letter[code]], text)  # the answer letter points to the right class
        self.assertEqual(self.q.options_for("ISIC_0024948"), (options, letter))  # the same every time

    def test_the_order_differs_between_cases_so_no_position_stands_for_a_class(self):
        ids = [f"ISIC_{i:07d}" for i in range(200)]
        first = {self.q.options_for(i)[0]["A"] for i in ids}
        self.assertEqual(len(first), 8)  # every class appears in position A
        melanoma = {self.q.options_for(i)[1]["MEL"] for i in ids}
        self.assertEqual(len(melanoma), 8)  # melanoma appears at every position


if __name__ == "__main__":
    unittest.main()
