"""Checks of the shared module (src/common.py) and of step 2 of the data pipeline (src/build_views.py): building the views again
from data/processed/master_clean.csv gives the committed view files byte for byte."""

import filecmp
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_views  # noqa: E402
import common  # noqa: E402

DATA = Path(os.environ.get("THESIS_DATA", ROOT / "data"))


class CommonTests(unittest.TestCase):
    def test_an_interrupted_write_leaves_the_old_file_and_no_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "table.csv")
            path.write_text("old", encoding="utf-8")
            with self.assertRaises(RuntimeError), common.atomic_path(path) as tmp:
                Path(tmp).write_text("half", encoding="utf-8")
                raise RuntimeError("interrupted")
            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(os.listdir(d), ["table.csv"])
            common.write_csv(path, [{"a": "1", "b": "x, y"}], ["a", "b"])
            self.assertEqual(path.read_bytes(), b'a,b\n1,"x, y"\n')     # LF endings, quoting as pandas writes

    def test_json_and_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            common.write_jsonl(Path(d, "x.jsonl"), [{"id": "a"}, {"id": "b"}])
            self.assertEqual(common.read_jsonl(Path(d, "x.jsonl")), [{"id": "a"}, {"id": "b"}])
            common.write_json(Path(d, "x.json"), {"a": 1})
            self.assertEqual(Path(d, "x.json").read_text(encoding="utf-8"), '{\n  "a": 1\n}')

    def test_the_layout_names(self):
        self.assertEqual(common.view_table("text_only"), "views/text_only/cases.csv")
        self.assertEqual(common.question_file("multimodal", "test"), "views/multimodal/qa_test.jsonl")
        self.assertEqual(len(common.CLASSES), len(common.LETTERS))
        self.assertEqual(set(common.CLASS_NAMES), set(common.CLASSES))
        self.assertEqual(set(common.OPTION_TEXT), set(common.CLASSES))


@unittest.skipUnless((DATA / common.MASTER).is_file(), "data/processed/master_clean.csv is not there")
class BuildViewsTests(unittest.TestCase):
    def test_building_the_views_again_gives_the_committed_files_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, common.PROCESSED).mkdir()
            shutil.copy(DATA / common.MASTER, Path(d, common.MASTER))
            build_views.build(d)
            names = [common.view_table(v) for v in common.VIEWS]
            names += [common.question_file(v, s) for v in common.VIEWS for s in common.SPLITS]
            different = [n for n in names if not filecmp.cmp(Path(d, n), DATA / n, shallow=False)]
            built = sorted(p.relative_to(d).as_posix() for p in Path(d, common.VIEWS_DIR).rglob("*") if p.is_file())
        self.assertEqual(different, [])
        self.assertEqual(built, sorted(names))                           # nothing else is written

    def test_an_unknown_class_stops_the_build(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, common.PROCESSED).mkdir()
            text = (DATA / common.MASTER).read_text(encoding="utf-8").replace(",MEL,", ",XYZ,", 1)
            Path(d, common.MASTER).write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                build_views.build(d)


if __name__ == "__main__":
    unittest.main()
