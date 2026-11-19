"""The helpers the builders share (src/common.py): the table writer."""

import tempfile
import unittest
from pathlib import Path

from src.common import read_rows, write_rows


class WriteRowsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)

    def test_a_table_is_written_with_unix_line_endings_and_read_back_and_no_temporary_file_is_left(self):
        path = self.folder / "deeper" / "t.csv"  # the folder is made
        write_rows(path, ["a", "b"], [{"a": "1", "b": "x, y"}, {"a": "2", "b": "line\nbreak"}])
        self.assertEqual(path.read_bytes(), b'a,b\n1,"x, y"\n2,"line\nbreak"\n')
        self.assertEqual(read_rows(path, "unused"), [{"a": "1", "b": "x, y"}, {"a": "2", "b": "line\nbreak"}])
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["t.csv"])

    def test_a_table_is_replaced_whole_and_a_stop_in_the_middle_leaves_the_old_one(self):
        path = self.folder / "t.csv"
        write_rows(path, ["a"], [{"a": "old"}])

        class Broken(dict):
            def get(self, *args):
                raise KeyboardInterrupt  # a stop while the rows are written

        with self.assertRaises(KeyboardInterrupt):
            write_rows(path, ["a"], [{"a": "new"}, Broken()])
        self.assertEqual(read_rows(path, "unused"), [{"a": "old"}])
        write_rows(path, ["a"], [{"a": "new"}])
        self.assertEqual(read_rows(path, "unused"), [{"a": "new"}])

    def test_a_missing_table_says_what_to_do(self):
        with self.assertRaisesRegex(FileNotFoundError, "t.csv not found: build it first"):
            read_rows(self.folder / "t.csv", "build it first")


if __name__ == "__main__":
    unittest.main()
