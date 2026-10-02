"""The hermeticity check must see a file being created, modified and deleted, and must not be
fooled by Python bytecode - otherwise it would report a clean tree while tests write into it."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.tests.check_hermetic import changes, snapshot


class SnapshotTests(unittest.TestCase):

    def test_an_untouched_tree_has_no_changes(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("one", encoding="utf-8")
            self.assertEqual(changes(snapshot(tmp), snapshot(tmp)), [])

    def test_created_modified_and_deleted_files_are_all_reported(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kept.txt").write_text("same", encoding="utf-8")
            (root / "edited.txt").write_text("short", encoding="utf-8")
            (root / "removed.txt").write_text("gone", encoding="utf-8")
            before = snapshot(root)

            (root / "edited.txt").write_text("a longer text", encoding="utf-8")
            (root / "removed.txt").unlink()
            (root / "sub").mkdir()
            (root / "sub" / "new.txt").write_text("x", encoding="utf-8")

            self.assertEqual(changes(before, snapshot(root)), [
                "created:  sub/new.txt",
                "deleted:  removed.txt",
                "modified: edited.txt",
            ])

    def test_a_rewrite_with_identical_size_is_still_detected_by_its_modification_time(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "marker.json"
            target.write_text("12345", encoding="utf-8")
            os.utime(target, ns=(1_000_000_000, 1_000_000_000))
            before = snapshot(tmp)
            target.write_text("54321", encoding="utf-8")
            os.utime(target, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(changes(before, snapshot(tmp)), ["modified: marker.json"])

    def test_bytecode_and_git_state_are_ignored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = snapshot(root)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"x")
            (root / ".git").mkdir()
            (root / ".git" / "index").write_bytes(b"y")
            self.assertEqual(changes(before, snapshot(root)), [])


if __name__ == "__main__":
    unittest.main()
