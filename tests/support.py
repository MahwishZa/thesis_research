"""What several test folders share: a temporary folder for each test."""

import shutil
import tempfile
import unittest
from pathlib import Path


class TempDirTestCase(unittest.TestCase):
    """Gives each test an empty folder ``self.dir``, removed afterwards, and ``self.write(name, text)`` to make a file in it."""

    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp(prefix="rq_test_"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path
