import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class LayoutTests(unittest.TestCase):
    """Placeholder until the pipeline parts exist: the folders the workflow relies on are present."""

    def test_folders_exist(self):
        for name in ("src", "tests", "experiments", "docs", "_archive"):
            self.assertTrue((ROOT / name).is_dir(), name)


if __name__ == "__main__":
    unittest.main()
