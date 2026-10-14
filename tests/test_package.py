import unittest

import medrag


class PackageTests(unittest.TestCase):
    def test_package_imports(self):
        self.assertTrue(medrag.__version__)


if __name__ == "__main__":
    unittest.main()
