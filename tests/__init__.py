"""The tests of the research code: unit, integration, system and acceptance. Run them all from the repository folder with
python -m unittest   (also: python -m unittest discover -s tests -t .)"""
import unittest

FILES = ("unit", "integration", "system", "acceptance")


def load_tests(loader, standard_tests, pattern):
    """The four test files have plain names, so the default discovery pattern (test*.py) would miss them: load them here."""
    suite = unittest.TestSuite()
    for name in FILES:
        suite.addTests(loader.loadTestsFromName(f"tests.{name}"))
    return suite
