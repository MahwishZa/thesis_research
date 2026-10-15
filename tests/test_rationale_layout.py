"""Guards for the layout of src/rationale_query and experiments/: docstrings, light imports, and the specification."""

import ast
import subprocess
import sys
import unittest

from tests.spec_targets import ROOT, SRC

PACKAGE = SRC / "rationale_query"
METHODOLOGY = ROOT / "docs" / "methodology.md"
TESTS = ROOT / "tests"


def modules():
    return sorted(path.stem for path in PACKAGE.glob("*.py") if path.stem not in ("__init__", "__main__"))


class PackageLayoutTests(unittest.TestCase):
    def test_every_module_has_a_docstring(self):
        for path in PACKAGE.glob("*.py"):
            with self.subTest(module=path.name):
                self.assertTrue(ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))))

    def test_the_package_docstring_lists_every_module(self):
        doc = ast.get_docstring(ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8")))
        for name in modules():
            with self.subTest(module=name):
                self.assertRegex(doc, rf"(?m)^\s+{name}\s{{2,}}\S")

    def test_every_module_is_named_by_a_test(self):
        tests = "".join(path.read_text(encoding="utf-8") for path in TESTS.glob("test_rationale_*.py"))
        for name in modules():
            with self.subTest(module=name):
                self.assertRegex(tests, rf"rationale_query(\.| import .*\b){name}\b")

    def test_importing_the_commands_loads_none_of_the_heavy_packages(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "import rationale_query, rationale_query.cli, rationale_query.generator, rationale_query.store\n"
            "import rationale_query.run, rationale_query.diagnostics, rationale_query.prompt, rationale_query.query\n"
            "print(' '.join(name for name in ('numpy', 'torch', 'transformers', 'llama_cpp') if name in sys.modules))\n"
        )
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "")


class SpecificationTests(unittest.TestCase):
    def test_the_specification_names_every_test_file_of_the_built_part(self):
        text = METHODOLOGY.read_text(encoding="utf-8")
        for path in sorted(TESTS.glob("test_rationale_*.py")):
            with self.subTest(file=path.name):
                self.assertIn(path.name, text)


class ExperimentFolderTests(unittest.TestCase):
    def test_every_experiment_folder_follows_the_convention_of_experiments_readme(self):
        for folder in sorted((ROOT / "experiments").iterdir()):
            if folder.is_dir():
                with self.subTest(folder=folder.name):
                    self.assertRegex(folder.name, r"^\d\d_[a-z0-9_]+$")  # numbered, and none of the names the ignore rules hide
                    self.assertTrue((folder / "NOTES.md").is_file())


if __name__ == "__main__":
    unittest.main()
