"""Guards for the layout of every package under src/ and of experiments/: docstrings, light imports, and the specification."""

import ast
import subprocess
import sys
import unittest

from tests import ROOT, SRC

METHODOLOGY = ROOT / "docs" / "methodology.md"
TESTS = ROOT / "tests"


#: The modules of each package that must import without loading numpy, torch, transformers or llama_cpp.
LIGHT_MODULES = {
    "rationale_query": ("cli", "generator", "store", "batch", "diagnostics", "prompt", "search_text"),
    "balanced_retrieval": ("cli", "pool", "retrieve", "diagnostics", "chunking", "records", "scope", "ncbi", "documents", "pubmed", "pmc", "guidelines", "textbooks"),
    "data_roles": ("splits", "questions", "overlap", "cli"),
    "answer_evaluation": ("prompt", "reader", "paired_tests", "ledger", "records", "runner", "results"),
    "rationale_filter": ("labels", "inference", "perplexity", "scorer", "label_records", "labelling", "training_data", "filter_eval"),
}


def packages():
    """Every package under src/ (a folder with an __init__.py), by name."""
    return sorted(path.name for path in SRC.iterdir() if (path / "__init__.py").is_file())


def modules(package):
    return sorted(path.stem for path in (SRC / package).glob("*.py") if path.stem not in ("__init__", "__main__"))


class RepositoryFolderTests(unittest.TestCase):
    def test_the_folders_the_workflow_relies_on_exist(self):
        for name in ("src", "tests", "experiments", "docs", "_archive"):
            self.assertTrue((ROOT / name).is_dir(), name)


def built_tests():
    """The test files of the built partss: every test file in the unit and integration folders, at any depth."""
    return sorted(path for folder in ("unit", "integration") for path in (TESTS / folder).rglob("test_*.py"))


def experiment_folders():
    """The folders of experiments/ that are part of the repository (a computer may also hold ignored local leftovers)."""
    try:
        done = subprocess.run(
            ["git", "ls-files", "experiments"], capture_output=True, text=True, cwd=ROOT, check=True
        )
        names = {line.split("/")[1] for line in done.stdout.splitlines() if line.count("/") >= 2}
    except (OSError, subprocess.CalledProcessError):
        return sorted((ROOT / "experiments").iterdir())
    return sorted(ROOT / "experiments" / name for name in names)


class PackageLayoutTests(unittest.TestCase):
    def test_every_package_is_known_to_the_guards(self):
        self.assertEqual(packages(), sorted(LIGHT_MODULES), "a new package under src/ needs an entry in LIGHT_MODULES")

    def test_every_module_has_a_docstring(self):
        for package in packages():
            for path in (SRC / package).glob("*.py"):
                with self.subTest(module=f"{package}.{path.name}"):
                    self.assertTrue(ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))))

    def test_the_package_docstring_lists_every_module(self):
        for package in packages():
            doc = ast.get_docstring(ast.parse((SRC / package / "__init__.py").read_text(encoding="utf-8")))
            for name in modules(package):
                with self.subTest(module=f"{package}.{name}"):
                    self.assertRegex(doc, rf"(?m)^\s+{name}\s{{2,}}\S")

    def test_every_module_is_named_by_a_test(self):
        tests = "".join(path.read_text(encoding="utf-8") for path in built_tests())
        for package in packages():
            for name in modules(package):
                with self.subTest(module=f"{package}.{name}"):
                    self.assertRegex(tests, rf"{package}(\.| import .*\b){name}\b")

    def test_importing_the_commands_loads_none_of_the_heavy_packages(self):
        for package, light in LIGHT_MODULES.items():
            code = (
                "import sys\n"
                f"sys.path.insert(0, {str(SRC)!r})\n"
                f"import {package}, " + ", ".join(f"{package}.{name}" for name in light) + "\n"
                "print(' '.join(name for name in ('numpy', 'torch', 'transformers', 'llama_cpp') if name in sys.modules))\n"
            )
            with self.subTest(package=package):
                done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(done.stdout.strip(), "")


class SpecificationTests(unittest.TestCase):
    def test_the_specification_names_every_test_file_of_the_built_part(self):
        text = METHODOLOGY.read_text(encoding="utf-8")
        for path in built_tests():
            with self.subTest(file=path.name):
                self.assertIn(path.name, text)


class ExperimentFolderTests(unittest.TestCase):
    def test_every_experiment_folder_follows_the_convention_of_experiments_readme(self):
        for folder in experiment_folders():
            if folder.is_dir():
                with self.subTest(folder=folder.name):
                    self.assertRegex(folder.name, r"^\d\d_[a-z0-9_]+$")  # numbered, and none of the names the ignore rules hide
                    self.assertTrue((folder / "NOTES.md").is_file())


if __name__ == "__main__":
    unittest.main()
