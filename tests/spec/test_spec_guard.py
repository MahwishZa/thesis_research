"""Guards for the acceptance tests themselves: the target map, the skip rule and the link to docs/methodology.md."""

import io
import keyword
import os
import re
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from tests import ROOT
from tests.spec import spec_targets
from tests.spec.spec_targets import (
    FILTER_INSTRUCTION,
    RATIONALE_PROMPT,
    TARGETS,
    SpecTestCase,
    part_started,
    resolve,
)

METHODOLOGY = ROOT / "docs" / "methodology.md"


def squeeze(text):
    return " ".join(text.split())


def acceptance_section():
    text = METHODOLOGY.read_text(encoding="utf-8")
    match = re.search(r"^## [^\n]*Acceptance tests[^\n]*\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    if match is None:
        raise AssertionError("docs/methodology.md has no 'Acceptance tests' section")
    return match.group(1)


class TargetMapTests(unittest.TestCase):
    def test_every_name_is_a_valid_module_path_and_function(self):
        for name, (module, attr) in TARGETS.items():
            with self.subTest(target=name):
                for part in module.split("."):
                    self.assertTrue(part.isidentifier() and not keyword.iskeyword(part), module)
                self.assertTrue(attr.isidentifier() and not keyword.iskeyword(attr), attr)

    def test_no_top_level_package_shadows_the_standard_library(self):
        for module, _ in TARGETS.values():
            with self.subTest(module=module):
                self.assertNotIn(module.split(".")[0], sys.stdlib_module_names)

    def test_every_target_is_used_by_a_test_and_every_use_is_a_target(self):
        used = set()
        for path in Path(__file__).parent.glob("test_spec_*.py"):
            if path.name == Path(__file__).name:
                continue
            source = path.read_text(encoding="utf-8")
            used |= set(re.findall(r"resolve\(\s*[\"'](\w+)[\"']", source))
            for needed in re.findall(r"^\s*requires = \(([^)]*)\)", source, re.MULTILINE):
                used |= set(re.findall(r"[\"'](\w+)[\"']", needed))
        self.assertEqual(set(TARGETS) - used, set(), "targets without a test")
        self.assertEqual(used - set(TARGETS), set(), "tests that name an unknown target")

    def test_the_specification_names_every_target(self):
        section = acceptance_section()
        for name, (module, attr) in TARGETS.items():
            with self.subTest(target=name):
                self.assertIn("`" + module + "`", section)
                self.assertIn("`" + attr + "`", section)

    def test_the_specification_explains_the_strict_run(self):
        self.assertIn("SPEC_STRICT", acceptance_section())

    def test_the_specification_quotes_the_texts_the_tests_check(self):
        text = squeeze(METHODOLOGY.read_text(encoding="utf-8"))
        self.assertIn(squeeze(RATIONALE_PROMPT), text)
        self.assertIn(squeeze(FILTER_INSTRUCTION), text)


class ResolveTests(unittest.TestCase):
    """The skip rule: skip only while a part is unstarted; afterwards every problem is loud."""

    def setUp(self):
        self.src = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.src, ignore_errors=True)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SPEC_STRICT", None)
        self.top = "specfake_" + uuid.uuid4().hex[:8]
        self.targets = {"f": (self.top + ".mod", "f"), "g": (self.top + ".other", "g")}
        self.addCleanup(self.forget)

    def forget(self):
        for name in [name for name in sys.modules if name == self.top or name.startswith(self.top + ".")]:
            del sys.modules[name]
        if str(self.src) in sys.path:
            sys.path.remove(str(self.src))

    def write(self, relative, text=""):
        path = self.src / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def get(self, name):
        return resolve(name, self.targets, self.src)

    def raised_by(self, name):
        """The exception that resolving `name` raises (a SkipTest counts), so that a skip can never hide a failure."""
        try:
            self.get(name)
        except Exception as error:  # noqa: BLE001 - any exception, SkipTest included, is what is being asked about
            return error
        self.fail(f"resolving {name!r} raised nothing")

    def test_an_unstarted_part_is_skipped(self):
        self.assertIsInstance(self.raised_by("f"), unittest.SkipTest)

    def test_a_folder_without_python_files_is_not_a_started_part(self):
        self.write(self.top + "/.gitkeep")
        self.assertFalse(part_started(self.top + ".mod", self.src))
        self.assertIsInstance(self.raised_by("f"), unittest.SkipTest)

    def test_the_strict_run_turns_a_skip_into_a_failure(self):
        os.environ["SPEC_STRICT"] = "1"
        self.assertIsInstance(self.raised_by("f"), AssertionError)

    def test_a_started_part_returns_the_function(self):
        self.write(self.top + "/__init__.py")
        self.write(self.top + "/mod.py", "def f():\n    return 41\n")
        self.assertEqual(self.get("f")(), 41)

    def test_a_single_file_module_starts_a_part(self):
        self.write(self.top + ".py", "def f():\n    return 1\n")
        self.assertEqual(resolve("f", {"f": (self.top, "f")}, self.src)(), 1)

    def test_a_started_part_with_a_missing_module_fails_instead_of_skipping(self):
        self.write(self.top + "/__init__.py")
        self.write(self.top + "/mod.py", "def f():\n    return 1\n")
        self.assertIsInstance(self.raised_by("g"), AssertionError)

    def test_a_missing_function_fails_instead_of_skipping(self):
        self.write(self.top + "/__init__.py")
        self.write(self.top + "/mod.py", "def other():\n    return 1\n")
        self.assertIsInstance(self.raised_by("f"), AssertionError)

    def test_an_error_inside_a_module_is_reported_as_it_is(self):
        self.write(self.top + "/__init__.py")
        self.write(self.top + "/mod.py", "raise RuntimeError('broken module')\n")
        self.assertIsInstance(self.raised_by("f"), RuntimeError)

    def test_a_missing_dependency_is_reported_as_it_is(self):
        self.write(self.top + "/__init__.py")
        self.write(self.top + "/mod.py", "import a_package_that_does_not_exist_xyz\n")
        self.assertIsInstance(self.raised_by("f"), ModuleNotFoundError)


class SpecTestCaseTests(unittest.TestCase):
    """The base class skips a whole test once, before its body runs, and runs it when the targets resolve."""

    @staticmethod
    def run_probe():
        ran = []

        class Probe(SpecTestCase):
            requires = ("label_pair", "dense_query")

            def test_body(self):
                ran.append(True)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Probe)
        return unittest.TextTestRunner(stream=io.StringIO()).run(suite), ran

    def test_an_unimplemented_target_skips_the_test_before_its_body_runs(self):
        with mock.patch.object(spec_targets, "resolve", side_effect=unittest.SkipTest("not implemented yet")):
            result, ran = self.run_probe()
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(ran, [])

    def test_a_failing_resolution_is_an_error_not_a_skip(self):
        with mock.patch.object(spec_targets, "resolve", side_effect=AssertionError("module missing")):
            result, ran = self.run_probe()
        self.assertEqual(len(result.skipped), 0)
        self.assertEqual(len(result.failures) + len(result.errors), 1)
        self.assertEqual(ran, [])

    def test_resolved_targets_let_the_body_run(self):
        with mock.patch.object(spec_targets, "resolve", return_value=len) as patched:
            result, ran = self.run_probe()
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(ran, [True])
        self.assertEqual([call.args[0] for call in patched.call_args_list], ["label_pair", "dense_query"])
