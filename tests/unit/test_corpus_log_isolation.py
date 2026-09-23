"""The test suite must never write to the corpus's tracked provenance logs.

``corpus/logs/*.log`` are not scratch output. They are the corpus's
provenance record: the stage-by-stage count reconciliation in
``_archive/docs_legacy/status_and_decisions.md`` §2 (114,157 normalized -> 111,315 unique ->
4,377,041 chunks -> all tagged) is read directly out of them, and the corpus
is marked COMPLETE/FROZEN on that evidence. A tracked file that changes when
anyone runs ``python -m unittest`` makes that evidence unreliable and buries
the real lines among fixture-scale noise.

This was a real defect, not a hypothetical one. ``_common.get_logger()``
resolved its directory from the loaded module's own ``__file__``, so tests
calling a stage's ``main()`` **in-process** against a temp corpus wrote into
the real log regardless: a suite run appended Stage-06 whitespace-tokenizer
lines naming ``/tmp`` paths to ``quality_control.log``. ``tests/__init__.py``
now redirects the directory for the duration of a test run, and these tests
keep it redirected.

2026-09-24: an apparent intermittent failure of this class, seen across
several earlier audit sessions and wrongly written up at the time as an
unexplained, sandbox-specific flake, is now root-caused and resolved -
not a defect in this repository's code at all. ``python -m unittest
discover -s tests -p "test_*.py"`` **without** ``-t .`` does not reliably
trigger ``tests/__init__.py``'s package-import side effect (it can leave
``ALZHEIMER_CORPUS_LOGS`` unset even before the first test runs, confirmed
by bisecting with a custom ``TestResult`` that checks the variable in
``startTest``), while the documented, correct invocation
(``python -m unittest discover -s tests -t .``, as README.md and
``docs/reproducibility.md`` §2 both specify) does not exhibit this at all,
reproduced clean across repeated runs. The earlier "flakiness" was this
same invocation mistake made inconsistently across sessions, compounded by
a leftover probe file in the tracked directory from one real occurrence
that then made subsequent runs fail for an unrelated, more confusing
reason (see the cleanup below). **Always run the test suite with
``-t .``** - omitting it is a client error, not something this file's
tests can detect or guard against from inside a test run.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "corpus" / "scripts"
CORPUS_LOGS = ROOT / "corpus" / "logs"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _common  # noqa: E402  - needs SCRIPT_DIR on the path first


class LogRedirectionTests(unittest.TestCase):

    def test_the_override_is_set_for_the_whole_test_run(self):
        self.assertTrue(
            os.environ.get("ALZHEIMER_CORPUS_LOGS"),
            "tests/__init__.py should have redirected corpus logging",
        )

    def test_the_override_does_not_point_into_the_repository(self):
        raw = os.environ.get("ALZHEIMER_CORPUS_LOGS")
        self.assertIsNotNone(
            raw, "ALZHEIMER_CORPUS_LOGS was unset when this test ran - "
                 "tests/__init__.py sets it once at test-suite start and "
                 "nothing in this repository unsets it afterward (verified "
                 "2026-09-23/24: it is the only file, besides this one and "
                 "_common.py, that touches this variable at all). If this "
                 "recurs, suspect the test-runner/environment, not this "
                 "file's own logic - re-run in isolation "
                 "(python -m unittest tests.unit.test_corpus_log_isolation) "
                 "to check whether it is reproducible outside a full-suite run.",
        )
        target = Path(raw).resolve()
        self.assertFalse(
            target.is_relative_to(ROOT),
            f"corpus test logs must not land inside the repository: {target}",
        )

    def test_get_logger_honours_the_override(self):
        """The mechanism itself, exercised rather than assumed - an
        in-process stage call must land in the redirected directory.

        Cleans up its own probe file in the tracked directory
        unconditionally (not just on the success path): if the override
        ever really is missing, get_logger() writes there for real, and an
        uncleaned leftover would silently make every later test run in this
        class fail with a confusing, unrelated-looking "wrote into the
        tracked directory" error long after the actual cause was gone -
        exactly the false trail a stray file from an earlier session left
        here once before."""
        self.addCleanup(
            lambda: (CORPUS_LOGS / "log_isolation_probe.log").unlink(missing_ok=True)
        )

        raw = os.environ.get("ALZHEIMER_CORPUS_LOGS")
        self.assertIsNotNone(raw, "ALZHEIMER_CORPUS_LOGS was unset when this "
                             "test ran - see test_the_override_does_not_"
                             "point_into_the_repository for the diagnostic note.")

        log = _common.get_logger("log_isolation_probe")
        log.info("probe")
        for handler in log.handlers:
            handler.flush()

        expected = Path(raw)
        written = expected / "log_isolation_probe.log"
        self.assertTrue(written.exists(),
                        f"logger did not write to {expected}")
        self.assertFalse(
            (CORPUS_LOGS / "log_isolation_probe.log").exists(),
            "logger wrote into the tracked corpus log directory",
        )

    def test_a_real_run_is_unaffected_when_the_variable_is_unset(self):
        """The override is a test-only escape hatch. With the variable
        absent, the resolved directory must be the corpus's own - otherwise
        this change would have altered where a real pipeline stage logs,
        which is a change to executed-corpus provenance."""
        saved = os.environ.pop("ALZHEIMER_CORPUS_LOGS", None)
        self.assertIsNotNone(saved, "ALZHEIMER_CORPUS_LOGS was already unset "
                             "before this test popped it - see the "
                             "diagnostic note on the sibling test in this "
                             "file for what that means.")
        try:
            self.assertEqual(Path(_common.LOGS).resolve(),
                             CORPUS_LOGS.resolve())
            self.assertIsNone(os.environ.get("ALZHEIMER_CORPUS_LOGS"))
        finally:
            os.environ["ALZHEIMER_CORPUS_LOGS"] = saved


if __name__ == "__main__":
    unittest.main()
