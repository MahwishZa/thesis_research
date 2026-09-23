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
        target = Path(os.environ["ALZHEIMER_CORPUS_LOGS"]).resolve()
        self.assertFalse(
            target.is_relative_to(ROOT),
            f"corpus test logs must not land inside the repository: {target}",
        )

    def test_get_logger_honours_the_override(self):
        """The mechanism itself, exercised rather than assumed - an
        in-process stage call must land in the redirected directory."""
        log = _common.get_logger("log_isolation_probe")
        log.info("probe")
        for handler in log.handlers:
            handler.flush()

        expected = Path(os.environ["ALZHEIMER_CORPUS_LOGS"])
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
        saved = os.environ.pop("ALZHEIMER_CORPUS_LOGS")
        try:
            self.assertEqual(Path(_common.LOGS).resolve(),
                             CORPUS_LOGS.resolve())
            self.assertIsNone(os.environ.get("ALZHEIMER_CORPUS_LOGS"))
        finally:
            os.environ["ALZHEIMER_CORPUS_LOGS"] = saved


if __name__ == "__main__":
    unittest.main()
