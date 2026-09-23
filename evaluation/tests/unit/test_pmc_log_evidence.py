"""Stage 02 finalization must be able to read the evidence it was given.

The PMC corpus was retrieved over several days by successive versions of
``02_pmc_download.py``. Each version worded the same finding - PMC holds no
article version for this PMCID - slightly differently. Finalization parses
those historical log lines as its authoritative record of which PMCIDs are
legitimately unavailable.

A regex that matched only the newest wording silently recovered **zero**
unavailable PMCIDs from a log containing 102 of them, so finalization could
never account for those records and refused to write the manifest. No rerun
could fix it: log lines already written cannot be reworded.

These tests lock the reader against that whole class of failure - a producer
and a consumer of the same evidence drifting apart.
"""

import importlib.util
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "corpus" / "scripts" / "02_pmc_download.py"
LOG = ROOT / "corpus" / "logs" / "retrieval.log"


def load_module():
    """Load the stage script by path - its name starts with a digit."""
    spec = importlib.util.spec_from_file_location("pmc_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quiet_logger():
    log = logging.getLogger("pmc_evidence_test")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


#: Every wording this repository has actually written for the same finding.
#: Taken verbatim from retrieval.log, not invented.
UNAVAILABLE_WORDINGS = (
    "2026-09-14 20:34:59,527 | WARNING | "
    "PMC6998867: no article version found",

    "2026-09-18 02:31:34,973 | WARNING | "
    "PMC6998867: no article version found during targeted retry",

    "2026-09-18 05:00:00,000 | WARNING | "
    "PMC6998867: no article version found; "
    "recording unavailable_current_dataset",
)


class UnavailableEvidenceTests(unittest.TestCase):
    """Behavioural: feed a synthetic log through the real reader."""

    def recover(self, *warning_lines):
        """Run the real evidence reader over a one-run synthetic log."""
        module = load_module()
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "retrieval.log"
            body = [
                "2026-01-01 00:00:00,000 | INFO | Stage 02 started.",
                "2026-01-01 00:00:01,000 | INFO | "
                "PMC Open Access records matched: 3",
                "2026-01-01 00:00:02,000 | INFO | "
                "Processing PMC record 1/3: PMC6998867",
                *warning_lines,
            ]
            log_path.write_text("\n".join(body) + "\n", encoding="utf-8")
            module.LOG_FILE = log_path
            _, _, unavailable, _, _ = module.extract_full_run_evidence(
                quiet_logger())
            return unavailable

    def test_every_historical_wording_is_recognised(self):
        for line in UNAVAILABLE_WORDINGS:
            with self.subTest(line=line):
                self.assertEqual(self.recover(line), {"PMC6998867"})

    def test_an_unrelated_line_yields_no_unavailable_record(self):
        """Reading more wordings must not mean reading anything at all."""
        for line in (
            "2026-01-01 00:00:03,000 | INFO | "
            "Processing PMC record 2/3: PMC7000000",
            "2026-01-01 00:00:03,000 | INFO | "
            "PMC7000000.1: article version is not marked PMC Open Access",
            "2026-01-01 00:00:03,000 | INFO | "
            "PMC Open Access records matched: 112260",
        ):
            with self.subTest(line=line):
                self.assertEqual(self.recover(line), set())


class RealLogEvidenceTests(unittest.TestCase):
    """Against the committed retrieval.log - the actual failing artifact."""

    @classmethod
    def setUpClass(cls):
        if not LOG.exists():
            raise unittest.SkipTest("retrieval.log is not present")
        cls.module = load_module()
        cls.log = quiet_logger()

    def test_the_selected_run_is_the_full_corpus_run(self):
        matched, processed, _, _, _ = self.module.extract_full_run_evidence(
            self.log)
        self.assertEqual(matched, 112260)
        self.assertEqual(len(processed), 112260,
                         "every matched PMCID should be recoverable")

    def test_unavailable_pmcids_are_recovered_from_the_historical_log(self):
        """The regression: this returned 0 before the fix."""
        _, _, unavailable, _, _ = self.module.extract_full_run_evidence(
            self.log)
        self.assertEqual(len(unavailable), 102)
        self.assertIn("PMC6998867", unavailable)

    def test_repair_evidence_is_still_read_correctly(self):
        """The fix must not disturb the repair reader beside it."""
        non_oa, stale = self.module.extract_repair_evidence(self.log)
        self.assertEqual(len(non_oa), 693)
        self.assertEqual(len(stale), 9)

    def test_the_accounting_closes(self):
        """Unaccounted PMCIDs are what made finalization refuse.

        112260 processed = 112161 with local evidence + 99 that remained
        unavailable after the targeted retry recovered 3 of the 102.
        """
        matched, _, unavailable, _, _ = self.module.extract_full_run_evidence(
            self.log)
        recovered_by_retry = 3
        still_unavailable = len(unavailable) - recovered_by_retry
        self.assertEqual(still_unavailable, 99)
        self.assertEqual(matched - still_unavailable, 112161)


if __name__ == "__main__":
    unittest.main()
