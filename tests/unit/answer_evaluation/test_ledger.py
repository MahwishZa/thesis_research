"""answer_evaluation.ledger: a test set is scored once per system; a second scoring needs a written reason."""

import json
import unittest

from answer_evaluation import ledger
from answer_evaluation.ledger import AlreadyScored, Ledger
from tests.support import TempDirTestCase


class LedgerTests(TempDirTestCase):
    def make(self):
        counter = iter(range(1, 100))
        return Ledger(self.dir / "scored.jsonl", clock=lambda: f"2026-10-23T12:00:{next(counter):02d}+00:00")

    def claim(self, book, system="rag2", settings="s1", **changes):
        return book.claim(dataset="medqa", split="test", system=system, settings_sha256=settings, **changes)

    def test_the_first_claim_is_recorded(self):
        book = self.make()
        entry = self.claim(book)
        self.assertEqual((entry["run"], entry["finished"], entry["rerun_reason"]), (1, False, None))
        events = book.events()
        self.assertEqual([e["event"] for e in events], ["claim"])
        self.assertEqual(events[0]["claimed_at"], "2026-10-23T12:00:01+00:00")

    def test_a_finished_run_cannot_be_scored_again(self):
        book = self.make()
        self.claim(book)
        book.finish(dataset="medqa", split="test", system="rag2")
        with self.assertRaises(AlreadyScored) as caught:
            self.claim(book)
        self.assertIn("already scored", str(caught.exception))
        self.assertEqual(len(book.events()), 2)

    def test_a_stopped_run_continues_with_the_same_settings_without_a_new_entry(self):
        book = self.make()
        first = self.claim(book)
        again = self.claim(book)
        self.assertEqual(again["claimed_at"], first["claimed_at"])
        self.assertEqual(len(book.events()), 1)

    def test_a_started_run_with_other_settings_is_refused(self):
        book = self.make()
        self.claim(book, settings="s1")
        with self.assertRaises(AlreadyScored) as caught:
            self.claim(book, settings="s2")
        self.assertIn("other settings", str(caught.exception))

    def test_a_second_scoring_with_a_reason_is_allowed_and_the_reason_is_kept(self):
        book = self.make()
        self.claim(book)
        book.finish(dataset="medqa", split="test", system="rag2")
        entry = self.claim(book, settings="s2", rerun_reason="  a fault in the answer reader was found after the first run ")
        self.assertEqual(entry["run"], 2)
        self.assertEqual(entry["rerun_reason"], "a fault in the answer reader was found after the first run")
        self.assertEqual([e["event"] for e in book.events()], ["claim", "finish", "claim"])

    def test_the_rerun_is_then_itself_a_run_that_can_finish_and_is_refused_again(self):
        book = self.make()
        self.claim(book)
        book.finish(dataset="medqa", split="test", system="rag2")
        self.claim(book, rerun_reason="reason")
        book.finish(dataset="medqa", split="test", system="rag2")
        with self.assertRaises(AlreadyScored):
            self.claim(book)

    def test_a_blank_reason_is_refused(self):
        book = self.make()
        self.claim(book)
        for bad in ("", "   "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.claim(book, rerun_reason=bad)

    def test_other_systems_datasets_and_splits_are_separate(self):
        book = self.make()
        self.claim(book, system="rag2")
        book.finish(dataset="medqa", split="test", system="rag2")
        self.claim(book, system="no_retrieval")
        book.claim(dataset="mmlu_med", split="test", system="rag2", settings_sha256="s1")
        book.claim(dataset="medqa", split="validation", system="rag2", settings_sha256="s1")
        self.assertEqual(len(book.events()), 5)

    def test_finishing_twice_writes_one_event_and_finishing_without_a_claim_is_an_error(self):
        book = self.make()
        self.claim(book)
        book.finish(dataset="medqa", split="test", system="rag2")
        book.finish(dataset="medqa", split="test", system="rag2")
        self.assertEqual([e["event"] for e in book.events()], ["claim", "finish"])
        with self.assertRaises(ledger.LedgerError):
            book.finish(dataset="medqa", split="test", system="nobody")

    def test_empty_names_are_refused(self):
        book = self.make()
        for name in ("dataset", "split", "system", "settings_sha256"):
            arguments = {"dataset": "medqa", "split": "test", "system": "rag2", "settings_sha256": "s", name: ""}
            with self.subTest(name=name), self.assertRaises(ValueError):
                book.claim(**arguments)

    def test_a_damaged_file_is_an_error_and_a_missing_file_is_an_empty_ledger(self):
        self.assertEqual(Ledger(self.dir / "none.jsonl").events(), [])
        self.write("damaged.jsonl", '{"event": "claim"}\n{broken\n')
        with self.assertRaises(ledger.LedgerError):
            Ledger(self.dir / "damaged.jsonl").events()

    def test_the_file_is_append_only_json_lines(self):
        book = self.make()
        self.claim(book)
        self.claim(book, system="other")
        lines = (self.dir / "scored.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["system"] for line in lines], ["rag2", "other"])


if __name__ == "__main__":
    unittest.main()
