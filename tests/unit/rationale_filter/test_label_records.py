"""rationale_filter.label_records: the record, the store, τ over the pool and the labels that follow."""

import json
import unittest

from rationale_filter import label_records
from rationale_filter.label_records import PairRecord, PairStore
from rationale_query.store import ConfigMismatch, CorruptStore
from tests.spec.targets import DISCARD, HELPFUL, NOT_HELPFUL
from tests.support import TempDirTestCase


def ok(question="q1", rank=1, *, without=True, with_=True, ppl_without=10.0, ppl_with=8.0, **changes):
    values = {
        "id": label_records.pair_id(question, rank), "question_id": question, "rank": rank, "question_sha256": "a" * 64,
        "snippet_sha256": "b" * 64, "status": "ok", "correct_without": without, "correct_with": with_,
        "ppl_without": ppl_without, "ppl_with": ppl_with, "generation_with": "the answer is (A)", "finish_reason_with": "stop",
    }
    values.update(changes)
    return PairRecord(**values)


def skipped(question="q9", rank=1, reason="with_unreadable"):
    return PairRecord(
        id=label_records.pair_id(question, rank), question_id=question, rank=rank, question_sha256="a" * 64, snippet_sha256="b" * 64,
        status="skipped", reason=reason,
    )


class RecordTests(unittest.TestCase):
    def test_a_record_survives_a_round_trip_through_json(self):
        record = ok(ppl_with=7.25)
        self.assertEqual(PairRecord.from_dict(json.loads(record.to_json())), record)

    def test_the_id_is_the_question_id_and_the_rank(self):
        self.assertEqual(label_records.pair_id("medqa:train:7", 3), "medqa:train:7#3")
        with self.assertRaises(ValueError):
            PairRecord.from_dict({**json.loads(ok().to_json()), "id": "other#1"})

    def test_refused_dictionaries(self):
        base = json.loads(ok().to_json())
        for name, change in {
            "unknown status": {"status": "maybe"},
            "correctness is a number": {"correct_with": 1},
            "correctness is missing": {"correct_without": None},
            "perplexity is zero": {"ppl_with": 0.0},
            "perplexity is not finite": {"ppl_without": float("nan")},
            "perplexity is a bool": {"ppl_without": True},
            "rank is text": {"rank": "1", "id": "q1#1"},
        }.items():
            with self.subTest(name), self.assertRaises(ValueError):
                PairRecord.from_dict({**base, **change})
        with self.assertRaises(TypeError):
            PairRecord.from_dict("text")

    def test_a_skipped_record_needs_one_of_the_two_reasons(self):
        base = json.loads(skipped().to_json())
        self.assertEqual(PairRecord.from_dict(base).reason, "with_unreadable")
        with self.assertRaises(ValueError):
            PairRecord.from_dict({**base, "reason": None})
        with self.assertRaises(ValueError):
            PairRecord.from_dict({**base, "reason": "because"})

    def test_an_error_record_needs_nothing_more(self):
        record = PairRecord(id="q#1", question_id="q", rank=1, question_sha256="a", snippet_sha256="b", status="error", error="Boom")
        self.assertEqual(PairRecord.from_dict(json.loads(record.to_json())).error, "Boom")


class StoreTests(TempDirTestCase):
    def test_records_are_appended_and_read_back_by_id(self):
        store = PairStore(self.dir / "run")
        store.bind({"dataset": "medqa"}, label_records.RELEVANT)
        store.load()
        store.append(ok("q1", 1))
        store.append(ok("q1", 2, ppl_with=9.0))
        loaded = PairStore(self.dir / "run").load()
        self.assertEqual(sorted(loaded), ["q1#1", "q1#2"])
        self.assertEqual(loaded["q1#2"].ppl_with, 9.0)

    def test_the_latest_record_of_a_pair_wins(self):
        store = PairStore(self.dir / "run")
        store.bind({}, ())
        store.load()
        store.append(PairRecord(id="q#1", question_id="q", rank=1, question_sha256="a", snippet_sha256="b", status="error", error="x"))
        store.append(ok("q", 1))
        self.assertEqual(PairStore(self.dir / "run").load()["q#1"].status, "ok")

    def test_another_setting_is_refused(self):
        store = PairStore(self.dir / "run")
        store.bind({"dataset": "medqa", "snippets_per_question": 4}, label_records.RELEVANT)
        with self.assertRaises(ConfigMismatch) as caught:
            PairStore(self.dir / "run").bind({"dataset": "medqa", "snippets_per_question": 5}, label_records.RELEVANT)
        self.assertIn("snippets_per_question", str(caught.exception))

    def test_a_damaged_line_is_refused(self):
        store = PairStore(self.dir / "run")
        store.bind({}, ())
        (self.dir / "run" / "pairs.jsonl").write_text('{"id": "q#1"}\n', encoding="utf-8")
        with self.assertRaises(CorruptStore):
            PairStore(self.dir / "run").load()

    def test_the_relevant_settings_are_the_ones_that_change_the_numbers(self):
        for name in ("generator", "scorer", "answer_prompt_sha256", "reader_pattern", "snippets_per_question", "dataset", "split"):
            self.assertIn(name, label_records.RELEVANT)


class LabelTests(unittest.TestCase):
    def pool(self):
        # Four pairs that went through the rule, with perplexity drops 5, 3, 1 and -2; the top quarter is the drop of 5.
        return [
            ok("a", ppl_without=15.0, ppl_with=10.0),
            ok("b", ppl_without=13.0, ppl_with=10.0),
            ok("c", ppl_without=11.0, ppl_with=10.0),
            ok("d", ppl_without=8.0, ppl_with=10.0),
        ]

    def test_tau_is_taken_over_all_pairs_that_went_through_the_rule(self):
        tau, labelled = label_records.label_records(self.pool())
        self.assertEqual(tau, 5.0)
        self.assertEqual([item.label for item in labelled], [HELPFUL, DISCARD, DISCARD, DISCARD])

    def test_a_changed_answer_decides_whatever_tau_is(self):
        pool = self.pool() + [ok("e", without=False, with_=True, ppl_without=5.0, ppl_with=9.0), ok("f", without=True, with_=False)]
        tau, labelled = label_records.label_records(pool)
        by_question = {item.record.question_id: item.label for item in labelled}
        self.assertEqual((by_question["e"], by_question["f"]), (HELPFUL, NOT_HELPFUL))
        self.assertEqual(tau, sorted([5.0, 3.0, 1.0, -2.0, -4.0, 2.0], reverse=True)[1])  # n = 6: the top two are kept

    def test_both_wrong_with_a_big_drop_is_not_helpful(self):
        pool = [ok("w", without=False, with_=False, ppl_without=20.0, ppl_with=10.0), ok("x", ppl_without=10.0, ppl_with=10.0)]
        tau, labelled = label_records.label_records(pool)
        self.assertEqual(labelled[0].label, NOT_HELPFUL)

    def test_skipped_and_error_records_are_not_in_the_pool(self):
        extra = [skipped(), PairRecord(id="z#1", question_id="z", rank=1, question_sha256="a", snippet_sha256="b", status="error", error="x")]
        tau_with, labelled_with = label_records.label_records(self.pool() + extra)
        tau_without, labelled_without = label_records.label_records(self.pool())
        self.assertEqual(tau_with, tau_without)
        self.assertEqual(len(labelled_with), 4)

    def test_a_mapping_is_labelled_in_the_order_of_its_ids(self):
        mapping = {record.id: record for record in reversed(self.pool())}
        _, labelled = label_records.label_records(mapping)
        self.assertEqual([item.record.question_id for item in labelled], ["a", "b", "c", "d"])

    def test_no_usable_pair_is_an_error(self):
        with self.assertRaises(ValueError):
            label_records.label_records([skipped()])
        with self.assertRaises(ValueError):
            label_records.label_records([])

    def test_another_fraction_changes_tau(self):
        tau, labelled = label_records.label_records(self.pool(), top_fraction=0.5)
        self.assertEqual(tau, 3.0)

    def test_the_summary_counts_everything(self):
        records = self.pool() + [skipped("s1", reason="without_unreadable"), skipped("s2")]
        tau, labelled = label_records.label_records(records)
        summary = label_records.summarise(records, labelled)
        self.assertEqual(summary["records"], 6)
        self.assertEqual(summary["status"], {"ok": 4, "skipped": 2, "error": 0})
        self.assertEqual(summary["skipped_reasons"], {"without_unreadable": 1, "with_unreadable": 1})
        self.assertEqual(summary["labels"], {HELPFUL: 1, NOT_HELPFUL: 0, DISCARD: 3})
        self.assertEqual(summary["helpful_share_of_kept"], 1.0)
        self.assertEqual(summary["questions"], 6)

    def test_the_summary_without_labels_has_no_label_counts(self):
        self.assertNotIn("labels", label_records.summarise([ok()]))


if __name__ == "__main__":
    unittest.main()
