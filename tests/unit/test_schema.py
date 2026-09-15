"""Stage-2 schema invariants."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.test_pairs.scripts.schema import (
    POOL_PRIMARY_EXTERNAL,
    POOL_SECONDARY_CURATED,
    ContradictionStatus,
    EvidenceRef,
    PairCategory,
    Provenance,
    SchemaError,
    TestPair,
    ValidationStatus,
    _parse_partial_date,
    assert_firewall,
    read_pairs,
    write_pairs,
)


def provenance(pool=POOL_PRIMARY_EXTERNAL):
    return Provenance(
        pool=pool,
        question_source="ExternalQA v1",
        answer_source="ExternalQA v1",
        evidence_source="alzheimer_corpus@test",
        extraction_date="2026-09-15",
    )


def pair(pair_id="P-1", question_id="Q-1", pool=POOL_PRIMARY_EXTERNAL, **kwargs):
    defaults = dict(
        pair_id=pair_id,
        question_id=question_id,
        question_text="Does X help?",
        provenance=provenance(pool),
        older=EvidenceRef(evidence_id="E-old", document_id="D-1",
                          publication_date="2019-05"),
        newer=EvidenceRef(evidence_id="E-new", document_id="D-2",
                          publication_date="2024-03"),
        question_date="2024-03",
        reference_answer="No",
    )
    defaults.update(kwargs)
    return TestPair(**defaults)


class ParseDateTests(unittest.TestCase):

    def test_precision_and_interval(self):
        cases = {
            "2024": ("2024-01-01", "2024-12-31", "year"),
            "2024-02": ("2024-02-01", "2024-02-29", "month"),
            "2023-02": ("2023-02-01", "2023-02-28", "month"),
            "2024-12": ("2024-12-01", "2024-12-31", "month"),
            "2024-07-09": ("2024-07-09", "2024-07-09", "day"),
        }
        for value, (start, end, precision) in cases.items():
            with self.subTest(value=value):
                earliest, latest, got = _parse_partial_date(value)
                self.assertEqual(earliest.isoformat(), start)
                self.assertEqual(latest.isoformat(), end)
                self.assertEqual(got, precision)

    def test_missing_and_malformed(self):
        for value in (None, "", "   "):
            self.assertEqual(_parse_partial_date(value)[2], "unknown")
        for value in ("not-a-date", "2024-13", "20xx"):
            self.assertIsNone(_parse_partial_date(value)[0])


class RequiredFieldTests(unittest.TestCase):

    def test_missing_identifiers_rejected(self):
        for field in ("pair_id", "question_id", "question_text"):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                pair(**{field: ""})

    def test_evidence_ref_requires_ids(self):
        with self.assertRaises(SchemaError):
            EvidenceRef(evidence_id="", document_id="D")
        with self.assertRaises(SchemaError):
            EvidenceRef(evidence_id="E", document_id="")

    def test_same_passage_on_both_sides_rejected(self):
        same = EvidenceRef(evidence_id="E", document_id="D")
        with self.assertRaises(SchemaError):
            pair(older=same, newer=same)

    def test_unknown_exclusion_reason_rejected(self):
        with self.assertRaises(SchemaError):
            pair(exclusion_reasons=("invented_reason",))

    def test_eligible_pair_cannot_carry_exclusions(self):
        with self.assertRaises(SchemaError):
            pair(temporal_eligible=True,
                 exclusion_reasons=("missing_question_date",))

    def test_invalid_enum_values_rejected(self):
        with self.assertRaises(SchemaError):
            pair(pair_category="sometimes")
        with self.assertRaises(SchemaError):
            pair(contradiction_status="probably")
        with self.assertRaises(SchemaError):
            pair(validation_status="looks_fine")

    def test_provenance_requires_known_pool_and_sources(self):
        with self.assertRaises(SchemaError):
            Provenance(pool="whatever", question_source="a", answer_source="b",
                       evidence_source="c", extraction_date="2026-09-15")
        with self.assertRaises(SchemaError):
            Provenance(pool=POOL_PRIMARY_EXTERNAL, question_source="",
                       answer_source="b", evidence_source="c",
                       extraction_date="2026-09-15")


class ExclusionTests(unittest.TestCase):

    def test_excluded_is_additive_and_clears_eligibility(self):
        base = pair(temporal_eligible=True)
        once = base.excluded("missing_question_date")
        twice = once.excluded("duplicate_evidence", "missing_question_date")

        self.assertFalse(twice.temporal_eligible)
        self.assertEqual(
            twice.exclusion_reasons,
            ("duplicate_evidence", "missing_question_date"),
        )
        self.assertFalse(twice.is_eligible)

    def test_is_eligible_requires_both_conditions(self):
        self.assertTrue(pair(temporal_eligible=True).is_eligible)
        self.assertFalse(pair(temporal_eligible=False).is_eligible)


class FirewallTests(unittest.TestCase):

    def test_single_pool_passes(self):
        pairs = [pair("P-1"), pair("P-2", question_id="Q-2")]
        self.assertEqual(assert_firewall(pairs), POOL_PRIMARY_EXTERNAL)

    def test_mixed_pools_rejected(self):
        pairs = [pair("P-1"), pair("P-2", pool=POOL_SECONDARY_CURATED)]
        with self.assertRaises(SchemaError) as ctx:
            assert_firewall(pairs)
        self.assertIn("firewall", str(ctx.exception).lower())

    def test_write_refuses_mixed_pools(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            with self.assertRaises(SchemaError):
                write_pairs(
                    [pair("P-1"), pair("P-2", pool=POOL_SECONDARY_CURATED)],
                    path,
                )

    def test_read_refuses_mixed_pools(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            write_pairs([pair("P-1")], path)
            mixed = pair("P-2", pool=POOL_SECONDARY_CURATED).to_dict()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(mixed, sort_keys=True) + "\n")
            with self.assertRaises(SchemaError):
                read_pairs(path)


class SerialisationTests(unittest.TestCase):

    def test_round_trip_preserves_every_field(self):
        original = pair(
            claim_class="AD-BIOM-01",
            claim_class_source="external",
            pair_category=PairCategory.CHANGED.value,
            contradiction_status=ContradictionStatus.EXTERNALLY_ESTABLISHED.value,
            contradiction_source="ExternalQA v1",
            validation_status=ValidationStatus.MANUALLY_VALIDATED.value,
            exclusion_reasons=("duplicate_evidence",),
            notes="checked by hand",
        )
        restored = TestPair.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)

    def test_serialisation_is_deterministic_and_sorted(self):
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            pairs = [pair("P-3", question_id="Q-3"), pair("P-1"),
                     pair("P-2", question_id="Q-2")]

            write_pairs(pairs, first)
            write_pairs(list(reversed(pairs)), second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            ids = [json.loads(line)["pair_id"] for line in
                   first.read_text().splitlines()]
            self.assertEqual(ids, ["P-1", "P-2", "P-3"])

    def test_duplicate_pair_ids_rejected(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SchemaError):
                write_pairs([pair("P-1"), pair("P-1")], Path(tmp) / "x.jsonl")

    def test_date_precision_is_recorded_in_output(self):
        record = pair().to_dict()
        self.assertEqual(record["older"]["date_precision"], "month")
        self.assertEqual(record["newer"]["date_precision"], "month")


if __name__ == "__main__":
    unittest.main()
