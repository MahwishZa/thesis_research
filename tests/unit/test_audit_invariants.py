"""Invariants the Stage-2 methodological decisions depend on.

Each test here locks a decision that would be easy to undo silently. They are
kept together so the reason for each one stays visible.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.test_pairs.scripts import build_pairs, power, validate_external
from experiments.test_pairs.scripts.schema import (
    EXCLUSION_REASONS,
    POOL_PRIMARY_EXTERNAL,
    EvidenceRef,
    Provenance,
    SchemaError,
    TestPair,
)


def pair(**kwargs):
    defaults = dict(
        pair_id="P-1",
        question_id="Q-1",
        question_text="Does X help?",
        provenance=Provenance(
            pool=POOL_PRIMARY_EXTERNAL,
            question_source="ExternalQA",
            answer_source="ExternalQA",
            evidence_source="ExternalQA",
            extraction_date="2026-09-15",
        ),
        older=EvidenceRef(evidence_id="E-old", document_id="D-1",
                          publication_date="2019-05"),
        newer=EvidenceRef(evidence_id="E-new", document_id="D-2",
                          publication_date="2024-03"),
        question_date="2026-09",
        reference_answer="No",
    )
    defaults.update(kwargs)
    return TestPair(**defaults)


class QuestionDateIndependenceTests(unittest.TestCase):
    """Decision 2: t_q never comes from the evidence under test."""

    def test_schema_rejects_evidence_derived_question_date(self):
        for source in ("derived:newer_publication_date",
                       "derived:older_publication_date"):
            with self.subTest(source=source), self.assertRaises(SchemaError):
                pair(question_date_source=source)

    def test_dataset_and_config_sources_are_accepted(self):
        for source in ("ExternalQA:question_date",
                       "config:evaluation_as_of_date"):
            with self.subTest(source=source):
                self.assertEqual(
                    pair(question_date_source=source).question_date_source,
                    source,
                )


class ClaimClassIsolationTests(unittest.TestCase):
    """Decision 3: corpus claim labels cannot reach the primary pool."""

    def test_external_claim_class_is_never_heuristic(self):
        records = [{
            "question_id": "Q-1", "question_text": "Does X help?",
            "reference_answer": "No",
            "older_document_id": "D-1", "newer_document_id": "D-2",
            "older_text": "old", "newer_text": "new",
            "older_publication_date": "2019-05",
            "newer_publication_date": "2024-03",
        }]
        built = build_pairs.build_external_pairs(
            records, dataset_name="ExternalQA", evidence_source="ExternalQA",
            extraction_date="2026-09-15", evaluation_as_of_date="2026-09",
        )
        self.assertIsNone(built[0].claim_class)
        self.assertEqual(built[0].claim_class_source, "unknown")

    def test_claim_class_absence_does_not_exclude_a_primary_pair(self):
        # If a missing claim class could exclude, the heuristic labels would
        # be an inclusion criterion by the back door.
        self.assertNotIn("unsupported_claim_class",
                         pair(claim_class=None).exclusion_reasons)

    def test_curated_claim_labels_carry_their_provenance(self):
        chunks = [
            {"chunk_id": "A#0", "document_id": "A", "publication_date": "2019-05",
             "claim_classes": ["AD-X"], "text": "a"},
            {"chunk_id": "B#0", "document_id": "B", "publication_date": "2024-03",
             "claim_classes": ["AD-X"], "text": "b"},
        ]
        built = build_pairs.build_curated_candidates(
            chunks, corpus_source="corpus@test", extraction_date="2026-09-15",
        )
        for item in built:
            self.assertEqual(item.claim_class_source, "heuristic")
            # ... and can never be mistaken for verified equivalence.
            self.assertIn("unverified_claim_equivalence",
                          item.exclusion_reasons)


class PilotSizingTests(unittest.TestCase):
    """Decision 4: the primary pool is used in full, not sampled."""

    def test_per_pool_sizing(self):
        configured = {"primary_external": None, "secondary_curated": 40}
        self.assertIsNone(
            build_pairs.resolve_pilot_size(configured, POOL_PRIMARY_EXTERNAL)
        )
        self.assertEqual(
            build_pairs.resolve_pilot_size(configured, "secondary_curated"), 40
        )

    def test_shipped_config_does_not_truncate_the_primary_pool(self):
        config = build_pairs.load_config(build_pairs.DEFAULT_CONFIG_PATH)
        self.assertIsNone(
            build_pairs.resolve_pilot_size(
                config["pilot"]["size"], POOL_PRIMARY_EXTERNAL
            ),
            "the primary pool must be evaluated in full (D-23)",
        )

    def test_plain_integer_still_applies_to_both_pools(self):
        self.assertEqual(build_pairs.resolve_pilot_size(40, "anything"), 40)

    def test_bad_sizing_value_is_refused(self):
        with self.assertRaises(ValueError):
            build_pairs.resolve_pilot_size("forty", POOL_PRIMARY_EXTERNAL)


class ExclusionVocabularyTests(unittest.TestCase):

    def test_no_dead_or_duplicate_reasons(self):
        self.assertEqual(len(EXCLUSION_REASONS), len(set(EXCLUSION_REASONS)))
        # Removed: it described a programming error, which the schema raises
        # on, not a property of an item that could be counted in attrition.
        self.assertNotIn("question_date_not_independent", EXCLUSION_REASONS)


class ExactPowerTests(unittest.TestCase):
    """Decision 4: sizing is exact, matching the exact test it sizes for."""

    def test_required_count_actually_reaches_nominal_power(self):
        for share in (0.60, 0.65, 0.70, 0.80):
            with self.subTest(share=share):
                result = power.required_pairs(
                    discordant_rate=0.3, older_share=share
                )
                self.assertGreaterEqual(
                    power.exact_power(result.discordant_pairs_needed,
                                      share, 0.05),
                    0.80,
                )

    def test_one_fewer_pair_would_not_suffice(self):
        result = power.required_pairs(discordant_rate=0.3, older_share=0.7)
        self.assertLess(
            power.exact_power(result.discordant_pairs_needed - 1, 0.7, 0.05),
            0.80,
        )

    def test_detectable_effect_clears_the_target(self):
        for total, rate in ((150, 0.4), (300, 0.3), (512, 0.25)):
            with self.subTest(total=total, rate=rate):
                share = power.detectable_effect(
                    total_pairs=total, discordant_rate=rate
                )
                self.assertGreaterEqual(
                    power.exact_power(int(total * rate), share, 0.05), 0.80
                )

    def test_null_split_has_no_power(self):
        self.assertLess(power.exact_power(100, 0.5, 0.05), 0.06)

    def test_tiny_counts_have_no_rejection_region(self):
        # With 4 discordant pairs no two-sided region reaches alpha = 0.05.
        self.assertIsNone(power._critical_value(4, 0.05))
        self.assertEqual(power.exact_power(4, 0.99, 0.05), 0.0)


class ExternalValidatorTests(unittest.TestCase):

    RECORD = {
        "question_id": "Q-1", "question_text": "Does X help?",
        "reference_answer": "No",
        "older_document_id": "D-1", "newer_document_id": "D-2",
        "older_text": "old", "newer_text": "new",
        "older_publication_date": "2019-05",
        "newer_publication_date": "2024-03",
    }

    def write(self, tmp, records):
        path = Path(tmp) / "external.jsonl"
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
            encoding="utf-8",
        )
        return path

    def test_complete_file_is_usable_and_digested(self):
        with TemporaryDirectory() as tmp:
            path = self.write(tmp, [self.RECORD])
            report = validate_external.run(
                input_path=path, dataset_name="ExternalQA v1",
                output_path=Path(tmp) / "acq.json",
            )
            self.assertTrue(report["usable"])
            self.assertEqual(report["records"], 1)
            self.assertEqual(len(report["sha256"]), 64)
            self.assertTrue((Path(tmp) / "acq.json").exists())

    def test_missing_required_field_makes_the_file_unusable(self):
        broken = dict(self.RECORD)
        broken.pop("reference_answer")
        with TemporaryDirectory() as tmp:
            report = validate_external.run(
                input_path=self.write(tmp, [broken]),
                dataset_name="ExternalQA v1",
            )
            self.assertFalse(report["usable"])
            self.assertIn("reference_answer", report["missing_required"])

    def test_cost_of_absent_optional_fields_is_reported_in_advance(self):
        thin = dict(self.RECORD)
        thin.pop("older_text")
        thin.pop("change_point_date", None)
        with TemporaryDirectory() as tmp:
            report = validate_external.run(
                input_path=self.write(tmp, [thin]),
                dataset_name="ExternalQA v1",
            )
            self.assertTrue(report["usable"])
            self.assertIn("older_text", report["anticipated_exclusions"])
            self.assertIn("missing_evidence_text",
                          report["anticipated_exclusions"]["older_text"])

    def test_duplicate_question_ids_are_surfaced(self):
        with TemporaryDirectory() as tmp:
            report = validate_external.run(
                input_path=self.write(tmp, [self.RECORD, dict(self.RECORD)]),
                dataset_name="ExternalQA v1",
            )
            self.assertEqual(report["duplicate_question_ids"], ["Q-1"])

    def test_absent_file_names_the_acquisition_document(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                validate_external.run(
                    input_path=Path(tmp) / "nope.jsonl",
                    dataset_name="ExternalQA v1",
                )
            self.assertIn("external_evaluation_data.md", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
