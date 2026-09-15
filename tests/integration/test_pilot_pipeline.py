"""End-to-end Stage-2 pilot: build, audit, serialise, split."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.test_pairs.scripts import build_pairs
from experiments.test_pairs.scripts.schema import (
    POOL_PRIMARY_EXTERNAL,
    POOL_SECONDARY_CURATED,
    SchemaError,
    read_pairs,
)


CONFIG = {
    "corpus_snapshot": "corpus@test",
    "external_dataset": "ExternalQA v1",
    "pilot": {"size": 40, "seed": "test-pilot"},
    "eligibility": {"min_separation_days": None, "exclude_retracted": True},
    "split": {"seed": "test-split", "dev": 0.2, "val": 0.2, "test": 0.6},
    "label_model_cutoffs": {"test-model": "2023-12"},
}


EXTERNAL_RECORDS = [
    {
        "question_id": "Q-1",
        "question_text": "Does treatment X improve outcome Y?",
        "reference_answer": "No",
        "older_evidence_id": "E-1a", "older_document_id": "D-1",
        "older_publication_date": "2018-04", "older_source_tier": "journal",
        "newer_evidence_id": "E-1b", "newer_document_id": "D-2",
        "newer_publication_date": "2024-09", "newer_source_tier": "journal",
        "change_point_date": "2024-09",
    },
    {
        # Undated older side: must be excluded, not silently dated.
        "question_id": "Q-2",
        "question_text": "Is biomarker Z sufficient for diagnosis?",
        "reference_answer": "Yes",
        "older_evidence_id": "E-2a", "older_document_id": "D-3",
        "newer_evidence_id": "E-2b", "newer_document_id": "D-4",
        "newer_publication_date": "2025-01",
    },
]


CHUNKS = [
    {
        "chunk_id": "DOC-1#abstract.0", "document_id": "DOC-1",
        "publication_date": "2019-05", "source_tier": "peer_reviewed_primary",
        "claim_classes": ["AD-BIOM-01"], "text": "older claim text here",
    },
    {
        "chunk_id": "DOC-2#abstract.0", "document_id": "DOC-2",
        "publication_date": "2024-03", "source_tier": "peer_reviewed_primary",
        "claim_classes": ["AD-BIOM-01"], "text": "newer claim text here",
    },
]


def write_jsonl(path: Path, records) -> Path:
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


class ExternalPoolTests(unittest.TestCase):

    def run_pilot(self, tmp, records=EXTERNAL_RECORDS):
        source = write_jsonl(Path(tmp) / "external.jsonl", records)
        return build_pairs.run(
            pool=POOL_PRIMARY_EXTERNAL,
            input_path=source,
            output_dir=Path(tmp) / "out",
            config=CONFIG,
        ), Path(tmp) / "out"

    def test_pilot_runs_and_writes_expected_artifacts(self):
        with TemporaryDirectory() as tmp:
            manifest, out = self.run_pilot(tmp)

            self.assertEqual(manifest["pool"], POOL_PRIMARY_EXTERNAL)
            self.assertEqual(manifest["candidates_enumerated"], 2)
            self.assertTrue((out / "primary_external_pilot.jsonl").exists())
            self.assertTrue((out / "primary_external_attrition.csv").exists())
            self.assertTrue((out / "primary_external_manifest.json").exists())

    def test_well_dated_pair_survives_and_undated_one_does_not(self):
        with TemporaryDirectory() as tmp:
            _, out = self.run_pilot(tmp)
            pairs = {p.question_id: p
                     for p in read_pairs(out / "primary_external_pilot.jsonl")}

            self.assertTrue(pairs["Q-1"].is_eligible)
            self.assertFalse(pairs["Q-2"].is_eligible)
            self.assertIn("missing_publication_date",
                          pairs["Q-2"].exclusion_reasons)

    def test_question_date_falls_back_per_item_not_globally(self):
        with TemporaryDirectory() as tmp:
            _, out = self.run_pilot(tmp)
            pairs = {p.question_id: p
                     for p in read_pairs(out / "primary_external_pilot.jsonl")}

            self.assertEqual(pairs["Q-1"].question_date, "2024-09")
            self.assertEqual(pairs["Q-2"].question_date, "2025-01")
            for item in pairs.values():
                self.assertEqual(item.question_date_source,
                                 "derived:newer_publication_date")

    def test_missing_required_field_is_refused(self):
        broken = [dict(EXTERNAL_RECORDS[0])]
        broken[0].pop("reference_answer")
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SchemaError) as ctx:
                self.run_pilot(tmp, broken)
            self.assertIn("reference_answer", str(ctx.exception))

    def test_missing_input_file_names_the_dependency(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                build_pairs.run(
                    pool=POOL_PRIMARY_EXTERNAL,
                    input_path=Path(tmp) / "absent.jsonl",
                    output_dir=Path(tmp) / "out",
                    config=CONFIG,
                )
            self.assertIn("externally-authored", str(ctx.exception))

    def test_attrition_and_check_a_are_reported(self):
        with TemporaryDirectory() as tmp:
            manifest, _ = self.run_pilot(tmp)

            self.assertEqual(manifest["attrition"]["candidates"], 2)
            self.assertEqual(manifest["attrition"]["eligible"], 1)
            self.assertEqual(manifest["check_a"][0]["post_cutoff"], 2)
            self.assertEqual(manifest["separation_summary"]["n"], 1)

    def test_manifest_records_which_rules_were_enforced(self):
        with TemporaryDirectory() as tmp:
            manifest, _ = self.run_pilot(tmp)
            enforced = manifest["eligibility_rules_enforced"]
            self.assertIn("unambiguous_temporal_order", enforced)
            self.assertNotIn("min_separation_days", enforced)

    def test_run_is_reproducible(self):
        with TemporaryDirectory() as tmp:
            first, out_a = self.run_pilot(tmp)
            source = write_jsonl(Path(tmp) / "external2.jsonl",
                                 list(reversed(EXTERNAL_RECORDS)))
            out_b = Path(tmp) / "out2"
            second = build_pairs.run(
                pool=POOL_PRIMARY_EXTERNAL, input_path=source,
                output_dir=out_b, config=CONFIG,
            )

            self.assertEqual(first["attrition"], second["attrition"])
            self.assertEqual(
                (out_a / "primary_external_pilot.jsonl").read_text(),
                (out_b / "primary_external_pilot.jsonl").read_text(),
            )


class CuratedPoolTests(unittest.TestCase):

    def run_pilot(self, tmp):
        source = write_jsonl(Path(tmp) / "chunks.jsonl", CHUNKS)
        out = Path(tmp) / "out"
        return build_pairs.run(
            pool=POOL_SECONDARY_CURATED, input_path=source,
            output_dir=out, config=CONFIG,
        ), out

    def test_corpus_only_candidates_are_never_eligible(self):
        with TemporaryDirectory() as tmp:
            manifest, out = self.run_pilot(tmp)

            self.assertEqual(manifest["attrition"]["eligible"], 0)
            for pair in read_pairs(out / "secondary_curated_pilot.jsonl"):
                self.assertFalse(pair.is_eligible)
                self.assertIn("missing_reference_answer",
                              pair.exclusion_reasons)
                self.assertIn("insufficient_contradiction",
                              pair.exclusion_reasons)

    def test_heuristic_claim_labels_are_marked_as_such(self):
        with TemporaryDirectory() as tmp:
            _, out = self.run_pilot(tmp)
            for pair in read_pairs(out / "secondary_curated_pilot.jsonl"):
                self.assertEqual(pair.claim_class_source, "heuristic")

    def test_older_and_newer_are_ordered_by_date(self):
        with TemporaryDirectory() as tmp:
            _, out = self.run_pilot(tmp)
            for pair in read_pairs(out / "secondary_curated_pilot.jsonl"):
                self.assertEqual(pair.older.document_id, "DOC-1")
                self.assertEqual(pair.newer.document_id, "DOC-2")

    def test_pools_are_written_to_separate_files(self):
        with TemporaryDirectory() as tmp:
            _, out = self.run_pilot(tmp)
            external = write_jsonl(Path(tmp) / "external.jsonl",
                                   EXTERNAL_RECORDS)
            build_pairs.run(pool=POOL_PRIMARY_EXTERNAL, input_path=external,
                            output_dir=out, config=CONFIG)

            curated = out / "secondary_curated_pilot.jsonl"
            primary = out / "primary_external_pilot.jsonl"
            self.assertTrue(curated.exists() and primary.exists())
            # Each file reads back cleanly, which the firewall check enforces.
            self.assertEqual(
                {p.provenance.pool for p in read_pairs(curated)},
                {POOL_SECONDARY_CURATED},
            )
            self.assertEqual(
                {p.provenance.pool for p in read_pairs(primary)},
                {POOL_PRIMARY_EXTERNAL},
            )


if __name__ == "__main__":
    unittest.main()
