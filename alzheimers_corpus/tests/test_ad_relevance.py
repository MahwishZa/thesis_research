"""Tests for the Alzheimer's relevance gate.

The three scenarios named in the corpus specification (section 32) are encoded
directly as tests, because they are the acceptance criteria for the gate:

  1. vascular dementia, no meaningful AD relevance  -> EXCLUDE
  2. AD compared with Lewy body disease             -> INCLUDE (differential dx)
  3. amyloid/tau with an established AD relationship-> INCLUDE
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "alzheimers_corpus" / "src" / "classification"))
from ad_relevance import ADRelevanceClassifier, _fold  # noqa: E402


class ADRelevanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = ADRelevanceClassifier()

    # -- spec section 32 acceptance scenarios ------------------------------
    def test_spec_vascular_dementia_without_ad_is_excluded(self):
        d = self.clf.classify({
            "document_id": "S1",
            "title": "Cerebral small vessel disease and vascular dementia outcomes",
            "abstract": "We studied cognitive decline in vascular dementia patients.",
            "mesh_terms": ["Dementia, Vascular", "Cognitive Dysfunction"],
        })
        self.assertFalse(d.ad_relevant)
        self.assertEqual(d.exclusion_reason, "comparator_disease_no_ad_anchor")

    def test_spec_ad_vs_lewy_body_is_included(self):
        d = self.clf.classify({
            "document_id": "S2",
            "title": "Differentiating Alzheimer's disease from dementia with Lewy bodies",
            "abstract": "Clinical features distinguishing the two conditions.",
            "mesh_terms": ["Alzheimer Disease", "Lewy Body Disease"],
        })
        self.assertTrue(d.ad_relevant)
        self.assertIn("R5_differential_with_anchor", d.rules_fired)

    def test_spec_amyloid_with_ad_relationship_is_included(self):
        d = self.clf.classify({
            "document_id": "S3",
            "title": "Plasma p-tau217 and amyloid beta in Alzheimer disease",
            "abstract": "Aβ42 and p-tau217 concentrations were measured.",
            "mesh_terms": ["Alzheimer Disease", "Amyloid beta-Peptides"],
        })
        self.assertTrue(d.ad_relevant)
        self.assertIn("R4_pathology_with_anchor", d.rules_fired)

    # -- the corpus must not drift into generic dementia -------------------
    def test_generic_dementia_only_is_excluded(self):
        d = self.clf.classify({
            "document_id": "G1", "title": "Prevalence of dementia in rural populations",
            "abstract": "Cognitive impairment was assessed across cohorts.",
            "mesh_terms": ["Dementia"],
        })
        self.assertFalse(d.ad_relevant)
        self.assertEqual(d.exclusion_reason, "generic_dementia_no_ad_anchor")

    def test_pathology_without_ad_anchor_is_not_relevant(self):
        """Amyloid outside AD (e.g. cardiac amyloidosis) must not qualify."""
        d = self.clf.classify({
            "document_id": "P1", "title": "Amyloid beta aggregation kinetics in vitro",
            "abstract": "We characterised tau filament assembly.", "mesh_terms": [],
        })
        self.assertFalse(d.ad_relevant)
        self.assertEqual(d.rules_fired, [])

    # -- abbreviation disambiguation ---------------------------------------
    def test_bare_AD_without_longform_is_excluded(self):
        """'AD' also means atopic dermatitis / autosomal dominant."""
        d = self.clf.classify({
            "document_id": "A1", "title": "Topical therapy for AD in children",
            "abstract": "AD severity was scored at baseline.", "mesh_terms": [],
        })
        self.assertFalse(d.ad_relevant)
        self.assertEqual(d.exclusion_reason, "ambiguous_abbreviation_only")

    def test_AD_with_longform_present_is_relevant(self):
        d = self.clf.classify({
            "document_id": "A2", "title": "Alzheimer's disease biomarkers",
            "abstract": "AD progression was tracked over 24 months.", "mesh_terms": [],
        })
        self.assertTrue(d.ad_relevant)
        self.assertIn("R2_title_anchor", d.rules_fired)

    # -- anchors -----------------------------------------------------------
    def test_mesh_anchor_alone_is_decisive(self):
        d = self.clf.classify({"document_id": "M1", "title": "A cohort study",
                               "abstract": "Methods and results.",
                               "mesh_terms": ["Alzheimer Disease"]})
        self.assertTrue(d.ad_relevant)
        self.assertIn("R1_mesh_anchor", d.rules_fired)

    def test_mesh_subheading_and_major_topic_forms_match(self):
        for m in ("*Alzheimer Disease", "Alzheimer Disease/diagnosis", "*Alzheimer Disease/therapy"):
            with self.subTest(mesh=m):
                d = self.clf.classify({"document_id": "M2", "title": "x",
                                       "abstract": "y", "mesh_terms": [m]})
                self.assertTrue(d.ad_relevant, m)

    def test_unicode_apostrophe_variants_match(self):
        for t in ("Alzheimer's disease staging", "Alzheimer’s disease staging",
                  "Alzheimer disease staging"):
            with self.subTest(title=t):
                d = self.clf.classify({"document_id": "U1", "title": t,
                                       "abstract": "", "mesh_terms": []})
                self.assertTrue(d.ad_relevant, t)

    def test_empty_record_is_not_relevant_and_does_not_crash(self):
        d = self.clf.classify({"document_id": "E1"})
        self.assertFalse(d.ad_relevant)
        self.assertEqual(d.score, 0.0)

    # -- trace integrity ---------------------------------------------------
    def test_decision_carries_full_trace(self):
        d = self.clf.classify({
            "document_id": "T1", "title": "Alzheimer's disease and Lewy body disease",
            "abstract": "Amyloid beta pathology compared.",
            "mesh_terms": ["Alzheimer Disease"]})
        self.assertTrue(d.anchor_evidence)
        self.assertTrue(d.supporting_evidence)
        self.assertTrue(d.comparator_evidence)
        self.assertTrue(d.config_version)
        self.assertIn("document_id", d.to_dict())

    def test_excluded_records_still_carry_a_reason(self):
        d = self.clf.classify({"document_id": "R1", "title": "Vascular dementia cohort",
                               "abstract": "", "mesh_terms": []})
        self.assertIsNotNone(d.exclusion_fired)
        self.assertIsNotNone(d.exclusion_reason)

    # -- matching must not corrupt biomedical surface forms ----------------
    def test_fold_is_match_only_and_preserves_source_text(self):
        src = "Aβ42 and p-tau217 with APOE ε4"
        self.assertNotEqual(_fold(src), src)          # folding changes the match key
        d = self.clf.classify({"document_id": "B1",
                               "title": "Alzheimer disease", "abstract": src,
                               "mesh_terms": []})
        self.assertTrue(d.ad_relevant)                # and the record still scores


class ConfigIntegrityTest(unittest.TestCase):
    def test_source_tiers_define_no_ordering(self):
        """Authority ordering is a tested thesis variable (ablation A12)."""
        clf = ADRelevanceClassifier()
        self.assertIsNone(clf.sources["source_tiers"]["ordering"])

    def test_licensing_fails_closed(self):
        clf = ADRelevanceClassifier()
        self.assertFalse(clf.sources["licensing"]["default_redistribution_allowed"])

    def test_retracted_documents_are_not_deleted(self):
        clf = ADRelevanceClassifier()
        self.assertFalse(clf.corpus_cfg["retraction_handling"]["delete_retracted"])

    def test_chunking_uses_a_model_tokenizer_not_whitespace(self):
        clf = ADRelevanceClassifier()
        ch = clf.corpus_cfg["chunking"]
        self.assertEqual(ch["tokenizer"], "ncbi/MedCPT-Article-Encoder")
        self.assertEqual(ch["chunk_size_tokens"], 256)
        self.assertEqual(ch["overlap_tokens"], 32)
        self.assertEqual(ch["stride_tokens"], 224)
        self.assertEqual(ch["chunk_size_tokens"] - ch["overlap_tokens"], ch["stride_tokens"])

    def test_only_alzheimer_disease_is_an_anchor_tier(self):
        clf = ADRelevanceClassifier()
        tiers = clf.mesh["tiers"]
        self.assertFalse(tiers["primary"]["requires_co_occurrence"])
        for name in ("clinical_context", "pathology", "differential"):
            self.assertTrue(tiers[name]["requires_co_occurrence"], name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
