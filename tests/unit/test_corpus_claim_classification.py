"""Stage 07's taxonomy-schema alignment.

``config/claim_taxonomy.yaml`` was restructured to ``claim_types`` (a dict of
category -> {label, description, subtypes}) and ``evidence_levels`` (group ->
[study/source types]) at some point after the script was last run - the
committed ``claim_class_distribution.csv`` lists codes like ``AD-CRIT-04``
that appear nowhere in the current YAML. The old ``load_classes()`` read
``cfg["groups"]``, which no longer exists, and crashed with ``KeyError``
before writing anything.

These tests lock the rewrite against the real, current schema and against
the honesty requirements the module docstring states: every current class
uses a taxonomy-derived (not curated) keyword fallback, reported as such;
claim_status/temporal_status/disease_relevance are not tagged here.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "alzheimer_corpus" / "scripts" / "07_claim_classification.py"
REAL_TAXONOMY = ROOT / "alzheimer_corpus" / "config" / "claim_taxonomy.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("claim_classification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LoadDimensionTests(unittest.TestCase):

    def setUp(self):
        self.m = load_module()

    def test_claim_types_shape_builds_one_entry_per_category(self):
        cfg = {
            "disease_definition": {
                "label": "Disease definition",
                "description": "x",
                "subtypes": ["biological_definition", "clinical_definition"],
            },
            "epidemiology": {
                "label": "Epidemiology", "description": "y",
                "subtypes": ["prevalence"],
            },
        }
        classes = self.m.load_dimension(cfg, keyed_by_group=False)
        self.assertEqual({c["id"] for c in classes},
                         {"disease_definition", "epidemiology"})

    def test_claim_types_keywords_come_from_label_and_subtypes_only(self):
        cfg = {"amyloid": {"label": "Amyloid", "description": "x",
                           "subtypes": ["amyloid_deposition", "amyloid_pet"]}}
        classes = self.m.load_dimension(cfg, keyed_by_group=False)
        keywords = classes[0]["keywords"]
        self.assertIn("amyloid", keywords)
        self.assertIn("deposition", keywords)
        # Never invents vocabulary the taxonomy doesn't contain.
        self.assertNotIn("plaque", keywords)

    def test_no_current_class_is_marked_as_having_curated_keywords(self):
        """The schema has no explicit keyword field anywhere yet."""
        cfg = {"x": {"label": "X", "description": "", "subtypes": ["y_z"]}}
        classes = self.m.load_dimension(cfg, keyed_by_group=False)
        self.assertFalse(classes[0]["has_curated_keywords"])

    def test_evidence_levels_shape_flattens_groups(self):
        cfg = {
            "primary_research": ["randomized_controlled_trial", "cohort_study"],
            "authoritative": ["clinical_guideline"],
        }
        classes = self.m.load_dimension(cfg, keyed_by_group=True)
        self.assertEqual(len(classes), 3)
        ids = {c["id"] for c in classes}
        self.assertEqual(ids, {"randomized_controlled_trial", "cohort_study",
                               "clinical_guideline"})
        rct = next(c for c in classes if c["id"] == "randomized_controlled_trial")
        self.assertEqual(rct["group"], "primary_research")

    def test_short_ids_produce_no_keywords_rather_than_crashing(self):
        """A four-letter-or-shorter id/label has nothing to match on."""
        cfg = {"ad": {"label": "AD", "description": "", "subtypes": []}}
        classes = self.m.load_dimension(cfg, keyed_by_group=False)
        self.assertEqual(classes[0]["keywords"], [])


class KeywordMatchTests(unittest.TestCase):

    def setUp(self):
        self.m = load_module()

    def test_a_matching_passage_is_tagged_at_reduced_confidence(self):
        classes = [{"id": "amyloid", "keywords": ["amyloid", "deposition"],
                   "has_curated_keywords": False}]
        hits = self.m.keyword_match(
            "Amyloid deposition was measured by PET.", classes)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["claim_class"], "amyloid")
        self.assertEqual(hits[0]["method"], "keyword-from-taxonomy")
        self.assertLess(hits[0]["confidence"], 1.0)

    def test_curated_keywords_would_report_full_confidence_method(self):
        """Exercises the path a future curated keyword list would take -
        not reachable from the current taxonomy, but must not bit-rot."""
        classes = [{"id": "amyloid", "keywords": ["amyloid"],
                   "has_curated_keywords": True}]
        hits = self.m.keyword_match("Amyloid was present.", classes)
        self.assertEqual(hits[0]["method"], "keyword")

    def test_a_class_with_no_keywords_never_matches(self):
        classes = [{"id": "empty", "keywords": [], "has_curated_keywords": False}]
        self.assertEqual(self.m.keyword_match("any text at all", classes), [])

    def test_word_boundaries_are_respected(self):
        """'ad' as a token must not match inside 'radiology'."""
        classes = [{"id": "x", "keywords": ["ad"], "has_curated_keywords": False}]
        self.assertEqual(self.m.keyword_match("radiology report", classes), [])
        self.assertEqual(len(self.m.keyword_match("the AD diagnosis", classes)), 1)


class RealTaxonomyIntegrationTests(unittest.TestCase):
    """Against the actual committed config/claim_taxonomy.yaml."""

    def setUp(self):
        if not REAL_TAXONOMY.exists():
            self.skipTest("real taxonomy config not present")
        self.m = load_module()
        import sys
        sys.path.insert(0, str(ROOT / "alzheimer_corpus" / "scripts"))
        from _common import load_config
        self.cfg = load_config(str(REAL_TAXONOMY))

    def test_claim_types_loads_without_a_keyerror(self):
        """The regression: this used to be cfg['groups'] and crashed."""
        classes = self.m.load_dimension(self.cfg["claim_types"], keyed_by_group=False)
        self.assertGreater(len(classes), 20)

    def test_evidence_levels_loads_all_four_groups(self):
        classes = self.m.load_dimension(self.cfg["evidence_levels"], keyed_by_group=True)
        groups = {c["group"] for c in classes}
        self.assertEqual(groups, {"primary_research", "secondary_research",
                                  "authoritative", "reference"})

    def test_every_claim_type_has_at_least_one_keyword(self):
        """label + subtypes should always yield something to match on."""
        classes = self.m.load_dimension(self.cfg["claim_types"], keyed_by_group=False)
        empty = [c["id"] for c in classes if not c["keywords"]]
        self.assertEqual(empty, [],
                         f"claim types with no derivable keywords: {empty}")

    def test_a_realistic_passage_gets_tagged_without_crashing(self):
        classes = self.m.load_dimension(self.cfg["claim_types"], keyed_by_group=False)
        hits = self.m.keyword_match(
            "Donepezil is a cholinesterase inhibitor used to treat "
            "symptomatic Alzheimer disease.", classes)
        self.assertIsInstance(hits, list)

    def test_claim_status_and_temporal_status_are_not_tagged(self):
        """Both dimensions require cross-evidence comparison a single
        passage's keywords cannot provide - intentionally out of scope."""
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('cfg["claim_status"]', source)
        self.assertNotIn('cfg["temporal_status"]', source)

    def test_disease_relevance_is_not_recomputed(self):
        """Stage 04's assess_ad_relevance() already makes this call; a
        second, coarser classifier here would risk disagreeing with it."""
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('cfg["disease_relevance"]', source)


if __name__ == "__main__":
    unittest.main()
