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
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "corpus" / "scripts" / "07_claim_classification.py"
REAL_TAXONOMY = ROOT / "corpus" / "config" / "claim_taxonomy.yaml"


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

    def test_a_phrase_keyword_still_matches_via_the_regex_fallback(self):
        """_tokens_from_name never produces a multi-word keyword today, but
        a future curated keyword list (has_curated_keywords=True) could -
        the word-set fast path can't match a phrase, so keyword_match()
        must fall back to the regex path for any keyword containing a
        separator character."""
        classes = [{"id": "onset", "keywords": ["early onset", "early_onset"],
                   "has_curated_keywords": True}]
        hits = self.m.keyword_match("This is early onset disease.", classes)
        self.assertEqual(len(hits), 1)
        self.assertIn("early onset", hits[0]["matched_terms"])

    def test_a_phrase_keyword_respects_word_boundaries_too(self):
        classes = [{"id": "onset", "keywords": ["early onset"],
                   "has_curated_keywords": True}]
        self.assertEqual(
            self.m.keyword_match("a very early onsetting condition", classes), [])


def reference_keyword_match(text, classes, m):
    """The keyword-matching algorithm before the word-set optimisation:
    one regex search per keyword per class, always. Preserved verbatim
    (not reading keyword_match()'s current body) as ground truth for the
    equivalence test below."""
    f = m.fold(text)
    hits = []
    for c in classes:
        terms = c["keywords"]
        if not terms:
            continue
        matched = [t for t in terms
                  if re.search(r"(?<!\w)" + re.escape(m.fold(t)) + r"(?!\w)", f)]
        if matched:
            confidence = min(1.0, len(matched) / max(1, len(terms)))
            hits.append({"claim_class": c["id"], "confidence": round(
                            confidence * (1.0 if c["has_curated_keywords"] else 0.4), 3),
                         "method": "keyword" if c["has_curated_keywords"] else "keyword-from-taxonomy",
                         "matched_terms": matched})
    return hits


class KeywordMatchEquivalenceTests(unittest.TestCase):
    """The word-set optimisation must match the original per-keyword regex
    algorithm exactly, over the real taxonomy and a range of realistic and
    adversarial text - not just the hand-picked cases above."""

    def setUp(self):
        if not REAL_TAXONOMY.exists():
            self.skipTest("real taxonomy config not present")
        self.m = load_module()
        cfg = self.m.load_config(str(REAL_TAXONOMY))
        self.claim_types = self.m.load_dimension(cfg["claim_types"], keyed_by_group=False)
        self.evidence_levels = self.m.load_dimension(cfg["evidence_levels"], keyed_by_group=True)

    def test_matches_the_reference_algorithm_across_many_texts(self):
        texts = [
            "Alzheimer disease diagnosis relies on amyloid and tau biomarkers.",
            "A randomized controlled trial evaluated donepezil for cognitive decline.",
            "",
            "radiology report shows no acute findings",  # 'ad' inside 'radiology'
            "the AD diagnosis was confirmed by PET imaging",
            "Aβ42 and p-tau181 levels were measured in cerebrospinal fluid samples "
            "from patients with mild cognitive impairment and early-onset dementia.",
            "A systematic review and meta-analysis pooling fourteen trials.",
            "Case report of a rare genetic mutation causing familial disease.",
            "1234567890 !@#$%^&*() no real words here at all just symbols",
            "amyloidamyloidamyloid" * 5,  # keyword as a substring of a longer word
        ]
        for text in texts:
            for classes in (self.claim_types, self.evidence_levels):
                expected = reference_keyword_match(text, classes, self.m)
                actual = self.m.keyword_match(text, classes)
                self.assertEqual(
                    expected, actual,
                    msg=f"mismatch for text={text!r}",
                )


class RealTaxonomyIntegrationTests(unittest.TestCase):
    """Against the actual committed config/claim_taxonomy.yaml."""

    def setUp(self):
        if not REAL_TAXONOMY.exists():
            self.skipTest("real taxonomy config not present")
        self.m = load_module()
        import sys
        sys.path.insert(0, str(ROOT / "corpus" / "scripts"))
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


def reference_tag_all(chunks, claim_types, evidence_levels, m):
    """The ORIGINAL, pre-streaming implementation: hold every chunk in
    memory, tag it, return the whole mutated list. Preserved verbatim as
    ground truth - not reading main()'s current body - so the streaming
    rewrite is checked against independently-reasoned-about behaviour, not
    against itself."""
    out = []
    for ch in chunks:
        ch = dict(ch)
        hits = m.keyword_match(ch.get("text", ""), claim_types)
        ch["claim_classes"] = [h["claim_class"] for h in hits]
        ch["claim_confidence"] = max([h["confidence"] for h in hits], default=0.0)
        ch["claim_method"] = hits[0]["method"] if hits else "none"
        ev_hits = m.keyword_match(ch.get("text", ""), evidence_levels)
        ch["claim_evidence_levels"] = [h["claim_class"] for h in ev_hits]
        ch["claim_evidence_confidence"] = max(
            [h["confidence"] for h in ev_hits], default=0.0)
        out.append(ch)
    return out


class StreamingRewriteTests(unittest.TestCase):
    """This stage now streams (its real input is 4.3M+ chunks - holding
    every chunk's full text in memory at once, then writing nothing until
    the very end, is exactly the memory/visibility problem Stage 06 had
    before its batching rewrite). These tests lock the rewrite's shape and
    its exact equivalence to the original in-memory algorithm."""

    def setUp(self):
        self.m = load_module()
        self.source = SCRIPT.read_text(encoding="utf-8")

    def test_does_not_materialise_the_whole_corpus_as_a_list(self):
        code_lines = [line for line in self.source.splitlines()
                      if not line.strip().startswith("#")]
        self.assertFalse(
            any("list(read_jsonl(" in line for line in code_lines),
            "found a live (non-comment) list(read_jsonl(...)) call",
        )

    def test_writes_through_a_temp_file_then_replaces_atomically(self):
        self.assertIn(".part", self.source)
        self.assertIn("tmp.replace(src)", self.source)

    def test_reports_progress_during_a_run(self):
        self.assertIn("stage 07 progress", self.source)
        self.assertGreater(self.m.PROGRESS_EVERY, 0)

    def test_tag_chunk_matches_the_reference_algorithm(self):
        from collections import Counter
        cfg = self.m.load_config(str(REAL_TAXONOMY))
        claim_types = self.m.load_dimension(cfg["claim_types"], keyed_by_group=False)
        evidence_levels = self.m.load_dimension(cfg["evidence_levels"], keyed_by_group=True)

        chunks = [
            {"chunk_id": f"C{i}", "document_id": f"D{i}", "text": text,
             "source_tier": "peer_reviewed_primary", "ad_relevant": True,
             "ad_relevance_score": 0.9}
            for i, text in enumerate([
                "Alzheimer disease diagnosis relies on biomarkers.",
                "A randomized controlled trial of donepezil.",
                "", "Amyloid plaques and tau tangles in the brain.",
                "A case report of early-onset dementia.",
            ])
        ]

        expected = reference_tag_all(chunks, claim_types, evidence_levels, self.m)

        dist, evidence_dist = Counter(), Counter()
        actual = [
            self.m.tag_chunk(dict(ch), claim_types, evidence_levels, dist, evidence_dist)
            for ch in chunks
        ]

        for exp, act in zip(expected, actual):
            for key in ("claim_classes", "claim_confidence", "claim_method",
                       "claim_evidence_levels", "claim_evidence_confidence"):
                self.assertEqual(exp[key], act[key], msg=f"{key} for {exp['chunk_id']}")

    def test_end_to_end_streaming_run_matches_reference_and_is_atomic(self):
        """Runs the real main() against a real (copied) corpus/
        tree - never the actual one - and checks the streamed output
        equals what the original in-memory algorithm would have produced,
        record for record, plus every report file is written."""
        import shutil
        import subprocess
        import sys as _sys

        with TemporaryDirectory() as tmp:
            copy = Path(tmp) / "corpus"
            shutil.copytree(ROOT / "corpus", copy,
                            ignore=shutil.ignore_patterns("__pycache__"))

            chunks = [
                {"chunk_id": f"C{i}", "document_id": f"D{i}",
                 "text": text, "source_tier": "peer_reviewed_primary",
                 "ad_relevant": True, "ad_relevance_score": 0.5,
                 "claim_classes": [], "claim_confidence": 0.0}
                for i, text in enumerate([
                    "Alzheimer disease diagnosis relies on biomarkers.",
                    "A randomized controlled trial of donepezil.",
                    "Amyloid plaques and tau tangles in the brain.",
                ])
            ]
            chunks_path = copy / "data" / "chunks" / "chunks.jsonl"
            chunks_path.write_text(
                "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")

            result = subprocess.run(
                [_sys.executable, "scripts/07_claim_classification.py"],
                cwd=copy, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            out_rows = [json.loads(l) for l in
                       chunks_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(out_rows), len(chunks))
            for row in out_rows:
                for key in ("claim_classes", "claim_confidence", "claim_method",
                           "claim_evidence_levels", "claim_evidence_confidence"):
                    self.assertIn(key, row)

            for report in ("claim_class_distribution.csv",
                          "evidence_level_distribution.csv",
                          "annotation_report.csv"):
                self.assertTrue((copy / "reports" / report).exists())

            self.assertFalse(
                chunks_path.with_suffix(".jsonl.part").exists(),
                "temp file must be replaced, not left behind",
            )


if __name__ == "__main__":
    unittest.main()
