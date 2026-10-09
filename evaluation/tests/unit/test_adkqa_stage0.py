"""Stage 0 of the Alzheimer's-specific evaluation: the counting script, its gate and its probe. No network is used: the
E-utilities client takes an injected opener, exactly as in the tests of ``pubmed_asof``."""

import contextlib
import io
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from experiments.adkqa import stage0 as S
from experiments.medchange.pubmed_asof import EUtils


def fake_eutils(universe):
    """An EUtils whose responses come from ``universe(term, mindate, maxdate) -> list of ids``."""
    def opener(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        ids = universe(q["term"][0], q["mindate"][0], q["maxdate"][0])
        return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids}}).encode()
    return EUtils(opener=opener, sleep=lambda s: None)


def make_universe(n_base=1200, per_qualifier=150):
    """Base records 1..n_base; every qualifier query returns its own block of ``per_qualifier`` records."""
    base = [str(100000 + i) for i in range(n_base)]
    qualifiers = [q for qs in {**S.CORE_AREAS, **S.OPTIONAL_AREAS}.values() for q in qs]

    def universe(term, mindate, maxdate):
        for k, q in enumerate(qualifiers):
            if f'"Alzheimer Disease/{q}"[majr]' in term:
                return base[k * per_qualifier:(k + 1) * per_qualifier] if (k + 1) * per_qualifier <= n_base else base[-per_qualifier:]
        for t in S.TYPES:                       # one publication type per query: each takes a quarter of the records
            if term.count("[pt]") == 2 and f'"{t}"[pt]' in term.split("AND")[1]:
                i = S.TYPES.index(t)
                return base[i * n_base // 4:(i + 1) * n_base // 4]
        # period slices: the whole base set in the first slice only, so the sum equals the total
        return base if mindate == "2023/04/01" else []
    return universe


class TermTests(unittest.TestCase):
    def test_the_term_names_the_condition_the_types_and_the_filters(self):
        term = S.build_term()
        for needle in ('"Alzheimer Disease"[majr]', '"Systematic Review"[pt]', '"Meta-Analysis"[pt]',
                       '"Practice Guideline"[pt]', 'hasabstract[text]', 'medline[sb]', 'english[lang]',
                       'NOT "retracted publication"[pt]'):
            self.assertIn(needle, term)

    def test_a_qualifier_restricts_the_condition(self):
        term = S.build_term("diagnosis")
        self.assertIn('"Alzheimer Disease/diagnosis"[majr]', term)
        self.assertNotIn('"Alzheimer Disease"[majr]', term)

    def test_periods_cover_the_window_without_overlap(self):
        p = S.periods("2023-04-01", "2026-10-09")
        self.assertEqual(p[0], ("2023-04-01", "2023-12-31"))
        self.assertEqual(p[-1], ("2026-01-01", "2026-10-09"))
        self.assertEqual(len(p), 4)

    def test_the_sample_is_deterministic_and_inside_the_set(self):
        ids = [str(i) for i in range(500)]
        self.assertEqual(S.sample_pmids(ids), S.sample_pmids(list(reversed(ids))))
        self.assertTrue(set(S.sample_pmids(ids)) <= set(ids))
        self.assertEqual(len(S.sample_pmids(ids)), S.SAMPLE_SIZE)


class GateTests(unittest.TestCase):
    def test_go_needs_enough_records_and_enough_covered_areas(self):
        enough = {a: 120 for a in S.CORE_AREAS}
        self.assertTrue(S.evaluate_gate(900, enough)["go"])
        self.assertFalse(S.evaluate_gate(650, enough)["go"], "total below the minimum")
        thin = dict(enough, care_management=40, symptoms=99)
        g = S.evaluate_gate(900, thin)
        self.assertFalse(g["areas_ok"])
        self.assertEqual(g["areas_below"], ["care_management", "symptoms"])
        self.assertFalse(g["go"])

    def test_one_thin_area_is_reported_but_does_not_stop_the_build(self):
        g = S.evaluate_gate(900, dict({a: 120 for a in S.CORE_AREAS}, care_management=40))
        self.assertTrue(g["go"])
        self.assertEqual(g["areas_below"], ["care_management"])

    def test_the_optional_strata_do_not_count_toward_the_gate(self):
        g = S.evaluate_gate(900, dict({a: 20 for a in S.CORE_AREAS}, background_history=500, terminology=500))
        self.assertFalse(g["go"])


class CollectTests(unittest.TestCase):
    def test_distinct_records_per_area_are_unions_of_their_qualifiers(self):
        result = S.collect(fake_eutils(make_universe()), "2023-04-01", "2026-10-09")
        self.assertEqual(result["total"], 1200)
        self.assertEqual(result["per_area"]["treatment"]["distinct"], 300)       # two qualifiers, two disjoint blocks
        self.assertEqual(result["per_area"]["prevention"]["distinct"], 150)
        self.assertEqual(sum(result["by_period"].values()), result["total"])
        self.assertEqual(len(result["sample_pmids"]), S.SAMPLE_SIZE)

    def test_pubmed_warnings_and_the_query_translation_are_kept(self):
        def opener(url):
            body = {"esearchresult": {"count": "1", "idlist": ["7"], "querytranslation": "(x[All Fields])",
                                      "warninglist": {"phrasesignored": ["hasabstract"]}}}
            return json.dumps(body).encode()
        ids, notes = S.search(EUtils(opener=opener, sleep=lambda s: None), "x", "2023-04-01", "2026-10-09")
        self.assertEqual(ids, ["7"])
        self.assertEqual(notes["warninglist"], {"phrasesignored": ["hasabstract"]})
        self.assertEqual(notes["querytranslation"], "(x[All Fields])")

    def test_a_truncated_search_is_refused(self):
        def opener(url):
            return json.dumps({"esearchresult": {"count": "20000", "idlist": ["1"] * 10000}}).encode()
        with self.assertRaises(RuntimeError):
            S.search_ids(EUtils(opener=opener, sleep=lambda s: None), "x", "2023-04-01", "2026-10-09")


class ReferenceProbeTests(unittest.TestCase):
    def test_reference_lists_are_counted_and_failures_are_reported(self):
        def opener(url):
            if "/MED/1/" in url:
                refs = [{"id": str(i), "source": "MED"} for i in range(30)] + [{"id": "x", "source": "PMC"}]
                return json.dumps({"referenceList": {"reference": refs}}).encode()
            if "/MED/2/" in url:
                return json.dumps({"hitCount": 0}).encode()
            raise OSError("unreachable")
        probe = S.probe_references(["1", "2", "3"], opener)
        by = {r["pmid"]: r for r in probe["rows"]}
        self.assertEqual(by["1"]["with_pubmed_id"], 30)
        self.assertEqual(by["2"]["references"], 0)
        self.assertIn("OSError", by["3"]["error"])
        self.assertEqual(probe["with_at_least_10_pubmed_references"], 1)


class RevisionTwoTests(unittest.TestCase):
    def test_prevention_is_queried_with_the_words_pubmed_knows(self):
        self.assertEqual(tuple(S.CORE_AREAS["prevention"]), ("prevention and control",))
        self.assertNotIn("&", json.dumps(S.CORE_AREAS))

    def test_routine_pubmed_messages_are_dropped_and_real_ones_kept(self):
        self.assertEqual(S.meaningful({"warninglist": {"outputmessages": ["Search result exceeds limit"]}}), {})
        got = S.meaningful({"warninglist": {"quotedphrasesnotfound": ["x"]}, "errorlist": {"phrasesnotfound": ["y"]}})
        self.assertIn("quotedphrasesnotfound", got["warninglist"])
        self.assertIn("errorlist", got)

    def test_exclusive_supply_lets_a_record_serve_one_area_only(self):
        sets = {"a": {"1", "2", "3"}, "b": {"3"}, "c": {"1", "2", "3", "4"}}
        got = S.exclusive_counts(sets)
        self.assertEqual(got, {"a": 2, "b": 1, "c": 1})
        self.assertEqual(sum(got.values()), 4)

    def test_the_gate_thresholds_are_unchanged_from_the_declaration(self):
        self.assertEqual((S.MIN_TOTAL, S.MIN_AREA, S.MIN_AREAS), (700, 100, 6))


class MainTests(unittest.TestCase):
    def test_main_writes_counts_only_and_signals_go_or_no_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage0.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = S.main(["--since", "2023-04-01", "--until", "2026-10-09", "--out", str(out)],
                              eu=fake_eutils(make_universe(n_base=1200, per_qualifier=150)))
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertTrue(data["gate"]["go"])
            self.assertNotIn("abstract", json.dumps({k: v for k, v in data.items() if k != "base_term"}).lower())
            with contextlib.redirect_stdout(io.StringIO()):
                code = S.main(["--out", str(out)], eu=fake_eutils(make_universe(n_base=1200, per_qualifier=40)))
            self.assertEqual(code, 3, "no-go is signalled by the exit code")
            self.assertFalse(json.loads(out.read_text(encoding="utf-8"))["gate"]["go"])


if __name__ == "__main__":
    unittest.main()
