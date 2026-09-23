"""Tests for the question-source adapters.

The source files are not vendored, so every test builds a synthetic record in
the real upstream format. That keeps the suite runnable anywhere and still
exercises the parsing, the anchoring rule and the provenance fields.
"""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.shared.questions import build_pool
from experiments.shared.questions.sources import cochrane, nih_medquad

CITATION = ("Cochrane Database Syst Rev. 2021 Dec 17;12(12):CD013304. "
            "doi: 10.1002/14651858.CD013304.pub2.")


def row(**kw):
    base = {
        "Question": "Are antipsychotics effective for agitation in Alzheimer disease?",
        "objectives": "To assess antipsychotics in Alzheimer disease.",
        "background": "Alzheimer disease commonly causes agitation.",
        "conclusions": ("We included 24 trials. There is some evidence that "
                        "typical antipsychotics slightly decrease agitation."),
        "Label": "SUPPORTED",
        "DOI_Date": CITATION,
        "PMID": "/34918337/",
        "Author": "Someone A.",
    }
    base.update(kw)
    return base


def write_csv(tmp, rows):
    path = Path(tmp) / "medrevqa.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class BottomLineTests(unittest.TestCase):

    def test_methodological_preamble_is_skipped(self):
        text = "We included 24 trials. Antipsychotics reduce agitation slightly."
        self.assertEqual(cochrane.bottom_line(text),
                         "Antipsychotics reduce agitation slightly.")

    def test_restated_heading_is_skipped(self):
        text = ("How is this condition inherited? "
                "It follows an autosomal dominant inheritance pattern here.")
        self.assertTrue(cochrane.bottom_line(text).startswith("It follows"))

    def test_entirely_methodological_conclusion_yields_nothing(self):
        self.assertEqual(cochrane.bottom_line("We included 3 trials."), "")

    def test_long_sentence_is_truncated_not_dropped(self):
        long = "Antipsychotics " + "reduce agitation " * 40 + "in patients."
        out = cochrane.bottom_line(long, limit=100)
        self.assertLessEqual(len(out), 104)
        self.assertTrue(out.endswith("..."))

    def test_empty_input(self):
        self.assertEqual(cochrane.bottom_line(""), "")


class CochraneAdapterTests(unittest.TestCase):

    def test_emits_candidate_with_full_provenance(self):
        with TemporaryDirectory() as tmp:
            [q] = list(cochrane.candidates(write_csv(tmp, [row()])))
        self.assertTrue(q.AD_anchor)
        self.assertTrue(q.determinate)
        self.assertEqual(q.status, "candidate")
        self.assertEqual(q.reference_date, "2021-12")
        self.assertIn("PMID:34918337", q.reference_locator)
        self.assertIn("CD013304.pub2", q.reference_locator)
        self.assertIn("doi:10.1002/14651858.CD013304.pub2", q.reference_locator)
        self.assertEqual(q.metadata["reference_source_type"],
                         "peer_reviewed_evidence_synthesis")
        self.assertTrue(q.metadata["verification_required"])

    def test_update_suffix_marks_a_temporal_candidate(self):
        with TemporaryDirectory() as tmp:
            [q] = list(cochrane.candidates(write_csv(tmp, [row()])))
        self.assertTrue(q.temporal_candidate)

    def test_first_edition_is_not_a_temporal_candidate(self):
        first = CITATION.replace(".pub2", "")
        with TemporaryDirectory() as tmp:
            [q] = list(cochrane.candidates(
                write_csv(tmp, [row(DOI_Date=first)])))
        self.assertFalse(q.temporal_candidate)

    def test_dementia_without_an_alzheimer_anchor_is_excluded(self):
        bare = row(
            Question="Is aromatherapy effective for people with dementia?",
            objectives="To assess aromatherapy in dementia.",
            background="Dementia affects many older people.",
            conclusions="Aromatherapy showed no clear benefit in this review.",
        )
        with TemporaryDirectory() as tmp:
            self.assertEqual(
                list(cochrane.candidates(write_csv(tmp, [bare]))), [])

    def test_anchor_may_come_from_elsewhere_in_the_record(self):
        anchored = row(
            Question="Is aromatherapy effective for people with dementia?",
            objectives="To assess aromatherapy in dementia.",
            background="Most dementia is due to Alzheimer disease.",
            conclusions="Aromatherapy showed no clear benefit in this review.",
        )
        with TemporaryDirectory() as tmp:
            [q] = list(cochrane.candidates(write_csv(tmp, [anchored])))
        self.assertTrue(q.AD_anchor)

    def test_not_enough_information_is_flagged_ambiguous(self):
        with TemporaryDirectory() as tmp:
            [q] = list(cochrane.candidates(
                write_csv(tmp, [row(Label="NOT ENOUGH INFORMATION")])))
        self.assertTrue(q.ambiguity_candidate)

    def test_missing_source_file_names_the_remedy(self):
        with self.assertRaises(cochrane.SourceUnavailable) as ctx:
            list(cochrane.candidates(Path("/nonexistent/medrevqa.csv")))
        self.assertIn("jvladika/MedChange", str(ctx.exception))

    def test_topics_are_assigned_deterministically(self):
        with TemporaryDirectory() as tmp:
            path = write_csv(tmp, [row()])
            a = [q.topic for q in cochrane.candidates(path)]
            b = [q.topic for q in cochrane.candidates(path)]
        self.assertEqual(a, b)


def write_medquad(tmp, *, cui="C0002395", qtype="treatment", answer=None):
    root = Path(tmp) / "6_NINDS_QA"
    root.mkdir(parents=True)
    body = answer if answer is not None else (
        "Treatment focuses on managing symptoms with cholinesterase inhibitors.")
    (root / "0000001.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Document id="0000001" source="NINDS" url="http://example.nih.gov/ad">\n'
        "<Focus>Alzheimer's Disease</Focus>\n"
        f"<FocusAnnotations><UMLS><CUIs><CUI>{cui}</CUI></CUIs>"
        "</UMLS></FocusAnnotations>\n"
        "<QAPairs><QAPair pid=\"1\">"
        f'<Question qid="0000001-1" qtype="{qtype}">'
        "What are the treatments for Alzheimer's Disease ?</Question>"
        f"<Answer>{body}</Answer>"
        "</QAPair></QAPairs>\n</Document>\n",
        encoding="utf-8")
    return Path(tmp)


class MedQuADAdapterTests(unittest.TestCase):

    def test_emits_candidate_with_cui_and_url_locator(self):
        with TemporaryDirectory() as tmp:
            [q] = list(nih_medquad.candidates(write_medquad(tmp),
                                              retrieved_on="2026-09"))
        self.assertTrue(q.AD_anchor)
        self.assertIn("CUI:C0002395", q.reference_locator)
        self.assertIn("example.nih.gov", q.reference_locator)
        self.assertEqual(q.metadata["reference_source_type"],
                         "government_public_health")

    def test_retrieval_date_is_flagged_as_not_a_publication_date(self):
        with TemporaryDirectory() as tmp:
            [q] = list(nih_medquad.candidates(write_medquad(tmp),
                                              retrieved_on="2026-09"))
        self.assertEqual(q.reference_date, "2026-09")
        self.assertTrue(q.metadata["reference_date_is_retrieval_date"])

    def test_non_alzheimer_cui_is_excluded(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(
                list(nih_medquad.candidates(write_medquad(tmp, cui="C0011265"),
                                            retrieved_on="2026-09")), [])

    def test_service_question_types_are_excluded(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(
                list(nih_medquad.candidates(
                    write_medquad(tmp, qtype="support groups"),
                    retrieved_on="2026-09")), [])

    def test_stripped_answer_is_skipped_not_reconstructed(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(
                list(nih_medquad.candidates(write_medquad(tmp, answer=""),
                                            retrieved_on="2026-09")), [])

    def test_missing_checkout_names_the_remedy(self):
        with self.assertRaises(cochrane.SourceUnavailable) as ctx:
            list(nih_medquad.candidates(Path("/nonexistent/MedQuAD"),
                                        retrieved_on="2026-09"))
        self.assertIn("abachaa/MedQuAD", str(ctx.exception))


class PoolTests(unittest.TestCase):

    def test_screen_rejects_near_duplicates_without_deleting_them(self):
        with TemporaryDirectory() as tmp:
            near = row(
                Question="Are antipsychotics effective for agitation in "
                         "Alzheimer disease patients?",
                PMID="/99999999/")
            pool = list(cochrane.candidates(write_csv(tmp, [row(), near])))
        screened, stats = build_pool.screen(pool)
        self.assertEqual(len(screened), len(pool))
        self.assertEqual(stats["auto_rejected"], 1)
        self.assertIn("near_duplicate_of_earlier_candidate",
                      stats["rejection_reasons"])

    def test_nothing_is_marked_final_by_the_builder(self):
        with TemporaryDirectory() as tmp:
            pool = list(cochrane.candidates(write_csv(tmp, [row()])))
        screened, _ = build_pool.screen(pool)
        self.assertTrue(all(q.status in ("validated", "rejected")
                            for q in screened))
        self.assertFalse(any(q.status == "final" for q in screened))
        self.assertFalse(any(q.is_usable for q in screened))

    def test_review_export_omits_rejected_and_has_blank_decisions(self):
        with TemporaryDirectory() as tmp:
            pool = list(cochrane.candidates(write_csv(tmp, [row()])))
            screened, _ = build_pool.screen(pool)
            paths = build_pool.write_pool(screened, Path(tmp) / "out")
            rows = list(csv.DictReader(open(paths["review"], encoding="utf-8")))
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["review_decision"], "")
            self.assertTrue(r["source_locator"])

    def test_pool_jsonl_round_trips(self):
        import json
        with TemporaryDirectory() as tmp:
            pool = list(cochrane.candidates(write_csv(tmp, [row()])))
            screened, _ = build_pool.screen(pool)
            paths = build_pool.write_pool(screened, Path(tmp) / "out")
            records = [json.loads(l) for l in
                       Path(paths["candidates"]).read_text().splitlines()]
        self.assertEqual(len(records), len(screened))
        self.assertIn("reference_locator", records[0])


if __name__ == "__main__":
    unittest.main()
