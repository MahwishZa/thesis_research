"""The independent check of the text dataset (src/text/audit.py), on a dataset built from the build's made-up cases."""

import contextlib
import io
import json
from unittest import mock

from src.text import audit, prepare

from .test_prepare import BCC_TEXT, BuildData


class AuditTests(BuildData):
    def setUp(self):
        super().setUp()
        prepare.build(self.data, classes=self.CLASSES)
        self.rows = {r["patient_uid"]: r for r in self._table()}

    def _file(self, uid):
        return self.data / "dataset" / self.rows[uid]["path"]

    def test_a_built_dataset_passes_every_check_and_two_questions_per_class_are_written_to_read(self):
        done = audit.audit(self.data)
        self.assertEqual({check: ids for check, ids in done["failures"].items() if ids}, {})
        self.assertEqual(len(done["rows"]), 9)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertLess(report.index("<!-- step: build -->"), report.index("<!-- step: audit -->"))
        for expected in ("## Audit: an independent check of the questions", "- Questions checked: 9",
                         "| question not word for word in its source | 0 |  |", "All checks passed."):
            self.assertIn(expected, report)
        sample = (self.data / "sources" / "audit_sample.txt").read_text(encoding="utf-8")
        heads = [line for line in sample.split("\n") if line.startswith("== ")]
        self.assertEqual(len(heads), 9)  # at most 2 per class; this dataset has 1 or 2
        self.assertIn("== actinic keratosis | pubmed-2101 | PubMed abstract | test", heads)
        self.assertIn(self._file("2001-1").read_text(encoding="utf-8"), sample)

    def test_a_changed_missing_or_extra_file_and_a_question_that_names_its_class_or_is_not_in_its_source_fail(self):
        self._file("2001-1").write_text(BCC_TEXT, encoding="utf-8")  # the whole case: names its class and the biopsy
        self._file("2010-1").unlink()
        changed = self._file("2008-1").read_text(encoding="utf-8").replace("six months", "five months")
        self._file("2008-1").write_text(changed, encoding="utf-8")
        self._file("2006-2").write_bytes(self._file("2006-1").read_bytes())
        self._file("2013-1").write_text("A 10-year-old girl had a brown mole on her back that developed a white ring around it "
                                        "over the past year.", encoding="utf-8")  # in its source, but short and no lesion word
        self._file("pubmed-2102").write_text("A 25-year-old woman had a small brown macule on her forearm", encoding="utf-8")
        self._file("2003-1").write_text(self._file("2003-1").read_text(encoding="utf-8") + " An SCC was suspected.",
                                        encoding="utf-8")  # the class's short form
        (self.data / "dataset" / "test" / "nevus" / "extra.txt").write_text("left over", encoding="utf-8")
        (self.data / "dataset" / "test" / "nevus" / "Thumbs.db").write_bytes(b"\0")  # written by Windows Explorer: ignored
        done = audit.audit(self.data)
        self.assertEqual({check: ids for check, ids in done["failures"].items() if ids}, {
            "question file missing": ["2010-1"],
            "question file differs from its fingerprint": ["2001-1", "2008-1", "2013-1", "2003-1", "pubmed-2102", "2006-2"],
            "file in the dataset folder but not in dataset.csv": ["test/nevus/extra.txt"],
            "question not word for word in its source": ["2008-1", "2003-1", "2006-2"],
            "question names its class or its short form": ["2001-1", "2003-1"],
            "question reports a test result": ["2001-1"],
            "question shorter than 25 words": ["2013-1", "pubmed-2102"],
            "question describes no lesion": ["2013-1"],
            "question the same as another": ["2006-2 = 2006-1"]})
        self.assertIn("Some checks failed", (self.data / prepare.REPORT).read_text(encoding="utf-8"))

    def test_a_short_form_the_source_defines_for_the_class_is_found_in_a_question(self):
        path = self.data / "sources" / "pubmed" / "abstracts.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        case = records[0]["sections"][1][1]  # pubmed-2101: its background defines a short form, its case part uses it
        records[0]["sections"] = [["BACKGROUND", "Actinic keratosis (AKT) is common."],
                                  ["CASE PRESENTATION", case.replace("It was 1 cm across", "The AKT was 1 cm across")]]
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        question = self._file("pubmed-2101").read_text(encoding="utf-8").replace("It was 1 cm across", "The AKT was 1 cm across")
        self._file("pubmed-2101").write_text(question, encoding="utf-8")
        failures = audit.audit(self.data)["failures"]
        self.assertEqual(failures["question uses a short form its source defines for the class"], ["pubmed-2101"])
        self.assertNotIn("pubmed-2101", failures.get("question not word for word in its source", []))
        self.assertEqual(audit._defined(["Seborrheic keratoses (SKs), not (Fig. 2) nor an MRI (MRI); PG (pyogenic granuloma)."],
                                        audit._name_pattern(["seborrheic keratosis"])), {"SK"})
        self.assertEqual(audit._defined(["PG (pyogenic granuloma) is common."], audit._name_pattern(["pyogenic granuloma"])), {"PG"})

    def test_the_audit_checks_the_build_s_rules_for_length_and_test_results(self):
        self.assertEqual(audit.MIN_WORDS, prepare.MIN_WORDS)
        for word in prepare.LESION_TYPES:  # every lesion type the build counts as a description, the audit counts too
            self.assertTrue(audit.TYPES.search(audit._plain(f"a red {word} here")), word)
        self.assertEqual((audit.SHARED_RUN, audit.SAME_PATIENT), (prepare.SHARED_RUN, prepare.SAME_PATIENT))
        for text in ("A 55 yr old man", "a 3-month-old infant", "aged 70", "The patient", "An 80-year-old farmer", "in childhood",
                     "Two cases of solitary tumours", "this patient's tumor", "A newborn"):  # the audit's own test of a patient finds what the build's finds
            self.assertEqual(bool(audit.PATIENT.search(text)), bool(prepare.PATIENT.search(text)), text)
        for cue in prepare.RESULT_CUES:  # every word beginning the build stops at is a test result for the audit too
            self.assertTrue(audit.RESULT.search(audit._plain(f"The {cue}x was seen.")), cue)

    def test_the_same_patient_in_two_articles_fails_and_a_shared_phrase_is_only_listed(self):
        words = self._file("2008-1").read_text(encoding="utf-8").split()
        scc = self._file("2003-1")
        scc.write_text(f"{scc.read_text(encoding='utf-8')} {' '.join(words[:12])}", encoding="utf-8")  # a shared phrase only
        start = " ".join(self._file("2010-1").read_text(encoding="utf-8").split()[:22])
        self._file("pubmed-2101").write_text(f"{start} {self._file('pubmed-2101').read_text(encoding='utf-8')}",
                                            encoding="utf-8")  # another article's case partly told again: about a quarter
        vascular = self._file("2006-2")  # the same article as 2006-1: kept in one split, not compared
        vascular.write_text(f"{vascular.read_text(encoding='utf-8')} {self._file('2006-1').read_text(encoding='utf-8')}",
                            encoding="utf-8")
        real = audit.audit
        with mock.patch.object(audit, "audit", lambda show=None: real(self.data, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown, self.assertRaises(SystemExit):
            audit.main([])
        found = real(self.data)["failures"][audit.DUPLICATE]
        self.assertEqual(len(found), 1)
        self.assertRegex(found[0], r"^pubmed-2101 ~ 2010-1 \(test/train, \d+%\)$")
        self.assertIn("2008-1 (melanoma, test) ~ 2003-1 (squamous cell carcinoma, test): ", shown.getvalue())  # listed only
        self.assertIn('shared: "a 52 year old man noticed that a mole on his upper"', shown.getvalue())
        self.assertNotIn("shared", (self.data / prepare.REPORT).read_text(encoding="utf-8"))  # no case text in the report

    def test_a_question_without_a_patient_or_with_a_lesion_inside_the_body_fails(self):
        self._file("2013-1").write_text("Brown macules on the back are common in childhood and most of them stay small, round and "
                                        "evenly coloured for many years without any change in their size or shape.", encoding="utf-8")
        self._file("2008-1").write_text(f"{self._file('2008-1').read_text(encoding='utf-8')} A second black macule was seen on the "
                                        "palatal mucosa.", encoding="utf-8")
        self._file("2010-1").write_text(f"{self._file('2010-1').read_text(encoding='utf-8')} Intraoral examination was normal.",
                                        encoding="utf-8")  # no lesion in that sentence: not inside the body
        self._file("2006-1").write_text("A 9-year-old boy came to the clinic because of lesions on his chin that had been present for "
                                        "three weeks; he had no other illness, no allergy and no family history of skin disease, and "
                                        "he took no medicines.", encoding="utf-8")  # nothing of what the lesion looks like
        self._file("2003-1").write_text("A 70-year-old man had a slowly growing nodule on his scalp for two years; he had worked "
                                        "outdoors all his life as a farmer and had no other illness or family history of cancer.",
                                        encoding="utf-8")  # "a nodule" tells what the lesion looks like
        failures = audit.audit(self.data)["failures"]
        self.assertEqual(failures["question describes no patient"], ["2013-1"])
        self.assertEqual(failures["question puts a lesion inside the body"], ["2008-1"])
        self.assertEqual(failures["question tells nothing of what the lesion looks like"], ["2006-1"])

    def test_names_are_found_in_any_spelling_plural_two_word_order_or_apostrophe_and_short_forms_in_any_case(self):
        cases = {"A pigmented naevus.": ["nevus"], "Two naevi, one atypical.": ["nevus"], "Melanomas were found.": ["melanoma"],
                 "Keratoses, actinic, on the scalp.": ["actinic keratosis"], "Bowen’s disease of the leg.": ["Bowen's disease"],
                 "A basal-cell carcinoma.": ["basal cell carcinoma"], "Lentigines on the back.": ["lentigo"]}
        for text, names in cases.items():
            self.assertTrue(audit._name_pattern(names).search(audit._plain(text)), text)
        self.assertFalse(audit._name_pattern(["nevus"]).search(audit._plain("A nevoid pattern; Nevada.")))
        self.assertFalse(audit._name_pattern(["melanoma"]).search(audit._plain("A melanocytic lesion.")))

    def test_the_command_line_prints_the_result_and_fails_when_a_check_fails(self):
        real = audit.audit
        with mock.patch.object(audit, "audit", lambda show=None: real(self.data, show=show)):
            with contextlib.redirect_stdout(io.StringIO()) as shown:
                audit.main([])
            for expected in ("Questions checked: 9", "All checks passed.", "audit_sample.txt (5 questions per class to read; not in Git)"):
                self.assertIn(expected, shown.getvalue())
            self._file("2010-1").unlink()
            with contextlib.redirect_stdout(io.StringIO()) as shown, self.assertRaises(SystemExit) as stop:
                audit.main([])
        self.assertEqual(stop.exception.code, 1)
        self.assertIn("FAILED question file missing: 1 (for example 2010-1)", shown.getvalue())
        with self.assertRaises(SystemExit):
            audit.main(["extra"])
