"""The independent check of the text dataset (src/build/text/audit.py) and the revision it leads to (src/build/text/prepare.py,
step revise), on a dataset built from the build's made-up cases."""

import contextlib
import io
import json
import re
from unittest import mock

from src.build.text import audit, prepare

from .test_prepare import BCC_TEXT, BuildData

REVIEW = audit.review


class AuditTests(BuildData):
    def setUp(self):
        super().setUp()
        prepare.build(self.data, classes=self.CLASSES)
        self.rows = {r["patient_uid"]: r for r in self._table()}

    def _file(self, uid):
        return self.data / prepare.TEXT_FILES / self.rows[uid]["path"]

    def test_a_built_dataset_passes_every_check_and_two_questions_per_class_are_written_to_read(self):
        done = audit.audit(self.data)
        self.assertEqual({check: ids for check, ids in done["failures"].items() if ids}, {})
        self.assertEqual(len(done["rows"]), 9)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertLess(report.index("<!-- step: build -->"), report.index("<!-- step: audit -->"))
        for expected in ("## Audit: an independent check of the questions", "- Questions checked: 9",
                         "| question not word for word in its source | 0 |  |", "All checks passed."):
            self.assertIn(expected, report)
        sample = (self.data / "downloads" / "audit_sample.txt").read_text(encoding="utf-8")
        heads = [line for line in sample.split("\n") if line.startswith("== ")]
        self.assertEqual(len(heads), 9)  # every question: this dataset has 1 or 2 per class
        self.assertIn("== actinic keratosis | pubmed-2101 | PubMed abstract | test", heads)
        self.assertIn(self._file("2001-1").read_text(encoding="utf-8"), sample)
        self.assertIn("-- after it in its source:\nA shave biopsy was taken. Histology showed an actinic keratosis.\n", sample)

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
        (self.data / prepare.TEXT_FILES / "test" / "nevus" / "extra.txt").write_text("left over", encoding="utf-8")
        (self.data / prepare.TEXT_FILES / "test" / "nevus" / "Thumbs.db").write_bytes(b"\0")  # written by Windows Explorer: ignored
        done = audit.audit(self.data)
        self.assertEqual({check: ids for check, ids in done["failures"].items() if ids}, {
            "question file missing": ["2010-1"],
            "question file differs from its fingerprint": ["2001-1", "2008-1", "2013-1", "2003-1", "pubmed-2102", "2006-2"],
            "file in the dataset folder but not in dataset.csv": ["test/nevus/extra.txt"],
            "question not word for word in its source": ["2008-1", "2003-1", "2006-2"],
            "question names its class or its short form": ["2001-1", "2003-1"],
            "question reports a test result": ["2001-1"],
            "no test result after the question names its class": ["2001-1", "2008-1", "2003-1", "2006-2"],  # not in their source
            "question shorter than 25 words": ["2013-1", "pubmed-2102"],
            "question describes no lesion": ["2013-1"],
            "question the same as another": ["2006-2 = 2006-1"]})
        self.assertIn("Some checks failed", (self.data / prepare.REPORT).read_text(encoding="utf-8"))

    def test_a_short_form_the_source_defines_for_the_class_is_found_in_a_question(self):
        path = self.data / "downloads" / "pubmed" / "abstracts.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        case = records[0]["sections"][1][1]  # pubmed-2101: its background defines a short form, its case part uses it
        records[0]["sections"] = [["BACKGROUND", "Actinic keratosis of the hand (AKT) is common."],
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
        keratosis, carcinoma = audit._name_pattern(["seborrheic keratosis"]), audit._name_pattern(["basal cell carcinoma"])
        bcc = re.compile(r"\b(?:BCC)s?\b", re.I)
        title = "Development of Seborrheic Keratosis and Human Papillomavirus (HPV) Infection"
        self.assertEqual(audit._defined([title], keratosis, ending=True), set())  # the name does not end right before the brackets
        self.assertEqual(audit._defined([title], keratosis), {"HPV"})  # but it is in the 5 words before them
        self.assertEqual(audit._defined(["A nodular basal cell carcinoma (nBCC) and a nodular BCC (nBCCs)."], carcinoma, bcc,
                                        ending=True), {"nBCC"})
        self.assertEqual(audit._defined(["A man had a nodular BCC (nBCC)."], carcinoma, bcc, ending=True), {"nBCC"})  # short form
        self.assertEqual(audit._defined(["A man had a nodular BCC (nBCC)."], carcinoma, ending=True), set())
        self.assertEqual(audit._defined(["A BCC or not (NOT)."], carcinoma, bcc, ending=True), set())
        self.assertEqual(audit._defined(["A BCC or not (NOT)."], carcinoma, bcc), {"NOT"})
        self.assertEqual(audit._defined(["Nevoid basal cell carcinoma syndrome (NBCCS) is rare."], carcinoma, bcc, ending=True), set())
        angiokeratoma = audit._name_pattern(["angiokeratoma"])
        self.assertEqual(audit._defined(["Angiokeratoma circumscriptum (AC) is rare."], angiokeratoma), {"AC"})
        self.assertEqual(audit._defined(["Angiokeratoma circumscriptum (AC) is rare."], angiokeratoma, ending=True), set())

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

    def test_a_case_heading_and_a_short_form_defined_on_the_class_s_short_form_are_read_as_the_build_reads_them(self):
        patients = self.folder / "PMC-Patients-V2.json"  # 2001-1's source defines nBCC on BCC; its result names only nBCC
        patients.write_text(patients.read_text(encoding="utf-8").replace("9 mm shiny papule", "9 mm shiny nBCC papule").replace(
            "A biopsy confirmed a pigmented basal cell carcinoma, which was excised.", "It looked like a nodular BCC (nBCC). She "
            "came back in May. A biopsy was taken. Histology confirmed an nBCC."), encoding="utf-8")
        self._file("2001-1").write_text(self._file("2001-1").read_text(encoding="utf-8").replace(
            "9 mm shiny papule", "9 mm shiny nBCC papule"), encoding="utf-8")  # and its question uses nBCC
        path = self.data / "downloads" / "pubmed" / "abstracts.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[0]["sections"][1][1] = "Case 2. " + records[0]["sections"][1][1].replace(  # pubmed-2101: a heading at its start,
            "Histology showed an actinic keratosis.", "Histology showed HPV.")  # and a result naming only a virus
        records[0]["sections"][0][1] = "Actinic keratosis and human papillomavirus (HPV) are common."
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        self._file("pubmed-2101").write_text("Case 2. " + self._file("pubmed-2101").read_text(encoding="utf-8"), encoding="utf-8")
        failures = audit.audit(self.data)["failures"]
        self.assertEqual(failures["question uses a short form its source defines for the class"], ["2001-1"])
        self.assertNotIn("2001-1", failures.get("no test result after the question names its class", []))
        self.assertNotIn("pubmed-2101", failures.get("question describes more than one patient", []))
        self.assertIn("pubmed-2101", failures["no test result after the question names its class"])  # HPV is not the class
        self.assertEqual(failures.get("question not word for word in its source", []), [])

    def test_two_patients_no_lesion_on_the_skin_and_no_test_result_naming_the_class_fail(self):
        self._file("2013-1").write_text("Case 1 was a 10-year-old girl with a brown papule on her back that grew over a year and "
                                        "developed a white ring; case 2, her brother, had a similar papule on his arm.", encoding="utf-8")
        self._file("2010-1").write_text("A 75-year-old retired sailor had a red, scaly mass in the esophagus for a year that was 8 mm "
                                        "across and bled when he swallowed. He had no history of skin disease.", encoding="utf-8")
        path = self.data / "downloads" / "pubmed" / "abstracts.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        case = records[0]["sections"][1][1]  # pubmed-2101: its question stays in it, its result no longer names the class
        records[0]["sections"][1][1] = case.replace("Histology showed an actinic keratosis.", "It was frozen. It healed. AK was likely.")
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        patients = self.folder / "PMC-Patients-V2.json"  # 2003-1's result names only keratoacanthoma, a name that does not count
        patients.write_text(patients.read_text(encoding="utf-8").replace("The excised SCC was well differentiated.",
                                                                         "Histology showed a keratoacanthoma (KA)."), encoding="utf-8")
        two = "A 52-year-old man and a 49-year-old man noticed"  # 2008-1, in its file and its source: two patients' ages
        patients.write_text(patients.read_text(encoding="utf-8").replace("A 52-year-old man noticed", two), encoding="utf-8")
        self._file("2008-1").write_text(self._file("2008-1").read_text(encoding="utf-8").replace("A 52-year-old man noticed", two),
                                        encoding="utf-8")
        failures = audit.audit(self.data)["failures"]
        self.assertEqual(failures["question describes more than one patient"], ["2008-1", "2013-1"])
        self.assertEqual(failures["question describes no lesion as seen on the skin"], ["2010-1"])
        for text, seen in {"Brown moles on both legs.": True, "Dusky papules on the upper and lower extremities.": True, "A red nodule (Fig. 2) on the left cheek.": True,
                           "A brown mole. It was on both legs.": False, "A chest CT showed a small nodule in the lung.": False,
                           "A nodule grew on the left cheek. It was red and firm.": True,
                           "It was red and firm. A nodule grew on the left cheek.": True,
                           "A nodule grew on the left cheek. It was 5 mm. It was red.": False,
                           "She had skin cancer. A nodule grew on her chest.": False,
                           "A port-wine stain on the left cheek was purple.": True, "A wound on the left shin was red.": True,
                           "A red papule on the prepuce.": True,
                           "Abdominal CT showed a mass in the bile duct. It was cystic.": False,
                           "A CT of the abdomen showed a polypoid lesion.": False}.items():
            self.assertEqual(audit._on_skin(text), seen, text)
        text = "A biopsy (Fig. 2) showed nests (Fig. 3). Of cells (e.g. Basal ones). It was new? Yes."
        self.assertEqual(audit._sentences(text), [text[a:b].strip() for a, b in zip(*(lambda s: (s, s[1:] + [len(text)]))(
            prepare._sentence_starts(text)))])  # the audit's sentences are the build's
        self.assertEqual(sorted(failures["no test result after the question names its class"]),
                         ["2003-1", "2010-1", "2013-1", "pubmed-2101"])
        scc = audit._name_pattern(["squamous cell carcinoma"])
        self.assertTrue(audit._confirms(" A punch biopsy was taken. It showed a squamous cell carcinoma.", scc, None, set()))
        self.assertTrue(audit._confirms(" A biopsy was taken. It healed. A squamous cell carcinoma was found.", scc, None, set()))
        self.assertFalse(audit._confirms(" A biopsy was taken. It healed. It was not sent. A squamous cell carcinoma was found.",
                                         scc, None, set()))
        self.assertFalse(audit._confirms(" A biopsy was taken. His father had a squamous cell carcinoma.", scc, None, set()))
        self.assertFalse(audit._confirms(" He was at risk of keratinocyte carcinomas. It was a squamous cell carcinoma.", scc, None,
                                         set()))  # a group of diseases, no test result
        self.assertTrue(audit._confirms(" Histology showed atypical keratinocytes. It was a squamous cell carcinoma.", scc, None,
                                        set()))
        self.assertNotIn("pubmed-2101", failures.get("question not word for word in its source", []))
        for text in ("Two patients were seen.", "The mean age was 54 years.", "Case 2 is a man.", "Both cases were alike.",
                     "The patient's mother had a mole.", "A patient with a case of acne.", "Case 2. A 40-year-old man.",
                     "Patient 3: a woman.", "CASE II - A boy.", "Case 1 is a woman; case 2, her son.", "A man. Case 2 is a woman.",
                     "Patient is a man; patient 2 is not.", "We report a 40-year-old man and a 48-year-old man.",
                     "A 45-year old African American woman; her 8-year-old daughter.", "A 61 years old female. The 61-year-old "
                     "female had a mole.", "One lesion was of a 53-year-old woman. The other, of a 20-year-old man."):  # the
            # build's test of one patient, a heading at the start left out
            self.assertEqual(bool(audit.ONE_PATIENT.search(audit.CASE_HEADING.sub("", text, count=1)))
                             or len({int(m.group(1)) for m in audit.AGE_OF.finditer(text)}) > 1,
                             bool(prepare.MANY.search(prepare.HEADING.sub("", text, count=1))) or len(prepare._ages(text)) > 1,
                             text)
        self.assertEqual(audit.CASE_HEADING.sub("", "Patient is a man.", count=1), "Patient is a man.")  # not a heading
        for word in prepare.LESION_WORDS + prepare.PLACED:  # what the build lets a sentence put on the skin, the audit does too
            self.assertTrue(audit.PLACED.search(audit._plain(f"a {word} here")), word)

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
            for expected in ("Questions checked: 9", "All checks passed.", "audit_sample.txt (10 questions per class to read; not in Git)"):
                self.assertIn(expected, shown.getvalue())
            self._file("2010-1").unlink()
            with contextlib.redirect_stdout(io.StringIO()) as shown, self.assertRaises(SystemExit) as stop:
                audit.main([])
        self.assertEqual(stop.exception.code, 1)
        self.assertIn("FAILED question file missing: 1 (for example 2010-1)", shown.getvalue())
        with self.assertRaises(SystemExit):
            audit.main(["extra"])


class ReviewData(BuildData):
    """The built made-up dataset (9 questions), a knowledge base with one passage that quotes a question, and three closed-book
    runs; no tests of its own."""

    def setUp(self):
        super().setUp()
        prepare.build(self.data, classes=self.CLASSES)
        self.rows = {r["patient_uid"]: r for r in self._table()}
        self.texts = {uid: (self.data / prepare.TEXT_FILES / r["path"]).read_text(encoding="utf-8") for uid, r in self.rows.items()}
        self.knowledge = self.data.parent / "knowledge"
        library = self.knowledge / "texts"
        library.mkdir(parents=True)
        leaked = " ".join(self.texts["2008-1"].split()[:14])  # a passage that quotes 14 words of a question
        (library / "passages.jsonl").write_text(json.dumps({"passage_id": "pmc-1", "text": f"As reported, {leaked} and more."}) + "\n"
                                                + json.dumps({"passage_id": "pmc-2", "text": "Unrelated words about something else entirely."})
                                                + "\n", encoding="utf-8")
        (library / "appearance.jsonl").write_text("", encoding="utf-8")
        self.results = self.data.parent / "results"
        for run, answer in (("a", "melanoma"), ("b", "melanoma"), ("c", None)):
            folder = self.results / "text" / run / "test_1"
            folder.mkdir(parents=True)
            (folder / "run_card.json").write_text("{}", encoding="utf-8")
            lines = [{"path": r["path"], "prediction": r["label"]} for r in self.rows.values()]
            for line in lines:
                if line["path"] == self.rows["2001-1"]["path"]:  # basal cell carcinoma: two runs say melanoma
                    line["prediction"] = answer if run != "c" else "melanoma"
            (folder / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")

    def sources(self, rests):
        found = audit._sources(self.data, list(self.rows.values()))
        for uid, rest in rests.items():
            found[uid] = [self.texts[uid] + " " + rest]
        return mock.patch.object(audit, "_sources", return_value=found)

    def _row(self, uid):
        return {r["patient_uid"]: r for r in audit.review(self.data, self.results, self.knowledge)["rows"]}[uid]


class ReviewTests(ReviewData):
    """The review: signals of a doubtful label or a leak."""

    def test_every_question_gets_a_row_and_a_built_dataset_shows_only_the_signals_it_has(self):
        audit.audit(self.data)
        done = audit.review(self.data, self.results, self.knowledge)
        rows = {r["patient_uid"]: r for r in done["rows"]}
        self.assertEqual(set(rows), set(self.rows))
        self.assertEqual(rows["2001-1"]["same_wrong_answer"], "melanoma")  # all three runs gave the same wrong answer
        self.assertEqual({uid for uid, r in rows.items() if r["same_wrong_answer"]}, {"2001-1"})  # right answers are no signal
        self.assertEqual(rows["2008-1"]["knowledge_passages_sharing_12_words"], "pmc-1")
        self.assertEqual({uid for uid, r in rows.items() if r["knowledge_passages_sharing_12_words"]}, {"2008-1"})
        self.assertEqual({uid for uid, r in rows.items() if r["result_names_other_class"]}, set())
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("## Review: signals of a doubtful label or of leakage", report)
        self.assertIn("Closed-book test runs read: 3.", report)
        self.assertLess(report.index("<!-- step: audit -->"), report.index("<!-- step: review -->"))
        with (self.data / audit.REVIEW_FLAGS).open(encoding="utf-8") as f:
            self.assertEqual(next(iter(f)).strip().split(","), audit.REVIEW_COLUMNS)

    def test_one_run_alone_gives_no_signal(self):
        for run in ("b", "c"):
            for path in (self.results / "text" / run).rglob("*"):
                if path.is_file():
                    path.unlink()
        self.assertEqual(self._row("2001-1")["same_wrong_answer"], "")

    def test_a_confirming_result_that_also_names_another_class_unless_negated(self):
        with self.sources({"2001-1": "Histology showed a basal cell carcinoma colliding with a melanoma."}):
            self.assertEqual(self._row("2001-1")["result_names_other_class"], "melanoma")
        with self.sources({"2001-1": "Histology showed a basal cell carcinoma, with no evidence of melanoma."}):
            self.assertEqual(self._row("2001-1")["result_names_other_class"], "")
        with self.sources({"2001-1": "Histology showed a basal cell carcinoma. The patient had a melanoma ten years ago."}):
            self.assertEqual(self._row("2001-1")["result_names_other_class"], "")  # not in a sentence that confirms the class

    def test_look_alike_wording_in_the_confirming_sentences(self):
        with self.sources({"2001-1": "Histology showed a basal cell carcinoma. It had been clinically diagnosed as a wart."}):
            self.assertEqual(self._row("2001-1")["lookalike_wording"], "yes")
        with self.sources({"2001-1": "It was suspected to be a wart. Much later, many things happened. And then more things. "
                                     "Histology showed a basal cell carcinoma."}):
            self.assertEqual(self._row("2001-1")["lookalike_wording"], "no")  # far before the result: not in its window

    def test_a_changed_question_or_a_missing_knowledge_base_stops_the_review(self):
        (self.knowledge / "texts" / "appearance.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "appearance.jsonl not found"):
            audit.review(self.data, self.results, self.knowledge)
        (self.knowledge / "texts" / "appearance.jsonl").write_text("", encoding="utf-8")
        (self.data / prepare.TEXT_FILES / self.rows["2001-1"]["path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from its fingerprint"):
            audit.review(self.data, self.results, self.knowledge)

    def test_the_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.object(audit, "review", side_effect=lambda show=None: REVIEW(self.data, self.results, self.knowledge, show)):
            audit.main(["review"])
        self.assertIn("Reviewed 9 questions.", out.getvalue())
        with self.assertRaises(SystemExit):
            audit.main(["other"])


class ReviseTests(ReviewData):
    """Version 3: the review flags 2001-1 (its confirming result also names melanoma; every run gave the same wrong answer) and
    2008-1 (look-alike wording, and a knowledge passage that quotes it); revise takes out 2001-1 only."""

    def setUp(self):
        super().setUp()
        audit.audit(self.data)
        with self.sources({"2001-1": "Histology showed a basal cell carcinoma colliding with a melanoma.",
                           "2008-1": "Histology showed a melanoma. It had been clinically diagnosed as a wart."}):
            flags = {r["patient_uid"]: r for r in audit.review(self.data, self.results, self.knowledge)["rows"]}
        self.assertEqual((flags["2008-1"]["lookalike_wording"], flags["2008-1"]["knowledge_passages_sharing_12_words"],
                          flags["2008-1"]["result_names_other_class"]), ("yes", "pmc-1", ""))  # signals that are never a reason
        self.before = self._table()
        self.path = self.rows["2001-1"]["path"]

    def test_only_the_flagged_question_is_taken_out_and_its_file_moved_not_deleted(self):
        content = (self.data / prepare.TEXT_FILES / self.path).read_bytes()
        done = prepare.revise(self.data)
        self.assertEqual([r["patient_uid"] for r in done["removed"]], ["2001-1"])
        self.assertEqual(self._table(), [r for r in self.before if r["patient_uid"] != "2001-1"])  # every other row as it was
        self.assertFalse((self.data / prepare.TEXT_FILES / self.path).exists())
        self.assertEqual((self.data / prepare.DOWNLOADS / prepare.REMOVED / self.path).read_bytes(), content)
        self.assertEqual((done["taken_out_now"], done["moved"]), (1, [self.path]))
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## Revision: version 3", "- Questions in version 2 (reviewed): 9; taken out: 1; version 3: 8",
                         "Taken out (the other class its confirming result names): 2001-1 (test, basal cell carcinoma: melanoma)."):
            self.assertIn(expected, report)
        self.assertLess(report.index("<!-- step: review -->"), report.index("<!-- step: revise -->"))
        self.assertEqual(len((self.data / audit.REVIEW_FLAGS).read_text(encoding="utf-8").splitlines()), 10)  # the review is kept

    def test_a_second_run_changes_nothing(self):
        prepare.revise(self.data)
        files = {name: (self.data / name).read_bytes() for name in ("dataset.csv", prepare.REPORT)}
        done = prepare.revise(self.data)
        self.assertEqual({name: (self.data / name).read_bytes() for name in files}, files)
        self.assertEqual((done["taken_out_now"], done["moved"], len(done["removed"])), (0, [], 1))

    def test_a_table_already_at_version_3_moves_the_files_the_same_way(self):
        prepare.revise(self.data)
        moved = self.data / prepare.DOWNLOADS / prepare.REMOVED / self.path
        (self.data / prepare.TEXT_FILES / self.path).write_bytes(moved.read_bytes())  # a computer with the new table and the old files
        moved.unlink()
        table = (self.data / "dataset.csv").read_bytes()
        done = prepare.revise(self.data)
        self.assertEqual(done["moved"], [self.path])
        self.assertTrue(moved.is_file())
        self.assertEqual((self.data / "dataset.csv").read_bytes(), table)

    def test_a_question_the_review_has_not_seen_or_in_another_place_stops_the_step(self):
        flags = (self.data / audit.REVIEW_FLAGS).read_text(encoding="utf-8")
        (self.data / audit.REVIEW_FLAGS).write_text("\n".join(line for line in flags.splitlines() if not line.startswith("2008-1,")) + "\n",
                                                   encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "has no row for 1 questions .for example 2008-1."):
            prepare.revise(self.data)
        (self.data / audit.REVIEW_FLAGS).write_text(flags.replace("2008-1,test,", "2008-1,train,"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "2008-1 is in test/melanoma in dataset.csv but in train/melanoma"):
            prepare.revise(self.data)
        (self.data / audit.REVIEW_FLAGS).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.build.text.audit review first"):
            prepare.revise(self.data)
        self.assertEqual(self._table(), self.before)  # nothing written
        self.assertTrue((self.data / prepare.TEXT_FILES / self.path).is_file())

    def test_the_command_line(self):
        real = prepare.revise
        with mock.patch.object(prepare, "revise", lambda show=None: real(self.data, show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["revise"])
        self.assertIn("Version 3: 8 questions, each in its split; taken out by the rule: 1 (from the table now: 1; files moved now: 1",
                      shown.getvalue())
