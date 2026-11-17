"""The knowledge base's additions (src/multimodal/enrich.py), on tiny invented data: no network, no real data."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.multimodal import enrich, prepare
from src.text import library as textlib
from tests.multimodal.test_prepare import Project

DERMOSCOPY = ("Dermoscopy of the lesion reveals an atypical pigment network with irregular globules and streaks at the periphery of the "
              "lesion, a pattern seen in many cases and described at length by several groups of authors over the years.")
TREATMENT = ("Treatment usually consists of surgical excision with clear margins, and cryotherapy or imiquimod therapy may be used for "
             "superficial disease in selected patients who cannot undergo surgery or who prefer a less invasive route.")
CLINICAL = ("A well demarcated dome-shaped nodule with a pearly surface and telangiectasia is the classic clinical presentation of this "
            "tumour in sun exposed skin of older people with a fair complexion and a long history of exposure.")
VESSELS = ("On dermoscopy, arborizing vessels, blue-gray ovoid nests and shiny white streaks are typical findings, while ulceration may "
           "also be present in larger tumours of long standing duration that are seen in elderly patients.")
EPIDEMIOLOGY = ("The incidence and prevalence of the disease have risen in many countries, and several large registries report that "
                "mortality and survival differ with the age of the patient and the stage at which the diagnosis was made.")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class TagTests(unittest.TestCase):
    def test_the_tags_by_hand(self):
        flags = enrich.aspect_flags(DERMOSCOPY)  # pigment network, globules, streaks, dermoscop: four words, threshold 2
        self.assertEqual((flags["dermoscopy"], flags["appearance"], flags["treatment"], flags["histology"]), (True, True, False, False))
        self.assertEqual(enrich.aspect_words(DERMOSCOPY)["dermoscopy"], 4)
        flags = enrich.aspect_flags(CLINICAL)  # well demarcated, dome-shaped, nodule, pearly, telangiectasia: clinical look, not dermoscopy
        self.assertEqual((flags["clinical look"], flags["dermoscopy"], flags["appearance"]), (True, False, True))
        self.assertEqual(enrich.aspect_words(TREATMENT)["treatment"], 7)  # treatment, therapy, excision, surgery, surgical, cryotherapy, imiquimod
        self.assertEqual((enrich.aspect_flags(TREATMENT)["treatment"], enrich.aspect_flags(TREATMENT)["appearance"]), (True, False))
        self.assertTrue(enrich.aspect_flags(EPIDEMIOLOGY)["epidemiology"])
        one = "Dermoscopy is useful."  # one word of the list is below the threshold of 2
        self.assertEqual((enrich.aspect_words(one)["dermoscopy"], enrich.aspect_flags(one)["dermoscopy"]), (1, False))
        self.assertEqual(enrich.aspect_words("DERMOSCOPY and Dermoscopic")["dermoscopy"], 1)  # a word counts once, in any case

    def test_a_section_title_names_one_aspect_or_none(self):
        for title, expected in (("Treatment / Management", "treatment"), ("Histopathology", "histology"), ("History and Physical", "appearance"),
                                ("Dermoscopic features", "appearance"), ("Epidemiology", "epidemiology"), ("Clinical presentation", "appearance"),
                                ("Abstract", None), ("Pathophysiology", None), ("Discussion", None),
                                ("Clinical features and treatment", None)):  # two aspects: none
            with self.subTest(title):
                self.assertEqual(enrich.heading_aspect(title), expected)

    def test_agreement_with_the_titles_by_hand(self):
        def row(heading, **tagged):
            base = {"heading_aspect": heading, "appearance": "no", "histology": "no", "treatment": "no", "epidemiology": "no"}
            return base | {k: "yes" for k in tagged}
        rows = [row("treatment", treatment=True), row("treatment", treatment=True), row("treatment"),  # 3 under treatment, 2 tagged
                row("appearance", appearance=True), row("appearance", treatment=True),  # 2 under appearance, 1 tagged; a treatment tag
                row("histology"),  # 1 under histology, none tagged
                row("", appearance=True)]  # no title: not counted
        check = enrich.agreement(rows)
        treatment = check["treatment"]  # tagged with a title: 3 (two right, one under appearance); base 3 of 6
        self.assertEqual((treatment["under_title"], treatment["tagged_under_title"]), (3, 3))
        self.assertAlmostEqual(treatment["recall"], 2 / 3)
        self.assertAlmostEqual(treatment["precision"], 2 / 3)
        self.assertAlmostEqual(treatment["base"], 0.5)
        self.assertAlmostEqual(treatment["lift"], (2 / 3) / 0.5)
        appearance = check["appearance"]  # tagged with a title: 1 (the untitled one is left out), right; base 2 of 6
        self.assertEqual((appearance["tagged_under_title"], appearance["recall"], appearance["precision"]), (1, 0.5, 1.0))
        self.assertAlmostEqual(appearance["lift"], 1.0 / (2 / 6))
        self.assertEqual((check["histology"]["recall"], check["histology"]["precision"], check["histology"]["lift"]), (0.0, None, None))
        self.assertIsNone(check["epidemiology"]["recall"])  # no passage under an epidemiology title


class TextFolder(unittest.TestCase):
    """data/text/ with library.csv and the passages' texts, and data/multimodal/ beside it."""

    PASSAGES = [("pmc-PMC1-1", "pmc", "PMC1", "Treatment / Management", TREATMENT, "melanoma"),
                ("pmc-PMC1-2", "pmc", "PMC1", "Dermoscopy", DERMOSCOPY, "melanoma"),
                ("pmc-PMC2-1", "pmc", "PMC2", "Epidemiology", EPIDEMIOLOGY, "nevus; melanoma"),
                ("book-NBK3-1", "textbook", "StatPearls NBK3", "Introduction", "A short general text about the disease and its names " * 3, "nevus")]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.text, self.data = self.base / "data" / "text", self.base / "data" / "multimodal"
        self.data.mkdir(parents=True)
        rows, texts = [], []
        for pid, corpus, source, section, body, topics in self.PASSAGES:
            rows.append({"passage_id": pid, "corpus": corpus, "source": source, "title": "T", "section": section, "year": "2024",
                         "licence": "CC BY", "topics": topics, "words": len(body.split()), "sha256": sha(body)})
            texts.append({**{k: v for k, v in rows[-1].items() if k not in ("words", "sha256")}, "text": body})
        write_csv(self.text / "library.csv", rows)
        (self.text / "library").mkdir()
        (self.text / "library" / "passages.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts), encoding="utf-8")

    def table(self, name):
        with (self.data / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


class AspectsTests(TextFolder):
    def test_every_passage_is_tagged_and_the_coverage_is_counted_per_class(self):
        with mock.patch.object(enrich, "TARGET_PASSAGES", 1), mock.patch.object(enrich, "TARGET_DOCUMENTS", 1):
            done = enrich.aspects(self.text, self.data)
        rows = {r["passage_id"]: r for r in self.table(enrich.ASPECTS_TABLE)}
        self.assertEqual(set(rows), {p[0] for p in self.PASSAGES})
        self.assertEqual((rows["pmc-PMC1-1"]["treatment"], rows["pmc-PMC1-1"]["appearance"], rows["pmc-PMC1-1"]["heading_aspect"]), ("yes", "no", "treatment"))
        self.assertEqual((rows["pmc-PMC1-2"]["dermoscopy"], rows["pmc-PMC1-2"]["appearance"], rows["pmc-PMC1-2"]["dermoscopy_words"]), ("yes", "yes", "4"))
        self.assertEqual((rows["pmc-PMC2-1"]["epidemiology"], rows["book-NBK3-1"]["heading_aspect"]), ("yes", ""))
        coverage = done["coverage"]
        self.assertEqual((coverage["melanoma"]["passages"], coverage["melanoma"]["total"], coverage["melanoma"]["documents"]), (3, 1, 1))
        self.assertEqual((coverage["nevus"]["passages"], coverage["nevus"]["total"], coverage["nevus"]["meets_target"]), (2, 0, False))
        self.assertEqual(done["short"], ["nevus"])
        self.assertTrue(coverage["melanoma"]["meets_target"])
        report = (self.data / enrich.REPORT).read_text(encoding="utf-8")
        for expected in ("<!-- step: aspects -->", "Do the tags follow the authors' section titles?", "Classes short of the target: nevus",
                         "the target is not lowered"):
            self.assertIn(expected, report)

    def test_a_class_meets_the_target_only_when_both_its_numbers_are_reached(self):
        for passages, documents, short in ((1, 1, ["nevus"]), (1, 2, ["melanoma", "nevus"]), (2, 1, ["melanoma", "nevus"]), (0, 0, ["melanoma", "nevus"])):
            with self.subTest((passages, documents)), mock.patch.object(enrich, "TARGET_PASSAGES", passages), \
                    mock.patch.object(enrich, "TARGET_DOCUMENTS", documents):
                self.assertEqual(enrich.aspects(self.text, self.data)["short"], [] if (passages, documents) == (0, 0) else short)

    def test_the_target_is_not_lowered_for_a_short_class(self):
        done = enrich.aspects(self.text, self.data)  # the real target: 30 passages from 8 documents
        self.assertEqual(done["short"], ["melanoma", "nevus"])
        self.assertEqual((enrich.TARGET_PASSAGES, enrich.TARGET_DOCUMENTS), (30, 8))

    def test_a_changed_text_or_a_missing_text_stops_the_step(self):
        path = self.text / "library" / "passages.jsonl"
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        lines[0]["text"] += " changed"
        path.write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not the one its fingerprint"):
            enrich.aspects(self.text, self.data)
        path.write_text("".join(json.dumps(t) + "\n" for t in lines[1:]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not list the same passages"):
            enrich.aspects(self.text, self.data)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "passages.jsonl not found"):
            enrich.aspects(self.text, self.data)

    def test_the_appearance_corpus_is_read_beside_the_library(self):
        row = {"passage_id": "appearance-PMC9-1", "corpus": "appearance", "source": "PMC9", "title": "T", "section": "Dermoscopy", "year": "2025",
               "licence": "CC BY", "topics": "nevus", "words": len(VESSELS.split()), "sha256": sha(VESSELS)}
        write_csv(self.data / enrich.APPEARANCE_TABLE, [row])
        (self.text / "library" / enrich.APPEARANCE_FILE).write_text(json.dumps({**row, "text": VESSELS}) + "\n", encoding="utf-8")
        done = enrich.aspects(self.text, self.data)
        self.assertEqual((done["coverage"]["nevus"]["library"], done["coverage"]["nevus"]["new"], done["coverage"]["nevus"]["total"]), (0, 1, 1))
        self.assertEqual({r["origin"] for r in done["rows"]}, {"library", "appearance"})
        (self.text / "library" / enrich.APPEARANCE_FILE).unlink()  # a table without its texts
        with self.assertRaisesRegex(FileNotFoundError, "run python -m src.multimodal.enrich appearance first"):
            enrich.aspects(self.text, self.data)


class AppearanceTests(TextFolder):
    """The step with NCBI replaced: the reviews it finds, their records and their full texts are invented."""

    def run_step(self, years=("2024", "2025"), **limits):
        records = {"101": {"pmid": "101", "pmcid": "PMC1", "title": "Review one", "year": years[0]},
                   "102": {"pmid": "102", "pmcid": "PMC2", "title": "Review two", "year": years[1]}}
        bodies = {"1": {"licence": "CC BY 4.0", "sections": [["Dermoscopy", [DERMOSCOPY]], ["Treatment", [TREATMENT]], ["Features", [PASSAGE_IN_LIBRARY]],
                                                             ["Presentation", [CLINICAL]]]},
                  "2": {"licence": "CC BY-NC 4.0", "sections": [["Dermoscopic findings", [VESSELS]]]}}

        def full_texts(chosen, store, say, what, cap=None):
            store.update({r["pmcid"].removeprefix("PMC"): bodies[r["pmcid"].removeprefix("PMC")] for r in chosen})
            return chosen[:cap]

        patches = [mock.patch.object(textlib, "_checked_headings", return_value=([], {"melanoma": ["Melanoma"], "nevus": ["Nevi"]})),
                   mock.patch.object(textlib, "_question_runs", return_value=enrich.textlib._runs(CLINICAL)),
                   mock.patch.object(textlib, "_dataset_articles", return_value=set()),
                   mock.patch.object(textlib, "_newest", return_value={"melanoma": ["101", "102"], "nevus": ["102"]}),
                   mock.patch.object(textlib, "_records", return_value=records),
                   mock.patch.object(textlib, "_full_texts", side_effect=full_texts)]
        patches += [mock.patch.object(enrich, name, value) for name, value in limits.items()]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return enrich.appearance(self.text, self.data, classes=["melanoma", "nevus"])

    def setUp(self):
        super().setUp()
        rows = [{"passage_id": "pmc-OLD-1", "corpus": "pmc", "source": "PMC0", "title": "T", "section": "S", "year": "2020",
                                   "licence": "CC BY", "topics": "melanoma", "words": 40, "sha256": sha(PASSAGE_IN_LIBRARY)}]
        write_csv(self.text / "library.csv", rows)  # the library already holds one of the passages
        (self.text / "library" / "passages.jsonl").write_text(json.dumps({**rows[0], "text": PASSAGE_IN_LIBRARY}) + "\n", encoding="utf-8")

    def test_only_new_passages_that_describe_appearance_and_share_no_run_with_a_question_are_kept(self):
        done = self.run_step()
        kept = {r["passage_id"]: r for r in self.table(enrich.APPEARANCE_TABLE)}
        self.assertEqual(set(kept), {"appearance-PMC1-1", "appearance-PMC2-1"})  # the dermoscopy passages of both reviews
        first = kept["appearance-PMC1-1"]
        self.assertEqual((first["corpus"], first["source"], first["section"], first["licence"], first["topics"]),
                         ("appearance", "PMC1", "Dermoscopy", "CC BY 4.0", "melanoma"))
        self.assertEqual(kept["appearance-PMC2-1"]["topics"], "melanoma; nevus")  # a review found for both classes
        self.assertEqual(kept["appearance-PMC2-1"]["sha256"], sha(VESSELS))
        self.assertEqual(dict(done["dropped"]), {"does not describe appearance": 1, "already in the library": 1,
                                                 "shares a run of words with a question": 1})
        texts = [json.loads(line) for line in (self.text / "library" / enrich.APPEARANCE_FILE).read_text(encoding="utf-8").splitlines()]
        self.assertEqual({t["passage_id"]: t["text"] for t in texts}, {"appearance-PMC1-1": DERMOSCOPY, "appearance-PMC2-1": VESSELS})
        report = (self.data / enrich.REPORT).read_text(encoding="utf-8")
        for expected in ("<!-- step: appearance -->", "| melanoma | Melanoma | first | 2 | 0 | 2 | 2 | 2 |", "| nevus | Nevi | first | 1 | 0 | 1 | 1 | 1 |",
                         "CC BY 4.0: 1", "No person has read the passages", "already in the library: 1"):
            self.assertIn(expected, report)
        self.assertEqual((self.text / "library.csv").read_text(encoding="utf-8").count("\n"), 2)  # the library's own table is not rewritten

    def test_at_most_so_many_passages_are_kept_per_class_newest_first(self):
        done = self.run_step(MAX_PER_CLASS=1)
        self.assertEqual([r["passage_id"] for r in done["rows"]], ["appearance-PMC2-1"])  # 2025 before 2024; it fills both classes
        self.assertEqual(done["dropped"]["over the limit of 1 per class"], 1)

    def test_a_passage_of_two_classes_is_kept_while_one_of_them_has_room(self):
        done = self.run_step(years=("2026", "2025"), MAX_PER_CLASS=1)  # the newest passage fills melanoma; the next is for melanoma and nevus
        self.assertEqual([r["passage_id"] for r in done["rows"]], ["appearance-PMC1-1", "appearance-PMC2-1"])
        self.assertEqual(done["rows"][1]["topics"], "melanoma; nevus")
        self.assertFalse(done["dropped"].get("over the limit of 1 per class"))

    def wider_step(self, **limits):
        """Nevus is a class of the wider search: found by its heading or by its names, without the dermoscopy words."""
        columns = ["class", "term", "counts", "in_mesh", "mesh_ids", "mesh_headings", "pubmed_articles"]
        write_csv(self.text / "term_check.csv", [dict(zip(columns, row)) for row in (
            ("nevus", "nevus", "yes", "heading", "D1", "Nevi", "50"), ("nevus", "blue nevus", "yes", "no", "", "", "3"),
            ("nevus", "pale nevus", "yes", "no", "", "", "0"), ("nevus", "pale", "no", "no", "", "", "4"),  # invented: a name nobody searches
            ("nevus", "dermal nevus", "yes", "no", "", "", "0"), ("nevus", "spider nevus", "no", "no", "", "", "4"),
            ("melanoma", "melanoma", "yes", "heading", "D2", "Melanoma", "90"))])
        sections = [["", "A review."]]
        records = {"101": {"pmid": "101", "pmcid": "PMC1", "title": "Review one", "year": "2024", "sections": sections},
                   "102": {"pmid": "102", "pmcid": "PMC2", "title": "Review two", "year": "2025", "sections": sections},
                   "103": {"pmid": "103", "pmcid": "PMC3", "title": "The blue nevus", "year": "2023", "sections": sections},
                   "104": {"pmid": "104", "pmcid": "PMC4", "title": "The spider nevus", "year": "2022", "sections": sections},
                   "106": {"pmid": "106", "pmcid": "PMC6", "title": "The pale nevus", "year": "2021", "sections": sections}}
        bodies = {"1": {"licence": "CC BY 4.0", "sections": [["Dermoscopy", [DERMOSCOPY]]]},
                  "2": {"licence": "CC BY-NC 4.0", "sections": [["Dermoscopic findings", [VESSELS]]]},
                  "3": {"licence": "CC BY 4.0", "sections": [["Dermoscopy", [BLUE]]]},
                  "6": {"licence": "CC BY 4.0", "sections": []}}  # the pale nevus: its name stands inside a longer name that counts  # the spider nevus review has no full text to ask for
        asked, searched = [], []

        def full_texts(chosen, store, say, what, cap=None):
            asked.append((what, [r["pmid"] for r in chosen]))
            store.update({r["pmcid"].removeprefix("PMC"): bodies[r["pmcid"].removeprefix("PMC")] for r in chosen})
            return chosen[:cap]

        def search(db, term, retmax=0, **extra):
            searched.append((term, retmax, extra))
            return {"idlist": ["102", "105", "103", "104", "106"]}  # 105 is an article of the dataset

        patches = [mock.patch.object(textlib, "_checked_headings", return_value=([], {"melanoma": ["Melanoma"], "nevus": ["Nevi"]})),
                   mock.patch.object(textlib, "_question_runs", return_value=enrich.textlib._runs(CLINICAL)),
                   mock.patch.object(textlib, "_dataset_articles", return_value={"105"}),
                   mock.patch.object(textlib, "_newest", return_value={"melanoma": ["101", "102"], "nevus": ["102"]}),
                   mock.patch.object(textlib, "_search", side_effect=search),
                   mock.patch.object(textlib, "_records", return_value=records),
                   mock.patch.object(textlib, "_full_texts", side_effect=full_texts),
                   mock.patch.object(enrich, "WIDER_CLASSES", ("nevus",))]
        patches += [mock.patch.object(enrich, name, value) for name, value in limits.items()]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return enrich.appearance(self.text, self.data, classes=["melanoma", "nevus"]), asked, searched

    def test_a_wider_class_is_searched_by_its_heading_or_its_names_without_the_dermoscopy_words(self):
        _, _, searched = self.wider_step()
        self.assertEqual(len(searched), 1)  # melanoma keeps its first search
        term, retmax, extra = searched[0]
        self.assertEqual((retmax, extra), (enrich.REVIEWS_ASKED, {"sort": "pub_date"}))
        self.assertTrue(term.startswith('(("Nevi"[MeSH Major Topic]) OR ("nevus"[Title/Abstract] OR "blue nevus"[Title/Abstract])) AND '), term)
        self.assertNotIn("dermal nevus", term)  # no article of PubMed uses it: PubMed would not understand the phrase
        self.assertNotIn("spider nevus", term)
        self.assertNotIn("dermoscopy[Title/Abstract]", term)
        for part in ("review[Publication Type]", "free full text[sb]", 'NOT "case reports"[Publication Type]', "english[Language]"):
            self.assertIn(part, term)

    def test_a_wider_class_leaves_out_the_dataset_s_articles_and_the_reviews_that_name_another_condition(self):
        done, asked, _ = self.wider_step()
        self.assertEqual(asked, [("melanoma", ["101", "102"]), ("nevus", ["102", "103", "106"])])  # 105 is the dataset's, 104 names the spider nevus
        kept = {r["passage_id"]: r for r in self.table(enrich.APPEARANCE_TABLE)}
        self.assertEqual(set(kept), {"appearance-PMC1-1", "appearance-PMC2-1", "appearance-PMC3-1"})
        self.assertEqual(kept["appearance-PMC3-1"]["topics"], "nevus")
        self.assertEqual(kept["appearance-PMC2-1"]["topics"], "melanoma; nevus")
        report = (self.data / enrich.REPORT).read_text(encoding="utf-8")
        self.assertIn("| nevus | Nevi | wider | 4 | 1 (spider nevus (1)) | 3 | 2 | 2 |", report)
        self.assertIn("- nevus: searched for nevus; blue nevus; not counted: pale; spider nevus", report)
        self.assertIn("| melanoma | Melanoma | first | 2 | 0 | 2 | 2 | 2 |", report)

    def test_the_wider_search_is_for_the_three_classes_the_first_search_left_short(self):
        self.assertEqual(enrich.WIDER_CLASSES, ("dermatofibroma", "pigmented benign keratosis", "vascular lesion"))

    def test_a_class_that_is_not_wider_keeps_its_first_search_and_leaves_no_review_out(self):
        done, asked, _ = self.wider_step(WIDER_CLASSES=())  # nothing wider: nevus is searched as before, from the patched first search
        self.assertEqual(asked, [("melanoma", ["101", "102"]), ("nevus", ["102"])])
        self.assertIn("| nevus | Nevi | first | 1 | 0 | 1 | 1 | 1 |", (self.data / enrich.REPORT).read_text(encoding="utf-8"))

    def test_the_query_asks_for_reviews_with_a_free_full_text_that_mention_dermoscopy_or_morphology(self):
        for part in ("review[Publication Type]", 'NOT "case reports"[Publication Type]', "free full text[sb]", "dermoscopy[Title/Abstract]",
                     "morphology[Title/Abstract]", "english[Language]", "NOT (animals[MeSH Terms] NOT humans[MeSH Terms])"):
            self.assertIn(part, enrich.APPEARANCE_QUERY)


BLUE = ("Dermoscopy of the blue nevus shows a homogeneous blue-gray pigmentation without any pigment network or globules, a finding "
        "that is helpful to tell this benign lesion from a nodular melanoma in the clinic and in the reading of images by experts.")


class NameTests(unittest.TestCase):
    """A review that names a condition that does not count is left out, unless the name only stands inside a longer name that counts."""

    def judge(self, title, abstract, counts, not_counts):
        return enrich.other_condition({"title": title, "sections": [["", abstract]]}, counts, not_counts)

    def test_a_name_that_does_not_count_is_found_in_the_title_or_the_abstract(self):
        counts, not_counts = ["lobular capillary hemangioma", "pyogenic granuloma"], ["hemangioma", "port-wine stain"]
        self.assertIsNone(self.judge("Pyogenic granuloma", "A common benign growth of vessels of the skin.", counts, not_counts))
        self.assertEqual(self.judge("Pyogenic granuloma", "It is not an infantile hemangioma.", counts, not_counts), "hemangioma")
        self.assertEqual(self.judge("Port-wine stains and pyogenic granuloma", "A review.", counts, not_counts), "port-wine stain")  # plural
        self.assertEqual(self.judge("Hemangiomas of infancy", "A review of pyogenic granuloma.", counts, not_counts), "hemangioma")

    def test_a_name_that_stands_inside_a_longer_name_that_counts_does_not_refuse_the_review(self):
        counts, not_counts = ["lobular capillary hemangioma"], ["hemangioma"]
        self.assertIsNone(self.judge("Lobular capillary hemangioma", "Lobular capillary hemangioma of the lip.", counts, not_counts))
        self.assertEqual(self.judge("Lobular capillary hemangioma", "Unlike a hemangioma of infancy.", counts, not_counts), "hemangioma")

    def test_a_longer_name_that_does_not_count_is_not_hidden_by_a_shorter_name_that_counts(self):
        self.assertEqual(self.judge("Spider nevus", "A review.", ["nevus"], ["spider nevus"]), "spider nevus")
        self.assertIsNone(self.judge("Nevus", "A review.", ["nevus"], ["spider nevus"]))

    def test_the_names_of_a_class_come_from_the_checked_survey(self):
        with tempfile.TemporaryDirectory() as tmp:
            columns = ["class", "term", "counts", "in_mesh", "mesh_ids", "mesh_headings", "pubmed_articles"]
            write_csv(Path(tmp) / "term_check.csv", [dict(zip(columns, row)) for row in (
                ("nevus", "nevus", "yes", "heading", "D1", "Nevus", "50"), ("nevus", "dermal nevus", "yes", "no", "", "", "0"),
                ("nevus", "spider nevus", "no", "no", "", "", "4"), ("melanoma", "melanoma", "yes", "heading", "D2", "Melanoma", "90"))])
            self.assertEqual(enrich._class_names(Path(tmp), "nevus"), (["nevus"], ["nevus", "dermal nevus"], ["spider nevus"]))


PASSAGE_IN_LIBRARY = ("Dermoscopy often shows a regular pigment network with evenly distributed globules in benign melanocytic lesions, "
                      "which helps to separate them from melanoma in daily practice according to many published series.")


class CardTests(Project):
    """The ISIC records of the multimodal tests, with the columns the cards need."""

    def setUp(self):
        super().setUp()
        photos = prepare._read_csv(self.image / "raw_photos.csv")
        for row in photos:
            row["diagnosis_confirm_type"] = "histopathology" if row["photo_id"] == "ISIC_0000002" else ""
        write_csv(self.image / "raw_photos.csv", photos)
        prepare.records()
        prepare.build()

    def test_a_card_for_each_training_photo_from_its_own_record(self):
        done = enrich.cards()
        by_id = {c["item_id"]: c for c in done["cards"]}
        self.assertEqual(set(by_id), {"image:train/melanoma/ISIC_0000002.jpg", "image:train/basal cell carcinoma/ISIC_0000004.jpg"})
        self.assertEqual(by_id["image:train/melanoma/ISIC_0000002.jpg"]["card"],
                         "Dermoscopic photo of a skin lesion. Diagnosis (ISIC): Diagnosis of melanoma. Confirmed by: histopathology. "
                         "Age: about 70 years. Sex: female. Body site: lower extremity. Lesion size: 8 mm (longest diameter). "
                         "Family history of melanoma: no.")
        second = by_id["image:train/basal cell carcinoma/ISIC_0000004.jpg"]
        self.assertEqual(second["card"], "Dermoscopic photo of a skin lesion. Diagnosis (ISIC): Diagnosis of basal cell carcinoma. "
                                         "Confirmed by: not recorded. Age: about 75 years. Sex: male. Body site: head and neck. "
                                         "Family history of melanoma: no.")  # no size recorded: left out, not guessed
        rows = self.table(enrich.CARDS_TABLE)
        self.assertEqual([r["source"] for r in rows], ["ISIC ISIC_0000004", "ISIC ISIC_0000002"])  # by class, then file
        report = (self.data / enrich.REPORT).read_text(encoding="utf-8")
        self.assertIn("<!-- step: cards -->", report)
        self.assertIn("| melanoma | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |", report)

    def test_no_validation_or_test_photo_has_a_card_and_no_card_names_a_collection(self):
        cards = enrich.cards()["cards"]
        text = " ".join(c["card"] for c in cards)
        self.assertNotIn("ISIC_0000001", text + " ".join(c["item_id"] for c in cards))  # a test photo
        self.assertNotIn("ISIC_0000005", " ".join(c["item_id"] for c in cards))  # a validation photo
        for name in ("Anonymous", "MILK study team", "CC-0", "CC-BY-NC"):
            self.assertNotIn(name, text)

    def test_a_card_that_breaks_a_rule_stops_the_step(self):
        photos = prepare._read_csv(self.image / "raw_photos.csv")
        for row in photos:  # a confirmation text that names the collection
            if row["photo_id"] == "ISIC_0000002":
                row["diagnosis_confirm_type"] = "reviewed by Anonymous"
        write_csv(self.image / "raw_photos.csv", photos)
        with self.assertRaisesRegex(ValueError, "names the photo's collection"):
            enrich.cards()
        diseases = prepare._read_csv(self.data / "diseases.csv")  # a diagnosis its class does not list
        for row in diseases:
            row["isic_diagnoses"] = "Something else (1)"
        write_csv(self.data / "diseases.csv", diseases)
        photos[0]["diagnosis_confirm_type"] = "histopathology"
        write_csv(self.image / "raw_photos.csv", photos)
        with self.assertRaisesRegex(ValueError, "not one that diseases.csv lists"):
            enrich.cards()

    def test_the_cards_must_be_exactly_the_training_photos_of_the_knowledge_table(self):
        knowledge = prepare._read_csv(self.data / "knowledge.csv")
        write_csv(self.data / "knowledge.csv", [r for r in knowledge if r["item_id"] != "image:train/melanoma/ISIC_0000002.jpg"])
        with self.assertRaisesRegex(ValueError, "not exactly the training photos"):
            enrich.cards()
        card = {"diagnosis": "d", "confirmation": "c", "age": "a", "sex": "s", "site": "t", "site_detail": "", "size_mm": "", "personal_hx_mm": "",
                "family_hx_mm": ""}
        self.assertEqual(enrich.render_card(card), "Dermoscopic photo of a skin lesion. Diagnosis (ISIC): d. Confirmed by: c. Age: a. Sex: s. Body site: t.")
        self.assertIn("Body site: t (u, v).", enrich.render_card(card | {"site_detail": "u, v"}))

    def test_a_card_whose_text_is_not_made_from_its_columns_stops_the_step(self):
        card = {"item_id": "image:p", "class": "melanoma", "diagnosis": "D", "confirmation": "c", "age": "a", "sex": "s", "site": "t",
                "site_detail": "", "size_mm": "", "personal_hx_mm": "", "family_hx_mm": ""}
        card["card"] = enrich.render_card(card)
        arguments = ([card], {"x": {"attribution": "Anonymous"}}, {}, {"p": {"split": "train"}}, {"image:p"}, {"melanoma": {"D"}})
        self.assertEqual(enrich._card_checks(*arguments)["cards"], 1)
        with self.assertRaisesRegex(ValueError, "not made from their columns"):
            enrich._card_checks([card | {"card": card["card"] + " The lesion is probably benign."}], *arguments[1:])

    def test_a_yes_no_field_is_read_only_when_it_is_one(self):
        self.assertEqual([enrich._flag(v) for v in (True, "True", "true", False, "False", "false", None, "", "maybe", 2)],
                         ["yes", "yes", "yes", "no", "no", "no", None, None, None, None])

    def table(self, name):
        with (self.data / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


class CommandLineTests(unittest.TestCase):
    def test_the_three_steps_are_the_commands(self):
        parser = enrich.build_parser()
        for step in ("appearance", "aspects", "cards"):
            self.assertEqual(parser.parse_args([step]).step, step)
        with self.assertRaises(SystemExit):
            parser.parse_args(["other"])

    def test_the_documentation_names_every_step(self):
        for step in enrich.STEPS:
            self.assertIn(f"python -m src.multimodal.enrich {step}", enrich.__doc__)


if __name__ == "__main__":
    unittest.main()
