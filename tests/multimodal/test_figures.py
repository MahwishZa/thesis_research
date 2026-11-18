"""The free count of the text dataset's figures (src/multimodal/figures.py), on invented data: no network, no real data."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.sax.saxutils import escape

from src.multimodal import figures

COLUMNS = ["split", "label", "path", "sha256", "patient_uid", "pmid", "source", "pubmed", "year", "age", "sex", "words"]


def article(number, figs, body=True, licence="https://creativecommons.org/licenses/by/4.0/", id_type="pmc"):
    figs = "".join(f'<fig id="f{i}"><label><bold>Figure {i}</bold></label><caption><p>{escape(c)}</p></caption><graphic/></fig>'
                   for i, c in enumerate(figs, start=1))
    return (f'<article><front><article-meta><article-id pub-id-type="{id_type}">{number}</article-id><permissions>'
            f'<license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="{licence}"/></permissions></article-meta></front>'
            + ("<body><p>Text.</p></body>" if body else "") + f"<floats-group>{figs}</floats-group></article>")


class JudgeTests(unittest.TestCase):
    def test_captions_by_hand(self):
        for caption, lesion, dermoscopy, reasons in (
                ("Clinical photograph of a pearly nodule on the nose.", True, False, []),
                ("Dermoscopy showing arborizing vessels and ulceration.", True, True, []),
                ("Histopathology (H&E, ×200) showing basaloid nests.", False, False, ["microscope or stain"]),
                ("(a) Clinical image of the lesion. (b) Histology, H&E stain.", True, False, ["microscope or stain"]),
                ("The lesion 6 months after excision.", True, False, ["after treatment or at follow-up"]),
                ("A 15 x 12 mm nodule on the cheek.", True, False, []),  # a size, not a magnification
                ("Original magnification x200.", False, False, ["microscope or stain"]),
                ("Magnetic resonance imaging of the mass.", True, False, ["scan"]),
                ("Clinical picture of patient 2.", True, False, [figures.MAY_BE_ANOTHER]),
                ("Lesions in both patients.", True, False, [figures.MAY_BE_ANOTHER]),
                ("Schematic drawing of the lesion.", True, False, ["drawing or chart"]),
                ("Photograph of the excised specimen.", True, False, ["removed specimen"]),
                ("Bilateral plaques on the legs.", True, False, []),  # "bilateral" is not "later"
                ("Before treatment: a plaque on the arm.", True, False, []),  # the state before treatment is the lesion itself
                ("Timeline of the disease.", False, False, ["drawing or chart"])):
            with self.subTest(caption):
                judged = figures.judge(caption)
                self.assertEqual((judged["lesion"], judged["dermoscopy"], judged["left_out_for"]), (lesion, dermoscopy, reasons))

    def test_a_photograph_is_no_graph(self):
        self.assertEqual(figures.judge("Photograph of the lesion")["left_out_for"], [])

    def test_the_figures_a_question_cites(self):
        self.assertEqual(figures._cited("A nodule (Figure 1A) and a plaque (Fig. 3); see Figs. 2."), {1, 2, 3})
        self.assertEqual(figures._cited("No figure is cited."), set())
        self.assertIsNone(figures._cited(None))


class ParseTests(unittest.TestCase):
    def test_figures_of_open_and_closed_full_texts(self):
        content = ("<pmc-articleset>" + article("11", ["Clinical photo of a nodule.", "H&E stain."])
                   + article("PMC12", ["Dermoscopy."], body=False, id_type="pmcid") + "</pmc-articleset>").encode()
        parsed = figures.parse_figures(content)
        self.assertEqual(set(parsed), {"11", "12"})
        self.assertTrue(parsed["11"]["open"])
        self.assertFalse(parsed["12"]["open"])
        self.assertEqual(parsed["11"]["licence"], "https://creativecommons.org/licenses/by/4.0/")
        first = parsed["11"]["figures"][0]
        self.assertEqual((first["figure"], first["label"], first["number"], first["caption"], first["pictures"]),
                         ("f1", "Figure 1", 1, "Clinical photo of a nodule.", 1))

    def test_a_figure_s_number_comes_from_its_label_else_its_place(self):
        content = ('<pmc-articleset><article><front><article-meta><article-id pub-id-type="pmc">9</article-id></article-meta></front>'
                   '<body><fig id="a"><label>Fig. 3</label></fig><fig id="b"/></body></article></pmc-articleset>').encode()
        found = figures.parse_figures(content)["9"]["figures"]
        self.assertEqual([(f["figure"], f["number"], f["caption"], f["pictures"]) for f in found], [("a", 3, "", 0), ("b", 2, "", 0)])
        self.assertEqual(figures.parse_figures(content)["9"]["licence"], "")

    def test_an_unexpected_answer_stops(self):
        with self.assertRaisesRegex(ValueError, "unexpected form"):
            figures.parse_figures(b"<html/>")


class CountTests(unittest.TestCase):
    """Eight cases: one counts, the others show every reason in turn."""

    CASES = [  # patient_uid, split, class, pmid, question
        ("c1", "test", "melanoma", "1", "A man had a dark nodule on his back (Figure 1)."),
        ("c2", "test", "melanoma", "2", "A woman had a dark macule on her leg."),
        ("c3", "test", "nevus", "3", "A child had a brown papule."),
        ("c4", "test", "nevus", "4", "A girl had a mole on the arm."),
        ("c5", "test", "nevus", "5", "A boy had a blue papule."),
        ("c6", "val", "nevus", "5", "His sister had a blue papule too."),
        ("c7", "val", "nevus", "7", None),  # its question is not on this computer
        ("c8", "train", "melanoma", "8", "A man had a nodule.")]
    PMC = {"1": "PMC11", "2": "", "3": "PMC13", "4": "PMC14", "5": "PMC15", "7": "PMC17", "8": "PMC18"}
    ARTICLES = {"11": article("11", ["Clinical photograph of the nodule.", "Histopathology, H&E."]),
                "13": article("13", ["Dermoscopy of the papule."], body=False),
                "14": article("14", []),
                "15": article("15", ["Dermoscopy of the papule."]),
                "17": article("17", ["(a) Dermoscopy. (b) Histopathology, H&E."], licence="https://creativecommons.org/licenses/by-nc/4.0/"),
                "18": article("18", ["Dermoscopy of patient 2."])}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.text, self.data = base / "data" / "text", base / "data" / "multimodal"
        rows = []
        for uid, split, label, pmid, question in self.CASES:
            path = f"{split}/{label}/{uid}.txt"
            body = (question or "missing").encode("utf-8")
            if question is not None:
                (self.text / "dataset" / split / label).mkdir(parents=True, exist_ok=True)
                (self.text / "dataset" / path).write_bytes(body)
            rows.append(dict(zip(COLUMNS, [split, label, path, hashlib.sha256(body).hexdigest(), uid, pmid, "PMC-Patients", "", "2020",
                                           "", "", "10"])))
        with (self.text / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.asked = []

        def efetch(pmids):
            return [{"pmid": p, "pmcid": self.PMC[p]} for p in pmids]

        def pmc(numbers):
            self.asked.append(list(numbers))
            return figures.parse_figures(("<pmc-articleset>" + "".join(self.ARTICLES.get(n, "") for n in numbers)
                                          + "</pmc-articleset>").encode())
        for patch in (mock.patch.object(figures.textprep, "_efetch", side_effect=efetch),
                      mock.patch.object(figures, "_pmc_figures", side_effect=pmc)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_every_case_counts_or_gets_its_reason(self):
        done = figures.count(self.text, self.data)
        rows = {r["patient_uid"]: r for r in done["rows"]}
        self.assertEqual({uid: r["reason"] for uid, r in rows.items()}, {
            "c1": "", "c2": "no PMC number", "c3": "no open full text in PMC", "c4": "no figure",
            "c5": "the article holds another case of the dataset", "c6": "the article holds another case of the dataset",
            "c7": "no figure whose caption shows the lesion alone", "c8": "no figure whose caption shows the lesion alone"})
        first = rows["c1"]
        self.assertEqual((first["counts"], first["pmcid"], first["figures"], first["lesion_figures"], first["dermoscopy_figures"],
                          first["lesion_figures_cited_in_question"]), ("yes", "PMC11", 2, 1, 0, 1))
        self.assertEqual((rows["c7"]["lesion_figures_with_other_parts"], rows["c7"]["lesion_figures_cited_in_question"]), (1, ""))
        self.assertEqual((rows["c8"]["lesion_figures_with_other_parts"], rows["c8"]["dermoscopy_figures"]), (0, 0))  # left out as possibly another patient, not for other parts
        self.assertEqual(self.asked, [["11", "13", "14", "15", "17", "18"]])  # every PMC number once, never an empty one

    def test_the_tables_and_the_local_captions(self):
        done = figures.count(self.text, self.data)
        with (self.data / figures.COUNT_TABLE).open(encoding="utf-8", newline="") as f:
            table = list(csv.DictReader(f))
        self.assertEqual(list(table[0]), figures.CASE_COLUMNS)
        self.assertNotIn("Clinical photograph of the nodule", (self.data / figures.COUNT_TABLE).read_text(encoding="utf-8"))  # no caption in Git
        captions = [json.loads(line) for line in (self.data / figures.CAPTIONS).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(captions), done["captions"])
        self.assertEqual(len(captions), 6)  # per case, its open article's figures: c1 2, c5 1, c6 1 (the same article), c7 1, c8 1
        self.assertIn({"patient_uid": "c1", "pmcid": "PMC11", "figure": "f1", "label": "Figure 1", "number": 1,
                       "caption": "Clinical photograph of the nodule.", "pictures": 1, "dermoscopy": False, "clinical": True,
                       "lesion": True, "left_out_for": []}, captions)
        report = (self.data / figures.COUNT_REPORT).read_text(encoding="utf-8")
        for expected in ("**Result: 1 test cases count; 0 classes have at least 8 (none). The rule is not met: the dataset is not built.**",
                         "| melanoma | 2 | 1 | 0 | 0 | 0 | 1 | 0 |", "| all | 5 | 1 | 0 | 2 | 0 | 1 | 0 |",
                         "| no PMC number | 1 |", "| the article holds another case of the dataset | 2 |",
                         "| no figure whose caption shows the lesion alone | 2 |", "1 have a figure that shows the lesion beside other parts",
                         "| microscope or stain | 1 |", f"| {figures.MAY_BE_ANOTHER} | 1 |",
                         "Cases whose question cites one of these figures by number: 1 of 1", "CC BY (any, no NC): 1"):
            self.assertIn(expected, report)

    def test_a_question_that_is_not_the_frozen_one_stops_the_count(self):
        (self.text / "dataset" / "test" / "melanoma" / "c1.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            figures.count(self.text, self.data)

    def test_the_rule_decides_with_the_counted_test_cases(self):
        with mock.patch.object(figures, "GO_TEST_CASES", 1), mock.patch.object(figures, "GO_CLASSES", 1), \
                mock.patch.object(figures, "GO_PER_CLASS", 1):
            done = figures.count(self.text, self.data)
        self.assertEqual(done["decision"], {"test_cases": 1, "per_class": {"melanoma": 1}, "classes_with_enough": ["melanoma"], "build": True})
        self.assertIn("The rule is met: the dataset can be built.", (self.data / figures.COUNT_REPORT).read_text(encoding="utf-8"))


class ReportTests(unittest.TestCase):
    def test_a_case_that_cites_no_figure_is_not_counted_as_citing_one(self):
        row = {"split": "test", "class": "melanoma", "counts": "yes", "reason": "", "lesion_figures": 1, "dermoscopy_figures": 0,
               "lesion_figures_with_other_parts": 0, "licence": ""}
        rows = [row | {"lesion_figures_cited_in_question": n} for n in (0, 1, 2, "")]
        report = "\n".join(figures._report(rows, [], figures.go(rows)))
        self.assertIn("Cases whose question cites one of these figures by number: 2 of 3 whose question is on this computer", report)


class RuleTests(unittest.TestCase):
    def rows(self, per_class):
        return [{"split": "test", "class": label, "counts": "yes"} for label, n in per_class.items() for _ in range(n)] + \
               [{"split": "val", "class": "x", "counts": "yes"}] * 50 + [{"split": "test", "class": "y", "counts": "no"}] * 50

    def test_the_rule_needs_enough_cases_and_enough_classes(self):
        self.assertEqual((figures.GO_TEST_CASES, figures.GO_CLASSES, figures.GO_PER_CLASS), (80, 6, 8))  # fixed before the count
        self.assertTrue(figures.go(self.rows({f"c{i}": 14 for i in range(6)}))["build"])  # 84 cases, 6 classes with 14
        self.assertFalse(figures.go(self.rows({**{f"c{i}": 16 for i in range(5)}, "c5": 7}))["build"])  # 87 cases, 5 classes with 8+
        self.assertFalse(figures.go(self.rows({**{f"c{i}": 13 for i in range(5)}, "c5": 8, "c6": 6}))["build"])  # 79 cases
        self.assertTrue(figures.go(self.rows({**{f"c{i}": 12 for i in range(5)}, "c5": 8, "c6": 12}))["build"])  # 80 cases, 6 with 8+

    def test_licence_kinds(self):
        for licence, kind in (("https://creativecommons.org/licenses/by/4.0/", "CC BY (any, no NC)"),
                              ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC BY-NC (any)"),
                              ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0"), ("open-access", "other: open-access"),
                              ("", "not stated")):
            self.assertEqual(figures.licence_kind(licence), kind)


class CommandLineTests(unittest.TestCase):
    def test_the_only_step_is_count(self):
        self.assertEqual(figures.build_parser().parse_args(["count"]).step, "count")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            figures.build_parser().parse_args(["build"])

    def test_an_error_is_one_line(self):
        with mock.patch.object(figures, "count", side_effect=FileNotFoundError("data/text/dataset.csv not found")):
            with self.assertRaisesRegex(SystemExit, "^ERROR: data/text/dataset.csv not found$"):
                figures.main(["count"])


if __name__ == "__main__":
    unittest.main()
