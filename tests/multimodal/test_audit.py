"""The multimodal dataset's audit (src/multimodal/audit.py) on the tiny synthetic project of test_prepare.py: no network."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path

from src.multimodal import audit, prepare
from tests.multimodal.test_prepare import Project


class AuditTests(Project):
    """The project with its files on disk, as on the researcher's computer, and the multimodal tables built."""

    def setUp(self):
        super().setUp()
        for row in self._source("image", "dataset.csv"):  # each image file holds its number, whose hash the table records
            file = self.image / "dataset" / row["path"]
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(row["photo_id"].encode())
        text = self.base / "data" / "text"
        cases = self._source("text", "dataset.csv")
        for row in cases:
            content = f"Case {row['patient_uid']}: a pigmented papule on the back.".encode()
            row["sha256"] = hashlib.sha256(content).hexdigest()
            file = text / "dataset" / row["path"]
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
        self._write(text / "dataset.csv", cases)
        passages = self._source("text", "library.csv")
        lines = []
        for row in passages:
            body = f"Passage {row['passage_id']} about {row['topics']}."
            row["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
            lines.append(json.dumps({"passage_id": row["passage_id"], "text": body}))
        self._write(text / "library.csv", passages)
        (text / "library").mkdir()
        (text / "library" / "passages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        prepare.records()
        prepare.build()

    def _source(self, dataset, name):
        with (self.base / "data" / dataset / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _failed(self):
        return {check: ids for check, ids in audit.audit()["failures"].items() if ids}

    def test_a_correct_build_passes_and_the_report_keeps_every_step_in_order(self):
        self.assertEqual(self._failed(), {})
        report = (self.data / "dataset_report.md").read_text(encoding="utf-8")
        self.assertIn("**All checks passed.**", report)
        self.assertLess(report.index("<!-- step: build -->"), report.index("<!-- step: audit -->"))
        prepare.build()  # building again keeps the audit's section
        self.assertIn("<!-- step: audit -->", (self.data / "dataset_report.md").read_text(encoding="utf-8"))

    def test_facts_are_worked_out_again_from_the_records(self):
        self.assertEqual(audit.expected_facts([{"metadata": {"clinical": {"age_approx": 65.0, "sex": "male", "anatom_site_1": "Trunk",
                                                                         "anatom_site_2": "Posterior trunk"}}}]),
                         {"age": "about 65 years", "sex": "male", "site": "trunk"})
        two = [{"metadata": {"clinical": {"age_approx": 80, "sex": "male"}}}, {"metadata": {"clinical": {"age_approx": 70, "sex": "male"}}}]
        self.assertEqual(audit.expected_facts(two), {"age": "not recorded", "sex": "male", "site": "not recorded"})
        rows = self._source("multimodal", "questions.csv")
        rows[0]["facts"] = rows[0]["facts"].replace("Sex: male", "Sex: female")
        rows[1]["site"] = "posterior trunk"
        self._write(self.data / "questions.csv", rows)
        self.assertEqual(len(self._failed()["facts differ from the ISIC records"]), 2)

    def test_a_held_out_item_in_the_knowledge_or_a_changed_file_fails(self):
        knowledge = self._source("multimodal", "knowledge.csv")
        knowledge.append({**knowledge[0], "item_id": "image:test/melanoma/ISIC_0000001.jpg",
                          "path": "data/image/dataset/test/melanoma/ISIC_0000001.jpg"})
        knowledge = [k for k in knowledge if k["item_id"] != "passage:book-2-1"]
        self._write(self.data / "knowledge.csv", knowledge)
        (self.image / "dataset" / "test" / "nevus" / "ISIC_0000003.jpg").write_bytes(b"changed")
        (self.base / "data" / "text" / "dataset" / "train" / "nevus" / "13-1.txt").unlink()
        failed = self._failed()
        self.assertEqual(failed["validation or test item in the knowledge"], ["image:test/melanoma/ISIC_0000001.jpg"])
        self.assertEqual(failed["knowledge item missing"], ["passage:book-2-1"])
        self.assertEqual(failed["image file missing or changed"], ["test/nevus/ISIC_0000003.jpg"])
        self.assertEqual(failed["case file missing or changed"], ["case:13-1"])
        self.assertIn("**", (self.data / "dataset_report.md").read_text(encoding="utf-8").split("checks failed")[0][-6:])

    def test_a_case_licence_that_differs_from_its_source_fails(self):
        knowledge = self._source("multimodal", "knowledge.csv")
        for k in knowledge:
            if k["item_id"] == "case:pubmed-16":
                k["licence"] = "the article's open-access licence in PMC"  # vague: not the licence PMC gives this full text
        self._write(self.data / "knowledge.csv", knowledge)
        self.assertEqual(self._failed(), {"knowledge case licence differs from its source": ["case:pubmed-16"]})

    def test_a_passage_from_a_candidate_article_and_a_wrong_count_fail(self):
        path = self.base / "data" / "text" / "library.csv"
        rows = self._source("text", "library.csv")
        rows[0]["source"] = "PMID 13"  # the article of a case of the text dataset
        self._write(path, rows)
        prepare.build()
        diseases = self._source("multimodal", "diseases.csv")
        diseases[0]["passages"] = "9"
        self._write(self.data / "diseases.csv", diseases)
        failed = self._failed()
        self.assertEqual(failed["passage from an article of the text dataset's candidates"], ["pmc-1-1"])
        self.assertEqual(failed["disease count differs from the sources"], [diseases[0]["class"]])

    def test_the_facts_alone_are_scored_against_chance(self):
        model = audit._counting_model([{"age": "about 30 years", "label": "nevus"}, {"age": "about 70 years", "label": "melanoma"},
                                       {"age": "about 70 years", "label": "melanoma"}, {"age": "about 30 years", "label": "melanoma"}],
                                      ("age",), "label")
        self.assertEqual(model({"age": "about 30 years"}), "nevus")  # every class weighs the same: 1 of 1 nevi, 1 of 3 melanomas
        self.assertEqual(model({"age": "about 70 years"}), "melanoma")
        shortcuts = {r["facts"]: r for r in audit.audit()["shortcuts"]}
        self.assertEqual(set(shortcuts), {"age", "sex", "body site", "which facts are missing", "all three facts",
                                          "all three facts, questions with all three"})
        self.assertEqual((shortcuts["age"]["n"], shortcuts["all three facts, questions with all three"]["n"]), (2, 1))
        report = (self.data / "dataset_report.md").read_text(encoding="utf-8")
        self.assertIn("| Facts used | Test questions | Class | Photo collection |", report)

    def test_the_command_line_stops_with_an_error_code_when_a_check_fails(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            audit.main([])
        self.assertIn("All checks passed.", shown.getvalue())
        (self.image / "dataset" / "test" / "nevus" / "ISIC_0000003.jpg").unlink()
        with contextlib.redirect_stdout(io.StringIO()) as shown, self.assertRaises(SystemExit) as stopped:
            audit.main([])
        self.assertEqual(stopped.exception.code, 1)
        self.assertIn("FAILED image file missing or changed: 1", shown.getvalue())
