"""data_roles.questions: the three readers, the inline form of a question, the manifest and the files and hub loaders."""

import json
import unittest

from data_roles import questions
from data_roles.questions import AnswersWithheld, QuestionDataError
from tests.support import TempDirTestCase


def medqa_row(**changes):
    row = {
        "question": "A 23-year-old pregnant woman has burning upon urination. Which is the best treatment?",
        "answer": "Nitrofurantoin",
        "options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
        "meta_info": "step2&3",
        "answer_idx": "D",
        "metamap_phrases": ["pregnant woman"],
    }
    row.update(changes)
    return row


def medmcqa_row(**changes):
    row = {
        "id": "4e1715fe-0bc3-494e-b6eb-2d4617245aef",
        "question": "Which term best describes these findings?",
        "opa": "Leukemoid reaction",
        "opb": "Leukopenia",
        "opc": "Myeloid metaplasia",
        "opd": "Neutrophilia",
        "cop": 3,
        "exp": "An explanation.",
        "subject_name": "Pathology",
        "topic_name": None,
        "choice_type": "single",
    }
    row.update(changes)
    return row


def mmlu_row(**changes):
    row = {
        "dataset_key": "mmlu_medical",
        "hash": "abc",
        "split": "test",
        "question_type": "mcq",
        "question": "Which nerve supplies the diaphragm?",
        "context": None,
        "options": [{"original_key": k, "text": t} for k, t in zip("ABCD", ("Vagus", "Phrenic", "Radial", "Ulnar"))],
        "answer": {"original_key": "B", "text": "Phrenic"},
        "metadata": {"subject": "anatomy"},
    }
    row.update(changes)
    return row


class RenderTests(unittest.TestCase):
    def test_the_options_follow_the_stem_inline_with_single_spaces(self):
        item = questions.read_medqa([medqa_row()], "test")[0]
        self.assertEqual(
            questions.render_question(item),
            "A 23-year-old pregnant woman has burning upon urination. Which is the best treatment? "
            "A) Ampicillin B) Ceftriaxone C) Doxycycline D) Nitrofurantoin",
        )

    def test_the_hash_changes_with_an_option_the_answer_or_the_stem(self):
        base = questions.question_sha256(questions.read_medqa([medqa_row()], "test")[0])
        for change in (
            {"answer_idx": "C", "answer": "Doxycycline"},
            {"question": "Another question?"},
            {"options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin."}, "answer": "Nitrofurantoin."},
        ):
            with self.subTest(change=change):
                self.assertNotEqual(base, questions.question_sha256(questions.read_medqa([medqa_row(**change)], "test")[0]))


class MedQATests(unittest.TestCase):
    def test_a_row_becomes_an_item_with_an_id_made_of_split_and_position(self):
        items = questions.read_medqa([medqa_row(), medqa_row(question="Second?")], "validation")
        self.assertEqual([item.id for item in items], ["medqa:validation:0", "medqa:validation:1"])
        self.assertEqual((items[0].dataset, items[0].split, items[0].answer), ("medqa", "validation", "D"))
        self.assertEqual(items[0].options[3], "Nitrofurantoin")

    def test_a_list_of_four_options_is_read_in_letter_order(self):
        item = questions.read_medqa([medqa_row(options=["w", "x", "y", "Nitrofurantoin"])], "test")[0]
        self.assertEqual(item.options, ("w", "x", "y", "Nitrofurantoin"))

    def test_refused_rows(self):
        for name, row in {
            "five options": medqa_row(options={"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}),
            "missing option": medqa_row(options={"A": "a", "B": "b", "C": "c"}),
            "lower-case keys": medqa_row(options={"a": "a", "b": "b", "c": "c", "d": "d"}),
            "answer letter outside": medqa_row(answer_idx="E"),
            "no answer letter": medqa_row(answer_idx=None),
            "answer text is not the option": medqa_row(answer="Ceftriaxone"),
            "empty question": medqa_row(question="  "),
            "empty option": medqa_row(options={"A": "a", "B": "", "C": "c", "D": "Nitrofurantoin"}),
            "options are text": medqa_row(options="A) a B) b"),
        }.items():
            with self.subTest(name), self.assertRaises(QuestionDataError):
                questions.read_medqa([row], "test")

    def test_the_message_says_which_row_is_wrong(self):
        with self.assertRaises(QuestionDataError) as caught:
            questions.read_medqa([medqa_row(), medqa_row(answer_idx="Z")], "train")
        self.assertIn("medqa train row 1", str(caught.exception))

    def test_a_row_that_is_not_a_mapping_is_refused(self):
        with self.assertRaises(QuestionDataError):
            questions.read_medqa(["text"], "test")


class MedMCQATests(unittest.TestCase):
    def test_cop_is_counted_from_zero(self):
        for cop, letter in enumerate("ABCD"):
            with self.subTest(cop=cop):
                self.assertEqual(questions.read_medmcqa([medmcqa_row(cop=cop)], "validation")[0].answer, letter)

    def test_the_card_says_one_to_four_but_four_is_refused(self):
        with self.assertRaises(QuestionDataError) as caught:
            questions.read_medmcqa([medmcqa_row(cop=4)], "train")
        self.assertIn("0 to 3", str(caught.exception))

    def test_a_split_without_answers_is_refused_by_name(self):
        rows = [medmcqa_row(id=f"id{i}", cop=-1) for i in range(3)]
        with self.assertRaises(AnswersWithheld) as caught:
            questions.read_medmcqa(rows, "test")
        self.assertIn("withheld", str(caught.exception))
        self.assertIn("3 rows", str(caught.exception))

    def test_a_split_that_is_part_withheld_is_refused_as_mixed(self):
        with self.assertRaises(QuestionDataError) as caught:
            questions.read_medmcqa([medmcqa_row(id="a", cop=-1), medmcqa_row(id="b", cop=1)], "test")
        self.assertNotIsInstance(caught.exception, AnswersWithheld)
        self.assertIn("mixed", str(caught.exception))

    def test_the_id_subject_and_choice_type_are_kept(self):
        item = questions.read_medmcqa([medmcqa_row(choice_type="multi")], "train")[0]
        self.assertEqual((item.id, item.subject, item.choice_type), ("4e1715fe-0bc3-494e-b6eb-2d4617245aef", "Pathology", "multi"))

    def test_other_refused_rows(self):
        for name, row in {
            "cop is text": medmcqa_row(cop="2"),
            "cop is a bool": medmcqa_row(cop=True),
            "cop is missing": medmcqa_row(cop=None),
            "option missing": medmcqa_row(opb=None),
            "no id": medmcqa_row(id=""),
        }.items():
            with self.subTest(name), self.assertRaises(QuestionDataError):
                questions.read_medmcqa([row], "train")

    def test_a_repeated_id_is_refused(self):
        with self.assertRaises(QuestionDataError) as caught:
            questions.read_medmcqa([medmcqa_row(id="x"), medmcqa_row(id="x", question="Other?")], "train")
        self.assertIn("appears twice", str(caught.exception))


class MMLUTests(unittest.TestCase):
    def test_a_row_becomes_an_item_with_its_subject(self):
        item = questions.read_mmlu_med([mmlu_row()], "test")[0]
        self.assertEqual((item.id, item.answer, item.subject), ("mmlu_med:test:0", "B", "anatomy"))
        self.assertEqual(item.options, ("Vagus", "Phrenic", "Radial", "Ulnar"))

    def test_options_without_original_keys_take_their_position(self):
        row = mmlu_row(options=[{"text": t} for t in ("Vagus", "Phrenic", "Radial", "Ulnar")])
        self.assertEqual(questions.read_mmlu_med([row], "test")[0].answer, "B")

    def test_refused_rows(self):
        shuffled = [{"original_key": k, "text": t} for k, t in zip("BACD", ("Vagus", "Phrenic", "Radial", "Ulnar"))]
        for name, row in {
            "keys out of order": mmlu_row(options=shuffled),
            "three options": mmlu_row(options=[{"original_key": "A", "text": "x"}] * 3),
            "answer text differs": mmlu_row(answer={"original_key": "B", "text": "Vagus"}),
            "answer key outside": mmlu_row(answer={"original_key": "E", "text": "Phrenic"}),
            "no answer": mmlu_row(answer=None),
            "option is text": mmlu_row(options=["a", "b", "c", "d"]),
        }.items():
            with self.subTest(name), self.assertRaises(QuestionDataError):
                questions.read_mmlu_med([row], "test")

    def test_the_subject_may_be_missing(self):
        self.assertIsNone(questions.read_mmlu_med([mmlu_row(metadata=None)], "test")[0].subject)


class ManifestAndSummaryTests(unittest.TestCase):
    def test_the_manifest_lists_every_question_by_id_in_sorted_order(self):
        items = questions.read_medqa([medqa_row(), medqa_row(question="Second?")], "test")
        result = questions.manifest(list(reversed(items)))
        self.assertEqual((result["dataset"], result["split"], result["count"]), ("medqa", "test", 2))
        self.assertEqual(list(result["questions"]), ["medqa:test:0", "medqa:test:1"])
        self.assertEqual(result["questions"]["medqa:test:0"], questions.question_sha256(items[0]))

    def test_an_empty_manifest_has_no_dataset(self):
        self.assertEqual(questions.manifest([]), {"dataset": None, "split": None, "count": 0, "questions": {}})

    def test_the_summary_counts_answers_repeats_subjects_and_choice_types(self):
        items = questions.read_medmcqa(
            [medmcqa_row(id="1", cop=0), medmcqa_row(id="2", cop=0, choice_type="multi"), medmcqa_row(id="3", cop=2, subject_name="Anatomy")],
            "train",
        )
        summary = questions.summarise(items)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["answers"], {"A": 2, "B": 0, "C": 1, "D": 0})
        self.assertEqual(summary["repeated_stems"], 2)
        self.assertEqual(summary["subjects"], {"Anatomy": 1, "Pathology": 2})
        self.assertEqual(summary["choice_types"], {"multi": 1, "single": 2})


class FileAndHubTests(TempDirTestCase):
    def test_a_json_list_json_lines_and_a_byte_order_mark_are_read(self):
        rows = [medqa_row(), medqa_row(question="Second?")]
        as_list = self.dir / "list.json"
        as_list.write_text(json.dumps(rows), encoding="utf-8")
        as_lines = self.dir / "lines.jsonl"
        as_lines.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8-sig")
        self.assertEqual(questions.read_rows(as_list), rows)
        self.assertEqual(questions.read_rows(as_lines), rows)

    def test_a_missing_or_broken_file_is_a_data_error(self):
        broken = self.write("broken.jsonl", "{not json}\n")
        for path in (self.dir / "missing.json", broken):
            with self.subTest(path=path.name), self.assertRaises(QuestionDataError):
                questions.read_rows(path)

    def test_a_truncated_json_list_is_a_data_error(self):
        self.write("object.json", "[1, 2")
        with self.assertRaises(QuestionDataError):
            questions.read_rows(self.dir / "object.json")

    def test_the_hub_loader_gets_the_split_the_configuration_and_the_revision(self):
        calls = []

        def loader(repo, *config, **options):
            calls.append((repo, config, options))
            return [{"question": "q"}]

        rows = questions.hub_rows("awinml/medqa", "test", config="questions", revision="abc", loader=loader)
        self.assertEqual(rows, [{"question": "q"}])
        self.assertEqual(calls, [("awinml/medqa", ("questions",), {"split": "test", "revision": "abc"})])
        questions.hub_rows("x/y", "train", loader=loader)
        self.assertEqual(calls[1], ("x/y", (), {"split": "train"}))


if __name__ == "__main__":
    unittest.main()
