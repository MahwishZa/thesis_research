"""Tests of src/rationale_query/diagnostics.py: the numbers that describe a set of rationales."""

import unittest

from rationale_query.diagnostics import (
    agreement,
    final_option,
    format_table,
    provenance,
    summarise,
)
from rationale_query.store import RationaleRecord


def record(question_id, text="The answer is B.", *, tokens=100, seconds=10.0, finish="stop"):
    return RationaleRecord(
        id=question_id,
        question_sha256="h",
        status="ok",
        rationale=text,
        finish_reason=finish,
        prompt_tokens=300,
        completion_tokens=tokens,
        seconds=seconds,
    )


def error(question_id, message="RuntimeError: boom"):
    return RationaleRecord(id=question_id, question_sha256="h", status="error", error=message)


class FinalOptionTests(unittest.TestCase):
    def test_phrases_that_name_the_answer(self):
        cases = {
            "The answer is B) Alzheimer disease.": "B",
            "Answer: C": "C",
            "**Final answer: D**": "D",
            "the best choice (C).": "C",
            "Therefore, the answer is E": "E",
            "Option B\nmore text": "B",
            "OPTION D.": "D",
            "answer is: B": "B",
            "Final Answer – B.": "B",
            "The correct answer would be (A).": "A",
            "The final answer should be D, because": "D",
            "The answer is **B) Donepezil**": "B",
            "Answer: A. Donepezil": "A",
        }
        for text, letter in cases.items():
            with self.subTest(text=text):
                self.assertEqual(final_option(text), letter)

    def test_the_last_phrase_wins(self):
        self.assertEqual(final_option("Option A is wrong. The final answer is C."), "C")
        self.assertEqual(final_option("The answer is B.\nOn reflection, the answer is (C)."), "C")

    def test_text_that_does_not_name_an_option_gives_none(self):
        cases = [
            "",
            "No answer given here",
            "The answer is A cholinesterase inhibitor",  # a letter that starts a word is not an option
            "answer is e.",  # lower case
            "The answer is F.",  # outside A-E
            "OPTION d.",
            "answerable B.",  # not the word 'answer'
            "The answer is not B.",
            "I think the answer is B or C.",
            "Answer is:\nB",  # a line break between the phrase and the letter
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(final_option(text))


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("a", "The answer is B.", tokens=100, seconds=10.0, finish="stop"),
            record("b", "Option C.", tokens=200, seconds=20.0, finish="length"),
            record("c", "No letter here", tokens=300, seconds=30.0, finish="stop"),
            error("d", "ContextOverflowError: too long"),
            error("e", "OSError: disk"),
        ]

    def test_counts_and_error_kinds(self):
        summary = summarise(self.records)
        self.assertEqual((summary["records"], summary["ok"], summary["errors"]), (5, 3, 2))
        self.assertEqual(summary["error_kinds"], {"ContextOverflowError": 1, "OSError": 1})
        self.assertEqual(summary["hit_generation_limit"], 1)

    def test_lengths_and_times_cover_the_successful_records_only(self):
        summary = summarise(self.records)
        self.assertEqual(summary["completion_tokens"], {"mean": 200.0, "median": 200.0, "p95": 300.0, "max": 300.0})
        self.assertEqual(summary["seconds_per_question"], {"mean": 20.0, "median": 20.0, "p95": 30.0, "max": 30.0})

    def test_a_failed_record_is_not_a_rationale_that_was_cut_off(self):
        empty = RationaleRecord(id="e", question_sha256="h", status="error", finish_reason="length", error="EmptyRationale: x")
        self.assertEqual(summarise([record("a", finish="stop"), empty])["hit_generation_limit"], 0)

    def test_the_final_options_that_could_be_read(self):
        summary = summarise(self.records)
        self.assertEqual(summary["final_option_found"], 2)
        self.assertEqual(summary["final_option_counts"], {"B": 1, "C": 1})

    def test_the_median_is_not_the_mean_and_an_even_count_is_averaged(self):
        skewed = [record("a", tokens=100, seconds=1.0), record("b", tokens=100, seconds=1.0), record("c", tokens=400, seconds=4.0)]
        summary = summarise(skewed)
        self.assertEqual((summary["completion_tokens"]["mean"], summary["completion_tokens"]["median"]), (200.0, 100.0))
        self.assertEqual((summary["seconds_per_question"]["mean"], summary["seconds_per_question"]["median"]), (2.0, 1.0))
        even = summarise([record("a", tokens=100), record("b", tokens=300)])
        self.assertEqual(even["completion_tokens"]["median"], 200.0)

    def test_a_percentile_is_the_nearest_rank(self):
        records = [record(f"q{n}", tokens=n) for n in range(1, 101)]
        self.assertEqual(summarise(records)["completion_tokens"]["p95"], 95.0)
        records = [record(f"q{n}", tokens=n) for n in range(1, 21)]
        self.assertEqual(summarise(records)["completion_tokens"]["p95"], 19.0)
        self.assertEqual(summarise([record("x", tokens=7)])["completion_tokens"]["p95"], 7.0)

    def test_nothing_to_describe(self):
        summary = summarise([])
        self.assertEqual((summary["records"], summary["ok"], summary["errors"]), (0, 0, 0))
        self.assertIsNone(summary["completion_tokens"])
        self.assertIsNone(summary["seconds_per_question"])
        summary = summarise([error("a")])
        self.assertIsNone(summary["completion_tokens"])

    def test_a_record_without_counts_is_left_out_of_the_spreads(self):
        bare = RationaleRecord(id="z", question_sha256="h", status="ok", rationale="Option A.")
        summary = summarise([record("a", tokens=100), bare])
        self.assertEqual(summary["completion_tokens"]["max"], 100.0)
        self.assertEqual(summary["ok"], 2)

    def test_encoder_tokens_are_not_reported_without_counts(self):
        self.assertNotIn("encoder_truncated", summarise(self.records))

    def test_a_text_is_cut_by_the_encoder_only_above_the_limit(self):
        counts = {"a": 512, "b": 513, "c": 100, "unrelated": 9999}
        summary = summarise(self.records, counts)
        self.assertEqual(summary["encoder_truncated"], 1)  # 513 only; 512 fits exactly
        self.assertEqual(summary["encoder_tokens"]["max"], 513.0)

    def test_the_limit_of_the_encoder_can_be_given(self):
        summary = summarise(self.records, {"a": 50, "b": 60, "c": 70}, encoder_limit=55)
        self.assertEqual(summary["encoder_truncated"], 2)


class AgreementTests(unittest.TestCase):
    def test_same_option_on_the_questions_both_writers_answered_readably(self):
        first = [record("a", "answer is B."), record("b", "answer is C."), record("c", "answer is D."), record("d", "no letter")]
        second = [record("a", "answer is B."), record("b", "answer is A."), record("c", "no letter"), record("d", "answer is A.")]
        self.assertEqual(agreement(first, second), {"compared": 2, "same_option": 1, "share": 0.5})

    def test_questions_only_one_writer_has_are_left_out(self):
        first = [record("a", "answer is B."), record("only first", "answer is B.")]
        second = [record("a", "answer is B."), record("only second", "answer is B.")]
        self.assertEqual(agreement(first, second)["compared"], 1)

    def test_failed_questions_are_left_out(self):
        first = [record("a", "answer is B."), error("b")]
        second = [record("a", "answer is B."), record("b", "answer is B.")]
        self.assertEqual(agreement(first, second), {"compared": 1, "same_option": 1, "share": 1.0})

    def test_no_shared_question_gives_no_share(self):
        self.assertEqual(agreement([record("a", "answer is B.")], [record("b", "answer is B.")]), {"compared": 0, "same_option": 0, "share": None})

    def test_the_share_is_rounded_to_three_places(self):
        first = [record(f"q{n}", "answer is B.") for n in range(3)]
        second = [record("q0", "answer is B."), record("q1", "answer is A."), record("q2", "answer is A.")]
        self.assertEqual(agreement(first, second)["share"], 0.333)


class ProvenanceTests(unittest.TestCase):
    def test_one_line_says_how_the_rationales_were_made(self):
        config = {
            "model_file": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "model_sha256": "8ba9baf3a7345f705a11878397500fb25174034f0fd784e83aa4a96aaa47735f",
            "chat_format": "llama-3",
            "template_check": "matches the file's own chat template",
            "max_new_tokens": 512,
            "n_ctx": 8192,
            "llama_cpp_python": "0.3.35",
        }
        self.assertEqual(
            provenance("baseline", config),
            "baseline: Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (SHA-256 8ba9baf3a734...), llama-3 format "
            "[template: matches the file's own chat template], at most 512 new tokens, context 8192, "
            "llama-cpp-python 0.3.35",
        )

    def test_a_config_with_gaps_does_not_break_the_line(self):
        self.assertIn("(SHA-256 ...)", provenance("x", {}))


class FormatTableTests(unittest.TestCase):
    def test_one_column_per_folder_and_one_row_per_number(self):
        records = [record("a", tokens=100), record("b", "Option C.", tokens=200, finish="length"), error("c")]
        table = format_table({"writer one": summarise(records), "w2": summarise([])})
        lines = table.split("\n")
        self.assertEqual(len(lines), 10)
        self.assertIn("writer one", lines[0])
        self.assertIn("w2", lines[0])
        rows = {line[:30].strip(): line[30:].split() for line in lines[1:]}
        self.assertEqual(rows["records"], ["3", "0"])
        self.assertEqual(rows["with a rationale"], ["2", "0"])
        self.assertEqual(rows["errors"], ["1", "0"])
        self.assertEqual(rows["new tokens, median"], ["150.0", "-"])
        self.assertEqual(rows["cut off at the limit"], ["1", "0"])
        self.assertEqual(rows["final option readable"], ["2", "0"])
        self.assertEqual(rows["longer than the encoder limit"], ["-", "-"])

    def test_the_encoder_row_shows_the_count_once_the_vectors_exist(self):
        records = [record("a"), record("b")]
        table = format_table({"x": summarise(records, {"a": 600, "b": 10})})
        self.assertTrue(table.split("\n")[-1].rstrip().endswith("1"))

    def test_the_columns_line_up(self):
        table = format_table({"first": summarise([record("a")]), "second folder name": summarise([record("a")])})
        widths = {len(line) for line in table.split("\n")}
        self.assertEqual(len(widths), 1)


if __name__ == "__main__":
    unittest.main()
