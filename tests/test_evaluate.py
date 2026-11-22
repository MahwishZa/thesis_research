"""Checks of src/evaluate.py with small made-up question files (Python only, no packages), plus one check on the committed test
questions that skips when data/qa/ is not there."""

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate as ev  # noqa: E402
from build_views import options_for  # noqa: E402
from common import CLASSES, OPTION_TEXT  # noqa: E402


def question(case_id, gold, group=None):
    options, letter = options_for(case_id)
    return {"id": case_id, "options": options, "answer": letter[gold], "gold_label": gold, "group_id": group or case_id,
            "question": f"Question {case_id}"}


def answer(q, code):
    """The letter that stands for a class in this question."""
    return next(k for k, v in q["options"].items() if v == OPTION_TEXT[code])


def as_dict(questions):
    return {q["id"]: q for q in questions}


class DiagnosisTests(unittest.TestCase):
    def test_a_letter_is_read_with_the_options_of_its_own_question(self):
        qa = as_dict(question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES))
        self.assertGreater(len({q["answer"] for q in qa.values()}), 1)  # the right letter differs between the questions
        pred = {i: {"id": i, "pred": q["answer"]} for i, q in qa.items()}  # every answer right
        d = ev.diagnosis_metrics(qa, pred)
        self.assertEqual(d["accuracy"], 1.0)
        self.assertEqual(d["macro_f1"], 1.0)
        self.assertTrue(all(r["sensitivity"] == 1.0 and r["specificity"] == 1.0 for r in d["per_class"].values()))
        self.assertEqual([row[k] for k, row in enumerate(d["confusion_matrix"]["matrix"])], [1] * 8)

    def test_per_class_counts_match_a_hand_count(self):
        # true: MEL MEL NV NV BCC; predicted: MEL NV NV MEL invalid
        y, p = ["MEL", "MEL", "NV", "NV", "BCC"], ["MEL", "NV", "NV", "MEL", ev.INVALID]
        r = ev.per_class(y, p)
        self.assertEqual((r["MEL"]["precision"], r["MEL"]["sensitivity"]), (0.5, 0.5))
        self.assertAlmostEqual(r["MEL"]["specificity"], 2 / 3)  # of NV, NV, BCC only one called MEL
        self.assertEqual((r["BCC"]["sensitivity"], r["BCC"]["specificity"], r["BCC"]["support"]), (0.0, 1.0, 1))
        self.assertEqual(r["DF"]["support"], 0)
        self.assertAlmostEqual(ev.macro_f1(y, p), (0.5 + 0.5) / 8)  # every class weighs the same, also those without cases
        self.assertEqual(ev.accuracy(y, p), 0.4)

    def test_a_missing_or_unusable_answer_counts_as_wrong(self):
        qs = [question(f"ISIC_{i:07d}", "MEL") for i in range(4)]
        qa = as_dict(qs)
        pred = {qs[0]["id"]: {"pred": qs[0]["answer"]}, qs[1]["id"]: {"pred": "Z"}, qs[2]["id"]: {"pred": None}}
        d = ev.diagnosis_metrics(qa, pred)
        self.assertEqual((d["n_questions"], d["missing_predictions"], d["invalid_answers"]), (4, 1, 2))
        self.assertEqual(d["accuracy"], 0.25)
        self.assertEqual(d["confusion_matrix"]["matrix"][0][-1], 3)  # three melanomas in the "invalid" column

    def test_lower_case_and_spaces_around_the_letter_are_accepted(self):
        q = question("ISIC_0000001", "SCC")
        self.assertEqual(ev.predicted_class(q, {"pred": f" {q['answer'].lower()} "}), "SCC")

    def test_a_prediction_for_a_question_that_is_not_in_the_file_stops_the_evaluation(self):
        qa = as_dict([question("ISIC_0000001", "MEL")])
        with self.assertRaises(ValueError):
            ev.diagnosis_metrics(qa, {"ISIC_0000001": {"pred": "A"}, "ISIC_9999999": {"pred": "A"}})

    def test_a_broken_question_file_stops_the_evaluation(self):
        q = question("ISIC_0000001", "MEL")
        q["gold_label"] = "NV"
        with self.assertRaises(ValueError):
            ev.gold_class(q)

    def test_the_report_has_accuracy_as_the_main_score_and_no_balanced_accuracy(self):
        qa = as_dict(question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES))
        d = ev.diagnosis_metrics(qa, {})
        self.assertIn("accuracy_95ci", d)
        self.assertFalse([k for k in d if "balanced" in k])

    def test_the_bootstrap_draws_whole_lesions(self):
        sizes = []
        ev.bootstrap_ci([1, 1, 1, 0], [1, 1, 1, 1], ["G1", "G1", "G1", "G2"], lambda y, p: sizes.append(len(y)) or 0.0, n=200)
        self.assertEqual(set(sizes), {2, 4, 6})  # G2+G2, G1+G2, G1+G1; drawing photos one by one would always give 4

    def test_the_bootstrap_interval_holds_the_accuracy_and_is_the_same_every_run(self):
        y = ["MEL"] * 30 + ["NV"] * 30
        p = ["MEL"] * 20 + ["NV"] * 10 + ["NV"] * 25 + ["MEL"] * 5
        groups = [f"G{i}" for i in range(60)]
        lo, hi = ev.bootstrap_ci(y, p, groups, ev.accuracy)
        self.assertLess(lo, ev.accuracy(y, p))
        self.assertGreater(hi, ev.accuracy(y, p))
        self.assertEqual(ev.bootstrap_ci(y, p, groups, ev.accuracy), [lo, hi])

    def test_the_percentile_matches_linear_interpolation(self):
        self.assertEqual(ev.percentile([0, 10, 20, 30, 40], 50), 20)
        self.assertAlmostEqual(ev.percentile([0, 10], 2.5), 0.25)

    def test_mcnemar_exact_gives_the_binomial_p_value(self):
        y = ["MEL"] * 12
        a = ["MEL"] * 10 + ["NV"] * 2
        b = ["NV"] * 10 + ["MEL"] * 2  # a alone right 10 times, b alone right 2 times
        r = ev.mcnemar_exact(y, a, b)
        self.assertEqual((r["a_right_b_wrong"], r["a_wrong_b_right"]), (10, 2))
        self.assertAlmostEqual(r["p_value"], 2 * (1 + 12 + 66) / 4096, places=6)  # two-sided, p = 0.0386
        tie = ev.mcnemar_exact(y, y, y)
        self.assertEqual(tie, {"a_right_b_wrong": 0, "a_wrong_b_right": 0, "p_value": 1.0})  # the same keys when nothing differs

    def test_a_second_system_is_compared_on_the_same_questions(self):
        qs = [question(f"ISIC_{i:07d}", "BCC") for i in range(3)]
        a = {q["id"]: {"pred": q["answer"]} for q in qs}
        d = ev.diagnosis_metrics(as_dict(qs), a, pred_b={})
        self.assertEqual(d["mcnemar_vs_system_b"]["a_right_b_wrong"], 3)


class ReportTests(unittest.TestCase):
    def test_the_chosen_letters_show_a_preference_for_one_position(self):
        qs = [question(f"ISIC_{i:07d}", "MEL") for i in range(5)]
        pred = {q["id"]: {"pred": "a"} for q in qs[:3]}
        pred[qs[3]["id"]] = {"pred": "maybe"}
        d = ev.diagnosis_metrics(as_dict(qs), pred)
        self.assertEqual(d["chosen_letters"], {"A": 3, "invalid": 2})

    def test_usage_is_summed_only_over_predictions_that_report_it(self):
        qa = as_dict([question("ISIC_0000001", "MEL"), question("ISIC_0000002", "NV")])
        pred = {"ISIC_0000001": {"pred": "A", "tokens_in": 100, "seconds": 1.5}, "ISIC_0000002": {"pred": "B"}}
        u = ev.usage_summary(qa, pred)
        self.assertEqual(u, {"tokens_in": {"total": 100.0, "mean": 100.0, "n": 1}, "seconds": {"total": 1.5, "mean": 1.5, "n": 1}})
        self.assertIsNone(ev.usage_summary(qa, {}))

    def test_accuracy_per_subgroup_and_per_recorded_facts(self):
        qs = [question(f"ISIC_{i:07d}", "MEL") for i in range(4)]
        cases = {q["id"]: {"isic_id": q["id"], "source": "HAM10000" if k < 2 else "archive", "age": "50.0", "sex": "male",
                           "site": "trunk" if k != 3 else ""} for k, q in enumerate(qs)}
        pred = {qs[0]["id"]: {"pred": qs[0]["answer"]}, qs[2]["id"]: {"pred": qs[2]["answer"]}}
        g = ev.subgroup_accuracy(as_dict(qs), pred, cases, ["source", "facts_recorded"])
        self.assertEqual(g["source"], {"HAM10000": {"n": 2, "accuracy": 0.5}, "archive": {"n": 2, "accuracy": 0.5}})
        self.assertEqual(g["facts_recorded"], {"all": {"n": 3, "accuracy": round(2 / 3, 4)}, "some missing": {"n": 1, "accuracy": 0.0}})
        with self.assertRaises(ValueError):
            ev.subgroup_accuracy(as_dict(qs), pred, cases, ["no_such_column"])

    def test_a_repeated_question_id_stops_the_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qa.jsonl")
            q = question("ISIC_0000001", "MEL")
            Path(path).write_text(json.dumps(q) + "\n" + json.dumps(q) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ev.load(path)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.qa = as_dict([question("ISIC_0000001", "BCC")])

    def contexts(self, tags):
        return {"ISIC_0000001": {"pred": "A", "contexts": [{"text": f"passage {j}", "tags": t} for j, t in enumerate(tags)]}}

    def test_hit_precision_mrr_and_context_precision(self):
        r = ev.retrieval_metrics(self.qa, self.contexts([["MEL"], ["BCC"], [], ["BCC", "SCC"], ["NV"]]))
        self.assertEqual((r["hit@1"], r["hit@3"]), (0.0, 1.0))
        self.assertAlmostEqual(r["precision@5"], 0.4)
        self.assertEqual(r["mrr"], 0.5)
        self.assertAlmostEqual(r["context_precision"], round((1 / 2 + 2 / 4) / 2, 4))
        self.assertNotIn("hit_recall@1", r)

    def test_ndcg_does_not_reward_returning_fewer_passages(self):
        r = ev.retrieval_metrics(self.qa, self.contexts([["BCC"]]))  # one passage only, and it is relevant
        self.assertEqual(r["ndcg@1"], 1.0)
        self.assertAlmostEqual(r["ndcg@5"], round(1 / sum(1 / math.log2(j + 2) for j in range(5)), 4))

    def test_no_contexts_means_no_retrieval_report(self):
        self.assertIsNone(ev.retrieval_metrics(self.qa, {"ISIC_0000001": {"pred": "A"}}))


class RelevanceTests(unittest.TestCase):
    def test_jaccard_of_the_content_words(self):
        stop = frozenset({"on", "the"})
        self.assertEqual(ev.jaccard("Basal cell carcinoma on the face", "basal cell carcinoma: arborizing vessels", stop), 0.5)
        self.assertEqual(ev.jaccard("", "", stop), 0.0)
        self.assertEqual(ev.jaccard("the", "the", stop), 0.0)  # only stop words: nothing in common that counts

    def test_relevance_uses_the_system_query_and_falls_back_to_the_question(self):
        qa = as_dict([question("ISIC_0000001", "BCC"), question("ISIC_0000002", "MEL")])
        ctx = [{"text": "arborizing vessels"}, {"text": "pigment network"}]
        pred = {"ISIC_0000001": {"query": "arborizing vessels", "contexts": ctx},
                "ISIC_0000002": {"contexts": ctx}}
        seen = []
        r = ev.relevance_metrics(qa, pred, scorers={"fake": lambda q, ps: seen.append(q) or [1.0, 0.0]}, stop_words=frozenset())
        self.assertEqual(seen, ["arborizing vessels", "Question ISIC_0000002"])
        self.assertEqual(r["query_used"], {"query": 1, "question": 1})
        self.assertEqual(r["fake"], 0.5)
        self.assertEqual(r["jaccard_content_words"], round((1.0 + 0.0) / 2 / 2, 4))  # per question, then over questions


class CommandTests(unittest.TestCase):
    def test_the_command_writes_a_report(self):
        qs = [question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES)]
        with tempfile.TemporaryDirectory() as d:
            paths = {k: os.path.join(d, f"{k}.jsonl") for k in ("qa", "pred")}
            with open(paths["qa"], "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(q) for q in qs) + "\n\n")  # a blank last line is ignored
            with open(paths["pred"], "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps({"id": q["id"], "pred": q["answer"]}) for q in qs[:4]))
            out = os.path.join(d, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                ev.main(["--qa", paths["qa"], "--pred", paths["pred"], "--out", out, "--skip-text"])
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            sha = hashlib.sha256(Path(paths["qa"]).read_bytes()).hexdigest()
        self.assertEqual((rep["diagnosis"]["accuracy"], rep["diagnosis"]["missing_predictions"]), (0.5, 4))
        self.assertIsNone(rep["retrieval"])
        self.assertEqual(rep["inputs"]["questions"]["sha256"], sha)  # the report says which question file it used


class CommittedQuestionTests(unittest.TestCase):
    def test_on_the_committed_test_questions_one_fixed_letter_is_about_chance(self):
        path = Path(os.environ.get("THESIS_DATA", ROOT / "data")) / "views" / "multimodal" / "qa_test.jsonl"
        if not path.is_file():
            self.skipTest("data/views/ is not there")
        qa = ev.load(path)
        right = ev.diagnosis_metrics(qa, {i: {"pred": q["answer"]} for i, q in qa.items()})
        self.assertEqual(right["accuracy"], 1.0)
        self.assertEqual(sum(r["support"] for r in right["per_class"].values()), len(qa))
        always_a = ev.diagnosis_metrics(qa, {i: {"pred": "A"} for i in qa})
        self.assertLess(always_a["accuracy"], 0.25)  # "A" is a different class in each question
        self.assertGreater(sum(r["sensitivity"] > 0 for r in always_a["per_class"].values()), 1)  # "A" picks several classes


if __name__ == "__main__":
    unittest.main()
