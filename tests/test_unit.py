"""Unit tests: one function or rule at a time, on made-up input, with no file from data/, no network and no API call.
Common (writing, reading, names), the balanced test rule, the cleaning functions, the question options, the metrics of
evaluate.py (diagnosis, top-k, selective, semantic similarity, retrieval, relevance, compare), and the prompt, the answer rule and the
request formats of run.py. Run: python -m unittest discover -s tests"""

import contextlib
import importlib.util
import io
import itertools
import json
import math
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build as bd  # noqa: E402
import check as cp  # noqa: E402
import common  # noqa: E402
import evaluate as ev  # noqa: E402
import run as rl  # noqa: E402
from common import CLASSES, OPTION_TEXT, write_jsonl  # noqa: E402
from build import options_for  # noqa: E402

HAVE = all(importlib.util.find_spec(p) for p in ("pandas", "numpy", "cv2", "PIL", "sklearn"))
SENTENCE = " A dermoscopic image of the lesion is available."


# ----------------------------------------------------------------------------------------------------------------------
# the commands of the scripts
# ----------------------------------------------------------------------------------------------------------------------
class DispatchTests(unittest.TestCase):
    """build.py, check.py, evaluate.py and run.py pass the rest of the command line to the right command, and refuse an unknown one."""

    def test_build_and_run_pass_the_arguments_after_the_command_name(self):
        for module, names in ((bd, ("views", "corpus", "references")), (rl, ("llm-only",))):
            for name in names:
                with self.subTest(f"{module.__name__} {name}"):
                    fake = mock.Mock(return_value=7)
                    with mock.patch.dict(module.COMMANDS, {name: fake}):
                        self.assertEqual(module.main([name, "--x", "1"]), 7)
                    fake.assert_called_once_with(["--x", "1"])

    def test_an_unknown_or_missing_command_prints_the_help_and_is_refused(self):
        for module in (bd, rl):
            with self.subTest(module.__name__), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(module.main(["nonsense"]), 2)
                self.assertEqual(module.main([]), 0)
                self.assertEqual(module.main(["--help"]), 0)
            self.assertTrue(out.getvalue().strip(), "no help text")

    def test_check_and_evaluate_run_their_default_and_pass_the_rest_to_the_other_command(self):
        for module, command, other, default in ((cp, "features", "features_main", "status_main"),
                                               (ev, "compare", "compare_main", "report_main")):
            with self.subTest(module.__name__):
                with mock.patch.object(module, other, return_value=5) as chosen, \
                        mock.patch.object(module, default, return_value=6) as plain:
                    self.assertEqual(module.main([command, "--x", "1"]), 5)
                    chosen.assert_called_once_with(["--x", "1"])
                    self.assertEqual(module.main(["--data", "d"]), 6)
                    plain.assert_called_once_with(["--data", "d"])


# ----------------------------------------------------------------------------------------------------------------------
# common
# ----------------------------------------------------------------------------------------------------------------------


class CommonTests(unittest.TestCase):
    def test_an_interrupted_write_leaves_the_old_file_and_no_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "table.csv")
            path.write_text("old", encoding="utf-8")
            with self.assertRaises(RuntimeError), common.atomic_path(path) as tmp:
                Path(tmp).write_text("half", encoding="utf-8")
                raise RuntimeError("interrupted")
            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(os.listdir(d), ["table.csv"])
            common.write_csv(path, [{"a": "1", "b": "x, y"}], ["a", "b"])
            self.assertEqual(path.read_bytes(), b'a,b\n1,"x, y"\n')     # LF endings, quoting as pandas writes

    def test_json_and_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            common.write_jsonl(Path(d, "x.jsonl"), [{"id": "a"}, {"id": "b"}])
            self.assertEqual(common.read_jsonl(Path(d, "x.jsonl")), [{"id": "a"}, {"id": "b"}])
            common.write_json(Path(d, "x.json"), {"a": 1})
            self.assertEqual(Path(d, "x.json").read_text(encoding="utf-8"), '{\n  "a": 1\n}')

    def test_the_layout_names(self):
        self.assertEqual(common.view_table("text_only"), "views/text_only/cases.csv")
        self.assertEqual(common.question_file("multimodal", "test"), "views/multimodal/test.jsonl")
        self.assertEqual(len(common.CLASSES), len(common.LETTERS))
        self.assertEqual(set(common.CLASS_NAMES), set(common.CLASSES))
        self.assertEqual(set(common.OPTION_TEXT), set(common.CLASSES))


# ----------------------------------------------------------------------------------------------------------------------
# the balanced test rule
# ----------------------------------------------------------------------------------------------------------------------


def spec_cases(spec):
    """spec: [(label, split, group size), ...] -> one case per image, group ids g0, g1, ..."""
    out = []
    for k, (label, split, size) in enumerate(spec):
        out += [{"isic_id": f"ISIC_{k:03d}_{j}", "group_id": f"g{k}", "label": label, "split": split} for j in range(size)]
    return out


def spec_counts(cs, new):
    return Counter((c["label"], new[c["isic_id"]]) for c in cs)


class BalanceRuleTests(unittest.TestCase):
    def test_every_class_gets_exactly_the_target_and_validation_is_untouched(self):
        cs = spec_cases([("MEL", "test", 1)] * 6 + [("MEL", "train", 1)] * 4 + [("MEL", "val", 1)] * 2 +
                   [("DF", "test", 1)] * 1 + [("DF", "train", 2)] * 3 + [("DF", "train", 1)] * 2 + [("DF", "val", 1)] * 2)
        new = common.balanced_splits(cs, per_class=3, seed=42)
        c = spec_counts(cs, new)
        self.assertEqual((c[("MEL", "test")], c[("DF", "test")]), (3, 3))
        self.assertEqual((c[("MEL", "val")], c[("DF", "val")]), (2, 2))
        self.assertTrue(all(new[x["isic_id"]] == "val" for x in cs if x["split"] == "val"))
        self.assertEqual((c[("MEL", "train")], c[("DF", "train")]), (7, 6))       # nothing lost: 12 and 11 images in all

    def test_whole_lesion_groups_move_and_a_group_that_would_go_over_is_skipped(self):
        cs = spec_cases([("AK", "test", 1), ("AK", "train", 3), ("AK", "train", 2), ("AK", "train", 1)])
        new = common.balanced_splits(cs, per_class=3, seed=42)
        by_group = {}
        for x in cs:
            by_group.setdefault(x["group_id"], set()).add(new[x["isic_id"]])
        self.assertTrue(all(len(s) == 1 for s in by_group.values()))
        self.assertEqual(spec_counts(cs, new)[("AK", "test")], 3)
        self.assertEqual(by_group["g1"], {"train"})                                  # 1 + 3 would be 4: skipped

    def test_test_cases_are_kept_first_and_only_what_is_needed_moves(self):
        cs = spec_cases([("NV", "test", 1)] * 2 + [("NV", "train", 1)] * 5)
        new = common.balanced_splits(cs, per_class=3, seed=42)
        self.assertTrue(all(new[x["isic_id"]] == "test" for x in cs if x["split"] == "test"))
        self.assertEqual(sum(x["split"] == "train" and new[x["isic_id"]] == "test" for x in cs), 1)

    def test_the_choice_is_fixed_by_the_seed_and_not_by_the_order_of_the_table(self):
        cs = spec_cases([("BCC", "test", 1)] * 8 + [("BCC", "train", 1)] * 8)
        a = common.balanced_splits(cs, per_class=4, seed=42)
        self.assertEqual(a, common.balanced_splits(list(reversed(cs)), per_class=4, seed=42))
        self.assertNotEqual(a, common.balanced_splits(cs, per_class=4, seed=7))
        self.assertEqual(a, common.balanced_splits(cs, per_class=4))                                  # the default seed is 42
        kept = sorted(x["group_id"] for x in cs if a[x["isic_id"]] == "test")
        self.assertEqual(kept, sorted(sorted({x["group_id"] for x in cs if x["split"] == "test"},
                                             key=lambda g: common.group_order(42, g))[:4]))

    def test_the_validation_pass_balances_val_from_train_and_never_touches_test(self):
        cs = (spec_cases([("MEL", "test", 1)] * 5 + [("MEL", "val", 1)] * 6 + [("MEL", "train", 1)] * 6 +
                         [("DF", "test", 1)] * 3 + [("DF", "val", 1)] * 1 + [("DF", "train", 1)] * 4))
        after_test = common.balanced_splits(cs, per_class=3, split="test")
        cs2 = [{**c, "split": after_test[c["isic_id"]]} for c in cs]
        new = common.balanced_splits(cs2, per_class=4, split="val")
        c = spec_counts(cs2, new)
        self.assertEqual((c[("MEL", "val")], c[("DF", "val")]), (4, 4))
        self.assertEqual((c[("MEL", "test")], c[("DF", "test")]), (3, 3))
        self.assertTrue(all(new[x["isic_id"]] == "test" for x in cs2 if x["split"] == "test"))      # the second pass leaves test alone
        self.assertEqual(sum(v == "train" for v in new.values()), len(cs) - 3 * 2 - 4 * 2)           # nothing lost
        again = common.balanced_splits([{**x, "split": new[x["isic_id"]]} for x in cs2], per_class=4, split="val")
        self.assertEqual(again, new)                                                                # a second run changes nothing
        with self.assertRaisesRegex(ValueError, "DF: only 4 val images|VASC|only .* val images"):
            common.balanced_splits(spec_cases([("DF", "val", 3), ("DF", "train", 3)]), per_class=4, split="val")
        self.assertEqual(common.VAL_PER_CLASS, 20)

    def test_it_stops_when_a_class_cannot_reach_the_target_or_a_group_is_mixed(self):
        with self.assertRaisesRegex(ValueError, "VASC: only 2 test images"):
            common.balanced_splits(spec_cases([("VASC", "test", 1), ("VASC", "train", 1), ("VASC", "val", 5)]), per_class=3)
        with self.assertRaisesRegex(ValueError, "only 2 test images"):
            common.balanced_splits(spec_cases([("SCC", "test", 2), ("SCC", "train", 2)]), per_class=3)    # 2 + 2 > 3, cannot be 3
        mixed = spec_cases([("MEL", "test", 2)])
        mixed[1]["label"] = "NV"
        with self.assertRaisesRegex(ValueError, "more than one class or split"):
            common.balanced_splits(mixed, per_class=1)


# ----------------------------------------------------------------------------------------------------------------------
# cleaning functions and question options
# ----------------------------------------------------------------------------------------------------------------------


@unittest.skipUnless(HAVE, "the packages of src/preprocess.py are not installed")
class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import preprocess
        cls.pd, cls.p = pd, preprocess

    def test_the_patient_text_says_only_the_recorded_facts_and_nothing_about_an_image(self):
        nan = float("nan")
        self.assertEqual(self.p.clinical_text(55.0, "female", "head/neck", "face"), "Patient: 55-year-old female. Skin lesion on the head/neck (face).")
        self.assertEqual(self.p.clinical_text(nan, nan, nan, nan), "Patient: age not recorded, sex not recorded. Lesion site not recorded.")
        self.assertEqual(self.p.clinical_text(70.0, "male", "trunk", "trunk"), "Patient: 70-year-old male. Skin lesion on the trunk.")  # no repeat

    def test_the_new_text_is_the_old_text_without_the_image_sentence_for_every_committed_case(self):
        path = ROOT / "data" / "processed_cases" / "all_cases.csv"
        if not path.is_file():
            self.skipTest("data/processed_cases/all_cases.csv is not there")
        m = self.pd.read_csv(path)
        old = [t.removesuffix(SENTENCE) for t in m.clinical_text]
        new = [self.p.clinical_text(*x) for x in zip(m.age, m.sex, m.site, m.site_detail)]
        self.assertEqual(new, old)

    def test_a_table_has_unix_line_endings_even_on_windows(self):
        df = self.pd.DataFrame({"isic_id": ["ISIC_1", "ISIC_2"], "label": ["MEL", "NV"]})
        with tempfile.TemporaryDirectory() as d, mock.patch("os.linesep", "\r\n"):   # what Windows uses
            path = Path(d, "t.csv")
            self.p.save_csv(df, path)
            self.assertEqual(path.read_bytes(), b"isic_id,label\nISIC_1,MEL\nISIC_2,NV\n")

    def test_an_image_path_uses_a_forward_slash_on_every_system(self):
        self.assertEqual(self.p.image_rel_path("ISIC_0000000"), "processed_images/ISIC_0000000.jpg")

    def test_a_missing_input_stops_the_run_with_its_name(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as stop:
                self.p.check_inputs(os.path.join(d, "raw"), os.path.join(d, "official_labels"))
            self.assertIn("Train", str(stop.exception))
            self.assertIn("HAM10000_metadata.csv", str(stop.exception))

    def test_the_audit_keeps_every_original_file_and_marks_the_removed_copies(self):
        pd = self.pd
        raw = pd.DataFrame({"isic_id": ["A", "A", "B", "C"], "path": ["p1", "p2", "p3", "p4"], "md5": ["x", "x", "y", "z"]})
        kept = pd.DataFrame({"isic_id": ["A", "B"], "label": ["MEL", "NV"], "split": ["train", "test"]})
        audit = self.p.audit_table(raw, kept, {"C": "A"}, pd.Series({"A": "MEL", "B": "NV", "C": "MEL"}))
        self.assertEqual(len(audit), 4)  # one row per original file
        self.assertEqual(audit[audit.isic_id == "C"][["label", "split"]].values.tolist(), [["MEL", "removed: same picture as A"]])
        self.assertNotIn("path", audit.columns)
        with self.assertRaises(AssertionError):  # a file that is neither kept nor a known copy
            self.p.audit_table(raw, kept, {}, pd.Series({"A": "MEL", "B": "NV"}))


class OptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = bd

    def test_the_options_are_the_eight_classes_in_a_fixed_order_for_each_case(self):
        options, letter = self.q.options_for("ISIC_0024948")
        self.assertEqual(sorted(options.values()), sorted(self.q.OPTION_TEXT.values()))
        self.assertEqual(list(options), list("ABCDEFGH"))
        for code, text in self.q.OPTION_TEXT.items():
            self.assertEqual(options[letter[code]], text)  # the answer letter points to the right class
        self.assertEqual(self.q.options_for("ISIC_0024948"), (options, letter))  # the same every time

    def test_the_order_differs_between_cases_so_no_position_stands_for_a_class(self):
        ids = [f"ISIC_{i:07d}" for i in range(200)]
        first = {self.q.options_for(i)[0]["A"] for i in ids}
        self.assertEqual(len(first), 8)  # every class appears in position A
        melanoma = {self.q.options_for(i)[1]["MEL"] for i in ids}
        self.assertEqual(len(melanoma), 8)  # melanoma appears at every position


# ----------------------------------------------------------------------------------------------------------------------
# metrics of evaluate.py
# ----------------------------------------------------------------------------------------------------------------------


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
        self.assertTrue(all(r["recall"] == 1.0 and r["specificity"] == 1.0 for r in d["per_class"].values()))
        self.assertEqual([row[k] for k, row in enumerate(d["confusion_matrix"]["matrix"])], [1] * 8)

    def test_per_class_counts_match_a_hand_count(self):
        # true: MEL MEL NV NV BCC; predicted: MEL NV NV MEL invalid
        y, p = ["MEL", "MEL", "NV", "NV", "BCC"], ["MEL", "NV", "NV", "MEL", ev.INVALID]
        r = ev.per_class(y, p)
        self.assertEqual((r["MEL"]["precision"], r["MEL"]["recall"]), (0.5, 0.5))
        self.assertAlmostEqual(r["MEL"]["specificity"], 2 / 3)  # of NV, NV, BCC only one called MEL
        self.assertEqual((r["BCC"]["recall"], r["BCC"]["specificity"], r["BCC"]["support"]), (0.0, 1.0, 1))
        self.assertEqual(r["DF"]["support"], 0)
        self.assertAlmostEqual(ev.macro_f1(y, p), (0.5 + 0.5) / 8)  # every class weighs the same, also those without cases
        self.assertEqual(ev.accuracy(y, p), 0.4)

    def test_invalid_answers_are_split_into_refusals_and_other_invalid_answers(self):
        qs = [question(f"ISIC_{i:07d}", "MEL") for i in range(7)]
        qa = {q["id"]: q for q in qs}
        right = answer(qs[0], "MEL")
        pred = {"ISIC_0000000": {"pred": right, "raw_output": "I'm sorry to say it looks like melanoma. Answer: " + right},
                "ISIC_0000001": {"pred": None, "finish_reason": "stop",
                                 "raw_output": "I'm sorry, I can't help with identifying or diagnosing images of skin lesions."},
                "ISIC_0000002": {"pred": None, "finish_reason": "refusal: cyber", "raw_output": ""},
                "ISIC_0000003": {"pred": None, "finish_reason": "stop", "raw_output": "I cannot provide a diagnosis."},
                "ISIC_0000004": {"pred": None, "finish_reason": "length", "raw_output": "The lesion shows a network and"},
                "ISIC_0000005": {"pred": "Z", "raw_output": "Answer: Z"}}
        d = ev.diagnosis_metrics(qa, pred)
        self.assertEqual((d["invalid_answers"], d["missing_predictions"]), (5, 1))
        self.assertEqual(d["invalid_answers_by_cause"],
                         {"refusals": 3, "other_invalid": 2, "refusal_ids": ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"],
                          "other_invalid_ids": ["ISIC_0000004", "ISIC_0000005"]})
        self.assertEqual(d["accuracy"], round(1 / 7, 4))                         # refusals count as wrong, as before
        for text in ("I am sorry, but I cannot assist.", "I’m sorry, that is outside what I do here.", "I’m unable to analyze this image.", "I won't diagnose this.",
                     "I am not able to evaluate medical images."):
            self.assertTrue(ev.REFUSAL.search(text), text)
        for text in ("Sorry state of the skin.", "I can see a pigment network.", "It cannot be excluded."):
            self.assertFalse(ev.REFUSAL.search(text), text)

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


EVIDENCE_IDS = itertools.count(1)
KEYWORD = {"MEL": "melanoma", "NV": "nevi", "BCC": "basal cell carcinoma", "AK": "actinic keratosis", "BKL": "seborrheic keratosis",
           "DF": "dermatofibroma", "VASC": "hemangioma", "SCC": "squamous cell carcinoma"}
MEL_TEXT = "Melanomas show asymmetry, and irregular borders here. Other text follows."


def keyword_scorer(weights=None, record=None):
    """A stand-in for the NLI model: a statement scores high when its diagnosis is named in the passage."""
    by_statement = {ev.diagnosis_statement(c): c for c in CLASSES}

    def scorer(premises, hypotheses):
        if record is not None:
            record.extend(hypotheses)
        return [(weights or {}).get(by_statement[h], 0.9) if KEYWORD[by_statement[h]] in p.lower() else 0.1
                for p, h in zip(premises, hypotheses)]
    scorer.model_name = "fake-nli"
    return scorer


def evidence_case(code, gold, chunk="c1", quote="Melanomas show asymmetry and irregular borders here", text=MEL_TEXT, tags=None,
                  none=False):
    """One answered question with evidence: (question, prediction). `code` is the answered class, `gold` the true class."""
    q = question(f"ISIC_{next(EVIDENCE_IDS):07d}", gold)
    p = {"pred": answer(q, code) if code else "Z", "contexts": [{"chunk_id": "c1", "text": text, "tags": tags if tags is not None else [gold]}],
         "evidence": None if none else {"chunk_id": chunk, "quote": quote}}
    return q, p


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        cases_ = [evidence_case("MEL", "MEL"),                                                       # 0 covered, right
                  evidence_case("MEL", "NV"),                                                        # 1 covered, wrong
                  evidence_case("MEL", "MEL", quote="Melanomas always grow into the deep layers of skin."),   # 2 made-up quote
                  evidence_case("MEL", "MEL", none=True, tags=["NV"]),                                # 3 "Evidence: none", nothing relevant
                  evidence_case("BCC", "BCC"),                                  # 4 quote found, but the passage is about melanoma
                  evidence_case(None, "MEL")]                                                         # 5 no valid answer
        self.qa = {q["id"]: q for q, _ in cases_}
        self.pred = {q["id"]: p for q, p in cases_}
        self.ids = [q["id"] for q, _ in cases_]

    def test_the_quote_must_be_in_the_cited_passage_word_for_word(self):
        passage = "Melanomas show Asymmetry, and irregular borders here. Blue-gray areas may appear."
        self.assertTrue(ev.quote_in_passage("melanomas show asymmetry and irregular borders here", passage))   # case and comma
        self.assertTrue(ev.quote_in_passage("“Blue gray areas may appear”", passage + " x"))                # quote marks, hyphen
        self.assertTrue(ev.quote_in_passage("Melanomas show ... irregular borders here. Blue-gray areas", passage))   # an ellipsis
        self.assertFalse(ev.quote_in_passage("Blue-gray areas ... Melanomas show asymmetry and irregular", passage))  # order
        self.assertFalse(ev.quote_in_passage("Melanomas show asymmetry and borders here", passage))             # a word missing
        self.assertFalse(ev.quote_in_passage("elanomas show asymmetry and irregular borders", passage))        # not inside a word
        self.assertFalse(ev.quote_in_passage("irregular borders", "irregular borders"))                         # too short to count
        self.assertFalse(ev.quote_in_passage("", passage))
        self.assertFalse(ev.quote_in_passage("some words that are not there", ""))

    def test_each_diagnosis_has_a_statement_in_the_words_of_the_literature(self):
        self.assertEqual(ev.diagnosis_statement("MEL"), "The diagnosis is melanoma.")
        self.assertEqual(len({ev.diagnosis_statement(c) for c in CLASSES}), 8)
        self.assertEqual(set(ev.SUPPORT_STATEMENTS), set(CLASSES))
        for word in ("seborrheic keratosis", "solar lentigo", "lichen planus-like keratosis"):
            self.assertIn(word, ev.diagnosis_statement("BKL"))                   # the dataset's umbrella name is not enough
        self.assertIn("hemangioma", ev.diagnosis_statement("VASC"))

    def test_coverage_risk_and_supported_accuracy_use_only_verified_evidence(self):
        seen = []
        m = ev.evidence_metrics(self.qa, self.pred, keyword_scorer(record=seen))
        self.assertEqual((m["n_questions"], m["valid_answers"]), (6, 5))
        self.assertEqual((m["evidence_given"], m["evidence_none"], m["quote_found"]), (4, 1, 3))
        self.assertEqual(m["quote_accuracy"], 0.75)                       # 3 of the 4 given quotes are in their passage
        self.assertEqual((m["supported"], m["support_rate"]), (2, round(2 / 3, 4)))       # the basal cell answer is not supported
        self.assertEqual((m["n_covered"], m["coverage_verified"]), (2, round(2 / 6, 4)))
        self.assertEqual(m["selective_risk_verified"], 0.5)               # of the 2 covered answers one is wrong
        self.assertEqual(m["supported_accuracy"], round(1 / 6, 4))        # right and covered: only the first
        self.assertEqual((m["support_model"], m["support_rule"], m["support_threshold"]), ("fake-nli", "top", None))
        self.assertEqual(len(m["coverage_verified_95ci"]), 2)
        self.assertEqual(len(seen), 3 * 8)                                # every found quote is scored against all 8 statements

    def test_the_answered_diagnosis_must_score_highest_and_a_tie_is_not_support(self):
        flat = lambda premises, hypotheses: [0.5] * len(premises)          # noqa: E731
        self.assertEqual(ev.evidence_metrics(self.qa, self.pred, flat)["n_covered"], 0)          # all 8 equal: nothing is supported
        statements = {ev.diagnosis_statement(c): c for c in CLASSES}
        rival = lambda premises, hyps: [0.95 if statements[h] == "BCC" else 0.9 if statements[h] == "MEL" else 0.1   # noqa: E731
                                        for h in hyps]
        self.assertEqual(ev.evidence_metrics(self.qa, self.pred, rival)["supported"], 1)    # BCC beats MEL: only the BCC answer is supported
        tied = lambda premises, hyps: [0.9 if statements[h] in ("BCC", "MEL") else 0.1 for h in hyps]   # noqa: E731
        self.assertEqual(ev.evidence_metrics(self.qa, self.pred, tied)["supported"], 0)     # a tie at the top is not support

    def test_the_threshold_rule_scores_only_the_answered_diagnosis(self):
        seen = []
        m = ev.evidence_metrics(self.qa, self.pred, keyword_scorer(record=seen), rule="threshold")
        self.assertEqual((m["n_covered"], m["support_rule"], m["support_threshold"]), (2, "threshold", 0.5))
        self.assertEqual(len(seen), 3)
        edge = ev.evidence_metrics(self.qa, self.pred, lambda p, h: [0.5] * len(p), rule="threshold")
        self.assertEqual(edge["n_covered"], 3)                             # a score exactly at the threshold counts as supported
        self.assertEqual(ev.evidence_metrics(self.qa, self.pred, keyword_scorer(), threshold=0.95, rule="threshold")["n_covered"], 0)
        with self.assertRaises(ValueError):
            ev.evidence_metrics(self.qa, self.pred, keyword_scorer(), rule="nonsense")

    def test_without_a_support_model_only_the_quote_is_checked(self):
        m = ev.evidence_metrics(self.qa, self.pred)
        self.assertEqual((m["n_covered"], m["supported"], m["support_rate"], m["support_model"], m["support_rule"]),
                         (3, None, None, None, None))
        self.assertEqual(m["selective_risk_verified"], round(1 / 3, 4))

    def test_negative_rejection_counts_none_when_no_retrieved_passage_is_tagged_with_the_true_class(self):
        m = ev.evidence_metrics(self.qa, self.pred, keyword_scorer())
        self.assertEqual(m["n_without_relevant_passage"], 1)              # only case 3 has no passage tagged with its true class
        self.assertEqual(m["negative_rejection"], 1.0)
        self.pred[self.ids[3]]["evidence"] = {"chunk_id": "c1", "quote": "made up words that are not in the passage"}
        self.assertEqual(ev.evidence_metrics(self.qa, self.pred, keyword_scorer())["negative_rejection"], 0.0)

    def test_the_selection_never_looks_at_the_true_class(self):
        same = ev.evidence_metrics(self.qa, self.pred, keyword_scorer())
        swapped = {i: question(i, "SCC" if q["gold_label"] != "SCC" else "MEL") for i, q in self.qa.items()}   # same options, other truth
        other = ev.evidence_metrics(swapped, self.pred, keyword_scorer())
        for key in ("n_covered", "coverage_verified", "quote_found", "supported", "evidence_given"):
            self.assertEqual(same[key], other[key], key)
        self.assertNotEqual(same["selective_risk_verified"], other["selective_risk_verified"])   # only the errors depend on it

    def test_predictions_without_an_evidence_field_have_no_evidence_metrics(self):
        self.assertIsNone(ev.evidence_metrics(self.qa, {i: {"pred": "A"} for i in self.qa}))
        self.assertIsNone(ev.evidence_metrics(self.qa, {}))

    def test_a_citation_of_a_passage_that_was_not_retrieved_does_not_count(self):
        q, p = evidence_case("MEL", "MEL", chunk="c9")
        m = ev.evidence_metrics({q["id"]: q}, {q["id"]: p}, keyword_scorer())
        self.assertEqual((m["evidence_given"], m["quote_found"], m["n_covered"]), (1, 0, 0))

    def test_the_nli_scorer_reads_the_entailment_probability_whatever_the_label_order(self):
        for labels, rows, expected in (({0: "contradiction", 1: "entailment", 2: "neutral"}, [[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], 0.787),
                                       ({0: "ENTAILMENT", 1: "neutral", 2: "contradiction"}, [[2.0, 0.0, 0.0]], 0.787)):
            class Model:
                config = types.SimpleNamespace(id2label=labels)

                def __init__(self, name):
                    self.name = name

                def predict(self, pairs):
                    return rows[:len(pairs)]

            module = types.ModuleType("sentence_transformers")
            module.CrossEncoder = Model
            with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
                score = ev.nli_support_scorer("any-model")
                probs = score(["p"] * len(rows), ["h"] * len(rows))
            self.assertAlmostEqual(probs[0], expected, places=3)
            self.assertEqual(score.model_name, "any-model")
        self.assertLess(probs[0], 1.0)

    def test_the_support_check_asks_whether_a_passage_scores_highest_for_its_own_diagnosis(self):
        passages = [{"passage_id": f"p{k}", "query_class": c, "tags": [c], "text": f"{KEYWORD[c]} appears with typical signs {k}."}
                    for k, c in enumerate(CLASSES * 3)]
        passages.append({"passage_id": "two", "query_class": "MEL", "tags": ["MEL", "NV"], "text": "melanoma and nevi are compared"})
        good = ev.support_check(passages, keyword_scorer(), per_class=3)
        self.assertEqual(good["n_passages"], 24)                           # a passage with two tags is not used
        self.assertEqual((good["top1_accuracy"], good["passed"]), (1.0, True))
        self.assertEqual(set(good["per_class"]), set(CLASSES))
        self.assertTrue(all(good["confusion"][c][c] == 3 for c in CLASSES))
        self.assertGreater(good["mean_score_own_statement"], good["mean_score_other_statements"])
        useless = ev.support_check(passages, lambda p, h: [0.5] * len(p), per_class=3)
        self.assertEqual((useless["top1_accuracy"], useless["passed"]), (0.0, False))          # all equal: ties are wrong
        statements = {ev.diagnosis_statement(c): c for c in CLASSES}
        confused = ev.support_check(passages, lambda p, h: [0.9 if statements[x] in ("BCC", "SCC") else 0.1 for x in h], per_class=3)
        self.assertLess(confused["top1_accuracy"], 1.0)
        self.assertFalse(confused["passed"])
        self.assertIsNone(ev.support_check([], keyword_scorer()))

    def test_the_pass_mark_needs_070_overall_and_050_in_every_class(self):
        passages = [{"passage_id": f"p{k}", "query_class": c, "tags": [c], "text": f"{KEYWORD[c]} appears with typical signs {k}."}
                    for k, c in enumerate(CLASSES * 3)]
        right = keyword_scorer()

        def wrong_for(blind_class):
            """The model never recognises one class: overall 0.875, but that class scores 0."""
            def scorer(premises, hyps):
                return [0.5 if KEYWORD[blind_class] in p else r for p, r in zip(premises, right(premises, hyps))]
            return scorer

        def wrong_for_a_third(premises, hyps):
            """Each class is right 2 times out of 3 (0.667 overall, 0.667 per class)."""
            third = [int(p.rstrip(".").split()[-1]) >= 16 for p in premises]
            return [0.5 if t else r for t, r in zip(third, right(premises, hyps))]

        one_blind = ev.support_check(passages, wrong_for("SCC"), per_class=3)
        self.assertEqual(one_blind["top1_accuracy"], 0.875)
        self.assertEqual((one_blind["per_class"]["SCC"], one_blind["passed"]), (0.0, False))
        low = ev.support_check(passages, wrong_for_a_third, per_class=3)
        self.assertEqual((low["top1_accuracy"], low["passed"]), (0.6667, False))
        self.assertTrue(min(low["per_class"].values()) >= 0.5)


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        self.qs = [question(f"ISIC_{i:07d}", c) for i, c in enumerate(["MEL", "NV", "BCC", "AK"])]
        self.qa = as_dict(self.qs)

    def test_coverage_and_selective_risk(self):
        q = self.qs
        pred = {q[0]["id"]: {"pred": answer(q[0], "MEL"), "evidence_sufficient": "yes"},   # covered, right
                q[1]["id"]: {"pred": answer(q[1], "MEL"), "evidence_sufficient": "yes"},   # covered, wrong
                q[2]["id"]: {"pred": answer(q[2], "BCC"), "evidence_sufficient": "no"},    # abstains
                q[3]["id"]: {"pred": "Z", "evidence_sufficient": "yes"}}                    # invalid: never covered
        d = ev.diagnosis_metrics(self.qa, pred)
        self.assertEqual((d["n_covered"], d["coverage"], d["selective_risk"]), (2, 0.5, 0.5))
        self.assertEqual(d["evidence_flag_missing"], 0)
        none = ev.diagnosis_metrics(self.qa, {i: {"pred": "A", "evidence_sufficient": "no"} for i in self.qa})
        self.assertEqual((none["coverage"], none["selective_risk"]), (0.0, None))

    def test_predictions_without_the_flag_get_no_selective_scores_and_no_top_k_accuracy_is_ever_reported(self):
        d = ev.diagnosis_metrics(self.qa, {i: {"pred": "A"} for i in self.qa})
        self.assertFalse({"coverage", "selective_risk"} & set(d))
        full = ev.diagnosis_metrics(self.qa, {i: {"pred": q["answer"], "top3": [q["answer"], "A", "B"], "evidence_sufficient": "yes"}
                                              for i, q in self.qa.items()})
        self.assertFalse([k for k in full if "top" in k])                      # the extra letters of a reply are not scored
        self.assertFalse(hasattr(ev, "top_k_accuracy"))

    def test_holm_correction(self):
        adjusted = ev.holm([0.01, 0.04, 0.03, 0.5])
        self.assertEqual([round(x, 4) for x in adjusted], [0.04, 0.09, 0.09, 0.5])  # 4x0.01; 3x0.03; max(2x0.04, 0.09); 0.5
        self.assertEqual(ev.holm([0.6, 0.9]), [1.0, 1.0])                            # capped at 1
        self.assertEqual(ev.holm([]), [])


def bag_of_words(texts):
    """A fake sentence encoder for the tests: word counts over a small fixed vocabulary."""
    vocab = ["pigment", "network", "veil", "vessels", "arborizing", "white", "patch", "melanoma", "nevus"]
    return [[t.lower().split().count(w) for w in vocab] for t in texts]


class SemanticSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.qs = [question("ISIC_0000001", "MEL", "G1"), question("ISIC_0000002", "MEL", "G2"), question("ISIC_0000003", "BCC", "G3")]
        self.qa = as_dict(self.qs)
        self.refs = {"MEL": {"text": "melanoma pigment network veil", "source": "a cited source"}, "BCC": "arborizing vessels"}

    def test_cosine(self):
        self.assertEqual(ev.cosine([1, 0], [1, 0]), 1.0)
        self.assertEqual(ev.cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(ev.cosine([0, 0], [1, 1]), 0.0)

    def test_each_explanation_is_compared_with_the_reference_of_its_true_class(self):
        pred = {"ISIC_0000001": {"rationale": "melanoma pigment network veil"},       # same words: 1.0
                "ISIC_0000002": {"rationale": "white patch"},                           # nothing shared: 0.0
                "ISIC_0000003": {"rationale": "arborizing vessels"}}                    # same words: 1.0
        r = ev.semantic_similarity(self.qa, pred, self.refs, bag_of_words)
        self.assertEqual((r["mean"], r["n_questions_with_rationale"]), (round(2 / 3, 4), 3))
        self.assertEqual(r["per_class"], {"MEL": 0.5, "BCC": 1.0})
        lo, hi = r["95ci"]
        self.assertTrue(0.0 <= lo <= r["mean"] <= hi <= 1.0)

    def test_questions_without_an_explanation_are_left_out_and_a_missing_reference_stops(self):
        r = ev.semantic_similarity(self.qa, {"ISIC_0000001": {"pred": "A", "rationale": ""}, "ISIC_0000002": {"pred": "B"},
                                             "ISIC_0000003": {"rationale": "arborizing vessels"}}, self.refs, bag_of_words)
        self.assertEqual((r["mean"], r["n_questions_with_rationale"]), (1.0, 1))
        self.assertIsNone(ev.semantic_similarity(self.qa, {}, self.refs, bag_of_words))
        with self.assertRaises(ValueError):
            ev.semantic_similarity(self.qa, {"ISIC_0000003": {"rationale": "x"}}, {"MEL": "melanoma"}, bag_of_words)

    def test_the_best_matching_reference_text_of_the_true_class_counts(self):
        refs = {"MEL": {"texts": [{"text": "pigment network"}, {"text": "melanoma veil"}]},
                "BCC": {"texts": [{"text": "arborizing vessels"}]}}
        pred = {"ISIC_0000001": {"rationale": "melanoma veil"},          # the second MEL text: 1.0
                "ISIC_0000002": {"rationale": "pigment network"},        # the first MEL text: 1.0
                "ISIC_0000003": {"rationale": "melanoma veil"}}          # nothing shared with the BCC text: 0.0
        r = ev.semantic_similarity(self.qa, pred, refs, bag_of_words)
        self.assertEqual(r["per_class"], {"MEL": 1.0, "BCC": 0.0})
        self.assertEqual(r["n_reference_texts"], {"BCC": 1, "MEL": 2})
        self.assertNotIn("longer_than_the_model_reads", r)
        with self.assertRaisesRegex(ValueError, "the class BCC has no reference text"):
            ev.semantic_similarity(self.qa, pred, {**refs, "BCC": {"texts": []}}, bag_of_words)
        self.assertEqual(ev.reference_lists({"MEL": "a", "NV": {"text": "b"}, "BCC": {"texts": [{"text": "c"}, {"text": "d"}]}}),
                         {"MEL": ["a"], "NV": ["b"], "BCC": ["c", "d"]})

    def test_texts_longer_than_the_model_reads_are_counted(self):
        def encode(texts):
            return bag_of_words(texts)
        encode.max_tokens, encode.n_tokens = 2, lambda t: len(t.split())
        refs = {"MEL": {"texts": [{"text": "pigment network veil"}, {"text": "veil"}]}, "BCC": "arborizing vessels"}
        pred = {"ISIC_0000001": {"rationale": "melanoma pigment network veil"}, "ISIC_0000003": {"rationale": "vessels"}}
        r = ev.semantic_similarity(self.qa, pred, refs, encode)
        self.assertEqual(r["longer_than_the_model_reads"], {"max_tokens": 2, "reference_texts": 1, "rationales": 1})

    def test_text_metrics_reports_it_only_with_reference_texts(self):
        pred = {"ISIC_0000003": {"rationale": "arborizing vessels"}}
        self.assertEqual(ev.text_metrics(self.qa, pred, None), {})
        out = ev.text_metrics(self.qa, pred, self.refs, encode=bag_of_words)
        self.assertEqual(out["semantic_similarity_cosine"]["mean"], 1.0)


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


# ----------------------------------------------------------------------------------------------------------------------
# compare (McNemar with Holm)
# ----------------------------------------------------------------------------------------------------------------------


def runs_of(right_lists):
    ids = [f"ISIC_{i:07d}" for i in range(len(next(iter(right_lists.values()))))]
    return {name: dict(zip(ids, right)) for name, right in right_lists.items()}


class CompareTests(unittest.TestCase):
    def test_every_pair_is_tested_and_corrected(self):
        rep = ev.compare(runs_of({"a": [True] * 10 + [False] * 2, "b": [False] * 10 + [True] * 2,
                                       "c": [True] * 12}))
        self.assertEqual(rep["accuracy"], {"a": 0.8333, "b": 0.1667, "c": 1.0})
        self.assertEqual([(p["a"], p["b"]) for p in rep["pairs"]], [("a", "b"), ("a", "c"), ("b", "c")])
        ab = rep["pairs"][0]
        self.assertEqual((ab["a_right_b_wrong"], ab["a_wrong_b_right"]), (10, 2))
        self.assertAlmostEqual(ab["p_value"], 2 * (1 + 12 + 66) / 4096, places=6)
        self.assertEqual([p["p_holm"] for p in rep["pairs"]], [round(x, 6) for x in ev.holm([p["p_value"] for p in rep["pairs"]])])
        self.assertTrue(all(p["p_holm"] >= p["p_value"] for p in rep["pairs"]))

    def test_runs_on_other_cases_are_refused(self):
        runs = runs_of({"a": [True, False]})
        runs["b"] = {"ISIC_0000000": True, "ISIC_9999999": False}
        with self.assertRaises(ValueError):
            ev.compare(runs)

    def test_the_command_reads_each_answer_with_its_own_questions(self):
        cases = [{"isic_id": f"ISIC_{i:07d}", "split": "test", "label": c, "group_id": f"ISIC_{i:07d}",
                  "clinical_text": "Patient: 50-year-old male.", "image_path": f"processed_images/ISIC_{i:07d}.jpg"}
                 for i, c in enumerate(["MEL", "NV", "BCC", "AK"])]
        with tempfile.TemporaryDirectory() as d:
            files = {}
            for view in ("text_only", "multimodal"):
                qs = [bd.question(c, view) for c in cases]
                files[view] = os.path.join(d, f"{view}.jsonl")
                write_jsonl(files[view], qs)
                write_jsonl(os.path.join(d, f"{view}_pred.jsonl"),
                            [{"id": q["id"], "pred": q["answer"] if view == "multimodal" or k < 2 else "Z"} for k, q in enumerate(qs)])
            out = os.path.join(d, "cmp.json")
            with contextlib.redirect_stdout(io.StringIO()):
                ev.compare_main(["--out", out,
                              "--run", f"text={files['text_only']},{os.path.join(d, 'text_only_pred.jsonl')}",
                              "--run", f"mm={files['multimodal']},{os.path.join(d, 'multimodal_pred.jsonl')}"])
            rep = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(rep["accuracy"], {"text": 0.5, "mm": 1.0})
        self.assertEqual(rep["pairs"][0]["a_wrong_b_right"], 2)
        self.assertIn("sha256", rep["inputs"]["text"]["questions"])


# ----------------------------------------------------------------------------------------------------------------------
# prompt, answer rule and requests of run.py
# ----------------------------------------------------------------------------------------------------------------------


def run_cases(n, view_has_image):
    labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    return [{"isic_id": f"ISIC_{i:07d}", "split": "val", "label": labels[i % 8], "group_id": f"ISIC_{i:07d}",
             "clinical_text": "Patient: 50-year-old male. Skin lesion on the trunk.", "image_path": f"processed_images/ISIC_{i:07d}.jpg"}
            for i in range(n)]


class PromptAndAnswerTests(unittest.TestCase):
    def test_the_prompt_lists_the_question_the_eight_options_and_one_instruction(self):
        q = bd.question(run_cases(1, False)[0], "text_only")
        text = rl.user_text(q)
        self.assertTrue(text.startswith(q["question"]))
        for letter, option in q["options"].items():
            self.assertIn(f"\n{letter}. {option}", text)
        self.assertTrue(text.endswith(rl.INSTRUCTIONS["rationale"]))
        self.assertNotIn(q["gold_label"], text.replace(q["options"][q["answer"]], ""))   # the code of the class is not shown

    def test_the_answer_rule(self):
        cases_ = {"Reasoning.\nAnswer: C": "C", "**Answer:** (D)": "D", "answer: b": None, "Answer: Actinic keratosis": None,
                  "Answer: A\nWait, Answer: B": "B", "E": "E", "(F).": "F", "The answer is probably G": None, "": None,
                  "Answer: Z": None, "Answer: C, A, F": "C", "Answer: C (Melanoma), A": "C"}
        for text, letter in cases_.items():
            with self.subTest(text):
                self.assertEqual(rl.parse_answer(text)["pred"], letter)
        self.assertEqual(rl.parse_answer("Reasoning and evidence: Asymmetric pigment network and a blue-white veil.\nAnswer: C"),
                         {"pred": "C", "rationale": "Asymmetric pigment network and a blue-white veil."})
        self.assertEqual(rl.parse_answer("**Reasoning and evidence:** Irregular borders.\n**Answer:** C (Melanoma)"),
                         {"pred": "C", "rationale": "Irregular borders."})
        self.assertEqual(rl.parse_answer("Answer: C (Melanoma).\nReasoning and evidence: The lesion is asymmetric."),
                         {"pred": "C", "rationale": "The lesion is asymmetric."})              # the answer first: the text after it
        for reply, letter in (("Answer: C (Melanoma)", "C"), ("**Answer:** C (Melanoma)", "C"),
                              ("Answer: C (Benign keratosis-like lesion (seborrheic keratosis / solar lentigo))", "C"),
                              ("Answer: Melanoma (C)", "C"), ("Answer: C (Squamous cell carcinoma)", "C"),   # the letter decides
                              ("Answer: Option D (Dermatofibroma).", "D")):
            self.assertEqual(rl.parse_answer(reply)["pred"], letter, reply)
        self.assertEqual(rl.parse_answer("Reasoning and evidence: Network.\nAnswer: C (Melanoma)"),
                         {"pred": "C", "rationale": "Network."})                          # the name is not part of the reasoning
        self.assertEqual(rl.parse_answer("Some reasoning.\nAnswer: B\nWait.\nAnswer: E")["pred"], "E")      # the last answer line counts
        self.assertEqual(rl.parse_answer("Answer: Z"), {"pred": None, "rationale": "Answer: Z"})
        self.assertEqual(set(rl.parse_answer("Answer: C")), {"pred", "rationale"})             # no top-3 list and no evidence flag

    def test_the_instruction_asks_for_the_reasoning_and_evidence_and_one_letter_and_nothing_else(self):
        text = rl.INSTRUCTIONS["rationale"]
        self.assertIn("Reasoning and evidence: <at most three sentences", text)
        self.assertIn("Answer: <the letter of the most likely option> (<the name of that option>)", text)
        self.assertIn("name the findings of this case, in the image and/or in the patient facts, that support your choice", text)
        self.assertIn("Answer: <the letter of the most likely option> (<the name of that option>)", rl.INSTRUCTIONS["answer_only"])
        for gone in ("three most likely", "Evidence sufficient", "yes or no"):
            self.assertNotIn(gone, text + rl.INSTRUCTIONS["answer_only"])
        self.assertEqual(rl.PROMPT_NAME, "llm_only")


class RequestTests(unittest.TestCase):
    def test_the_four_research_models_and_their_fixed_settings(self):
        self.assertEqual(sorted(rl.MODELS), ["claude-sonnet-5-5", "gemini-3.8-flash", "gpt-4o", "llama-4-maverick"])
        for name, cfg in rl.MODELS.items():
            with self.subTest(name):
                if cfg["api"] == "anthropic":
                    self.assertNotIn("temperature", cfg)                 # Claude rejects a temperature
                    self.assertEqual(cfg["thinking"], {"type": "between_tools"})  # no hidden reasoning
                else:
                    self.assertEqual((cfg["temperature"], cfg["seed"]), (0.0, 0))
                    self.assertEqual(len(cfg["route"]), 1)              # one pinned host

    def test_claude_request(self):
        p = rl.anthropic_params(rl.MODELS["claude-sonnet-5-5"], "sys", "txt", "IMG")
        self.assertEqual(p["model"], "claude-sonnet-5-5")
        self.assertEqual(p["system"], "sys")
        self.assertEqual(p["messages"][0]["content"][0]["source"], {"type": "base64", "media_type": "image/jpeg", "data": "IMG"})
        self.assertEqual(p["messages"][0]["content"][1], {"type": "text", "text": "txt"})
        self.assertEqual((p["thinking"], p["output_config"]), ({"type": "between_tools"}, {"effort": "low"}))
        self.assertNotIn("temperature", p)
        self.assertEqual(len(rl.anthropic_params(rl.MODELS["claude-sonnet-5-5"], "s", "t", None)["messages"][0]["content"]), 1)

    def test_claude_reply_and_refusal(self):
        ns = types.SimpleNamespace
        r = ns(content=[ns(type="thinking", text=""), ns(type="text", text="Answer: B")], stop_reason="end_turn",
               stop_details=None, model="claude-sonnet-5-5", usage=ns(input_tokens=7, output_tokens=3))
        self.assertEqual(rl.anthropic_reply(r), {"text": "Answer: B", "finish_reason": "end_turn", "served_by": "claude-sonnet-5-5",
                                                 "tokens_in": 7, "tokens_out": 3})
        r = ns(content=[], stop_reason="refusal", stop_details=ns(category="bio"), model="m", usage=ns(input_tokens=1, output_tokens=0))
        self.assertEqual(rl.anthropic_reply(r)["finish_reason"], "refusal: bio")

    def test_openrouter_request_pins_the_host(self):
        url, body = rl.openrouter_request(rl.MODELS["gpt-4o"], "sys", "txt", "IMG")
        self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(body["model"], "openai/gpt-4o-2024-08-06")
        self.assertEqual(body["provider"], {"order": ["openai"], "allow_fallbacks": False})
        self.assertEqual(body["messages"][1]["content"][1]["image_url"], {"url": "data:image/jpeg;base64,IMG", "detail": "high"})
        self.assertEqual((body["temperature"], body["seed"], body["max_tokens"]), (0.0, 0, 1024))
        _, body = rl.openrouter_request(rl.MODELS["gemini-3.8-flash"], "sys", "txt", None)
        self.assertEqual((body["reasoning"], len(body["messages"][1]["content"])), ({"effort": "minimal"}, 1))
        _, body = rl.openrouter_request(rl.MODELS["llama-4-maverick"], "sys", "txt", "IMG")
        self.assertNotIn("detail", body["messages"][1]["content"][1]["image_url"])
        self.assertNotIn("reasoning", body)

    def test_openrouter_replies(self):
        ok = rl.openrouter_reply({"model": "openai/gpt-4o-2024-08-06", "provider": "OpenAI",
                                  "choices": [{"message": {"content": "Answer: A"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
        self.assertEqual(ok, {"text": "Answer: A", "finish_reason": "stop", "served_by": "openai/gpt-4o-2024-08-06@OpenAI",
                              "tokens_in": 5, "tokens_out": 2})
        self.assertEqual(rl.openrouter_reply({"choices": [{"message": {"content": None, "refusal": "no"}}]})["finish_reason"],
                         "refusal")
        with self.assertRaises(rl.Refused):
            rl.openrouter_reply({"error": {"code": 403, "message": "Input was flagged by moderation"}})
        with self.assertRaises(rl.ApiError):
            rl.openrouter_reply({"error": {"code": 402, "message": "Insufficient credits"}})

    def test_the_claude_client_uses_the_official_package_and_stops_on_an_api_error(self):
        ns = types.SimpleNamespace
        sent = {}

        class StatusError(Exception):
            def __init__(self, status, message):
                super().__init__(message)
                self.status_code, self.message = status, message

        class Messages:
            def create(self, **kw):
                sent.update(kw)
                if kw["system"] == "fail":
                    raise StatusError(401, "invalid x-api-key")
                return ns(content=[ns(type="text", text="Answer: C")], stop_reason="end_turn", stop_details=None,
                          model="claude-sonnet-5-5", usage=ns(input_tokens=9, output_tokens=4))

        def make(api_key, max_retries, timeout):
            sent["client"] = (api_key, max_retries, timeout)
            return ns(messages=Messages())

        fake = ns(Anthropic=make, APIStatusError=StatusError, APIConnectionError=ConnectionError)
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            client = rl.AnthropicClient(rl.MODELS["claude-sonnet-5-5"], "KEY")
            self.assertEqual(client.ask("sys", "txt", None)["text"], "Answer: C")
            self.assertEqual(sent["client"], ("KEY", 6, 180.0))
            self.assertEqual(sent["model"], "claude-sonnet-5-5")
            with self.assertRaises(rl.ApiError):
                client.ask("fail", "txt", None)

    def test_load_env_reads_keys_without_overriding_the_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, ".env")
            path.write_text("# comment\nA_KEY=\"one\"\nB_KEY = two\nbroken line\nEMPTY=\n=novalue\nC_KEY='x=y'\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"B_KEY": "terminal"}, clear=True):
                unusable = common.load_env(path)
                self.assertEqual((os.environ["A_KEY"], os.environ["B_KEY"], os.environ["C_KEY"]), ("one", "terminal", "x=y"))
                self.assertNotIn("EMPTY", os.environ)
                self.assertEqual(unusable, [4, 5, 6])                    # line numbers only, never the lines themselves
            self.assertEqual(common.load_env(Path(d, "missing.env")), [])   # a missing file is fine
            self.assertIs(rl.load_env, common.load_env)

    def test_an_empty_variable_in_the_terminal_does_not_hide_the_env_file_and_a_secret_is_never_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, ".env")
            path.write_text("HF_TOKEN=hf_secret\nOPENROUTER_API_KEY=sk-or-secret-without-name-part\nsk-or-just-a-value\n",
                            encoding="utf-8")
            with mock.patch.dict(os.environ, {"HF_TOKEN": "  "}, clear=True):
                unusable = common.load_env(path)
                self.assertEqual(os.environ["HF_TOKEN"], "hf_secret")
            self.assertEqual(unusable, [3])
            self.assertNotIn("sk-or-just-a-value", repr(unusable))

    def test_busy_servers_are_retried_and_other_errors_stop_at_once(self):
        class Answer(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def error(code):
            return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b"detail"))

        calls = []

        def busy_then_ok(req, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise error(429)
            return Answer(b'{"ok": true}')

        with mock.patch("urllib.request.urlopen", busy_then_ok), mock.patch("time.sleep"):
            self.assertEqual(rl.post_json("https://x", {}, {}), {"ok": True})
        self.assertEqual(len(calls), 2)
        with mock.patch("urllib.request.urlopen", side_effect=error(400)) as bad, mock.patch("time.sleep"):
            with self.assertRaises(rl.ApiError):
                rl.post_json("https://x", {}, {})
        self.assertEqual(bad.call_count, 1)                           # a bad request is not sent again
        moderated = urllib.error.HTTPError("u", 403, "x", {}, io.BytesIO(b'{"error": "flagged by moderation"}'))
        with mock.patch("urllib.request.urlopen", side_effect=moderated), mock.patch("time.sleep"):
            with self.assertRaises(rl.Refused):
                rl.post_json("https://x", {}, {})


if __name__ == "__main__":
    unittest.main()
