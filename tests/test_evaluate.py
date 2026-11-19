"""Scores of finished runs (src/evaluate.py): the report of a run, and comparing two runs on the same questions."""

import contextlib
import io
import json
import unittest
from pathlib import Path

from src import datasets, evaluate

from tests.fixtures import FakeModel, RunProject, json_answer, listed

ROOT = Path(__file__).resolve().parents[1]


class Nevus(FakeModel):
    """Says "nevus" for every image, with "evidence not sufficient"."""

    def predict(self, images, prompt):
        return super().predict(images, prompt) | {"text": json_answer("nevus", listed(prompt), "no")}


def answered(path: str, label: str, group_column: str, group: str, prediction: str | None) -> dict:
    """One saved answer of the current format: the diagnosis alone, "yes", no evidence; None is an invalid answer."""
    return {"path": path, "label": label, group_column: group, "prediction": prediction, "differential": None if prediction is None else [prediction],
            "evidence_sufficient": None if prediction is None else "yes", "reason": None if prediction is None else "r",
            "claims": None if prediction is None else [{"statement": "c", "evidence": []}], "evidence": None if prediction is None else [], "invalid": "not asked" if prediction is None else None, "raw_text": prediction,
            "finish_reason": "STOP", "model_version": "m-1"}


def write_run(folder: Path, truth: list[str], predicted: list, groups: list[str], dataset: str = "clinic_photo", answers: list | None = None,
              first_phase: bool = False):
    """A finished run of the current answer format. ``answers`` (differential and flag per question) say more than the diagnosis; without
    them each valid answer is its diagnosis, "yes" and nothing else. ``first_phase``: the answer format of Version 1 is missing."""
    folder.mkdir(parents=True)
    kind = datasets.DATASETS[dataset]
    card = {"status": "complete", "config": {"dataset": dataset}, "split": "test", "labels": sorted(set(truth)),
            f"n_{kind.item}s": len(truth), f"{kind.item}s_sha256": "same", "name": "m", "run_id": folder.name,
            "model": {"model_id": "m-1"}, "model_versions_returned": ["m-1"], "started_utc": "s", "finished_utc": "f",
            "prompt_file": "p.txt", "prompt_sha256": "0" * 64, "dataset_table": {"file": "t.csv", "sha256": "0" * 64},
            "git": {"commit": "c" * 40}, "sessions": []}
    if not first_phase:
        card["answer_format"] = "json-v2"
    (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
    rows = []
    for i, (t, p, g) in enumerate(zip(truth, predicted, groups)):
        differential, flag = answers[i] if answers is not None else (None if p is None else [p], None if p is None else "yes")
        rows.append(answered(f"test/{t}/{i}.jpg", t, kind.group_column, g, p) | {"differential": differential, "evidence_sufficient": flag})
    (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return folder


class ReportTests(RunProject):
    def test_a_run_of_the_first_phase_is_refused_with_a_pointer_to_the_archive(self):
        folder = write_run(self.base / "phase1" / "test_1", ["a", "a", "b", "b"], ["a", "b", "b", None], ["L1", "L2", "L3", "L4"],
                           first_phase=True)
        with self.assertRaisesRegex(ValueError, "'class name \\(first phase\\)', not 'json-v2'.*archive/v1/"):
            evaluate.write_report(folder)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["predictions.jsonl", "run_card.json"])  # nothing was written

    def test_a_run_of_the_current_form_reports_top3_and_the_selective_scores(self):
        truth = ["melanoma", "melanoma", "nevus", "nevus"]
        answers = [(["melanoma", "nevus"], "yes"), (["nevus", "melanoma"], "no"), (["nevus", "melanoma"], "yes"), None]
        folder = write_run(self.base / "current" / "test_1", truth, ["melanoma", "nevus", "nevus", None], ["L1", "L2", "L3", "L4"],
                           answers=[a or (None, None) for a in answers])
        path = evaluate.write_report(folder)
        self.assertEqual(path.name, "report.md")
        m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(m["main"]["top3_balanced_accuracy"], 0.75)  # melanoma 2 of 2 in the differential; nevus 1 of 2 (one invalid)
        d = m["main"]["selective"]
        self.assertEqual((d["coverage"], d["selective_risk"]), (0.5, 0.0))
        text = path.read_text(encoding="utf-8")
        for expected in ("| Macro-averaged top-3 accuracy (the true class in the differential) | 75.0% |", "### Selective prediction"):
            self.assertIn(expected, text)
        self.assertNotIn("first phase", text)


class VerifyAnswersTests(RunProject):
    """verify_answers on a run of 4 answers (3 valid, 1 invalid) that is damaged one way at a time."""

    def setUp(self):
        super().setUp()
        self.folder = write_run(self.base / "r" / "test_1", ["a", "a", "b", "b"], ["a", "b", "b", None], ["L1", "L2", "L3", "L4"],
                                answers=[(["a", "b"], "yes"), (["b", "a"], "no"), (["b"], "yes"), (None, None)])
        self.card = json.loads((self.folder / "run_card.json").read_text(encoding="utf-8"))
        self.rows = evaluate.read_jsonl(self.folder / "predictions.jsonl")

    def problems(self, rows=None, complete=True):
        return evaluate.verify_answers(self.card, self.rows if rows is None else rows, complete)

    def damaged(self, number, **changes):
        return [row | changes if i == number else row for i, row in enumerate(self.rows)]

    def test_a_consistent_run_has_no_problem_and_is_scored(self):
        self.assertEqual(self.problems(), [])
        self.assertEqual(evaluate.score_run(self.folder)[2]["main"]["n"], 4)

    def test_an_answer_that_is_missing_a_field_is_named(self):
        rows = [{k: v for k, v in row.items() if k != "evidence"} if i == 1 else row for i, row in enumerate(self.rows)]
        self.assertEqual(self.problems(rows), ["answer 2: no field evidence"])
        no_claims = [{k: v for k, v in row.items() if k != "claims"} if i == 0 else row for i, row in enumerate(self.rows)]
        self.assertEqual(self.problems(no_claims), ["answer 1: no field claims"])
        self.assertEqual(self.problems([{k: v for k, v in self.rows[0].items() if k != "patient_id"}], complete=False),
                         ["answer 1: no field patient_id"])

    def test_a_question_answered_twice_would_count_twice_and_is_refused(self):
        self.assertEqual(self.problems(self.rows + [self.rows[0]], complete=False), ["test/a/0.jpg is answered 2 times"])

    def test_a_complete_run_must_answer_every_question_an_unfinished_one_may_not(self):
        self.assertEqual(self.problems(self.rows[:3]), ["3 questions answered, 4 expected"])
        self.assertEqual(self.problems(self.rows[:3], complete=False), [])

    def test_classes_outside_the_run_are_refused(self):
        self.assertIn("the true class 'z' is not a class of the run", self.problems(self.damaged(0, label="z"))[0])
        self.assertIn("the differential is not different classes of the run", self.problems(self.damaged(0, differential=["a", "z"]))[0])
        self.assertIn("the differential is not different classes of the run", self.problems(self.damaged(1, differential=["b", "a", "a"]))[0])

    def test_the_diagnosis_must_be_the_first_of_the_differential_and_the_flag_yes_or_no(self):
        self.assertIn("not the first of the differential", self.problems(self.damaged(0, prediction="b"))[0])
        self.assertIn("not the first of the differential", self.problems(self.damaged(0, differential=[]))[0])
        self.assertIn("the flag 'maybe' is not yes or no", self.problems(self.damaged(0, evidence_sufficient="maybe"))[0])

    def test_a_closed_book_answer_holds_no_evidence(self):
        self.assertIn("a closed-book answer must hold no evidence", self.problems(self.damaged(0, evidence=["K:1"]))[0])
        self.assertIn("a closed-book claim must rely on no passage", self.problems(self.damaged(0, claims=[{"statement": "c", "evidence": ["K:1"]}]))[0])
        self.card["config"]["approach"] = "standard_rag"  # a retrieval approach may cite what it was given
        self.assertEqual(self.problems(self.damaged(0, evidence=["K:1"], claims=[{"statement": "c", "evidence": ["K:1"]}])), [])

    def test_a_valid_answer_holds_at_least_one_claim_and_an_invalid_one_none(self):
        self.assertIn("a valid answer must hold at least one claim", self.problems(self.damaged(0, claims=[]))[0])
        self.assertIn("a valid answer must hold at least one claim", self.problems(self.damaged(0, claims=None))[0])
        self.assertIn("an invalid answer must hold its reason", self.problems(self.damaged(3, claims=[{"statement": "c", "evidence": []}]))[0])

    def test_an_invalid_answer_holds_its_reason_and_nothing_else_and_a_valid_one_no_reason(self):
        self.assertIn("an invalid answer must hold its reason", self.problems(self.damaged(3, invalid=None))[0])
        self.assertIn("an invalid answer must hold its reason", self.problems(self.damaged(3, differential=["a"]))[0])
        self.assertIn("a valid answer must not hold an invalid reason", self.problems(self.damaged(0, invalid="x"))[0])

    def test_a_comparison_refuses_a_damaged_run_and_a_run_of_the_first_phase(self):
        other = write_run(self.base / "r2" / "test_1", ["a", "a", "b", "b"], ["a", "a", "b", "b"], ["L1", "L2", "L3", "L4"])
        self.assertEqual(evaluate.compare(self.folder, other)["n"], 4)
        (other / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evaluate.read_jsonl(other / "predictions.jsonl")[:3]
                                                          + [evaluate.read_jsonl(other / "predictions.jsonl")[0]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not consistent, so nothing is scored: test/a/0.jpg is answered 2 times"):
            evaluate.compare(self.folder, other)
        old = write_run(self.base / "r3" / "test_1", ["a", "a", "b", "b"], ["a", "a", "b", "b"], ["L1", "L2", "L3", "L4"], first_phase=True)
        with self.assertRaisesRegex(ValueError, "'class name \\(first phase\\)', not 'json-v2'"):
            evaluate.compare(self.folder, old)

    def test_scoring_and_comparing_stop_at_a_damaged_file_before_any_score(self):
        (self.folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows + [self.rows[0]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not consistent, so nothing is scored: test/a/0.jpg is answered 2 times"):
            evaluate.write_report(self.folder)
        self.assertFalse((self.folder / "report.md").exists())


class CompareTests(RunProject):
    def test_two_runs_on_the_same_questions_are_compared_and_others_are_refused(self):
        a, b = self._run(FakeModel(), "val", run_id="v1"), self._run(Nevus(), "val", run_id="v2")
        c = evaluate.compare(a, b)
        self.assertEqual((c["n"], c["both_right"], c["only_a"], c["only_b"], c["both_wrong"]), (4, 0, 2, 2, 0))
        self.assertEqual((c["mcnemar_p"], c["balanced_accuracy_difference"], c["macro_f1_difference"]), (1.0, 0.0, 0.0))
        low, high = c["balanced_accuracy_difference_ci95"]
        self.assertTrue(low <= 0 <= high)
        text = evaluate.comparison_text(c)
        for expected in ("**Balanced accuracy, B minus A: +0.0 points**", "paired bootstrap over patients", "The interval holds 0",
                         "| **Run B right** | 0 | 2 |", "McNemar's exact test (supporting", "p = 1.0000"):
            self.assertIn(expected, text)
        same = evaluate.compare(a, a)
        self.assertEqual((same["only_a"], same["only_b"], same["balanced_accuracy_difference_ci95"]), (0, 0, (0.0, 0.0)))
        short = self._run(FakeModel(), "val", run_id="v3", limit=3)
        with self.assertRaisesRegex(ValueError, "differ in their questions"):
            evaluate.compare(a, short)
        test = self._run(FakeModel(), "test", run_id="t1")
        with self.assertRaisesRegex(ValueError, "differ in their split"):
            evaluate.compare(a, test)
        card = json.loads((b / "run_card.json").read_text(encoding="utf-8"))
        card["status"] = "running"
        (b / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a finished run"):
            evaluate.compare(a, b)
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            evaluate.main(["compare", str(a), str(a)])
        self.assertIn("McNemar's exact test (supporting", shown.getvalue())

    def test_the_paired_interval_resamples_whole_patients_of_both_runs_together(self):
        """Run B misses all 4 photos of patient a1 and nothing else. Resampling the two patients of class a (with replacement) gives
        B's recall of a as 0, 1/2 or 1 with chances 1/4, 1/2, 1/4, so the difference in balanced accuracy is -50, -25 or 0 points
        and the 95% interval is exactly -50 to 0. Resampling the 8 photos one by one would make losing all of them rare (under 1%)
        and give an interval of about -44 to -6 instead."""
        photos = [("a", "a1")] * 4 + [("a", "a2")] * 4 + [("b", "b1")] * 2 + [("b", "b2")] * 2
        truth, groups = [t for t, _ in photos], [g for _, g in photos]
        a = write_run(self.base / "run_A", truth, truth, groups)
        b = write_run(self.base / "run_B", truth, ["b" if i < 4 else t for i, t in enumerate(truth)], groups)
        c = evaluate.compare(a, b)
        self.assertEqual((c["only_a"], c["only_b"], c["balanced_accuracy_difference"]), (4, 0, -0.25))
        self.assertEqual(c["balanced_accuracy_difference_ci95"], (-0.5, 0.0))
        self.assertEqual(c["mcnemar_p"], 0.125)  # 2 x (1/2)^4

    def test_the_comparison_calls_a_difference_beyond_chance_only_when_its_interval_leaves_out_0(self):
        c = evaluate.compare(self._run(FakeModel(), "val", run_id="v1"), self._run(Nevus(), "val", run_id="v2"))
        for ci, beyond in (((0.01, 0.2), True), ((-0.3, -0.02), True), ((-0.1, 0.1), False), ((0.0, 0.2), False)):
            with self.subTest(ci):
                text = evaluate.comparison_text(c | {"balanced_accuracy_difference_ci95": ci})
                self.assertEqual("The interval does not hold 0: the runs differ by more than chance." in text, beyond)
                self.assertEqual("The interval holds 0: the difference may be chance." in text, not beyond)

    def test_the_photo_run_and_the_photo_record_run_of_the_same_photos_can_be_compared_but_no_other_datasets(self):
        """The photo + record questions are the photo dataset's photos with their records: the two runs differ in dataset and in
        the questions' fingerprint, yet answer the same photos, and `compare` must accept exactly that pair."""
        truth = ["a", "a", "b", "b"]
        runs = {}
        for name, dataset, predicted in (("image", "clinic_photo", ["a", "b", "b", "a"]), ("multi", "clinic_photo_record", ["a", "a", "b", "b"]),
                                         ("text", "published_text", ["a", "a", "b", "b"])):
            folder = self.base / f"pair_{name}"
            folder.mkdir()
            kind = datasets.DATASETS[dataset]
            card = {"status": "complete", "config": {"dataset": dataset}, "split": "test", "labels": ["a", "b"],
                    f"n_{kind.item}s": 4, f"{kind.item}s_sha256": f"fingerprint of {dataset}", "answer_format": "json-v2"}
            (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
            rows = [answered(f"test/{t}/{i}.jpg", t, kind.group_column, f"L{i}", p) for i, (t, p) in enumerate(zip(truth, predicted))]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs[name] = folder
        c = evaluate.compare(runs["image"], runs["multi"])
        self.assertEqual((c["dataset"], c["n"], c["only_a"], c["only_b"]), ("clinic_photo and clinic_photo_record", 4, 0, 2))  # the facts fixed 2
        self.assertEqual(c["balanced_accuracy_difference"], 0.5)
        self.assertIn("clinic_photo and clinic_photo_record dataset, test split, 4 questions", evaluate.comparison_text(c))
        with self.assertRaisesRegex(ValueError, "differ in their dataset"):
            evaluate.compare(runs["image"], runs["text"])
        with self.assertRaisesRegex(ValueError, "differ in their dataset"):
            evaluate.compare(runs["multi"], runs["text"])
        rows = [json.loads(x) for x in (runs["multi"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["label"] = "b"  # another photo, or another class for the same photo
        (runs["multi"] / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same photos with the same classes"):
            evaluate.compare(runs["image"], runs["multi"])


    def test_the_three_clinic_views_are_compared_by_question_although_their_files_end_differently(self):
        """A clinic photo, its record and the two together are one question: <split>/<class>/<img_id> with .jpg or .txt."""
        truth = ["a", "a", "b", "b"]
        runs = {}
        for name, dataset, predicted, ending in (("photo", "clinic_photo", ["a", "b", "b", "a"], "jpg"),
                                                 ("record", "clinic_record", ["a", "a", "b", "b"], "txt"),
                                                 ("both", "clinic_photo_record", ["a", "a", "b", "a"], "jpg")):
            folder = self.base / f"clinic_{name}"
            folder.mkdir()
            kind = datasets.DATASETS[dataset]
            card = {"status": "complete", "config": {"dataset": dataset}, "split": "test", "labels": ["a", "b"],
                    f"n_{kind.item}s": 4, f"{kind.item}s_sha256": f"fingerprint of {dataset}", "answer_format": "json-v2"}
            (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
            rows = [answered(f"test/{t}/P{i}_1_a.{ending}", t, kind.group_column, f"P{i}", p) for i, (t, p) in enumerate(zip(truth, predicted))]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs[name] = folder
        c = evaluate.compare(runs["photo"], runs["record"])  # the record fixes two photos the photo got wrong
        self.assertEqual((c["dataset"], c["n"], c["only_a"], c["only_b"]), ("clinic_photo and clinic_record", 4, 0, 2))
        self.assertEqual(c["groups"], "patients")
        c = evaluate.compare(runs["record"], runs["both"])
        self.assertEqual((c["n"], c["only_a"], c["only_b"]), (4, 1, 0))
        self.assertEqual(evaluate.compare(runs["both"], runs["photo"])["only_b"], 0)
        rows = [json.loads(x) for x in (runs["record"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["path"] = "test/a/P9_1_a.txt"  # another question
        (runs["record"] / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same photos with the same classes"):
            evaluate.compare(runs["photo"], runs["record"])
        rows[0]["path"], rows[0]["label"] = "test/a/P0_1_a.txt", "b"  # the same question, another class
        (runs["record"] / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same photos with the same classes"):
            evaluate.compare(runs["photo"], runs["record"])


if __name__ == "__main__":
    unittest.main()
