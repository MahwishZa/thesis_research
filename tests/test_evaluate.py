"""Scores of finished runs (src/evaluate.py): comparing two runs on the same questions."""

import contextlib
import io
import json
import unittest
from pathlib import Path

from src import datasets, evaluate, metrics

from tests.fixtures import FakeModel, RunProject

ROOT = Path(__file__).resolve().parents[1]


class CompareTests(RunProject):
    def test_two_runs_on_the_same_questions_are_compared_and_others_are_refused(self):
        class Nevus(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": "nevus"}

        a, b = self._run(FakeModel(), "val", run_id="v1"), self._run(Nevus(), "val", run_id="v2")
        c = evaluate.compare(a, b)
        self.assertEqual((c["n"], c["both_right"], c["only_a"], c["only_b"], c["both_wrong"]), (4, 0, 2, 2, 0))
        self.assertEqual((c["mcnemar_p"], c["balanced_accuracy_difference"], c["accuracy_difference"]), (1.0, 0.0, 0.0))
        self.assertEqual((c["hallucination_difference"], c["hallucination_reduction"], c["hallucination_reduction_ci95"]),
                         (0.0, 0.0, (0.0, 0.0)))
        self.assertEqual((c["hallucinations_removed"], c["hallucinations_introduced"]), (2, 2))  # each fixes the other's wrong half
        low, high = c["balanced_accuracy_difference_ci95"]
        self.assertTrue(low <= 0 <= high)
        text = evaluate.comparison_text(c)
        for expected in ("| **Run B right** | 0 | 2 |", "McNemar's exact test: p = 1.0000", "+0.0 points", "paired bootstrap over lesions",
                         "Hallucination reduction, A minus B as a share of A: +0.0% (95% interval +0.0% to +0.0%)",
                         "Hallucinations removed (A named a wrong diagnosis, B is right): 2; introduced (A was right, B named a wrong diagnosis): 2"):
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
        self.assertIn("McNemar's exact test: p = 1.0000", shown.getvalue())

    def test_the_paired_interval_resamples_whole_lesions_of_both_runs_together(self):
        """Run B misses all 4 photos of lesion a1 and nothing else. Resampling the two lesions of class a (with replacement) gives
        B's recall of a as 0, 1/2 or 1 with chances 1/4, 1/2, 1/4, so the difference in balanced accuracy is -50, -25 or 0 points
        and the 95% interval is exactly -50 to 0. Resampling the 8 photos one by one would make losing all of them rare (under 1%)
        and give an interval of about -44 to -6 instead."""
        photos = [("a", "a1")] * 4 + [("a", "a2")] * 4 + [("b", "b1")] * 2 + [("b", "b2")] * 2
        runs = []
        for name, wrong in (("A", set()), ("B", {0, 1, 2, 3})):
            folder = self.base / f"run_{name}"
            folder.mkdir()
            card = {"status": "complete", "config": {"dataset": "image"}, "n_images": 12, "images_sha256": "same", "split": "test",
                    "labels": ["a", "b"]}
            (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
            rows = [{"path": f"test/{label}/{i}.jpg", "label": label, "lesion_group": group,
                     "prediction": ("b" if label == "a" else "a") if i in wrong else label} for i, (label, group) in enumerate(photos)]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs.append(folder)
        c = evaluate.compare(*runs)
        self.assertEqual((c["only_a"], c["only_b"], c["balanced_accuracy_difference"]), (4, 0, -0.25))
        self.assertEqual(c["balanced_accuracy_difference_ci95"], (-0.5, 0.0))
        self.assertEqual(c["mcnemar_p"], 0.125)  # 2 x (1/2)^4
        # B's balanced hallucination rate is 0.5, 0.25 or 0 (a1 twice, once, never in the resample) with chances 1/4, 1/2, 1/4; A's is 0
        self.assertEqual((c["hallucination_difference"], c["hallucination_difference_ci95"]), (0.25, (0.0, 0.5)))
        self.assertEqual((c["hallucination_reduction"], c["hallucination_reduction_ci95"]), (None, None))  # A has none to reduce
        self.assertEqual((c["hallucinations_removed"], c["hallucinations_introduced"]), (0, 4))
        self.assertIn("undefined (run A has no hallucination)", evaluate.comparison_text(c))
        back = evaluate.compare(runs[1], runs[0])  # the other way round: B has removed all 4 of A's hallucinations
        self.assertEqual((back["hallucination_reduction"], back["hallucinations_removed"], back["hallucinations_introduced"]), (1.0, 4, 0))
        self.assertIsNone(back["hallucination_reduction_ci95"])  # in some resamples A has none: no interval
        self.assertIn("+100.0% (no interval: run A has no hallucination in some resamples)", evaluate.comparison_text(back))

    def test_the_comparison_says_a_difference_is_beyond_chance_only_when_p_is_below_0_05(self):
        class Nevus(FakeModel):
            def predict(self, *args):
                return super().predict(*args) | {"text": "nevus"}

        c = evaluate.compare(self._run(FakeModel(), "val", run_id="v1"), self._run(Nevus(), "val", run_id="v2"))
        for p, beyond in ((0.0003, True), (0.0499, True), (0.05, False), (0.0534, False), (1.0, False)):
            with self.subTest(p):
                text = evaluate.comparison_text(c | {"mcnemar_p": p})
                self.assertEqual("Below 0.05: the two runs differ by more than chance." in text, beyond)
                self.assertEqual("Not below 0.05: the difference may be chance." in text, not beyond)

    def test_the_image_run_and_the_image_metadata_run_of_the_same_photos_can_be_compared_but_no_other_datasets(self):
        """The image + metadata questions are the image dataset's photos with their facts: the two runs differ in dataset and in
        the questions' fingerprint, yet answer the same photos, and `compare` must accept exactly that pair. A run card of the
        first phase names the dataset by its earlier name, "multimodal", and is read under the current one."""
        truth = ["a", "a", "b", "b"]
        runs = {}
        for name, dataset, predicted in (("image", "image", ["a", "b", "b", "a"]), ("multi", "image_metadata", ["a", "a", "b", "b"]),
                                         ("text", "text", ["a", "a", "b", "b"]), ("phase1", "multimodal", ["a", "a", "b", "b"])):
            folder = self.base / f"pair_{name}"
            folder.mkdir()
            kind = datasets.DATASETS[datasets.ALIASES.get(dataset, dataset)]
            card = {"status": "complete", "config": {"dataset": dataset}, "split": "test", "labels": ["a", "b"],
                    f"n_{kind.item}s": 4, f"{kind.item}s_sha256": f"fingerprint of {dataset}"}
            (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
            rows = [{"path": f"test/{t}/{i}.jpg", "label": t, kind.group_column: f"L{i}", "prediction": p}
                    for i, (t, p) in enumerate(zip(truth, predicted))]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs[name] = folder
        c = evaluate.compare(runs["image"], runs["multi"])
        self.assertEqual((c["dataset"], c["n"], c["only_a"], c["only_b"]), ("image and image_metadata", 4, 0, 2))  # the facts fixed 2
        self.assertEqual(c["balanced_accuracy_difference"], 0.5)
        self.assertIn("image and image_metadata dataset, test split, 4 questions", evaluate.comparison_text(c))
        with self.assertRaisesRegex(ValueError, "differ in their dataset"):
            evaluate.compare(runs["image"], runs["text"])
        with self.assertRaisesRegex(ValueError, "differ in their dataset"):
            evaluate.compare(runs["multi"], runs["text"])
        self.assertEqual(evaluate.compare(runs["image"], runs["phase1"])["dataset"], "image and image_metadata")
        rows = [json.loads(x) for x in (runs["multi"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["label"] = "b"  # another photo, or another class for the same photo
        (runs["multi"] / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same photos with the same classes"):
            evaluate.compare(runs["image"], runs["multi"])

    def test_removed_and_introduced_hallucinations_are_counted_by_hand(self):
        """Six questions of two classes. Run A: right, no valid answer, wrong, right, wrong, wrong. Run B: right, right, wrong,
        wrong, right, wrong. Hallucinated (a wrong diagnosis named): A on questions 2, 4, 5; B on 2, 3, 5. Removed (A hallucinated,
        B right): question 4 only; question 1 is no hallucination of A's (it named nothing). Introduced (A right, B hallucinated):
        question 3 only; questions 2 and 5 are hallucinations of both."""
        truth = ["a", "a", "a", "b", "b", "b"]
        answers = {"A": ["a", None, "b", "b", "a", "a"], "B": ["a", "a", "b", "a", "b", "a"]}
        runs = []
        for name, predicted in answers.items():
            folder = self.base / f"hand_{name}"
            folder.mkdir()
            card = {"status": "complete", "config": {"dataset": "image"}, "n_images": 6, "images_sha256": "same", "split": "test",
                    "labels": ["a", "b"]}
            (folder / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
            rows = [{"path": f"test/{t}/{i}.jpg", "label": t, "lesion_group": f"L{i}", "prediction": p}
                    for i, (t, p) in enumerate(zip(truth, predicted))]
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs.append(folder)
        c = evaluate.compare(*runs)
        self.assertEqual((c["hallucinations_removed"], c["hallucinations_introduced"]), (1, 1))
        # class a: A wrong 1 of 3 (question 2), B 1 of 3; class b: A wrong 2 of 3 (4, 5), B 2 of 3 (3, 5): the same rate, 0 difference
        self.assertAlmostEqual(c["hallucination_difference"], 0.0)
        self.assertAlmostEqual(c["hallucination_reduction"], 0.0)

    def test_the_hallucination_interval_is_its_own_not_the_accuracy_s(self):
        """Class b is never answered with a valid answer: a balanced accuracy of exactly 50%, but no hallucination at all."""
        m = metrics.summarize(["a", "a", "b", "b"], ["a", "a", None, None], ["a", "b"], groups=["L1", "L2", "L3", "L4"])
        self.assertEqual((m["balanced_accuracy_ci95"], m["hallucination"]["balanced_rate"], m["hallucination"]["balanced_rate_ci95"]),
                         ((0.5, 0.5), 0.0, (0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
