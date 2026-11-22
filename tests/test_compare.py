"""Checks of src/compare.py: every pair of runs gets McNemar's test, and the p-values are corrected with Holm's method."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import compare  # noqa: E402
import evaluate as ev  # noqa: E402
from build_views import question  # noqa: E402
from common import write_jsonl  # noqa: E402


def runs_of(right_lists):
    ids = [f"ISIC_{i:07d}" for i in range(len(next(iter(right_lists.values()))))]
    return {name: dict(zip(ids, right)) for name, right in right_lists.items()}


class CompareTests(unittest.TestCase):
    def test_every_pair_is_tested_and_corrected(self):
        rep = compare.compare(runs_of({"a": [True] * 10 + [False] * 2, "b": [False] * 10 + [True] * 2,
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
            compare.compare(runs)

    def test_the_command_reads_each_answer_with_its_own_questions(self):
        cases = [{"isic_id": f"ISIC_{i:07d}", "split": "test", "label": c, "group_id": f"ISIC_{i:07d}",
                  "clinical_text": "Patient: 50-year-old male.", "image_path": f"processed_images/ISIC_{i:07d}.jpg"}
                 for i, c in enumerate(["MEL", "NV", "BCC", "AK"])]
        with tempfile.TemporaryDirectory() as d:
            files = {}
            for view in ("text_only", "multimodal"):
                qs = [question(c, view) for c in cases]
                files[view] = os.path.join(d, f"{view}.jsonl")
                write_jsonl(files[view], qs)
                write_jsonl(os.path.join(d, f"{view}_pred.jsonl"),
                            [{"id": q["id"], "pred": q["answer"] if view == "multimodal" or k < 2 else "Z"} for k, q in enumerate(qs)])
            out = os.path.join(d, "cmp.json")
            with contextlib.redirect_stdout(io.StringIO()):
                compare.main(["--out", out,
                              "--run", f"text={files['text_only']},{os.path.join(d, 'text_only_pred.jsonl')}",
                              "--run", f"mm={files['multimodal']},{os.path.join(d, 'multimodal_pred.jsonl')}"])
            rep = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(rep["accuracy"], {"text": 0.5, "mm": 1.0})
        self.assertEqual(rep["pairs"][0]["a_wrong_b_right"], 2)
        self.assertIn("sha256", rep["inputs"]["text"]["questions"])


if __name__ == "__main__":
    unittest.main()
