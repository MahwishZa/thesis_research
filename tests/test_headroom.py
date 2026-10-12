"""Headroom from the MedChange authors' released answers (no retrieval): scoring and the splits it covers (synthetic files)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import headroom
from experiments.medchange.headroom import score
from experiments.medchange.released_answers import FILES


class HeadroomTests(unittest.TestCase):

    def test_scoring_against_the_newest_and_the_outdated_verdict(self):
        items = [{"kind": "changed", "likely_label_noise": False,
                  "newest": {"row": 0, "label": "SUPPORTED"},
                  "previous": {"row": 1, "label": "REFUTED"}}]
        s = score(items, ["REFUTED", None])
        self.assertEqual(s["changed"]["newest_verdict_accuracy"], 0.0)
        self.assertEqual(s["changed"]["previous_verdict_match"], 1.0)


class HeadroomSplitTests(unittest.TestCase):
    def test_the_check_covers_the_main_benchmark_only_not_the_ad_and_fresh_questions_appended_to_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            folder = d / "Code" / "GeneratedAnswers"
            folder.mkdir(parents=True)
            (folder / FILES["qwen25-7b"]).write_text("SUPPORTED\nREFUTED\nSUPPORTED\nSUPPORTED\n", encoding="utf-8")
            rows = [{"item_id": f"X{k}", "kind": "unchanged", "split": split, "likely_label_noise": False,
                     "newest": {"row": k, "label": "SUPPORTED"}, "previous": {"label": "SUPPORTED"}}
                    for k, split in enumerate(("dev", "confirm", "ad", "fresh"))]
            bench = d / "benchmark.jsonl"
            bench.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")       # a blank line is tolerated
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                headroom.main(["--medchange-dir", str(d), "--benchmark", str(bench)])
            self.assertEqual(json.loads(out.getvalue())["qwen25-7b"]["unchanged"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
