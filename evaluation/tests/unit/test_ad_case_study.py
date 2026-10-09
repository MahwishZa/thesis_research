"""Case study of the Alzheimer's-named questions (no model, no network): selection rule and outputs."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import ad_case_study as C


def fixture(tmp: Path):
    items, answers = [], []
    for k in range(30):
        lab = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")[k % 3]
        items.append({"item_id": f"AD-{k:05d}", "split": "ad", "likely_label_noise": False, "kind": "unchanged",
                      "question": "Does X help in Alzheimer's disease?" if k < 25 else "Does X help in dementia?",
                      "newest": {"label": lab, "date": "2010-01-01", "cochrane_id": f"CD{k:06d}", "row": k}, "previous": {"label": lab}})
        for arm in C.ARMS:
            rec = {"item_id": f"AD-{k:05d}", "arm": arm, "verdict": lab if arm == "R2V" and k % 2 == 0 else "SUPPORTED", "text": "DIRECT STUDIES: NONE",
                   "admitted": ["1", "2"], "admitted_stratum": ["RCT", "other"]}
            if arm == "R2V":
                rec.update(draft_verdict="SUPPORTED", changed=k % 2 == 0 and lab != "SUPPORTED")
            answers.append(rec)
    (tmp / "benchmark.jsonl").write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    (tmp / "answers_ad.jsonl").write_text("\n".join(json.dumps(a) for a in answers if a["arm"] in ("B0", "B1")), encoding="utf-8")
    (tmp / "rag2_answers_ad.jsonl").write_text("\n".join(json.dumps(a) for a in answers if a["arm"] not in ("B0", "B1")), encoding="utf-8")


class CaseStudyTests(unittest.TestCase):
    def test_only_questions_naming_alzheimers_disease_are_used_in_a_fixed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fixture(d)
            items, rec = C.load(d, d)
            self.assertEqual(len(items), 25)
            a = [c["item_id"] for c in C.build(items, rec)]
            b = [c["item_id"] for c in C.build(dict(reversed(list(items.items()))), rec)]
            self.assertEqual(a, b, "the order depends on the seed and the id only")

    def test_fixed_and_broken_answers_are_named_and_a_failure_is_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fixture(d)
            cases = C.build(*C.load(d, d))
            kinds = {c["change"] for c in cases}
            self.assertIn("fixed", kinds)
            text = C.to_markdown(cases, n=3)
            self.assertIn("Nothing was chosen by outcome except those two", text)
            extra = sum(1 for k in ("fixed", "broke") if any(c["change"] == k for c in cases[:3]) is False and any(c["change"] == k for c in cases))
            self.assertEqual(sum(1 for line in text.splitlines() if line.startswith("## ")), 3 + extra)

    def test_main_writes_three_files_and_a_validation_sheet_of_twenty(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fixture(d)
            out = d / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main(["--data-dir", str(d), "--results-dir", str(d), "--out-dir", str(out)]), 0)
            rows = list(csv.DictReader(open(out / "validation_sheet.csv", encoding="utf-8-sig")))
            self.assertEqual(len(rows), 20)
            self.assertIn("clinician_verdict", rows[0])
            self.assertEqual(rows[0]["clinician_verdict"], "")
            self.assertTrue((out / "ad_cases_all.csv").is_file() and (out / "ad_case_study.md").is_file())


if __name__ == "__main__":
    unittest.main()
