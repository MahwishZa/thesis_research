"""Automatic label audit (synthetic data; a fake in place of the model)."""

import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import label_audit as LA


S, R, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"


def item(i, kind, new, prev, row_new, row_prev):
    return {"item_id": i, "kind": kind, "split": "dev", "likely_label_noise": False, "question": f"Q {i}?",
            "newest": {"label": new, "row": row_new, "date": "2020-01-01"},
            "previous": {"label": prev, "row": row_prev, "date": "2010-01-01"}}


MEDREV = {1: {"objectives": "obj one", "conclusions": "conc one"}, 2: {"objectives": "o2", "conclusions": "c2"},
          3: {"objectives": "o3", "conclusions": "c3"}, 4: {"objectives": "o4", "conclusions": "c4"}}


class LabelAuditTests(unittest.TestCase):

    def test_the_item_level_file_is_copied_to_the_results_folder_once_and_never_onto_itself(self):
        with TemporaryDirectory() as tmp:
            data, results = Path(tmp) / "data", Path(tmp) / "results"
            data.mkdir()
            rows = data / "label_audit_fresh.jsonl"
            rows.write_text(json.dumps({"item_id": "FR-1", "which": "newest", "gold": S, "relabel": S}) + "\n", encoding="utf-8")
            copy = LA.publish_rows(rows, results)
            self.assertEqual(copy, results / "label_audit_fresh.jsonl")
            self.assertEqual(copy.read_text(encoding="utf-8"), rows.read_text(encoding="utf-8"))
            self.assertIsNone(LA.publish_rows(rows, data))                         # same folder: nothing to copy

    def test_prompt_carries_the_authors_rubric_and_the_text(self):
        p = LA.build_prompt("Does X help?", "obj", "conc")
        for needle in ("only select the third label if not enough studies were found", "at least partially supported",
                       "Does X help?", "obj", "conc", "LABEL:"):
            self.assertIn(needle, p)

    def test_kappa(self):
        self.assertEqual(LA.cohens_kappa([(S, S), (R, R), (N, N)]), 1.0)
        self.assertEqual(LA.cohens_kappa([(S, R), (R, S)]), -1.0)
        self.assertIsNone(LA.cohens_kappa([]))

    def test_run_is_resumable_and_summary_finds_stable_items_and_reproduced_changes(self):
        items = [item("a", "changed", S, N, 1, 2), item("b", "unchanged", R, R, 3, 4)]
        answers = {"obj one": "LABEL: SUPPORTED", "o2": "LABEL: NOT ENOUGH INFORMATION", "o3": "LABEL: SUPPORTED"}
        calls = []

        def fake(system, user):
            calls.append(user)
            for k, v in answers.items():
                if f"OBJECTIVES: {k}" in user:
                    return v
            return "no idea"

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit.jsonl"
            self.assertEqual(LA.run(items, MEDREV, fake, out), 3)          # a newest, a previous, b newest
            self.assertEqual(LA.run(items, MEDREV, fake, out), 0)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        rep = LA.summarize(rows, {i["item_id"]: i for i in items})
        self.assertEqual(rep["n_items"], 2)
        self.assertAlmostEqual(rep["agreement"], 0.5)                       # a agrees, b (REFUTED vs SUPPORTED) not
        self.assertEqual(rep["stable_item_ids"], ["a"])
        self.assertEqual(rep["label_change_reproduced"], 1.0)
        self.assertEqual(rep["per_gold_class"][R]["agreement"], 0.0)
        self.assertEqual(len(calls), 3)

    def test_unparsed_relabels_are_counted_not_guessed(self):
        items = [item("a", "unchanged", S, S, 1, 2)]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.jsonl"
            LA.run(items, MEDREV, lambda s, u: "hmm", out)
            rep = LA.summarize([json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()], {"a": items[0]})
        self.assertEqual((rep["n_unparsed"], rep["agreement"], rep["stable_item_ids"]), (1, None, []))

    def test_cli_refuses_the_confirmatory_split_before_the_design_is_frozen(self):
        with TemporaryDirectory() as tmp, mock.patch("sys.stderr"):
            code = LA.main(["--split", "confirm", "--medchange-dir", tmp, "--model-path", str(Path(tmp) / "m.gguf"),
                            "--design-record", str(Path(tmp) / "absent.json")])
        self.assertEqual(code, 2)

    def test_cli_writes_reports_with_a_fake_model(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, res, ds = tmp / "data", tmp / "results", tmp / "mc" / "Datasets"
            data.mkdir(); ds.mkdir(parents=True)
            (data / "benchmark.jsonl").write_text(json.dumps(item("a", "unchanged", S, S, 1, 2)) + "\n", encoding="utf-8")
            with open(ds / "MedRevQA.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["", "objectives", "conclusions"])
                w.writeheader()
                for k, v in MEDREV.items():
                    w.writerow({"": k, **v})
            model = tmp / "m.gguf"
            model.write_bytes(b"x")
            with mock.patch.object(LA, "llama_generator", lambda *a, **k: (lambda s, u: "LABEL: SUPPORTED")), \
                    mock.patch("sys.stdout"):
                code = LA.main(["--split", "dev", "--medchange-dir", str(tmp / "mc"), "--model-path", str(model),
                                "--data-dir", str(data), "--out-dir", str(res)])
            rep = json.loads((res / "label_audit_dev.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(rep["stable_item_ids"], ["a"])


class AuditSampleTests(unittest.TestCase):
    def test_the_registered_audit_sample_of_the_committed_question_list_is_these_300_questions(self):
        """The rule of protocol §10.8 (SHA-256 order of ``fresh-audit-v1|<item id>``) applied to the committed manifest; a change of the rule or of the
        list changes the sample and fails here."""
        manifest = json.loads((Path(LA.__file__).resolve().parent / "manifest_fresh.json").read_text(encoding="utf-8"))
        items = [{"item_id": f"FR-{row:05d}"} for row in manifest["rows"]]
        sample = [x["item_id"] for x in LA.audit_sample(items, 300)]
        self.assertEqual(sample[:5], ["FR-00725", "FR-03135", "FR-07991", "FR-03991", "FR-02782"])
        self.assertEqual(hashlib.sha256(json.dumps(sorted(sample)).encode()).hexdigest()[:16], "0dd10528022c2eae")
        self.assertEqual(len(set(sample)), 300)

    def test_the_audit_sample_is_the_same_items_every_time(self):
        items = [{"item_id": f"FR-{i:05d}"} for i in range(100)]
        a, b = LA.audit_sample(items, 20), LA.audit_sample(list(reversed(items)), 20)
        self.assertEqual([x["item_id"] for x in a], [x["item_id"] for x in b])
        self.assertEqual(len(a), 20)


if __name__ == "__main__":
    unittest.main()
