"""data_roles.cli: check, export and overlap, end to end on small files."""

import io
import json
import unittest
from pathlib import Path

from data_roles import cli
from tests.support import TempDirTestCase
from tests.unit.data_roles.test_questions import medmcqa_row, medqa_row


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    status = cli.main(argv, streams=cli.Streams(out, err))
    return status, out.getvalue(), err.getvalue()


class CliTests(TempDirTestCase):
    @staticmethod
    def lines(path):
        return Path(path).read_text(encoding="utf-8").splitlines()

    def medqa_file(self, name="medqa.jsonl", rows=None):
        rows = rows if rows is not None else [medqa_row(), medqa_row(question="Second question?", answer_idx="A", answer="Ampicillin")]
        return str(self.write(name, "".join(json.dumps(row) + "\n" for row in rows)))

    def test_check_prints_the_summary_as_json(self):
        status, out, err = run(["check", "--dataset", "medqa", "--file", self.medqa_file(), "--split", "test"])
        self.assertEqual((status, err), (0, ""))
        summary = json.loads(out)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["answers"], {"A": 1, "B": 0, "C": 0, "D": 1})

    def test_export_writes_the_questions_the_answers_and_the_manifest(self):
        paths = {name: str(self.dir / name) for name in ("q.jsonl", "a.jsonl", "m.json")}
        status, out, err = run(
            ["export", "--dataset", "medqa", "--file", self.medqa_file(), "--split", "test",
             "--questions-out", paths["q.jsonl"], "--answers-out", paths["a.jsonl"], "--manifest-out", paths["m.json"]]
        )
        self.assertEqual((status, err, out), (0, "", "wrote 2 questions\n"))
        questions = [json.loads(line) for line in self.lines(paths["q.jsonl"])]
        self.assertEqual(questions[0]["id"], "medqa:test:0")
        self.assertTrue(questions[0]["question"].endswith("A) Ampicillin B) Ceftriaxone C) Doxycycline D) Nitrofurantoin"))
        answers = [json.loads(line) for line in self.lines(paths["a.jsonl"])]
        self.assertEqual([a["answer"] for a in answers], ["D", "A"])
        manifest = json.loads(Path(paths["m.json"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 2)

    def test_the_exported_questions_are_readable_by_part_one(self):
        from rationale_query.store import read_questions

        target = str(self.dir / "q.jsonl")
        run(["export", "--dataset", "medqa", "--file", self.medqa_file(), "--split", "test",
             "--questions-out", target, "--answers-out", str(self.dir / "a"), "--manifest-out", str(self.dir / "m")])
        loaded = read_questions(target)
        self.assertEqual([q.id for q in loaded], ["medqa:test:0", "medqa:test:1"])

    def test_export_does_not_overwrite(self):
        existing = self.write("q.jsonl", "keep")
        status, out, err = run(
            ["export", "--dataset", "medqa", "--file", self.medqa_file(), "--split", "test",
             "--questions-out", str(existing), "--answers-out", str(self.dir / "a"), "--manifest-out", str(self.dir / "m")]
        )
        self.assertEqual(status, 1)
        self.assertIn("does not overwrite", err)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.assertFalse((self.dir / "a").exists())

    def test_a_split_with_withheld_answers_exits_with_3(self):
        rows = [medmcqa_row(id=f"i{n}", cop=-1) for n in range(2)]
        path = self.write("test.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        status, out, err = run(["check", "--dataset", "medmcqa", "--file", str(path), "--split", "test"])
        self.assertEqual(status, 3)
        self.assertIn("withheld", err)

    def test_a_bad_row_exits_with_1_and_names_it(self):
        path = self.medqa_file(rows=[medqa_row(answer_idx="Q")])
        status, out, err = run(["check", "--dataset", "medqa", "--file", path, "--split", "test"])
        self.assertEqual(status, 1)
        self.assertIn("row 0", err)

    def test_exactly_one_source_is_needed(self):
        status, out, err = run(["check", "--dataset", "medqa", "--split", "test"])
        self.assertEqual(status, 1)
        self.assertIn("exactly one", err)
        status, out, err = run(["check", "--dataset", "medqa", "--file", self.medqa_file(), "--hub", "x/y", "--split", "test"])
        self.assertEqual(status, 1)

    def test_missing_arguments_exit_with_2(self):
        self.assertEqual(run(["check"])[0], 2)
        self.assertEqual(run([])[0], 2)

    def test_overlap_counts_the_candidates_that_repeat_a_reference_question(self):
        shared = "A 58 year old man presents with five days of productive cough and fever and a chest radiograph shows an infiltrate"
        reference = self.write("train.jsonl", "".join(json.dumps(medmcqa_row(id=f"t{i}", question=q)) + "\n" for i, q in enumerate([shared, "Another question entirely about something else here today"])))
        candidate = self.write("valid.jsonl", "".join(json.dumps(medmcqa_row(id=f"v{i}", question=q)) + "\n" for i, q in enumerate([shared, "A new unrelated question about the heart and its valves today"])))
        status, out, err = run(
            ["overlap", "--dataset", "medmcqa", "--reference-file", str(reference), "--reference-split", "train",
             "--candidate-file", str(candidate), "--candidate-split", "validation"]
        )
        self.assertEqual((status, err), (0, ""))
        self.assertIn("1 of 2 candidate questions repeat a reference question (1 pairs are exact)", out)
        self.assertIn("v0 ~ t0", out)


if __name__ == "__main__":
    unittest.main()
