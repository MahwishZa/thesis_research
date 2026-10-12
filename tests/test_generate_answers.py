"""Resumable answer generation for B0 and B1 and its command (a stub generator in place of the model; no network)."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import generate_answers as generate_cli
from experiments.medchange.files import load_jsonl
from experiments.medchange.generate_answers import run


CUTOFF = "2020-01-01"


def cand(pmid, rank, year, helpful=None, month="01"):
    c = {"pmid": pmid, "rank": rank, "lower": f"{year}-{month}-01", "upper": f"{year}-{month}-01",
         "title": f"T{pmid}", "abstract": f"abstract {pmid}", "pubtypes": [], "journal": "J"}
    if helpful is not None:
        c["helpful"] = helpful
    return c


def pool(n=10, with_helpful=True):
    # rank 1 is the OLDEST; helpfulness is highest for rank 3
    out = []
    for r in range(1, n + 1):
        out.append(cand(f"{r:03d}", r, 2000 + r * 2,
                        helpful=(0.9 if r == 3 else 0.1 + 0.01 * (n - r)) if with_helpful else None))
    return out


def item(iid, kind, newest, previous="REFUTED"):
    return {"item_id": iid, "kind": kind, "newest": {"label": newest, "date": CUTOFF},
            "previous": {"label": previous}, "question": f"Q {iid}"}


class RunTests(unittest.TestCase):

    def setUp(self):
        self.items = [item("a", "changed", "SUPPORTED"), item("b", "changed", "REFUTED")]
        self.pools = {i["item_id"]: {"candidates": pool()} for i in self.items}
        self.helpful = {}

    def test_resume_skips_finished_pairs_and_tolerates_truncated_last_line(self):
        calls = []
        def gen(system, user):
            calls.append(user)
            return "VERDICT: SUPPORTED\nok"
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ans.jsonl"
            n1 = run(self.items, self.pools, self.helpful, ["B0", "B1"], out, gen)
            self.assertEqual((n1, len(calls)), (4, 4))
            with open(out, "a") as f:
                f.write('{"item_id": "zz", "arm": "B0", "text": "trunc')    # killed mid-write
            n2 = run(self.items, self.pools, self.helpful, ["B0", "B1"], out, gen)
            self.assertEqual(n2, 0)
            self.assertEqual(len(calls), 4)
            rows = load_jsonl(out)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["arm"], "B0")           # items interleaved across arms
        self.assertEqual(rows[1]["arm"], "B1")
        self.assertEqual(rows[0]["admitted"], [])
        self.assertEqual(len(rows[1]["admitted"]), 5)

    def test_prompt_hash_differs_between_arms_that_admit_different_evidence(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.jsonl"
            run(self.items[:1], self.pools, self.helpful, ["B0", "B1"], out, lambda s, u: "x")
            r = load_jsonl(out)
        self.assertNotEqual(r[0]["prompt_sha256"], r[1]["prompt_sha256"])


class GenerateCliTests(unittest.TestCase):
    """``generate_answers.main`` end to end with a stub generator: the configuration is recorded
    before any answer is written, and a changed setting stops a resume before the model loads."""

    def _data_dir(self, tmp):
        d = Path(tmp) / "data"
        d.mkdir()
        bench = [dict(item("a", "changed", "SUPPORTED"), split="dev", likely_label_noise=False)]
        pools = [{"item_id": "a", "candidates": pool()}]
        for name, rows in (("benchmark.jsonl", bench), ("frozen_dev.jsonl", pools)):
            (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        model = Path(tmp) / "m.gguf"
        model.write_bytes(b"not a real model")
        return d, model

    def _main(self, tmp, model, *extra):
        loaded = []

        def fake_generator(*args):
            loaded.append(args)
            return lambda system, user: "VERDICT: SUPPORTED\nok"

        with mock.patch.object(generate_cli, "HERE", Path(tmp)), \
                mock.patch.object(generate_cli, "llama_generator", fake_generator), \
                mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            code = generate_cli.main(["--split", "dev", "--arms", "B0", "B1",
                                      "--model-path", str(model), *extra])
        return code, loaded

    def test_the_configuration_is_recorded_and_a_resume_with_the_same_settings_continues(self):
        with TemporaryDirectory() as tmp:
            d, model = self._data_dir(tmp)
            self.assertEqual(self._main(tmp, model)[0], 0)
            cfg = json.loads((d / "answers_dev.config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["model_sha256"], hashlib.sha256(b"not a real model").hexdigest())
            self.assertEqual(len(load_jsonl(d / "answers_dev.jsonl")), 2)
            self.assertEqual(self._main(tmp, model)[0], 0)
            self.assertEqual(len(load_jsonl(d / "answers_dev.jsonl")), 2)

    def test_a_resume_with_a_different_setting_or_model_is_refused_before_loading_the_model(self):
        with TemporaryDirectory() as tmp:
            d, model = self._data_dir(tmp)
            self._main(tmp, model)
            code, loaded = self._main(tmp, model, "--max-new-tokens", "320")
            self.assertEqual((code, loaded), (2, []))
            other = Path(tmp) / "other.gguf"
            other.write_bytes(b"a different model")
            code, loaded = self._main(tmp, other)
            self.assertEqual((code, loaded), (2, []))
            self.assertEqual(len(load_jsonl(d / "answers_dev.jsonl")), 2)

    def test_a_missing_model_file_is_reported_not_a_traceback(self):
        with TemporaryDirectory() as tmp:
            d, _ = self._data_dir(tmp)
            code, loaded = self._main(tmp, Path(tmp) / "absent.gguf")
            self.assertEqual((code, loaded), (2, []))
            self.assertFalse((d / "answers_dev.jsonl").exists())

    def test_answers_made_before_configurations_were_recorded_are_adopted_not_refused(self):
        with TemporaryDirectory() as tmp:
            d, model = self._data_dir(tmp)
            (d / "answers_dev.jsonl").write_text(
                json.dumps({"item_id": "a", "arm": "B0", "text": "t", "verdict": "SUPPORTED"}) + "\n",
                encoding="utf-8")
            self.assertEqual(self._main(tmp, model)[0], 0)
            self.assertTrue((d / "answers_dev.config.json").exists())
            self.assertEqual(len(load_jsonl(d / "answers_dev.jsonl")), 2)


if __name__ == "__main__":
    unittest.main()
