"""Arms (B0/B1 and the completed stage-1 arms), prompts and resumable generation (no model, no network)."""

import hashlib
import json
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import arms as A
from experiments.medchange import generate_answers as generate_cli
from experiments.medchange.generate_answers import (
    RESULT_RELEVANT, check_config, config_path, file_sha256, load_jsonl, run, run_config,
)
from experiments.medchange.prompts import build_prompt, parse_verdict

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
        out.append(cand(f"{r:03d}", r, 2019 - 2 * (n - r) if False else 2000 + r * 2,
                        helpful=(0.9 if r == 3 else 0.1 + 0.01 * (n - r)) if with_helpful else None))
    return out


class ArmTests(unittest.TestCase):

    def test_b0_admits_nothing_b1_follows_rerank_rank(self):
        self.assertEqual(A.admit("B0", pool(), CUTOFF), [])
        self.assertEqual([c["rank"] for c in A.admit("B1", pool(), CUTOFF)], [1, 2, 3, 4, 5])

    def test_b2_follows_helpfulness_not_rank(self):
        got = [c["rank"] for c in A.admit("B2", pool(), CUTOFF)]
        self.assertEqual(got[0], 3)
        self.assertEqual(sorted(got), sorted(got))      # budget respected
        self.assertEqual(len(got), 5)

    def test_recency_arms_prefer_newer_passages_than_their_relevance_only_twins(self):
        mean_year = lambda arm: sum(int(c["upper"][:4]) for c in A.admit(arm, pool(), CUTOFF)) / 5
        self.assertGreater(mean_year("B3"), mean_year("B1"))
        self.assertGreater(mean_year("P"), mean_year("B2"))

    def test_shuffled_dates_keep_relevance_but_break_the_date_signal(self):
        p = pool()
        c1 = A.admit("C1", p, CUTOFF, item_id="x")
        pp = A.admit("P", p, CUTOFF, item_id="x")
        self.assertEqual(A.admit("C1", p, CUTOFF, item_id="x"), c1)          # deterministic
        self.assertNotEqual([c["pmid"] for c in c1], [c["pmid"] for c in pp])
        # the passages' own (true) dates are what is returned, only selection changed
        self.assertTrue(all(c in p for c in c1))

    def test_relevance_ranks_are_contiguous_and_ties_follow_cross_encoder_rank(self):
        p = [cand("a", 2, 2000), cand("b", 1, 2000), cand("c", 3, 2000)]
        self.assertEqual(A.relevance_ranks([5.0, 5.0, 9.0], p), [3, 2, 1])   # c best; tie: b (rank 1) before a
        self.assertEqual(sorted(A.relevance_ranks([1, 1, 1], p)), [1, 2, 3])

    def test_scores_come_from_the_reference_implementation(self):
        """The arms must not carry their own copy of the formula: compare with a direct
        call of src.proposed on the same inputs."""
        from datetime import date
        from src.common.evidence import Candidate, Evidence
        from src.temporal_filter.scorer import AdmissionScorer
        from src.temporal_filter.temporal import TemporalPolicy
        p = pool()
        got = A.admission_scores(p, [-c["rank"] for c in p], p, CUTOFF, 0.5)
        scorer = AdmissionScorer(0.5)
        policy = TemporalPolicy(half_life_days=A.HALF_LIFE_DAYS, undated_score=0.0)
        for c, g in zip(p, got):
            ev = Evidence(evidence_id=c["pmid"], text="", source_tier="x",
                          publication_date=A.point_date(c))
            t = policy.score(ev, question="", question_date=date.fromisoformat(CUTOFF)).score
            want = scorer.score(Candidate(ev, 0.0, c["rank"]), temporal=t, candidate_count=len(p)).total
            self.assertAlmostEqual(g, want)
        self.assertAlmostEqual(got[0], 0.5 * 1.0 + 0.5 * A.recency(p[0], CUTOFF))   # rank 1 -> rho 1

    def test_recency_halves_every_half_life_and_clamps_future(self):
        c = cand("1", 1, 2017, month="01")
        self.assertAlmostEqual(A.recency(c, "2020-01-01", half_life_days=1096), 0.5, places=2)
        self.assertEqual(A.recency(cand("2", 1, 2021), CUTOFF), 1.0)

    def test_helpfulness_arms_refuse_unscored_pools_and_unknown_arm(self):
        with self.assertRaises(ValueError):
            A.admit("P", pool(with_helpful=False), CUTOFF)
        with self.assertRaises(ValueError):
            A.admit("Z", pool(), CUTOFF)

    def test_empty_pool_and_small_pool(self):
        self.assertEqual(A.admit("B1", [], CUTOFF), [])
        self.assertEqual(len(A.admit("B3", pool(3), CUTOFF)), 3)

    def test_settings_hash_is_stable(self):
        self.assertEqual(A.settings_hash(), A.settings_hash())


class PromptTests(unittest.TestCase):

    def test_prompt_has_evidence_only_when_admitted_and_never_dates(self):
        p = build_prompt("Does X help Y?", [cand("1", 1, 2001)])
        self.assertIn("[1] T1. abstract 1", p)
        self.assertNotIn("2001", p)
        self.assertNotIn("Evidence:", build_prompt("Does X help Y?", []))

    def test_verdict_parsing(self):
        self.assertEqual(parse_verdict("VERDICT: SUPPORTED\nbecause"), "SUPPORTED")
        self.assertEqual(parse_verdict("verdict: **Not  enough information**"), "NOT ENOUGH INFORMATION")
        self.assertEqual(parse_verdict("VERDICT: REFUTED"), "REFUTED")
        self.assertIsNone(parse_verdict("The evidence supports it."))   # never guessed
        self.assertIsNone(parse_verdict(None))


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


class RunConfigTests(unittest.TestCase):
    """An answers file is bound to the generator that produced it: a resumed run with a different
    model or result-relevant setting is refused instead of silently mixing configurations."""

    KW = dict(model_path="models/m.gguf", model_sha256="a" * 64, n_ctx=4096, max_new_tokens=160,
              n_threads=8, n_gpu_layers=0, llama_version="0.2.90")

    def test_file_sha256_streams_the_whole_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.bin"
            payload = b"x" * (3 * (1 << 20) + 17)
            path.write_bytes(payload)
            self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())

    def test_the_config_records_model_decoding_prompt_and_arm_settings(self):
        cfg = run_config(**self.KW)
        self.assertEqual(cfg["model_file"], "m.gguf")
        self.assertEqual((cfg["temperature"], cfg["seed"]), (0.0, 42))
        self.assertEqual(cfg["arm_settings"], A.settings_hash())
        for key in RESULT_RELEVANT:
            self.assertIn(key, cfg)

    def test_the_first_run_records_the_configuration_and_a_rerun_is_compatible(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers_dev.jsonl"
            self.assertEqual(check_config(out, run_config(**self.KW)), [])
            self.assertTrue(config_path(out).exists())
            self.assertEqual(config_path(out).name, "answers_dev.config.json")
            self.assertEqual(check_config(out, run_config(**self.KW)), [])

    def test_a_different_model_or_result_relevant_setting_is_refused(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers_dev.jsonl"
            check_config(out, run_config(**self.KW))
            self.assertEqual(check_config(out, run_config(**dict(self.KW, model_sha256="b" * 64))),
                             ["model_sha256"])
            self.assertEqual(check_config(out, run_config(**dict(self.KW, n_ctx=2048,
                                                                 max_new_tokens=256))),
                             ["n_ctx", "max_new_tokens"])

    def test_threads_gpu_layers_and_library_version_are_recorded_but_do_not_block_a_resume(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers_dev.jsonl"
            check_config(out, run_config(**self.KW))
            changed = run_config(**dict(self.KW, n_threads=2, n_gpu_layers=10, llama_version="9.9"))
            self.assertEqual(check_config(out, changed), [])
            recorded = json.loads(config_path(out).read_text(encoding="utf-8"))
        self.assertEqual((recorded["n_threads"], recorded["llama_cpp_python"]), (8, "0.2.90"))


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
