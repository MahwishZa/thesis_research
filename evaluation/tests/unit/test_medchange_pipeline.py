"""The stage-2 pipeline: integrity checks, the frozen-model guard and the plans (nothing is executed)."""

import contextlib
import io
import json
import subprocess
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import findings as F
from experiments.medchange import pipeline as P


def write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def bench(n_dev=2, n_confirm=3):
    rows = []
    for k in range(n_dev + n_confirm):
        rows.append({"item_id": f"MC-{k}", "split": "dev" if k < n_dev else "confirm", "likely_label_noise": False,
                     "newest": {"pmid": f"R{k}", "label": "SUPPORTED"}, "previous": {"pmid": f"O{k}", "label": "REFUTED"}})
    return rows


def pool(item_id, n=8, pmids=None):
    return {"item_id": item_id, "candidates": [{"pmid": (pmids or [f"{item_id}-{i}" for i in range(n)])[i], "rank": i + 1}
                                               for i in range(n)]}


class PreflightTests(unittest.TestCase):

    def setUp(self):
        self.expected = dict(P.EXPECTED_ITEMS)
        P.EXPECTED_ITEMS.update(dev=2, confirm=3)

    def tearDown(self):
        P.EXPECTED_ITEMS.update(self.expected)

    def _data(self, tmp, **kw):
        d = Path(tmp)
        write(d / "benchmark.jsonl", bench())
        write(d / "frozen_dev.jsonl", [pool("MC-0"), pool("MC-1", **kw)])
        return d

    def test_clean_data_passes_every_check(self):
        with TemporaryDirectory() as tmp:
            rows = P.preflight(self._data(tmp), "dev")
        self.assertTrue(all(ok for _, ok, _ in rows), rows)

    def test_each_integrity_failure_is_caught(self):
        with TemporaryDirectory() as tmp:
            d = self._data(tmp, n=5)                                          # a short pool
            failed = {name for name, ok, _ in P.preflight(d, "dev") if not ok}
            self.assertIn("every pool has at least 8 candidates", failed)
            relaxed = {name for name, ok, _ in P.preflight(d, "dev", P.ANSWER_BUDGET) if not ok}
            self.assertEqual(relaxed, set())                                  # 5 candidates suffice without stance
            write(d / "frozen_dev.jsonl", [pool("MC-0"), pool("MC-1", pmids=["R1"] + [f"x{i}" for i in range(7)])])
            self.assertIn("no pool contains its own review", {n for n, ok, _ in P.preflight(d, "dev") if not ok})
            write(d / "frozen_dev.jsonl", [pool("MC-0")])
            self.assertIn("every item has a frozen pool", {n for n, ok, _ in P.preflight(d, "dev") if not ok})
            rows = bench()
            rows[2]["item_id"] = "MC-0"                                       # an id in both splits
            write(d / "benchmark.jsonl", rows)
            self.assertIn("dev and confirm share no item", {n for n, ok, _ in P.preflight(d, "dev") if not ok})

    def test_a_low_parse_rate_fails(self):
        with TemporaryDirectory() as tmp:
            d = self._data(tmp)
            write(d / "answers_dev.jsonl", [{"item_id": "MC-0", "arm": "B1", "verdict": None}] * 3
                  + [{"item_id": "MC-1", "arm": "B1", "verdict": "SUPPORTED"}])
            self.assertIn("answers parse rate >= 95%", {n for n, ok, _ in P.preflight(d, "dev") if not ok})


class ConfigTests(unittest.TestCase):

    def test_config_differences_lists_only_result_relevant_fields(self):
        a = {"model_sha256": "x", "n_ctx": 4096, "n_threads": 4}
        self.assertEqual(P.config_differences(a, {"model_sha256": "x", "n_ctx": 4096, "n_threads": 8}), [])
        self.assertEqual(P.config_differences(a, {"model_sha256": "y", "n_ctx": 2048}), ["model_sha256", "n_ctx"])

    def test_generation_precheck_compares_with_the_recorded_dev_run(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"weights")
            cfg = Path(tmp) / "answers_dev.config.json"
            self.assertEqual(P.generation_precheck(str(model), cfg), ["no recorded dev generator configuration"])
            expected = P.run_config(model_path=str(model), model_sha256=P.file_sha256(str(model)), n_ctx=4096,
                                    max_new_tokens=P.MAX_NEW_TOKENS, n_threads=None, n_gpu_layers=0, llama_version="")
            cfg.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(P.generation_precheck(str(model), cfg), [])
            expected["model_sha256"] = "other"
            cfg.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(P.generation_precheck(str(model), cfg), ["model_sha256"])

    def test_stance_model_precheck(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"w")
            cfg = Path(tmp) / "stance_dev.config.json"
            cfg.write_text(json.dumps({"model_sha256": P.file_sha256(str(model))}), encoding="utf-8")
            self.assertEqual(P.stance_model_precheck(str(model), cfg), [])
            cfg.write_text(json.dumps({"model_sha256": "zzz"}), encoding="utf-8")
            self.assertEqual(P.stance_model_precheck(str(model), cfg), ["model_sha256"])


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          capture_output=True, text=True, check=True)


class NetworkRetryTests(unittest.TestCase):
    """A connection cut in the middle of a download (IncompleteRead) is retried, not fatal."""

    def test_efetch_retries_an_incomplete_read(self):
        import http.client
        from experiments.medchange import freeze_candidates as FC
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"partial")
            return b"<PubmedArticleSet></PubmedArticleSet>"

        eu = types.SimpleNamespace(api_key=None, delay=0, _sleep=lambda s: None, _open=opener)
        self.assertEqual(FC.fetch_abstracts(eu, ["1"]), {})
        self.assertEqual(len(calls), 2)


class FrozenGuardTests(unittest.TestCase):

    def test_the_frozen_model_must_match_origin_main(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-q")
            rel = "results/synthesis_model.json"
            ok, why = P.frozen_is_pushed(repo, rel)
            self.assertFalse(ok)
            (repo / "results").mkdir()
            (repo / rel).write_text("{}", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, "commit", "-q", "-m", "x")
            ok, why = P.frozen_is_pushed(repo, rel)
            self.assertFalse(ok)
            self.assertIn("not on origin/main", why)                           # committed but not pushed
            git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
            self.assertTrue(P.frozen_is_pushed(repo, rel)[0])
            (repo / rel).write_text('{"changed": 1}', encoding="utf-8")
            ok, why = P.frozen_is_pushed(repo, rel)
            self.assertFalse(ok)
            self.assertIn("differs", why)


def args(**kw):
    base = dict(model_path="llama.gguf", judge_path="qwen.gguf", medchange_dir="mc", n_threads="6", commit=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


def names(steps):
    return [n for n, _ in steps]


class PlanTests(unittest.TestCase):

    def test_dev_plan_has_the_audits_only_when_a_judge_is_given_and_stops_after_the_report(self):
        full = P.dev_plan(args(), Path("d"), Path("r"))
        self.assertEqual(names(full)[:3], ["preflight (dev)", "label audit (dev)", "consistency check (dev)"])
        self.assertEqual(names(full)[-3:], ["dev report", "tables and figures (dev)", "commit and push"])
        self.assertFalse(any("confirm" in n for n in names(full)))
        lean = P.dev_plan(args(judge_path=None, commit=False), Path("d"), Path("r"))
        self.assertNotIn("label audit (dev)", names(lean))
        self.assertEqual(names(lean)[-1], "tables and figures (dev)")
        stance = dict(full)["stance, both wordings (dev)"]
        self.assertIn("both", stance)

    def test_confirm_plan_with_gate_2_pass_runs_stance_and_prediction_after_the_answers(self):
        n = names(P.confirm_plan(args(), Path("d"), Path("r"), "PASS"))
        self.assertLess(n.index("B0 and B1 answers (confirm)"), n.index("stance, both wordings (confirm)"))
        self.assertLess(n.index("frozen-model prediction"), n.index("label audit (confirm, after freeze)"))
        self.assertLess(n.index("label audit (confirm, after freeze)"), n.index("analysis"))
        self.assertEqual(n[-3:], ["findings", "tables and figures (confirm)", "commit and push"])

    def test_confirm_plan_with_gate_2_fail_is_rq1_only(self):
        steps = P.confirm_plan(args(commit=False), Path("d"), Path("r"), "FAIL")
        n = names(steps)
        self.assertNotIn("stance, both wordings (confirm)", n)
        self.assertNotIn("frozen-model prediction", n)
        self.assertIn("--rq1-only", dict(steps)["analysis"])
        self.assertNotIn("--rq1-only", dict(P.confirm_plan(args(), Path("d"), Path("r"), "PASS"))["analysis"])

    def test_execute_stops_at_the_first_failure_and_dry_run_runs_nothing(self):
        ran = []
        steps = [("a", lambda: (ran.append("a") or True, "ok")), ("b", lambda: (False, "bad")),
                 ("c", lambda: (ran.append("c") or True, "never"))]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(P.execute(steps, False), 2)
            self.assertEqual(ran, ["a"])
            self.assertEqual(P.execute(steps, True), 0)
        self.assertEqual(ran, ["a"])


class MainTests(unittest.TestCase):

    def run_main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return P.main(argv)

    def test_confirm_refuses_without_go_and_without_a_pushed_frozen_model(self):
        with TemporaryDirectory() as tmp:
            base = ["confirm", "--model-path", "x.gguf", "--data-dir", tmp, "--results-dir", tmp]
            self.assertEqual(self.run_main(base), 2)                                      # no --go
            with mock.patch.object(P, "frozen_is_pushed", return_value=(False, "no")):
                self.assertEqual(self.run_main(base + ["--go"]), 2)

    def test_dev_refuses_when_gate_1_is_not_pass(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(self.run_main(["dev", "--model-path", __file__, "--data-dir", tmp, "--results-dir", tmp]), 2)

    def test_a_confirmatory_generator_that_differs_from_dev_is_refused(self):
        with TemporaryDirectory() as tmp:
            res = Path(tmp)
            (res / "synthesis_model.json").write_text(json.dumps({"gate2": {"gate2": "PASS"}}), encoding="utf-8")
            model = res / "m.gguf"
            model.write_bytes(b"w")
            with mock.patch.object(P, "frozen_is_pushed", return_value=(True, "ok")):
                code = self.run_main(["confirm", "--go", "--model-path", str(model), "--data-dir", tmp,
                                      "--results-dir", tmp])
        self.assertEqual(code, 2)                                        # no recorded dev configuration

    def test_dry_run_prints_the_plan_without_executing(self):
        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = P.main(["confirm", "--go", "--dry-run", "--model-path", "m.gguf", "--data-dir", tmp,
                               "--results-dir", tmp])
        self.assertEqual(code, 0)
        self.assertIn("analyze_stage2", out.getvalue())


class FindingsTests(unittest.TestCase):

    def test_dev_report_states_the_gate_and_what_happens_next(self):
        with TemporaryDirectory() as tmp:
            res = Path(tmp)
            model = {"selected": "H0", "stance_direction_auc": 0.64,
                     "gate2": {"gate2": "FAIL", "checks": {"stance_direction_auc>=0.6": True, "x": False}},
                     "cv": {"prior_accuracy": 0.46, "variants": {"B1R": {"accuracy": 0.52}, "H0": {"accuracy": 0.53}}}}
            (res / "synthesis_model.json").write_text(json.dumps(model), encoding="utf-8")
            text, gate2 = F.dev_report(res, {"gate1": "PASS", "checks": {"wording_agreement>=0.80": True}})
        self.assertEqual(gate2, "FAIL")
        for needle in ("Gate 1", "Gate 2", "RQ1 only", "H0", "0.640"):
            self.assertIn(needle, text)

    def test_findings_picks_the_pre_written_sentence_for_the_tier(self):
        rep = {"primary_arm": "H0", "reading": {"RQ1": "confirmed", "RQ1_note": "mostly abstention",
                                                "RQ2_tier": "not confirmed", "criteria": {"RQ2 confirmed against B1R": False}},
               "primary_family": {"RQ1 B1 vs B0": {"diff_a_minus_b": 0.06, "ci95": [0.02, 0.1], "holm_p": 0.01},
                                  "RQ2 H0 vs B1R": {"diff_a_minus_b": 0.0, "ci95": [-0.03, 0.03], "holm_p": 1.0}}}
        with TemporaryDirectory() as tmp:
            res = Path(tmp)
            self.assertIn("has not been run", F.findings(res))
            (res / "stage2_analysis_confirm.json").write_text(json.dumps(rep), encoding="utf-8")
            F.write_findings(res)
            text = (res / "FINDINGS.md").read_text(encoding="utf-8")
        self.assertIn("RQ2 was not confirmed", text)
        self.assertIn("mostly abstention", text)


if __name__ == "__main__":
    unittest.main()
