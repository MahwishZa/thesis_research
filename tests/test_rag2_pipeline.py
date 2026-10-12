"""The phase drivers of the realigned study: preflight, dev check, reports, plans, guards, the fresh phase, the recorded environment and the committed configurations."""

import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import analyze_rag2 as AR
from experiments.medchange import pipeline_support as P
from experiments.medchange import rag2 as R
from experiments.medchange import rag2_pipeline as RP
from experiments.medchange import rag2_run as RR
from experiments.medchange.encoders import HashingEncoder, LexicalOverlapReranker
from tests.fixtures import F_, N, S, fake_generate, item, lists_for, records, write


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(P.EXPECTED_ITEMS)
        P.EXPECTED_ITEMS.update(dev=2, confirm=1)

    def tearDown(self):
        P.EXPECTED_ITEMS.update(self.expected)

    def _data(self, tmp, b1=True):
        d = Path(tmp)
        rows = [item("MC-1"), item("MC-2"), item("MC-3", split="confirm")]
        write(d / "benchmark.jsonl", rows)
        (d / "pubmed_g0").mkdir()
        for r in rows:
            (d / "pubmed_g0" / f"{r['item_id']}.json").write_text("{}", encoding="utf-8")
        write(d / "abstracts.jsonl", [])
        arms = ("B0", "B1") if b1 else ("B0",)
        write(d / "answers_dev.jsonl", [{"item_id": i, "arm": a} for i in ("MC-1", "MC-2") for a in arms])
        return d

    def test_preflight(self):
        with TemporaryDirectory() as tmp:
            self.assertTrue(all(ok for _, ok, _ in RP.preflight(self._data(tmp), "dev")))
        with TemporaryDirectory() as tmp:
            failed = {n for n, ok, _ in RP.preflight(self._data(tmp, b1=False), "dev") if not ok}
            self.assertEqual(failed, {"B0 and B1 answers exist for every item (reused)"})

    def _rep(self, r2=0.50, r2v=0.52, b1=0.53, unparsed=0, valid=1.0):
        gen = {a: {"all": {"n": 100, "accuracy": acc}, "unparsed": unparsed}
               for a, acc in (("B1", b1), ("R2", r2), ("R2V", r2v))}
        return {"generation": gen, "verification": {"R2V": {"valid_rate": valid}},
                "primary": {"diff_a_minus_b": r2v - r2, "ci95": [-0.03, 0.07], "mcnemar_p": 0.4, "confirmed": False},
                "secondary": {}, "reading": "development estimate only (exploratory)"}

    def test_dev_check(self):
        self.assertEqual(RP.dev_check(self._rep())["status"], "READY")
        self.assertEqual(RP.dev_check(self._rep(r2v=0.48))["status"], "REVISE ONCE")
        self.assertEqual(RP.dev_check(self._rep(r2=0.47, r2v=0.49))["status"], "DEFECT")      # R2 < B1 - 5 pp
        self.assertEqual(RP.dev_check(self._rep(unparsed=10))["status"], "DEFECT")
        self.assertEqual(RP.dev_check(self._rep(valid=0.9))["status"], "DEFECT")

    def test_reports(self):
        with TemporaryDirectory() as tmp:
            res = Path(tmp)
            self.assertFalse(RP.write_dev_report(res)[0])
            (res / "rag2_analysis_dev.json").write_text(json.dumps(self._rep()), encoding="utf-8")
            ok, msg = RP.write_dev_report(res)
            self.assertTrue(ok)
            self.assertIn("READY", msg)
            self.assertIn("Pre-declared dev check", (res / "RAG2_DEV_REPORT.md").read_text(encoding="utf-8"))
            (res / "rag2_analysis_confirm.json").write_text(json.dumps(self._rep()), encoding="utf-8")
            self.assertTrue(RP.write_findings(res)[0])
            self.assertIn("Reading:", (res / "RAG2_FINDINGS_CONFIRM.md").read_text(encoding="utf-8"))

    def test_publish_strips_publisher_text(self):
        with TemporaryDirectory() as tmp:
            d, res = Path(tmp) / "data", Path(tmp) / "results"
            d.mkdir()
            write(d / "rag2_lists_dev.jsonl", [{"item_id": "MC-1", "lists": lists_for(2)}])
            write(d / "rag2_answers_dev.jsonl", [{"item_id": "MC-1", "arm": "R2"}])
            RP.publish(d, res, "dev")
            self.assertTrue((res / "rag2_answers_dev.jsonl").is_file())
            body = (res / "rag2_lists_dev.ids.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("abstract", body)
            self.assertNotIn("title", body)

    def _args(self, **kw):
        base = dict(model_path="m.gguf", judge_path=None, n_threads="6", ablations=False,
                    no_temporal_ablation=False, commit=False)
        base.update(kw)
        return mock.Mock(**base)

    def test_plans(self):
        d, res = Path("data"), Path("results")
        names = [n for n, _ in RP.dev_plan(self._args(), d, res)]
        self.assertIn("answers (dev): R2 R2C R2V R2V-ND", names)
        self.assertIn("design record", names)
        names = [n for n, _ in RP.dev_plan(self._args(ablations=True, judge_path="q.gguf", commit=True), d, res)]
        self.assertIn("answers (dev): R2 R2C R2V R2V-ND R2-NF R2-RQ R2-BR", names)
        self.assertIn("directness judge (dev)", names)
        self.assertEqual(names[-1], "commit")
        names = [n for n, _ in RP.confirm_plan(self._args(no_temporal_ablation=True), d, res)]
        self.assertIn("answers (confirm): R2 R2C R2V", names)
        self.assertIn("findings", names)

    def test_confirm_guards(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()) as err:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"model")
            res = Path(tmp) / "results"
            res.mkdir()
            base = ["confirm", "--model-path", str(model), "--results-dir", str(res), "--data-dir", tmp]
            self.assertEqual(RP.main(base), 2)                                   # no --go
            self.assertEqual(RP.main(base + ["--go"]), 2)                        # no dev report
            (res / "RAG2_DEV_REPORT.md").write_text("x", encoding="utf-8")
            with mock.patch.object(RP, "frozen_is_pushed", return_value=(False, "not on origin/main")):
                self.assertEqual(RP.main(base + ["--go"]), 2)
            with mock.patch.object(RP, "frozen_is_pushed", return_value=(True, "")):
                self.assertEqual(RP.main(base + ["--go"]), 2)                    # no design record yet
                RP.write_design(res, str(model))
                self.assertEqual(RP.design_differences(res, str(model)), [])
                with mock.patch.dict(R.SETTINGS, {"budget": 4}):
                    self.assertIn("settings", RP.design_differences(res, str(model)))
        self.assertIn("--go", err.getvalue())

    def test_dry_run(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(RP.main(["dev", "--model-path", "m.gguf", "--dry-run"]), 0)
        self.assertIn("rag2_run rationale", out.getvalue())

    def test_the_ad_phase_plans_the_standard_answers_first_and_is_guarded(self):
        args = mock.Mock(model_path="m.gguf", judge_path=None, n_threads="6", commit=False)
        names = [n for n, _ in RP.ad_plan(args, Path("data"), Path("results"))]
        self.assertIn("B0 and B1 answers (ad)", names)
        self.assertIn("answers (ad): R2 R2C R2V", names)
        self.assertEqual(names[-2:], ["findings (ad)", "environment record"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(RP.main(["ad", "--model-path", "m.gguf", "--dry-run"]), 2)   # no --go

    def test_every_analysis_step_looks_for_the_label_audit_of_its_split(self):
        """The label-stable subset is reported whenever the audit exists. The audit runs after the phase, so the analysis step names the file in
        advance and a missing file is skipped (``analyze_rag2.stable_ids_from_audit``); the dev and confirm plans did this from the start, the ad plan did not."""
        args = mock.Mock(model_path="m.gguf", judge_path=None, n_threads="6", ablations=False, no_temporal_ablation=False, commit=False)
        for split, plan in (("dev", RP.dev_plan), ("confirm", RP.confirm_plan), ("ad", RP.ad_plan)):
            steps = {name: action for name, action in plan(args, Path("data"), Path("results"))}
            action = steps[f"analysis ({split})"]
            self.assertEqual(action[action.index("--label-audit") + 1], str(Path("results") / f"label_audit_{split}.jsonl"), split)


class FreshPhaseTests(unittest.TestCase):
    def test_the_status_line_that_puts_the_preregistration_in_force(self):
        self.assertTrue(RP.IN_FORCE.search("**Status.** This section is IN FORCE since 2026-11-01 (commit abc)."))
        self.assertIsNone(RP.IN_FORCE.search("**Status.** This section is a draft written on 2026-10-09."))
        text = (Path(RP.HERE).parents[1] / "docs" / "protocol.md").read_text(encoding="utf-8")
        draft = "**Status.** This section is a draft" in text
        self.assertTrue(draft or RP.IN_FORCE.search(text), "§10 must read either as a draft or as in force, never neither")
        self.assertFalse(draft and RP.IN_FORCE.search(text))

    def test_without_the_freeze_the_fresh_run_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "protocol.md").write_text("**Status.** draft", encoding="utf-8")
            problems = RP.freeze_problems(root / "data", root)
            self.assertTrue(any("IN FORCE" in p for p in problems))
            self.assertTrue(any("manifest_fresh.json" in p for p in problems))

    def test_the_fresh_plan_runs_r2_r2c_r2v_and_nothing_else(self):
        a = argparse.Namespace(model_path="m.gguf", judge_path="j.gguf", n_threads="6", commit=False, ablations=False, no_temporal_ablation=False)
        steps = RP.fresh_plan(a, Path("data"), Path("results"))
        names = " | ".join(n for n, _ in steps)
        argv = " ".join(" ".join(map(str, act)) for _, act in steps if isinstance(act, list))
        self.assertIn("--arms R2 R2C R2V", argv)
        self.assertNotIn("R2V-ND", argv)
        self.assertNotIn("generate_answers", argv)                                   # no B0 / B1
        self.assertNotIn("judge", names.lower().replace("no directness judge", ""))
        self.assertIn("--abstracts-only", argv)
        self.assertIn("analyze_fresh", argv)
        self.assertEqual(RP.EXPECTED_ITEMS["fresh"], 850)

    def test_preflight_of_the_fresh_split_does_not_ask_for_b0_and_b1(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "benchmark.jsonl").write_text("", encoding="utf-8")
            names = [n for n, _, _ in RP.preflight(d, "fresh")]
            self.assertFalse(any("B0 and B1" in n for n in names))
            self.assertTrue(any("B0 and B1" in n for n in [n for n, _, _ in RP.preflight(d, "ad")]))

    def test_the_items_in_the_benchmark_file_must_match_the_manifest_when_the_list_is_pushed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "protocol.md").write_text("**Status.** This section is IN FORCE since 2026-10-10 (list `abc`).", encoding="utf-8")
            ids = ["FR-00001", "FR-00002"]
            manifest = root / "experiments" / "medchange" / "manifest_fresh.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"items": 2, "item_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}), encoding="utf-8")
            data = root / "data"
            data.mkdir()

            def write(listed):
                (data / "benchmark.jsonl").write_text("\n".join(json.dumps({"item_id": i, "split": "fresh"}) for i in listed) + "\n", encoding="utf-8")
            with mock.patch.object(RP, "frozen_is_pushed", return_value=(True, "file is on origin/main")):
                write(ids)
                self.assertEqual(RP.freeze_problems(data, root), [])
                write(ids[:1])
                self.assertEqual(len(RP.freeze_problems(data, root)), 1)
                write(ids[::-1])
                self.assertTrue(any("differ" in p for p in RP.freeze_problems(data, root)))               # the order is part of the list
            with mock.patch.object(RP, "frozen_is_pushed", return_value=(False, "differs from origin/main")):
                write(ids)
                self.assertEqual(RP.freeze_problems(data, root), ["differs from origin/main"])


class TraceabilityTests(unittest.TestCase):
    def test_environment_record_has_versions_and_the_code_commit_and_is_written_with_lf(self):
        rec = RP.environment_record("dev", "models/Meta-Llama.gguf", "models/Qwen.gguf")
        self.assertEqual(set(rec["packages"]), set(RP.PACKAGES))
        self.assertEqual(rec["generator_file"], "Meta-Llama.gguf")
        self.assertEqual(rec["judge_file"], "Qwen.gguf")
        self.assertRegex(rec["python"], r"^3\.")
        with TemporaryDirectory() as tmp:
            ok, message = RP.write_environment(Path(tmp), "dev", "m.gguf")
            self.assertTrue(ok)
            raw = (Path(tmp) / "rag2_environment_dev.json").read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(json.loads(raw)["phase"], "dev")

    def test_environment_record_survives_a_machine_without_git(self):
        with mock.patch.object(RP.subprocess, "run", side_effect=OSError("no git")):
            rec = RP.environment_record("confirm", "m.gguf")
        self.assertIsNone(rec["git_commit"])
        self.assertIsNone(rec["tracked_code_modified"])

    def test_files_the_study_writes_use_lf_line_endings(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            RP._write(d / "a.md", "x\ny\n")
            RP.write_design(d, str(d / "m.gguf")) if (d / "m.gguf").write_bytes(b"m") else None
            for path in d.iterdir():
                self.assertNotIn(b"\r", path.read_bytes(), path.name)


class PipelineWiringTests(unittest.TestCase):
    """The commands the pipeline launches must only use flags that their module declares."""

    @staticmethod
    def _declared(module: str) -> set[str]:
        import ast
        path = Path(*module.split(".")).with_suffix(".py")
        flags = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
                flags.update(a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--"))
        return flags

    def test_every_flag_of_every_planned_command_is_declared(self):
        args = mock.Mock(model_path="m.gguf", judge_path="j.gguf", n_threads="6", ablations=True,
                         no_temporal_ablation=False, commit=True)
        plans = (RP.dev_plan(args, Path("data"), Path("results")), RP.confirm_plan(args, Path("data"), Path("results")),
                 RP.ad_plan(args, Path("data"), Path("results")))
        checked = 0
        for plan in plans:
            for name, action in plan:
                if not isinstance(action, list):
                    continue
                module = action[2]
                flags = {x for x in map(str, action[3:]) if x.startswith("--")}
                if module.endswith("rag2_run"):                   # rag2_run declares its flags per sub-command
                    module_flags = self._declared(module)
                else:
                    module_flags = self._declared(module)
                self.assertLessEqual(flags, module_flags, f"{name}: {flags - module_flags} is not declared by {module}")
                checked += 1
        self.assertGreater(checked, 15)


class EndToEndFlowTests(unittest.TestCase):
    """Every step of a phase, with fake models, on synthetic data: wiring, files, analysis, report, design record."""

    def test_dev_phase_end_to_end(self):
        import argparse
        S_, F__, N_ = S, F_, N
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, results = tmp / "data", tmp / "results"
            (data / "pubmed_g0").mkdir(parents=True)
            results.mkdir()
            items = [item(f"MC-{i}", kind="changed" if i % 2 else "unchanged", gold=[S_, F__, N_][i % 3]) for i in range(6)]
            for it in items:
                it["newest"].update(pmid=f"N{it['item_id']}", row=1, cochrane_id="CD1")
                it["previous"].update(pmid=f"O{it['item_id']}", row=2)
            write(data / "benchmark.jsonl", items)
            abstract_rows = []
            for it in items:
                recs = records(n_per_type=6, upper="2011-12-31", lower="2011-01-01")
                for r in recs:
                    r["pmid"] = f"{it['item_id']}-{r['pmid']}"
                    abstract_rows.append({"pmid": r["pmid"], "title": f"Trial {r['pmid']} of drug pain",
                                          "abstract": "RESULTS: pain fell. CONCLUSIONS: the drug reduces pain. " * 6})
                (data / "pubmed_g0" / f"{it['item_id']}.json").write_text(json.dumps({"records": recs}), encoding="utf-8")
            write(data / "abstracts.jsonl", abstract_rows)
            write(data / "answers_dev.jsonl", [
                {"item_id": it["item_id"], "arm": arm, "verdict": N_ if arm == "B0" else it["newest"]["label"],
                 "admitted": [], "admitted_upper": [], "text": "VERDICT: x"} for it in items for arm in ("B0", "B1")])
            write(data / "frozen_dev.jsonl", [{"item_id": it["item_id"], "candidates": []} for it in items])
            write(results / "label_audit_dev.jsonl", [{"item_id": it["item_id"], "which": "newest",
                                                       "gold": it["newest"]["label"], "relabel": it["newest"]["label"]}
                                                      for it in items])
            model = tmp / "m.gguf"
            model.write_bytes(b"fake model file")

            class Scorer:
                def __init__(self, *a, **k):
                    self.name, self.model_sha256, self.n_ctx = "fake", "0", 1

                def score(self, question, study):
                    return 0.8, 0.95

            def ns(**kw):
                base = dict(split="dev", limit=None, out=None, model_path=str(model), n_threads=1, device="cpu",
                            variants=["R2"], arms=["R2", "R2C", "R2V", "R2V-ND"])
                base.update(kw)
                return argparse.Namespace(**base)
            patches = [mock.patch.object(RR, "llama_generator", lambda *a, **k: fake_generate),
                       mock.patch.object(RR, "LlamaYesNo", Scorer),
                       mock.patch("experiments.medchange.encoders.medcpt_query_encoder", lambda **k: HashingEncoder()),
                       mock.patch("experiments.medchange.encoders.medcpt_article_encoder", lambda **k: HashingEncoder()),
                       mock.patch("experiments.medchange.encoders.MedCPTReranker", lambda **k: LexicalOverlapReranker())]
            for patch in patches:
                patch.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    codes = [RR.cmd_rationale(ns(), data), RR.cmd_lists(ns(), data), RR.cmd_filter(ns(), data),
                             RR.cmd_answers(ns(), data), RR.cmd_judge(ns(), data)]
                    analysis = AR.main(["--split", "dev", "--data-dir", str(data), "--out-dir", str(results),
                                       "--label-audit", str(results / "label_audit_dev.jsonl")])
            finally:
                for patch in patches:
                    patch.stop()
            self.assertEqual(codes, [0, 0, 0, 0, 0])
            self.assertEqual(analysis, 0)
            for step in ("rationales", "lists", "filter", "answers", "directness"):
                self.assertTrue((data / f"rag2_{step}_dev.jsonl").is_file(), step)
                self.assertTrue((data / f"rag2_{step}_dev.config.json").is_file(), step)
            self.assertTrue(RP.publish(data, results, "dev")[0])
            self.assertTrue((results / "rag2_lists_dev.ids.jsonl").is_file())
            self.assertNotIn("abstract", (results / "rag2_lists_dev.ids.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(RP.write_dev_report(results)[0])
            self.assertTrue(RP.write_design(results, str(model))[0])
            self.assertTrue(RP.write_environment(results, "dev", str(model))[0])
            self.assertEqual(RP.design_differences(results, str(model)), [])
            rep = json.loads((results / "rag2_analysis_dev.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["arms"], ["B0", "B1", "R2", "R2C", "R2V", "R2V-ND"])
            self.assertEqual(rep["reading"].split(" (")[0], "development estimate only")
            self.assertIn("label_stable", rep)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                RP.status(data, results)
            self.assertEqual(json.loads(out.getvalue())["dev_answers"], 24)
            # a rerun of every step finds everything finished and writes nothing new
            before = {p.name: p.read_bytes() for p in data.glob("rag2_*")}
            for patch in patches:
                patch.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    RR.cmd_rationale(ns(), data)
                    RR.cmd_lists(ns(), data)
                    RR.cmd_filter(ns(), data)
                    RR.cmd_answers(ns(), data)
            finally:
                for patch in patches:
                    patch.stop()
            self.assertEqual(before, {p.name: p.read_bytes() for p in data.glob("rag2_*")})


class CommittedConfigurationTests(unittest.TestCase):
    """Every committed output of the R2 family was made under the frozen design on file (generator file, settings, prompts), for every split
    that has run, so that no split was produced under a changed design. The independent judge's files differ on purpose and are skipped."""

    def test_every_recorded_configuration_agrees_with_the_design_record(self):
        results = Path(RP.__file__).resolve().parent / "results"
        design = json.loads((results / "rag2_design.json").read_text(encoding="utf-8"))
        checked = 0
        for path in sorted(results.glob("rag2_*_*.config.json")):
            if "directness" in path.name:
                continue
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if "model_sha256" in cfg:
                self.assertEqual(cfg["model_sha256"], design["generator_model_sha256"], path.name)
            if "settings_sha256" in cfg:
                self.assertEqual(cfg["settings_sha256"], design["settings_sha256"], path.name)
            if "prompts" in cfg:
                self.assertEqual(cfg["prompts"], design["prompts"], path.name)
            checked += 1
        self.assertGreaterEqual(checked, 12)                  # rationales, lists, filter and answers of dev, confirm and ad (and fresh, once committed)
        self.assertEqual(design["settings_sha256"], R.settings_hash())
        self.assertEqual(design["prompts"], R.prompt_hashes())


if __name__ == "__main__":
    unittest.main()
