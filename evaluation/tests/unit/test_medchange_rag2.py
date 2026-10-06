"""The realigned study: adapted RAG² (R2) and evidence-criteria verification (R2V).

Synthetic data and fakes only: no model, no MedCPT download, no network.
"""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import analyze_rag2 as A
from experiments.medchange import pipeline as P
from experiments.medchange import rag2 as R
from experiments.medchange import rag2_pipeline as RP
from experiments.medchange import rag2_run as RR
from experiments.medchange.encoders import HashingEncoder, LexicalOverlapReranker

S, F_, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"
TYPES = {"SR/MA": ["Systematic Review"], "RCT": ["Randomized Controlled Trial"], "other": ["Journal Article"]}


def write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def item(iid, kind="changed", gold=S, prev=F_, date="2015-06-01", split="dev", ad=False):
    return {"item_id": iid, "kind": kind, "split": split, "likely_label_noise": False, "ad_related": ad,
            "question": f"Does drug {iid} reduce pain in adults?",
            "newest": {"label": gold, "date": date, "pmid": f"N{iid}"},
            "previous": {"label": prev, "date": "2010-01-01", "pmid": f"O{iid}"}}


def records(n_per_type=10, upper="2012-12-31", lower="2012-01-01"):
    out = []
    for kind, pubtypes in TYPES.items():
        for k in range(n_per_type):
            out.append({"pmid": f"{kind}-{k}", "lower": lower, "upper": upper, "pubtypes": pubtypes, "journal": "J"})
    return out


def abstracts(recs):
    return {r["pmid"]: {"pmid": r["pmid"], "title": f"Trial of drug pain {r['pmid']}",
                        "abstract": ("RESULTS: pain fell. CONCLUSIONS: drug reduces pain. " * 8)} for r in recs}


def cand(pmid, rank, stratum="RCT", upper="2012-12-31", lower="2012-01-01"):
    return {"pmid": pmid, "rank": rank, "rerank_score": 1.0 / rank, "stratum": stratum, "lower": lower,
            "upper": upper, "pubtypes": TYPES[stratum], "journal": "J", "title": f"Title {pmid}",
            "abstract": f"RESULTS: findings {pmid}. CONCLUSIONS: conclusion {pmid}."}


def lists_for(n=8):
    return {v: [cand(f"{v}-{i}", i + 1) for i in range(n)] for v in R.LISTS}


def scores_for(lists, p=0.9, valid=True):
    return {c["pmid"]: {"pmid": c["pmid"], "p_yes": p, "mass": 0.99, "valid": valid}
            for cands in lists.values() for c in cands}


# --------------------------------------------------------------------------------------
# rag2.py
# --------------------------------------------------------------------------------------

class RetrievalTests(unittest.TestCase):
    def test_balanced_takes_the_quota_of_each_type_in_order(self):
        strata = ["other"] * 5 + ["RCT"] * 3 + ["SR/MA"]
        order = list(range(9))
        self.assertEqual(R.balanced(order, strata, quota=2), [0, 1, 5, 6, 8])

    def _build(self, recs, rationale="Population adults; intervention drug; outcome pain.", date="2015-06-01"):
        return R.build_lists(item("MC-1", date=date), recs, abstracts(recs), rationale,
                             query_encoder=HashingEncoder(), article_encoder=HashingEncoder(),
                             reranker=LexicalOverlapReranker())

    def test_lists_are_capped_deterministic_and_typed(self):
        recs = records()
        a, b = self._build(recs), self._build(recs)
        self.assertEqual(a["lists_hash"], b["lists_hash"])
        self.assertEqual(set(a["lists"]), set(R.LISTS))
        for variant, cands in a["lists"].items():
            self.assertEqual(len(cands), R.RERANK_KEEP)
            self.assertEqual([c["rank"] for c in cands], list(range(1, R.RERANK_KEEP + 1)))
            scores = [c["rerank_score"] for c in cands]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(c["stratum"] in R.STRATA for c in cands))
        self.assertTrue(a["rationale_used"])

    def test_records_after_the_cutoff_or_ineligible_are_never_listed(self):
        recs = records(n_per_type=2)
        recs.append({"pmid": "late", "lower": "2016-01-01", "upper": "2016-12-31", "pubtypes": TYPES["RCT"], "journal": "J"})
        recs.append({"pmid": "erratum", "lower": "2010-01-01", "upper": "2010-12-31",
                     "pubtypes": ["Published Erratum"], "journal": "J"})
        rec = self._build(recs)
        listed = {c["pmid"] for cands in rec["lists"].values() for c in cands}
        self.assertNotIn("late", listed)
        self.assertNotIn("erratum", listed)
        self.assertEqual(rec["n_eligible"], 6)

    def test_an_empty_rationale_falls_back_to_the_question(self):
        rec = self._build(records(n_per_type=3), rationale="  ")
        self.assertFalse(rec["rationale_used"])
        self.assertEqual(rec["lists"]["R2"], rec["lists"]["R2-RQ"])

    def test_no_eligible_record_gives_empty_lists(self):
        rec = self._build(records(n_per_type=2, upper="2020-01-01", lower="2019-01-01"))
        self.assertEqual(rec["n_eligible"], 0)
        self.assertTrue(all(v == [] for v in rec["lists"].values()))

    def test_strip_text_removes_publisher_text_only(self):
        rec = self._build(records(n_per_type=2))
        stripped = R.strip_text(rec)
        for cands in stripped["lists"].values():
            for c in cands:
                self.assertNotIn("title", c)
                self.assertNotIn("abstract", c)
                self.assertIn("pmid", c)
        self.assertIn("title", rec["lists"]["R2"][0])            # the original is untouched


class FilterTests(unittest.TestCase):
    def test_yes_probability_from_first_token(self):
        p, mass = R.yes_probability([{"token": "Yes", "logprob": -0.1}, {"token": " no", "logprob": -2.5},
                                     {"token": "Maybe", "logprob": -3.0}])
        self.assertAlmostEqual(mass, 0.9048 + 0.0821, places=3)
        self.assertGreater(p, 0.9)
        self.assertEqual(R.yes_probability([{"token": "A", "logprob": -0.1}]), (None, 0.0))

    def test_admission_filters_keeps_order_and_budget(self):
        lists = lists_for()
        scores = scores_for(lists)
        scores["R2-0"]["p_yes"] = 0.2                      # filtered out
        scores["R2-1"]["valid"] = False                    # invalid: kept
        scores["R2-1"]["p_yes"] = 0.0
        admitted = R.admit("R2", lists, scores)
        self.assertEqual([c["pmid"] for c in admitted], ["R2-1", "R2-2", "R2-3", "R2-4", "R2-5"])
        self.assertEqual([c["pmid"] for c in R.admit("R2-NF", lists, {})], [f"R2-{i}" for i in range(5)])
        self.assertEqual(R.admit("R2V", lists, scores), admitted)          # the criteria arms read R2's evidence

    def test_everything_filtered_admits_nothing_and_a_missing_judgement_is_an_error(self):
        lists = lists_for()
        self.assertEqual(R.admit("R2", lists, scores_for(lists, p=0.1)), [])
        with self.assertRaises(KeyError):
            R.admit("R2", lists, {})


class CriteriaTests(unittest.TestCase):
    def test_dated_and_undated_prompts(self):
        passages = [cand("1", 1, "RCT", upper="2009-05-31"), cand("2", 2, "SR/MA", upper="2011-03-31")]
        _, dated = R.build_criteria_prompt("Q?", passages, draft="VERDICT: SUPPORTED", dated=True)
        _, undated = R.build_criteria_prompt("Q?", passages, draft="VERDICT: SUPPORTED", dated=False)
        self.assertIn("(randomized or controlled trial, 2009)", dated)
        self.assertIn("(systematic review or meta-analysis, 2011)", dated)
        self.assertIn("newer evidence take precedence", dated)
        self.assertNotIn("2009", undated)
        self.assertNotIn("newer evidence", undated)
        self.assertIn("direct randomized evidence.", undated)

    def test_criteria_control_and_verification_share_everything_before_the_task(self):
        passages = [cand("1", 1)]
        _, crit = R.build_criteria_prompt("Q?", passages)
        _, verify = R.build_criteria_prompt("Q?", passages, draft="VERDICT: SUPPORTED\nBecause [1].")
        shared = crit.split("Answer the question by applying")[0]
        self.assertTrue(verify.startswith(shared))
        self.assertIn("Draft answer:\nVERDICT: SUPPORTED", verify)
        self.assertNotIn("Draft answer", crit)

    def test_parse_final_verdict(self):
        self.assertEqual(R.parse_final_verdict("DIRECT STUDIES: 1\nFINDINGS: x.\nFINAL VERDICT: REFUTED"), F_)
        self.assertEqual(R.parse_final_verdict("**FINAL VERDICT:** not enough  information"), N)
        self.assertEqual(R.parse_final_verdict("FINAL VERDICT: SUPPORTED\nFINAL VERDICT: REFUTED"), F_)
        self.assertIsNone(R.parse_final_verdict("VERDICT: SUPPORTED"))
        self.assertIsNone(R.parse_final_verdict(R.FORMAT))                  # the format line copied verbatim

    def test_cited_and_direct_numbers(self):
        self.assertEqual(R.cited_numbers("See [1] and [2, 4] and [3-5]."), [1, 2, 3, 4, 5])
        self.assertEqual(R.direct_numbers("DIRECT STUDIES: 1, 3\nFINAL VERDICT: SUPPORTED"), [1, 3])
        self.assertEqual(R.direct_numbers("DIRECT STUDIES: NONE"), [])
        self.assertIsNone(R.direct_numbers("VERDICT: SUPPORTED"))

    def test_unsupported_decisive(self):
        self.assertIsNone(R.unsupported_decisive(S, "VERDICT: SUPPORTED", 0))
        self.assertFalse(R.unsupported_decisive(N, "VERDICT: NOT ENOUGH INFORMATION", 5))
        self.assertFalse(R.unsupported_decisive(S, "VERDICT: SUPPORTED because [2] shows it", 5))
        self.assertTrue(R.unsupported_decisive(S, "VERDICT: SUPPORTED, as is well known", 5))
        self.assertTrue(R.unsupported_decisive(F_, "VERDICT: REFUTED per [7]", 5))
        self.assertFalse(R.unsupported_decisive(F_, "DIRECT STUDIES: 2\nFINAL VERDICT: REFUTED", 5))
        self.assertTrue(R.unsupported_decisive(F_, "DIRECT STUDIES: NONE\nFINAL VERDICT: REFUTED", 5))

    def test_anachronistic_years(self):
        text = "A 2019 meta-analysis and a 2010 trial; see also 2015."
        self.assertEqual(R.anachronistic_years(text, "2015-06-01"), [2019])
        self.assertEqual(R.anachronistic_years("no years here", "2015-06-01"), [])

    def test_design_record_is_stable_and_complete(self):
        a, b = R.design_record("abc"), R.design_record("abc")
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"settings", "settings_sha256", "prompts", "generator_model_sha256", "encoders"})
        self.assertEqual(len(a["settings_sha256"]), 12)
        self.assertIn("verify_task", a["prompts"])


# --------------------------------------------------------------------------------------
# rag2_run.py
# --------------------------------------------------------------------------------------

def fake_generate(system, user):
    if "Draft answer" in user:
        return "DIRECT STUDIES: 1\nFINDINGS: no benefit over placebo.\nFINAL VERDICT: REFUTED"
    if "Evidence criteria" in user:
        return "DIRECT STUDIES: NONE\nFINDINGS: no direct study.\nFINAL VERDICT: NOT ENOUGH INFORMATION"
    if "step-by-step" in user:
        return "Population: adults. Intervention: drug. Outcome: pain. Tentative answer: supported."
    return "VERDICT: SUPPORTED\nStudy [1] shows benefit."


class FakeScorer:
    name, model_sha256, n_ctx = "fake", "0", 1

    def __init__(self, invalid=()):
        self.invalid = set(invalid)
        self.calls = 0

    def score(self, question, study):
        self.calls += 1
        if any(p in study for p in self.invalid):
            return None, 0.0
        return 0.8, 0.95


class RunnerTests(unittest.TestCase):
    def test_rationales_are_resumable(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.jsonl"
            items = [item("MC-1"), item("MC-2")]
            self.assertEqual(RR.run_rationales(items, fake_generate, out), 2)
            self.assertEqual(RR.run_rationales(items, fake_generate, out), 0)
            rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(rec["rationale"].startswith("Population"))

    def test_lists_step_writes_all_variants_and_resumes(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "l.jsonl"
            recs = records(n_per_type=4)
            kw = dict(query_encoder=HashingEncoder(), article_encoder=HashingEncoder(),
                      reranker=LexicalOverlapReranker())
            n = RR.run_lists([item("MC-1")], {"MC-1": {"records": recs}}, abstracts(recs), {"MC-1": "adults drug pain"},
                             out, **kw)
            self.assertEqual(n, 1)
            self.assertEqual(RR.run_lists([item("MC-1")], {"MC-1": {"records": recs}}, abstracts(recs),
                                          {"MC-1": "x"}, out, **kw), 0)

    def test_judgements_deduplicate_resume_and_mark_invalid(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "f.jsonl"
            lists = lists_for(3)
            it = item("MC-1")
            tasks = RR.filter_tasks([it], {"MC-1": {"lists": lists}}, ["R2", "R2-RQ"]) + [(it, lists["R2"][0])]
            scorer = FakeScorer(invalid={"R2-RQ-0"})
            self.assertEqual(RR.run_judgements(tasks, scorer, out), 6)
            self.assertEqual(RR.run_judgements(tasks, scorer, out), 0)
            recs = {r["pmid"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
            self.assertFalse(recs["R2-RQ-0"]["valid"])
            self.assertTrue(recs["R2-0"]["valid"])
            self.assertEqual(scorer.calls, 6)

    def _answers(self, tmp, lists, scores, arms, generate=fake_generate):
        out = Path(tmp) / "a.jsonl"
        it = item("MC-1")
        RR.run_answers([it], {"MC-1": {"lists": lists}}, {"MC-1": scores}, arms, out, generate)
        return {r["arm"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}

    def test_answer_arms(self):
        lists = lists_for()
        with TemporaryDirectory() as tmp:
            recs = self._answers(tmp, lists, scores_for(lists), ["R2V", "R2C", "R2", "R2V-ND", "R2-NF"])
        self.assertEqual(recs["R2"]["verdict"], S)
        self.assertEqual(recs["R2C"]["verdict"], N)
        self.assertEqual(recs["R2V"]["verdict"], F_)
        self.assertTrue(recs["R2V"]["changed"])
        self.assertEqual(recs["R2V"]["draft_verdict"], S)
        self.assertEqual(recs["R2V"]["admitted"], recs["R2"]["admitted"])
        self.assertEqual(len(recs["R2"]["admitted"]), 5)
        self.assertEqual(recs["R2"]["admitted_stratum"], ["RCT"] * 5)

    def test_no_evidence_keeps_r2s_answer_and_invalid_verification_keeps_the_draft(self):
        lists = lists_for()
        with TemporaryDirectory() as tmp:
            recs = self._answers(tmp, lists, scores_for(lists, p=0.1), ["R2", "R2C", "R2V"])
        self.assertEqual(recs["R2"]["admitted"], [])
        for arm in ("R2C", "R2V"):
            self.assertEqual(recs[arm]["fallback"], "no evidence")
            self.assertEqual(recs[arm]["verdict"], recs["R2"]["verdict"])
        broken = lambda s, u: "VERDICT: SUPPORTED [1]" if "Evidence criteria" not in u else "I am not sure."
        with TemporaryDirectory() as tmp:
            recs = self._answers(tmp, lists, scores_for(lists), ["R2", "R2C", "R2V"], broken)
        self.assertEqual(recs["R2V"]["verdict"], S)
        self.assertFalse(recs["R2V"]["valid"])
        self.assertIsNone(recs["R2C"]["verdict"])                       # R2C has no draft to fall back on

    def test_criteria_arms_need_r2(self):
        lists = lists_for()
        with TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            self._answers(tmp, lists, scores_for(lists), ["R2V"])

    def test_cli_refuses_without_benchmark(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(RR.main(["lists", "--split", "dev", "--data-dir", tmp]), 2)


# --------------------------------------------------------------------------------------
# analyze_rag2.py
# --------------------------------------------------------------------------------------

def answer(iid, arm, verdict, admitted=(), text=None, **extra):
    rec = {"item_id": iid, "arm": arm, "verdict": verdict, "admitted": list(admitted),
           "text": text if text is not None else f"VERDICT: {verdict} [1]"}
    if arm.startswith("R2"):
        rec.update(admitted_lower=["2012-01-01"] * len(admitted), admitted_upper=["2012-12-31"] * len(admitted),
                   admitted_stratum=["RCT"] * len(admitted))
    rec.update(extra)
    return rec


def world(n=40, r2v_extra=6):
    """n items, gold SUPPORTED; R2 right on half; R2V right on half plus ``r2v_extra`` more."""
    items = {f"MC-{k:03d}": item(f"MC-{k:03d}", kind="changed" if k % 2 else "unchanged",
                                 split="confirm", ad=(k == 0)) for k in range(n)}
    answers = {}
    for k, iid in enumerate(sorted(items)):
        r2_ok = k < n // 2
        r2v_ok = r2_ok or k < n // 2 + r2v_extra
        for arm, ok in (("B0", k < n // 3), ("B1", r2_ok), ("R2", r2_ok), ("R2C", r2v_ok), ("R2V", r2v_ok)):
            v = S if ok else N
            extra = {}
            if arm in ("R2C", "R2V"):
                extra = dict(draft_verdict=S if r2_ok else N, valid=True, changed=(v != (S if r2_ok else N)),
                             fallback=None)
            answers[(iid, arm)] = answer(iid, arm, v, admitted=() if arm == "B0" else ("p1", "p2"), **extra)
    return items, answers


class AnalysisTests(unittest.TestCase):
    def test_requirement_reading(self):
        primary = {"diff_a_minus_b": 0.02, "ci95": [0.005, 0.04], "mcnemar_p": 0.01, "confirmed": True}
        self.assertEqual(A.requirement_reading(primary, "confirm"), "met and confirmed")
        self.assertEqual(A.requirement_reading(dict(primary, confirmed=False), "confirm"),
                         "met as a point estimate, not confirmed")
        self.assertEqual(A.requirement_reading(dict(primary, diff_a_minus_b=0.009), "confirm"), "not met")
        self.assertTrue(A.requirement_reading(primary, "dev").startswith("dev estimate only"))
        self.assertEqual(A.requirement_reading(None, "confirm"), "not run")

    def test_report_on_a_synthetic_world(self):
        items, answers = world()
        rep = A.report(items, answers, "confirm")
        self.assertEqual(rep["arms"], ["B0", "B1", "R2", "R2C", "R2V"])
        p = rep["primary"]
        self.assertAlmostEqual(p["diff_a_minus_b"], 6 / 40)
        self.assertEqual((p["a_only_correct"], p["b_only_correct"]), (6, 0))
        self.assertEqual(rep["requirement"], "met and confirmed")
        self.assertIn("R2 vs B1", rep["secondary"])
        self.assertEqual(rep["verification"]["R2V"]["changes_fixed"], 6)
        self.assertEqual(rep["verification"]["R2V"]["changes_broke"], 0)
        self.assertEqual(rep["retrieval"]["R2"]["evidence_mix"], {"SR/MA": 0.0, "RCT": 1.0, "other": 0.0})
        self.assertEqual(rep["case_study_alzheimers"]["n"], 1)
        text = A.to_markdown(rep)
        for needle in ("verdict accuracy", "The requirement", "Secondary comparisons", "Retrieval", "met and confirmed"):
            self.assertIn(needle, text)

    def test_unsupported_answer_indicators(self):
        items, answers = world(n=10)
        iid = sorted(items)[0]
        answers[(iid, "B1")]["text"] = "VERDICT: SUPPORTED. A 2019 meta-analysis shows it."
        rows = A.hallucination_rows(items, answers, ["B0", "B1"])
        self.assertAlmostEqual(rows["B1"]["anachronism_rate"], 0.1)
        self.assertIsNone(rows["B0"]["unsupported_decisive_rate"])            # B0 has no evidence
        self.assertAlmostEqual(rows["B1"]["unsupported_decisive_rate"], 0.1)  # one decisive answer cites nothing

    def test_stable_ids_and_cli(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            write(d / "audit.jsonl", [{"item_id": "a", "which": "newest", "gold": S, "relabel": S},
                                      {"item_id": "b", "which": "newest", "gold": S, "relabel": N},
                                      {"item_id": "a", "which": "previous", "gold": N, "relabel": N}])
            self.assertEqual(A.stable_ids_from_audit(d / "audit.jsonl"), ["a"])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(A.main(["--split", "confirm", "--data-dir", tmp]), 2)
            items, answers = world(n=12)
            write(d / "benchmark.jsonl", list(items.values()))
            write(d / "answers_confirm.jsonl", [r for (i, a), r in answers.items() if a in ("B0", "B1")])
            write(d / "rag2_answers_confirm.jsonl", [r for (i, a), r in answers.items() if a.startswith("R2")])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main(["--split", "confirm", "--data-dir", tmp, "--out-dir", tmp]), 0)
            self.assertTrue((d / "rag2_analysis_confirm.md").is_file())


# --------------------------------------------------------------------------------------
# rag2_pipeline.py
# --------------------------------------------------------------------------------------

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
                "secondary": {}, "requirement": "dev estimate only (exploratory)"}

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
            self.assertIn("Reading:", (res / "RAG2_FINDINGS.md").read_text(encoding="utf-8"))

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



# --------------------------------------------------------------------------------------
# The Alzheimer's/dementia secondary test set
# --------------------------------------------------------------------------------------

from experiments.medchange import ad_benchmark as AD  # noqa: E402


def medrev_row(k, question, label=S, date="2015 Mar 2"):
    return {"": str(k), "Question": question, "objectives": "", "Label": label, "PMID": f"https://pubmed/{9000 + k}",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD{100000 + k:06d}. doi: x"}


class AlzheimersBenchmarkTests(unittest.TestCase):
    def _world(self):
        rows = {1: medrev_row(1, "Does donepezil help people with Alzheimer's disease?"),
                2: medrev_row(2, "Does donepezil help people with Alzheimer's disease?", F_, "2010 Jan 1"),
                3: medrev_row(3, "Does exercise slow cognitive decline in older adults?", N),
                4: medrev_row(4, "Does exercise slow cognitive decline in older adults?", N),   # duplicate question
                5: medrev_row(5, "Do acetylcholinesterase inhibitors improve autism?"),           # off topic
                6: medrev_row(6, "Is music therapy effective for dementia?", F_),
                7: medrev_row(7, "Is music therapy effective for dementia?", F_, "2009 May 5")}
        groups = {10: [1, 2], 11: [6, 7], 12: [8]}
        rows[8] = medrev_row(8, "Does aspirin prevent dementia?")
        return rows, groups

    def test_selection(self):
        rows, groups = self._world()
        items = AD.build_ad_items(rows, groups, used_groups={12})
        by_id = {i["item_id"]: i for i in items}
        self.assertEqual(set(by_id), {"AD-00001", "AD-00003", "AD-00006"})
        self.assertEqual(by_id["AD-00001"]["kind"], "changed")           # unused multi-version review, label changed
        self.assertEqual(by_id["AD-00006"]["kind"], "unchanged")
        single = by_id["AD-00003"]
        self.assertEqual(single["previous"], single["newest"])
        self.assertTrue(single["notes"])
        self.assertTrue(all(i["split"] == "ad" and i["ad_related"] for i in items))

    def test_an_older_version_of_a_review_already_in_dev_or_confirm_is_excluded(self):
        """MedRevQA can hold an old version of a review as an ungrouped row of its own, so excluding by study
        group alone lets a review through that dev or confirm already holds (four such questions were in the
        first build of the set). The Cochrane ID closes that gap; a missing ID cannot exclude anything."""
        rows, groups = self._world()
        rows[9] = medrev_row(9, "Does donepezil improve well-being in mild Alzheimer's disease?", S, "2001 Jan 1")
        rows[9]["DOI_Date"] = rows[9]["DOI_Date"].replace("CD100009", "CD100008")   # same review as row 8
        rows[10] = medrev_row(10, "Does memantine help in dementia?", S, "2002 Jan 1")
        rows[10]["DOI_Date"] = "Cochrane Database Syst Rev. 2002 Jan 1;1:no-id. doi: x"      # a date, no CD id
        before = {i["item_id"] for i in AD.build_ad_items(rows, groups, used_groups={12})}
        self.assertIn("AD-00009", before)                                   # the group check alone keeps it
        main = [dict(item("MC-1"), group_id=12,
                     newest={"cochrane_id": "CD100008", "row": 8}, previous={"cochrane_id": "CD100008", "row": 8})]
        used = AD.used_review_ids(main)
        self.assertEqual(used, frozenset({"CD100008"}))
        after = {i["item_id"] for i in AD.build_ad_items(rows, groups, used_groups={12}, used_reviews=used)}
        self.assertNotIn("AD-00009", after)
        self.assertIn("AD-00010", after)                                    # no ID: nothing to match, kept
        self.assertEqual(after, before - {"AD-00009"})

    def test_expected_item_counts_equal_the_committed_manifests(self):
        """``pipeline.EXPECTED_ITEMS`` is what the preflight of both pipelines demands. It is a hand-typed copy
        of the manifests' counts, and it went stale once (212, against the corrected 208)."""
        base = Path(P.__file__).resolve().parent
        main = json.loads((base / "manifest.json").read_text(encoding="utf-8"))["counts"]
        ad = json.loads((base / "manifest_ad.json").read_text(encoding="utf-8"))
        self.assertEqual(P.EXPECTED_ITEMS["dev"] + P.EXPECTED_ITEMS["confirm"],
                         main["items"] - main["likely_label_noise_excluded"])
        self.assertEqual(P.EXPECTED_ITEMS["ad"], ad["items"])
        self.assertEqual(ad["kinds"], {"unchanged": ad["items"]})          # no changed verdict can be in this set

    def test_cli_appends_once_and_keeps_the_main_benchmark(self):
        rows, groups = self._world()
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "Datasets").mkdir()
            import csv
            with open(d / "Datasets" / "MedRevQA.csv", "w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["", "Question", "objectives", "Label", "PMID", "DOI_Date"])
                w.writeheader()
                for k in sorted(rows):
                    w.writerow(rows[k])
            with open(d / "Datasets" / "AllStudyGroups.csv", "w", newline="", encoding="utf-8") as h:
                w = csv.writer(h)
                w.writerow(["Group_ID", "Study_ID"])
                for gid, keys in groups.items():
                    for j, k in enumerate(keys):
                        w.writerow([gid if j == 0 else "", k])
            write(d / "benchmark.jsonl", [dict(item("MC-1"), group_id=12)])
            args = ["--medchange-dir", tmp, "--data-dir", tmp, "--manifest", str(d / "m.json")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(AD.main(args), 0)
                self.assertEqual(AD.main(args), 0)                        # a rerun replaces, never duplicates
            bench = [json.loads(l) for l in (d / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["split"] for r in bench], ["dev", "ad", "ad", "ad"])
            self.assertEqual(json.loads((d / "m.json").read_text(encoding="utf-8"))["items"], 3)

    def test_requirement_is_read_on_the_ad_split_and_the_phase_is_guarded(self):
        primary = {"diff_a_minus_b": 0.02, "ci95": [-0.01, 0.05], "mcnemar_p": 0.2, "confirmed": False}
        self.assertEqual(A.requirement_reading(primary, "ad"), "met as a point estimate, not confirmed")
        args = mock.Mock(model_path="m.gguf", judge_path=None, n_threads="6", commit=False)
        names = [n for n, _ in RP.ad_plan(args, Path("data"), Path("results"))]
        self.assertIn("B0 and B1 answers (ad)", names)
        self.assertIn("answers (ad): R2 R2C R2V", names)
        self.assertEqual(names[-2:], ["findings (ad)", "environment record"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(RP.main(["ad", "--model-path", "m.gguf", "--dry-run"]), 2)   # no --go


# --------------------------------------------------------------------------------------
# Robustness, traceability and wiring (added by the 2026-10-05 audit)
# --------------------------------------------------------------------------------------

class ContextOverflowTests(unittest.TestCase):
    """A prompt that does not fit the context window must shorten the abstracts, not stop a multi-day run."""

    def _lists(self):
        lists = lists_for()
        for cands in lists.values():
            for c in cands:
                c["abstract"] = "word " * 400
        return lists

    def test_overflow_is_retried_once_with_shortened_abstracts_and_recorded(self):
        calls = []

        def generate(system, user):
            calls.append(len(user))
            if len(user) > 7000:
                raise ValueError("Requested tokens (7000) exceed context window of 6144")
            return "VERDICT: SUPPORTED\nStudy [1] shows benefit."
        lists = self._lists()
        rec = RR.answer_one(item("MC-1"), "R2", lists, scores_for(lists), {}, generate)
        self.assertTrue(rec["context_truncated"])
        self.assertEqual(rec["verdict"], S)
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[1], calls[0])
        self.assertLessEqual(len(RR.shorten_abstracts([{"abstract": "w " * 500}])[0]["abstract"].split()),
                             RR.OVERFLOW_ABSTRACT_WORDS)

    def test_criteria_arms_use_the_same_fallback(self):
        def generate(system, user):
            if len(user) > 9000:
                raise ValueError("Requested tokens (9000) exceed context window of 6144")
            return ("VERDICT: SUPPORTED [1]" if "Draft answer" not in user and "Evidence criteria" not in user
                    else "DIRECT STUDIES: 1\nFINDINGS: x.\nFINAL VERDICT: REFUTED")
        lists = self._lists()
        existing = {("MC-1", "R2"): {"verdict": S, "text": "VERDICT: SUPPORTED [1]"}}
        rec = RR.answer_one(item("MC-1"), "R2V", lists, scores_for(lists), existing, generate)
        self.assertTrue(rec["context_truncated"])
        self.assertEqual(rec["verdict"], F_)

    def test_an_answer_that_fits_carries_no_flag_and_other_errors_are_not_swallowed(self):
        lists = lists_for()
        rec = RR.answer_one(item("MC-1"), "R2", lists, scores_for(lists), {}, fake_generate)
        self.assertNotIn("context_truncated", rec)

        def broken(system, user):
            raise ValueError("something else went wrong")
        with self.assertRaises(ValueError):
            RR.answer_one(item("MC-1"), "R2", lists, scores_for(lists), {}, broken)

    def test_the_analysis_reports_truncated_answers(self):
        items, answers = world(n=6)
        answers[("MC-000", "R2")]["context_truncated"] = True
        rep = A.report(items, answers, "confirm")
        self.assertEqual(rep["context_truncated"], {"R2": 1})
        self.assertIn("shortened", A.to_markdown(rep))


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
                    analysis = A.main(["--split", "dev", "--data-dir", str(data), "--out-dir", str(results),
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
            self.assertEqual(rep["requirement"].split(" (")[0], "dev estimate only")
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


if __name__ == "__main__":
    unittest.main()
