"""The runner of the realigned study: resumable steps, the context-overflow fallback and the arms' prompts (fakes in place of the models)."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import rag2 as R
from experiments.medchange import rag2_run as RR
from tests.fixtures import F_, N, S, FakeScorer, HashingEncoder, LexicalOverlapReranker, abstracts, fake_generate, item, lists_for, records, scores_for


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

    def test_a_judgement_is_valid_only_when_yes_and_no_carry_enough_of_the_probability(self):
        class Fixed:
            def __init__(self, p, mass):
                self.p, self.mass = p, mass

            def score(self, question, study):
                return self.p, self.mass
        cand = {"pmid": "1", "title": "T", "abstract": "A"}
        self.assertTrue(RR.judge_one(Fixed(0.9, R.MIN_YES_NO_MASS), "Q?", cand)["valid"])             # exactly the minimum mass is valid
        self.assertFalse(RR.judge_one(Fixed(0.9, R.MIN_YES_NO_MASS - 0.01), "Q?", cand)["valid"])     # yes + no below it: unusable
        self.assertFalse(RR.judge_one(Fixed(None, 0.0), "Q?", cand)["valid"])

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
        calls = []

        def broken(system, user):
            calls.append(user)
            raise ValueError("something else went wrong")
        with self.assertRaises(ValueError):
            RR.answer_one(item("MC-1"), "R2", lists, scores_for(lists), {}, broken)
        self.assertEqual(len(calls), 1)                              # only a context overflow is retried

    def test_the_undated_ablation_shows_neither_years_nor_the_currency_criterion_and_the_dated_arms_show_both(self):
        lists = lists_for()
        existing = {("MC-1", "R2"): {"verdict": S, "text": "VERDICT: SUPPORTED [1]"}}
        prompts = {}
        for arm in ("R2V", "R2V-ND", "R2C"):
            def capture(system, user, arm=arm):
                prompts[arm] = user
                return "DIRECT STUDIES: 1\nFINDINGS: x.\nFINAL VERDICT: SUPPORTED"
            RR.answer_one(item("MC-1"), arm, lists, scores_for(lists), existing, capture)
        for arm in ("R2V", "R2C"):
            self.assertIn("2012", prompts[arm])
            self.assertIn("newer evidence take precedence", prompts[arm])
        self.assertNotIn("2012", prompts["R2V-ND"])
        self.assertNotIn("newer evidence", prompts["R2V-ND"])
        self.assertIn("Draft answer", prompts["R2V"] + prompts["R2V-ND"])
        self.assertNotIn("Draft answer", prompts["R2C"])


if __name__ == "__main__":
    unittest.main()
