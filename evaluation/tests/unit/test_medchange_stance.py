"""Stage-2 stance step, its pilot checks and the P0 diagnostics (no model, no network)."""

import csv
import json
import math
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import diagnostics as D
from experiments.medchange import stance as ST
from experiments.medchange import stance_check as SC

ABSTRACT = ("BACKGROUND: Cranberries may prevent urinary tract infection. METHODS: We randomised 120 women "
            "(BMI: 25). RESULTS: Infection occurred in 18% versus 32% (RR 0.56, 95% CI 0.35 to 0.89). "
            "CONCLUSIONS: Cranberry juice reduced recurrence.")


def cand(pmid, rank, abstract=ABSTRACT, title="Cranberry for infection.", pubtypes=(), lower="2010-01-01",
         upper="2010-01-01"):
    return {"pmid": pmid, "rank": rank, "title": title, "abstract": abstract, "pubtypes": list(pubtypes),
            "lower": lower, "upper": upper}


class SnippetTests(unittest.TestCase):

    def test_labelled_sections_are_found_and_running_text_acronyms_are_not_boundaries(self):
        sections = ST.split_sections(ABSTRACT)
        self.assertEqual([label for label, _ in sections], ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"])
        self.assertIn("(BMI: 25)", dict(sections)["METHODS"])

    def test_key_text_takes_results_and_conclusions(self):
        results, conclusions = ST.key_text(ABSTRACT)
        self.assertTrue(results.startswith("Infection occurred"))
        self.assertEqual(conclusions, "Cranberry juice reduced recurrence.")

    def test_an_unlabelled_abstract_contributes_its_last_three_sentences(self):
        results, conclusions = ST.key_text("One is first. Two is second. Three is third. Four is fourth.")
        self.assertEqual(results, "")
        self.assertEqual(conclusions, "Two is second. Three is third. Four is fourth.")

    def test_snippet_has_title_results_and_conclusions(self):
        text = ST.study_snippet(cand("1", 1))
        self.assertTrue(text.startswith("Cranberry for infection."))
        self.assertIn("Results: Infection occurred", text)
        self.assertIn("Conclusions: Cranberry juice reduced recurrence.", text)

    def test_snippet_respects_the_word_budget_and_keeps_half_for_conclusions(self):
        long = ("RESULTS: " + " ".join(f"r{i}" for i in range(400)) + ". CONCLUSIONS: "
                + " ".join(f"c{i}" for i in range(400)) + ".")
        text = ST.study_snippet(cand("1", 1, abstract=long), max_words=100)
        self.assertLessEqual(len(text.split()), 100 + 4)
        self.assertGreaterEqual(text.count(" c"), 45)

    def test_a_missing_abstract_does_not_crash(self):
        self.assertEqual(ST.study_snippet({"title": "", "abstract": ""}), "No abstract available.")
        self.assertEqual(ST.study_snippet({"title": "Only a title", "abstract": None}), "Only a title.")


class PromptTests(unittest.TestCase):

    def test_the_study_comes_last_and_the_question_is_stripped(self):
        system, user = ST.build_messages("A", "  Does it work?  ", "STUDY TEXT")
        self.assertIn("Does it work?", user)
        self.assertNotIn("  Does", user)
        self.assertLess(user.index("A = "), user.index("STUDY TEXT"))
        self.assertTrue(user.rstrip().endswith("Answer with a single letter: A, B or C."))
        self.assertTrue(system)

    def test_the_two_wordings_use_different_letter_orders(self):
        self.assertEqual(ST.WORDINGS["A"]["letters"], {"A": "supports", "B": "contradicts", "C": "neither"})
        self.assertEqual(ST.WORDINGS["B"]["letters"], {"A": "neither", "B": "supports", "C": "contradicts"})
        a = ST.build_messages("A", "q", "s")[1]
        b = ST.build_messages("B", "q", "s")[1]
        self.assertNotEqual(a, b)

    def test_letters_map_to_the_canonical_class_order(self):
        letters = {"A": 0.1, "B": 0.2, "C": 0.7}
        self.assertEqual(ST.canonical_probs("A", letters), (0.1, 0.2, 0.7))
        self.assertEqual(ST.canonical_probs("B", letters), (0.2, 0.7, 0.1))

    def test_argmax_resolves_ties_and_invalid_outputs_to_neither(self):
        self.assertEqual(ST.argmax_class(None), "neither")
        self.assertEqual(ST.argmax_class((0.4, 0.4, 0.2)), "supports")
        self.assertEqual(ST.argmax_class((0.4, 0.3, 0.4)), "neither")
        self.assertEqual(ST.argmax_class((0.1, 0.8, 0.1)), "contradicts")


class LetterProbTests(unittest.TestCase):

    def test_letters_are_found_whatever_their_spacing_and_renormalised(self):
        top = [{"token": " A", "logprob": math.log(0.5)}, {"token": "B", "logprob": math.log(0.25)},
               {"token": "The", "logprob": math.log(0.2)}, {"token": "c", "logprob": math.log(0.05)}]
        probs, mass = ST.letter_probs_from_top_logprobs(top)
        self.assertAlmostEqual(mass, 0.8, places=6)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertAlmostEqual(probs["A"], 0.625, places=6)

    def test_no_letter_in_the_top_tokens_is_invalid(self):
        self.assertEqual(ST.letter_probs_from_top_logprobs([{"token": "The", "logprob": -0.1}]), (None, 0.0))


def fake_llama_module(top_by_call):
    """A stand-in for llama_cpp whose chat completion returns the next prepared top-logprobs list."""
    seen = {"init": None, "calls": []}

    class Llama:
        def __init__(self, **kw):
            seen["init"] = kw

        def create_chat_completion(self, messages, max_tokens, temperature, logprobs=None, top_logprobs=None):
            seen["calls"].append({"max_tokens": max_tokens, "temperature": temperature, "logprobs": logprobs,
                                  "top_logprobs": top_logprobs})
            top = top_by_call.pop(0)
            if logprobs:
                return {"choices": [{"logprobs": {"content": [{"token": "x", "logprob": 0.0, "top_logprobs": top}]}}]}
            return {"choices": [{"message": {"content": top}}]}

    module = types.ModuleType("llama_cpp")
    module.Llama = Llama
    return module, seen


class LlamaStanceTests(unittest.TestCase):

    def _model(self, tmp):
        path = Path(tmp) / "m.gguf"
        path.write_bytes(b"fake model")
        return path

    def test_log_probabilities_are_requested_with_logits_all_and_parsed(self):
        module, seen = fake_llama_module([[{"token": "A", "logprob": math.log(0.6)},
                                           {"token": "C", "logprob": math.log(0.3)}]])
        with TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"llama_cpp": module}):
            scorer = ST.LlamaStance(self._model(tmp))
            probs, mass = scorer.score(lambda s: ("sys", f"user {s}"), "study")
        self.assertTrue(seen["init"]["logits_all"])
        self.assertEqual(seen["init"]["n_ctx"], 1536)
        self.assertEqual(seen["calls"][0], {"max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20})
        self.assertAlmostEqual(probs["A"], 2 / 3, places=6)
        self.assertAlmostEqual(mass, 0.9, places=6)
        self.assertTrue(scorer.name.startswith("llama-logprobs:"))

    def test_hard_labels_read_the_plain_letter_and_skip_logits_all(self):
        module, seen = fake_llama_module(["B", "Sorry"])
        with TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"llama_cpp": module}):
            scorer = ST.LlamaStance(self._model(tmp), hard_labels=True)
            first = scorer.score(lambda s: ("sys", s), "study")
            second = scorer.score(lambda s: ("sys", s), "study")
        self.assertFalse(seen["init"]["logits_all"])
        self.assertEqual(first, ({"A": 0.0, "B": 1.0, "C": 0.0}, 1.0))
        self.assertEqual(second, (None, 0.0))


class FitStudyTests(unittest.TestCase):

    def test_the_study_is_shortened_from_the_end_until_the_whole_prompt_fits(self):
        build = lambda s: ("system words", "question words " + s + " answer with a letter")
        count = lambda text: len(text.split())
        fitted = ST.fit_study(build, " ".join(f"w{i}" for i in range(200)), count, limit=60)
        self.assertLessEqual(count("\n".join(build(fitted))), 60 + 1)
        self.assertTrue(fitted.startswith("w0 w1"))
        self.assertGreater(len(fitted.split()), 20)

    def test_a_short_study_is_left_alone(self):
        build = lambda s: ("sys", s)
        self.assertEqual(ST.fit_study(build, "a b c", lambda t: len(t.split()), limit=100), "a b c")


class FakeScorer:
    name = "fake:test"
    model_sha256 = "x" * 64
    n_ctx = 1536
    min_mass = 0.5

    def __init__(self, probs=None):
        self.calls = 0
        self.probs = probs or {"A": 0.7, "B": 0.2, "C": 0.1}

    def score(self, build, study):
        self.calls += 1
        build(study)
        return dict(self.probs), 0.95


def world(n_items=3, n_cands=4):
    items = [{"item_id": f"MC-{i:03d}", "question": f"Question {i}?", "split": "dev", "likely_label_noise": False,
              "kind": "changed", "newest": {"label": "SUPPORTED", "date": "2020-01-01"},
              "previous": {"label": "REFUTED", "date": "2010-01-01"}} for i in range(n_items)]
    pools = {it["item_id"]: {"item_id": it["item_id"],
                             "candidates": [cand(f"{i}{k}", k) for k in range(1, n_cands + 1)]}
             for i, it in enumerate(items)}
    return items, pools


class RunTests(unittest.TestCase):

    def test_scores_the_top_k_in_rank_order_and_resumes_without_rescoring(self):
        items, pools = world(2, 5)
        scorer = FakeScorer()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "stance.jsonl"
            n1 = ST.run(items, pools, scorer, "A", out, top_k=3)
            n2 = ST.run(items, pools, scorer, "A", out, top_k=3)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual((n1, n2, scorer.calls), (6, 0, 6))
        self.assertEqual([r["rank"] for r in rows[:3]], [1, 2, 3])
        self.assertEqual(rows[0]["probs"], [0.7, 0.2, 0.1])
        self.assertEqual(rows[0]["argmax"], "supports")
        self.assertFalse(rows[0]["control"])

    def test_a_second_wording_is_scored_separately(self):
        items, pools = world(1, 2)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "s.jsonl"
            ST.run(items, pools, FakeScorer(), "A", out)
            self.assertEqual(ST.run(items, pools, FakeScorer(), "B", out), 2)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["wording"] for r in rows}, {"A", "B"})
        self.assertEqual([r["probs"] for r in rows if r["wording"] == "B"], [[0.2, 0.1, 0.7]] * 2)

    def test_the_control_judges_another_items_papers_against_this_items_question(self):
        items, pools = world(3, 2)
        control = ST.derangement([it["item_id"] for it in items], seed=1)
        self.assertTrue(all(k != v for k, v in control.items()))
        self.assertEqual(sorted(control.values()), sorted(control))
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "s.jsonl"
            ST.run(items, pools, FakeScorer(), "A", out, control=control)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            self.assertTrue(r["control"])
            self.assertEqual(r["source_item"], control[r["item_id"]])
            self.assertTrue(r["pmid"].startswith(str(int(r["source_item"][-3:]))))

    def test_an_invalid_output_is_recorded_as_neither_with_no_probabilities(self):
        items, pools = world(1, 1)
        scorer = FakeScorer()
        scorer.score = lambda build, study: (None, 0.0)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "s.jsonl"
            ST.run(items, pools, scorer, "A", out)
            row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        self.assertIsNone(row["probs"])
        self.assertEqual(row["argmax"], "neither")

    def test_pilot_items_are_a_seeded_sample(self):
        ids = [f"MC-{i:03d}" for i in range(50)]
        self.assertEqual(ST.pilot_items(ids, 10), ST.pilot_items(list(reversed(ids)), 10))
        self.assertEqual(len(set(ST.pilot_items(ids, 10))), 10)
        self.assertNotEqual(ST.pilot_items(ids, 10), ST.pilot_items(ids, 10, seed=1))


def write_data_dir(tmp, n_items=4, n_cands=3):
    d = Path(tmp) / "data"
    d.mkdir()
    items, pools = world(n_items, n_cands)
    (d / "benchmark.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    (d / "frozen_dev.jsonl").write_text("\n".join(json.dumps(p) for p in pools.values()) + "\n", encoding="utf-8")
    model = Path(tmp) / "m.gguf"
    model.write_bytes(b"x")
    return d, model


class StanceCliTests(unittest.TestCase):

    def _run(self, argv):
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return ST.main(argv)

    def test_the_pilot_scores_both_wordings_plus_the_control_and_binds_its_configuration(self):
        with TemporaryDirectory() as tmp:
            d, model = write_data_dir(tmp)
            fake = FakeScorer()
            with mock.patch.object(ST, "LlamaStance", lambda *a, **k: fake):
                code = self._run(["--split", "dev", "--pilot", "--pilot-n", "2", "--model-path", str(model),
                                  "--data-dir", str(d)])
                rows = [json.loads(l) for l in (d / "stance_pilot.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(code, 0)
                self.assertEqual(len(rows), 2 * 3 * 2 + 2 * 3)
                self.assertEqual(sum(r["control"] for r in rows), 6)
                self.assertTrue((d / "stance_pilot.config.json").exists())
                other = FakeScorer()
                other.name = "fake:other"
                with mock.patch.object(ST, "LlamaStance", lambda *a, **k: other):
                    again = self._run(["--split", "dev", "--pilot", "--pilot-n", "2", "--model-path", str(model),
                                       "--data-dir", str(d)])
                self.assertEqual(again, 2)

    def test_a_full_run_needs_a_chosen_wording_and_then_uses_it(self):
        with TemporaryDirectory() as tmp:
            d, model = write_data_dir(tmp, 2, 2)
            with mock.patch.object(ST, "LlamaStance", lambda *a, **k: FakeScorer()):
                self.assertEqual(self._run(["--split", "dev", "--model-path", str(model), "--data-dir", str(d)]), 2)
                (d / "stance_choice.json").write_text(json.dumps({"wording": "B"}), encoding="utf-8")
                self.assertEqual(self._run(["--split", "dev", "--model-path", str(model), "--data-dir", str(d)]), 0)
            rows = [json.loads(l) for l in (d / "stance_dev.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["wording"] for r in rows}, {"B"})
        self.assertEqual(len(rows), 4)

    def test_a_missing_model_file_is_reported(self):
        with TemporaryDirectory() as tmp:
            d, _ = write_data_dir(tmp)
            self.assertEqual(self._run(["--split", "dev", "--wording", "A", "--model-path",
                                        str(Path(tmp) / "absent.gguf"), "--data-dir", str(d)]), 2)

    def test_the_pilot_is_dev_only(self):
        with TemporaryDirectory() as tmp:
            d, model = write_data_dir(tmp)
            self.assertEqual(self._run(["--split", "confirm", "--pilot", "--model-path", str(model),
                                        "--data-dir", str(d)]), 2)


def pilot_records(agree=True, control_neither=1.0, backend="llama-logprobs:m.gguf", seconds=5.0, n=10):
    rows = []
    for k in range(n):
        for wording in ("A", "B"):
            cls = "supports" if (agree or wording == "A") else "contradicts"
            probs = {"supports": [0.8, 0.1, 0.1], "contradicts": [0.1, 0.8, 0.1]}[cls]
            rows.append({"item_id": f"MC-{k // 5}", "pmid": f"p{k}", "wording": wording, "control": False,
                         "backend": backend, "probs": probs, "argmax": cls, "seconds": seconds})
    for k in range(n):
        neither = k < int(control_neither * n)
        rows.append({"item_id": f"MC-{k // 5}", "pmid": f"c{k}", "wording": "A", "control": True,
                     "source_item": "MC-9", "backend": backend,
                     "probs": [0.1, 0.1, 0.8] if neither else [0.8, 0.1, 0.1],
                     "argmax": "neither" if neither else "supports", "seconds": seconds})
    return rows


class PilotReportTests(unittest.TestCase):

    def test_report_measures_agreement_control_validity_and_speed(self):
        rep = SC.pilot_report(pilot_records())
        self.assertEqual(rep["wording_agreement"], 1.0)
        self.assertEqual(rep["control_neither_share"], 1.0)
        self.assertEqual(rep["invalid_rate"], 0.0)
        self.assertEqual(rep["seconds_per_paper_mean"], 5.0)
        self.assertGreater(rep["mean_side_strength_real"], rep["mean_side_strength_control"])

    def test_gate_is_incomplete_until_the_hand_check_is_scored_then_passes_or_fails(self):
        rep = SC.pilot_report(pilot_records())
        self.assertEqual(SC.gate1(rep, None)["gate1"], "INCOMPLETE")
        self.assertEqual(SC.gate1(rep, {"chosen_accuracy": 0.75})["gate1"], "PASS")
        self.assertEqual(SC.gate1(rep, {"chosen_accuracy": 0.65})["gate1"], "FAIL")

    def test_each_pre_stated_threshold_can_fail_the_gate(self):
        ok = {"chosen_accuracy": 0.9}
        disagree = SC.pilot_report(pilot_records(agree=False))
        self.assertEqual(SC.gate1(disagree, ok)["gate1"], "FAIL")
        leaky = SC.pilot_report(pilot_records(control_neither=0.5))
        self.assertEqual(SC.gate1(leaky, ok)["gate1"], "FAIL")
        slow = SC.pilot_report(pilot_records(seconds=11.0))
        self.assertEqual(SC.gate1(slow, ok)["gate1"], "FAIL")
        flan_fast = SC.pilot_report(pilot_records(backend="flan:google/flan-t5-large", seconds=3.0))
        self.assertEqual(SC.gate1(flan_fast, ok)["gate1"], "PASS")
        flan_slow = SC.pilot_report(pilot_records(backend="flan:google/flan-t5-large", seconds=5.0))
        self.assertEqual(SC.gate1(flan_slow, ok)["gate1"], "FAIL")
        broken = pilot_records()
        for r in broken[:3]:
            r["probs"], r["argmax"] = None, "neither"
        self.assertEqual(SC.gate1(SC.pilot_report(broken), ok)["gate1"], "FAIL")


class HandCheckTests(unittest.TestCase):

    def test_export_hides_the_model_answer_and_score_compares_both_wordings(self):
        items = {"MC-0": {"question": "Does it work?"}, "MC-1": {"question": "Is it safe?"}}
        pools = {k: {"candidates": [cand(f"p{i}", 1) for i in range(10)]} for k in items}
        records = [r for r in pilot_records(n=10) if not r.get("control")]
        with TemporaryDirectory() as tmp:
            sheet, key = Path(tmp) / "sheet.csv", Path(tmp) / "key.json"
            n = SC.export_handcheck(records, pools, items, 6, 3, sheet, key)
            text = sheet.read_text(encoding="utf-8-sig")
            self.assertEqual(n, 6)
            self.assertNotIn("contradicts", text.lower())
            self.assertNotIn("argmax", text.lower())
            with open(sheet, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), ["row", "question", "study_text", "your_label"])
            keys = json.loads(key.read_text(encoding="utf-8"))
            for row in rows:                       # the researcher agrees with wording A everywhere
                row["your_label"] = "s"
            result = SC.score_handcheck(rows, keys)
        self.assertEqual(result["n_checked"], 6)
        self.assertEqual(result["accuracy"]["A"], 1.0)
        self.assertEqual(result["accuracy"]["B"], 1.0)
        self.assertEqual(result["chosen_wording"], "A")

    def test_wording_b_is_chosen_only_when_it_scores_strictly_higher(self):
        keys = {"1": {"A": "supports", "B": "contradicts"}, "2": {"A": "neither", "B": "contradicts"}}
        rows = [{"row": "1", "your_label": "C"}, {"row": "2", "your_label": "C"}]
        result = SC.score_handcheck(rows, keys)
        self.assertEqual(result["chosen_wording"], "B")
        self.assertEqual(result["chosen_accuracy"], 1.0)

    def test_an_unlabelled_or_invalid_row_is_refused_with_its_row_number(self):
        with self.assertRaisesRegex(ValueError, "row 2"):
            SC.score_handcheck([{"row": "1", "your_label": "S"}, {"row": "2", "your_label": ""}],
                               {"1": {"A": "supports"}, "2": {"A": "supports"}})

    def test_cli_report_export_and_score_round_trip(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            items = [{"item_id": f"MC-{i}", "question": f"Q{i}?", "split": "dev", "likely_label_noise": False,
                      "kind": "changed", "newest": {"label": "SUPPORTED", "date": "2020-01-01"}} for i in range(2)]
            pools = [{"item_id": f"MC-{i}", "candidates": [cand(f"p{k}", 1) for k in range(10)]} for i in range(2)]
            (d / "benchmark.jsonl").write_text("\n".join(map(json.dumps, items)) + "\n", encoding="utf-8")
            (d / "frozen_dev.jsonl").write_text("\n".join(map(json.dumps, pools)) + "\n", encoding="utf-8")
            (d / "stance_pilot.jsonl").write_text("\n".join(map(json.dumps, pilot_records(n=10))) + "\n", encoding="utf-8")
            with mock.patch("sys.stdout"):
                self.assertEqual(SC.main(["report", "--data-dir", str(d)]), 0)
                self.assertEqual(SC.main(["export", "--data-dir", str(d), "--n", "8"]), 0)
            with open(d / "stance_handcheck.csv", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row["your_label"] = "S"
            with open(d / "stance_handcheck.csv", "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            with mock.patch("sys.stdout"):
                self.assertEqual(SC.main(["score", "--data-dir", str(d)]), 0)
            choice = json.loads((d / "stance_choice.json").read_text(encoding="utf-8"))
        self.assertEqual(choice["wording"], "A")
        self.assertEqual(choice["gate1"], "PASS")


class DiagnosticsTests(unittest.TestCase):

    def setUp(self):
        long_abs = "RESULTS: " + " ".join(["word"] * 400) + ". CONCLUSIONS: It worked."
        self.items = {"MC-1": {"item_id": "MC-1", "question": "Does it work?", "kind": "changed",
                               "newest": {"label": "SUPPORTED", "date": "2020-01-01"}},
                      "MC-2": {"item_id": "MC-2", "question": "Is it safe?", "kind": "unchanged",
                               "newest": {"label": "REFUTED", "date": "2020-01-01"}}}
        self.pools = {
            "MC-1": {"candidates": [cand("1", 1, abstract=long_abs, pubtypes=["Meta-Analysis"]),
                                    cand("2", 2, pubtypes=["Randomized Controlled Trial"]),
                                    cand("3", 3, abstract="Plain unlabelled abstract. It says things.")]},
            "MC-2": {"candidates": [cand("4", 1, pubtypes=["Case Reports"]), cand("5", 2, abstract=long_abs)]}}
        self.answers = {("MC-1", "B1"): {"verdict": "NOT ENOUGH INFORMATION", "admitted": ["1", "2"]},
                        ("MC-2", "B1"): {"verdict": "SUPPORTED", "admitted": ["4"]},
                        ("MC-1", "B2"): {"verdict": "SUPPORTED", "admitted": ["1", "3"]}}

    def test_helpfulness_inputs_over_the_limit_are_counted_overall_and_among_b2_papers(self):
        rep = D.helpfulness_truncation(self.items, self.pools, self.answers, lambda t: len(t.split()), limit=300)
        self.assertEqual(rep["inputs"], 5)
        self.assertEqual(rep["over_limit"], 2)                    # the two 400-word abstracts
        self.assertEqual(rep["b2_admitted_papers"], 2)
        self.assertEqual(rep["b2_admitted_over_limit"], 1)
        self.assertEqual(rep["b2_admitted_share_over_limit"], 0.5)

    def test_snippet_statistics_count_labelled_sections(self):
        rep = D.snippet_stats(self.items, self.pools)
        self.assertEqual(rep["candidates"], 5)
        self.assertEqual(rep["with_results_or_conclusions"], 0.8)        # all but the unlabelled one
        self.assertLessEqual(rep["snippet_words_max"], ST.MAX_SNIPPET_WORDS + 6)

    def test_study_types_and_b1_composition_groups(self):
        rep = D.study_type_stats(self.items, self.pools, self.answers)
        self.assertEqual(rep["items"], 2)
        self.assertEqual(rep["items_with_sr_ma_in_top8"], 0.5)
        self.assertEqual(rep["items_with_sr_ma_in_b1_admitted"], 0.5)
        self.assertTrue(rep["study_type_weight_has_material"])
        self.assertEqual(rep["b1_by_sr_ma"]["with_sr_ma_in_b1"]["b1_verdicts"], {"NOT ENOUGH INFORMATION": 1})
        self.assertEqual(rep["b1_by_sr_ma"]["without_sr_ma_in_b1"]["gold"], {"REFUTED": 1})

    def test_cli_without_a_tokenizer_writes_both_files(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "data"
            d.mkdir()
            bench = [dict(it, split="dev", likely_label_noise=False) for it in self.items.values()]
            answers = [dict(r, item_id=k[0], arm=k[1]) for k, r in self.answers.items()]
            for name, rows in (("benchmark.jsonl", bench), ("frozen_dev.jsonl",
                                [dict(p, item_id=i) for i, p in self.pools.items()]), ("answers_dev.jsonl", answers)):
                (d / name).write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            with mock.patch("sys.stdout"):
                code = D.main(["--data-dir", str(d), "--no-tokenizer", "--out-dir", str(Path(tmp) / "out")])
            md = (Path(tmp) / "out" / "diagnostics_dev.md").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertIn("Skipped (no tokenizer)", md)
        self.assertIn("Study types", md)


if __name__ == "__main__":
    unittest.main()
