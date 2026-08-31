"""The audit module itself must be trustworthy: its checks have to be able to fail."""

import json
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rag2_audit import checks_filter, checks_pipeline, checks_structure  # noqa: E402,F401
from rag2_audit.registry import CHECKS, Status, run_all  # noqa: E402
from rag2_audit.run import verdict  # noqa: E402


def test_every_check_runs_and_returns_a_result():
    results = run_all()
    assert len(results) == len(CHECKS)
    assert all(isinstance(r.status, Status) for r in results)
    assert all(r.summary for r in results)


def test_non_pass_results_explain_themselves():
    """The brief requires paper/code/impact/fix for every non-PASS finding."""
    for result in run_all():
        if result.status in (Status.PASS, Status.MANUAL):
            continue
        assert result.paper_says, f"{result.check_id} has no paper_says"
        assert result.code_does, f"{result.check_id} has no code_does"
        assert result.why_it_matters, f"{result.check_id} has no why_it_matters"
        assert result.how_to_fix, f"{result.check_id} has no how_to_fix"


def test_check_ids_are_unique_and_namespaced():
    ids = [c.check_id for c in CHECKS]
    assert len(ids) == len(set(ids))
    assert all(i.split("-")[0] in {"STR", "FLT", "PPL", "LBL", "RET", "MOD", "GEN", "EVA", "DET"} for i in ids)


# --- the checks must actually be able to fail -------------------------------
def test_delta_sign_check_would_catch_a_reversed_sign(monkeypatch):
    from rag2.filter_training import perplexity as ppl_module

    class Reversed(ppl_module.PerplexityPair):
        @property
        def delta(self):
            return self.ppl_with - self.ppl_without  # the bug

    monkeypatch.setattr(ppl_module, "PerplexityPair", Reversed)
    result = checks_filter.check_delta_sign()
    assert result.status is Status.FAIL
    assert "sign" in result.summary.lower()


def test_truth_table_check_would_catch_a_wrong_branch(monkeypatch):
    from rag2.filter_training import labeling

    monkeypatch.setattr(labeling, "decide_label", lambda a, b, c: "[HELPFUL]")
    result = checks_filter.check_figure_2_truth_table()
    assert result.status is Status.FAIL
    assert result.evidence["wrong"]


def test_admission_check_would_catch_an_inverted_filter(monkeypatch):
    from rag2.filtering import rag2_filter
    from rag2.schema import FilterDecision

    def inverted(probabilities, threshold=0.5):
        return [
            FilterDecision(keep=p < threshold, label="[NOT_HELPFUL]" if p >= threshold else "[HELPFUL]", score=p)
            for p in probabilities
        ]

    monkeypatch.setattr(rag2_filter, "decisions_from_probabilities", inverted)
    result = checks_filter.check_admission_direction()
    assert result.status is Status.FAIL


def test_threshold_check_would_catch_a_wrong_percentile(monkeypatch):
    from rag2.filter_training import perplexity as ppl_module

    monkeypatch.setattr(
        ppl_module, "top_percent_threshold",
        lambda deltas, top_percent=25.0: ppl_module.percentile(deltas, top_percent),  # 25th not 75th
    )
    result = checks_filter.check_tau_threshold()
    assert result.status is Status.FAIL


def test_leakage_check_would_catch_a_leaking_prompt(monkeypatch):
    from rag2 import prompts as prompts_module

    original = prompts_module.PromptSet.render_filter_prompt

    def leaky(self, question, evidence):
        rendered = original(self, question, evidence)
        source = getattr(evidence, "source", "")
        return f"{rendered}\n[source: {source}]"

    monkeypatch.setattr(prompts_module.PromptSet, "render_filter_prompt", leaky)
    result = checks_filter.check_no_leakage_into_filter()
    assert result.status is Status.FAIL
    assert "pubmed" in result.evidence["leaked"]


def test_balance_check_would_catch_an_unbalanced_pool(monkeypatch):
    from rag2.retrieval import balanced as balanced_module

    original = balanced_module.balanced_retrieve

    def lopsided(corpora, embeddings, config, **kwargs):
        pooled = original(corpora, embeddings, config, **kwargs)
        return [[e for e in candidates if e.source != "cpg"] for candidates in pooled]

    monkeypatch.setattr(balanced_module, "balanced_retrieve", lopsided)
    result = checks_pipeline.check_balanced_retrieval()
    assert result.status is Status.FAIL


# --- verdict logic ----------------------------------------------------------
def test_verdict_scale():
    from rag2_audit.registry import Result

    def make(status, component="misc"):
        return Result("X-01", component, status, "s")

    assert verdict([make(Status.PASS)]) == "VERIFIED"
    assert verdict([make(Status.PASS), make(Status.PARTIAL)]) == "MOSTLY VERIFIED"
    assert verdict([make(Status.PASS), make(Status.UNKNOWN)]) == "MOSTLY VERIFIED"
    assert verdict([make(Status.FAIL, "determinism")]) == "PARTIALLY VERIFIED"
    assert verdict([make(Status.FAIL, "perplexity")]) == "NOT VERIFIED"
    assert verdict([make(Status.FAIL, "filter scoring")]) == "NOT VERIFIED"


def test_cli_runs_and_emits_json(tmp_path):
    out = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, "-m", "rag2_audit.run", "--quiet", "--json", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert "VERDICT:" in result.stdout
    payload = json.loads(out.read_text())
    assert payload["verdict"] in {"VERIFIED", "MOSTLY VERIFIED", "PARTIALLY VERIFIED", "NOT VERIFIED"}
    assert len(payload["results"]) == len(CHECKS)
    # exit code is non-zero iff something FAILed
    assert (result.returncode != 0) == (payload["counts"].get("FAIL", 0) > 0)


def test_trace_script_produces_every_required_section(tmp_path):
    out = tmp_path / "trace.json"
    result = subprocess.run(
        [sys.executable, "-m", "rag2_audit.trace", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    trace = json.loads(out.read_text())
    # the sections the audit brief enumerates
    for section in ("01_question", "02_rationale", "03_retrieval_query", "04_retrieval",
                    "05_reranking", "06_top_k_selection", "07_perplexity", "08_threshold",
                    "10_filtering", "11_generation", "12_provenance_isolation"):
        assert section in trace, section
    assert trace["04_retrieval"]["balanced"] is True
    assert trace["03_retrieval_query"]["is_the_rationale_not_the_question"] is True
    assert trace["12_provenance_isolation"]["any_date_in_model_input"] is False
    assert trace["11_generation"]["extracted_prediction"] in list("ABCD")
    # perplexity rows must carry both terms and their difference
    row = trace["07_perplexity"]["rows"][0]
    assert {"ppl_without_document_PPL_x", "ppl_with_document_PPL_x_d", "delta_ppl"} <= set(row)
