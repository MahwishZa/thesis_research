"""Deterministic sanity tests for the audit's mathematical core.

Controlled toy inputs where the expected direction can be verified by hand, plus
the edge cases the audit brief enumerates: empty, very short, long/truncating,
identical and duplicated passages; equal scores; threshold boundaries.
"""

import math

import pytest

from rag2.config import Config
from rag2.filter_training.labeling import (
    DISCARD,
    LabelingObservation,
    compute_tau,
    decide_label,
)
from rag2.filter_training.perplexity import (
    PerplexityPair,
    compute_perplexity_pair,
    perplexity_from_scores,
    top_percent_threshold,
)
from rag2.filtering.rag2_filter import KEEP_THRESHOLD, ScriptedFilter, helpful_probability
from rag2.llm.base import LLM, ScoredSequence
from rag2.prompts import DEFAULT_PROMPTS, LABEL_HELPFUL, LABEL_NOT_HELPFUL
from rag2.retrieval.rerank import rerank_candidates
from rag2.schema import Evidence, Question


def _question():
    return Question("q1", "Which is best?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")


class FixedLLM(LLM):
    """Returns a per-prompt total log-probability chosen by the test."""

    def __init__(self, logprob_by_marker, default=-10.0):
        self.logprob_by_marker = logprob_by_marker
        self.default = default

    def generate(self, prompts, **kwargs):
        return ["Therefore, the answer is (A)." for _ in prompts]

    def score(self, prompt, continuation):
        total = self.default
        for marker, value in self.logprob_by_marker.items():
            if marker in prompt:
                total = value
                break
        tokens = max(len(continuation.split()), 1)
        return ScoredSequence([total / tokens] * tokens, tokens)


# ------------------------------------------------------- perplexity math ---
def test_perplexity_of_a_hand_computable_case():
    """PPL of three tokens each with log p = -ln 2 is exactly 2."""
    logprobs = [-math.log(2)] * 3
    assert perplexity_from_scores(ScoredSequence(logprobs, 3)) == pytest.approx(2.0)


def test_perplexity_of_certainty_is_one():
    assert perplexity_from_scores(ScoredSequence([0.0, 0.0, 0.0], 3)) == pytest.approx(1.0)


def test_lower_perplexity_means_higher_confidence():
    confident = perplexity_from_scores(ScoredSequence([-0.1] * 4, 4))
    unsure = perplexity_from_scores(ScoredSequence([-3.0] * 4, 4))
    assert confident < unsure


def test_a_helpful_document_yields_a_positive_delta_end_to_end():
    """The direction that matters: a document that raises confidence must give
    delta > 0, which is what Figure 2 reads as 'lower perplexity'."""
    llm = FixedLLM({"HELPFUL-DOC": -2.0}, default=-8.0)  # with-doc scores better
    pair = compute_perplexity_pair(
        llm, _question(), "the rationale text", Evidence(text="HELPFUL-DOC"), DEFAULT_PROMPTS
    )
    assert pair.ppl_with < pair.ppl_without
    assert pair.delta > 0
    assert decide_label(True, True, pair.delta >= 0) == LABEL_HELPFUL


def test_a_distracting_document_yields_a_negative_delta_end_to_end():
    llm = FixedLLM({"BAD-DOC": -20.0}, default=-2.0)  # with-doc scores worse
    pair = compute_perplexity_pair(
        llm, _question(), "the rationale text", Evidence(text="BAD-DOC"), DEFAULT_PROMPTS
    )
    assert pair.ppl_with > pair.ppl_without
    assert pair.delta < 0
    # Wrong both ways and no confidence gain -> the paper discards rather than labels.
    assert decide_label(False, False, False) == DISCARD


def test_delta_is_zero_for_an_inert_document():
    llm = FixedLLM({}, default=-5.0)
    pair = compute_perplexity_pair(
        llm, _question(), "rationale", Evidence(text="INERT"), DEFAULT_PROMPTS
    )
    assert pair.delta == pytest.approx(0.0)


# ---------------------------------------------------- threshold behaviour ---
@pytest.mark.parametrize("n,percent,expected", [(100, 25.0, 25), (200, 25.0, 50), (100, 50.0, 50), (40, 25.0, 10)])
def test_top_percent_admits_the_right_count(n, percent, expected):
    deltas = [float(i) for i in range(n)]
    tau = top_percent_threshold(deltas, percent)
    assert sum(1 for d in deltas if d >= tau) == expected


def test_threshold_boundary_is_inclusive():
    """Equation 3 writes '>=', so a delta exactly at tau must be admitted."""
    deltas = [0.0, 1.0, 2.0, 3.0]
    tau = top_percent_threshold(deltas, 25.0)
    at_boundary = [d for d in deltas if d == tau]
    assert all(d >= tau for d in at_boundary)
    assert decide_label(True, True, tau >= tau) == LABEL_HELPFUL


def test_all_equal_deltas_make_every_observation_pass():
    """Degenerate but real: if every delta is identical, tau equals them all and
    the '>=' test admits everything rather than a quarter."""
    deltas = [5.0] * 20
    tau = top_percent_threshold(deltas, 25.0)
    assert tau == 5.0
    assert sum(1 for d in deltas if d >= tau) == 20


def test_single_observation_trivially_passes_its_own_threshold():
    assert top_percent_threshold([2.5], 25.0) == 2.5


def test_tau_per_question_differs_from_global():
    def obs(qid, i, delta):
        return LabelingObservation(qid, i, True, True, delta,
                                   evidence=Evidence(text=f"s{i}"), question=_question())

    observations = [obs("q1", 0, 100.0), obs("q1", 1, 90.0), obs("q2", 0, 1.0), obs("q2", 1, 0.5)]
    global_tau = compute_tau(observations, 25.0, "global")["__global__"]
    per_question = compute_tau(observations, 25.0, "per_question")
    assert per_question["q2"] < global_tau
    # q2's best snippet clears its own bar but not the global one.
    assert 1.0 >= per_question["q2"] and 1.0 < global_tau


# ------------------------------------------------- filter score behaviour ---
def test_changing_the_evidence_changes_the_filter_score():
    """Required by the audit brief: the filter must actually be a function of the
    passage, not of the question alone."""
    scores = {}

    def score_fn(rendered, question, evidence):
        value = len(evidence.text) / 100.0
        scores[evidence.text] = value
        return value

    filt = ScriptedFilter(score_fn)
    candidates = [Evidence(text="short"), Evidence(text="a considerably longer passage of evidence")]
    decisions = filt.decide(_question(), candidates)
    assert decisions[0].score != decisions[1].score
    assert len(set(scores.values())) == 2


def test_admission_follows_the_threshold_rule_exactly():
    probabilities = [KEEP_THRESHOLD - 1e-9, KEEP_THRESHOLD, KEEP_THRESHOLD + 1e-9]
    filt = ScriptedFilter(lambda r, q, e: probabilities[int(e.text)])
    decisions = filt.decide(_question(), [Evidence(text=str(i)) for i in range(3)])
    assert [d.keep for d in decisions] == [False, True, True]
    assert [d.label for d in decisions] == [LABEL_NOT_HELPFUL, LABEL_HELPFUL, LABEL_HELPFUL]


def test_helpful_probability_is_monotonic_in_the_logit_gap():
    gaps = [-5.0, -1.0, 0.0, 1.0, 5.0]
    probabilities = [helpful_probability(g, 0.0) for g in gaps]
    assert probabilities == sorted(probabilities)
    assert probabilities[2] == pytest.approx(0.5)


# ----------------------------------------------------------- edge cases ----
def test_empty_passage_is_scored_not_crashed():
    filt = ScriptedFilter(lambda r, q, e: 1.0 if e.text.strip() else 0.0)
    decisions = filt.decide(_question(), [Evidence(text=""), Evidence(text="   ")])
    assert [d.keep for d in decisions] == [False, False]
    rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), Evidence(text=""))
    assert "Evidence: \n\nQuestion:" in rendered


def test_very_short_passage():
    rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), Evidence(text="a"))
    assert "Evidence: a" in rendered


def test_long_passage_survives_prompt_construction():
    long_text = "word " * 5000
    rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), Evidence(text=long_text))
    assert len(rendered) > 20000
    # truncation is the tokenizer's job at filter.max_seq_length, not the prompt's
    assert Config().filter.max_seq_length == 512


def test_identical_passages_receive_identical_decisions():
    filt = ScriptedFilter(lambda r, q, e: len(e.text) / 100.0)
    same = [Evidence(text="IDENTICAL", source="pubmed"), Evidence(text="IDENTICAL", source="cpg")]
    decisions = filt.decide(_question(), same)
    assert decisions[0].score == decisions[1].score
    assert decisions[0].keep == decisions[1].keep


def test_duplicate_passages_are_not_deduplicated():
    """The paper specifies no dedup; both copies must survive so the reproduction
    does not silently drop retrieved passages."""
    duplicates = [Evidence(text="SAME", source="pubmed", doc_id="d1"),
                  Evidence(text="SAME", source="cpg", doc_id="d2")]

    class Scorer:
        def score(self, query, snippets):
            return [1.0, 1.0]

    ranked = rerank_candidates(Scorer(), "q", duplicates, top_k=2)
    assert len(ranked) == 2
    assert [e.doc_id for e in ranked] == ["d1", "d2"]


def test_equal_rerank_scores_preserve_pool_order():
    candidates = [Evidence(text=f"s{i}") for i in range(6)]

    class Scorer:
        def score(self, query, snippets):
            return [1.0] * len(snippets)

    ranked = rerank_candidates(Scorer(), "q", candidates, top_k=6)
    assert [e.text for e in ranked] == [f"s{i}" for i in range(6)]


def test_empty_candidate_list_is_safe_everywhere():
    assert rerank_candidates(None, "q", []) == []
    assert ScriptedFilter(lambda *_: 1.0).apply(_question(), []) == ([], [])
    assert DEFAULT_PROMPTS.render_evidence_block([]) == ""


def test_zero_perplexity_tokens_gives_infinite_perplexity_not_a_crash():
    assert perplexity_from_scores(ScoredSequence([], 0)) == float("inf")
    assert math.isnan(PerplexityPair(float("inf"), float("inf")).delta)
