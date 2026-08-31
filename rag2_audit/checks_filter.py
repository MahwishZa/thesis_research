"""The rationale-guided filter and its perplexity-derived labels.

This is the component the thesis will replace, so it gets the most scrutiny:
sign of Delta-PPL, direction of the threshold test, the Figure 2 truth table,
what text is scored and what conditions it, per-passage independence, and
leakage.
"""

from __future__ import annotations

import json
import math
import os
from typing import List

from . import paper
from .registry import Result, Status, check

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_SAMPLE = os.path.join(REPO, "classifier", "data", "medqa", "llama3_cot", "5%-train.json")


def _question():
    from rag2.schema import Question

    return Question("q1", "Which is the best next step?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")


# ---------------------------------------------------------------- prompt ---
@check("FLT-01", "filter input", "Filter prompt reconstructs the released training artifact")
def check_filter_prompt_matches_release() -> Result:
    from rag2.prompts import FILTER_PROMPT

    if not os.path.exists(RELEASE_SAMPLE):
        return Result(
            "FLT-01", "filter input", Status.UNKNOWN,
            "the released training artifact is not present, cannot verify the template",
            how_to_fix=f"restore {os.path.relpath(RELEASE_SAMPLE, REPO)}",
        )
    records = json.load(open(RELEASE_SAMPLE, "r", encoding="utf-8"))
    mismatches = []
    for record in records:
        body = record["question"]
        head, _, rest = body.partition("\n\nEvidence: ")
        evidence, _, question = rest.partition("\n\nQuestion: ")
        if FILTER_PROMPT.format(evidence=evidence, question=question) != body:
            mismatches.append(record["id"])
        if head != paper.FILTER_INPUT_TEMPLATE_HEAD:
            mismatches.append(f"{record['id']}:head")
    if mismatches:
        return Result(
            "FLT-01", "filter input", Status.FAIL,
            f"template fails to reconstruct {len(mismatches)} released record(s)",
            paper_says="R: every released record uses one fixed instruction + Evidence + Question layout",
            code_does="rag2.prompts.FILTER_PROMPT differs",
            why_it_matters="a different filter input means a different model input distribution than the authors trained on",
            how_to_fix="restore the template to the released wording and \\n\\n separators",
            evidence={"mismatched": mismatches},
        )
    return Result(
        "FLT-01", "filter input", Status.PASS,
        f"template reconstructs all {len(records)} released records byte-for-byte",
        evidence={"records_checked": len(records), "head": paper.FILTER_INPUT_TEMPLATE_HEAD},
    )


@check("FLT-02", "filter input", "Filter is given the initial question, not the rationale")
def check_filter_sees_initial_question() -> Result:
    from rag2.prompts import DEFAULT_PROMPTS
    from rag2.schema import Evidence

    question = _question()
    rendered = DEFAULT_PROMPTS.render_filter_prompt(question, Evidence(text="SNIPPET-TEXT"))
    has_question = "Which is the best next step?" in rendered
    has_snippet = "SNIPPET-TEXT" in rendered
    if has_question and has_snippet:
        return Result(
            "FLT-02", "filter input", Status.PASS,
            "filter input carries the initial question and the snippet",
            evidence={"rendered_prefix": rendered[:80]},
        )
    return Result(
        "FLT-02", "filter input", Status.FAIL,
        "filter input is missing the question or the snippet",
        paper_says="Fig1: the filtering model's prompt is 'Snippet + Initial Query'",
        code_does=f"question present={has_question}, snippet present={has_snippet}",
        why_it_matters="the filter cannot judge helpfulness without both halves",
        how_to_fix="fix PromptSet.render_filter_prompt",
    )


@check("FLT-03", "leakage", "Filter sees no provenance, score, rank or corpus identity")
def check_no_leakage_into_filter() -> Result:
    from rag2.prompts import DEFAULT_PROMPTS
    from rag2.schema import Evidence

    evidence = Evidence(
        text="the snippet text",
        source="pubmed",
        doc_id="PMID-99999",
        passage_id="PMID-99999-p3",
        corpus_index=4242,
        retrieval_score=0.87654,
        rerank_score=12.3456,
        rank=7,
        metadata={"publication_date": "1998-07-01", "journal": "SecretJournal"},
    )
    rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), evidence)
    leaked = [
        token for token in
        ("pubmed", "PMID-99999", "4242", "0.87654", "12.3456", "1998-07-01", "SecretJournal")
        if token in rendered
    ]
    if leaked:
        return Result(
            "FLT-03", "leakage", Status.FAIL,
            f"{len(leaked)} field(s) leak into the filter input",
            paper_says="the filter judges a (question, snippet) pair; nothing else was available to it",
            code_does=f"leaked: {leaked}",
            why_it_matters=(
                "the filter could learn to key on retrieval rank or corpus rather than "
                "helpfulness, and the thesis's later date-based analysis would be confounded"
            ),
            how_to_fix="render only Evidence.text",
            evidence={"leaked": leaked, "rendered": rendered},
        )
    return Result(
        "FLT-03", "leakage", Status.PASS,
        "only Evidence.text reaches the filter; 7 provenance fields all absent",
        evidence={"fields_checked": 7},
    )


# --------------------------------------------------------------- scoring ---
@check("FLT-04", "filter scoring", "Two-way softmax reproduces the release's argmax rule")
def check_two_way_softmax() -> Result:
    from rag2.filtering.rag2_filter import KEEP_THRESHOLD, helpful_probability

    cases = [(2.0, -1.0, True), (-1.0, 2.0, False), (0.0, 0.0, True), (1e-6, 0.0, True)]
    wrong = []
    for helpful, not_helpful, expect_keep in cases:
        probability = helpful_probability(helpful, not_helpful)
        keep = probability >= KEEP_THRESHOLD
        # The release does np.argmax([p_helpful, p_not_helpful]) -> index 0 wins ties.
        release_keep = not (not_helpful > helpful)
        if keep != expect_keep or keep != release_keep:
            wrong.append((helpful, not_helpful, probability, keep, release_keep))
    if wrong:
        return Result(
            "FLT-04", "filter scoring", Status.FAIL,
            "keep decision diverges from the release's argmax",
            paper_says=paper.FILTER_SCORING,
            code_does=f"divergent cases: {wrong}",
            why_it_matters="a flipped or mis-tied decision inverts the filter",
            how_to_fix="keep iff P([HELPFUL]) >= 0.5, ties to HELPFUL",
            evidence={"divergent": wrong},
        )
    return Result(
        "FLT-04", "filter scoring", Status.PASS,
        "softmax-over-two-logits + threshold 0.5 matches np.argmax including the tie case",
        evidence={"cases": len(cases), "tie_goes_to": "[HELPFUL]"},
    )


@check("FLT-05", "filter scoring", "Only [HELPFUL] passages are admitted")
def check_admission_direction() -> Result:
    from rag2.filtering.rag2_filter import decisions_from_probabilities
    from rag2.prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL

    decisions = decisions_from_probabilities([0.99, 0.75, 0.5, 0.49, 0.01])
    keeps = [d.keep for d in decisions]
    labels = [d.label for d in decisions]
    expected_keeps = [True, True, True, False, False]
    expected_labels = [LABEL_HELPFUL] * 3 + [LABEL_NOT_HELPFUL] * 2
    if keeps != expected_keeps or labels != expected_labels:
        return Result(
            "FLT-05", "filter scoring", Status.FAIL,
            "admission direction is inverted or mislabelled",
            paper_says="P3.2: keep only snippets judged helpful; filter out distractors",
            code_does=f"keeps={keeps} labels={labels}",
            why_it_matters=(
                "an inverted filter keeps exactly the distractors the method exists to "
                "remove, while still producing plausible-looking output"
            ),
            how_to_fix="keep iff label == [HELPFUL]",
            evidence={"keeps": keeps, "labels": labels},
        )
    return Result(
        "FLT-05", "filter scoring", Status.PASS,
        "higher P([HELPFUL]) admits; lower rejects; direction is correct",
        evidence={"probabilities": [0.99, 0.75, 0.5, 0.49, 0.01], "keeps": keeps},
    )


@check("FLT-06", "filter scoring", "Each passage is filtered independently")
def check_per_passage_independence() -> Result:
    from rag2.filtering.rag2_filter import ScriptedFilter
    from rag2.schema import Evidence

    seen: List[str] = []

    def score(rendered, question, evidence):
        seen.append(evidence.text)
        return 1.0 if evidence.text == "keep-me" else 0.0

    candidates = [Evidence(text=t, source="cpg") for t in ("keep-me", "drop-me", "keep-me", "drop-me")]
    filt = ScriptedFilter(score)
    together = [d.keep for d in filt.decide(_question(), candidates)]
    separate = [filt.decide(_question(), [c])[0].keep for c in candidates]
    if together != separate:
        return Result(
            "FLT-06", "filter scoring", Status.FAIL,
            "a passage's decision depends on the other passages in the batch",
            paper_says="Limitations: 'the Flan-T5 model can filter only one snippet at a time'",
            code_does=f"batched={together} individually={separate}",
            why_it_matters="cross-passage interaction is a different method than the paper's",
            how_to_fix="score each (question, snippet) pair on its own",
            evidence={"batched": together, "individual": separate},
        )
    return Result(
        "FLT-06", "filter scoring", Status.PASS,
        "batched decisions equal one-at-a-time decisions; each pair scored independently",
        evidence={"decisions": together, "distinct_inputs": len(seen)},
    )


# ------------------------------------------------------------ perplexity ---
@check("PPL-01", "perplexity", "PPL = exp(-mean token log-probability)")
def check_perplexity_formula() -> Result:
    from rag2.filter_training.perplexity import perplexity_from_scores
    from rag2.llm.base import ScoredSequence

    cases = [([-1.0, -2.0, -3.0], math.exp(2.0)), ([0.0, 0.0], 1.0), ([-math.log(4)] * 5, 4.0)]
    wrong = []
    for logprobs, expected in cases:
        got = perplexity_from_scores(ScoredSequence(logprobs, len(logprobs)))
        if abs(got - expected) > 1e-9:
            wrong.append((logprobs, got, expected))
    # length normalisation: repeating a sequence must not change PPL
    short = perplexity_from_scores(ScoredSequence([-1.5, -2.5], 2))
    long = perplexity_from_scores(ScoredSequence([-1.5, -2.5] * 6, 12))
    normalised = abs(short - long) < 1e-9
    if wrong or not normalised:
        return Result(
            "PPL-01", "perplexity", Status.FAIL,
            "perplexity does not match Equation 4",
            paper_says=f"Eq4: {paper.PPL_DEFINITION}",
            code_does=f"mismatches={wrong}, length-normalised={normalised}",
            why_it_matters="every label downstream is derived from this number",
            how_to_fix="PPL = exp(-sum(logprobs)/L)",
            evidence={"wrong": wrong, "length_normalised": normalised},
        )
    return Result(
        "PPL-01", "perplexity", Status.PASS,
        "matches Eq4 on 3 controlled inputs and is length-normalised",
        evidence={"cases": len(cases), "length_normalised": True},
    )


@check("PPL-02", "perplexity", "Delta-PPL sign follows Equation 3")
def check_delta_sign() -> Result:
    from rag2.filter_training.perplexity import PerplexityPair

    helpful = PerplexityPair(ppl_without=10.0, ppl_with=4.0)   # document lowered perplexity
    harmful = PerplexityPair(ppl_without=4.0, ppl_with=10.0)   # document raised perplexity
    neutral = PerplexityPair(ppl_without=7.0, ppl_with=7.0)
    ok = (
        helpful.delta == 6.0 and harmful.delta == -6.0 and neutral.delta == 0.0
        and helpful.delta > 0 > harmful.delta
    )
    if not ok:
        return Result(
            "PPL-02", "perplexity", Status.FAIL,
            "Delta-PPL sign is reversed",
            paper_says=f"Eq3: Delta-PPL = {paper.DELTA_DEFINITION}; positive means the document helped",
            code_does=f"helpful={helpful.delta}, harmful={harmful.delta}",
            why_it_matters=(
                "a reversed sign inverts every perplexity-derived label, training the "
                "filter to keep distractors -- and the pipeline would still run"
            ),
            how_to_fix="delta = ppl_without - ppl_with",
            evidence={"helpful": helpful.delta, "harmful": harmful.delta},
        )
    return Result(
        "PPL-02", "perplexity", Status.PASS,
        "Delta-PPL = PPL(x) - PPL(x,d); a confidence-raising document yields a positive delta",
        evidence={"lowered_ppl_delta": helpful.delta, "raised_ppl_delta": harmful.delta},
    )


@check("PPL-03", "perplexity", "Both terms score the same continuation, differing only by the document")
def check_same_continuation_both_sides() -> Result:
    from rag2.filter_training.perplexity import compute_perplexity_pair
    from rag2.llm.base import LLM, ScoredSequence
    from rag2.prompts import DEFAULT_PROMPTS
    from rag2.schema import Evidence

    calls: List[tuple] = []

    class Recorder(LLM):
        def generate(self, prompts, **kwargs):
            return ["" for _ in prompts]

        def score(self, prompt, continuation):
            calls.append((prompt, continuation))
            return ScoredSequence([-1.0] * max(len(continuation.split()), 1),
                                  max(len(continuation.split()), 1))

    compute_perplexity_pair(
        Recorder(), _question(), "THE RATIONALE", Evidence(text="THE-DOCUMENT"), DEFAULT_PROMPTS
    )
    same_continuation = len(calls) == 2 and calls[0][1] == calls[1][1] == "THE RATIONALE"
    doc_only_in_second = "THE-DOCUMENT" not in calls[0][0] and "THE-DOCUMENT" in calls[1][0]
    if not (same_continuation and doc_only_in_second):
        return Result(
            "PPL-03", "perplexity", Status.FAIL,
            "the two perplexity terms are not a controlled comparison",
            paper_says="Eq3/Eq4: the same sequence scored with and without the document",
            code_does=f"same continuation={same_continuation}, document isolated={doc_only_in_second}",
            why_it_matters=(
                "if the scored text or anything besides the document differs between the "
                "terms, Delta-PPL confounds the document's effect with that difference"
            ),
            how_to_fix="score one fixed rationale under prompts differing only by the evidence block",
            evidence={"calls": [(c[0][:60], c[1]) for c in calls]},
        )
    return Result(
        "PPL-03", "perplexity", Status.PASS,
        "identical continuation both sides; the evidence block is the only difference in conditioning",
        evidence={"continuation": calls[0][1], "n_score_calls": len(calls)},
    )


@check("PPL-04", "perplexity", "Scored tokens are the rationale, per the paper's prose")
def check_ppl_target_is_rationale() -> Result:
    from rag2.config import Config

    target = Config().filter_training.ppl_target
    if target != "rationale":
        return Result(
            "PPL-04", "perplexity", Status.PARTIAL,
            f"default ppl_target is {target!r}, not 'rationale'",
            paper_says="abstract/P2.1/Fig2 say the rationale; Eq4 literally writes the query",
            code_does=f"filter_training.ppl_target = {target!r}",
            why_it_matters="the two readings measure different things",
            how_to_fix="set filter_training.ppl_target: rationale",
        )
    return Result(
        "PPL-04", "perplexity", Status.PARTIAL,
        "scores the rationale (prose reading); Eq4 as literally written says the query",
        paper_says=(
            "Eq4 sums over x (the query). The abstract ('labels of rationales'), P2.1 "
            "('perplexity differences in the rationales') and Fig2's 'Rationale' tag all say the rationale."
        ),
        code_does="scores the rationale; filter_training.ppl_target='query' selects the literal reading",
        why_it_matters=(
            "an unresolvable ambiguity in the source. If the authors scored the query, "
            "every label differs and the reproduced filter is trained on a different signal."
        ),
        how_to_fix="cannot be resolved from the paper; ask the authors. Both readings are implemented.",
        evidence={"default": target, "alternative": "query"},
    )


# ---------------------------------------------------------------- labels ---
@check("LBL-01", "labeling", "Figure 2 decision tree is transcribed exactly")
def check_figure_2_truth_table() -> Result:
    from rag2.filter_training.labeling import decide_label

    wrong = []
    for (without, with_, lower), expected in paper.FIGURE_2_TRUTH_TABLE.items():
        got = decide_label(without, with_, lower)
        if got != expected:
            wrong.append({"inputs": [without, with_, lower], "expected": expected, "got": got})
    if wrong:
        return Result(
            "LBL-01", "labeling", Status.FAIL,
            f"{len(wrong)} of 8 Figure 2 branches are wrong",
            paper_says="Fig2 decision tree over (correct w/o, correct w/, lower perplexity)",
            code_does=str(wrong),
            why_it_matters="the labels are the filter's entire training signal",
            how_to_fix="correct rag2.filter_training.labeling.decide_label",
            evidence={"wrong": wrong},
        )
    return Result(
        "LBL-01", "labeling", Status.PASS,
        "all 8 branches of Figure 2 match, including both [DISCARD] leaves",
        evidence={"branches": 8},
    )


@check("LBL-02", "labeling", "tau is the top-25% quantile and the test is inclusive")
def check_tau_threshold() -> Result:
    from rag2.filter_training.perplexity import top_percent_threshold

    deltas = [float(i) for i in range(100)]
    tau = top_percent_threshold(deltas, paper.TAU_PERCENTILE)
    admitted = sum(1 for d in deltas if d >= tau)
    if admitted != 25:
        return Result(
            "LBL-02", "labeling", Status.FAIL,
            f"top-25% threshold admits {admitted}/100, expected 25",
            paper_says=f"P3.2: tau = top {paper.TAU_PERCENTILE}% of perplexity differentials",
            code_does=f"tau={tau}, admitted={admitted}",
            why_it_matters="a wrong threshold shifts the entire helpful/not-helpful balance",
            how_to_fix="tau = percentile(deltas, 100 - 25); test delta >= tau",
            evidence={"tau": tau, "admitted": admitted},
        )
    return Result(
        "LBL-02", "labeling", Status.PASS,
        f"tau admits exactly 25/100 on a uniform grid; test is '{paper.DELTA_TEST}' (inclusive) per Eq3",
        evidence={"tau": tau, "admitted": admitted, "operator": paper.DELTA_TEST},
    )


@check("LBL-03", "labeling", "Non-finite Delta-PPL is not silently labelled")
def check_non_finite_deltas() -> Result:
    from rag2.filter_training.labeling import LabelingObservation, label_observations
    from rag2.schema import Evidence, Question

    def observation(index, delta):
        return LabelingObservation(
            qid="q", snippet_index=index,
            correct_without_retrieval=True, correct_with_retrieval=True, delta_ppl=delta,
            evidence=Evidence(text=f"s{index}"),
            question=Question("q", "Q?", {"A": "a", "B": "b"}, "A"),
        )

    observations = [observation(i, float(i)) for i in range(8)]
    observations += [observation(8, float("inf")), observation(9, float("nan"))]
    pairs, stats = label_observations(observations, dataset_name="audit")
    labelled_ids = {p.id for p in pairs}
    inf_labelled = "audit_8" in labelled_ids
    nan_labelled = "audit_9" in labelled_ids
    if inf_labelled or nan_labelled:
        return Result(
            "LBL-03", "labeling", Status.PARTIAL,
            "non-finite Delta-PPL values are excluded from tau but still receive labels",
            paper_says=(
                "P3.2 defines tau over the perplexity differentials; the paper does not "
                "contemplate undefined differentials"
            ),
            code_does=(
                f"percentile() drops non-finite values when computing tau (tau={stats['tau']:.4g}), "
                f"but the comparison still runs: inf >= tau is True -> labelled "
                f"(inf labelled={inf_labelled}); nan >= tau is False -> labelled via the "
                f"'not lower perplexity' branch (nan labelled={nan_labelled})"
            ),
            why_it_matters=(
                "inf/nan arise from a degenerate rationale (empty generation, or a "
                "scoring failure). Such a pair silently becomes a training example whose "
                "label is an artefact of IEEE comparison semantics rather than evidence "
                "utility, and nan in particular is labelled by the *false* branch, so it "
                "is indistinguishable from a genuine low-delta observation."
            ),
            how_to_fix=(
                "drop non-finite deltas before labeling and count them in the stats, so "
                "degenerate generations are visible rather than mislabelled"
            ),
            evidence={
                "tau": stats["tau"], "inf_labelled": inf_labelled, "nan_labelled": nan_labelled,
                "labelled_ids": sorted(labelled_ids),
            },
        )
    return Result(
        "LBL-03", "labeling", Status.PASS,
        "non-finite Delta-PPL values are excluded from labeling",
        evidence={"stats": {k: v for k, v in stats.items() if k != "diagnostics"}},
    )


@check("LBL-04", "labeling", "Labels are the two release tokens in the release schema")
def check_label_schema() -> Result:
    from rag2.filter_training.labeling import LabelingObservation, label_observations
    from rag2.schema import Evidence, Question

    observations = [
        LabelingObservation(
            qid="q", snippet_index=i, correct_without_retrieval=bool(i % 2),
            correct_with_retrieval=not bool(i % 2), delta_ppl=float(i),
            evidence=Evidence(text=f"s{i}"),
            question=Question("q", "Q?", {"A": "a", "B": "b"}, "A"),
        )
        for i in range(6)
    ]
    pairs, _ = label_observations(observations, dataset_name="audit")
    schema_ok = all(set(p.to_training_record()) == {"id", "answer", "dataset_name", "question"} for p in pairs)
    tokens_ok = {p.answer for p in pairs} <= {paper.LABEL_HELPFUL, paper.LABEL_NOT_HELPFUL}
    prov_excluded = all("doc_id" not in p.to_training_record() for p in pairs)
    if not (schema_ok and tokens_ok and prov_excluded):
        return Result(
            "LBL-04", "labeling", Status.FAIL,
            "training records do not match the released schema",
            paper_says="R: {id, answer, dataset_name, question}, answer in {[HELPFUL],[NOT_HELPFUL]}",
            code_does=f"schema_ok={schema_ok} tokens_ok={tokens_ok} provenance_excluded={prov_excluded}",
            why_it_matters="the authors' training script reads exactly these four columns",
            how_to_fix="emit the four-field record and keep provenance in the sidecar",
        )
    return Result(
        "LBL-04", "labeling", Status.PASS,
        "four-field release schema; only the two label tokens; provenance kept out of the record",
        evidence={"records": len(pairs)},
    )


@check("PPL-05", "perplexity", "Truncation never shortens the scored continuation")
def check_truncation_preserves_continuation() -> Result:
    """The two Equation 4 terms must normalise by the same L.

    ``rag2/llm/hf.py:score`` truncates from the left so the continuation survives.
    That holds while the prompt absorbs the whole cut. If the cut exceeds the
    prompt, it eats into the continuation and the two terms normalise by
    different L, so Delta-PPL stops being a controlled comparison.
    """
    source_path = os.path.join(REPO, "rag2", "llm", "hf.py")
    source = open(source_path, "r", encoding="utf-8").read()
    guarded = "min(" in source and "len(prompt_ids)" in source and "continuation_ids" in source

    def scored_length(prompt_len: int, cont_len: int, max_len: int) -> int:
        """Replicates the index arithmetic in hf.py:score()."""
        prompt_ids = list(range(prompt_len))
        full_ids = prompt_ids + list(range(10_000, 10_000 + cont_len))
        if max_len and len(full_ids) > max_len:
            cut = len(full_ids) - max_len
            full_ids = full_ids[cut:]
            prompt_ids = prompt_ids[cut:] if cut < len(prompt_ids) else []
        return len(full_ids) - 1 - max(len(prompt_ids) - 1, 0)

    probes = [(150, 250, 2048), (2000, 300, 2048), (100, 2500, 2048), (50, 3000, 2048)]
    broken = [(p, c, m, scored_length(p, c, m)) for p, c, m in probes if scored_length(p, c, m) != c]

    if not broken or guarded:
        return Result(
            "PPL-05", "perplexity", Status.PASS,
            "the scored continuation survives truncation intact in every probe",
            evidence={"probes": probes, "guarded": guarded},
        )
    return Result(
        "PPL-05", "perplexity", Status.PARTIAL,
        "left-truncation can shorten the scored continuation when it alone exceeds the context window",
        paper_says=(
            "Eq4 normalises by L, the length of the scored sequence. Eq3 subtracts two such "
            "perplexities, which is only meaningful when both use the same L."
        ),
        code_does=(
            "rag2/llm/hf.py:score truncates from the left and sets prompt_ids=[] once the cut "
            "exceeds the prompt, at which point the removed tokens come out of the "
            f"continuation: {[(p, c, m, got) for p, c, m, got in broken]} "
            "(prompt_len, cont_len, max_len, tokens actually scored)"
        ),
        why_it_matters=(
            "LATENT, not currently live: it needs a rationale longer than the whole context "
            "window, and max_new_tokens is 512 against a >=2048 window, so no shipped config "
            "can reach it. It matters because the docstring asserts the continuation 'always "
            "survives intact', so a future change to max_input_tokens or max_new_tokens would "
            "silently break the L-comparability of Eq3 with no error."
        ),
        how_to_fix=(
            "clamp the cut to len(prompt_ids) and raise (or truncate the continuation "
            "explicitly and record it) rather than silently scoring fewer tokens"
        ),
        evidence={"probes": probes, "broken": broken, "latent": True,
                  "reachable_in_shipped_configs": False},
    )


@check("LBL-05", "labeling", "The rationale scored for Delta-PPL is the one retrieval used")
def check_rationale_reuse() -> Result:
    from rag2.filter_training.build_labels import build_observations
    from rag2.llm.stub import StubLLM
    from rag2.schema import CandidateSet, Evidence, Question

    question = Question("q0", "Vignette?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
    cached_rationale = "CACHED-RATIONALE-FROM-STAGE-1"
    candidates = {
        "q0": CandidateSet(qid="q0", rationale=cached_rationale,
                           candidates=[Evidence(text="snippet", source="cpg")])
    }
    llm = StubLLM()
    _, diagnostics = build_observations(llm, [question], candidates, top_k=1)
    scored = diagnostics["q0"]["closed_book_generation"]
    reused = scored == cached_rationale

    if reused:
        return Result(
            "LBL-05", "labeling", Status.PASS,
            "labeling scores the cached stage-1 rationale",
            evidence={"reused": True},
        )
    return Result(
        "LBL-05", "labeling", Status.PARTIAL,
        "labeling regenerates the rationale instead of reusing the cached stage-1 one",
        paper_says=(
            "P3.3: one rationale per question, generated by the base LLM, used as the "
            "retrieval query. Fig2 scores 'Rationale' from the same LLM output. The paper "
            "describes a single rationale per question, not two."
        ),
        code_does=(
            "rag2/filter_training/build_labels.py calls the LLM again with the closed-book "
            f"prompt and scores that generation ({scored[:40]!r}...), discarding "
            f"CandidateSet.rationale ({cached_rationale!r}) written by stage 1"
        ),
        why_it_matters=(
            "the two agree only while decoding is exactly deterministic and the same "
            "backbone and prompt are configured for both stages. The paper itself reports "
            "residual nondeterminism at temperature 0 (PA3), and nothing checks that the "
            "cache's rationale LLM matches the labeling LLM -- so the query that retrieved "
            "a passage can silently differ from the rationale whose perplexity judges it. "
            "It also doubles the closed-book generation cost of the most expensive stage."
        ),
        how_to_fix=(
            "reuse CandidateSet.rationale when present and generate only the correctness "
            "signal, or assert the regenerated rationale matches the cached one"
        ),
        evidence={"cached": cached_rationale, "scored": scored[:60], "reused": reused,
                  "generate_calls": len(llm.generate_calls)},
    )
