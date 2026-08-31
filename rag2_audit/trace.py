"""End-to-end audit trace for a single question.

Runs one question through every stage of the original RAG2 pipeline and records
what each stage actually did, so a researcher can check the behaviour against
the paper by reading rather than by trusting the code:

    python -m rag2_audit.trace                       # synthetic, no models needed
    python -m rag2_audit.trace -c configs/medqa_llama3.yaml --qid <id> \
        --candidates cache/candidates/<...>.jsonl    # a real run

The trace records, per the audit brief: the original question, the generated
rationale, the retrieval query, the retrieved passages with retrieval and
rerank scores, the final candidate ordering, the perplexity of the rationale
with and without each passage, the perplexity differences, the threshold, which
passages are admitted or rejected and why, the exact context handed to the
generator, and the generated answer.

Every number is accompanied by the paper reference it should be checked against.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rag2 import __version__ as rag2_version  # noqa: E402
from rag2.config import Config, DatasetConfig, RetrievalConfig, load_config, merge_overrides  # noqa: E402
from rag2.evaluation import extract_choice, is_correct  # noqa: E402
from rag2.filter_training.labeling import decide_label  # noqa: E402
from rag2.filter_training.perplexity import compute_perplexity_pair, top_percent_threshold  # noqa: E402
from rag2.retrieval.balanced import corpus_distribution  # noqa: E402
from rag2.schema import CandidateSet, Question  # noqa: E402

PAPER_REFS = {
    "rationale": "P3.3 -- chain-of-thought prompt, printed verbatim in the paper",
    "retrieval_query": "P3.3 -- 'search ... solely using the rationale, excluding the initial query'",
    "balanced_retrieval": "P3.4 -- 'an equal number of documents from each corpus'",
    "rerank": "P3.4 + Fig1 -- cross-encode the INITIAL query with each snippet (code uses the rationale)",
    "perplexity": "Eq4 -- PPL = exp(-(1/L) sum log P); Eq3 -- Delta-PPL = PPL(x) - PPL(x,d)",
    "threshold": "P3.2 -- tau = top 25% of perplexity differentials, fixed across all experiments",
    "labeling": "Fig2 -- decision tree over (correct w/o, correct w/, Delta-PPL >= tau)",
    "filter": "P3.2 -- Flan-T5 keeps only [HELPFUL] snippets; one snippet at a time (Limitations)",
    "generation": "Fig1 -- prompt = kept snippets + initial query; PA3 -- greedy, temperature 0",
}


def _synthetic_setup():
    """A self-contained example so the trace runs with no models or corpus."""
    import numpy as np

    from rag2.corpora.base import InMemoryCorpus
    from rag2.llm.stub import StubLLM

    dim = 16
    rng = np.random.default_rng(0)
    question = Question(
        qid="synthetic-1",
        question=(
            "A 62-year-old man with COPD presents with severe dyspnoea, bilateral wheezes "
            "and poor air movement despite bronchodilators. What is the best next step?"
        ),
        options={"A": "BiPAP", "B": "Chest tube placement", "C": "Intubation", "D": "Needle decompression"},
        answer="A",
        dataset="synthetic", split="test",
    )
    corpora = []
    for name in ("pubmed", "pmc", "cpg", "textbook"):
        passages = [
            {
                "id": f"{name}-doc{j}",
                "passage_id": f"{name}-doc{j}-p0",
                "text": f"[{name}] Passage {j} on management of respiratory failure and ventilation.",
                "publication_date": f"{1998 + j}-01-01",
            }
            for j in range(3)
        ]
        corpora.append(InMemoryCorpus(name, passages, rng.normal(size=(3, dim)).astype("float32")))

    config = Config()
    config.experiment.name = "audit-trace"
    config.dataset = DatasetConfig(loader="inline", name="synthetic")
    config.retrieval = RetrievalConfig(
        embedding_dim=dim, candidates_per_corpus=2, final_top_k=8,
        query_encoder="stub", reranker="none", rerank_query="initial",
    )
    config.llm.backend = "stub"
    config.llm.model = "stub"
    return config, question, corpora, StubLLM(config.llm)


class _HashEncoder:
    """Deterministic stand-in encoder (sha256-seeded, stable across processes)."""

    def __init__(self, dim):
        self.dim = dim

    def encode(self, queries, batch_size=None):
        import hashlib

        import numpy as np

        rows = []
        for query in queries:
            seed = int.from_bytes(hashlib.sha256(query.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).normal(size=(self.dim,)).astype("float32")
            rows.append(vector / (np.linalg.norm(vector) or 1.0))
        return np.vstack(rows)


def build_trace(
    config: Config,
    question: Question,
    llm,
    corpora=None,
    candidate_set: Optional[CandidateSet] = None,
    evidence_filter=None,
    score_perplexity: bool = True,
) -> Dict[str, Any]:
    prompts = config.prompt_set()
    trace: Dict[str, Any] = {
        "_about": "End-to-end audit trace of one question through the original RAG2 pipeline.",
        "_paper_references": PAPER_REFS,
        "_versions": {"rag2": rag2_version, "config_fingerprint": config.fingerprint(),
                      "prompt_fingerprint": prompts.fingerprint()},
        "_backend": llm.describe(),
    }

    # -- stage 0: the question -------------------------------------------
    trace["01_question"] = {
        "qid": question.qid,
        "stem": question.question,
        "options": question.options,
        "gold_answer": question.answer,
        "rendered_for_model": prompts.render_question(question),
    }

    # -- stage 1: rationale ----------------------------------------------
    if candidate_set is not None and candidate_set.rationale:
        rationale = candidate_set.rationale
        source = "replayed from the candidate cache"
    else:
        rationale = llm.generate([prompts.render_rationale_prompt(question)])[0].strip()
        source = "generated now"
    trace["02_rationale"] = {
        "paper": PAPER_REFS["rationale"],
        "prompt_sent": prompts.render_rationale_prompt(question),
        "rationale": rationale,
        "source": source,
    }

    # -- stage 2: retrieval query ----------------------------------------
    from rag2.rationale import retrieval_query

    query = candidate_set.retrieval_query if candidate_set and candidate_set.retrieval_query \
        else retrieval_query(rationale, question)
    trace["03_retrieval_query"] = {
        "paper": PAPER_REFS["retrieval_query"],
        "query_sent_to_retriever": query,
        "is_the_rationale_not_the_question": query.strip() == rationale.strip(),
        "initial_question_excluded": question.question not in query,
    }

    # -- stage 3: balanced retrieval + rerank ----------------------------
    if candidate_set is None:
        from rag2.pipeline import run_retrieval
        from rag2.retrieval.rerank import IdentityReranker

        candidate_set = run_retrieval(
            config, [question], llm=llm, corpora=corpora,
            encoder=_HashEncoder(config.retrieval.embedding_dim),
            reranker=IdentityReranker(), prompts=prompts,
            rationales={question.qid: rationale},
        )[0]

    candidates = candidate_set.candidates
    trace["04_retrieval"] = {
        "paper": PAPER_REFS["balanced_retrieval"],
        "candidates_per_corpus_configured": config.retrieval.candidates_per_corpus,
        "pool_size": len(candidates),
        "pool_by_source": corpus_distribution(candidates),
        "balanced": len(set(corpus_distribution(candidates).values())) <= 1,
        "passages": [
            {
                "rank": e.rank, "source": e.source, "doc_id": e.doc_id, "passage_id": e.passage_id,
                "retrieval_score_inner_product": e.retrieval_score,
                "rerank_score_cross_encoder_logit": e.rerank_score,
                "text": e.text,
                "metadata_carried_but_never_read_by_baseline": e.metadata,
            }
            for e in candidates
        ],
    }
    trace["05_reranking"] = {
        "paper": PAPER_REFS["rerank"],
        "rerank_query_setting": config.retrieval.rerank_query,
        "rerank_query_text": candidate_set.rerank_query,
        "final_ordering": [
            {"rank": e.rank, "source": e.source, "rerank_score": e.rerank_score, "text": e.text[:70]}
            for e in candidates
        ],
        "descending": all(
            (a.rerank_score or 0) >= (b.rerank_score or 0)
            for a, b in zip(candidates, candidates[1:])
        ),
    }

    top_k = config.retrieval.final_top_k
    selected = candidate_set.top(top_k)
    trace["06_top_k_selection"] = {
        "paper": "Fig3 -- k swept over {1,2,4,8,16,32}, selected on validation",
        "final_top_k": top_k,
        "selected": [e.text[:70] for e in selected],
        "dropped": [e.text[:70] for e in candidates[len(selected):]],
    }

    # -- stage 4: perplexity (the filter's training signal) --------------
    if score_perplexity:
        rows = []
        for index, evidence in enumerate(selected):
            pair = compute_perplexity_pair(
                llm, question, rationale, evidence, prompts=prompts,
                target=config.filter_training.ppl_target,
            )
            rows.append({
                "index": index, "source": evidence.source, "doc_id": evidence.doc_id,
                "ppl_without_document_PPL_x": pair.ppl_without,
                "ppl_with_document_PPL_x_d": pair.ppl_with,
                "delta_ppl": pair.delta,
                "document_lowered_perplexity": pair.delta > 0,
                "scored_tokens": pair.num_tokens,
            })
        deltas = [r["delta_ppl"] for r in rows]
        tau = top_percent_threshold(deltas, config.filter_training.tau_percentile) if deltas else float("nan")
        for row in rows:
            row["passes_threshold_delta_ge_tau"] = row["delta_ppl"] >= tau
        trace["07_perplexity"] = {
            "paper": PAPER_REFS["perplexity"],
            "scored_text": "the rationale" if config.filter_training.ppl_target == "rationale" else "the query",
            "scored_text_value": rationale,
            "conditioning_without": "answer prompt with no evidence block",
            "conditioning_with": "the same prompt plus this one passage",
            "rows": rows,
        }
        trace["08_threshold"] = {
            "paper": PAPER_REFS["threshold"],
            "tau_percentile": config.filter_training.tau_percentile,
            "tau_scope": config.filter_training.tau_scope,
            "tau_value_over_this_question_only": tau,
            "note": (
                "In a real labeling run tau is computed once over the whole split "
                "(tau_scope=global), not per question. The per-question value here is "
                "illustrative of the mechanism only."
            ),
            "n_passing": sum(1 for r in rows if r["passes_threshold_delta_ge_tau"]),
        }
        trace["09_label_demonstration"] = {
            "paper": PAPER_REFS["labeling"],
            "note": "how Figure 2 would label each passage, given correctness both ways",
            "truth_table_applied": [
                {
                    "index": row["index"],
                    "delta_ppl": row["delta_ppl"],
                    "lower_perplexity": row["passes_threshold_delta_ge_tau"],
                    "label_if_correct_both_ways": decide_label(True, True, row["passes_threshold_delta_ge_tau"]),
                    "label_if_wrong_both_ways": decide_label(False, False, row["passes_threshold_delta_ge_tau"]),
                }
                for row in rows
            ],
        }

    # -- stage 5: filtering ----------------------------------------------
    if evidence_filter is None:
        from rag2.filtering.base import build_filter

        try:
            evidence_filter = build_filter(config.filter, prompts)
        except Exception as error:
            from rag2.filtering.passthrough import PassthroughFilter

            evidence_filter = PassthroughFilter()
            trace["_filter_substitution"] = (
                f"the configured filter could not be built ({error}); this trace uses "
                "PassthroughFilter, so the admit/reject column below is NOT the RAG2 filter"
            )
    kept, decisions = evidence_filter.apply(question, selected)
    trace["10_filtering"] = {
        "paper": PAPER_REFS["filter"],
        "filter": evidence_filter.describe(),
        "example_filter_input": prompts.render_filter_prompt(question, selected[0]) if selected else "",
        "decisions": [
            {
                "index": i, "source": e.source, "doc_id": e.doc_id,
                "p_helpful": d.score, "label": d.label,
                "admitted": d.keep, "text": e.text[:70],
            }
            for i, (e, d) in enumerate(zip(selected, decisions))
        ],
        "n_admitted": len(kept),
        "n_rejected": len(selected) - len(kept),
        "on_empty_policy": config.filter.on_empty,
    }

    # -- stage 6: generation ---------------------------------------------
    context = prompts.render_answer_prompt(question, kept)
    generation = llm.generate([context], max_new_tokens=config.generation.max_new_tokens,
                              temperature=config.generation.temperature)[0].strip()
    prediction = extract_choice(generation, question.options,
                                config.evaluation.extraction_patterns or None)
    trace["11_generation"] = {
        "paper": PAPER_REFS["generation"],
        "context_sent_to_generator": context,
        "n_passages_in_context": len(kept),
        "temperature": config.generation.temperature,
        "generation": generation,
        "extracted_prediction": prediction,
        "gold": question.answer,
        "correct": is_correct(prediction, question.answer),
    }

    # -- provenance isolation --------------------------------------------
    dates = [e.metadata.get("publication_date") for e in kept if e.metadata.get("publication_date")]
    model_inputs = context + "\n" + "\n".join(
        prompts.render_filter_prompt(question, e) for e in selected
    )
    trace["12_provenance_isolation"] = {
        "check": "publication metadata is carried but must never reach a model input",
        "dates_carried": dates,
        "any_date_in_model_input": any(str(d) in model_inputs for d in dates),
        "doc_ids_carried": [e.doc_id for e in kept],
        "any_doc_id_in_model_input": any(str(e.doc_id) in model_inputs for e in kept if e.doc_id),
    }
    return trace


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default="", help="config file; omit for the synthetic example")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--qid", default="", help="question id to trace (default: the first)")
    parser.add_argument("--candidates", default="", help="replay candidates from a cache")
    parser.add_argument("--out", default="audit/trace.json", help="where to write the trace")
    parser.add_argument("--no-perplexity", action="store_true", help="skip the perplexity stage")
    args = parser.parse_args(argv)

    corpora = candidate_set = None
    if not args.config:
        config, question, corpora, llm = _synthetic_setup()
        print("no --config given: tracing the built-in synthetic example (no models needed)")
    else:
        from rag2.datasets.base import build_dataset
        from rag2.llm.base import build_llm

        config = load_config(args.config, merge_overrides(args.override))
        questions = build_dataset(config.dataset).questions()
        question = next((q for q in questions if q.qid == args.qid), None) if args.qid else questions[0]
        if question is None:
            print(f"question id {args.qid!r} not found", file=sys.stderr)
            return 2
        llm = build_llm(config.llm)
        if args.candidates:
            from rag2.cache import index_by_qid, load_candidates

            candidate_set = index_by_qid(
                load_candidates(args.candidates, config.retrieval_fingerprint(),
                                config.cache.allow_config_mismatch)
            ).get(question.qid)
            if candidate_set is None:
                print(f"no cached candidates for {question.qid}", file=sys.stderr)
                return 2
        else:
            from rag2.pipeline import build_corpora

            corpora = build_corpora(config)

    trace = build_trace(config, question, llm, corpora=corpora, candidate_set=candidate_set,
                        score_perplexity=not args.no_perplexity)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, ensure_ascii=False, default=str)

    print(f"\ntraced question {trace['01_question']['qid']}")
    print(f"  rationale        : {trace['02_rationale']['rationale'][:70]}...")
    print(f"  retrieval query  : rationale-only = {trace['03_retrieval_query']['is_the_rationale_not_the_question']}")
    print(f"  pool             : {trace['04_retrieval']['pool_size']} passages, "
          f"balanced = {trace['04_retrieval']['balanced']}, {trace['04_retrieval']['pool_by_source']}")
    print(f"  rerank query     : {trace['05_reranking']['rerank_query_setting']}, "
          f"descending = {trace['05_reranking']['descending']}")
    if "08_threshold" in trace:
        print(f"  tau (this q only): {trace['08_threshold']['tau_value_over_this_question_only']:.6g}, "
              f"{trace['08_threshold']['n_passing']} passage(s) pass")
    print(f"  filter           : {trace['10_filtering']['n_admitted']} admitted, "
          f"{trace['10_filtering']['n_rejected']} rejected")
    print(f"  answer           : {trace['11_generation']['extracted_prediction']} "
          f"(gold {trace['11_generation']['gold']}, correct = {trace['11_generation']['correct']})")
    print(f"  no date leakage  : {not trace['12_provenance_isolation']['any_date_in_model_input']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
