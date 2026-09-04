"""Retrieval, reranking, model fidelity, generation, evaluation, determinism."""

from __future__ import annotations

import os
from typing import List

from . import paper
from .registry import Result, Status, check

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 8


def _corpus(name, n=10, seed=0):
    import numpy as np

    from rag2.corpora.base import InMemoryCorpus

    rng = np.random.default_rng(seed)
    passages = [{"id": f"{name}-{i}", "text": f"{name} passage {i}"} for i in range(n)]
    return InMemoryCorpus(name, passages, rng.normal(size=(n, DIM)).astype("float32"))


# ------------------------------------------------------------- retrieval ---
@check("RET-01", "retrieval", "MedCPT checkpoints and index geometry match the release")
def check_retrieval_checkpoints() -> Result:
    from rag2.config import Config

    retrieval = Config().retrieval
    expected = {
        "query_encoder": paper.QUERY_ENCODER,
        "reranker": paper.RERANKER,
        "article_encoder": paper.ARTICLE_ENCODER,
        "embedding_dim": paper.EMBEDDING_DIM,
        "query_max_length": paper.QUERY_MAX_LENGTH,
        "rerank_max_length": paper.RERANK_MAX_LENGTH,
    }
    wrong = {k: (getattr(retrieval, k), v) for k, v in expected.items() if getattr(retrieval, k) != v}
    if wrong:
        return Result(
            "RET-01", "retrieval", Status.FAIL,
            f"{len(wrong)} retrieval setting(s) differ from the release",
            paper_says=str(expected),
            code_does=str({k: v[0] for k, v in wrong.items()}),
            why_it_matters="a different encoder or truncation length changes every retrieved passage",
            how_to_fix="restore the release values in configs/default.yaml",
            evidence={"wrong": wrong},
        )
    return Result(
        "RET-01", "retrieval", Status.PASS,
        "MedCPT query encoder, cross-encoder, 768-dim index and 512-token truncation all match",
        evidence=expected,
    )


@check("RET-02", "retrieval", "Search is exact inner product (IndexFlatIP semantics)")
def check_exact_mips() -> Result:
    import numpy as np

    from rag2.retrieval.index import faiss_available, search_corpus

    corpus = _corpus("cpg", n=25, seed=1)
    matrix = list(corpus.embedding_shards())[0][1]
    queries = np.random.default_rng(5).normal(size=(4, DIM)).astype("float32")
    scores, indices = search_corpus(corpus, queries, top_k=6)

    wrong = []
    for q in range(queries.shape[0]):
        brute = np.argsort(-(matrix @ queries[q]))[:6]
        if list(indices[q]) != list(brute):
            wrong.append({"query": q, "got": list(map(int, indices[q])), "brute": list(map(int, brute))})
    if wrong:
        return Result(
            "RET-02", "retrieval", Status.FAIL,
            "search does not return the exact inner-product top-k",
            paper_says=f"R:retriever/retrieve.py builds {paper.INDEX_TYPE}",
            code_does=str(wrong[:2]),
            why_it_matters="an approximate index silently changes the candidate pool",
            how_to_fix="use faiss.IndexFlatIP or the exact numpy fallback",
            evidence={"wrong": wrong},
        )
    return Result(
        "RET-02", "retrieval", Status.PASS,
        "top-k equals brute-force inner product on every probe query",
        evidence={"faiss_installed": faiss_available(),
                  "backend": "faiss" if faiss_available() else "exact numpy fallback"},
    )


@check("RET-03", "retrieval", "Balanced retrieval draws equally from every corpus")
def check_balanced_retrieval() -> Result:
    import numpy as np

    from rag2.config import RetrievalConfig
    from rag2.retrieval.balanced import balanced_retrieve, corpus_distribution

    corpora = [_corpus(name, seed=i) for i, name in enumerate(paper.CORPORA)]
    queries = np.random.default_rng(3).normal(size=(3, DIM)).astype("float32")
    per_corpus = 4
    pooled = balanced_retrieve(corpora, queries, RetrievalConfig(embedding_dim=DIM, candidates_per_corpus=per_corpus))

    distributions = [corpus_distribution(p) for p in pooled]
    expected = {name: per_corpus for name in paper.CORPORA}
    if any(d != expected for d in distributions):
        return Result(
            "RET-03", "retrieval", Status.FAIL,
            "the candidate pool is not balanced across corpora",
            paper_says="P3.4: 'extracts an equal number of documents from each corpus'",
            code_does=str(distributions[0]),
            why_it_matters="balanced retrieval is one of the paper's three contributions",
            how_to_fix="search each corpus separately for candidates_per_corpus and pool",
            evidence={"distributions": distributions},
        )
    return Result(
        "RET-03", "retrieval", Status.PASS,
        f"every corpus contributes exactly {per_corpus} candidates to the pool",
        evidence={"distribution": distributions[0], "corpora": list(paper.CORPORA)},
    )


@check("RET-04", "retrieval", "Sharding a corpus does not change its top-k")
def check_shard_merge_preserves_balance() -> Result:
    import numpy as np

    from rag2.corpora.base import Corpus
    from rag2.retrieval.index import search_corpus

    base = _corpus("pubmed", n=24, seed=7)
    matrix = list(base.embedding_shards())[0][1]

    class Sharded(Corpus):
        name = "pubmed"

        def __len__(self):
            return len(base)

        def passage(self, index):
            return base.passage(index)

        def embedding_shards(self):
            for s in range(4):
                yield s * 6, matrix[s * 6:(s + 1) * 6]

    queries = np.random.default_rng(11).normal(size=(3, DIM)).astype("float32")
    single_scores, single_indices = search_corpus(base, queries, top_k=5)
    merged_scores, merged_indices = search_corpus(Sharded(), queries, top_k=5, shard_merge="score")
    _, concat_indices = search_corpus(Sharded(), queries, top_k=5, shard_merge="concat")

    identical = bool((single_indices == merged_indices).all())
    concat_width = int(concat_indices.shape[1])
    if not identical:
        return Result(
            "RET-04", "retrieval", Status.FAIL,
            "score-merged sharding changes the corpus top-k",
            paper_says="P3.4: balance means an equal count per corpus, however it is stored",
            code_does=f"single={single_indices[0].tolist()} sharded={merged_indices[0].tolist()}",
            why_it_matters="sharding is a memory workaround; it must not alter results",
            how_to_fix="merge shard results by score and keep candidates_per_corpus",
            evidence={"single": single_indices.tolist(), "merged": merged_indices.tolist()},
        )
    return Result(
        "RET-04", "retrieval", Status.PASS,
        "score-merged shards reproduce the single-index top-k exactly; 'concat' reproduces "
        f"the release's imbalance ({concat_width} candidates instead of 5)",
        evidence={"identical_to_single_index": identical, "concat_width": concat_width,
                  "release_behaviour_available": True},
    )


@check("RET-05", "reranking", "Reranking sorts descending and is stable on ties")
def check_rerank_ordering() -> Result:
    from rag2.retrieval.rerank import rerank_candidates
    from rag2.schema import Evidence

    class Scorer:
        def __init__(self, values):
            self.values = values

        def score(self, query, snippets):
            return list(self.values)

    candidates = [Evidence(text=f"s{i}", source="x") for i in range(5)]
    ranked = rerank_candidates(Scorer([0.0, 3.0, 1.0, 4.0, 2.0]), "q", list(candidates), top_k=3)
    order_ok = [e.text for e in ranked] == ["s3", "s1", "s4"]
    ranks_ok = [e.rank for e in ranked] == [1, 2, 3]
    scores_ok = [e.rerank_score for e in ranked] == [4.0, 3.0, 2.0]

    tied = rerank_candidates(Scorer([1.0] * 5), "q", [Evidence(text=f"s{i}") for i in range(5)], top_k=5)
    release_order = [f"s{i}" for i in sorted(range(5), key=lambda k: [1.0] * 5, reverse=True)]
    tie_ok = [e.text for e in tied] == release_order

    if not (order_ok and ranks_ok and scores_ok and tie_ok):
        return Result(
            "RET-05", "reranking", Status.FAIL,
            "reranked ordering is wrong",
            paper_says="R:retriever/rerank.py sorts by cross-encoder logit, descending, stable",
            code_does=f"order={order_ok} ranks={ranks_ok} scores={scores_ok} ties={tie_ok}",
            why_it_matters="ordering decides which passages the top-k sweep admits",
            how_to_fix="sort descending with a stable sort; assign 1-based ranks",
            evidence={"ranked": [e.text for e in ranked]},
        )
    return Result(
        "RET-05", "reranking", Status.PASS,
        "descending order, 1-based ranks, tie order matches the release's stable sort",
        evidence={"ranked": [e.text for e in ranked], "tie_order": [e.text for e in tied]},
    )


@check("RET-06", "reranking", "Reranker cross-encodes the initial query, as the paper states")
def check_rerank_query_source() -> Result:
    from rag2.config import Config

    configured = Config().retrieval.rerank_query
    if configured != paper.RERANK_QUERY:
        return Result(
            "RET-06", "reranking", Status.FAIL,
            f"rerank_query defaults to {configured!r}, not the paper's {paper.RERANK_QUERY!r}",
            paper_says="Fig1 caption and P3.4 both say the reranker encodes the initial/original query",
            code_does=f"retrieval.rerank_query = {configured!r}",
            why_it_matters="reranking with a different query changes the final ordering entirely",
            how_to_fix=f"set retrieval.rerank_query: {paper.RERANK_QUERY}",
        )
    return Result(
        "RET-06", "reranking", Status.PARTIAL,
        "defaults to the paper's 'initial query'; note the released code disagrees",
        paper_says="Fig1 caption: 'cross-encoding the initial query and each snippet'; P3.4: 'encodes the original query'",
        code_does=(
            "defaults to 'initial' (paper). R:retriever/main.py:124 passes the rationale file "
            "instead, so the authors' released code reranks with the rationale. "
            "configs/ablation_release_retrieval.yaml reproduces the code path."
        ),
        why_it_matters=(
            "the paper and the authors' own code disagree, so one of them does not "
            "describe the run that produced Table 2. Which was used is unresolvable from "
            "the artefacts, and the two give different final top-k orderings."
        ),
        how_to_fix="ask the authors; both behaviours are implemented and switchable",
        evidence={"default": configured, "release_behaviour": "rationale"},
    )


@check("RET-07", "retrieval", "The paper's four corpora are the configured sources")
def check_corpora_sources() -> Result:
    from rag2.config import load_config

    path = os.path.join(REPO, "configs", "corpora.example.yaml")
    if not os.path.exists(path):
        return Result("RET-07", "retrieval", Status.UNKNOWN, "corpora config not found")
    names = tuple(c.name for c in load_config(path).retrieval.corpora)
    if names != paper.CORPORA:
        return Result(
            "RET-07", "retrieval", Status.FAIL,
            f"configured corpora {names} differ from the paper's {paper.CORPORA}",
            paper_says="PA3 Table A1: PubMed, PMC, CPG, Textbooks (the Self-BioRAG corpus)",
            code_does=str(names),
            why_it_matters="a different source mix is a different retrieval system",
            how_to_fix="configure the four corpora",
        )
    return Result(
        "RET-07", "retrieval", Status.APPROXIMATION,
        "the four corpus slots are configured, but the corpora themselves are unavailable",
        paper_says="PA3 Table A1: 37.6M docs / 116.7M passages / 564.2GB (Self-BioRAG corpus)",
        code_does="four named slots with no data; the medical corpus is being prepared separately",
        why_it_matters=(
            "this is the single largest expected source of divergence from Table 2. "
            "Retrieval quality dominates a RAG pipeline's accuracy."
        ),
        how_to_fix="not fixable here; supply the corpus and re-run, then record the delta",
        evidence={"corpora": list(names)},
    )


# ---------------------------------------------------------------- models ---
@check("MOD-01", "models", "Backbone LLM identifier matches the paper")
def check_backbone_model() -> Result:
    from rag2.config import Config

    configured = Config().llm.model
    if configured != paper.BACKBONE_LLAMA:
        return Result(
            "MOD-01", "models", Status.FAIL,
            f"default backbone is {configured!r}",
            paper_says=f"P4.2 footnote 2 gives the explicit URL for {paper.BACKBONE_LLAMA}",
            code_does=f"llm.model = {configured!r}",
            why_it_matters="a different backbone changes rationales, labels and answers",
            how_to_fix=f"set llm.model: {paper.BACKBONE_LLAMA}",
        )
    return Result(
        "MOD-01", "models", Status.PASS,
        f"default backbone is {paper.BACKBONE_LLAMA}, the one the paper names explicitly",
        evidence={"model": configured, "revision_pinned": bool(Config().llm.revision)},
    )


@check("MOD-02", "models", "Meerkat checkpoint path")
def check_meerkat_checkpoint() -> Result:
    from rag2.config import load_config

    path = os.path.join(REPO, "configs", "medqa_meerkat.yaml")
    if not os.path.exists(path):
        return Result("MOD-02", "models", Status.UNKNOWN, "meerkat config not found")
    configured = load_config(path).llm.model
    return Result(
        "MOD-02", "models", Status.UNKNOWN,
        f"configured as {configured!r}, which the paper never states",
        paper_says=(
            "P4.2 cites Kim et al. (2024) and describes the model (Mistral-7B init, "
            "GPT-4-rationale instruction tuning, MedQA+MedMCQA fine-tuning) but gives no "
            "checkpoint path"
        ),
        code_does=f"llm.model = {configured!r}, inferred from that description",
        why_it_matters=(
            "if the inference is wrong, the Meerkat rows are produced by a different model "
            "than the paper's and are not comparable"
        ),
        how_to_fix="confirm against the Meerkat paper/release before the final run",
        evidence={"configured": configured, "inferred": True},
    )


@check("MOD-03", "models", "Filter base model and trained checkpoint")
def check_filter_checkpoint() -> Result:
    from rag2.config import Config

    filter_config = Config().filter
    base_ok = paper.FILTER_BASE_MODEL_FAMILY in filter_config.base_model
    if not base_ok:
        return Result(
            "MOD-03", "models", Status.FAIL,
            f"filter base model is {filter_config.base_model!r}",
            paper_says="P4.2: Flan-T5-large, 770M parameters",
            code_does=f"filter.base_model = {filter_config.base_model!r}",
            why_it_matters="a different filter architecture is a different method",
            how_to_fix="set filter.base_model: google/flan-t5-large",
        )
    return Result(
        "MOD-03", "models", Status.APPROXIMATION,
        "correct base model, but the paper's trained checkpoint is unavailable and must be retrained",
        paper_says="P4.2: Flan-T5-large filter trained on perplexity labels",
        code_does=(
            f"filter.base_model={filter_config.base_model!r}; filter.checkpoint is empty "
            "and the filter refuses to run without one"
        ),
        why_it_matters=(
            "the reproduced filter is retrained from reconstructed labels. The authors ran "
            "unseeded (R: --seed defaults to None), so an exact checkpoint match is "
            "impossible in principle, not merely inconvenient."
        ),
        how_to_fix="train with scripts/03 + scripts/04 and report filter metrics alongside accuracy",
        evidence={"base_model": filter_config.base_model, "max_seq_length": filter_config.max_seq_length,
                  "doc_stride": filter_config.doc_stride},
    )


@check("MOD-04", "models", "Filter training hyperparameters match the paper")
def check_training_hyperparameters() -> Result:
    from rag2.config import Config

    training = Config().filter_training
    wrong = {}
    for key, expected in paper.TRAINING.items():
        actual = getattr(training, key, None)
        if isinstance(expected, float):
            if actual is None or abs(actual - expected) > 1e-12:
                wrong[key] = (actual, expected)
        elif actual != expected:
            wrong[key] = (actual, expected)
    if wrong:
        return Result(
            "MOD-04", "models", Status.FAIL,
            f"{len(wrong)} training hyperparameter(s) differ",
            paper_says=str(paper.TRAINING),
            code_does=str({k: v[0] for k, v in wrong.items()}),
            why_it_matters="the filter is the paper's contribution; its training recipe is specified",
            how_to_fix="restore the values from PA3 and run/run_large_train_xl_000.sh",
            evidence={"wrong": wrong},
        )
    return Result(
        "MOD-04", "models", Status.PASS,
        f"all {len(paper.TRAINING)} specified hyperparameters match (lr 3e-5, 40 epochs, batch 16, 512/128)",
        evidence=paper.TRAINING,
    )


@check("MOD-05", "models", "Runtime model/tokenizer loading")
def check_runtime_loading() -> Result:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        installed = True
    except Exception:
        installed = False
    return Result(
        "MOD-05", "models", Status.MANUAL,
        "runtime checkpoint/tokenizer/dtype fidelity needs a GPU run to verify",
        paper_says="MedCPT encoders, Flan-T5-large filter, Llama-3-8B-Instruct backbone",
        code_does=(
            "identifiers are verified statically (MOD-01..04). Actual weight loading, "
            "tokenizer vocabulary, dtype and device placement are exercised only when "
            "torch/transformers are installed"
        ),
        why_it_matters="a correct identifier can still load a revision that has since changed upstream",
        how_to_fix=(
            "pin llm.revision and run rag2_audit with the models installed; the run manifest "
            "records the resolved revision hash"
        ),
        evidence={"torch_and_transformers_installed": installed},
    )


# ------------------------------------------------------------ generation ---
@check("GEN-01", "generation", "Decoding is greedy at temperature 0")
def check_greedy_decoding() -> Result:
    from rag2.config import Config

    config = Config()
    temperatures = {"llm.temperature": config.llm.temperature,
                    "generation.temperature": config.generation.temperature}
    if any(t != 0.0 for t in temperatures.values()):
        return Result(
            "GEN-01", "generation", Status.FAIL,
            "decoding is not greedy",
            paper_says=f"PA3 Inference: {paper.DECODING}",
            code_does=str(temperatures),
            why_it_matters="sampling makes the baseline non-reproducible run to run",
            how_to_fix="set both temperatures to 0",
            evidence=temperatures,
        )
    source = open(os.path.join(REPO, "rag2", "llm", "hf.py"), "r", encoding="utf-8").read()
    honours = "do_sample=False" in source
    return Result(
        "GEN-01", "generation", Status.PASS if honours else Status.PARTIAL,
        "temperature 0 in config and the HF backend maps it to do_sample=False",
        evidence={**temperatures, "hf_backend_sets_do_sample_false": honours},
    )


@check("GEN-02", "generation", "Answer-generation prompt")
def check_answer_prompt() -> Result:
    from rag2.prompts import ANSWER_PROMPT_WITH_EVIDENCE, RATIONALE_PROMPT

    reuses_cot = ANSWER_PROMPT_WITH_EVIDENCE.startswith(RATIONALE_PROMPT[:80])
    return Result(
        "GEN-02", "generation", Status.UNKNOWN,
        "the paper never publishes the answer-generation prompt; this one is reconstructed",
        paper_says=(
            "Fig1 shows only the structure: prompt = retrieved snippets + initial query. "
            "The wording is not given anywhere in the paper or the release."
        ),
        code_does=(
            "reuses the paper's chain-of-thought prompt (P3.3, printed verbatim) with an "
            f"evidence block prepended (reuses_cot_wording={reuses_cot})"
        ),
        why_it_matters=(
            "prompt wording materially affects multiple-choice accuracy, and Meerkat was "
            "instruction-tuned on a specific prompt. This is the largest single prompt-level "
            "assumption in the reproduction and a prime suspect if accuracy misses Table 2."
        ),
        how_to_fix="unresolvable from the artefacts; vary it first when investigating a gap",
        evidence={"reuses_cot_wording": reuses_cot},
    )


@check("GEN-03", "generation", "Context is the kept passages in rank order")
def check_context_construction() -> Result:
    from rag2.prompts import DEFAULT_PROMPTS
    from rag2.schema import Evidence, Question

    question = Question("q", "Stem?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
    kept = [Evidence(text=f"passage-{i}", rank=i + 1) for i in range(3)]
    rendered = DEFAULT_PROMPTS.render_answer_prompt(question, kept)
    positions = [rendered.index(f"passage-{i}") for i in range(3)]
    ordered = positions == sorted(positions)
    question_after = rendered.index("Stem?") > max(positions)
    empty = DEFAULT_PROMPTS.render_answer_prompt(question, [])
    closed_book = "retrieved documents" not in empty
    if not (ordered and question_after and closed_book):
        return Result(
            "GEN-03", "generation", Status.FAIL,
            "context is not assembled as Figure 1 shows",
            paper_says="Fig1: prompt = snippets then initial query",
            code_does=f"ordered={ordered} question_last={question_after} closed_book_fallback={closed_book}",
            why_it_matters="passage order affects attention and answer choice",
            how_to_fix="emit kept passages in rank order, then the question",
        )
    return Result(
        "GEN-03", "generation", Status.PASS,
        "kept passages appear in rank order before the question; empty evidence falls back to closed book",
        evidence={"positions": positions},
    )


@check("GEN-04", "generation", "Rationale and answer lengths are governed consistently")
def check_generation_length_keys() -> Result:
    rationale_src = open(os.path.join(REPO, "rag2", "rationale.py"), "r", encoding="utf-8").read()
    pipeline_src = open(os.path.join(REPO, "rag2", "pipeline.py"), "r", encoding="utf-8").read()
    accepts = "max_new_tokens" in rationale_src and "temperature" in rationale_src
    wired = "max_new_tokens=config.llm.max_new_tokens" in pipeline_src
    if accepts and wired:
        return Result(
            "GEN-04", "generation", Status.PASS,
            "rationale decoding settings are passed explicitly from llm.*, not left implicit",
            evidence={"rationale_accepts_explicit_settings": accepts,
                      "pipeline_wires_llm_max_new_tokens": wired},
        )
    return Result(
        "GEN-04", "generation", Status.PARTIAL,
        "rationale length is governed by llm.max_new_tokens, answer length by generation.max_new_tokens",
        paper_says="the paper states neither length",
        code_does=(
            "rag2/rationale.py calls llm.generate(chunk) with no max_new_tokens, so it "
            "falls back to LLMConfig.max_new_tokens; rag2/generation.py explicitly passes "
            "GenerationConfig.max_new_tokens"
        ),
        why_it_matters=(
            "two keys with the same name and default silently govern different stages. "
            "Setting generation.max_new_tokens to study answer length would leave the "
            "rationale -- and therefore the retrieval query and every perplexity label -- "
            "unchanged, which is easy to miss."
        ),
        how_to_fix="pass an explicit length from one key, or rename them to name their stage",
        evidence={"rationale_accepts_explicit_settings": accepts, "pipeline_wires_it": wired},
    )


# ------------------------------------------------------------ evaluation ---
@check("EVA-01", "evaluation", "Accuracy is the reported metric")
def check_metric() -> Result:
    from rag2.evaluation import accuracy
    from rag2.schema import PipelineResult

    results = [PipelineResult(qid=str(i), correct=(i < 3), gold="A", prediction="A") for i in range(4)]
    metrics = accuracy(results)
    ok = abs(metrics["accuracy"] - 75.0) < 1e-9 and metrics["num_correct"] == 3
    if not ok:
        return Result(
            "EVA-01", "evaluation", Status.FAIL, "accuracy is miscomputed",
            paper_says="Table 2 reports accuracy in percent",
            code_does=str(metrics), why_it_matters="the headline number would be wrong",
            how_to_fix="accuracy = 100 * correct / scored",
        )
    return Result(
        "EVA-01", "evaluation", Status.PASS,
        "accuracy in percent over scored examples, matching Table 2's metric",
        evidence=metrics,
    )


@check("EVA-02", "evaluation", "Answer extraction rule")
def check_answer_extraction() -> Result:
    from rag2.evaluation import extract_choice

    options = {"A": "BiPAP", "B": "Chest tube", "C": "Intubation", "D": "Needle decompression"}
    paper_example = "... Therefore, the answer is (C) Intubation."
    got = extract_choice(paper_example, options)
    return Result(
        "EVA-02", "evaluation", Status.UNKNOWN,
        "the paper never states how the option is extracted from a free-form generation",
        paper_says=(
            "P3.3's prompt asks for 'your explanation and single option ... as the final "
            "answer'; Fig4's worked example ends 'Therefore, the answer is (C) Intubation'. "
            "No extraction rule is published."
        ),
        code_does=(
            f"ordered regex list, last match wins, falls back to option-text matching; "
            f"parses the paper's own Fig4 example correctly (-> {got!r}); unparsed counts as incorrect"
        ),
        why_it_matters=(
            "extraction strictness directly moves reported accuracy, and an unparsed "
            "generation scored as incorrect versus abstained is a different number"
        ),
        how_to_fix="unresolvable; report num_unparsed alongside accuracy (the code does)",
        evidence={"figure_4_example": got, "unparsed_as_incorrect": True},
    )


@check("EVA-03", "evaluation", "Benchmark split sizes match Table 1")
def check_dataset_sizes() -> Result:
    from rag2.datasets.benchmarks import EXPECTED_SIZES

    if EXPECTED_SIZES != paper.DATASET_SIZES:
        return Result(
            "EVA-03", "evaluation", Status.FAIL,
            "hard-coded split sizes differ from Table 1",
            paper_says=str(paper.DATASET_SIZES), code_does=str(EXPECTED_SIZES),
            why_it_matters="the size assertion is what catches evaluating on the wrong split",
            how_to_fix="restore Table 1's counts",
        )
    return Result(
        "EVA-03", "evaluation", Status.PASS,
        "Table 1 split sizes are encoded and assertable via dataset.options.assert_size",
        evidence=paper.DATASET_SIZES,
    )


# ----------------------------------------------------------- determinism ---
@check("DET-01", "determinism", "Seeds are set and recorded")
def check_seeding() -> Result:
    from rag2.config import Config
    from rag2.experiment import set_all_seeds

    seed = Config().experiment.seed
    info = set_all_seeds(seed)
    return Result(
        "DET-01", "determinism", Status.PASS,
        f"seed {seed} is set across {len(info['seeded'])} RNG(s) and written to the manifest",
        paper_says="R:classifier/run_classifier.py defaults --seed to None: the authors ran unseeded",
        code_does=f"experiment.seed={seed}, seeded={info['seeded']}",
        evidence=info,
    )


@check("DET-02", "determinism", "No process-randomised hashing in pipeline code")
def check_hash_determinism() -> Result:
    import ast

    offenders: List[str] = []
    for root in ("rag2", "scripts"):
        for dirpath, _, filenames in os.walk(os.path.join(REPO, root)):
            if "__pycache__" in dirpath:
                continue
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                source = open(path, "r", encoding="utf-8").read()
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue
                lines = source.splitlines()
                for node in ast.walk(tree):
                    # A bare call to the builtin hash(); obj.hash(...) is not it.
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                            and node.func.id == "hash":
                        line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                        offenders.append(f"{os.path.relpath(path, REPO)}:{node.lineno}: {line}")
    if offenders:
        return Result(
            "DET-02", "determinism", Status.FAIL,
            f"{len(offenders)} use(s) of Python's process-randomised hash() to seed randomness",
            paper_says="PA3: greedy decoding to minimise randomness",
            code_does="; ".join(offenders),
            why_it_matters=(
                "Python randomises str/tuple hashing per process unless PYTHONHASHSEED is "
                "set, so any RNG seeded from hash() produces different values on every run. "
                "Two 'identical' runs then diverge silently."
            ),
            how_to_fix="seed from hashlib.sha256 of the text instead of hash()",
            evidence={"offenders": offenders},
        )
    return Result(
        "DET-02", "determinism", Status.PASS,
        "no RNG is seeded from Python's process-randomised hash()",
    )


@check("DET-03", "determinism", "Cached candidates are replay-verified by fingerprint")
def check_cache_fingerprint_guard() -> Result:
    from rag2.config import load_config

    base = load_config(None, {})
    same_k = load_config(None, {"retrieval": {"final_top_k": 32}})
    different_depth = load_config(None, {"retrieval": {"candidates_per_corpus": 50}})
    different_rerank = load_config(None, {"retrieval": {"rerank_query": "rationale"}})

    k_insensitive = base.retrieval_fingerprint() == same_k.retrieval_fingerprint()
    depth_sensitive = base.retrieval_fingerprint() != different_depth.retrieval_fingerprint()
    rerank_sensitive = base.retrieval_fingerprint() != different_rerank.retrieval_fingerprint()
    if not (k_insensitive and depth_sensitive and rerank_sensitive):
        return Result(
            "DET-03", "determinism", Status.FAIL,
            "the cache fingerprint does not track the right settings",
            paper_says="n/a -- a reproduction-side guarantee",
            code_does=f"k_insensitive={k_insensitive} depth={depth_sensitive} rerank={rerank_sensitive}",
            why_it_matters="silently swapping the candidate set between experiments invalidates comparisons",
            how_to_fix="include retrieval-determining fields in the fingerprint, exclude final_top_k",
        )
    return Result(
        "DET-03", "determinism", Status.PASS,
        "fingerprint ignores final_top_k (so one cache serves the k sweep) and tracks depth "
        "and rerank-query changes; a mismatched replay raises",
        evidence={"k_insensitive": k_insensitive, "depth_sensitive": depth_sensitive,
                  "rerank_sensitive": rerank_sensitive},
    )
