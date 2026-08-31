"""What the original RAG2 work specifies, as machine-checkable constants.

Single source of truth for the audit. Every value carries its citation so a
reviewer can check the constant itself against the paper, not just the code
against the constant.

Sources:
  P<n>   paper section n (Sohn et al., NAACL 2025, 2025.naacl-long.635)
  PA<n>  paper appendix section n
  Fig<n> paper figure n
  Eq<n>  paper equation n
  R:<path>  the authors' released repository
"""

from __future__ import annotations

from typing import Dict, Tuple

# --- models and checkpoints ------------------------------------------------
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"          # R:retriever/query_encode.py:52
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"      # R:README.md "Data & corpora"
RERANKER = "ncbi/MedCPT-Cross-Encoder"               # R:retriever/rerank.py:19-20
FILTER_BASE_MODEL_FAMILY = "flan-t5-large"           # P4.2 "Flan-T5-large ... 770 million"
BACKBONE_LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"  # P4.2 footnote 2 (explicit URL)

# The paper cites Kim et al. (2024) with no checkpoint path; the id below is an
# inference from the paper's description, not a published fact.
BACKBONE_MEERKAT_INFERRED = "dmis-lab/meerkat-7b-v1.0"  # P4.2, path NOT published

# --- retrieval -------------------------------------------------------------
EMBEDDING_DIM = 768                                   # R:retriever/retrieve.py IndexFlatIP(768)
QUERY_MAX_LENGTH = 512                                # R:retriever/query_encode.py:70
RERANK_MAX_LENGTH = 512                               # R:retriever/rerank.py:32
INDEX_TYPE = "exact inner product (IndexFlatIP)"      # R:retriever/retrieve.py
CORPORA = ("pubmed", "pmc", "cpg", "textbook")        # P3.4, PA3 Table A1
TOP_K_GRID = (1, 2, 4, 8, 16, 32)                     # Fig3 x-axis
# Table 2 numbers, read off Figure 3 at the k where each peaks.
OPTIMAL_K = {
    ("llama3", "medqa"): 32,
    ("llama3", "medmcqa"): 16,
    ("meerkat", "medqa"): 2,
    ("meerkat", "medmcqa"): 8,
}
# P3.4 + Fig1 caption: the reranker cross-encodes the INITIAL query.
RERANK_QUERY = "initial"

# --- filter ----------------------------------------------------------------
LABEL_HELPFUL = "[HELPFUL]"                           # R:classifier/run_classifier.py:93
LABEL_NOT_HELPFUL = "[NOT_HELPFUL]"                   # R:classifier/run_classifier.py:94
FILTER_MAX_SEQ_LENGTH = 512                           # R:classifier/run/run_large_train_xl_000.sh
FILTER_DOC_STRIDE = 128                               # R: same
FILTER_INPUT_TEMPLATE_HEAD = (
    "Given the following evidence, determine whether it helps answer the provided question."
)                                                     # R:classifier/data/medqa/llama3_cot/5%-train.json
FILTER_SCORING = "softmax over the two label-token logits at the first decoded position"
# R:classifier/run_classifier.py:696-713

# --- perplexity labeling ---------------------------------------------------
TAU_PERCENTILE = 25.0                                 # P3.2 "top 25% of perplexity differentials"
DELTA_DEFINITION = "PPL(x) - PPL(x, d)"               # Eq3
DELTA_TEST = ">="                                     # Eq3
PPL_DEFINITION = "exp(-(1/L) * sum log P)"            # Eq4
# Fig2, transcribed: (correct_without, correct_with, delta>=tau) -> label
FIGURE_2_TRUTH_TABLE: Dict[Tuple[bool, bool, bool], str] = {
    (True, True, True): LABEL_HELPFUL,
    (True, True, False): "[DISCARD]",
    (True, False, True): LABEL_NOT_HELPFUL,
    (True, False, False): LABEL_NOT_HELPFUL,
    (False, True, True): LABEL_HELPFUL,
    (False, True, False): LABEL_HELPFUL,
    (False, False, True): LABEL_NOT_HELPFUL,
    (False, False, False): "[DISCARD]",
}

# --- filter training hyperparameters (PA3 + R:run/run_large_train_xl_000.sh) --
TRAINING = {
    "learning_rate": 3e-5,
    "num_train_epochs": 40,
    "per_device_train_batch_size": 16,
    "max_seq_length": 512,
    "doc_stride": 128,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.0,
    "lr_scheduler_type": "linear",
    "num_warmup_steps": 0,
    "max_answer_length": 30,
    "checkpointing_steps": "epoch",
}

# --- generation / evaluation ----------------------------------------------
DECODING = "greedy, temperature 0"                    # PA3 "Inference"
METRIC = "accuracy"                                   # Table 2
DATASET_SIZES = {                                     # Table 1
    "medqa": {"train": 10178, "validation": 1272, "test": 1273},
    "medmcqa": {"train": 182822, "validation": 4183, "test": 6150},
    "mmlu_med": {"test": 1089},
}
MMLU_MED_SUBJECTS = (
    "anatomy", "clinical_knowledge", "college_biology",
    "college_medicine", "medical_genetics", "professional_medicine",
)                                                     # P4.1 (PA2 says "human genetics")

# --- things the paper/repo never specify ----------------------------------
# Audited as UNKNOWN or APPROXIMATION; listed here so the report cannot forget one.
UNSPECIFIED = {
    "answer_generation_prompt": "never published; Fig1 shows only snippets + initial query",
    "answer_extraction_rule": "never published",
    "candidates_per_corpus": "pre-rerank retrieval depth never stated",
    "chunk_size_overlap": "PA3 says 'sliding window with overlap', no sizes",
    "tau_population": "top-25% global vs per-question not stated",
    "ppl_scored_tokens": "Eq4 literally scores the query; prose/Fig2 say the rationale",
    "label_top_k": "snippets per question sent through labeling not stated",
    "random_seed": "R:classifier/run_classifier.py defaults --seed to None (unseeded)",
    "filter_checkpoint": "not distributed (R:README.md)",
    "corpora_and_embeddings": "not distributed, 564.2GB (PA3 Table A1)",
    "meerkat_checkpoint_path": "cited as Kim et al. 2024, no path given",
    "gpt4o_snapshot": "'the latest version', no snapshot string",
    "optimal_k_mmlu_and_gpt4o": "not recoverable from the published figures",
    "five_percent_artifact": "R:classifier/data/.../5%-train.json filename undocumented vs tau=25%",
    "preprocess_py": "R:classifier/data/preprocess.py is an empty file (labeling code unreleased)",
}
