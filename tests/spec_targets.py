"""Where each testable requirement of docs/methodology.md lives in the code, and how a test finds it.

The acceptance tests (test_spec_*.py) fix behaviour, not code. Each asks `resolve(name)` for the function
that implements a requirement. This file is the only place that maps a name to a module under src/. When an
interface changes while the code is written, change it here and in section 11 of docs/methodology.md.

Skip rule: a test is skipped only while its part has not been started, that is, while no .py file exists under
src/<top-level package of the module>. Once the part exists, a missing module, a missing function or a failing
check is an error, never a skip. With SPEC_STRICT=1 an unstarted part is an error too; baseline acceptance
requires a strict run in which no test is skipped.

Contract (plain Python values only, no models, no third-party packages; names after * are passed by keyword):

  rationale_filter.labels
    label_pair(*, correct_without, correct_with, ppl_without, ppl_with, tau) -> "helpful" | "not_helpful" | "discard"
    to_training_token(label) -> "[HELPFUL]" | "[NOT_HELPFUL]"           (ValueError for "discard" or anything else)
    perplexity_threshold(deltas, top_fraction=0.25) -> float             (tau; ValueError for an empty pool)
  rationale_filter.inference
    build_filter_input(*, evidence, question) -> str
    decide_helpful(*, logit_helpful, logit_not_helpful) -> bool
    judge_snippets(*, question, snippets, judge) -> list[bool]           (judge(filter_input) -> bool, once per snippet)
    keep_helpful(items, decisions) -> list
  rationale_query.prompt
    build_rationale_prompt(question) -> str
  rationale_query.query
    dense_query(rationale) -> str
    rerank_query(*, question, rationale, against="question") -> str      (against is "question" or "rationale")
  balanced_retrieval.pool
    pool_candidates(per_corpus, top_k) -> list                           (per_corpus: corpus name -> ranked candidates)
    rerank(candidates, scores, top_k) -> list
  balanced_retrieval.retrieve
    balanced_retrieve(*, question, rationale, searchers, scorer, top_k, rerank_against="question") -> list
        searchers: corpus name -> search(query_text, k) -> ranked candidates; scorer(query_text, candidates) -> scores
  balanced_retrieval.diagnostics
    corpus_share(final) -> dict[str, float]                              (a key for every corpus in CORPUS_ORDER)
    duplicate_rate(final) -> float
  data_roles.splits
    split_by_question(*, question_ids, fractions, seed) -> dict[str, str]
    assert_disjoint(roles) -> None                                       (ValueError naming the shared ids)

A candidate is a mapping with at least "corpus" (one of CORPUS_ORDER), "doc_key" (one canonical identity key per
document, equal across corpora for the same document) and "text".
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

HELPFUL = "helpful"
NOT_HELPFUL = "not_helpful"
DISCARD = "discard"
TOKEN_HELPFUL = "[HELPFUL]"
TOKEN_NOT_HELPFUL = "[NOT_HELPFUL]"

CORPUS_ORDER = ("pubmed", "pmc", "guidelines", "textbooks")

FILTER_INSTRUCTION = "Given the following evidence, determine whether it helps answer the provided question."
RATIONALE_PROMPT = (
    "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step "
    "fashion, starting by summarizing the available information. Output your explanation and single option "
    "from the given options as the final answer. Here is the question:"
)

TARGETS = {
    "label_pair": ("rationale_filter.labels", "label_pair"),
    "to_training_token": ("rationale_filter.labels", "to_training_token"),
    "perplexity_threshold": ("rationale_filter.labels", "perplexity_threshold"),
    "build_filter_input": ("rationale_filter.inference", "build_filter_input"),
    "decide_helpful": ("rationale_filter.inference", "decide_helpful"),
    "judge_snippets": ("rationale_filter.inference", "judge_snippets"),
    "keep_helpful": ("rationale_filter.inference", "keep_helpful"),
    "build_rationale_prompt": ("rationale_query.prompt", "build_rationale_prompt"),
    "dense_query": ("rationale_query.query", "dense_query"),
    "rerank_query": ("rationale_query.query", "rerank_query"),
    "pool_candidates": ("balanced_retrieval.pool", "pool_candidates"),
    "rerank": ("balanced_retrieval.pool", "rerank"),
    "balanced_retrieve": ("balanced_retrieval.retrieve", "balanced_retrieve"),
    "corpus_share": ("balanced_retrieval.diagnostics", "corpus_share"),
    "duplicate_rate": ("balanced_retrieval.diagnostics", "duplicate_rate"),
    "split_by_question": ("data_roles.splits", "split_by_question"),
    "assert_disjoint": ("data_roles.splits", "assert_disjoint"),
}


def part_started(module, src=SRC):
    """True once any .py file exists under src/<top-level package of module> (or src/<top>.py)."""
    top = module.split(".")[0]
    if (src / (top + ".py")).is_file():
        return True
    folder = src / top
    return folder.is_dir() and any(folder.rglob("*.py"))


def resolve(name, targets=TARGETS, src=SRC):
    """Return the function behind `name`; skip while its part is unstarted, fail loudly afterwards."""
    module, attr = targets[name]
    where = module + "." + attr
    if not part_started(module, src):
        if os.environ.get("SPEC_STRICT") == "1":
            raise AssertionError("strict run, not implemented yet: " + where)
        raise unittest.SkipTest("not implemented yet: " + where)
    if sys.path[:1] != [str(src)]:
        sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name and (module == error.name or module.startswith(error.name + ".")):
            raise AssertionError(f"the part exists but the module {module} is missing (needed for {attr})") from error
        raise
    try:
        return getattr(loaded, attr)
    except AttributeError:
        raise AssertionError(f"{module} exists but defines no {attr}") from None


class SpecTestCase(unittest.TestCase):
    """Base class of the acceptance tests: a test whose `requires` are not implemented is skipped as a whole, once."""

    requires = ()

    def setUp(self):
        for name in self.requires:
            resolve(name)
