"""Structural checks: components exist, nothing thesis-specific leaked in."""

from __future__ import annotations

import dataclasses
import importlib
import io
import os
import re
import subprocess
import tokenize
from typing import List

from .registry import Result, Status, check

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_COMPONENTS = {
    "rationale generation": ("rag2.rationale", "generate_rationales"),
    "retrieval query formulation": ("rag2.rationale", "retrieval_query"),
    "query encoding": ("rag2.retrieval.encoder", "build_query_encoder"),
    "MIPS index search": ("rag2.retrieval.index", "search_corpus"),
    "balanced retrieval": ("rag2.retrieval.balanced", "balanced_retrieve"),
    "reranking": ("rag2.retrieval.rerank", "rerank_candidates"),
    "candidate cache": ("rag2.cache", "save_candidates"),
    "filter interface": ("rag2.filtering.base", "EvidenceFilter"),
    "RAG2 filter": ("rag2.filtering.rag2_filter", "RAG2PerplexityFilter"),
    "perplexity": ("rag2.filter_training.perplexity", "compute_perplexity_pair"),
    "labeling": ("rag2.filter_training.labeling", "decide_label"),
    "filter training": ("rag2.filter_training.train", "build_train_command"),
    "answer generation": ("rag2.generation", "generate_answers"),
    "evaluation": ("rag2.evaluation", "accuracy"),
    "orchestration": ("rag2.pipeline", "run_filter_and_generate"),
}


@check("STR-01", "architecture", "All four pipeline stages exist and are importable")
def check_components_exist() -> Result:
    missing: List[str] = []
    for label, (module_name, attr) in REQUIRED_COMPONENTS.items():
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, attr):
                missing.append(f"{label} ({module_name}.{attr} absent)")
        except Exception as error:
            missing.append(f"{label} ({module_name} import failed: {error})")
    if missing:
        return Result(
            "STR-01", "architecture", Status.FAIL,
            f"{len(missing)} pipeline component(s) missing",
            paper_says="P3/Fig1: rationale -> balanced retrieval + rerank -> filter -> generate",
            code_does="; ".join(missing),
            why_it_matters="a missing stage means the pipeline is not the paper's pipeline",
            how_to_fix="implement or re-export the missing callables",
            evidence={"missing": missing},
        )
    return Result(
        "STR-01", "architecture", Status.PASS,
        f"all {len(REQUIRED_COMPONENTS)} components present",
        evidence={"components": sorted(REQUIRED_COMPONENTS)},
    )


@check("STR-02", "scope", "No SCAF or thesis-specific machinery in the baseline path")
def check_no_thesis_code() -> Result:
    pattern = re.compile(
        r"scaf|frb.?pair|recency|publication_date|pub_date|temporal.?weight|"
        r"contested|entail|abstain|abstention|currency",
        re.IGNORECASE,
    )
    offenders: List[str] = []
    for root in ("rag2",):
        for dirpath, _, filenames in os.walk(os.path.join(REPO, root)):
            if "__pycache__" in dirpath:
                continue
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                source = open(path, "r", encoding="utf-8").read()
                spans = _docstring_spans(source)
                for token in tokenize.generate_tokens(io.StringIO(source).readline):
                    if token.type == tokenize.COMMENT:
                        continue
                    if token.type == tokenize.STRING and any(
                        lo <= token.start[0] <= hi for lo, hi in spans
                    ):
                        continue
                    if pattern.search(token.string):
                        offenders.append(
                            f"{os.path.relpath(path, REPO)}:{token.start[0]}: {token.string[:60]}"
                        )
    if offenders:
        return Result(
            "STR-02", "scope", Status.FAIL,
            f"{len(offenders)} thesis-specific reference(s) in executable baseline code",
            paper_says="the baseline must implement original RAG2 only",
            code_does="; ".join(offenders[:5]),
            why_it_matters="thesis machinery in the baseline invalidates it as a comparison point",
            how_to_fix="move the code into a separate thesis package",
            evidence={"offenders": offenders},
        )
    return Result(
        "STR-02", "scope", Status.PASS,
        "no SCAF/recency/entailment/abstention logic in executable baseline code",
        evidence={"scanned_root": "rag2/", "note": "docstrings may discuss these; code does not touch them"},
    )


def _docstring_spans(source: str):
    import ast

    spans = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            spans.append((body[0].lineno, body[0].end_lineno or body[0].lineno))
    return spans


@check("STR-03", "scope", "The authors' released code is unmodified")
def check_release_untouched() -> Result:
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--stat", "86add43", "HEAD", "--", "retriever/", "classifier/", "environment.yml"],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as error:
        return Result(
            "STR-03", "scope", Status.UNKNOWN,
            f"could not diff against the release commit: {error}",
            how_to_fix="run inside a git checkout containing commit 86add43",
        )
    if diff:
        return Result(
            "STR-03", "scope", Status.PARTIAL,
            "the authors' released files have been modified",
            paper_says="the release is the reference implementation",
            code_does=diff,
            why_it_matters="modifying the release makes it non-citable as the authors' code",
            how_to_fix="revert retriever/ and classifier/ and put changes in rag2/",
            evidence={"diff": diff},
        )
    return Result(
        "STR-03", "scope", Status.PASS,
        "retriever/, classifier/ and environment.yml are byte-identical to release commit 86add43",
    )


@check("STR-04", "configuration", "No dead configuration keys (silent no-ops)")
def check_dead_config_keys() -> Result:
    from rag2 import config as cfg

    source = ""
    for root in ("rag2", "scripts"):
        for dirpath, _, filenames in os.walk(os.path.join(REPO, root)):
            if "__pycache__" in dirpath:
                continue
            for filename in sorted(filenames):
                if filename.endswith(".py") and not filename.endswith("config.py"):
                    source += open(os.path.join(dirpath, filename), "r", encoding="utf-8").read()

    sections = [
        cfg.ExperimentConfig, cfg.DatasetConfig, cfg.CorpusConfig, cfg.RetrievalConfig,
        cfg.LLMConfig, cfg.FilterConfig, cfg.FilterTrainingConfig, cfg.GenerationConfig,
        cfg.EvaluationConfig, cfg.CacheConfig,
    ]
    dead: List[str] = []
    for section in sections:
        for field in dataclasses.fields(section):
            if not re.search(rf"\.{re.escape(field.name)}\b", source):
                dead.append(f"{section.__name__}.{field.name}")
    if dead:
        return Result(
            "STR-04", "configuration", Status.PARTIAL,
            f"{len(dead)} config key(s) declared but never read",
            paper_says="n/a -- these are reproduction-side knobs",
            code_does=", ".join(dead),
            why_it_matters=(
                "a key that looks like a control but is a no-op invites a silent "
                "deviation: setting it changes nothing while appearing to"
            ),
            how_to_fix="wire each key, or delete it and keep the value in the manifest only",
            evidence={"dead_keys": dead},
        )
    return Result("STR-04", "configuration", Status.PASS, "every config key is read somewhere")
