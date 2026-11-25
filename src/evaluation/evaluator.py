"""One evaluation script for all systems (LLM only, standard RAG, RAG^2, proposed) and all views (image only, text only,
image + text).

Inputs
  --questions    data/views/<view>/<split>.jsonl from the views command: the options of each case, the answer letter,
                 the gold class, the lesion group
  --predictions  one JSON line per question (the llm-only command writes this format):
            {"id": "...", "pred": "C",                               # the chosen option letter
             "rationale": "free-text explanation",                  # optional, for the explanation metrics
             "query": "the text the system searched with",          # optional (RAG systems); the question is used if absent
             "contexts": [{"chunk_id": .., "title": .., "text": .., "tags": [..]}, ...]}   # retrieved passages, best first
             optional per line: "tokens_in", "tokens_out", "seconds" (cost and time of the call)
  --predictions-b  (optional) a second system on the same questions, for McNemar's test

The options are shuffled for each case, so a letter is first turned into its class with that case's own options.
A question with no prediction, or with an answer that is not one of its option letters, counts as wrong.

Metrics
  Diagnosis : accuracy (the main score) with a 95% bootstrap interval, macro-F1 with its interval, per-class precision,
              recall (= sensitivity), specificity and F1, confusion matrix, missing and invalid answers (the invalid ones split
              into refusals, by REFUSAL, and other invalid answers),
              McNemar's exact test against --pred-b. The bootstrap draws whole lesions, so photos of one lesion stay together.
              Also: how often each option letter was chosen (a model that prefers one position shows here), and cost/time.
              Accuracy is top-1 accuracy (the chosen option is the true class); no top-k accuracy is reported (decided 2026-11-22).
              Several runs are compared with the compare command (McNemar for every pair, Holm's correction).
  Evidence  : RAG answers that carry "evidence": {"chunk_id", "quote"} (null = an explicit "Evidence: none"; a missing or unreadable line has an empty quote and is counted as unparsed, not as a rejection). Coverage and selective
              risk with verified evidence (an answer is covered when it has a valid diagnosis and its quote is word for word in the
              cited passage; this replaces the yes/no flag), and negative rejection (the share of "Evidence: none" when no
              retrieved passage is tagged with the true class, after RGB). Quote accuracy, supported accuracy and accuracy by
              subgroup were removed on the researcher's decision (2026-11-24: they repeat other metrics). Whether the quote really supports the diagnosis is NOT tested: an NLI check was tried on
              2026-11-23, failed its pre-set pass mark twice (0.83 pairwise, then 0.50 top-1) and was removed on the researcher's decision.
  Retrieval : Hit@k, Recall@k (with --passages), Precision@k, MRR and nDCG@k (context precision was dropped, 2026-11-24). Weak labels: a passage counts as relevant when its
              tags contain the gold class. nDCG@k assumes the corpus holds at least k relevant passages for each class.
  Relevance : context relevance: how well each retrieved passage fits the query, scored with --relevance-models by the MedCPT
              cross-encoder and the cosine of MedCPT embeddings (Jaccard was dropped on the researcher's decision, 2026-11-24).
  Text      : semantic similarity (cosine of sentence embeddings, rationale vs the cited reference texts of the true class, the
              best-matching one kept). No LLM-judge metric (RAGAS) is used: removed on the researcher's decision, 2026-11-24.
  Image claims in the text-only view : the share of reasonings that describe an image or a dermoscopy for the case although no image was
              given (a word list, sentences that deny or hypothesise are skipped). An approximate lower bound, decided 2026-11-24; it
              is never applied to the image views, where a described finding cannot be checked.

Sources: McNemar (1947) Psychometrika 12:153-157; nDCG: Jarvelin & Kekalainen (2002)
ACM TOIS 20:422-446; MedCPT: Jin et al. (2023) Bioinformatics 39:btad651;
coverage and selective risk: Geifman & El-Yaniv (2017) NeurIPS; Holm (1979) Scandinavian Journal of Statistics 6:65-70;
negative rejection: RGB, Chen et al. (2024) AAAI
38:17754-17762; recall, precision and ranking of retrieval: BEIR, Thakur et al. (2021) NeurIPS Datasets and Benchmarks.

Every report records its input files with their SHA-256, so two reports can be checked to use the same question file.

Usage: python -m src.main evaluate --questions data/views/multimodal/test.jsonl --predictions predictions.jsonl
       --out report.json [--predictions-b other.jsonl] [--relevance-models] [--reference-texts class_reference.json]
       [--passages data/corpus/passages.jsonl]
       [--skip-text]
       python -m src.main similarity-check             (is the semantic similarity informative? finished runs + reference texts)
       python -m src.main collect                      (a report.json for every finished run and the table results/comparisons/summary.csv)
       python -m src.main compare --run name=questions,predictions --run name=questions,predictions --out compare.json
       (several runs on the same cases: McNemar's exact test for every pair, Holm-corrected; see the end of this file)"""
import argparse
import itertools
import json
import os

from src.data.loaders import recorded_question_path
from src.evaluation.diagnostic import (
    diagnosis_metrics,
    gold_class,
    predicted_class,
    usage_summary,
)
from src.evaluation.hallucination import (
    SIMILARITY_MODEL,
    SIMILARITY_PASS,
    evidence_metrics,
    image_claim_metrics,
    semantic_similarity,
    sentence_encoder,
    similarity_check,
)
from src.evaluation.retrieval import (
    medcpt_scorers,
    relevance_metrics,
    relevant_counts,
    retrieval_metrics,
)
from src.evaluation.statistics import holm, mcnemar_exact, pct
from src.utils.common import REFERENCE_TEXTS
from src.utils.io import read_jsonl, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.reproducibility import text_matches, text_sha256


def load(path):
    """A JSONL file as {id: record}; stops if an id is repeated."""
    records = read_jsonl(path)
    by_id = {d["id"]: d for d in records}
    if len(by_id) != len(records):
        raise ValueError(f"{path}: {len(records) - len(by_id)} repeated ids")
    return by_id


def file_record(path):
    return {"path": str(path).replace("\\", "/"), "sha256": text_sha256(path)}


def similarity_check_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main similarity-check",
                                 description="Does the semantic similarity carry information? Uses finished runs and the reference texts only.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--root", default=".")
    ap.add_argument("--reference-texts", default=os.path.join("data", REFERENCE_TEXTS))
    ap.add_argument("--out", default="results/comparisons/similarity_check.json")
    a = ap.parse_args(argv)
    with open(a.reference_texts, encoding="utf-8") as f:
        refs = json.load(f)
    encode = sentence_encoder()
    out, log = {"model": SIMILARITY_MODEL, "runs": {}}, get_logger("similarity_check")
    for system, model, run, folder, info in finished_runs(a.results):
        qa = load(os.path.join(a.root, info["settings"]["questions"]))
        res = similarity_check(qa, load(os.path.join(folder, "predictions.jsonl")), refs, encode)
        out["runs"][f"{system}/{model}/{run}"] = res
        if res:
            log.info(f"{model} {run}: the answered disease has the highest similarity in {pct(res['identified'])} of {res['n_answers']} answers "
                     f"(chance 12.5%); answered {res['mean_similarity_to_answered_class']} vs the other seven "
                     f"{res['mean_similarity_to_the_other_seven']}; {'PASSED' if res['passed'] else 'NOT PASSED'} ({SIMILARITY_PASS})")
    write_json(a.out, out)
    return out


def text_metrics(qa, pred, reference_texts, encode=None):
    """Semantic similarity of the explanations (when reference texts are given)."""
    out = {}
    if reference_texts:
        sim = semantic_similarity(qa, pred, reference_texts, encode or sentence_encoder())
        if sim:
            out["semantic_similarity_cosine"] = {**sim, "model": SIMILARITY_MODEL if encode is None else "given"}
    return out


def evaluate(qa_path, pred_path, pred_b_path=None, relevance_models=False, reference_texts=None,
             skip_text=False, passages_path=None, answered_only=False):
    """The full report of one system on one question file (the same code for every system and every view)."""
    qa, pred = load(qa_path), load(pred_path)
    if answered_only:      # a trial run (--limit): score only the questions that were asked; the report says so
        qa = {i: q for i, q in qa.items() if i in pred}
    inputs = {"questions": file_record(qa_path), "predictions": file_record(pred_path)}
    if pred_b_path:
        inputs["predictions_b"] = file_record(pred_b_path)
    rep = {"inputs": inputs, "trial_answered_only": bool(answered_only),
           "diagnosis": diagnosis_metrics(qa, pred, load(pred_b_path) if pred_b_path else None),
           "usage": usage_summary(qa, pred),
           "retrieval": retrieval_metrics(qa, pred, n_relevant=relevant_counts(read_jsonl(passages_path)) if passages_path else None)}
    if rep["retrieval"] and relevance_models:
        rep["relevance"] = relevance_metrics(qa, pred, medcpt_scorers())
    has_evidence = any("evidence" in p for p in pred.values())
    rep["evidence"] = evidence_metrics(qa, pred) if has_evidence else None
    if not skip_text:
        refs = None
        if reference_texts:
            with open(reference_texts, encoding="utf-8") as f:
                refs = json.load(f)
        rep["text"] = text_metrics(qa, pred, refs)
    rep["text_only_image_claims"] = image_claim_metrics(qa, pred)
    return rep


def report_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main evaluate", description="Evaluates one system's predictions on one question file.")
    ap.add_argument("--questions", required=True, help="data/views/<view>/<split>.jsonl")
    ap.add_argument("--predictions", required=True, help="the predictions of one system, one JSON line per question")
    ap.add_argument("--predictions-b", default=None, help="a second system on the same questions, for McNemar's test")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--relevance-models", action="store_true", help="also score relevance with MedCPT (downloads 3 models)")
    ap.add_argument("--reference-texts", default=None)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--answered-only", action="store_true",
                    help="trial runs: score only the questions that have a prediction (without it, an unasked question counts as wrong)")
    ap.add_argument("--passages", default=None,
                    help="the corpus (data/corpus/passages.jsonl): with it Recall@k is reported, as the share of all the "
                         "corpus passages tagged with the true class that are found in the top k")
    a = ap.parse_args(argv)
    rep = evaluate(a.questions, a.predictions, a.predictions_b, a.relevance_models, a.reference_texts,
                   a.skip_text, a.passages, a.answered_only)
    write_json(a.out, rep)
    log = get_logger("evaluate")
    d = rep["diagnosis"]
    log.info(f"accuracy {pct(d['accuracy'])} (95% CI {pct(d['accuracy_95ci'][0])} to {pct(d['accuracy_95ci'][1])}) | macro-F1 {pct(d['macro_f1'])} | "
             f"melanoma recall {pct(d['per_class']['MEL']['recall'])} | missing {d['missing_predictions']}, invalid {d['invalid_answers']} (refusals {d['invalid_answers_by_cause']['refusals']})")
    if rep["text_only_image_claims"]:
        c = rep["text_only_image_claims"]
        log.info(f"text-only reasonings that describe an image: {c['n_claiming_an_image']} of {c['n_text_only_with_reasoning']} "
                 f"(approximate; ids in the report)")
    if rep["retrieval"]:
        top = max(int(k.split("@")[1]) for k in rep["retrieval"] if k.startswith("hit@"))
        log.info(f"retrieval: { {k: rep['retrieval'][k] for k in (f'hit@{top}', f'ndcg@{top}', 'mrr')} }")
        if "relevance" in rep:
            log.info(f"relevance: {rep['relevance']}")
    return rep


# ---------------------------------------------------------------------------------------------------------------------------
# compare: several runs on the same cases (python -m src.main compare ...)
# ---------------------------------------------------------------------------------------------------------------------------
# Compares several runs on the same cases: each run's accuracy, and for every pair McNemar's exact test on the questions exactly one
# of the two answers right, with Holm's correction for the number of pairs (Holm 1979), since a page of many comparisons would
# otherwise show some "significant" differences by chance.
#
# A run is name=questions_file,predictions_file. The question files may differ (e.g. two views of the same cases, whose options are
# the same per case); every run must hold the same case ids. An answer is read with its own question's options; a missing or invalid
# answer is wrong.
#
# Usage:
#   python -m src.main compare --out results/comparisons/llm_only/compare_text_only_test.json       --run claude=data/views/text_only/test.jsonl,results/llm_only/claude-sonnet-5-5/text_only_test/predictions.jsonl       --run gpt-4o=data/views/text_only/test.jsonl,results/llm_only/gpt-4o/text_only_test/predictions.jsonl

def correct_by_id(questions, predictions):
    unknown = set(predictions) - set(questions)
    if unknown:
        raise ValueError(f"{len(unknown)} predictions are not in the question file, e.g. {sorted(unknown)[:3]}")
    return {i: predicted_class(q, predictions.get(i)) == gold_class(q) for i, q in questions.items()}


def compare(runs):
    """runs: {name: {id: right (bool)}}, all on the same ids. Returns the accuracy of each run and the tests of every pair."""
    ids = None
    for name, right in runs.items():
        if ids is None:
            ids = sorted(right)
        elif sorted(right) != ids:
            raise ValueError(f"run {name!r} holds other cases than the first run")
    accuracy = {name: round(sum(right.values()) / len(right), 4) for name, right in runs.items()}
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        truth = [True] * len(ids)                    # McNemar on right/wrong: "gold" is True, a run's answer is its rightness
        test = mcnemar_exact(truth, [runs[a][i] for i in ids], [runs[b][i] for i in ids])
        pairs.append({"a": a, "b": b, "accuracy_a": accuracy[a], "accuracy_b": accuracy[b], **test})
    for pair, adjusted in zip(pairs, holm([p["p_value"] for p in pairs])):
        pair["p_holm"] = round(adjusted, 6)
    return {"n_questions": len(ids), "accuracy": accuracy, "pairs": pairs}


def compare_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main compare", description="Compares several runs on the same cases (McNemar + Holm).")
    ap.add_argument("--run", action="append", required=True, help="name=questions_file,predictions_file (two or more)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.run) < 2:
        ap.error("give at least two --run")
    runs, inputs = {}, {}
    for spec in a.run:
        name, _, files = spec.partition("=")
        qpath, _, ppath = files.partition(",")
        if not (name and qpath and ppath) or name in runs:
            ap.error(f"--run must be a new name=questions_file,predictions_file (got {spec!r})")
        runs[name] = correct_by_id(load(qpath), load(ppath))
        inputs[name] = {"questions": file_record(qpath), "predictions": file_record(ppath)}
    rep = {"inputs": inputs, **compare(runs)}
    write_json(a.out, rep)
    log = get_logger("compare")
    for p in rep["pairs"]:
        log.info(f"{p['a']} {pct(p['accuracy_a'])} vs {p['b']} {pct(p['accuracy_b'])}: only a right {p['a_right_b_wrong']}, only b right "
                 f"{p['a_wrong_b_right']}, p = {p['p_value']}, Holm p = {p['p_holm']}")
    return rep


# ---------------------------------------------------------------------------------------------------------------------------
# collect: one report per finished run and one table of all runs (decided 2026-11-24).
# A finished run is a folder results/<system>/<view>/<model>_<split>/ (no ".partial" ending) whose run.json says "complete". The report
# goes next to its predictions as report.json (it is written again when the predictions or the code settings changed); the table
# results/comparisons/summary.csv has one row per run, so a thesis table is never copied by hand. Trial runs (.partial) are never collected.
#   python -m src.main collect [--results results] [--out results/comparisons/summary.csv] [--skip-text] [--force]
# ---------------------------------------------------------------------------------------------------------------------------
SUMMARY_COLUMNS = ["system", "model", "model_id", "view", "split", "status", "questions", "answered", "accuracy", "accuracy_ci_low",
                   "accuracy_ci_high", "macro_f1", "missing", "invalid", "refusals", "semantic_similarity", "text_only_image_claim_rate",
                   "tokens_in", "tokens_out", "seconds", "served_by", "questions_sha256", "folder"]


def finished_runs(results_dir):
    """The run folders (system/view/model_split) that are complete, sorted: (system, model, "<view>_<split>", folder, run.json)."""
    found = []
    for system in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        for view in sorted(os.listdir(os.path.join(results_dir, system))) if os.path.isdir(os.path.join(results_dir, system)) else []:
            vdir = os.path.join(results_dir, system, view)
            for run in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
                folder = os.path.join(vdir, run)
                if run.endswith(".partial") or not os.path.isfile(os.path.join(folder, "run.json")):
                    continue
                with open(os.path.join(folder, "run.json"), encoding="utf-8") as f:
                    info = json.load(f)
                if info.get("status") == "complete":
                    model, _, split = run.rpartition("_")
                    found.append((system, model, f"{view}_{split}", folder, info))
    return found


def collect(results_dir="results", out="results/comparisons/summary.csv", root=".", skip_text=False, reference_texts=None, force=False,
            passages_path=None, relevance_models=False):
    rows = []
    for system, model, run, folder, info in finished_runs(results_dir):
        settings = info["settings"]
        qpath, ppath = recorded_question_path(root, settings["questions"]), os.path.join(folder, "predictions.jsonl")
        if not os.path.isfile(qpath):
            raise FileNotFoundError(f"{folder}: the question file {qpath} is not there (use --root)")
        rpath = os.path.join(folder, "report.json")
        report = None
        if os.path.isfile(rpath) and not force:
            with open(rpath, encoding="utf-8") as f:
                report = json.load(f)
            if not text_matches(ppath, report["inputs"]["predictions"]["sha256"]) or report.get("skip_text") != skip_text:
                report = None
        if report is None:
            report = evaluate(qpath, ppath, reference_texts=reference_texts, skip_text=skip_text, passages_path=passages_path,
                              relevance_models=relevance_models)
            report["skip_text"] = skip_text
            write_json(rpath, report)
        d, usage = report["diagnosis"], report.get("usage") or {}
        sim = ((report.get("text") or {}).get("semantic_similarity_cosine") or {}).get("mean")
        claims = report.get("text_only_image_claims") or {}
        served = sorted({r.get("served_by") for r in read_jsonl(ppath) if r.get("served_by")})
        view, _, split = run.rpartition("_")
        rows.append({"system": settings.get("baseline", system), "model": settings.get("model", model),
                     "model_id": (settings.get("model_settings") or {}).get("id", ""), "view": settings.get("view", view),
                     "split": settings.get("split", split), "status": info["status"], "questions": d["n_questions"],
                     "answered": info.get("answered", ""), "accuracy": d["accuracy"], "accuracy_ci_low": d["accuracy_95ci"][0],
                     "accuracy_ci_high": d["accuracy_95ci"][1], "macro_f1": d["macro_f1"], "missing": d["missing_predictions"],
                     "invalid": d["invalid_answers"], "refusals": d["invalid_answers_by_cause"]["refusals"],
                     "semantic_similarity": "" if sim is None else sim, "text_only_image_claim_rate": claims.get("rate", ""),
                     "tokens_in": (usage.get("tokens_in") or {}).get("total", ""), "tokens_out": (usage.get("tokens_out") or {}).get("total", ""),
                     "seconds": (usage.get("seconds") or {}).get("total", ""), "served_by": " | ".join(served),
                     "questions_sha256": settings.get("questions_sha256", ""), "folder": folder.replace("\\", "/")})
    write_csv(out, rows, SUMMARY_COLUMNS)
    return rows


def collect_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main collect", description="A report for every finished run and one table of all runs.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/comparisons/summary.csv")
    ap.add_argument("--root", default=".", help="the folder the question paths in run.json are relative to")
    ap.add_argument("--reference-texts", default=os.path.join("data", REFERENCE_TEXTS))
    ap.add_argument("--skip-text", action="store_true", help="no semantic similarity (no sentence model needed)")
    ap.add_argument("--passages", default=None)
    ap.add_argument("--relevance-models", action="store_true")
    ap.add_argument("--force", action="store_true", help="compute every report again")
    a = ap.parse_args(argv)
    refs = a.reference_texts if not a.skip_text and os.path.isfile(a.reference_texts) else None
    rows = collect(a.results, a.out, a.root, a.skip_text, refs, a.force, a.passages, a.relevance_models)
    log = get_logger("collect")
    log.info(f"{len(rows)} finished runs -> {a.out}")
    for r in rows:
        log.info(f"{r['system']} {r['model']} {r['view']} {r['split']}: accuracy {pct(r['accuracy'])} ({pct(r['accuracy_ci_low'])} to {pct(r['accuracy_ci_high'])}), "
                 f"macro-F1 {pct(r['macro_f1'])}, invalid {r['invalid']}, refusals {r['refusals']}")
    return rows
