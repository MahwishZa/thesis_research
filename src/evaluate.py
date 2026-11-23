#!/usr/bin/env python3
"""
One evaluation script for all systems (LLM only, standard RAG, RAG^2, proposed) and all views (image only, text only,
image + text).

Inputs
  --questions    data/views/<view>/<split>.jsonl from build.py views: the options of each case, the answer letter,
                 the gold class, the lesion group
  --predictions  one JSON line per question (run.py llm-only writes this format):
            {"id": "...", "pred": "C",                               # the chosen option letter
             "rationale": "free-text explanation",                  # optional, for the explanation metrics
             "query": "the text the system searched with",          # optional (RAG systems); the question is used if absent
             "contexts": [{"chunk_id": .., "title": .., "text": .., "tags": [..]}, ...]}   # retrieved passages, best first
             optional per line: "tokens_in", "tokens_out", "seconds" (cost and time of the call)
  --predictions-b  (optional) a second system on the same questions, for McNemar's test
  --cases   (optional) data/processed_cases/all_cases.csv, with --by: accuracy per subgroup (e.g. --by source facts_recorded)

The options are shuffled for each case, so a letter is first turned into its class with that case's own options.
A question with no prediction, or with an answer that is not one of its option letters, counts as wrong.

Metrics
  Diagnosis : accuracy (the main score) with a 95% bootstrap interval, macro-F1 with its interval, per-class precision,
              recall (= sensitivity), specificity and F1, confusion matrix, missing and invalid answers (the invalid ones split
              into refusals, by REFUSAL, and other invalid answers),
              McNemar's exact test against --pred-b. The bootstrap draws whole lesions, so photos of one lesion stay together.
              Also: how often each option letter was chosen (a model that prefers one position shows here), and cost/time.
              Accuracy is top-1 accuracy (the chosen option is the true class); no top-k accuracy is reported (decided 2026-11-22).
              When the predictions carry them: selective classification from the "evidence_sufficient" flag: coverage (share of all questions answered
              with a valid diagnosis marked "yes") and selective risk (share of those that are wrong: confident misdiagnoses).
              Several runs are compared with src/evaluate.py compare (McNemar for every pair, Holm's correction).
  Evidence  : RAG answers that carry "evidence": {"chunk_id", "quote"} (or null = "Evidence: none"). Quote accuracy (the quote is
              word for word in the cited passage), coverage and selective risk with verified evidence (an answer is covered when its
              quote is found in the cited passage; this replaces the yes/no flag), supported accuracy (correct and covered, after
              KILT-AC), and negative rejection (the share of "Evidence: none" when no retrieved passage is tagged with the true
              class, after RGB). Whether the quote really supports the diagnosis is NOT tested: an NLI check was tried on
              2026-11-23, failed its pre-set pass mark twice (0.83 pairwise, then 0.50 top-1) and was removed on the researcher's decision.
  Retrieval : Hit@k, Recall@k (with --passages), Precision@k, MRR, nDCG@k and context precision. Weak labels: a passage counts as relevant when its
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
answer counted only with right evidence: KILT-AC, Petroni et al. (2021) NAACL; negative rejection: RGB, Chen et al. (2024) AAAI
38:17754-17762; recall, precision and ranking of retrieval: BEIR, Thakur et al. (2021) NeurIPS Datasets and Benchmarks.

Every report records its input files with their SHA-256, so two reports can be checked to use the same question file.

Usage: python src/evaluate.py --questions data/views/multimodal/test.jsonl --predictions predictions.jsonl
       --out report.json [--predictions-b other.jsonl] [--relevance-models] [--reference-texts class_reference.json]
       [--passages data/knowledge_corpus/passages.jsonl]
       [--skip-text] [--cases data/processed_cases/all_cases.csv --by source facts_recorded]
       python src/evaluate.py compare --run name=questions,predictions --run name=questions,predictions --out compare.json
       (several runs on the same cases: McNemar's exact test for every pair, Holm-corrected; see the end of this file)
"""
import argparse
import hashlib
import itertools
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from common import CLASSES, OPTION_TEXT, get_logger, read_csv, read_jsonl, write_json

TEXT_TO_CLASS = {text: code for code, text in OPTION_TEXT.items()}
INVALID = "invalid"
# A refusal: the host flagged it (finish_reason "refusal...") or the reply says so in one of these fixed phrases (set on 2026-11-22
# from a real reply: "I'm sorry, I can't help with identifying or diagnosing images of skin lesions."). Only an invalid answer
# (no usable letter) is checked, so a valid answer that apologises is never counted.
REFUSAL = re.compile(r"\bI(?:['’]m| am) sorry\b|\bI(?: can['’]?t| cannot| can not| won['’]t| am unable to|['’]m unable to"
                     r"| am not able to|['’]m not able to) (?:help|assist|provide|identify|diagnose|answer|comply|interpret"
                     r"|analy[sz]e|evaluate)", re.I)


def invalid_causes(ids, classes, pred):
    """Splits the invalid answers into refusals (REFUSAL) and other invalid answers (a reply without a usable letter)."""
    out = {"refusals": [], "other": []}
    for i, c in zip(ids, classes):
        r = pred.get(i)
        if r is None or c != INVALID:
            continue
        refused = str(r.get("finish_reason") or "").startswith("refusal") or REFUSAL.search(str(r.get("raw_output") or ""))
        out["refusals" if refused else "other"].append(i)
    return {"refusals": len(out["refusals"]), "other_invalid": len(out["other"]), "refusal_ids": out["refusals"],
            "other_invalid_ids": out["other"]}
MEDCPT = ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", "ncbi/MedCPT-Cross-Encoder")


def load(path):
    """A JSONL file as {id: record}; stops if an id is repeated."""
    records = read_jsonl(path)
    by_id = {d["id"]: d for d in records}
    if len(by_id) != len(records):
        raise ValueError(f"{path}: {len(records) - len(by_id)} repeated ids")
    return by_id


def file_record(path):
    with open(path, "rb") as f:
        return {"path": str(path).replace("\\", "/"), "sha256": hashlib.sha256(f.read()).hexdigest()}


def gold_class(q):
    """The gold class of a question; stops if the answer letter does not point to it (a broken question file)."""
    code = TEXT_TO_CLASS[q["options"][q["answer"]]]
    if code != q["gold_label"]:
        raise ValueError(f"{q['id']}: answer {q['answer']} is {code}, gold label is {q['gold_label']}")
    return code


def predicted_class(q, p):
    """The class a prediction chose, read with this case's options; 'invalid' when there is no usable option letter."""
    letter = str((p or {}).get("pred") or "").strip().upper()
    return TEXT_TO_CLASS[q["options"][letter]] if letter in q["options"] else INVALID


def per_class(y, p):
    """Precision, recall (= sensitivity), specificity and F1 of each class; an invalid answer is wrong for every class."""
    pairs, n, rows = Counter(zip(y, p)), len(y), {}
    for c in CLASSES:
        tp = pairs[(c, c)]
        fn = sum(v for (t, q), v in pairs.items() if t == c and q != c)
        fp = sum(v for (t, q), v in pairs.items() if t != c and q == c)
        tn = n - tp - fn - fp
        prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        rows[c] = {"precision": prec, "recall": rec, "specificity": tn / (tn + fp) if tn + fp else 0.0,
                   "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn}
    return rows


def accuracy(y, p):
    return sum(t == q for t, q in zip(y, p)) / len(y)


def macro_f1(y, p):
    """The mean F1 over the eight classes (every class weighs the same)."""
    return sum(r["f1"] for r in per_class(y, p).values()) / len(CLASSES)


def percentile(sorted_values, q):
    """The q-th percentile with linear interpolation (the same as numpy's default)."""
    k = (len(sorted_values) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def bootstrap_ci(y, p, groups, fn, n=1000, seed=42):
    """95% percentile interval of fn. Whole lesions are drawn with replacement, so photos of one lesion stay together."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    keys, rng, values = list(members), random.Random(seed), []
    for _ in range(n):
        idx = [i for g in rng.choices(keys, k=len(keys)) for i in members[g]]
        values.append(fn([y[i] for i in idx], [p[i] for i in idx]))
    values.sort()
    return [round(percentile(values, 2.5), 4), round(percentile(values, 97.5), 4)]


def coverage(y, answers):
    """Selective classification: share of all questions answered with a valid diagnosis marked "evidence sufficient: yes"."""
    return sum(c != INVALID and flag == "yes" for c, flag in answers) / len(y)


def selective_risk(y, answers):
    """Share of wrong diagnoses among the questions marked "evidence sufficient: yes" (None when none is)."""
    covered = [(t, c) for t, (c, flag) in zip(y, answers) if c != INVALID and flag == "yes"]
    return sum(t != c for t, c in covered) / len(covered) if covered else None


def holm(p_values):
    """Holm's step-down correction: the i-th smallest of m p-values is multiplied by (m - i + 1), the results are made
    non-decreasing in that order and capped at 1. Returns the adjusted p-values in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def mcnemar_exact(y, pa, pb):
    """McNemar's exact test: a two-sided binomial test (p = 0.5) on the questions that exactly one system got right."""
    b = sum(a == t and c != t for t, a, c in zip(y, pa, pb))
    c = sum(a != t and x == t for t, a, x in zip(y, pa, pb))
    n = b + c
    p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": round(p_value, 6)}


def diagnosis_metrics(qa, pred, pred_b=None):
    unknown = sorted(set(pred) - set(qa))
    if unknown:
        raise ValueError(f"{len(unknown)} predictions are not in the question file, e.g. {unknown[:3]}")
    ids = list(qa)
    y = [gold_class(qa[i]) for i in ids]
    p = [predicted_class(qa[i], pred.get(i)) for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    rows = per_class(y, p)
    columns = [*CLASSES, INVALID]
    out = {"n_questions": len(ids),
           "missing_predictions": sum(i not in pred for i in ids),
           "invalid_answers": sum(i in pred and c == INVALID for i, c in zip(ids, p)),
           "invalid_answers_by_cause": invalid_causes(ids, p, pred),
           "accuracy": round(accuracy(y, p), 4), "accuracy_95ci": bootstrap_ci(y, p, groups, accuracy),
           "macro_f1": round(macro_f1(y, p), 4), "macro_f1_95ci": bootstrap_ci(y, p, groups, macro_f1),
           "per_class": {c: {k: (v if k == "support" else round(v, 3)) for k, v in r.items()} for c, r in rows.items()},
           "confusion_matrix": {"rows_true": CLASSES, "columns_predicted": columns,
                                "matrix": [[sum(t == r and q == c for t, q in zip(y, p)) for c in columns] for r in CLASSES]}}
    if any("evidence_sufficient" in (pred.get(i) or {}) for i in ids):
        answers = [(c, (pred.get(i) or {}).get("evidence_sufficient")) for i, c in zip(ids, p)]
        risk = selective_risk(y, answers)
        out["coverage"] = round(coverage(y, answers), 4)
        out["coverage_95ci"] = bootstrap_ci(y, answers, groups, coverage)
        out["n_covered"] = sum(c != INVALID and f == "yes" for c, f in answers)
        out["selective_risk"] = None if risk is None else round(risk, 4)
        out["evidence_flag_missing"] = sum(c != INVALID and f not in ("yes", "no") for c, f in answers)
    out["chosen_letters"] = {k: v for k, v in sorted(Counter(
        (str((pred.get(i) or {}).get("pred") or "").strip().upper() or "none") if c != INVALID else "invalid"
        for i, c in zip(ids, p)).items())}
    if pred_b is not None:
        out["mcnemar_vs_system_b"] = mcnemar_exact(y, p, [predicted_class(qa[i], pred_b.get(i)) for i in ids])
    return out


def mean(values):
    return sum(values) / len(values) if values else 0.0


def usage_summary(qa, pred):
    """Totals and means of tokens and seconds, for the predictions that report them (None when none does)."""
    out = {}
    for key in ("tokens_in", "tokens_out", "seconds"):
        values = [float(pred[i][key]) for i in qa if i in pred and pred[i].get(key) is not None]
        if values:
            out[key] = {"total": round(sum(values), 2), "mean": round(mean(values), 2), "n": len(values)}
    return out or None


def facts_recorded(case):
    """'all' when age, sex and body site are recorded, else 'some missing' (missing facts are common in NV and MEL)."""
    return "all" if all(case.get(k) for k in ("age", "sex", "site")) else "some missing"


def subgroup_accuracy(qa, pred, cases, columns):
    """Accuracy within each value of each column of the case table (or 'facts_recorded'), with the number of questions."""
    out = {}
    for col in columns:
        groups = defaultdict(list)
        for i, q in qa.items():
            case = cases.get(i)
            if case is None:
                raise ValueError(f"{i} is not in the case table")
            if col != "facts_recorded" and col not in case:
                raise ValueError(f"the case table has no column {col!r}")
            groups[facts_recorded(case) if col == "facts_recorded" else case[col]].append(
                predicted_class(q, pred.get(i)) == gold_class(q))
        out[col] = {g: {"n": len(v), "accuracy": round(mean(v), 4)} for g, v in sorted(groups.items())}
    return out


def relevant_counts(passages):
    """How many corpus passages carry each class tag (the weak labels): the denominator of Recall@k."""
    counts = Counter(t for p in passages for t in set(p.get("tags") or []))
    return {c: counts[c] for c in CLASSES}


def retrieval_metrics(qa, pred, ks=(1, 3, 5, 10), n_relevant=None):
    rows = [(qa[i]["gold_label"], pred[i]["contexts"]) for i in qa if i in pred and pred[i].get("contexts")]
    if not rows:
        return None
    res = {}
    ideal = [1 / math.log2(j + 2) for j in range(max(ks))]
    for k in ks:
        hit, prec, nd, rec = [], [], [], []
        for g, ctx in rows:
            rel = [1 if g in c.get("tags", []) else 0 for c in ctx[:k]]
            hit.append(1.0 if any(rel) else 0.0)
            prec.append(sum(rel) / k)
            if n_relevant:
                rec.append(sum(rel) / n_relevant[g] if n_relevant.get(g) else 0.0)
            nd.append(sum(r / math.log2(j + 2) for j, r in enumerate(rel)) / sum(ideal[:k]))
        res[f"hit@{k}"], res[f"precision@{k}"], res[f"ndcg@{k}"] = (round(mean(v), 4) for v in (hit, prec, nd))
        if n_relevant:
            res[f"recall@{k}"] = round(mean(rec), 4)
    mrr, cp = [], []
    for g, ctx in rows:
        rel = [1 if g in c.get("tags", []) else 0 for c in ctx]
        mrr.append(next((1 / (j + 1) for j, r in enumerate(rel) if r), 0.0))
        # context precision: the mean of precision@j over the positions j that hold a relevant passage
        cp.append(mean([sum(rel[:j + 1]) / (j + 1) for j, r in enumerate(rel) if r]))
    res["mrr"], res["context_precision"] = round(mean(mrr), 4), round(mean(cp), 4)
    res["n_questions_with_contexts"] = len(rows)
    return res


def medcpt_scorers(models=MEDCPT):
    """Two scorers built on MedCPT: the cosine of the query and passage embeddings, and the cross-encoder score (higher = more
    relevant; not a probability). Each takes (query, passages) and returns one score per passage."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    q_tok, q_enc = AutoTokenizer.from_pretrained(models[0]), AutoModel.from_pretrained(models[0]).eval()
    a_tok, a_enc = AutoTokenizer.from_pretrained(models[1]), AutoModel.from_pretrained(models[1]).eval()
    c_tok, c_enc = AutoTokenizer.from_pretrained(models[2]), AutoModelForSequenceClassification.from_pretrained(models[2]).eval()

    def articles(passages):
        return [[c.get("title") or "", c["text"]] for c in passages]

    @torch.no_grad()
    def cosine(query, passages):
        q = q_enc(**q_tok([query], truncation=True, padding=True, return_tensors="pt", max_length=64)).last_hidden_state[:, 0]
        a = a_enc(**a_tok(articles(passages), truncation=True, padding=True, return_tensors="pt",
                          max_length=512)).last_hidden_state[:, 0]
        return torch.nn.functional.cosine_similarity(q, a).tolist()

    @torch.no_grad()
    def cross(query, passages):
        pairs = [[query, f"{c.get('title') or ''} {c['text']}".strip()] for c in passages]
        return c_enc(**c_tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)).logits.squeeze(-1).tolist()

    return {"medcpt_cosine": cosine, "medcpt_cross_encoder": cross}


def relevance_metrics(qa, pred, scorers):
    """Mean relevance of the retrieved passages to the query the system searched with (its 'query', else the question):
    first averaged over the passages of a question, then over the questions."""
    ids = [i for i in qa if i in pred and pred[i].get("contexts")]
    if not ids:
        return None
    scores, source = defaultdict(list), Counter()
    for i in ids:
        query, passages = pred[i].get("query") or qa[i]["question"], pred[i]["contexts"]
        source["query" if pred[i].get("query") else "question"] += 1
        for name, fn in scorers.items():
            scores[name].append(mean(fn(query, passages)))
    return {**{k: round(mean(v), 4) for k, v in scores.items()}, "n_questions": len(ids), "query_used": dict(source)}


SIMILARITY_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"     # a biomedical sentence encoder (PubMedBERT tuned on MS MARCO)


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def sentence_encoder(model=SIMILARITY_MODEL):
    """A function that turns a list of texts into a list of embedding vectors (downloads the model on first use)."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model)

    def encode(texts):
        return enc.encode(list(texts), convert_to_numpy=True).tolist()
    # the model reads at most max_seq_length tokens of a text and ignores the rest: counted so a report can say so
    encode.max_tokens = enc.max_seq_length
    encode.n_tokens = lambda text: len(enc.tokenizer(text)["input_ids"])
    return encode


def reference_lists(reference_texts):
    """{class: [text, ...]} from any of the accepted forms: {class: text}, {class: {"text": ...}} or the file of
    src/build.py references, {class: {"texts": [{"text": ...}, ...]}}."""
    out = {}
    for c, r in reference_texts.items():
        if isinstance(r, str):
            out[c] = [r]
        elif "texts" in r:
            out[c] = [t["text"] for t in r["texts"]]
        else:
            out[c] = [r["text"]]
        if not out[c]:
            raise ValueError(f"the class {c} has no reference text")
    return out


def semantic_similarity(qa, pred, reference_texts, encode):
    """Cosine similarity between the embedding of each model explanation ("rationale") and the embeddings of the reference texts of
    the question's true class, keeping the highest (the best-matching reference, as in multi-reference scoring); then the mean
    over the questions, with a 95% interval over lesions and the mean per class.
    reference_texts: see reference_lists; the texts are quoted from a cited source, never written by us."""
    ids = [i for i in qa if (pred.get(i) or {}).get("rationale")]
    if not ids:
        return None
    refs = reference_lists(reference_texts)
    missing = sorted({qa[i]["gold_label"] for i in ids} - set(refs))
    if missing:
        raise ValueError(f"no reference text for the classes {missing}")
    classes = sorted({qa[i]["gold_label"] for i in ids})
    flat = [(c, t) for c in classes for t in refs[c]]
    ref_vec = defaultdict(list)
    for (c, _), v in zip(flat, encode([t for _, t in flat])):
        ref_vec[c].append(v)
    rationales = [pred[i]["rationale"] for i in ids]
    sims = [max(cosine(h, r) for r in ref_vec[qa[i]["gold_label"]]) for i, h in zip(ids, encode(rationales))]
    groups = [qa[i].get("group_id", i) for i in ids]
    per_class = defaultdict(list)
    for i, v in zip(ids, sims):
        per_class[qa[i]["gold_label"]].append(v)
    out = {"mean": round(mean(sims), 4), "95ci": bootstrap_ci(sims, sims, groups, lambda a, b: mean(a)),
           "per_class": {c: round(mean(per_class[c]), 4) for c in CLASSES if c in per_class},
           "n_questions_with_rationale": len(ids), "n_reference_texts": {c: len(refs[c]) for c in classes},
           "aggregation": "highest cosine over the reference texts of the true class"}
    if hasattr(encode, "n_tokens"):
        out["longer_than_the_model_reads"] = {"max_tokens": encode.max_tokens,
                                              "reference_texts": sum(encode.n_tokens(t) > encode.max_tokens for _, t in flat),
                                              "rationales": sum(encode.n_tokens(t) > encode.max_tokens for t in rationales)}
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Evidence (RAG systems): the quoted sentence and whether it is really in the cited passage. Decided 2026-11-23 (option B): the answer
# is "covered" only if its quote is verified, instead of a yes/no flag. The NLI "support" step was removed 2026-11-24 (failed its check).
# ---------------------------------------------------------------------------------------------------------------------------
MIN_QUOTE_WORDS = 5                                      # a shorter "quote" is too easy to find by chance


def normalise_quote(text):
    """The words of a text in lower case: every run of other characters (punctuation, quote marks, dashes, spaces) becomes one space, so
    a quote that drops a comma or writes "blue gray" for "blue-gray" is still the same words."""
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text or "").lower()).strip()


def quote_in_passage(quote, passage):
    """True when the quote is in the passage word for word (after normalise_quote). A quote with "..." or an ellipsis is checked
    piece by piece, the pieces in order; the words of all pieces together must be at least MIN_QUOTE_WORDS."""
    pieces = [normalise_quote(x) for x in re.split(r"\[?\.\.\.\]?|\u2026", quote or "")]
    pieces = [x for x in pieces if x]
    if not pieces or sum(len(x.split()) for x in pieces) < MIN_QUOTE_WORDS:
        return False
    text, at = f" {normalise_quote(passage)} ", 0
    for piece in pieces:
        found = text.find(f" {piece} ", at)                    # whole words only: no match inside a longer word
        if found < 0:
            return False
        at = found + len(piece) + 1
    return True


def cited_passage(p):
    """The text of the passage a prediction cites in its evidence, or None."""
    ev = p.get("evidence") or {}
    for c in p.get("contexts") or []:
        if ev.get("chunk_id") is not None and str(c.get("chunk_id")) == str(ev["chunk_id"]):
            return c.get("text")
    return None


def evidence_metrics(qa, pred):
    """RAG answers that carry an "evidence" field: {"chunk_id": ..., "quote": "..."}, or null for "Evidence: none".
    An answer is covered when it has a valid diagnosis and a quote that is word for word in the cited passage. No model is used.
    Nothing here uses the true class, except negative rejection and the errors counted among the covered answers."""
    ids = [i for i in qa if i in pred and "evidence" in pred[i]]
    if not ids:
        return None
    y = [gold_class(qa[i]) for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    given, found, answers = [], [], []
    for i in ids:
        p = pred[i]
        code = predicted_class(qa[i], p)
        quote = ((p.get("evidence") or {}).get("quote") or "").strip()
        passage = cited_passage(p)
        g = code != INVALID and bool(quote)
        f = g and passage is not None and quote_in_passage(quote, passage)
        given.append(g)
        found.append(f)
        answers.append((code, "yes" if f else "no"))
    n, valid = len(ids), sum(predicted_class(qa[i], pred[i]) != INVALID for i in ids)
    right_and_covered = sum(f and t == a[0] for f, t, a in zip(found, y, answers))
    risk = selective_risk(y, answers)
    irrelevant = [k for k, i in enumerate(ids) if not any(y[k] in (c.get("tags") or []) for c in pred[i].get("contexts") or [])]
    out = {"n_questions": n, "valid_answers": valid,
           "evidence_given": sum(given), "evidence_none": sum(1 for i in ids if not (pred[i].get("evidence") or {}).get("quote")),
           "quote_found": sum(found), "quote_accuracy": round(sum(found) / sum(given), 4) if sum(given) else None,
           "coverage_verified": round(coverage(y, answers), 4), "coverage_verified_95ci": bootstrap_ci(y, answers, groups, coverage),
           "n_covered": sum(found), "selective_risk_verified": None if risk is None else round(risk, 4),
           "supported_accuracy": round(right_and_covered / n, 4),
           "negative_rejection": None, "n_without_relevant_passage": len(irrelevant)}
    if irrelevant:
        out["negative_rejection"] = round(sum(not (pred[ids[k]].get("evidence") or {}).get("quote") for k in irrelevant)
                                          / len(irrelevant), 4)
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Image claims in the text-only view (researcher's decision, 2026-11-24). In that view the model gets no picture, so a reasoning that
# says "dermoscopy reveals ..." or "the image shows ..." describes something that does not exist. Found by a word list, so it is a
# lower bound (a paraphrase is missed) and it can flag a harmless sentence: report it as approximate, never as "the hallucination rate".
# ---------------------------------------------------------------------------------------------------------------------------
IMAGE_CLAIM = re.compile(
    r"\bdermos?cop\w*|\bdermatos?cop\w*|\btrichoscop\w*"                                    # dermoscopy, dermoscopic, dermatoscopic
    r"|\b(?:the|this|these|an?|given|provided|attached|dermoscopic|clinical)\s+(?:images?|pictures?|photos?|photographs?)\b"
    r"|\b(?:images?|pictures?|photos?|photographs?)\s+(?:shows?|reveals?|displays?|demonstrates?|depicts?|exhibits?)\b", re.I)
NOT_A_CLAIM = re.compile(r"\b(?:no|not|without|absence|absent|lack(?:s|ing)?|unavailable|cannot|can['’]?t|unable|never|neither|nor|"
                         r"if|would|could|unless|needs?|needed|requires?|required|limited)\b|n['’]t\b", re.I)


def claims_an_image(text):
    """True when a sentence of the text describes an image or a dermoscopy and does not deny or hypothesise it."""
    sentences = re.split(r"(?<=[.!?;:])\s+|\n+", text or "")
    return any(IMAGE_CLAIM.search(x) and not NOT_A_CLAIM.search(x) for x in sentences)


def image_claim_metrics(qa, pred):
    """The share of text-only reasonings that claim an image. A record counts when it says "view": "text_only" (run.py writes it) and has a
    reasoning. Returns None when there is no such record."""
    ids = [i for i in qa if i in pred and pred[i].get("view") == "text_only" and (pred[i].get("rationale") or "").strip()]
    if not ids:
        return None
    flags = [1.0 if claims_an_image(pred[i]["rationale"]) else 0.0 for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    return {"n_text_only_with_reasoning": len(ids), "n_claiming_an_image": int(sum(flags)), "rate": round(mean(flags), 4),
            "rate_95ci": bootstrap_ci(flags, flags, groups, lambda a, b: mean(a)),
            "ids_claiming_an_image": [i for i, f in zip(ids, flags) if f],
            "method": "word list; approximate lower bound"}


def text_metrics(qa, pred, reference_texts, encode=None):
    """Semantic similarity of the explanations (when reference texts are given)."""
    out = {}
    if reference_texts:
        sim = semantic_similarity(qa, pred, reference_texts, encode or sentence_encoder())
        if sim:
            out["semantic_similarity_cosine"] = {**sim, "model": SIMILARITY_MODEL if encode is None else "given"}
    return out


def evaluate(qa_path, pred_path, pred_b_path=None, cases_path=None, by=(), relevance_models=False, reference_texts=None,
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
    if by:
        if not cases_path:
            raise ValueError("--by needs --cases (data/processed_cases/all_cases.csv)")
        rep["subgroups"] = subgroup_accuracy(qa, pred, {r["isic_id"]: r for r in read_csv(cases_path)}, by)
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
    ap = argparse.ArgumentParser(prog="evaluate.py", description="Evaluates one system's predictions on one question file.")
    ap.add_argument("--questions", required=True, help="data/views/<view>/<split>.jsonl")
    ap.add_argument("--predictions", required=True, help="the predictions of one system, one JSON line per question")
    ap.add_argument("--predictions-b", default=None, help="a second system on the same questions, for McNemar's test")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--cases", default=None, help="the case table, for --by (data/processed_cases/all_cases.csv)")
    ap.add_argument("--by", nargs="*", default=[], help="subgroups: columns of the case table, or facts_recorded")
    ap.add_argument("--relevance-models", action="store_true", help="also score relevance with MedCPT (downloads 3 models)")
    ap.add_argument("--reference-texts", default=None)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--answered-only", action="store_true",
                    help="trial runs: score only the questions that have a prediction (without it, an unasked question counts as wrong)")
    ap.add_argument("--passages", default=None,
                    help="the corpus (data/knowledge_corpus/passages.jsonl): with it Recall@k is reported, as the share of all the "
                         "corpus passages tagged with the true class that are found in the top k")
    a = ap.parse_args(argv)
    rep = evaluate(a.questions, a.predictions, a.predictions_b, a.cases, a.by, a.relevance_models, a.reference_texts,
                   a.skip_text, a.passages, a.answered_only)
    write_json(a.out, rep)
    log = get_logger("evaluate")
    d = rep["diagnosis"]
    log.info(f"accuracy {d['accuracy']} (95% CI {d['accuracy_95ci'][0]}-{d['accuracy_95ci'][1]}) | macro-F1 {d['macro_f1']} | "
             f"melanoma recall {d['per_class']['MEL']['recall']} | missing {d['missing_predictions']}, invalid {d['invalid_answers']} (refusals {d['invalid_answers_by_cause']['refusals']})")
    if rep["text_only_image_claims"]:
        c = rep["text_only_image_claims"]
        log.info(f"text-only reasonings that describe an image: {c['n_claiming_an_image']} of {c['n_text_only_with_reasoning']} "
                 f"(approximate; ids in the report)")
    if rep["retrieval"]:
        log.info(f"retrieval: { {k: rep['retrieval'][k] for k in ('hit@5', 'ndcg@5', 'mrr', 'context_precision')} }")
        if "relevance" in rep:
            log.info(f"relevance: {rep['relevance']}")
    return rep


# ---------------------------------------------------------------------------------------------------------------------------
# compare: several runs on the same cases (python src/evaluate.py compare ...)
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
#   python src/evaluate.py compare --out results/llm_only/compare_text_only_test.json       --run claude=data/views/text_only/test.jsonl,results/llm_only/claude-sonnet-5-5/text_only_test/predictions.jsonl       --run gpt-4o=data/views/text_only/test.jsonl,results/llm_only/gpt-4o/text_only_test/predictions.jsonl

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
    ap = argparse.ArgumentParser(prog="evaluate.py compare", description="Compares several runs on the same cases (McNemar + Holm).")
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
        log.info(f"{p['a']} {p['accuracy_a']} vs {p['b']} {p['accuracy_b']}: only a right {p['a_right_b_wrong']}, only b right "
                 f"{p['a_wrong_b_right']}, p = {p['p_value']}, Holm p = {p['p_holm']}")
    return rep


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "compare":
        return compare_main(argv[1:])
    return report_main(argv)


if __name__ == "__main__":
    main()
