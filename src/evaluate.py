#!/usr/bin/env python3
"""
One evaluation script for all systems (LLM only, standard RAG, RAG^2, proposed) and all views (image only, text only,
image + text).

Inputs
  --qa      qa_<view>_<split>.jsonl from build_qa.py: the options of each case, the answer letter, the gold class, the lesion group
  --pred    predictions, one JSON line per question:
            {"id": "...", "pred": "C",                               # the chosen option letter
             "rationale": "free-text explanation",                  # optional, for the explanation metrics
             "query": "the text the system searched with",          # optional (RAG systems); the question is used if absent
             "contexts": [{"chunk_id": .., "title": .., "text": .., "tags": [..]}, ...]}   # retrieved passages, best first
  --pred-b  (optional) a second system on the same questions, for McNemar's test

The options are shuffled for each case, so a letter is first turned into its class with that case's own options.
A question with no prediction, or with an answer that is not one of its option letters, counts as wrong.

Metrics
  Diagnosis : accuracy (the main score) with a 95% bootstrap interval, macro-F1 with its interval, per-class precision,
              sensitivity (recall), specificity and F1, melanoma sensitivity, confusion matrix, missing and invalid answers,
              McNemar's exact test against --pred-b. The bootstrap draws whole lesions, so photos of one lesion stay together.
  Retrieval : Hit@k, Precision@k, MRR, nDCG@k and context precision. Weak labels: a passage counts as relevant when its
              tags contain the gold class. nDCG@k assumes the corpus holds at least k relevant passages for each class.
  Relevance : how well each retrieved passage fits the query: Jaccard similarity of the content words (free, no model);
              with --relevance-models also the cosine similarity of MedCPT embeddings and the MedCPT cross-encoder score.
  Text      : semantic similarity (cosine of sentence embeddings, rationale vs a cited reference text), BERTScore F1
              (optional), RAGAS faithfulness / context relevance / answer relevancy (optional; an LLM judge that costs credit).

Sources: McNemar (1947) Psychometrika 12:153-157; Jaccard (1912) New Phytologist 11:37-50; nDCG: Jarvelin & Kekalainen (2002)
ACM TOIS 20:422-446; MedCPT: Jin et al. (2023) Bioinformatics 39:btad651; RAGAS: Es et al. (2024) EACL demos 150-158.

Usage: python src/evaluate.py --qa data/qa/qa_multimodal_test.jsonl --pred preds.jsonl --out report.json
       [--pred-b preds_other.jsonl] [--relevance-models] [--reference-texts class_reference.json] [--bertscore] [--ragas]
       [--skip-text]
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

from build_qa import CLASSES, OPTION_TEXT

TEXT_TO_CLASS = {text: code for code, text in OPTION_TEXT.items()}
INVALID = "invalid"
MEDCPT = ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", "ncbi/MedCPT-Cross-Encoder")


def load(path):
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in (json.loads(line) for line in f if line.strip())}


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
    """Precision, sensitivity (recall), specificity and F1 of each class; an invalid answer is wrong for every class."""
    pairs, n, rows = Counter(zip(y, p)), len(y), {}
    for c in CLASSES:
        tp = pairs[(c, c)]
        fn = sum(v for (t, q), v in pairs.items() if t == c and q != c)
        fp = sum(v for (t, q), v in pairs.items() if t != c and q == c)
        tn = n - tp - fn - fp
        prec, sens = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        rows[c] = {"precision": prec, "sensitivity": sens, "specificity": tn / (tn + fp) if tn + fp else 0.0,
                   "f1": 2 * prec * sens / (prec + sens) if prec + sens else 0.0, "support": tp + fn}
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
           "accuracy": round(accuracy(y, p), 4), "accuracy_95ci": bootstrap_ci(y, p, groups, accuracy),
           "macro_f1": round(macro_f1(y, p), 4), "macro_f1_95ci": bootstrap_ci(y, p, groups, macro_f1),
           "melanoma_sensitivity": round(rows["MEL"]["sensitivity"], 4),
           "per_class": {c: {k: (v if k == "support" else round(v, 3)) for k, v in r.items()} for c, r in rows.items()},
           "confusion_matrix": {"rows_true": CLASSES, "columns_predicted": columns,
                                "matrix": [[sum(t == r and q == c for t, q in zip(y, p)) for c in columns] for r in CLASSES]}}
    if pred_b is not None:
        out["mcnemar_vs_system_b"] = mcnemar_exact(y, p, [predicted_class(qa[i], pred_b.get(i)) for i in ids])
    return out


def mean(values):
    return sum(values) / len(values) if values else 0.0


def retrieval_metrics(qa, pred, ks=(1, 3, 5, 10)):
    rows = [(qa[i]["gold_label"], pred[i]["contexts"]) for i in qa if i in pred and pred[i].get("contexts")]
    if not rows:
        return None
    res = {}
    ideal = [1 / math.log2(j + 2) for j in range(max(ks))]
    for k in ks:
        hit, prec, nd = [], [], []
        for g, ctx in rows:
            rel = [1 if g in c.get("tags", []) else 0 for c in ctx[:k]]
            hit.append(1.0 if any(rel) else 0.0)
            prec.append(sum(rel) / k)
            nd.append(sum(r / math.log2(j + 2) for j, r in enumerate(rel)) / sum(ideal[:k]))
        res[f"hit@{k}"], res[f"precision@{k}"], res[f"ndcg@{k}"] = (round(mean(v), 4) for v in (hit, prec, nd))
    mrr, cp = [], []
    for g, ctx in rows:
        rel = [1 if g in c.get("tags", []) else 0 for c in ctx]
        mrr.append(next((1 / (j + 1) for j, r in enumerate(rel) if r), 0.0))
        # context precision: the mean of precision@j over the positions j that hold a relevant passage
        cp.append(mean([sum(rel[:j + 1]) / (j + 1) for j, r in enumerate(rel) if r]))
    res["mrr"], res["context_precision"] = round(mean(mrr), 4), round(mean(cp), 4)
    res["n_questions_with_contexts"] = len(rows)
    return res


def content_words(text, stop_words):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - stop_words


def jaccard(a, b, stop_words=frozenset()):
    """Jaccard similarity of the two sets of content words: shared words / all words (0 when both are empty)."""
    x, y = content_words(a, stop_words), content_words(b, stop_words)
    return len(x & y) / len(x | y) if x | y else 0.0


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


def relevance_metrics(qa, pred, scorers=None, stop_words=None):
    """Mean relevance of the retrieved passages to the query the system searched with (its 'query', else the question):
    first averaged over the passages of a question, then over the questions."""
    ids = [i for i in qa if i in pred and pred[i].get("contexts")]
    if not ids:
        return None
    if stop_words is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
    scores, source = defaultdict(list), Counter()
    for i in ids:
        query, passages = pred[i].get("query") or qa[i]["question"], pred[i]["contexts"]
        source["query" if pred[i].get("query") else "question"] += 1
        scores["jaccard_content_words"].append(mean([jaccard(query, c["text"], frozenset(stop_words)) for c in passages]))
        for name, fn in (scorers or {}).items():
            scores[name].append(mean(fn(query, passages)))
    return {**{k: round(mean(v), 4) for k, v in scores.items()}, "n_questions": len(ids), "query_used": dict(source)}


def text_metrics(qa, pred, reference_texts, use_bertscore=False, use_ragas=False):
    """reference_texts: {gold class: a reference text quoted from a cited source}."""
    ids = [i for i in qa if i in pred and pred[i].get("rationale")]
    out = {}
    if reference_texts and ids:
        from sentence_transformers import SentenceTransformer, util
        enc = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")   # biomedical sentence encoder
        hyp = [pred[i]["rationale"] for i in ids]
        ref = [reference_texts[qa[i]["gold_label"]] for i in ids]
        eh, er = enc.encode(hyp, convert_to_tensor=True), enc.encode(ref, convert_to_tensor=True)
        out["semantic_similarity_cosine"] = round(float(util.cos_sim(eh, er).diag().mean()), 4)
        if use_bertscore:
            from bert_score import score
            _, _, f = score(hyp, ref, lang="en", model_type="microsoft/deberta-xlarge-mnli")
            out["bertscore_f1"] = round(float(f.mean()), 4)
    ctx_ids = [i for i in qa if i in pred and pred[i].get("contexts")]
    if use_ragas and ctx_ids:
        # LLM-as-judge metrics (needs an LLM API key configured for ragas; costs credit)
        import numpy as np
        from ragas import EvaluationDataset
        from ragas import evaluate as ragas_eval
        from ragas.metrics import ContextRelevance, Faithfulness, ResponseRelevancy
        ds = EvaluationDataset.from_list([{"user_input": qa[i]["question"],
                                           "response": pred[i].get("rationale") or pred[i].get("pred", ""),
                                           "retrieved_contexts": [x["text"] for x in pred[i]["contexts"]]} for i in ctx_ids])
        r = ragas_eval(ds, metrics=[Faithfulness(), ResponseRelevancy(), ContextRelevance()])
        out["ragas"] = {k: round(float(np.nanmean(v)), 4) for k, v in r.to_pandas().select_dtypes("number").items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-b", default=None)
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--relevance-models", action="store_true", help="also score relevance with MedCPT (downloads 3 models)")
    ap.add_argument("--reference-texts", default=None)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--skip-text", action="store_true")
    a = ap.parse_args(argv)
    qa, pred = load(a.qa), load(a.pred)
    rep = {"diagnosis": diagnosis_metrics(qa, pred, load(a.pred_b) if a.pred_b else None),
           "retrieval": retrieval_metrics(qa, pred)}
    if rep["retrieval"]:
        rep["relevance"] = relevance_metrics(qa, pred, medcpt_scorers() if a.relevance_models else None)
    if not a.skip_text:
        refs = None
        if a.reference_texts:
            with open(a.reference_texts, encoding="utf-8") as f:
                refs = json.load(f)
        rep["text"] = text_metrics(qa, pred, refs, a.bertscore, a.ragas)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    d = rep["diagnosis"]
    print(f"accuracy {d['accuracy']} (95% CI {d['accuracy_95ci'][0]}-{d['accuracy_95ci'][1]}) | macro-F1 {d['macro_f1']} | "
          f"melanoma sensitivity {d['melanoma_sensitivity']} | missing {d['missing_predictions']}, invalid {d['invalid_answers']}")
    if rep["retrieval"]:
        print("retrieval:", {k: rep["retrieval"][k] for k in ("hit@5", "ndcg@5", "mrr", "context_precision")})
        print("relevance:", rep["relevance"])
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])
