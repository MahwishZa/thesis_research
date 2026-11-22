#!/usr/bin/env python3
"""
One evaluation script for all baselines (LLM-only, Standard RAG, RAG^2, Proposed) x
modalities (image-only, text-only, multimodal).

Inputs
  --qa      qa_<modality>_test.jsonl      (gold answers, from build_qa.py)
  --pred    predictions JSONL, one line per question:
            {"id": "...", "pred": "C",                        # chosen option letter
             "probs": {"A":0.1,...},                         # optional, for AUC
             "rationale": "free-text explanation",           # optional, for text metrics
             "contexts": [{"chunk_id":..,"text":..,"tags":[..]}, ...]}   # retrieved evidence (RAG systems)
  --pred-b  (optional) a second system for McNemar significance testing

Metrics
  Diagnosis : accuracy, balanced accuracy, macro-F1, per-class precision/recall/specificity,
              macro AUC (if probs), MCC, Cohen kappa, 95% bootstrap CI, McNemar vs --pred-b.
  Retrieval : Recall@k, Precision@k, MRR, nDCG@k, context precision (rank-weighted)  [weak labels:
              a chunk is relevant if its tags contain the gold diagnosis].
  Text      : semantic similarity (cosine of sentence embeddings, rationale vs reference),
              context relevance (embedding cosine question<->contexts),
              BERTScore F1 (optional), RAGAS faithfulness / context relevance / answer relevancy (optional, LLM judge).

Usage: python evaluate.py --qa qa/qa_multimodal_test.jsonl --pred preds_rag2.jsonl --out report_rag2.json
       [--reference-texts class_reference.json] [--ragas] [--bertscore]
"""
import argparse, json, math
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef, cohen_kappa_score, roc_auc_score)

LETTERS = list("ABCDEFGH")


def load(p):
    return {d["id"]: d for d in map(json.loads, open(p))}


def bootstrap_ci(y, p, fn, n=1000, seed=42):
    r = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); vals = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y)); vals.append(fn(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def mcnemar(y, pa, pb):
    a_ok, b_ok = np.asarray(pa) == np.asarray(y), np.asarray(pb) == np.asarray(y)
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    if b + c == 0:
        return dict(b=b, c=c, p_value=1.0)
    from scipy.stats import binomtest
    return dict(a_right_b_wrong=b, a_wrong_b_right=c, p_value=float(binomtest(b, b + c, 0.5).pvalue))


def diagnosis_metrics(qa, pred, pred_b=None):
    ids = [i for i in qa if i in pred]
    y = [qa[i]["answer"] for i in ids]; p = [pred[i].get("pred", "?") for i in ids]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LETTERS, zero_division=0)
    cm = confusion_matrix(y, p, labels=LETTERS)
    spec = [float((cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]) / max(cm.sum() - cm[k].sum(), 1)) for k in range(8)]
    names = {L: qa[ids[0]]["options"][L] for L in LETTERS}
    out = dict(n=len(ids), missing_predictions=len(qa) - len(ids),
               accuracy=round(accuracy_score(y, p), 4),
               balanced_accuracy=round(balanced_accuracy_score(y, p), 4),
               balanced_accuracy_95ci=bootstrap_ci(y, p, balanced_accuracy_score),
               macro_f1=round(f1_score(y, p, labels=LETTERS, average="macro", zero_division=0), 4),
               macro_f1_95ci=bootstrap_ci(y, p, lambda a, b: f1_score(a, b, labels=LETTERS, average="macro", zero_division=0)),
               mcc=round(matthews_corrcoef(y, p), 4), cohen_kappa=round(cohen_kappa_score(y, p), 4),
               per_class={names[L]: dict(precision=round(pr[k], 3), recall_sensitivity=round(rc[k], 3),
                                         specificity=round(spec[k], 3), f1=round(f1[k], 3), support=int(sup[k]))
                          for k, L in enumerate(LETTERS)},
               confusion_matrix=dict(labels=LETTERS, matrix=cm.tolist()))
    if all("probs" in pred[i] for i in ids):
        P = np.array([[pred[i]["probs"].get(L, 0) for L in LETTERS] for i in ids]); P = P / P.sum(1, keepdims=True)
        out["macro_auc_ovr"] = round(roc_auc_score(y, P, multi_class="ovr", labels=LETTERS), 4)
    if pred_b:
        pb = [pred_b.get(i, {}).get("pred", "?") for i in ids]
        out["mcnemar_vs_system_b"] = mcnemar(y, p, pb)
    return out


def retrieval_metrics(qa, pred, ks=(1, 3, 5, 10)):
    rows = [(qa[i]["gold_label"], pred[i]["contexts"]) for i in qa if i in pred and pred[i].get("contexts")]
    if not rows:
        return None
    res = {}
    for k in ks:
        rec, prec, nd = [], [], []
        for g, ctx in rows:
            rel = [1 if g in c.get("tags", []) else 0 for c in ctx[:k]]
            rec.append(1.0 if any(rel) else 0.0); prec.append(sum(rel) / k)
            dcg = sum(r / math.log2(j + 2) for j, r in enumerate(rel)); ideal = sum(1 / math.log2(j + 2) for j in range(min(k, len(ctx))))
            nd.append(dcg / ideal if ideal else 0)
        res[f"hit_recall@{k}"] = round(float(np.mean(rec)), 4); res[f"precision@{k}"] = round(float(np.mean(prec)), 4)
        res[f"ndcg@{k}"] = round(float(np.mean(nd)), 4)
    mrr, cp = [], []
    for g, ctx in rows:
        rel = [1 if g in c.get("tags", []) else 0 for c in ctx]
        mrr.append(next((1 / (j + 1) for j, r in enumerate(rel) if r), 0))
        # RAGAS-style context precision: mean of precision@j at every relevant position
        hits = [sum(rel[:j + 1]) / (j + 1) for j, r in enumerate(rel) if r]
        cp.append(float(np.mean(hits)) if hits else 0.0)
    res["mrr"] = round(float(np.mean(mrr)), 4); res["context_precision"] = round(float(np.mean(cp)), 4)
    res["n_questions_with_contexts"] = len(rows)
    return res


def text_metrics(qa, pred, reference_texts, use_bertscore=False, use_ragas=False):
    """reference_texts: {gold_label: reference explanation written from a cited source}."""
    from sentence_transformers import SentenceTransformer, util
    enc = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")   # biomedical sentence encoder
    ids = [i for i in qa if i in pred and pred[i].get("rationale")]
    out = {}
    if reference_texts and ids:
        hyp = [pred[i]["rationale"] for i in ids]; ref = [reference_texts[qa[i]["gold_label"]] for i in ids]
        eh, er = enc.encode(hyp, convert_to_tensor=True), enc.encode(ref, convert_to_tensor=True)
        out["semantic_similarity_cosine"] = round(float(util.cos_sim(eh, er).diag().mean()), 4)
        if use_bertscore:
            from bert_score import score
            _, _, F = score(hyp, ref, lang="en", model_type="microsoft/deberta-xlarge-mnli")
            out["bertscore_f1"] = round(float(F.mean()), 4)
    ctx_ids = [i for i in qa if i in pred and pred[i].get("contexts")]
    if ctx_ids:
        sims = []
        for i in ctx_ids:
            q = enc.encode(qa[i]["question"], convert_to_tensor=True)
            c = enc.encode([x["text"] for x in pred[i]["contexts"]], convert_to_tensor=True)
            sims.append(float(util.cos_sim(q, c).mean()))
        out["context_relevance_embedding"] = round(float(np.mean(sims)), 4)
    if use_ragas and ctx_ids:
        # LLM-as-judge metrics (needs an LLM API key configured for ragas)
        from ragas import evaluate as ragas_eval, EvaluationDataset
        from ragas.metrics import Faithfulness, ResponseRelevancy, ContextRelevance
        ds = EvaluationDataset.from_list([dict(user_input=qa[i]["question"],
                                               response=pred[i].get("rationale", pred[i]["pred"]),
                                               retrieved_contexts=[x["text"] for x in pred[i]["contexts"]])
                                          for i in ctx_ids])
        r = ragas_eval(ds, metrics=[Faithfulness(), ResponseRelevancy(), ContextRelevance()])
        out["ragas"] = {k: round(float(np.nanmean(v)), 4) for k, v in r.to_pandas().select_dtypes("number").items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-b", default=None); ap.add_argument("--out", default="report.json")
    ap.add_argument("--reference-texts", default=None); ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--ragas", action="store_true"); ap.add_argument("--skip-text", action="store_true")
    a = ap.parse_args()
    qa, pred = load(a.qa), load(a.pred)
    rep = dict(diagnosis=diagnosis_metrics(qa, pred, load(a.pred_b) if a.pred_b else None),
               retrieval=retrieval_metrics(qa, pred))
    if not a.skip_text:
        refs = json.load(open(a.reference_texts)) if a.reference_texts else None
        rep["text"] = text_metrics(qa, pred, refs, a.bertscore, a.ragas)
    json.dump(rep, open(a.out, "w"), indent=2)
    d = rep["diagnosis"]
    print(f"balanced acc {d['balanced_accuracy']} {d['balanced_accuracy_95ci']} | macro-F1 {d['macro_f1']} | MCC {d['mcc']}")
    if rep["retrieval"]:
        print("retrieval:", {k: v for k, v in rep["retrieval"].items() if k in ("hit_recall@5", "ndcg@5", "mrr", "context_precision")})


if __name__ == "__main__":
    main()
