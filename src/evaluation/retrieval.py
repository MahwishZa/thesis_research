"""Retrieval metrics (RAG systems): Hit@k, Precision@k, Recall@k, nDCG@k, MRR and the MedCPT relevance scores."""
import math
from collections import Counter, defaultdict

from src.evaluation.statistics import mean
from src.utils.common import CLASSES

MEDCPT = ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", "ncbi/MedCPT-Cross-Encoder")


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
    mrr = []
    for g, ctx in rows:
        rel = [1 if g in c.get("tags", []) else 0 for c in ctx]
        mrr.append(next((1 / (j + 1) for j, r in enumerate(rel) if r), 0.0))
    res["mrr"] = round(mean(mrr), 4)
    res["n_questions_with_contexts"] = len(rows)
    return res


def medcpt_scorers(models=MEDCPT):
    """Two scorers built on MedCPT: the cosine of the query and passage embeddings, and the cross-encoder score (higher = more
    relevant; not a probability). Each takes (query, passages) and returns one score per passage."""
    import torch
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
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
