"""Retrieval metrics (RAG systems): Hit@k, Precision@k, Recall@k, nDCG@k, MRR and the MedCPT relevance scores."""
import math
from collections import Counter, defaultdict

from src.evaluation.metrics import mean
from src.utils.common import CLASSES


def relevant_counts(passages):
    """How many corpus passages carry each class tag (the weak labels): the denominator of Recall@k."""
    counts = Counter(t for p in passages for t in set(p.get("tags") or []))
    return {c: counts[c] for c in CLASSES}


CUTOFFS = (1, 3, 5, 10)


def retrieval_metrics(qa, pred, ks=None, n_relevant=None):
    """Hit@k, Precision@k, nDCG@k (and Recall@k with the corpus counts) and MRR. Without ks, k is each of 1, 3, 5 and 10 that is not
    larger than the number of passages the system kept for an answer (researcher's decision, 2026-11-25): a system that keeps 3
    passages has no Precision@5 (it would be divided by 5 with 3 passages) and no Hit@5 (it would repeat Hit@3)."""
    rows = [(qa[i]["gold_label"], pred[i]["contexts"]) for i in qa if i in pred and pred[i].get("contexts")]
    if not rows:
        return None
    if ks is None:
        kept = max(len(ctx) for _, ctx in rows)
        ks = tuple(k for k in CUTOFFS if k <= kept)
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


def medcpt_scorers():
    """Two scorers built on MedCPT (the same models as the RAG retrieval, src/models.py): the cosine of the query and passage
    embeddings, and the cross-encoder score (higher = more relevant; not a probability). Each takes (query, passages) and returns one
    score per passage."""
    from src.models import (
        MEDCPT_ARTICLE,
        MEDCPT_QUERY,
        medcpt_cross_encoder,
        medcpt_encoder,
    )
    query, article, cross_encoder = medcpt_encoder(MEDCPT_QUERY, 64), medcpt_encoder(MEDCPT_ARTICLE, 512), medcpt_cross_encoder()

    def articles(passages):
        return [[c.get("title") or "", c["text"]] for c in passages]

    def cosine(q, passages):
        qv = query([q])[0]
        return [sum(a * b for a, b in zip(qv, v)) for v in article(articles(passages))]      # unit vectors: the dot product

    def cross(q, passages):
        return cross_encoder(q, articles(passages))

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
