"""The standard RAG system (python -m src.main standard-rag) and the retrieval-only comparison on validation
(retrieval-eval)."""
import argparse
import json
import os
import random

from src.data.views import question_file
from src.evaluation.metrics import holm, mcnemar_exact, percentile, relevant_counts, retrieval_metrics
from src.models.embeddings import MEDCPT_CROSS, MEDCPT_QUERY, medcpt_cross_encoder, medcpt_encoder
from src.models.llm import make_client, require_free_or_allowed
from src.pipelines.base import APPROVED_RETRIEVAL, add_run_arguments, rag_instruction, rag_system, rag_text, run, run_folder
from src.retrieval.indexer import read_index
from src.retrieval.retriever import BM25Index, distinct, image_query_vector, reciprocal_rank_fusion, retrieval_query
from src.utils import IMAGE_LIBRARY, INDEX_DIR, SEED, SPLIT_FILES, TEXT_LIBRARY, VIEWS, get_logger, load_env, read_jsonl, write_json

RETRIEVAL_MODES = ("bm25", "bm25-rerank", "dense", "hybrid")     # all but the approved one: comparisons and trials
APPROVED = APPROVED_RETRIEVAL     # the standard RAG's text search (defined in src/pipelines/base.py with the folder rule)
POOL = 20      # passages each searcher gives to the fusion and the re-ranker (a fixed, common value; not tuned)
FUSION_LIST = 10   # image + text view: the top 10 of the text search and of the image search are fused by rank (as validated; not tuned)


class StandardRag:
    """The retrieval of one run: the index, the query encoder, k, and the retrieval mode.
    bm25: the k passages of highest BM25 keyword score (Robertson & Zaragoza 2009).  bm25-rerank: the POOL best BM25 passages,
    re-ordered by the MedCPT cross-encoder (Jin et al. 2023), the one re-ranking technique of the research.  dense: the k passages of
    highest MedCPT cosine similarity.  hybrid: that list and a BM25 keyword list, merged by reciprocal rank fusion."""

    def __init__(self, index, encode_query, k=3, mode="dense", bm25=None, rerank=None, pool=POOL, visual=None):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if mode.endswith("-rerank") and rerank is None:
            raise ValueError(f"{mode} needs a re-ranker")
        self.index, self.encode_query, self.k, self.mode, self.rerank, self.pool = index, encode_query, k, mode, rerank, pool
        self.bm25 = bm25 if bm25 is not None else (BM25Index(index.passages) if mode != "dense" else None)
        self.visual = visual          # (the image index, {Kaggle image file: its saved vector}) for the image views; see load_visual

    def retrieve(self, query, k):
        """[(passage, score), ...], best first, for one query text."""
        pool = max(self.pool, k)
        if self.mode == "bm25":
            return self.bm25.search(query, k)
        if self.mode == "bm25-rerank":
            candidates = self.bm25.search(query, pool)
            scores = self.rerank(query, [[p.get("title") or "", p["text"]] for p, _ in candidates])
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            return [(candidates[i][0], round(scores[i], 6)) for i in order[:k]]
        vector = self.encode_query([query])[0]
        if self.mode == "dense":
            return self.index.search(vector, k)
        return reciprocal_rank_fusion([[p for p, _ in self.index.search(vector, pool)], [p for p, _ in self.bm25.search(query, pool)]],
                                      top=pool)[:k]

    def ready(self, view):
        """Stops before any question is asked when a view needs what is not loaded."""
        if view != "text_only" and self.visual is None:
            raise ValueError("the image views search by the image: the image index is needed (python -m src.main visual-index)")

    def image_search(self, q, n):
        """The n library images (picture-and-caption records) closest to the question's own image (its saved MONET vector)."""
        self.ready("image_only")
        vindex, vectors = self.visual
        return vindex.search(image_query_vector(q, vectors), n)

    def ranked(self, q, view, n):
        """The n best records for a question, best first, as validated with retrieval-eval (researcher's decisions, 2026-11-24/25):
        text only: the text search of the patient facts; image only: the image search by the question's image; image + text: the
        text search and the image search, fused by rank (reciprocal rank fusion of the top FUSION_LIST of each)."""
        m = max(n, FUSION_LIST)                     # candidates, so that n remain after a repeated text is removed
        if view == "image_only":
            return distinct(self.image_search(q, m))[:n]
        query = retrieval_query(q, view)
        if view == "multimodal":
            return distinct(reciprocal_rank_fusion([[p for p, _ in self.retrieve(query, FUSION_LIST)],
                                                    [p for p, _ in self.image_search(q, FUSION_LIST)]], top=2 * m))[:n]
        return distinct(self.retrieve(query, m))[:n]

    def settings(self, view="text_only"):
        meta = self.index.meta
        s = {"query_encoder": None if self.mode.startswith("bm25") else MEDCPT_QUERY, "article_encoder": meta.get("encoder"),
             "reranker": MEDCPT_CROSS if self.mode == "bm25-rerank" else None, "k": self.k, "query": "question", "retrieval": self.mode,
             "pool": None if self.mode in ("bm25", "dense") else self.pool,
             "index_passages_sha256": meta.get("passages_sha256"), "index_n_passages": len(self.index.passages),
             "exclude_reference_pages": meta.get("exclude_reference_pages")}
        if view != "text_only":
            vmeta = self.visual[0].meta if self.visual else {}
            s["image_search"] = {"encoder": vmeta.get("encoder"), "items_sha256": vmeta.get("passages_sha256"),
                                 "n_items": len(self.visual[0].passages) if self.visual else 0}
            s["query"] = "image" if view == "image_only" else "question and image, fused by rank"
            if view == "multimodal":
                s["fusion"] = {"method": "reciprocal rank fusion", "k": 60, "list": FUSION_LIST}
        return s

    def instruction(self, view):
        return rag_instruction(view, self.k)

    def prepare(self, q, view, client, image):
        """(the text to send, the fields to add to the record, no extra tokens): one retrieval, no extra model call."""
        hits = self.ranked(q, view, self.k)
        contexts = [{"rank": r, "chunk_id": p["passage_id"], "title": p.get("title"), "text": p["text"], "tags": p.get("tags", []),
                     "score": score} | ({"image_file": p["image_file"]} if p.get("image_file") else {})
                    for r, (p, score) in enumerate(hits, 1)]
        extra = {"query": None if view == "image_only" else retrieval_query(q, view), "contexts": contexts}
        if view != "text_only":
            extra["query_image"] = os.path.basename(q["image_path"])
        return rag_text(q, view, contexts), extra, None, None


def rag_parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main standard-rag", description="Standard RAG: one model, one view, one split.")
    add_run_arguments(ap)
    ap.add_argument("--k", type=int, default=3, help="passages given to the model (approved: 3)")
    ap.add_argument("--retrieval", default=APPROVED, choices=RETRIEVAL_MODES,
                    help=f"{APPROVED} (the standard RAG); the other modes are trials in their own folders")
    ap.add_argument("--index", default=None, help=f"the index folder (default: data/{INDEX_DIR})")
    return ap.parse_args(argv)


def make_rag(args):
    """The StandardRag of the command line: loads the index and the encoders (laptop: torch and transformers)."""
    folder = args.index or os.path.join(args.data, INDEX_DIR)
    index = read_index(folder, os.path.join(args.data, TEXT_LIBRARY, "passages.jsonl"))      # stops with a clear message
    rerank = medcpt_cross_encoder() if args.retrieval.endswith("-rerank") else None
    visual = load_visual(args.data) if args.view != "text_only" else None     # the image views search by the image
    encode = None if args.retrieval.startswith("bm25") else medcpt_encoder(MEDCPT_QUERY, 64)      # BM25 needs no query encoder
    return StandardRag(index, encode, args.k, args.retrieval, rerank=rerank, visual=visual)


def standard_rag_main(argv=None, rag=None):
    args = rag_parse_args(argv)
    final, _ = run_folder(args.results, args.model, args.view, args.split, rag_system(args.retrieval))
    name = f"{rag_system(args.retrieval)}.{args.model}.{args.view}.{args.split}"
    log, client = get_logger(name), None
    if os.path.isdir(final) and not args.dry_run:
        log.info(f"already complete: {final}")
        return final
    if not args.dry_run:
        require_free_or_allowed(args.model, args.allow_paid)     # before anything is loaded
    if rag is None:
        rag = make_rag(args)
    if not args.dry_run:
        unusable = load_env(args.env_file)
        if unusable:
            log.warning(f"{args.env_file}: lines {unusable} are not NAME=value (or have no value) and were skipped")
        client = make_client(args.model, log)                       # stops here when the key is missing
    return run(args, client, log, rag)


# ---- retrieval-only comparison (no model call, no cost) ---------------------------------------------------------------------
# For the validation questions of a text view, retrieves 10 passages with each retrieval mode and scores them with the retrieval
# metrics of src/evaluation/metrics.py (Hit@k, Precision@k, Recall@k, nDCG@k, MRR; a passage is relevant when its tag names the question's true
# class: the tags are used for this score only, never to retrieve). The choice of the retrieval mode is made on this validation
# score, before any paid run; the test split is refused. Image-only questions are not scored: their text query is the same for all.
def retrieval_eval_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main retrieval-eval", description="Retrieval metrics of the retrieval modes on validation.")
    ap.add_argument("--view", default="text_only", choices=VIEWS,
                    help="text_only: the text modes; image_only: the image search; multimodal: text, image and both fused")
    ap.add_argument("--split", default="val", choices=("val",), help="validation only")
    ap.add_argument("--modes", nargs="+", default=list(RETRIEVAL_MODES), choices=RETRIEVAL_MODES)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default=None, help="default: results/standard_rag/<view>/retrieval_validation.json")
    return ap.parse_args(argv)


INTERVAL_KEYS = ("hit@3", "precision@3", "ndcg@3", "mrr")


def hit3_by_question(qa, pred):
    """{question id: 1 if a passage of the true class reached the top 3, else 0} (the tags score the search, they never steer it)."""
    return {i: int(any(qa[i]["gold_label"] in c.get("tags", []) for c in pred[i]["contexts"][:3])) for i in qa}


def paired_hit3(qa, preds, approved):
    """Each search against the approved one on the same questions: McNemar's exact test on Hit@3 (did a passage of the true class
    reach the top 3?), with Holm's correction over the comparisons. Empty when the approved search is not among them."""
    if approved not in preds:
        return {}

    def hits(pred):
        by_question = hit3_by_question(qa, pred)
        return [by_question[i] for i in qa]
    base, others = hits(preds[approved]), [m for m in preds if m != approved]
    tests = {m: mcnemar_exact([1] * len(qa), hits(preds[m]), base) for m in others}
    adjusted = holm([t["p_value"] for t in tests.values()])
    return {m: {"against": approved, "a_hit_b_miss": t["a_right_b_wrong"], "a_miss_b_hit": t["a_wrong_b_right"],
                "p_value": t["p_value"], "p_holm": round(p, 6)} for (m, t), p in zip(tests.items(), adjusted, strict=True)}


def retrieval_intervals(qa, pred, n_boot=1000, seed=42):
    """95% percentile intervals of the main retrieval scores. Whole lesions are drawn with replacement (photos of one lesion stay
    together), as in src/evaluation/metrics.py."""
    members = {}
    for i, q in qa.items():
        members.setdefault(q.get("group_id", i), []).append(i)
    groups, rng, values = list(members), random.Random(seed), {k: [] for k in INTERVAL_KEYS}
    for _ in range(n_boot):
        qa_b, pred_b = {}, {}
        for g in rng.choices(groups, k=len(groups)):
            for i in members[g]:
                key = f"{i}#{len(qa_b)}"
                qa_b[key], pred_b[key] = qa[i], pred[i]
        scores = retrieval_metrics(qa_b, pred_b, ks=(3,))
        for k in INTERVAL_KEYS:
            values[k].append(scores[k])
    return {k: [round(percentile(sorted(v), 2.5), 4), round(percentile(sorted(v), 97.5), 4)]
            for k, v in values.items()}


def random_retrieval(qa, passages, rounds=20, seed=SEED):
    """The reference line: the same scores when 10 passages are drawn at random from the whole corpus (mean of `rounds` draws)."""
    rng, scores = random.Random(seed), []
    for _ in range(rounds):
        pred = {i: {"contexts": [{"chunk_id": p["passage_id"], "tags": p.get("tags", [])} for p in rng.sample(passages, min(10, len(passages)))]} for i in qa}
        scores.append(retrieval_metrics(qa, pred, ks=(3,)))
    return {k: round(sum(s[k] for s in scores) / rounds, 4) for k in INTERVAL_KEYS}


def load_visual(data):
    """The image index and the saved image vectors of the Kaggle images ({file name: vector}), both made by the visual-index command.
    The query vector of a question is the saved vector of its own image, so no image model is needed here."""
    folder = os.path.join(data, IMAGE_LIBRARY)
    index = read_index(os.path.join(folder, "index"), os.path.join(folder, "items_final.jsonl"))
    path = os.path.join(folder, "kaggle_vectors.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is not there: run python -m src.main visual-index")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    return index, dict(zip(saved["names"], saved["vectors"], strict=True))


def retrieval_eval_main(argv=None, rag_for=None, visual=None):
    args = retrieval_eval_args(argv)
    log = get_logger("retrieval-eval")
    questions = read_jsonl(os.path.join(args.data, question_file(args.view, args.split)))
    qa = {q["id"]: q for q in questions}
    text_modes = list(args.modes) if args.view != "image_only" else []          # image only has no text query: the image search alone
    if rag_for is None and text_modes:
        index = read_index(args.index or os.path.join(args.data, INDEX_DIR), os.path.join(args.data, TEXT_LIBRARY, "passages.jsonl"))
        encode = medcpt_encoder(MEDCPT_QUERY, 64)
        bm25 = BM25Index(index.passages)
        rerank = medcpt_cross_encoder() if any(m.endswith("-rerank") for m in text_modes) else None

        def rag_for(mode):
            return StandardRag(index, encode, 10, mode, bm25=bm25, rerank=rerank)
    rankers, pools, text_sha, image_sha = {}, {}, None, None
    for mode in text_modes:
        rag = rag_for(mode)
        text_sha = rag.index.meta.get("passages_sha256")
        rankers[mode] = (lambda q, rag=rag: [p for p, _ in rag.ranked(q, "text_only", 10)])     # the text search (same query in every view)
        pools[mode] = rag.index.passages
    if args.view != "text_only":
        vindex, vectors = visual if visual is not None else load_visual(args.data)
        image_sha = vindex.meta.get("passages_sha256")
        rankers["image"] = lambda q: [p for p, _ in distinct(vindex.search(image_query_vector(q, vectors), 10))]
        pools["image"] = vindex.passages
        if args.view == "multimodal":
            for mode in text_modes:                          # the image search fused with each text search, as in the RAG run
                fused = rag_for(mode)
                fused.visual = (vindex, vectors)
                rankers[f"fused:{mode}"] = (lambda q, fused=fused: [p for p, _ in fused.ranked(q, "multimodal", 10)])
                pools[f"fused:{mode}"] = pools[mode] + vindex.passages
    visual_ids = ({p["passage_id"] for p in pools.get("image", [])}                     # the image library: image records, and
                  | {p["passage_id"] for pool in pools.values() for p in pool if p.get("library") == "visual"})   # captions as text
    result, intervals, chance, image_share, preds = {}, {}, {}, {}, {}
    for mode, rank in rankers.items():
        pred = preds[mode] = {q["id"]: {"contexts": [{"chunk_id": p["passage_id"], "tags": p.get("tags", [])} for p in rank(q)]}
                              for q in questions}
        top3 = [c["chunk_id"] for v in pred.values() for c in v["contexts"][:3]]
        image_share[mode] = round(sum(c in visual_ids for c in top3) / len(top3), 4) if top3 else 0.0
        result[mode] = retrieval_metrics(qa, pred, n_relevant=relevant_counts(pools[mode]))
        intervals[mode] = retrieval_intervals(qa, pred)
        chance[mode] = random_retrieval(qa, pools[mode])
        log.info(f"{mode}: " + ", ".join(f"{k} {100 * result[mode][k]:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"
                                          for k, (lo, hi) in intervals[mode].items()))
        log.info(f"  random from the same pool ({len(pools[mode])} items): "
                 + ", ".join(f"{k} {100 * v:.1f}%" for k, v in chance[mode].items()))
    paired = paired_hit3(qa, preds, f"fused:{APPROVED}" if args.view == "multimodal" else APPROVED)
    for mode, r in paired.items():
        log.info(f"  {mode} vs {r['against']}: Hit@3 differs on {r['a_hit_b_miss'] + r['a_miss_b_hit']} questions "
                 f"({r['a_hit_b_miss']} only {mode}, {r['a_miss_b_hit']} only {r['against']}); McNemar p = {r['p_value']}, "
                 f"Holm p = {r['p_holm']}")
    out = args.out or os.path.join(args.results, "standard_rag", args.view, f"retrieval_{SPLIT_FILES[args.split]}.json")
    # which libraries these numbers belong to (status compares them with the current indexes), and each question's Hit@3, so that
    # a later library can be compared with this one by a paired test
    libraries = {"text_index_passages_sha256": text_sha, "image_items_sha256": image_sha}
    write_json(out, {"view": args.view, "split": args.split, "n_questions": len(questions), "approved": APPROVED, "libraries": libraries,
                     "modes": result, "intervals_95": intervals, "random_from_the_same_pool": chance,
                     "share_of_top3_from_the_image_library": image_share, "paired_hit3_vs_approved": paired,
                     "hit3_by_question": {mode: hit3_by_question(qa, pred) for mode, pred in preds.items()}})
    return result
