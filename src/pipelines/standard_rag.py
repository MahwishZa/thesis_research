"""The standard RAG system (python -m src.main standard-rag) and the retrieval-only comparison on validation
(retrieval-eval)."""
import argparse
import json
import os
import random

from src.data.loaders import question_file
from src.evaluation.retrieval import relevant_counts, retrieval_metrics
from src.evaluation.statistics import percentile
from src.knowledge.references import OVERLAP_LIMIT, overlap_share, shingles
from src.models.embeddings import MEDCPT_QUERY, medcpt_encoder
from src.models.llm import MODELS, make_client
from src.pipelines.base import rag_system, run, run_folder
from src.pipelines.prompts import rag_instruction, rag_text
from src.retrieval.indexer import read_index
from src.retrieval.query import image_query_vector, retrieval_query
from src.retrieval.reranker import medcpt_cross_encoder
from src.retrieval.retriever import BM25Index, reciprocal_rank_fusion
from src.utils.common import (
    CORPUS_DIR,
    INDEX_DIR,
    SPLIT_FILES,
    SPLITS,
    VIEWS,
    load_env,
)
from src.utils.io import read_jsonl, write_json
from src.utils.logging import get_logger
from src.utils.reproducibility import SEED

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid-rerank")
POOL = 20      # passages each searcher gives to the fusion and the re-ranker (a fixed, common value; not tuned)
FUSION_LIST = 10   # image + text view: the top 10 of the text search and of the image search are fused by rank (as validated; not tuned)


def distinct(hits):
    """The hits without a record whose text repeats a better-ranked one (the same text, or half or more of its 8-word stretches):
    a figure caption can be both a corpus passage and a picture-and-caption record, and the model must not get it twice."""
    out, seen, seen_sh = [], set(), set()
    for p, score in hits:
        text = " ".join(p["text"].lower().split())
        if text in seen or overlap_share(p["text"], seen_sh) >= OVERLAP_LIMIT:
            continue
        out.append((p, score))
        seen.add(text)
        seen_sh |= shingles(p["text"])
    return out


class StandardRag:
    """The retrieval of one run: the index, the query encoder, k, and the retrieval mode.
    dense: the k passages of highest MedCPT cosine similarity.  hybrid: that list and a BM25 keyword list, merged by reciprocal rank
    fusion.  hybrid-rerank: the hybrid pool, re-ordered by the MedCPT cross-encoder."""

    def __init__(self, index, encode_query, k=3, mode="dense", bm25=None, rerank=None, pool=POOL, visual=None):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if mode == "hybrid-rerank" and rerank is None:
            raise ValueError("hybrid-rerank needs a re-ranker")
        self.index, self.encode_query, self.k, self.mode, self.rerank, self.pool = index, encode_query, k, mode, rerank, pool
        self.bm25 = bm25 if bm25 is not None else (BM25Index(index.passages) if mode != "dense" else None)
        self.visual = visual          # (the image index, {Kaggle image file: its saved vector}) for the image views; see load_visual

    def retrieve(self, query, k):
        """[(passage, score), ...], best first, for one query text."""
        vector = self.encode_query([query])[0]
        if self.mode == "dense":
            return self.index.search(vector, k)
        pool = max(self.pool, k)
        fused = reciprocal_rank_fusion([[p for p, _ in self.index.search(vector, pool)], [p for p, _ in self.bm25.search(query, pool)]],
                                       top=pool)
        if self.mode == "hybrid":
            return fused[:k]
        scores = self.rerank(query, [[p.get("title") or "", p["text"]] for p, _ in fused])
        order = sorted(range(len(fused)), key=lambda i: (-scores[i], i))
        return [(fused[i][0], round(scores[i], 6)) for i in order[:k]]

    def ready(self, view):
        """Stops before any question is asked when a view needs what is not loaded."""
        if view != "text_only" and self.visual is None:
            raise ValueError("the image views search by the image: the image index is needed (python -m src.main visual-index)")

    def image_search(self, q, n):
        """The n library images (picture-and-caption records) closest to the question's own image (its saved BiomedCLIP vector)."""
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
        s = {"query_encoder": MEDCPT_QUERY, "article_encoder": meta.get("encoder"), "k": self.k, "query": "question", "retrieval": self.mode, "pool": None if self.mode == "dense" else self.pool,
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
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--view", required=True, choices=VIEWS)
    ap.add_argument("--split", default="val", choices=SPLITS, help="val to try things out; test only for the final runs")
    ap.add_argument("--k", type=int, default=3, help="passages given to the model (approved: 3)")
    ap.add_argument("--retrieval", default="hybrid-rerank", choices=RETRIEVAL_MODES,
                    help="hybrid-rerank (approved 2026-11-24: dense + BM25, rank fusion, MedCPT cross-encoder); dense or hybrid: trials")
    ap.add_argument("--limit", type=int, default=0, help="ask at most this many questions now (a trial)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--dry-run", action="store_true", help="show the prompts of the first questions; no API call, no key needed")
    ap.add_argument("--env-file", default=".env", help="the file with the API keys")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--index", default=None, help=f"the index folder (default: data/{INDEX_DIR})")
    ap.add_argument("--image-encoder", default="biomedclip", help="the image index of the image views (made by visual-index)")
    return ap.parse_args(argv)


def make_rag(args):
    """The StandardRag of the command line: loads the index and the encoders (laptop: torch and transformers)."""
    folder = args.index or os.path.join(args.data, INDEX_DIR)
    index = read_index(folder, os.path.join(args.data, CORPUS_DIR, "passages.jsonl"))      # stops with a clear message
    rerank = medcpt_cross_encoder() if args.retrieval == "hybrid-rerank" else None
    visual = load_visual(args.data, args.image_encoder) if args.view != "text_only" else None     # the image views search by the image
    return StandardRag(index, medcpt_encoder(MEDCPT_QUERY, 64), args.k, args.retrieval, rerank=rerank, visual=visual)


def standard_rag_main(argv=None, rag=None):
    args = rag_parse_args(argv)
    final, partial = run_folder(args.results, args.model, args.view, args.split, rag_system(args.retrieval))
    name = f"{rag_system(args.retrieval)}.{args.model}.{args.view}.{args.split}"
    log, client = get_logger(name), None
    if os.path.isdir(final) and not args.dry_run:
        log.info(f"already complete: {final}")
        return final
    if rag is None:
        rag = make_rag(args)
    if not args.dry_run:
        unusable = load_env(args.env_file)
        if unusable:
            log.warning(f"{args.env_file}: lines {unusable} are not NAME=value (or have no value) and were skipped")
        client = make_client(args.model, log)                       # stops here when the key is missing
        log = get_logger(name, os.path.join(partial, "run.log"))
    return run(args, client, log, rag)


# ---- retrieval-only comparison (no model call, no cost) ---------------------------------------------------------------------
# For the validation questions of a text view, retrieves 10 passages with each retrieval mode and scores them with the retrieval
# metrics of src/evaluation/retrieval.py (Hit@k, Precision@k, Recall@k, nDCG@k, MRR; a passage is relevant when its tag names the question's true
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
    ap.add_argument("--image-encoder", default="biomedclip")
    ap.add_argument("--out", default=None, help="default: results/standard_rag/<view>/retrieval_validation.json")
    return ap.parse_args(argv)


INTERVAL_KEYS = ("hit@3", "precision@3", "ndcg@3", "mrr")


def retrieval_intervals(qa, pred, n_boot=1000, seed=42):
    """95% percentile intervals of the main retrieval scores. Whole lesions are drawn with replacement (photos of one lesion stay
    together), as in src/evaluation/statistics.py."""
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


def load_visual(data, encoder):
    """The image index and the saved image vectors of the Kaggle images ({file name: vector}), both made by the visual-index command.
    The query vector of a question is the saved vector of its own image, so no image model is needed here."""
    folder = os.path.join(data, CORPUS_DIR, "visual")
    index = read_index(os.path.join(folder, f"index_{encoder}"), os.path.join(folder, "items_final.jsonl"))
    path = os.path.join(folder, f"kaggle_vectors_{encoder}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is not there: run python -m src.main visual-index")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    return index, dict(zip(saved["names"], saved["vectors"]))


def retrieval_eval_main(argv=None, rag_for=None, visual=None):
    args = retrieval_eval_args(argv)
    log = get_logger("retrieval-eval")
    questions = read_jsonl(os.path.join(args.data, question_file(args.view, args.split)))
    qa = {q["id"]: q for q in questions}
    text_modes = list(args.modes) if args.view == "text_only" else (["hybrid-rerank"] if args.view == "multimodal" else [])
    if rag_for is None and text_modes:
        index = read_index(args.index or os.path.join(args.data, INDEX_DIR), os.path.join(args.data, CORPUS_DIR, "passages.jsonl"))
        encode = medcpt_encoder(MEDCPT_QUERY, 64)
        bm25 = BM25Index(index.passages)
        rerank = medcpt_cross_encoder() if "hybrid-rerank" in text_modes else None

        def rag_for(mode):
            return StandardRag(index, encode, 10, mode, bm25=bm25, rerank=rerank)
    rankers, pools = {}, {}
    for mode in text_modes:
        rag = rag_for(mode)
        rankers[mode] = (lambda q, rag=rag: [p for p, _ in rag.ranked(q, "text_only", 10)])     # the text search (same query in every view)
        pools[mode] = rag.index.passages
    if args.view != "text_only":
        vindex, vectors = visual if visual is not None else load_visual(args.data, args.image_encoder)
        rankers["image"] = lambda q: [p for p, _ in distinct(vindex.search(image_query_vector(q, vectors), 10))]
        pools["image"] = vindex.passages
        if args.view == "multimodal":
            fused = rag_for("hybrid-rerank")
            fused.visual = (vindex, vectors)
            rankers["fused"] = lambda q: [p for p, _ in fused.ranked(q, "multimodal", 10)]      # the same code as the RAG run
            pools["fused"] = pools["hybrid-rerank"] + vindex.passages
    visual_ids = ({p["passage_id"] for p in pools.get("image", [])}                     # the image library: image records, and
                  | {p["passage_id"] for pool in pools.values() for p in pool if p.get("library") == "visual"})   # captions as text
    result, intervals, chance, image_share = {}, {}, {}, {}
    for mode, rank in rankers.items():
        pred = {q["id"]: {"contexts": [{"chunk_id": p["passage_id"], "tags": p.get("tags", [])} for p in rank(q)]} for q in questions}
        top3 = [c["chunk_id"] for v in pred.values() for c in v["contexts"][:3]]
        image_share[mode] = round(sum(c in visual_ids for c in top3) / len(top3), 4) if top3 else 0.0
        result[mode] = retrieval_metrics(qa, pred, n_relevant=relevant_counts(pools[mode]))
        intervals[mode] = retrieval_intervals(qa, pred)
        chance[mode] = random_retrieval(qa, pools[mode])
        log.info(f"{mode}: " + ", ".join(f"{k} {100 * result[mode][k]:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"
                                          for k, (lo, hi) in intervals[mode].items()))
        log.info(f"  random from the same pool ({len(pools[mode])} items): "
                 + ", ".join(f"{k} {100 * v:.1f}%" for k, v in chance[mode].items()))
    out = args.out or os.path.join(args.results, "standard_rag", args.view, f"retrieval_{SPLIT_FILES[args.split]}.json")
    write_json(out, {"view": args.view, "split": args.split, "n_questions": len(questions), "modes": result,
                     "intervals_95": intervals, "random_from_the_same_pool": chance,
                     "share_of_top3_from_the_image_library": image_share})
    return result
