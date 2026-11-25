"""The vector indexes: the text index (python -m src.main index) and the image index (visual-index), saved as float32
vectors with their passage ids."""
import argparse
import heapq
import json
import operator
import os
from array import array

from src.knowledge.lookalike import (
    dhash_variants,
    drop_near_duplicates,
    kaggle_hashes,
    kaggle_vectors,
    max_similarities,
    nearest_by_hash,
)
from src.knowledge.references import OVERLAP_LIMIT, overlap_share, reference_shingles
from src.models.embeddings import MEDCPT_ARTICLE, image_encoder, medcpt_encoder
from src.utils.common import (
    CORPUS_DIR,
    INDEX_DIR,
    PROCESSED_IMAGES,
    REFERENCE_TEXTS,
)
from src.utils.io import atomic_path, read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger
from src.utils.reproducibility import file_sha256

log = get_logger("corpus")      # the corpus, reference, index and visual commands share this logger


def write_index(folder, passages, vectors, meta):
    """Saves the vectors (float32, one row per passage, in the order of `passages`) and index.json (the passage ids and how the index
    was made). The vectors are derived from licensed text, so the folder stays on the laptop (see .gitignore)."""
    if len(vectors) != len(passages) or not passages:
        raise ValueError("an index needs one vector per passage, and at least one passage")
    dim = len(vectors[0])
    flat = array("f")
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors of an index must have the same length")
        flat.extend(v)
    os.makedirs(folder, exist_ok=True)
    with atomic_path(os.path.join(folder, "embeddings.f32")) as tmp, open(tmp, "wb") as f:
        flat.tofile(f)
    write_json(os.path.join(folder, "index.json"),
               {**meta, "dim": dim, "n_passages": len(passages), "passage_ids": [p["passage_id"] for p in passages]})


class DenseIndex:
    """The passages and their vectors; `search` gives the k best passages for one query vector."""

    def __init__(self, passages, flat, dim, meta=None):
        self.passages, self.flat, self.dim, self.meta = passages, flat, dim, meta or {}
        self.rows = memoryview(flat)

    def search(self, query_vector, k):
        """[(passage, score), ...], best first. A tie is broken by the order of the passages, so a search always gives the same list."""
        if len(query_vector) != self.dim:
            raise ValueError(f"the query vector has {len(query_vector)} numbers, the index {self.dim}")
        scores = (sum(map(operator.mul, query_vector, self.rows[i * self.dim:(i + 1) * self.dim])) for i in range(len(self.passages)))
        best = heapq.nlargest(k, enumerate(scores), key=lambda t: (t[1], -t[0]))
        return [(self.passages[i], round(score, 6)) for i, score in best]


def read_index(folder, passages_path):
    """Loads an index written by write_index with the passages it was made from; stops if the passage file has changed since."""
    meta_path = os.path.join(folder, "index.json")
    if not os.path.isfile(meta_path) or not os.path.isfile(os.path.join(folder, "embeddings.f32")):
        raise FileNotFoundError(f"no index in {folder}: run python -m src.main index")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("passages_sha256") != file_sha256(passages_path):
        raise ValueError(f"{passages_path} has changed since the index was built: run python -m src.main index again")
    by_id = {p["passage_id"]: p for p in read_jsonl(passages_path)}
    flat = array("f")
    with open(os.path.join(folder, "embeddings.f32"), "rb") as f:
        flat.fromfile(f, meta["n_passages"] * meta["dim"])
    return DenseIndex([by_id[i] for i in meta["passage_ids"]], flat, meta["dim"], meta)


# ----------------------------------------------------------------------------------------------------------------------
# index: the dense index of the corpus for the RAG systems (python -m src.main index)
# ----------------------------------------------------------------------------------------------------------------------
# Each passage of data/corpus/passages.jsonl is embedded once, as [title, text], with the MedCPT article encoder (Jin et al.
# 2023), and the unit-length vectors are saved in data/corpus/index/ (laptop only: they are derived from licensed text). The
# corpus is already cut into passages by the corpus command (whole sentences, at most 200 words, one sentence shared with the next
# passage), so nothing is cut again here. Tags and query classes are not embedded and never used for retrieval.
#
# --exclude-reference-pages leaves out the passages of the Dermoscopedia pages that also supply the reference texts of the
# semantic-similarity score (src/evaluation/hallucination.py). Without it, a RAG answer that quotes such a passage resembles the reference text of its
# own disease almost word for word, which inflates that score. The choice is the researcher's; it is recorded in index.json.
ARTICLE_MAX_TOKENS = 512


def index_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main index", description="Embeds the corpus passages for the RAG systems (MedCPT).")
    ap.add_argument("--corpus", default=os.path.join("data", CORPUS_DIR), help="the folder with passages.jsonl")
    ap.add_argument("--out", default=os.path.join("data", INDEX_DIR), help="where the index is written")
    ap.add_argument("--encoder", default=MEDCPT_ARTICLE)
    ap.add_argument("--reference-texts", default=os.path.join("data", REFERENCE_TEXTS))
    ap.add_argument("--exclude-reference-pages", action="store_true",
                    help="leave out the passages of the pages that supply data/annotations/reference_texts.json")
    ap.add_argument("--exclude-reference-overlap", action="store_true",
                    help="also leave out any passage that copies a reference text (half or more of its 8-word stretches)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N passages (a trial of the command)")
    return ap.parse_args(argv)


def reference_urls(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is needed for --exclude-reference-pages")
    with open(path, encoding="utf-8") as f:
        return sorted({r["url"] for r in json.load(f).values()})


def build_index(passages, encode, exclude_urls=(), ref_shingles=None):
    """(kept passages, their vectors, the passages left out): every kept passage is embedded as [title, text]."""
    drop = set(exclude_urls)

    def out(p):
        return p["url"] in drop or (ref_shingles is not None and overlap_share(p["text"], ref_shingles) >= OVERLAP_LIMIT)
    kept = [p for p in passages if not out(p)]
    left_out = [p for p in passages if out(p)]
    vectors = encode([[p.get("title") or "", p["text"]] for p in kept]) if kept else []
    return kept, vectors, left_out


def index_main(argv=None, encode=None):
    a = index_args(argv)
    get_logger("index")
    path = os.path.join(a.corpus, "passages.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is not there: run python -m src.main corpus first (on the laptop)")
    passages = read_jsonl(path)
    if a.limit:
        passages = passages[:a.limit]
    urls = reference_urls(a.reference_texts) if a.exclude_reference_pages else []
    encode = encode or medcpt_encoder(a.encoder, ARTICLE_MAX_TOKENS)
    refs = reference_shingles(a.reference_texts) if a.exclude_reference_overlap else None
    kept, vectors, left_out = build_index(passages, encode, urls, refs)
    write_index(a.out, kept, vectors, {"encoder": a.encoder, "max_tokens": ARTICLE_MAX_TOKENS, "normalised": True,
                                       "passages_file": path.replace("\\", "/"), "passages_sha256": file_sha256(path),
                                       "exclude_reference_pages": bool(a.exclude_reference_pages),
                                       "exclude_reference_overlap": bool(a.exclude_reference_overlap), "excluded_urls": urls,
                                       "n_excluded_passages": len(left_out), "limit": a.limit})
    log.info(f"{len(kept)} passages embedded ({len(left_out)} left out) -> {a.out}")
    return kept


# ----------------------------------------------------------------------------------------------------------------------
# visual-index: the image index and the check against the Kaggle images (python -m src.main visual-index)
# ----------------------------------------------------------------------------------------------------------------------
# Every library image is embedded with an image encoder (laptop: open_clip_torch, Pillow). So that no test image can be "looked up",
# every Kaggle image of the study (all splits, data/processed/images/) is embedded too, and a library image whose largest cosine
# similarity with any Kaggle image is at least --threshold is left out. The threshold is a safety choice made before any result was
# seen (default 0.95: images that are almost the same picture); it is NOT tuned. So that it can be judged, the card records the
# 20 highest similarities (library item, Kaggle case) and the quantiles of all of them. The Kaggle vectors are used for this check
# only; they are never part of the index or of any answer.
def visual_index_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main visual-index", description="Embeds the library images and removes Kaggle look-alikes.")
    ap.add_argument("--corpus", default=os.path.join("data", CORPUS_DIR))
    ap.add_argument("--data", default="data")
    ap.add_argument("--encoder", default="biomedclip", choices=("biomedclip",))
    ap.add_argument("--threshold", type=float, default=0.95, help="a library image this similar to a Kaggle image is left out")
    ap.add_argument("--hash-threshold", type=int, default=4,
                    help="a library image whose 64-bit picture hash (also mirrored or turned) differs from a Kaggle image's in at most "
                         "this many bits is left out")
    ap.add_argument("--library-threshold", type=float, default=0.98,
                    help="of two library images this similar, the later one is left out (the same picture in two papers)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N library images (a trial)")
    ap.add_argument("--kaggle-limit", type=int, default=0,
                    help="compare with only the first N Kaggle images (a trial of the command; the check is then NOT complete)")
    return ap.parse_args(argv)


def visual_index_main(argv=None, encode=None, hash_fn=dhash_variants):
    a = visual_index_args(argv)
    log = get_logger("visual-index", os.path.join(a.corpus, "build_visual.log"))
    folder = os.path.join(a.corpus, "visual")
    items = read_jsonl(os.path.join(folder, "items_with_images.jsonl"))
    if a.limit:
        items = items[:a.limit]
    kaggle = sorted(f for f in os.listdir(os.path.join(a.data, PROCESSED_IMAGES)) if f.lower().endswith(".jpg"))
    if not items or not kaggle:
        raise SystemExit("no library images or no Kaggle images: run visual-images first, and check data/processed/images/")
    encode = encode or image_encoder(a.encoder)
    vectors = encode([os.path.join(folder, "images", it["image_file"]) for it in items])
    if a.kaggle_limit:
        kaggle = kaggle[:a.kaggle_limit]
        log.warning("--kaggle-limit: only part of the Kaggle images is compared; this index must not be used for results")
    reference = kaggle_vectors(encode, kaggle, a.data, folder, a.encoder)
    best = max_similarities(vectors, reference)
    ref_hashes = kaggle_hashes(kaggle, a.data, folder, hash_fn)
    by_hash = [nearest_by_hash(hash_fn(os.path.join(folder, "images", it["image_file"])), ref_hashes) for it in items]
    not_kaggle = [i for i, (sim, _) in enumerate(best) if sim < a.threshold and by_hash[i][0] > a.hash_threshold]
    keep_idx, twins = drop_near_duplicates([vectors[i] for i in not_kaggle], a.library_threshold)
    kept = [(items[not_kaggle[i]], vectors[not_kaggle[i]]) for i in keep_idx]
    dropped = [{"item_id": it["item_id"], "similarity": round(sim, 4), "kaggle_image": kaggle[j], "hash_bits_different": hb[0]}
               for it, (sim, j), hb in zip(items, best, by_hash) if sim >= a.threshold or hb[0] <= a.hash_threshold]
    sims = sorted(sim for sim, _ in best)
    quantiles = {f"q{q}": round(sims[min(len(sims) - 1, int(len(sims) * q / 100))], 4) for q in (50, 90, 99)} | {"max": round(sims[-1], 4)}
    top = [{"item_id": items[i]["item_id"], "similarity": round(best[i][0], 4), "kaggle_image": kaggle[best[i][1]]}
           for i in sorted(range(len(items)), key=lambda i: -best[i][0])[:20]]
    if not kept:
        raise SystemExit("every library image was dropped by the Kaggle check: nothing to index")
    out = os.path.join(folder, f"index_{a.encoder}")
    write_jsonl(os.path.join(folder, "items_final.jsonl"), [it for it, _ in kept])
    write_index(out, [it for it, _ in kept], [v for _, v in kept],
                {"encoder": a.encoder, "passages_file": "items_final.jsonl",
                 "passages_sha256": file_sha256(os.path.join(folder, "items_final.jsonl")), "threshold": a.threshold,
                 "n_dropped_as_kaggle_lookalikes": len(dropped), "n_kaggle_images_compared": len(kaggle)})
    card_path = os.path.join(a.corpus, "visual_card.json")
    card = {}
    if os.path.isfile(card_path):
        with open(card_path, encoding="utf-8") as f:
            card = json.load(f)
    card.update({"image_encoder": a.encoder, "kaggle_check": {"threshold": a.threshold, "n_kaggle_images": len(kaggle),
                                                              "n_library_images": len(items), "n_dropped": len(dropped),
                                                              "hash_threshold_bits": a.hash_threshold,
                                                              "n_dropped_by_embedding": sum(sim >= a.threshold for sim, _ in best),
                                                              "n_dropped_by_hash": sum(hb[0] <= a.hash_threshold for hb in by_hash),
                                                              "closest_hashes_bits": sorted(hb[0] for hb in by_hash)[:10],
                                                              "library_threshold": a.library_threshold,
                                                              "n_dropped_as_library_duplicates": len(twins),
                                                              "similarity_quantiles": quantiles, "highest_20": top},
                 "n_indexed_images": len(kept)})
    write_json(card_path, card)
    log.info(f"{len(kept)} images indexed, {len(dropped)} dropped as Kaggle look-alikes (threshold {a.threshold}) and "
             f"{len(twins)} as duplicates inside the library (threshold {a.library_threshold}); "
             f"similarity quantiles {quantiles}")
    return [it for it, _ in kept]
