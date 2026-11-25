"""The knowledge corpus command (python -m src.main corpus), which joins steps 1-4, and the corpus audit (audit)."""
import argparse
import datetime
import hashlib
import json
import os
from collections import Counter

from src.knowledge.chunking import chunk
from src.knowledge.collect import (
    GUIDELINE_TOPIC,
    GUIDELINE_WORDS,
    PMC_QUERIES,
    Fetcher,
    meditron_guideline_docs,
    pmc_docs,
    pubmed_docs,
    textbook_docs,
)
from src.knowledge.extract import is_ai_paper, junk_reason, strip_markup
from src.knowledge.metadata import LIBRARIES, MANIFEST_COLUMNS, corpus_card, tag, topics
from src.knowledge.references import (
    OVERLAP_LIMIT,
    overlap_share,
    reference_shingles,
    shingles,
)
from src.utils.common import CLASSES, KNOWLEDGE_CORPUS, REFERENCE_TEXTS, load_env
from src.utils.io import read_jsonl, write_csv, write_json, write_jsonl
from src.utils.logging import get_logger

log = get_logger("corpus")      # the corpus, reference, index and visual commands share this logger

# ----------------------------------------------------------------------------------------------------------------------
# corpus: the knowledge corpus (python -m src.main corpus)
# ----------------------------------------------------------------------------------------------------------------------
# Builds the knowledge corpus for RAG: real, published, citable text about the 8 diagnoses. Nothing is written by an LLM or by us.
#
# Four libraries, as in RAG^2 (PubMed, PMC, clinical guidelines, textbooks; Sohn et al. 2025, Appendix A.3), limited to skin cancer:
#   pubmed      PubMed abstracts found per diagnosis (NCBI E-utilities)
#   pmc         full open-access papers from PubMed Central (search: E-utilities; licence: the paper's own Creative Commons link, from E-utilities; text: BioC API)
#   guidelines  open-access guideline and consensus papers in PMC, and the public part of the Meditron guideline collection
#               (Hugging Face epfl-llm/guidelines) without WikiDoc (crowdsourced) and NICE (AI use needs NICE's written permission)
#   textbook    Dermoscopedia, the dermoscopy wiki of the International Dermoscopy Society (CC BY-NC-SA 4.0)
#
# Rules (all automatic):
#   - papers about AI or image datasets (deep learning, ISIC, HAM10000 ...) are left out: they describe datasets, not medicine;
#   - PMC papers need an open licence (CC0 or CC BY*) and must not be retracted;
#   - texts are split at sentence ends into passages of at most --max-words words, with one sentence of overlap;
#   - exact duplicates and passages under --min-words words are dropped;
#   - at most --per-class passages per library and diagnosis, so no diagnosis or library dominates the corpus;
#   - a passage is tagged with every diagnosis whose name or synonym appears in it or in its title (weak relevance labels).
#
# Output (in --out, default data/knowledge/corpus/):
#   passages.jsonl       the passages with their source, link and licence (stays on the laptop; see .gitignore)
#   manifest.csv         one row per passage without the text: id, library, document, link, licence, tags, MD5 (committed)
#   corpus_card.json     counts per library and diagnosis, what was removed and why, the topic coverage check, settings, date
#
# No e-mail address is sent. NCBI requests carry the tool name only, and the NCBI API key if the environment variable NCBI_API_KEY
# is set (optional; it allows more requests per second). Downloads are cached in --cache, so a rerun does not download again.
#
# Usage (on the laptop, with internet):
#   python -m src.main corpus --out data/knowledge/corpus
#   python -m src.main corpus --out data/knowledge/corpus --libraries pubmed pmc   # some libraries only


def passages_from(docs, library, cls, args, seen_text, seen_docs, removed):
    """Turns documents into passages, applying the rules, until the library has args.per_class passages for this class."""
    out = []
    for d in docs:
        if len(out) >= args.per_class:
            break
        body = " ".join(d["paragraphs"])
        reason = ("document already used" if d["doc_id"] in seen_docs else "retracted" if d.get("retracted")
                  else "AI or image-dataset paper" if is_ai_paper(d["title"], body) else "no text" if not body else None)
        if reason:
            removed[f"{library}: {reason}"] += 1
            continue
        seen_docs.add(d["doc_id"])
        k = 0
        for para in d["paragraphs"]:
            for piece in chunk(para, args.max_words):
                if len(out) >= args.per_class:
                    break
                piece = strip_markup(piece)
                junk = junk_reason(piece)
                if junk:
                    removed[f"{library}: {junk}"] += 1
                    continue
                if len(piece.split()) < args.min_words:
                    removed[f"{library}: passage too short"] += 1
                    continue
                h = hashlib.md5(piece.lower().encode()).hexdigest()
                if h in seen_text:
                    removed[f"{library}: duplicate passage"] += 1
                    continue
                seen_text.add(h)
                out.append({"passage_id": f"{d['doc_id']}#{k}", "library": library, "doc_id": d["doc_id"], "title": d["title"],
                            "text": piece, "n_words": len(piece.split()), "url": d["url"], "year": d.get("year", ""),
                            "licence": d["licence"], "query_class": cls, "tags": tag(piece, d["title"]),
                            "topics": topics(piece), "md5": h})
                k += 1
    return out


def write_outputs(passages, card, out):
    """passages.jsonl (with the text), manifest.csv (without it) and corpus_card.json, each written in one step."""
    write_jsonl(os.path.join(out, "passages.jsonl"), passages)
    write_csv(os.path.join(out, "manifest.csv"), [{**p, "tags": " ".join(p["tags"])} for p in passages], MANIFEST_COLUMNS)
    write_json(os.path.join(out, "corpus_card.json"), card)


def build_corpus(args, fetch, meditron_rows=None):
    passages, removed, seen_text, seen_docs = [], Counter(), set(), set()
    for library in [lib for lib in LIBRARIES if lib in args.libraries]:
        if library == "guidelines":
            meditron = []
            if getattr(args, "no_meditron", False):
                log.info("Meditron guideline collection skipped (--no-meditron); PMC guideline papers only")
            else:
                cache = os.path.join(fetch.cache, "meditron_skin_cancer.json") if hasattr(fetch, "cache") else None
                try:
                    if meditron_rows is None and not (cache and os.path.isfile(cache)):
                        log.info("Meditron guideline collection: one large download, saved after one complete pass "
                                 "(it can take several minutes; Ctrl-C is safe, nothing is saved until it completes)")
                    meditron = meditron_guideline_docs(removed, meditron_rows, cache)
                except Exception as e:                      # e.g. no network to Hugging Face: the PMC guideline papers still count
                    log.warning(f"Meditron guideline collection not available ({type(e).__name__}); using PMC guideline papers only")
                    removed["guidelines: Meditron collection not available"] += 1
        for cls in CLASSES:
            if library == "pubmed":
                docs = pubmed_docs(fetch, cls, args.pubmed_docs)
            elif library == "pmc":
                docs = pmc_docs(fetch, PMC_QUERIES[cls], args.pmc_docs, removed)
            elif library == "guidelines":
                own = [d for d in meditron if cls in tag(" ".join(d["paragraphs"])[:5000], d["title"])]
                docs = [*pmc_docs(fetch, f"{GUIDELINE_TOPIC[cls]} AND {GUIDELINE_WORDS}", args.guideline_docs, removed), *own]
            else:
                docs = textbook_docs(fetch, cls, args.textbook_pages)
            got = passages_from(docs, library, cls, args, seen_text, seen_docs, removed)
            passages.extend(got)
            log.info(f"{library:10s} {cls:4s} {len(got):5d} passages")
    return passages, removed


def corpus_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main corpus", description="Builds the knowledge corpus for RAG: real, published, citable text about the 8 diagnoses. Nothing is written by an LLM or by us.")
    ap.add_argument("--out", default=os.path.join("data", KNOWLEDGE_CORPUS))
    ap.add_argument("--cache", default=None, help="download cache (default: <out>/cache)")
    ap.add_argument("--libraries", nargs="+", default=list(LIBRARIES), choices=LIBRARIES)
    ap.add_argument("--per-class", type=int, default=150, help="most passages per library and diagnosis")
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--pubmed-docs", type=int, default=300, help="abstracts searched per diagnosis")
    ap.add_argument("--pmc-docs", type=int, default=40, help="open-access papers searched per diagnosis")
    ap.add_argument("--guideline-docs", type=int, default=20, help="open-access guideline papers searched per diagnosis")
    ap.add_argument("--textbook-pages", type=int, default=15, help="Dermoscopedia pages per diagnosis")
    ap.add_argument("--no-meditron", action="store_true", help="skip the large Meditron download (PMC guideline papers only)")
    ap.add_argument("--env-file", default=".env", help="keys: NCBI_API_KEY and HF_TOKEN are read from it if present (both optional)")
    return ap.parse_args(argv)


def corpus_main(argv=None):
    args = corpus_args(argv)
    get_logger("corpus", os.path.join(args.out, "build_corpus.log"))
    unusable = load_env(args.env_file)                       # a variable already set in the terminal keeps its value
    if unusable:
        log.warning(f"{args.env_file}: lines {unusable} are not NAME=value (or have no value) and were skipped")
    for name, what in (("NCBI_API_KEY", "faster NCBI requests"), ("HF_TOKEN", "faster Hugging Face downloads")):
        log.info(f"{name}: " + (f"found ({what})" if os.environ.get(name, "").strip() else "not set (optional; anonymous access)"))
    fetch = Fetcher(args.cache or os.path.join(args.out, "cache"))
    passages, removed = build_corpus(args, fetch)
    card = corpus_card(passages, removed, args, datetime.date.today().isoformat())
    write_outputs(passages, card, args.out)
    log.info(f"{card['n_passages']} passages from {card['n_documents']} documents -> {args.out}")
    if card["empty_topic_boxes"]:
        log.info("topics with no passage yet: " + ", ".join(card["empty_topic_boxes"]))


def audit_corpus(passages, ref_shingles=frozenset(), near_duplicate=0.8):
    """Checks that need the passage text. Returns counts and the ids of what was found; nothing is changed or removed."""
    sh = {p["passage_id"]: shingles(p["text"]) for p in passages}
    seen, dup = {}, []
    for p in passages:                                      # near-duplicates: an inverted index on the 8-word stretches
        a = sh[p["passage_id"]]
        cand = {}
        for g in a:
            for q in seen.get(g, ()):
                cand[q] = cand.get(q, 0) + 1
        for q, n in cand.items():
            if n / len(a | sh[q]) >= near_duplicate:
                dup.append([q, p["passage_id"]])
        for g in a:
            seen.setdefault(g, []).append(p["passage_id"])
    over = [p["passage_id"] for p in passages if overlap_share(p["text"], ref_shingles) >= OVERLAP_LIMIT]
    by_doc = {}
    for p in passages:
        by_doc[p["doc_id"]] = by_doc.get(p["doc_id"], 0) + 1
    top = sorted(by_doc.items(), key=lambda t: (-t[1], t[0]))[:5]
    lib = {}
    for p in passages:
        row = lib.setdefault(p["library"], {"passages": 0, "untagged": 0, "short_under_40_words": 0})
        row["passages"] += 1
        row["untagged"] += not p["tags"]
        row["short_under_40_words"] += p["n_words"] < 40
    retag = [p["passage_id"] for p in passages if sorted(p["tags"]) != tag(p["text"], p.get("title", ""))]
    return {"n_passages": len(passages), "by_library": lib, "near_duplicate_pairs": dup, "reference_copies": over,
            "largest_documents": [{"doc_id": d, "passages": n} for d, n in top], "tags_that_would_change": retag}


def audit_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main audit", description="Checks the corpus text (no internet, nothing is changed).")
    ap.add_argument("--corpus", default=os.path.join("data", KNOWLEDGE_CORPUS))
    ap.add_argument("--reference-texts", default=os.path.join("data", REFERENCE_TEXTS))
    a = ap.parse_args(argv)
    get_logger("audit")
    passages = read_jsonl(os.path.join(a.corpus, "passages.jsonl"))
    result = audit_corpus(passages, reference_shingles(a.reference_texts))
    out = os.path.join(a.corpus, "audit.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    log.info(f"{result['n_passages']} passages: {len(result['near_duplicate_pairs'])} near-duplicate pairs, "
             f"{len(result['reference_copies'])} copies of reference texts, {len(result['tags_that_would_change'])} tags that the "
             f"current rule would change; details in {out}")
    for name, row in sorted(result["by_library"].items()):
        log.info(f"  {name}: {row}")
    return result
