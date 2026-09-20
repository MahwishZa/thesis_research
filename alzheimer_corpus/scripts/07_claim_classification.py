#!/usr/bin/env python3
"""Stage 07 - automatic chunk-level tagging (multi-label).

Two inspectable stages, never collapsed into one opaque score:

  Stage 1  keyword / rule matching  -> claim_class, confidence, method, matched_terms
  Stage 2  similarity to class prototypes -> claim_class, similarity_score, method

Classification is NOT a deletion filter. It drives retrieval ranking,
down-weighting and manual review. Down-weighting rather than deleting is
deliberate: tagging precision is unverified, so a mis-tagged passage must
remain retrievable.

Reads config/claim_taxonomy.yaml's current schema:

  claim_types      -> the topical dimension this stage tags as claim_class.
                       Every current entry has no explicit keyword list, so
                       Stage 1 always falls back to tokens drawn from the
                       category's label and its subtypes - the same
                       fallback the original script already used when a
                       class's keywords were unpopulated, generalised to
                       the schema this taxonomy actually has today.
  evidence_levels   -> a second, independent dimension (what kind of source
                       this passage is drawn from), tagged the same way and
                       reported separately as claim_evidence_level*.
  claim_status,
  temporal_status   -> INTENTIONALLY NOT tagged here. Both require
                       comparing a claim against other evidence (is it
                       contradicted? has it been superseded?), which a
                       single passage's keywords cannot determine.
                       Temporal status is the thesis's actual experimental
                       treatment (systems/proposed/temporal.py scores it from
                       publication date at admission time); pre-baking a
                       "current vs superseded" label onto the corpus here
                       would duplicate that treatment with a much weaker
                       method and risk disagreeing with it. See
                       docs/research_experimental_specification.md §4.
  disease_relevance -> NOT re-tagged here. Stage 04's assess_ad_relevance()
                       already makes this decision, with a full rule trace,
                       for every chunk's source document. This stage
                       propagates that existing ad_relevant/
                       ad_relevance_score rather than recomputing a coarser
                       version of the same judgement, which could disagree
                       with Stage 04's and would then be an inconsistency
                       rather than a second opinion.
  claim_record      -> describes a future, more granular unit (individual
                       claims within a chunk, with claim_id/evidence_span/
                       etc.) that this stage does not produce. It tags
                       *chunks* (~256 tokens), not extracted claims -
                       extracting individual claims is a separate task this
                       stage does not attempt.

Also emits the stratified sample for the 300-passage human validation: a
simple random sample would under-represent small claim classes, which is
exactly where tagging is weakest.

Input : data/chunks/chunks.jsonl   Output: in-place claim_classes + reports
"""
from __future__ import annotations
import argparse, random, re, sys, time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DATA, load_config, get_logger, read_jsonl,
                     write_jsonl, write_report, fold)

SRC = DATA / "chunks" / "chunks.jsonl"

#: Words too common in this domain's prose to serve as a keyword on their
#: own (they would match almost every chunk). Filtered out of the
#: label/subtype-derived fallback vocabulary, not out of curated keyword
#: lists (none exist yet - see module docstring).
_STOPWORDS = {
    "and", "the", "for", "with", "from", "that", "this", "its", "are",
    "was", "were", "has", "have", "not", "disease", "alzheimer",
}


def _tokens_from_name(name: str) -> list[str]:
    """Split a label or snake_case subtype into matchable word tokens."""
    words = re.split(r"[\s_/-]+", name.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS]


def load_dimension(cfg_section: dict, *, keyed_by_group: bool) -> list[dict]:
    """Build a flat list of {id, keywords, has_curated_keywords} entries
    from one taxonomy dimension.

    Two shapes are supported, because claim_taxonomy.yaml uses both:

      claim_types:        {id: {label, description, subtypes: [...]}}
      evidence_levels:     {group: [id, id, ...]}

    ``keyed_by_group`` selects which. Neither shape carries an explicit
    keyword list in the current schema, so every entry's keywords are
    derived from what the taxonomy does specify - the id, its label
    (claim_types only) and its subtypes (claim_types only) - never
    invented vocabulary the taxonomy does not contain.
    """
    out = []
    if keyed_by_group:
        for group, ids in cfg_section.items():
            for item_id in ids:
                keywords = _tokens_from_name(item_id)
                out.append({"id": item_id, "group": group,
                           "keywords": keywords, "has_curated_keywords": False})
    else:
        for item_id, spec in cfg_section.items():
            keywords = set(_tokens_from_name(item_id))
            keywords.update(_tokens_from_name(spec.get("label", "")))
            for subtype in spec.get("subtypes") or []:
                keywords.update(_tokens_from_name(subtype))
            out.append({"id": item_id, "group": None,
                       "keywords": sorted(keywords),
                       "has_curated_keywords": False})
    return out


@lru_cache(maxsize=None)
def _term_pattern(term: str) -> re.Pattern:
    """Compile one keyword's match pattern once, not once per chunk.

    Only used for a PHRASE keyword (contains whitespace/underscore/hyphen) -
    single-word keywords use the much faster word-set path in
    keyword_match() below. A keyword's folded form and escaped pattern never
    change between calls - only the chunk text does - so rebuilding the
    pattern string and recompiling it on every chunk was pure waste even
    for the phrase path. ``term`` is already folded by the caller.
    """
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


@lru_cache(maxsize=None)
def _is_phrase(term: str) -> bool:
    return bool(re.search(r"[\s_-]", term))


def _word_set(text: str) -> frozenset[str]:
    """Every whole word in ``text``, folded, as a set.

    Every keyword _tokens_from_name() currently produces is already a
    single word (it splits category ids/labels/subtypes on the same
    separator characters), so "does this keyword appear as a standalone
    word" reduces to set membership - one O(text length) tokenization
    per chunk instead of one O(text length) regex scan per keyword per
    class (up to ~43 classes). This is what made Stage 07 slow at the
    real corpus's 4.3M-chunk, ~256-MedCPT-token-per-chunk scale even
    after precompiling patterns: precompilation avoided rebuilding a
    pattern, not the O(text length) cost of re-scanning the same text
    once per keyword. A future CURATED (has_curated_keywords=True)
    keyword list could contain a multi-word phrase, which this set can't
    match - keyword_match() below falls back to the regex path for any
    keyword that isn't a single word, so that case still works correctly,
    just without this speedup.
    """
    return frozenset(re.findall(r"\w+", fold(text)))


def keyword_match(text: str, classes: list[dict]) -> list[dict]:
    """Stage 1. Every current class uses the label/subtype-derived fallback
    (module docstring), reported at reduced confidence so the gap between
    'a curated keyword list' and 'tokens borrowed from the taxonomy's own
    naming' is visible in the output, not hidden by it."""
    words = _word_set(text)
    f = None  # only folded if a phrase keyword actually needs it
    hits = []
    for c in classes:
        terms = c["keywords"]
        if not terms:
            continue
        matched = []
        for t in terms:
            ft = fold(t)
            if _is_phrase(ft):
                if f is None:
                    f = fold(text)
                if _term_pattern(ft).search(f):
                    matched.append(t)
            elif ft in words:
                matched.append(t)
        if matched:
            confidence = min(1.0, len(matched) / max(1, len(terms)))
            hits.append({"claim_class": c["id"], "confidence": round(
                            confidence * (1.0 if c["has_curated_keywords"] else 0.4), 3),
                         "method": "keyword" if c["has_curated_keywords"] else "keyword-from-taxonomy",
                         "matched_terms": matched})
    return hits


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over (source, claim class) cells so small classes appear.

    ``records`` only needs to carry the fields a cell key and the final
    annotation_report.csv row need (see ``_annotation_fields`` in main()) -
    not a full chunk with its text - so the caller can pass a lightweight
    projection instead of holding every chunk's full text in memory at once
    just to sample from it.
    """
    rng = random.Random(seed)
    cells = defaultdict(list)
    for c in records:
        key = (c.get("source_tier", "") or "unknown",
               (c.get("claim_classes") or ["<untagged>"])[0])
        cells[key].append(c)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells)
    out = []
    while len(out) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out


#: Reported the same way Stage 06 reports chunking progress: frequent
#: enough to be useful on a multi-million-chunk corpus, infrequent enough
#: not to flood the terminal.
PROGRESS_EVERY = 100_000

#: Fields annotation_report.csv actually reads from a sampled chunk (see
#: the write_report call below) - the lightweight projection tag_chunks()
#: keeps for stratified_sample() instead of every chunk's full text.
_ANNOTATION_FIELDS = (
    "chunk_id", "document_id", "source_tier", "claim_classes",
    "claim_confidence", "claim_evidence_levels", "ad_relevant",
    "ad_relevance_score",
)


def tag_chunk(ch: dict, claim_types: list[dict], evidence_levels: list[dict],
              dist: Counter, evidence_dist: Counter) -> dict:
    """Tag one chunk in place with both dimensions' labels, updating the
    running distribution counters. Pulled out of main() so the streaming
    loop and any future caller share one tagging path."""
    hits = keyword_match(ch.get("text", ""), claim_types)
    ch["claim_classes"] = [h["claim_class"] for h in hits]
    ch["claim_confidence"] = max([h["confidence"] for h in hits], default=0.0)
    ch["claim_method"] = hits[0]["method"] if hits else "none"
    for h in hits:
        dist[h["claim_class"]] += 1
    if not hits:
        dist["<untagged>"] += 1

    ev_hits = keyword_match(ch.get("text", ""), evidence_levels)
    ch["claim_evidence_levels"] = [h["claim_class"] for h in ev_hits]
    ch["claim_evidence_confidence"] = max([h["confidence"] for h in ev_hits], default=0.0)
    for h in ev_hits:
        evidence_dist[h["claim_class"]] += 1
    if not ev_hits:
        evidence_dist["<untagged>"] += 1
    return ch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(SRC))
    ap.add_argument("--sample", type=int, default=300, help="validation sample size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    log = get_logger("07_claim_classification", "quality_control.log")

    cfg = load_config("claim_taxonomy.yaml")
    claim_types = load_dimension(cfg["claim_types"], keyed_by_group=False)
    evidence_levels = load_dimension(cfg["evidence_levels"], keyed_by_group=True)
    log.info("taxonomy | claim_types=%d classes | evidence_levels=%d classes across %d groups | "
             "all classes use taxonomy-derived keywords (no curated keyword lists exist yet)",
             len(claim_types), len(evidence_levels), len(cfg["evidence_levels"]))

    src = Path(args.input)
    if not src.exists():
        log.error("no input at %s - run stage 06 first", src)
        return 2

    # Streams src -> a temp file, then atomically replaces src, instead of
    # `chunks = list(read_jsonl(src))` + write_jsonl(src, chunks) at the
    # end: on a multi-million-chunk corpus (the real corpus produces 4.3M+
    # chunks) holding every chunk's full text in memory at once, then
    # writing nothing until the very end, is the same memory/visibility
    # problem Stage 06 had before its batching rewrite - just relocated
    # one stage later. This keeps peak memory to O(1) chunks in flight
    # plus one lightweight (no text) record per chunk for the stratified
    # sample, and reports progress instead of running silently.
    tmp = src.with_suffix(src.suffix + ".part")
    dist: Counter = Counter()
    evidence_dist: Counter = Counter()
    sample_pool: list[dict] = []
    start = time.monotonic()

    def tag_and_track():
        n = 0
        for ch in read_jsonl(src):
            ch = tag_chunk(ch, claim_types, evidence_levels, dist, evidence_dist)
            sample_pool.append({k: ch.get(k) for k in _ANNOTATION_FIELDS})
            n += 1
            if n % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - start
                log.info(
                    "stage 07 progress | chunks=%d | %.1f chunks/sec | elapsed=%.0fs",
                    n, n / elapsed if elapsed > 0 else 0, elapsed,
                )
            yield ch

    n_chunks = write_jsonl(tmp, tag_and_track())
    tmp.replace(src)

    write_report("claim_class_distribution.csv",
                 [{"claim_class": k, "chunks": v} for k, v in dist.most_common()],
                 ["claim_class", "chunks"])
    write_report("evidence_level_distribution.csv",
                 [{"evidence_level": k, "chunks": v} for k, v in evidence_dist.most_common()],
                 ["evidence_level", "chunks"])
    sample = stratified_sample(sample_pool, args.sample, args.seed)
    write_report("annotation_report.csv",
                 [{"chunk_id": c["chunk_id"], "document_id": c["document_id"],
                   "source_tier": c.get("source_tier", ""),
                   "automatic_labels": ";".join(c.get("claim_classes") or []),
                   "automatic_confidence": c.get("claim_confidence", 0.0),
                   "automatic_evidence_levels": ";".join(c.get("claim_evidence_levels") or []),
                   # Propagated from Stage 04's assess_ad_relevance(), not
                   # recomputed - see module docstring's disease_relevance note.
                   "ad_relevant": c.get("ad_relevant", ""),
                   "ad_relevance_score": c.get("ad_relevance_score", ""),
                   "human_labels": "", "annotator_notes": ""} for c in sample],
                 ["chunk_id", "document_id", "source_tier", "automatic_labels",
                  "automatic_confidence", "automatic_evidence_levels",
                  "ad_relevant", "ad_relevance_score",
                  "human_labels", "annotator_notes"])
    elapsed = time.monotonic() - start
    log.info("stage 07 | chunks=%d | tagged claim_types=%d | tagged evidence_levels=%d | "
             "validation sample=%d (stratified, seed=%d) | elapsed=%.0fs",
             n_chunks, len([k for k in dist if k != "<untagged>"]),
             len([k for k in evidence_dist if k != "<untagged>"]),
             len(sample), args.seed, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
