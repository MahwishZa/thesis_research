#!/usr/bin/env python3
"""
Builds the reference text of each of the 8 diseases for the semantic-similarity score of src/evaluate.py: a paragraph quoted from
Dermoscopedia (the dermoscopy wiki of the International Dermoscopy Society, CC BY-NC-SA 4.0), never written by us or by an AI.

The fixed rule (applied the same way to every disease; set on 2026-11-22 from the site's page layout, seen with --inspect):
  1. page      search Dermoscopedia for the disease's search terms, in their order; for the first term with any result, take the
               result whose title equals the term (ignoring case), else that term's first result.
  2. chapter   if that page has no sections and its introduction lists subchapters, take the first page it links to instead (once):
               the chapter pages of melanoma and vascular lesions are only lists of their subchapters.
  3. sections  the first section whose heading mentions dermoscopy ("dermoscop" or "dermatoscop") and the sections after it; if no
               heading does, the sections in their order from the first (their headings are the criteria themselves, e.g. "Large
               blue-gray ovoid nests"), skipping the introduction; if the page has no sections, its introduction.
  4. text      whole sentences from the start, up to --max-words words (a first sentence longer than that is cut), without
               tables, references, edit links or the wiki's template lines ("cite!", "participate!", "(message)").
The search term of melanocytic nevus is the site's general page on nevus patterns: the page titled "Nevus" is about nail nevi.

Output: data/knowledge_corpus/reference_texts.json
  {"MEL": {"text": ..., "source": "Dermoscopedia: <page>", "url": ..., "section": ..., "rule": ..., "licence": ..., "retrieved": ...}}
Check the 8 texts once before any run (the script prints them): the page and section must be about the disease named.
Downloads are cached (data/knowledge_corpus/cache), so a rerun gives the same texts.

Usage (on the laptop, with internet): python src/build_reference_texts.py
Then:  python src/evaluate.py ... --reference-texts data/knowledge_corpus/reference_texts.json
"""
import argparse
import datetime
import html
import json
import os
import re

from build_knowledge_corpus import TEXTBOOK_LICENCE, Fetcher, dermoscopedia_api, html_paragraphs, page_url, sentences
from common import CLASSES, KNOWLEDGE_CORPUS, get_logger, write_json

SEARCH_TERMS = {
    "MEL": ["melanoma"],
    "NV": ["classification of nevi / benign nevus pattern", "nevus"],
    "BCC": ["basal cell carcinoma"],
    "AK": ["actinic keratosis"],
    "BKL": ["seborrheic keratosis", "solar lentigo", "lichen planus-like keratosis"],
    "DF": ["dermatofibroma"],
    "VASC": ["vascular lesions", "hemangioma", "angioma"],
    "SCC": ["squamous cell carcinoma"],
}
DERMOSCOPY_HEADING = re.compile(r"dermoscop|dermatoscop", re.I)
# The wiki's page templates ("<page> – cite!", "<page> (message)", "<page> – participate!") are not text about the disease.
TEMPLATE = re.compile(r"cite!|participate!|\(message\)", re.I)
CHAPTER = re.compile(r"following subchapters", re.I)


def clean_heading(line):
    """A section heading as plain text: no HTML tags, entities decoded, footnote marks like [1] removed."""
    text = html.unescape(re.sub(r"<[^>]+>", "", line or ""))
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def text_paragraphs(page_html):
    """The paragraphs of a page section, without the wiki's template lines."""
    return [p for p in html_paragraphs(page_html) if not TEMPLATE.search(p)]


def choose_page(search_results):
    """search_results: [(term, [titles in rank order]), ...] in the order of the terms. The rule of step 1."""
    for term, titles in search_results:
        if titles:
            for title in titles:
                if title.strip().lower() == term.lower():
                    return title, f"title equals '{term}'"
            return titles[0], f"first result for '{term}'"
    return None, "no page found"


def choose_sections(sections):
    """Step 3. sections: the page's sections from the wiki API ({"index", "line"}). Returns (indexes to read in order, label)."""
    for k, s in enumerate(sections):
        if DERMOSCOPY_HEADING.search(clean_heading(s.get("line"))):
            return [x["index"] for x in sections[k:]], f"from '{clean_heading(s['line'])}'"
    if sections:
        return [x["index"] for x in sections], f"all sections from '{clean_heading(sections[0]['line'])}' (no dermoscopy heading)"
    return [0], "(introduction)"


def first_words(paragraphs, max_words):
    """Step 3: whole sentences from the start of the section, up to max_words words; a first sentence longer than that is cut
    at max_words words."""
    out, n = [], 0
    for sentence in sentences(" ".join(paragraphs)):
        words = sentence.split()
        if n + len(words) > max_words:
            if not out:
                out.append(" ".join(words[:max_words]))
            break
        out.append(sentence)
        n += len(words)
    return " ".join(out)


def parse(fetch, api, title, **params):
    return json.loads(fetch.get(api, {"action": "parse", "page": title, "format": "json", "formatversion": 2, **params}))["parse"]


def reference_for(fetch, api, cls, max_words, today):
    results = []
    for term in SEARCH_TERMS[cls]:
        hits = json.loads(fetch.get(api, {"action": "query", "list": "search", "srsearch": term, "srlimit": 10,
                                          "format": "json"}))["query"]["search"]
        results.append((term, [h["title"] for h in hits]))
    title, page_rule = choose_page(results)
    if title is None:
        raise SystemExit(f"{cls}: Dermoscopedia gave no page for {SEARCH_TERMS[cls]}")
    sections = parse(fetch, api, title, prop="sections")["sections"]
    if not sections:
        intro = parse(fetch, api, title, prop="text|links", section=0)
        links = [x["title"] for x in intro.get("links", []) if x.get("ns") == 0]
        if links and CHAPTER.search(" ".join(text_paragraphs(intro["text"]))):
            page_rule += f"; chapter page '{title}': its first subchapter"
            title = links[0]
            sections = parse(fetch, api, title, prop="sections")["sections"]
    indexes, section_rule = choose_sections(sections)
    paragraphs = []
    for index in indexes:
        paragraphs += text_paragraphs(parse(fetch, api, title, prop="text", section=index)["text"])
        if sum(len(p.split()) for p in paragraphs) >= max_words:
            break
    text = first_words(paragraphs, max_words)
    if not text:
        raise SystemExit(f"{cls}: '{title}' ({section_rule}) has no text")
    return {"text": text, "source": f"Dermoscopedia: {title}", "url": page_url(title), "section": section_rule,
            "rule": f"page: {page_rule}", "licence": TEXTBOOK_LICENCE, "retrieved": today, "n_words": len(text.split())}


def inspect(fetch, api, log):
    """Prints, without choosing anything, what the rule sees for each disease: the search results of every term, and for each
    page whose title equals a term (or the first result) its section headings, the start of its introduction and the pages it
    links to there. Used to set the rule once; it writes nothing."""
    for cls in CLASSES:
        log.info(f"\n===== {cls}")
        pages = []
        for term in SEARCH_TERMS[cls]:
            hits = json.loads(fetch.get(api, {"action": "query", "list": "search", "srsearch": term, "srlimit": 10,
                                              "format": "json"}))["query"]["search"]
            titles = [h["title"] for h in hits]
            log.info(f"search '{term}': {titles}")
            pages += [t for t in titles if t.lower() == term.lower()] or titles[:1]
        for title in dict.fromkeys(pages):
            info = json.loads(fetch.get(api, {"action": "parse", "page": title, "prop": "sections|links", "section": 0,
                                              "format": "json", "formatversion": 2}))["parse"]
            sections = json.loads(fetch.get(api, {"action": "parse", "page": title, "prop": "sections", "format": "json",
                                                  "formatversion": 2}))["parse"]["sections"]
            intro = json.loads(fetch.get(api, {"action": "parse", "page": title, "prop": "text", "section": 0, "format": "json",
                                               "formatversion": 2}))["parse"]["text"]
            links = [x["title"] for x in info.get("links", []) if x.get("ns") == 0][:20]
            log.info(f"  page '{title}': sections {[clean_heading(x['line']) for x in sections]}")
            log.info(f"    introduction: {' '.join(text_paragraphs(intro))[:250]!r}")
            log.info(f"    links in the introduction: {links}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reference texts of the 8 diseases, quoted from Dermoscopedia by a fixed rule.")
    ap.add_argument("--out", default=os.path.join("data", KNOWLEDGE_CORPUS, "reference_texts.json"))
    ap.add_argument("--cache", default=os.path.join("data", KNOWLEDGE_CORPUS, "cache"))
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--inspect", action="store_true", help="print what the rule sees for each disease; write nothing")
    a = ap.parse_args(argv)
    log = get_logger("reference_texts")
    fetch = Fetcher(a.cache)
    api = dermoscopedia_api(fetch)
    if a.inspect:
        return inspect(fetch, api, log)
    today = datetime.date.today().isoformat()
    refs = {cls: reference_for(fetch, api, cls, a.max_words, today) for cls in CLASSES}
    write_json(a.out, refs)
    for cls, r in refs.items():
        log.info(f"\n{cls}: {r['source']} / {r['section']} ({r['rule']}; {r['n_words']} words)\n  {r['text'][:300]}...")
    log.info(f"\n-> {a.out}. Check that each page and section is about the disease named before any run.")
    return refs


if __name__ == "__main__":
    main()
