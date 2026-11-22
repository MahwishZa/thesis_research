#!/usr/bin/env python3
"""
Builds the reference text of each of the 8 diseases for the semantic-similarity score of src/evaluate.py: a paragraph quoted from
Dermoscopedia (the dermoscopy wiki of the International Dermoscopy Society, CC BY-NC-SA 4.0), never written by us or by an AI.

The fixed rule (applied the same way to every disease; steps 1-3 set on 2026-11-22 from the site's page layout, seen with
--inspect; step 4 and the split into several texts per disease approved the same day, option B):
  1. page      search Dermoscopedia for the disease's search terms, in their order; for the first term with any result, take the
               result whose title equals the term (ignoring case), else that term's first result.
  2. chapter   if that page has no sections and its introduction lists subchapters, take the first page it links to instead (once):
               the chapter pages of melanoma and vascular lesions are only lists of their subchapters.
  3. sections  the first section whose heading mentions dermoscopy ("dermoscop" or "dermatoscop") and the sections after it; if no
               heading does, all sections in their order (their headings are the criteria themselves, e.g. "Large blue-gray ovoid
               nests"); if the page has no sections, its introduction. Sections that are not text about the disease itself are left
               out by their heading: references, literature, see also, external links, further reading, differential diagnosis.
               Each section is read without its subsections (they are sections of their own), so no text is read twice.
  4. texts     every section is split into texts of whole sentences, at most --max-words words each (a longer sentence is cut), in
               order, without tables, references, edit links, the wiki's template lines ("cite!", "participate!", "(message)") or
               leftovers of removed figure numbers ("(Figure )", ". ."), and without a reference list written as plain text at the end of a
               section (cut from its first entry; see BIBLIOGRAPHY). A text under --min-words words is left out and listed under
               "skipped". So a disease has one reference text per criterion section, not only the first 200 words of its page.
The semantic-similarity score (src/evaluate.py) compares an explanation with every reference text of its true class and keeps the
highest cosine (the best-matching reference).
The search term of melanocytic nevus is the site's general page on nevus patterns: the page titled "Nevus" is about nail nevi.

Output: data/knowledge_corpus/reference_texts.json
  {"MEL": {"source": "Dermoscopedia: <page>", "url": ..., "rule": ..., "sections": ..., "licence": ..., "retrieved": ...,
           "texts": [{"section": <heading>, "text": ..., "n_words": ...}, ...], "skipped": [{"section": ..., "n_words": ...}],
           "bibliography_removed": [{"section": ..., "n_words": ...}]}}
Check the texts once before any run (the script prints them): the page and sections must be about the disease named.
--token-check counts each text's tokens with the similarity model's own tokenizer and compares them with the most the model reads
(needs sentence-transformers; the model is downloaded once). Downloads are cached (data/knowledge_corpus/cache), so a rerun gives
the same texts.

Usage (on the laptop, with internet): python src/build_reference_texts.py [--token-check]
       python src/build_reference_texts.py --show-tables   (the text of the tables the rule leaves out; writes nothing)
       python src/build_reference_texts.py --show-skipped  (the other text it leaves out: captions, divs, image alt text)
Then:  python src/evaluate.py ... --reference-texts data/knowledge_corpus/reference_texts.json
"""
import argparse
import datetime
import html
import json
import os
import re
from html.parser import HTMLParser

from build_knowledge_corpus import TEXTBOOK_LICENCE, Fetcher, chunk, dermoscopedia_api, html_paragraphs, page_url, sentences
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
NOT_ABOUT_THE_DISEASE = re.compile(r"reference|literature|see also|external link|further reading|differential", re.I)
HEADING_TAG = re.compile(r"<h[1-6]\b", re.I)
# Leftovers of figure numbers the wiki fills in on the page itself: "(Figure )", "(Fig. )", and the ". ." they leave.
EMPTY_FIGURE = re.compile(r"\s*\(\s*(?:see\s+)?fig(?:ure)?s?\.?\s*\)", re.I)
DOUBLE_STOP = re.compile(r"\.(?:\s+\.)+")
# The start of a reference list written as plain text at the end of a section (seen in AK and SCC on 2026-11-22): a sentence that
# begins with an author ("Kittler, H., Riedl, E. ...", "Kreusch, J. & Koch, F.") or names a book edition ("..., Second Edition.").
# A citation inside a sentence, such as "(Kittler et al., 2008)", does not match.
BIBLIOGRAPHY = re.compile(r"^[A-Z][\w'’-]+,\s(?:[A-Z]\.\s?)+[,&]|\b(?:\d+(?:st|nd|rd|th)|First|Second|Third|Fourth|Fifth|Sixth)"
                          r"\s+[Ee]dition\b")


def clean_heading(line):
    """A section heading as plain text: no HTML tags, entities decoded, footnote marks like [1] removed."""
    text = html.unescape(re.sub(r"<[^>]+>", "", line or ""))
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def text_paragraphs(page_html):
    """The paragraphs of a page section, without the wiki's template lines and without empty figure references."""
    out = [DOUBLE_STOP.sub(".", EMPTY_FIGURE.sub("", p)).strip() for p in html_paragraphs(page_html) if not TEMPLATE.search(p)]
    return [p for p in out if p]


def own_html(section_html):
    """The part of a section before its first subsection: the wiki returns a section with its own heading first, followed by its
    subsections, which are read as sections of their own. (The introduction, section 0, has no heading.)"""
    starts = [m.start() for m in HEADING_TAG.finditer(section_html or "")]
    return section_html[:starts[1]] if len(starts) > 1 else section_html


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
    about = [x for x in sections if not NOT_ABOUT_THE_DISEASE.search(clean_heading(x.get("line")))]
    for k, s in enumerate(about):
        if DERMOSCOPY_HEADING.search(clean_heading(s.get("line"))):
            return [x["index"] for x in about[k:]], f"from '{clean_heading(s['line'])}'"
    if about:
        return [x["index"] for x in about], f"all sections from '{clean_heading(about[0]['line'])}' (no dermoscopy heading)"
    return [0], "(introduction)"


def without_bibliography(paragraphs):
    """The section's sentences up to the first one that starts a reference list (BIBLIOGRAPHY), and the number of words cut."""
    parts = sentences(" ".join(paragraphs))
    cut = next((k for k, x in enumerate(parts) if BIBLIOGRAPHY.search(x)), len(parts))
    return [" ".join(parts[:cut])] if cut else [], sum(len(x.split()) for x in parts[cut:])


def split_texts(paragraphs, max_words, min_words):
    """Step 4: (texts of whole sentences of at most max_words words, word counts of the texts under min_words that are left out)."""
    texts = chunk(" ".join(paragraphs), max_words=max_words, overlap=0)
    return [t for t in texts if len(t.split()) >= min_words], [len(t.split()) for t in texts if len(t.split()) < min_words]


def parse(fetch, api, title, **params):
    return json.loads(fetch.get(api, {"action": "parse", "page": title, "format": "json", "formatversion": 2, **params}))["parse"]


def choose(fetch, api, cls):
    """Steps 1-3 for one disease: (page title, page rule, the page's sections, indexes of the sections to read, section rule)."""
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
    return title, page_rule, sections, indexes, section_rule


def reference_for(fetch, api, cls, max_words, min_words, today):
    title, page_rule, sections, indexes, section_rule = choose(fetch, api, cls)
    headings = {x["index"]: clean_heading(x["line"]) for x in sections}
    texts, skipped, bibliography = [], [], []
    for index in indexes:
        html_ = parse(fetch, api, title, prop="text", section=index)["text"]
        paragraphs, cut = without_bibliography(text_paragraphs(own_html(html_)))
        kept, short = split_texts(paragraphs, max_words, min_words)
        heading = headings.get(index, "(introduction)")
        texts += [{"section": heading, "text": t, "n_words": len(t.split())} for t in kept]
        skipped += [{"section": heading, "n_words": n} for n in short]
        bibliography += [{"section": heading, "n_words": cut}] if cut else []
    if not texts:
        raise SystemExit(f"{cls}: '{title}' ({section_rule}) has no text of {min_words} words or more")
    return {"source": f"Dermoscopedia: {title}", "url": page_url(title), "rule": f"page: {page_rule}", "sections": section_rule,
            "licence": TEXTBOOK_LICENCE, "retrieved": today, "texts": texts, "skipped": skipped,
            "bibliography_removed": bibliography}


class _Tables(HTMLParser):
    """The text of each top-level <table> of a page section, which the reference texts leave out (one string per table)."""

    def __init__(self):
        super().__init__()
        self.out, self.depth, self.buf = [], 0, []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.buf = []
        elif self.depth and tag in ("td", "th", "li", "p", "br", "tr"):
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "table" and self.depth:
            self.depth -= 1
            if not self.depth:
                text = re.sub(r"\s+", " ", html.unescape("".join(self.buf))).strip()
                if text:
                    self.out.append(text)

    def handle_data(self, data):
        if self.depth:
            self.buf.append(data)


def table_texts(section_html):
    parser = _Tables()
    parser.feed(own_html(section_html or ""))
    return parser.out


class _Skipped(HTMLParser):
    """The text the reference texts leave out, other than tables: text outside paragraphs and list items (for example image
    captions, definition lists, plain divs) and the alt text of images. One (where, text) per block, where = the element."""
    READ = {"p", "li", "table", "script", "style", "sup", "h1", "h2", "h3", "h4", "h5", "h6"}
    BLOCK = {"div", "dl", "dd", "dt", "blockquote", "figure", "figcaption", "center", "section", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self.out, self.stack, self.buf, self.where = [], [], [], ""

    def ignored(self):
        return any(t in self.READ or "editsection" in c for t, c in self.stack)

    def flush(self):
        text = re.sub(r"\s+", " ", html.unescape("".join(self.buf))).strip()
        if text:
            self.out.append((self.where, text))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            if a.get("alt", "").strip() and not self.ignored():
                self.out.append(("img alt", a["alt"].strip()))
            return
        if tag in ("br", "hr", "wbr", "input", "meta", "link", "source", "area", "col"):
            if tag == "br" and self.buf:
                self.buf.append(" ")
            return
        if tag in self.BLOCK:
            self.flush()
        self.stack.append((tag, a.get("class") or ""))

    def handle_endtag(self, tag):
        if tag not in (t for t, _ in self.stack):
            return
        while self.stack and self.stack.pop()[0] != tag:
            pass
        if tag in self.BLOCK:
            self.flush()

    def handle_data(self, data):
        if not self.ignored() and data.strip():
            if not self.buf:
                t, c = self.stack[-1] if self.stack else ("(no element)", "")
                self.where = f"{t}.{c.split()[0]}" if c else t
            self.buf.append(data)

    def close(self):
        super().close()
        self.flush()


def skipped_texts(section_html):
    parser = _Skipped()
    parser.feed(own_html(section_html or ""))
    parser.close()
    return parser.out


def show_skipped(fetch, api, log):
    """Prints, for each disease, the text outside paragraphs, list items and tables in the sections the rule reads, and the alt
    text of images (decision of 2026-11-22, option A: to find melanoma's lists of criteria). It writes nothing."""
    for cls in CLASSES:
        title, _, sections, indexes, section_rule = choose(fetch, api, cls)
        headings = {x["index"]: clean_heading(x["line"]) for x in sections}
        found = [(headings.get(i, "(introduction)"), where, t) for i in indexes
                 for where, t in skipped_texts(parse(fetch, api, title, prop="text", section=i)["text"])]
        log.info(f"\n===== {cls}: {title} / {section_rule}: {len(found)} pieces of text left out")
        for heading, where, text in found:
            log.info(f"  [{heading}, {where}, {len(text.split())} words] {text[:400]}")
    return None


def show_tables(fetch, api, log):
    """Prints, for each disease, the text of the tables in the sections the rule reads (decision of 2026-11-22: to see whether
    criteria, e.g. melanoma's lists, are in tables). It writes nothing and changes no rule."""
    for cls in CLASSES:
        title, _, sections, indexes, section_rule = choose(fetch, api, cls)
        headings = {x["index"]: clean_heading(x["line"]) for x in sections}
        found = [(headings.get(i, "(introduction)"), t) for i in indexes
                 for t in table_texts(parse(fetch, api, title, prop="text", section=i)["text"])]
        log.info(f"\n===== {cls}: {title} / {section_rule}: {len(found)} tables")
        for heading, text in found:
            log.info(f"  [{heading}, {len(text.split())} words] {text[:600]}")
    return None


def token_check(refs, log, model=None):
    """Counts every text's tokens with the similarity model's tokenizer; the model reads at most max_seq_length tokens and ignores
    the rest. Returns (the largest count, the model's limit)."""
    from sentence_transformers import SentenceTransformer

    from evaluate import SIMILARITY_MODEL
    enc = SentenceTransformer(model or SIMILARITY_MODEL)
    counts = {(c, k): len(enc.tokenizer(t["text"])["input_ids"]) for c, r in refs.items() for k, t in enumerate(r["texts"])}
    limit = enc.max_seq_length
    for (c, k), n in counts.items():
        log.info(f"  {c} text {k + 1}: {n} tokens" + ("  <- longer than the model reads" if n > limit else ""))
    log.info(f"token check: the longest text has {max(counts.values())} tokens; the model reads at most {limit}; "
             f"{sum(n > limit for n in counts.values())} texts are longer")
    if max(counts.values()) > limit:
        log.warning("token check: some texts are longer than the model reads; rebuild with a lower --max-words")
    return max(counts.values()), limit


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
    ap.add_argument("--max-words", type=int, default=200, help="the longest reference text, in words")
    ap.add_argument("--min-words", type=int, default=20, help="shorter texts are left out (listed under 'skipped')")
    ap.add_argument("--token-check", action="store_true", help="count each text's tokens with the similarity model's tokenizer")
    ap.add_argument("--inspect", action="store_true", help="print what the rule sees for each disease; write nothing")
    ap.add_argument("--show-tables", action="store_true", help="print the text of the tables the rule leaves out; write nothing")
    ap.add_argument("--show-skipped", action="store_true", help="print the other text the rule leaves out; write nothing")
    a = ap.parse_args(argv)
    log = get_logger("reference_texts")
    fetch = Fetcher(a.cache)
    api = dermoscopedia_api(fetch)
    if a.show_skipped:
        return show_skipped(fetch, api, log)
    if a.show_tables:
        return show_tables(fetch, api, log)
    if a.inspect:
        return inspect(fetch, api, log)
    today = datetime.date.today().isoformat()
    refs = {cls: reference_for(fetch, api, cls, a.max_words, a.min_words, today) for cls in CLASSES}
    write_json(a.out, refs)
    for cls, r in refs.items():
        log.info(f"\n{cls}: {r['source']} / {r['sections']} ({r['rule']}): {len(r['texts'])} texts"
                 + (f"; skipped as too short: {[x['section'] for x in r['skipped']]}" if r["skipped"] else "")
                 + (f"; reference list removed from: {[x['section'] for x in r['bibliography_removed']]}"
                    if r["bibliography_removed"] else ""))
        for t in r["texts"]:
            log.info(f"  [{t['section']}, {t['n_words']} words] {t['text'][:150]}...")
    log.info(f"\n-> {a.out}. Check that each page and section is about the disease named before any run.")
    if a.token_check:
        token_check(refs, log)
    return refs


if __name__ == "__main__":
    main()
