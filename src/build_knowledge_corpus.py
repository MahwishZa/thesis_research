#!/usr/bin/env python3
"""
Builds the knowledge corpus for RAG: real, published, citable text about the 8 diagnoses. Nothing is written by an LLM or by us.

Four libraries, as in RAG^2 (PubMed, PMC, clinical guidelines, textbooks; Sohn et al. 2025, Appendix A.3), limited to skin cancer:
  pubmed      PubMed abstracts found per diagnosis (NCBI E-utilities)
  pmc         full open-access papers from PubMed Central (search: E-utilities; licence: PMC OA service; text: BioC API)
  guidelines  open-access guideline and consensus papers in PMC, and the public part of the Meditron guideline collection
              (Hugging Face epfl-llm/guidelines) without WikiDoc (crowdsourced) and NICE (AI use needs NICE's written permission)
  textbook    Dermoscopedia, the dermoscopy wiki of the International Dermoscopy Society (CC BY-NC-SA 4.0)

Rules (all automatic):
  - papers about AI or image datasets (deep learning, ISIC, HAM10000 ...) are left out: they describe datasets, not medicine;
  - PMC papers need an open licence (CC0 or CC BY*) and must not be retracted;
  - texts are split at sentence ends into passages of at most --max-words words, with one sentence of overlap;
  - exact duplicates and passages under --min-words words are dropped;
  - at most --per-class passages per library and diagnosis, so no diagnosis or library dominates the corpus;
  - a passage is tagged with every diagnosis whose name or synonym appears in it or in its title (weak relevance labels).

Output (in --out, default data/corpus/):
  passages.jsonl       the passages with their source, link and licence (stays on the laptop; see .gitignore)
  manifest.csv         one row per passage without the text: id, library, document, link, licence, tags, MD5 (committed)
  corpus_card.json     counts per library and diagnosis, what was removed and why, the topic coverage check, settings, date

No e-mail address is sent. NCBI requests carry the tool name only, and the NCBI API key if the environment variable NCBI_API_KEY
is set (optional; it allows more requests per second). Downloads are cached in --cache, so a rerun does not download again.

Usage (on the laptop, with internet):
  python src/build_knowledge_corpus.py --out data/corpus
  python src/build_knowledge_corpus.py --out data/corpus --libraries pubmed pmc   # some libraries only
"""
import argparse
import datetime
import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser

from common import CLASSES, CORPUS, get_logger, write_csv, write_json, write_jsonl

log = get_logger("corpus")
LIBRARIES = ("guidelines", "pmc", "pubmed", "textbook")     # guidelines first, so a guideline paper is not counted again as pmc
SYNONYMS = {
    "MEL": ["melanoma", "lentigo maligna", "melanoma in situ"],
    "NV": ["nevus", "naevus", "nevi", "naevi", "melanocytic nevus"],
    "BCC": ["basal cell carcinoma", "basal-cell carcinoma", "basalioma"],
    "AK": ["actinic keratosis", "actinic keratoses", "solar keratosis", "solar keratoses"],
    "BKL": ["seborrheic keratosis", "seborrhoeic keratosis", "seborrheic keratoses", "solar lentigo", "solar lentigines",
            "lichen planus-like keratosis", "benign keratosis"],
    "DF": ["dermatofibroma", "dermatofibromas", "benign fibrous histiocytoma"],
    "VASC": ["hemangioma", "haemangioma", "cherry angioma", "angiokeratoma", "pyogenic granuloma", "vascular lesion"],
    "SCC": ["squamous cell carcinoma", "keratoacanthoma", "bowen's disease", "bowen disease"],
}
PUBMED_QUERIES = {
    "MEL": '"melanoma"[MeSH Terms] AND "skin neoplasms"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "NV": '"nevus, pigmented"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "BCC": '"carcinoma, basal cell"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "AK": '"keratosis, actinic"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "BKL": '("keratosis, seborrheic"[MeSH Terms] OR "solar lentigo"[tiab] OR "lichen planus-like keratosis"[tiab]) '
           'AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "DF": '("histiocytoma, benign fibrous"[MeSH Terms] OR dermatofibroma[tiab]) AND (dermoscopy[tiab] OR dermatoscopy[tiab] '
          'OR diagnosis[sh])',
    "VASC": '("hemangioma"[MeSH Terms] OR angiokeratoma[tiab] OR "pyogenic granuloma"[tiab] OR "cherry angioma"[tiab]) '
            'AND skin[tiab] AND (dermoscopy[tiab] OR dermatoscopy[tiab])',
    "SCC": '"carcinoma, squamous cell"[MeSH Terms] AND "skin neoplasms"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] '
           'OR diagnosis[sh])',
}
PMC_QUERIES = {
    "MEL": '"cutaneous melanoma"[Title/Abstract] AND (dermoscopy OR dermatoscopy OR diagnosis)',
    "NV": '(nevus[Title] OR naevus[Title] OR nevi[Title] OR naevi[Title]) AND (dermoscopy OR dermatoscopy)',
    "BCC": '"basal cell carcinoma"[Title] AND (dermoscopy OR dermatoscopy OR diagnosis)',
    "AK": '("actinic keratosis"[Title] OR "actinic keratoses"[Title]) AND (dermoscopy OR dermatoscopy OR diagnosis)',
    "BKL": '("seborrheic keratosis"[Title] OR "seborrhoeic keratosis"[Title] OR "solar lentigo"[Title]) '
           'AND (dermoscopy OR dermatoscopy OR diagnosis)',
    "DF": 'dermatofibroma[Title] AND (dermoscopy OR dermatoscopy OR diagnosis)',
    "VASC": '(hemangioma[Title] OR angiokeratoma[Title] OR "pyogenic granuloma"[Title] OR "cherry angioma"[Title]) '
            'AND skin AND (dermoscopy OR dermatoscopy)',
    "SCC": '"cutaneous squamous cell carcinoma"[Title/Abstract] AND (dermoscopy OR dermatoscopy OR diagnosis)',
}
GUIDELINE_WORDS = '(guideline[Title] OR guidelines[Title] OR consensus[Title] OR recommendations[Title])'
GUIDELINE_TOPIC = {"MEL": '"melanoma"[Title]', "NV": '(nevus[Title] OR naevus[Title] OR nevi[Title])',
                   "BCC": '"basal cell carcinoma"[Title]', "AK": '("actinic keratosis"[Title] OR "actinic keratoses"[Title])',
                   "BKL": '("seborrheic keratosis"[Title] OR "seborrhoeic keratosis"[Title])', "DF": 'dermatofibroma[Title]',
                   "VASC": '("hemangioma"[Title] OR "haemangioma"[Title] OR "vascular anomalies"[Title])',
                   "SCC": '"squamous cell carcinoma"[Title] AND (skin[Title] OR cutaneous[Title])'}
TEXTBOOK_SEARCH = {"MEL": "melanoma", "NV": "nevus", "BCC": "basal cell carcinoma", "AK": "actinic keratosis",
                   "BKL": "seborrheic keratosis", "DF": "dermatofibroma", "VASC": "hemangioma", "SCC": "squamous cell carcinoma"}
AI_PAPER = re.compile(r"\b(deep learning|machine learning|artificial intelligence|neural networks?|convolutional|"
                      r"computer[- ]aided|ISIC|HAM10000)\b", re.I)
OPEN_LICENCES = re.compile(r"^(CC0|CC BY(-SA|-ND|-NC|-NC-SA|-NC-ND)?)(\s|$)", re.I)
GUIDELINE_EXCLUDED_SOURCES = {"wikidoc", "nice"}
PMC_KEEP_SECTIONS = {"ABSTRACT", "INTRO", "CASE", "RESULTS", "DISCUSS", "CONCL", "FIG"}
PMC_KEEP_TYPES = {"abstract", "paragraph", "fig_caption"}
TEXTBOOK_LICENCE = "CC BY-NC-SA 4.0"
# Topic coverage check (keyword-based, approximate): does the corpus speak about each diagnosis in each of these four ways?
TOPICS = {
    "appearance": r"dermoscop|dermatoscop|pigment network|globule|streak|blue-white|veil|vessel|vascular pattern|"
                  r"regression|milia|lacun|white patch|rosette|ulceration|border|asymmetr|colou?r",
    "patients": r"\bage[ds]?\b|elderly|older|young|incidence|prevalen|\bmen\b|\bwomen\b|male|female|sun|ultraviolet|uv\b|"
                r"risk factor|located|location|trunk|face|head|neck|extremit|\blegs?\b|\barms?\b",
    "differential": r"differential|distinguish|differentiat|mimic|simulat|confused with|versus|\bvs\.?\b|resembl",
    "basics": r"benign|malignan|carcinoma|tumou?r|neoplasm|lesion|prognos|histolog|patholog|definition|defined as",
}
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
DERMOSCOPEDIA_APIS = ("https://dermoscopedia.org/api.php", "https://dermoscopedia.org/w/api.php")
TOOL = "thesis_skin_rag_corpus"


# ---- text handling (no internet) -------------------------------------------------------------------------------------------
def clean(text):
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def sentences(text):
    """Splits at a full stop, question or exclamation mark followed by a space and a capital letter, digit or bracket."""
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", clean(text)) if s]


def chunk(text, max_words=200, overlap=1):
    """Passages of whole sentences, at most max_words words (a longer single sentence is cut), with `overlap` sentences shared."""
    out, cur = [], []
    for s in sentences(text):
        words = s.split()
        while len(words) > max_words:                       # a very long "sentence" (e.g. a list): cut it
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if cur and len(" ".join(cur).split()) + len(words) > max_words:
            out.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            if len(" ".join(cur).split()) + len(words) > max_words:
                cur = []
        cur.append(" ".join(words))
    if cur and (not out or " ".join(cur) not in out[-1]):
        out.append(" ".join(cur))
    return [p for p in out if p]


def tag(text, title=""):
    t = f"{title} {text}".lower()
    return sorted(c for c, words in SYNONYMS.items() if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words))


def is_ai_paper(title, text):
    """True for papers about AI or image datasets, judged from the title and the start of the text (about an abstract)."""
    return bool(AI_PAPER.search(f"{title} {text[:1500]}"))


def open_licence(licence):
    return bool(licence) and bool(OPEN_LICENCES.match(licence.strip()))


def topics(text):
    t = text.lower()
    return sorted(k for k, rx in TOPICS.items() if re.search(rx, t))


class _Paragraphs(HTMLParser):
    """Collects the text of <p> and <li> elements, skipping tables, scripts, styles, references, navigation and edit links."""
    SKIP_TAGS = {"table", "script", "style", "sup", "math", "nav"}
    SKIP_CLASSES = ("mw-editsection", "reference", "navbox", "toc", "printfooter", "catlinks")
    VOID = {"br", "img", "hr", "input", "meta", "link", "wbr", "source", "area", "col"}

    def __init__(self):
        super().__init__()
        self.out, self.buf, self.stack = [], [], []        # stack: (tag, skipped)

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            if tag == "br":
                self.handle_data(" ")
            return
        cls = dict(attrs).get("class") or ""
        skipped = bool(self.stack and self.stack[-1][1]) or tag in self.SKIP_TAGS or any(k in cls for k in self.SKIP_CLASSES)
        self.stack.append((tag, skipped))
        if tag in ("p", "li") and not skipped:
            self.buf = []

    def handle_endtag(self, tag):
        if tag not in (t for t, _ in self.stack):
            return
        while self.stack:
            t, skipped = self.stack.pop()
            if t in ("p", "li") and not skipped:
                text = clean("".join(self.buf))
                self.buf = []
                if text:
                    self.out.append(text)
            if t == tag:
                break

    def handle_data(self, data):
        if self.stack and not self.stack[-1][1] and any(t in ("p", "li") for t, _ in self.stack):
            self.buf.append(data)


def html_paragraphs(page_html):
    parser = _Paragraphs()
    parser.feed(page_html or "")
    return parser.out


# ---- parsing of the downloaded formats (no internet) ------------------------------------------------------------------------
def parse_pubmed_xml(xml_text):
    docs = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title_el = art.find(".//ArticleTitle")
        title = clean("".join(title_el.itertext())) if title_el is not None else ""
        abstract = clean(" ".join("".join(x.itertext()) for x in art.findall(".//AbstractText")))
        year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
        pub_types = [clean(x.text) for x in art.findall(".//PublicationType")]
        docs.append({"doc_id": f"PMID:{pmid}", "title": title, "paragraphs": [abstract] if abstract else [], "year": year,
                     "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "licence": "PubMed abstract (research use; cite PMID)",
                     "pub_types": pub_types, "retracted": "Retracted Publication" in pub_types})
    return docs


def parse_oa_record(xml_text):
    """The PMC OA service answer for one paper: (licence, retracted), or (None, None) when it is not open access."""
    rec = ET.fromstring(xml_text).find(".//record")
    if rec is None:
        return None, None
    return rec.get("license"), rec.get("retracted", "no") == "yes"


def parse_bioc(bioc):
    """Title, year and kept paragraphs of a BioC JSON paper (the service answers with a collection or a list of them)."""
    collection = bioc[0] if isinstance(bioc, list) else bioc
    doc = collection["documents"][0]
    title, year, paragraphs = "", "", []
    for p in doc.get("passages", []):
        inf = p.get("infons", {})
        sec, typ = inf.get("section_type", ""), inf.get("type", "")
        if sec == "TITLE" and not title:
            title, year = clean(p.get("text")), str(inf.get("year", ""))
        elif sec in PMC_KEEP_SECTIONS and typ in PMC_KEEP_TYPES and p.get("text"):
            paragraphs.append(clean(p["text"]))
    return title, year, paragraphs


# ---- downloading (internet; cached) -----------------------------------------------------------------------------------------
class Fetcher:
    def __init__(self, cache_dir, pause=0.4):
        self.cache, self.pause, self.key = cache_dir, pause, os.environ.get("NCBI_API_KEY")
        if self.key:
            self.pause = 0.12
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, url, params=None, ncbi=False):
        params = dict(params or {})
        if ncbi:
            params["tool"] = TOOL
        name = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        path = os.path.join(self.cache, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        if ncbi and self.key:
            params["api_key"] = self.key                   # never written to the cache name or to any output
        full = url + ("?" + urllib.parse.urlencode(params) if params else "")
        for attempt in range(4):
            try:
                req = urllib.request.Request(full, headers={"User-Agent": TOOL})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read().decode("utf-8", "replace")
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        time.sleep(self.pause)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return body


def esearch(fetch, db, term, n):
    r = json.loads(fetch.get(NCBI + "esearch.fcgi", {"db": db, "term": term, "retmax": n, "retmode": "json", "sort": "relevance"},
                             ncbi=True))
    return r["esearchresult"]["idlist"]


def pubmed_docs(fetch, cls, n):
    ids = esearch(fetch, "pubmed", PUBMED_QUERIES[cls] + " AND hasabstract[text] AND english[lang]", n)
    for i in range(0, len(ids), 100):
        yield from parse_pubmed_xml(fetch.get(NCBI + "efetch.fcgi", {"db": "pubmed", "id": ",".join(ids[i:i + 100]),
                                                                      "retmode": "xml"}, ncbi=True))


def pmc_docs(fetch, term, n, removed):
    for uid in esearch(fetch, "pmc", term + ' AND "open access"[filter]', n):
        pmcid = f"PMC{uid}"
        licence, retracted = parse_oa_record(fetch.get(PMC_OA, {"id": pmcid}))
        if licence is None or not open_licence(licence):
            removed["pmc: no open licence"] += 1
            continue
        try:
            title, year, paragraphs = parse_bioc(json.loads(fetch.get(BIOC.format(pmcid))))
        except Exception:                                    # no BioC version of this paper, or an unexpected answer
            removed["pmc: full text not available"] += 1
            continue
        yield {"doc_id": pmcid, "title": title, "paragraphs": paragraphs, "year": year, "licence": licence,
               "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/", "retracted": retracted}


def meditron_guideline_docs(removed, rows=None):
    """The public Meditron guideline collection, skin-cancer articles only, without the excluded sources."""
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset("epfl-llm/guidelines", split="train", streaming=True)
    for r in rows:
        source = (r.get("source") or "").lower()
        text = r.get("clean_text") or r.get("raw_text") or ""
        if source in GUIDELINE_EXCLUDED_SOURCES:
            removed[f"guidelines: source {source} excluded"] += 1
            continue
        if not tag(text[:5000], r.get("title", "")):
            continue                                            # not about the 8 diagnoses
        yield {"doc_id": f"meditron:{r.get('id', '')}", "title": clean(r.get("title")), "year": "",
               "paragraphs": [p for p in re.split(r"\n\s*\n", text) if p.strip()], "url": r.get("url") or "",
               "licence": f"Meditron guidelines collection (source: {source}); see the source's terms", "retracted": False}


def textbook_docs(fetch, cls, n):
    api = None
    for candidate in DERMOSCOPEDIA_APIS:
        try:
            json.loads(fetch.get(candidate, {"action": "query", "meta": "siteinfo", "format": "json"}))
            api = candidate
            break
        except Exception:
            continue
    if api is None:
        raise RuntimeError("Dermoscopedia's API did not answer at " + " or ".join(DERMOSCOPEDIA_APIS))
    hits = json.loads(fetch.get(api, {"action": "query", "list": "search", "srsearch": TEXTBOOK_SEARCH[cls], "srlimit": n,
                                      "format": "json"}))["query"]["search"]
    for h in hits:
        page = json.loads(fetch.get(api, {"action": "parse", "page": h["title"], "prop": "text", "format": "json",
                                          "formatversion": 2}))
        yield {"doc_id": f"dermoscopedia:{h['title']}", "title": h["title"], "year": "", "licence": TEXTBOOK_LICENCE,
               "url": "https://dermoscopedia.org/" + urllib.parse.quote(h["title"].replace(" ", "_")),
               "paragraphs": html_paragraphs(page["parse"]["text"]), "retracted": False}


# ---- from documents to passages (no internet) -------------------------------------------------------------------------------
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


def corpus_card(passages, removed, args, snapshot):
    count = Counter((p["library"], p["query_class"]) for p in passages)
    coverage = {c: {t: sum(c in p["tags"] and t in p["topics"] for p in passages) for t in TOPICS} for c in CLASSES}
    return {"snapshot_date": snapshot, "n_passages": len(passages), "n_documents": len({p["doc_id"] for p in passages}),
            "passages_per_library_and_class": {lib: {c: count[(lib, c)] for c in CLASSES} for lib in LIBRARIES},
            "passages_tagged_per_class": {c: sum(c in p["tags"] for p in passages) for c in CLASSES},
            "topic_coverage_per_class": coverage,
            "empty_topic_boxes": [f"{c}/{t}" for c, row in coverage.items() for t, n in row.items() if n == 0],
            "licences": dict(Counter(p["licence"] for p in passages).most_common()),
            "removed": dict(sorted(removed.items())),
            "settings": {k: v for k, v in vars(args).items() if k not in ("out", "cache")}}


MANIFEST_COLUMNS = ["passage_id", "library", "doc_id", "url", "licence", "query_class", "tags", "n_words", "md5"]


def write_outputs(passages, card, out):
    """passages.jsonl (with the text), manifest.csv (without it) and corpus_card.json, each written in one step."""
    write_jsonl(os.path.join(out, "passages.jsonl"), passages)
    write_csv(os.path.join(out, "manifest.csv"), [{**p, "tags": " ".join(p["tags"])} for p in passages], MANIFEST_COLUMNS)
    write_json(os.path.join(out, "corpus_card.json"), card)


def build(args, fetch, meditron_rows=None):
    passages, removed, seen_text, seen_docs = [], Counter(), set(), set()
    for library in [lib for lib in LIBRARIES if lib in args.libraries]:
        if library == "guidelines":
            try:
                meditron = list(meditron_guideline_docs(removed, meditron_rows))
            except Exception as e:                          # e.g. not logged in to Hugging Face for this gated collection
                log.warning(f"Meditron guideline collection not available ({type(e).__name__}); using PMC guideline papers only")
                removed["guidelines: Meditron collection not available"] += 1
                meditron = []
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join("data", CORPUS))
    ap.add_argument("--cache", default=None, help="download cache (default: <out>/cache)")
    ap.add_argument("--libraries", nargs="+", default=list(LIBRARIES), choices=LIBRARIES)
    ap.add_argument("--per-class", type=int, default=150, help="most passages per library and diagnosis")
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--pubmed-docs", type=int, default=300, help="abstracts searched per diagnosis")
    ap.add_argument("--pmc-docs", type=int, default=40, help="open-access papers searched per diagnosis")
    ap.add_argument("--guideline-docs", type=int, default=20, help="open-access guideline papers searched per diagnosis")
    ap.add_argument("--textbook-pages", type=int, default=15, help="Dermoscopedia pages per diagnosis")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    get_logger("corpus", os.path.join(args.out, "build_corpus.log"))
    fetch = Fetcher(args.cache or os.path.join(args.out, "cache"))
    passages, removed = build(args, fetch)
    card = corpus_card(passages, removed, args, datetime.date.today().isoformat())
    write_outputs(passages, card, args.out)
    log.info(f"{card['n_passages']} passages from {card['n_documents']} documents -> {args.out}")
    if card["empty_topic_boxes"]:
        log.info("topics with no passage yet: " + ", ".join(card["empty_topic_boxes"]))


if __name__ == "__main__":
    main()
