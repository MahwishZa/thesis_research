#!/usr/bin/env python3
"""
Builds what the experiments read from the cleaned cases and from public sources. Three commands (each has --help):

  python src/build.py views        the three views and their question files, from data/processed_cases/all_cases.csv (step 2)
  python src/build.py corpus       the knowledge corpus for RAG: four libraries of real, published text (internet; laptop)
  python src/build.py references   the reference texts of the 8 diseases for the semantic-similarity score (internet; laptop)

The rules of each command are written above its code. Only the Python standard library is used.
"""
import argparse
import datetime
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser

from common import (ALL_CASES, CLASSES, KNOWLEDGE_CORPUS, LETTERS, OPTION_TEXT, SEED, SPLITS, VIEWS, get_logger, question_file,
                    read_csv, view_table, write_csv, write_json, write_jsonl)

log = get_logger("corpus")


# ----------------------------------------------------------------------------------------------------------------------
# views: step 2 of the data pipeline (python src/build.py views)
# ----------------------------------------------------------------------------------------------------------------------
# Step 2 of the data pipeline: turns the cleaned cases (data/processed_cases/all_cases.csv) into the three views of the same cases.
#
#   data/views/image_only/   the image only
#   data/views/text_only/    the patient's recorded facts only (age, sex, body site in one fixed sentence)
#   data/views/multimodal/   the image and the same sentence
#
# Each view folder holds
#   cases.csv                one row per case: id, split, fold, lesion group, label, and only the inputs of that view
#   train.jsonl, val.jsonl, test.jsonl
#                            one 8-option multiple-choice question per case (MedQA style) for the LLM and RAG systems
#
# The questions use one fixed template and real data only; nothing is generated. The 8 options are shuffled for each case with a
# fixed seed (the same order for a case in all three views), so a letter never stands for one class. The question never names a
# class, and a text-only question never mentions an image.
#
# Usage: python src/build.py views --data data

OPTION_SEED = SEED
CORE = ["isic_id", "split", "cv_fold", "group_id", "label", "label_idx", "label_name"]
VIEW_INPUTS = {"image_only": ["image_path"],
               "text_only": ["clinical_text", "age", "sex", "site"],
               "multimodal": ["image_path", "clinical_text", "age", "sex", "site"]}
Q_IMAGE = "Look at the dermoscopic image of the skin lesion. What is the most likely diagnosis?"
Q_TEXT = "{vignette} What is the most likely diagnosis?"
Q_MM = "{vignette} Look at the dermoscopic image of the lesion. What is the most likely diagnosis?"


def options_for(case_id, seed=OPTION_SEED):
    """The options of one case: the eight classes in an order fixed by the seed and the case id (the same on every computer).
    Returns {letter: option text} and {class: letter}."""
    order = random.Random(f"{seed}:{case_id}").sample(CLASSES, len(CLASSES))
    return {LETTERS[i]: OPTION_TEXT[c] for i, c in enumerate(order)}, {c: LETTERS[i] for i, c in enumerate(order)}


def question(case, view):
    """The question record of one case in one view."""
    options, letter = options_for(case["isic_id"])
    q = {"id": case["isic_id"], "split": case["split"], "options": options, "answer": letter[case["label"]],
         "answer_text": OPTION_TEXT[case["label"]], "gold_label": case["label"], "group_id": case["group_id"]}
    if view == "image_only":
        return {**q, "question": Q_IMAGE, "image_path": case["image_path"]}
    if view == "text_only":
        return {**q, "question": Q_TEXT.format(vignette=case["clinical_text"])}
    return {**q, "question": Q_MM.format(vignette=case["clinical_text"]), "image_path": case["image_path"]}


def build_views(data):
    cases = read_csv(os.path.join(data, ALL_CASES))
    unknown = sorted({c["label"] for c in cases} - set(CLASSES))
    if unknown:
        raise ValueError(f"{ALL_CASES} holds classes that are not among the 8: {unknown}")
    counts = {}
    for view in VIEWS:
        write_csv(os.path.join(data, view_table(view)), cases, CORE + VIEW_INPUTS[view])
        for split in SPLITS:
            items = [question(c, view) for c in cases if c["split"] == split]
            write_jsonl(os.path.join(data, question_file(view, split)), items)
            counts[(view, split)] = len(items)
    return counts


def views_main(argv=None):
    ap = argparse.ArgumentParser(prog="build.py views", description="Builds the three views and their question files from the cleaned cases.")
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    log = get_logger("build_views")
    for (view, split), n in build_views(a.data).items():
        log.info(f"{view:11s} {split:5s} {n:5d} questions")


# ----------------------------------------------------------------------------------------------------------------------
# corpus: the knowledge corpus (python src/build.py corpus)
# ----------------------------------------------------------------------------------------------------------------------
# Builds the knowledge corpus for RAG: real, published, citable text about the 8 diagnoses. Nothing is written by an LLM or by us.
#
# Four libraries, as in RAG^2 (PubMed, PMC, clinical guidelines, textbooks; Sohn et al. 2025, Appendix A.3), limited to skin cancer:
#   pubmed      PubMed abstracts found per diagnosis (NCBI E-utilities)
#   pmc         full open-access papers from PubMed Central (search: E-utilities; licence: PMC OA service; text: BioC API)
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
# Output (in --out, default data/knowledge_corpus/):
#   passages.jsonl       the passages with their source, link and licence (stays on the laptop; see .gitignore)
#   manifest.csv         one row per passage without the text: id, library, document, link, licence, tags, MD5 (committed)
#   corpus_card.json     counts per library and diagnosis, what was removed and why, the topic coverage check, settings, date
#
# No e-mail address is sent. NCBI requests carry the tool name only, and the NCBI API key if the environment variable NCBI_API_KEY
# is set (optional; it allows more requests per second). Downloads are cached in --cache, so a rerun does not download again.
#
# Usage (on the laptop, with internet):
#   python src/build.py corpus --out data/knowledge_corpus
#   python src/build.py corpus --out data/knowledge_corpus --libraries pubmed pmc   # some libraries only

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


def dermoscopedia_api(fetch):
    """The address of Dermoscopedia's wiki API (the first of the known addresses that answers)."""
    for candidate in DERMOSCOPEDIA_APIS:
        try:
            json.loads(fetch.get(candidate, {"action": "query", "meta": "siteinfo", "format": "json"}))
            return candidate
        except Exception:
            continue
    raise RuntimeError("Dermoscopedia's API did not answer at " + " or ".join(DERMOSCOPEDIA_APIS))


def page_url(title):
    return "https://dermoscopedia.org/" + urllib.parse.quote(title.replace(" ", "_"))


def textbook_docs(fetch, cls, n):
    api = dermoscopedia_api(fetch)
    hits = json.loads(fetch.get(api, {"action": "query", "list": "search", "srsearch": TEXTBOOK_SEARCH[cls], "srlimit": n,
                                      "format": "json"}))["query"]["search"]
    for h in hits:
        page = json.loads(fetch.get(api, {"action": "parse", "page": h["title"], "prop": "text", "format": "json",
                                          "formatversion": 2}))
        yield {"doc_id": f"dermoscopedia:{h['title']}", "title": h["title"], "year": "", "licence": TEXTBOOK_LICENCE,
               "url": page_url(h["title"]),
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


def build_corpus(args, fetch, meditron_rows=None):
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


def corpus_args(argv=None):
    ap = argparse.ArgumentParser(prog="build.py corpus", description="Builds the knowledge corpus for RAG: real, published, citable text about the 8 diagnoses. Nothing is written by an LLM or by us.")
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
    return ap.parse_args(argv)


def corpus_main(argv=None):
    args = corpus_args(argv)
    get_logger("corpus", os.path.join(args.out, "build_corpus.log"))
    fetch = Fetcher(args.cache or os.path.join(args.out, "cache"))
    passages, removed = build_corpus(args, fetch)
    card = corpus_card(passages, removed, args, datetime.date.today().isoformat())
    write_outputs(passages, card, args.out)
    log.info(f"{card['n_passages']} passages from {card['n_documents']} documents -> {args.out}")
    if card["empty_topic_boxes"]:
        log.info("topics with no passage yet: " + ", ".join(card["empty_topic_boxes"]))


# ----------------------------------------------------------------------------------------------------------------------
# references: the reference texts (python src/build.py references)
# ----------------------------------------------------------------------------------------------------------------------
# Builds the reference text of each of the 8 diseases for the semantic-similarity score of src/evaluate.py: a paragraph quoted from
# Dermoscopedia (the dermoscopy wiki of the International Dermoscopy Society, CC BY-NC-SA 4.0), never written by us or by an AI.
#
# The fixed rule (applied the same way to every disease; steps 1-3 set on 2026-11-22 from the site's page layout, seen with
# --inspect; step 4 and the split into several texts per disease approved the same day, option B):
#   1. page      search Dermoscopedia for the disease's search terms, in their order; for the first term with any result, take the
#                result whose title equals the term (ignoring case), else that term's first result.
#   2. chapter   if that page has no sections and its introduction lists subchapters, take the first page it links to instead (once):
#                the chapter pages of melanoma and vascular lesions are only lists of their subchapters.
#   3. sections  the first section whose heading mentions dermoscopy ("dermoscop" or "dermatoscop") and the sections after it; if no
#                heading does, all sections in their order (their headings are the criteria themselves, e.g. "Large blue-gray ovoid
#                nests"); if the page has no sections, its introduction. Sections that are not text about the disease itself are left
#                out by their heading: references, literature, see also, external links, further reading, differential diagnosis.
#                Each section is read without its subsections (they are sections of their own), so no text is read twice.
#   4. texts     every section is split into texts of whole sentences, at most --max-words words each (a longer sentence is cut), in
#                order, without tables, references, edit links, the wiki's template lines ("cite!", "participate!", "(message)") or
#                leftovers of removed figure numbers ("(Figure )", ". ."), and without a reference list written as plain text at the end of a
#                section (cut from its first entry; see BIBLIOGRAPHY). A text under --min-words words is left out and listed under
#                "skipped". So a disease has one reference text per criterion section, not only the first 200 words of its page.
# The semantic-similarity score (src/evaluate.py) compares an explanation with every reference text of its true class and keeps the
# highest cosine (the best-matching reference).
# The search term of melanocytic nevus is the site's general page on nevus patterns: the page titled "Nevus" is about nail nevi.
#
# Output: data/knowledge_corpus/reference_texts.json
#   {"MEL": {"source": "Dermoscopedia: <page>", "url": ..., "rule": ..., "sections": ..., "licence": ..., "retrieved": ...,
#            "texts": [{"section": <heading>, "text": ..., "n_words": ...}, ...], "skipped": [{"section": ..., "n_words": ...}],
#            "bibliography_removed": [{"section": ..., "n_words": ...}]}}
# Check the texts once before any run (the script prints them): the page and sections must be about the disease named.
# --token-check counts each text's tokens with the similarity model's own tokenizer and compares them with the most the model reads
# (needs sentence-transformers; the model is downloaded once). Downloads are cached (data/knowledge_corpus/cache), so a rerun gives
# the same texts.
#
# Usage (on the laptop, with internet): python src/build.py references [--token-check]
#        python src/build.py references --show-tables   (the text of the tables the rule leaves out; writes nothing)
#        python src/build.py references --show-skipped  (the other text it leaves out: captions, divs, image alt text)
# Then:  python src/evaluate.py ... --reference-texts data/knowledge_corpus/reference_texts.json

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


def references_main(argv=None):
    ap = argparse.ArgumentParser(prog="build.py references", description="Reference texts of the 8 diseases, quoted from Dermoscopedia by a fixed rule.")
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

# ---------------------------------------------------------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------------------------------------------------------
COMMANDS = {"views": views_main, "corpus": corpus_main, "references": references_main}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if argv in ([], ["-h"], ["--help"]) else 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
