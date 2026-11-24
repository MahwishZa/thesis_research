#!/usr/bin/env python3
"""
Builds what the experiments read from the cleaned cases and from public sources. Eight commands (each has --help):

  python src/build.py views        the three views and their question files, from data/processed_cases/all_cases.csv (step 2)
  python src/build.py corpus       the knowledge corpus for RAG: four libraries of real, published text (internet; laptop)
  python src/build.py references   the reference texts of the 8 diseases for the semantic-similarity score (internet; laptop)
  python src/build.py index        the dense index of the corpus passages for the RAG systems (MedCPT article encoder; laptop)
  python src/build.py audit        checks of the corpus text: near-duplicates, copies of the reference texts, tags (no internet)
  python src/build.py visual       the figure list of the visual library: skin-image figures and captions of the corpus papers
  python src/build.py visual-images  the images of those figures (internet; laptop; --probe N first)
  python src/build.py visual-index   the image index, without any image that looks like a Kaggle image (laptop; open_clip_torch)

The rules of each command are written above its code. The standard library is enough, except for the index (torch and transformers).
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

from common import (ALL_CASES, CLASSES, INDEX_DIR, KNOWLEDGE_CORPUS, LETTERS, MEDCPT_ARTICLE, OPTION_TEXT, SEED, SPLITS, VIEWS,
                    drop_near_duplicates, file_sha256, get_logger, image_encoder, load_env, max_similarities, medcpt_encoder, question_file, read_csv, read_jsonl, view_table, write_csv,
                    write_index, write_json, write_jsonl)

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
    "MEL": ["melanoma", "lentigo maligna", "melanoma in situ", "cutaneous melanoma"],
    "NV": ["nevus", "naevus", "nevi", "naevi", "melanocytic nevus"],
    "BCC": ["basal cell carcinoma", "basal-cell carcinoma", "basalioma", "basal cell epithelioma"],
    "AK": ["actinic keratosis", "actinic keratoses", "solar keratosis", "solar keratoses"],
    "BKL": ["seborrheic keratosis", "seborrhoeic keratosis", "seborrheic keratoses", "solar lentigo", "solar lentigines",
            "lichen planus-like keratosis", "benign keratosis"],
    "DF": ["dermatofibroma", "dermatofibromas", "benign fibrous histiocytoma"],
    "VASC": ["hemangioma", "haemangioma", "cherry angioma", "angioma", "angiokeratoma", "pyogenic granuloma", "vascular lesion",
             "lobular capillary hemangioma"],
    "SCC": ["squamous cell carcinoma", "squamous-cell carcinoma", "keratoacanthoma", "bowen's disease", "bowen disease"],
}
PUBMED_QUERIES = {
    "MEL": '"melanoma"[MeSH Terms] AND "skin neoplasms"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "NV": '"nevus, pigmented"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    # NLM renamed this MeSH heading for 2026 ("Carcinoma, Basal Cell" -> "Basal Cell Carcinoma"): a quoted heading that is not found
    # makes the whole AND query empty, so both names and the plain words are listed
    "BCC": '("carcinoma, basal cell"[MeSH Terms] OR "basal cell carcinoma"[MeSH Terms] OR "basal cell carcinoma"[tiab]) '
           'AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
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


def _synonym_pattern(words):
    """Whole words, with an optional plural ending: 'melanoma' also finds 'melanomas', 'carcinoma' finds 'carcinomas' (checked
    2026-11-24: the exact-word rule left such passages untagged)."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es)?\b")


SYNONYM_PATTERNS = {c: _synonym_pattern(words) for c, words in SYNONYMS.items()}


def tag(text, title=""):
    t = f"{title} {text}".lower()
    return sorted(c for c, rx in SYNONYM_PATTERNS.items() if rx.search(t))


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


XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
CC_URL = re.compile(r"creativecommons\.org/(?:licenses|publicdomain)/([a-z-]+)(?:/([\d.]+))?", re.I)


def parse_pmc_licence(xml_text):
    """The licence and the retraction flag of one PMC paper, from its full-text XML (JATS, E-utilities efetch db=pmc):
    (licence such as "CC BY 4.0", retracted), or (None, retracted) when the paper carries no machine-readable Creative Commons
    licence. The licence is read from the <license> element's link to creativecommons.org (licenses/by, by-nc, ... or
    publicdomain/zero for CC0). A retraction notice (article-type "retraction") or a paper with a related article of a "retract..."
    type is flagged as retracted. (The PMC OA web service that gave this before is being retired by NLM in 2026.)"""
    root = ET.fromstring(xml_text)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        return None, False
    retracted = article.get("article-type") == "retraction" or any(
        "retract" in (r.get("related-article-type") or "") for r in article.iter("related-article"))
    for lic in article.iter("license"):
        m = CC_URL.search(lic.get(XLINK_HREF) or "") or CC_URL.search("".join(lic.itertext()))
        if m:
            kind, version = m.group(1).lower(), m.group(2)
            name = "CC0" if kind == "zero" else "CC " + kind.upper()
            return (name + (f" {version}" if version else "")), retracted
    return None, retracted


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
                             ncbi=True))["esearchresult"]
    missing = (r.get("warninglist") or {}).get("quotedphrasesnotfound")
    if not r["idlist"] or missing:                          # say why, with PubMed's own reading of the query (no secret in it)
        what = "found nothing" if not r["idlist"] else f"did not find {missing}"
        log.warning(f"{db} search {what} (count {r.get('count')}); query as read by NCBI: {r.get('querytranslation')!r}; "
                    f"errors: {r.get('errorlist')}; warnings: {r.get('warninglist')}")
    return r["idlist"]


def pubmed_docs(fetch, cls, n):
    ids = esearch(fetch, "pubmed", PUBMED_QUERIES[cls] + " AND hasabstract[text] AND english[lang]", n)
    for i in range(0, len(ids), 100):
        yield from parse_pubmed_xml(fetch.get(NCBI + "efetch.fcgi", {"db": "pubmed", "id": ",".join(ids[i:i + 100]),
                                                                      "retmode": "xml"}, ncbi=True))


def pmc_docs(fetch, term, n, removed):
    for uid in esearch(fetch, "pmc", term + ' AND "open access"[filter]', n):
        pmcid = f"PMC{uid}"
        try:
            licence, retracted = parse_pmc_licence(fetch.get(NCBI + "efetch.fcgi", {"db": "pmc", "id": uid, "retmode": "xml"},
                                                              ncbi=True))
        except Exception as e:                               # one paper must not stop the whole build; the card counts them
            log.warning(f"{pmcid}: licence lookup failed ({type(e).__name__}: {e})")
            removed["pmc: licence lookup failed"] += 1
            continue
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


def meditron_rows_from_hub():
    """The public Meditron guideline collection, streamed from Hugging Face (one large file; no token is needed, but a token in the
    environment as HF_TOKEN is used by the Hugging Face library and makes the download faster)."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")            # the library's default (10 s) is too short on a slow line
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")      # Windows without developer mode: a harmless warning
    from datasets import load_dataset
    return load_dataset("epfl-llm/guidelines", split="train", streaming=True)


def meditron_guideline_docs(removed, rows=None, cache=None, load=meditron_rows_from_hub):
    """The public Meditron guideline collection, skin-cancer articles only, without the excluded sources.
    With `cache` (a file path) the answer is saved there after one complete pass and read from there afterwards, so the large
    download is made once; an interrupted pass saves nothing."""
    if rows is None and cache and os.path.isfile(cache):
        with open(cache, encoding="utf-8") as f:
            saved = json.load(f)
        removed.update(saved["removed"])
        return saved["docs"]
    docs, dropped = [], Counter()
    for r in (load() if rows is None else rows):
        source = (r.get("source") or "").lower()
        text = r.get("clean_text") or r.get("raw_text") or ""
        if source in GUIDELINE_EXCLUDED_SOURCES:
            dropped[f"guidelines: source {source} excluded"] += 1
            continue
        if not tag(text[:5000], r.get("title", "")):
            continue                                            # not about the 8 diagnoses
        docs.append({"doc_id": f"meditron:{r.get('id', '')}", "title": clean(r.get("title")), "year": "",
                     "paragraphs": [p for p in re.split(r"\n\s*\n", text) if p.strip()], "url": r.get("url") or "",
                     "licence": f"Meditron guidelines collection (source: {source}); see the source's terms", "retracted": False})
    removed.update(dropped)
    if rows is None and cache:
        write_json(cache, {"docs": docs, "removed": dict(dropped)})
    return docs


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
# Text that is not about a disease (seen in the laptop corpus on 2026-11-24): a wiki's placeholder text, its page-template lines and
# literature lists cut into passages. Such a passage is left out; the markup left in a real passage is removed.
CITATION_START = re.compile(r"^[A-Z][\w'’-]+,?\s(?:[A-Z]\.?){1,3}[,.&]")
CITATION_MARK = re.compile(r"\bPMID\b|\bdoi\b|\bEpub\b|\bet al\.,|\bJ [A-Z][a-z]+\b|\bBr J\b|\bAm J\b|\b[12]\d{3}[;,.]\s*\d+")
VOLUME_PAGES = re.compile(r"\b\d+\(\d+\):\d+[-–]\d+")
MARKUP = re.compile(r"={2,}\s*(.*?)\s*={2,}")
URL = re.compile(r"https?://\S+")


def junk_reason(text):
    """Why a passage is not useful text about a disease, or None."""
    if re.search(r"lorem ipsum", text, re.IGNORECASE):
        return "placeholder text"
    if TEMPLATE.search(text):
        return "wiki template line"
    if BIBLIOGRAPHY.search(text) or VOLUME_PAGES.search(text) or (CITATION_START.search(text) and CITATION_MARK.search(text)):
        return "reference list entry"
    return None


def strip_markup(text):
    """Removes wiki heading marks ('===== Title =====') and web addresses from a passage."""
    return clean(URL.sub(" ", MARKUP.sub(lambda m: m.group(1) + ".", text)))


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
            "settings": {k: v for k, v in vars(args).items() if k not in ("out", "cache", "env_file")}}


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

# ----------------------------------------------------------------------------------------------------------------------
# index: the dense index of the corpus for the RAG systems (python src/build.py index)
# ----------------------------------------------------------------------------------------------------------------------
# Each passage of data/knowledge_corpus/passages.jsonl is embedded once, as [title, text], with the MedCPT article encoder (Jin et al.
# 2023), and the unit-length vectors are saved in data/knowledge_corpus/index/ (laptop only: they are derived from licensed text). The
# corpus is already cut into passages by `build.py corpus` (whole sentences, at most 200 words, one sentence shared with the next
# passage), so nothing is cut again here. Tags and query classes are not embedded and never used for retrieval.
#
# --exclude-reference-pages leaves out the passages of the Dermoscopedia pages that also supply the reference texts of the
# semantic-similarity score (src/evaluate.py). Without it, a RAG answer that quotes such a passage resembles the reference text of its
# own disease almost word for word, which inflates that score. The choice is the researcher's; it is recorded in index.json.
ARTICLE_MAX_TOKENS = 512
SHINGLE_WORDS = 8          # a passage and a reference text "share a stretch" when they have the same 8 words in a row
OVERLAP_LIMIT = 0.5        # a passage is a copy of a reference text when half or more of its 8-word stretches are in that text


def shingles(text, n=SHINGLE_WORDS):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def reference_shingles(corpus):
    """The 8-word stretches of every reference text of every class."""
    path = os.path.join(corpus, "reference_texts.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is needed for the reference-text checks")
    with open(path, encoding="utf-8") as f:
        refs = json.load(f)
    out = set()
    for r in refs.values():
        for t in r["texts"]:
            out |= shingles(t["text"])
    return out


def overlap_share(text, ref_shingles):
    """Share of the passage's 8-word stretches that also occur in the reference texts (0 for a passage shorter than 8 words)."""
    sh = shingles(text)
    return len(sh & ref_shingles) / len(sh) if sh else 0.0


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
    ap = argparse.ArgumentParser(prog="build.py audit", description="Checks the corpus text (no internet, nothing is changed).")
    ap.add_argument("--corpus", default=os.path.join("data", KNOWLEDGE_CORPUS))
    a = ap.parse_args(argv)
    get_logger("audit")
    passages = read_jsonl(os.path.join(a.corpus, "passages.jsonl"))
    result = audit_corpus(passages, reference_shingles(a.corpus))
    out = os.path.join(a.corpus, "audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    log.info(f"{result['n_passages']} passages: {len(result['near_duplicate_pairs'])} near-duplicate pairs, "
             f"{len(result['reference_copies'])} copies of reference texts, {len(result['tags_that_would_change'])} tags that the "
             f"current rule would change; details in {out}")
    for name, row in sorted(result["by_library"].items()):
        log.info(f"  {name}: {row}")
    return result


def index_args(argv=None):
    ap = argparse.ArgumentParser(prog="build.py index", description="Embeds the corpus passages for the RAG systems (MedCPT).")
    ap.add_argument("--corpus", default=os.path.join("data", KNOWLEDGE_CORPUS), help="the folder with passages.jsonl")
    ap.add_argument("--out", default=os.path.join("data", INDEX_DIR), help="where the index is written")
    ap.add_argument("--encoder", default=MEDCPT_ARTICLE)
    ap.add_argument("--exclude-reference-pages", action="store_true",
                    help="leave out the passages of the pages that supply data/knowledge_corpus/reference_texts.json")
    ap.add_argument("--exclude-reference-overlap", action="store_true",
                    help="also leave out any passage that copies a reference text (half or more of its 8-word stretches)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N passages (a trial of the command)")
    return ap.parse_args(argv)


def reference_urls(corpus):
    path = os.path.join(corpus, "reference_texts.json")
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
        raise SystemExit(f"{path} is not there: run python src/build.py corpus first (on the laptop)")
    passages = read_jsonl(path)
    if a.limit:
        passages = passages[:a.limit]
    urls = reference_urls(a.corpus) if a.exclude_reference_pages else []
    encode = encode or medcpt_encoder(a.encoder, ARTICLE_MAX_TOKENS)
    refs = reference_shingles(a.corpus) if a.exclude_reference_overlap else None
    kept, vectors, left_out = build_index(passages, encode, urls, refs)
    write_index(a.out, kept, vectors, {"encoder": a.encoder, "max_tokens": ARTICLE_MAX_TOKENS, "normalised": True,
                                       "passages_file": path.replace("\\", "/"), "passages_sha256": file_sha256(path),
                                       "exclude_reference_pages": bool(a.exclude_reference_pages),
                                       "exclude_reference_overlap": bool(a.exclude_reference_overlap), "excluded_urls": urls,
                                       "n_excluded_passages": len(left_out), "limit": a.limit})
    log.info(f"{len(kept)} passages embedded ({len(left_out)} left out) -> {a.out}")
    return kept


# ----------------------------------------------------------------------------------------------------------------------
# visual: the figures of the visual library (python src/build.py visual)
# ----------------------------------------------------------------------------------------------------------------------
# The visual library is made of real figures with their real captions, taken from the open-access PMC papers that are already in the
# text corpus (same papers, same licences). Nothing is drawn, described or captioned by a model. This step makes the LIST of figures
# (no image is downloaded here): it reads each paper's full-text XML (the same request the corpus build already made, so it comes
# from the download cache) and keeps a figure only when
#   1. it has an image file,
#   2. its caption has at least 8 words and is not a duplicate of another kept caption,
#   3. its caption is not about a chart, a flow diagram, a table of results or another imaging method (MRI, CT, ultrasound, OCT,
#      confocal, X-ray, endoscopy),
#   4. its caption names a dermoscopic image, a clinical photograph or a lesion image, and is not only about histology,
#   5. its caption or the paper's title names one of the 8 diagnoses (the same word lists as the text corpus).
# The list is written to data/knowledge_corpus/visual/items.jsonl (licensed text: not committed) and a card with the counts to
# data/knowledge_corpus/visual_card.json (committed). Downloading the images and the check against the Kaggle images are separate
# steps, run only after this list has been looked at.
FIG_NOT_A_SKIN_IMAGE = re.compile(
    r"flow ?chart|flow diagram|forest plot|kaplan|prisma|algorithm|schematic|workflow|bar (?:chart|graph)|histogram|roc curve|"
    r"scatter|heat ?map|pathway|\bmri\b|magnetic resonance|\bct\b|computed tomograph|ultrasound|ultrasonograph|\boct\b|"
    r"optical coherence|confocal|\bpet\b|radiograph|x-ray|mammogra|endoscop|colonoscop|fundus|flow-chart|pie chart|word cloud",
    re.IGNORECASE)
FIG_SKIN_IMAGE = re.compile(
    r"dermoscop|dermatoscop|trichoscop|clinical (?:image|photo|picture|presentation|appearance|view|feature|finding)|photograph|"
    r"macroscopic|close-?up|\blesions?\b", re.IGNORECASE)
FIG_HISTOLOGY = re.compile(
    r"h ?& ?e\b|hematoxylin|haematoxylin|histolog|histopatholog|immunohistochem|immunostain|magnification|biopsy specimen|"
    r"original magnification|×\s?\d+|\bx ?\d{2,3}\b|panoramic|low[- ]power|high[- ]power|\bstain", re.IGNORECASE)
# words that say the picture is of the skin itself ("clinical-pathological correlation" is not one of them)
FIG_CLINICAL_WORD = re.compile(
    r"dermoscop|dermatoscop|trichoscop|clinical (?:image|photo|picture|presentation|appearance|view)|photograph|macroscopic|close-?up",
    re.IGNORECASE)


def parse_pmc_figures(xml_text):
    """The figures of one PMC paper, from its full-text XML: [{"label", "caption", "href"}, ...] (href is None for a figure without an
    image file). The caption is its own text, whitespace tidied."""
    root = ET.fromstring(xml_text)
    out = []
    for fig in root.iter("fig"):
        label = clean("".join(fig.find("label").itertext())) if fig.find("label") is not None else ""
        cap = fig.find("caption")
        blocks = [b for b in cap if b.tag in ("title", "p")] if cap is not None else []
        caption = clean(" ".join("".join(b.itertext()) for b in blocks) if blocks else "".join(cap.itertext()) if cap is not None else "")
        graphic = next((g for g in fig.iter("graphic") if g.get(XLINK_HREF)), None)
        out.append({"label": label, "caption": caption, "href": graphic.get(XLINK_HREF) if graphic is not None else None})
    return out


def figure_reject_reason(fig, title, seen_captions, min_words=8):
    """Why a figure is left out of the visual library, or None to keep it (and then remember its caption in `seen_captions`)."""
    caption = fig["caption"]
    if not fig["href"]:
        return "no image file"
    if len(caption.split()) < min_words:
        return "caption too short"
    if FIG_NOT_A_SKIN_IMAGE.search(caption):
        return "chart, diagram or other imaging method"
    if not FIG_SKIN_IMAGE.search(caption):
        return "not a clinical or dermoscopic image"
    if FIG_HISTOLOGY.search(caption) and not FIG_CLINICAL_WORD.search(caption):
        return "histology only"
    if not tag(caption, title):
        return "no diagnosis named"
    key = hashlib.md5(re.sub(r"\W+", " ", caption.lower()).strip().encode()).hexdigest()
    if key in seen_captions:
        return "duplicate caption"
    seen_captions.add(key)
    return None


def visual_items(papers, fetch_xml, removed):
    """The kept figures of the given papers ([{"doc_id", "title", "licence", "query_class", "url"}]); fetch_xml(doc_id) gives the XML."""
    seen, out = set(), []
    for d in papers:
        try:
            figs = parse_pmc_figures(fetch_xml(d["doc_id"]))
        except Exception as e:                                       # one paper must not stop the build; the card counts them
            log.warning(f"{d['doc_id']}: figures not read ({type(e).__name__}: {e})")
            removed["paper: figures not read"] += 1
            continue
        for k, fig in enumerate(figs):
            reason = figure_reject_reason(fig, d["title"], seen)
            if reason:
                removed[reason] += 1
                continue
            item_id = f"{d['doc_id']}#fig{k}"
            # an item is also a passage (passage_id, title, text) so that the same index, prompt and evidence code serve it
            out.append({"item_id": item_id, "passage_id": item_id, "doc_id": d["doc_id"], "doc_title": d["title"],
                        "title": f"{d['title']} ({fig['label'] or 'Figure'})", "text": fig["caption"], "label": fig["label"],
                        "caption": fig["caption"], "href": fig["href"], "library": "visual", "licence": d["licence"], "url": d["url"],
                        "query_class": d["query_class"], "tags": tag(fig["caption"], d["title"]),
                        "n_words": len(fig["caption"].split())})
    return out


def corpus_papers(passages):
    """One record per PMC paper of the text corpus (the first passage of each gives its title, licence, class and web address)."""
    papers = {}
    for p in passages:
        if re.fullmatch(r"PMC\d+", p["doc_id"]) and p["doc_id"] not in papers:
            papers[p["doc_id"]] = {"doc_id": p["doc_id"], "title": p["title"], "licence": p["licence"],
                                   "query_class": p["query_class"], "url": p["url"]}
    return list(papers.values())


def paper_from_xml(doc_id, xml_text, cls):
    """The record of a PMC paper read from its own full-text XML, or None with the reason: (record, None) or (None, reason).
    Left out: no open Creative Commons licence, retracted, or a paper about AI or image datasets (the same rules as the text corpus)."""
    licence, retracted = parse_pmc_licence(xml_text)
    if retracted:
        return None, "retracted"
    if licence is None or not open_licence(licence):
        return None, "no open licence"
    root = ET.fromstring(xml_text)
    title_el, abstract_el = root.find(".//article-title"), root.find(".//abstract")
    title = clean("".join(title_el.itertext())) if title_el is not None else ""
    abstract = clean("".join(abstract_el.itertext())) if abstract_el is not None else ""
    if is_ai_paper(title, abstract):
        return None, "AI or image-dataset paper"
    return {"doc_id": doc_id, "title": title, "licence": licence, "query_class": cls,
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{doc_id}/"}, None


def extra_papers(fetch, per_class, known, removed):
    """More open-access PMC papers for the figure list: for each diagnosis, the papers found by the same PMC search as the text corpus
    (PMC_QUERIES), up to `per_class` per diagnosis, without the passage cap of the text corpus. Papers in `known` are skipped."""
    out, seen = [], set(known)
    for cls in CLASSES:
        for uid in esearch(fetch, "pmc", PMC_QUERIES[cls] + ' AND "open access"[filter]', per_class):
            doc_id = f"PMC{uid}"
            if doc_id in seen:
                continue
            seen.add(doc_id)
            try:
                paper, why = paper_from_xml(doc_id, fetch.get(NCBI + "efetch.fcgi", {"db": "pmc", "id": uid, "retmode": "xml"},
                                                              ncbi=True), cls)
            except Exception as e:                           # one paper must not stop the build; the card counts them
                log.warning(f"{doc_id}: not read ({type(e).__name__}: {e})")
                removed["extra paper: not read"] += 1
                continue
            if paper is None:
                removed[f"extra paper: {why}"] += 1
                continue
            out.append(paper)
    return out


def visual_args(argv=None):
    ap = argparse.ArgumentParser(prog="build.py visual", description="Lists the skin-image figures of the corpus papers.")
    ap.add_argument("--corpus", default=os.path.join("data", KNOWLEDGE_CORPUS))
    ap.add_argument("--cache", default=None, help="download cache (default: <corpus>/cache)")
    ap.add_argument("--limit", type=int, default=0, help="read only the first N papers (a trial)")
    ap.add_argument("--extra-papers", type=int, default=0,
                    help="also search PMC for up to N more open-access papers per diagnosis (internet), beyond the text corpus papers")
    return ap.parse_args(argv)


def visual_main(argv=None, fetch_xml=None, fetch=None):
    a = visual_args(argv)
    get_logger("visual", os.path.join(a.corpus, "build_visual.log"))
    passages = read_jsonl(os.path.join(a.corpus, "passages.jsonl"))
    papers = corpus_papers(passages)
    if a.limit:
        papers = papers[:a.limit]
    removed = Counter()
    if fetch is None and (fetch_xml is None or a.extra_papers):
        fetch = Fetcher(a.cache or os.path.join(a.corpus, "cache"))
    if a.extra_papers:
        papers += extra_papers(fetch, a.extra_papers, {p["doc_id"] for p in papers}, removed)
    if fetch_xml is None:
        def fetch_xml(doc_id):
            return fetch.get(NCBI + "efetch.fcgi", {"db": "pmc", "id": doc_id[3:], "retmode": "xml"}, ncbi=True)
    items = visual_items(papers, fetch_xml, removed)
    out_dir = os.path.join(a.corpus, "visual")
    write_jsonl(os.path.join(out_dir, "items.jsonl"), items)
    per_class = Counter(t for it in items for t in it["tags"])
    card = {"snapshot": datetime.date.today().isoformat(), "n_papers_read": len(papers), "n_figures_kept": len(items),
            "n_papers_with_a_figure": len({it["doc_id"] for it in items}), "figures_per_tag": {c: per_class.get(c, 0) for c in CLASSES},
            "left_out": dict(sorted(removed.items())), "images_downloaded": False,
            "rule": "see the comment above visual_items in src/build.py"}
    write_json(os.path.join(a.corpus, "visual_card.json"), card)
    log.info(f"{len(items)} figures kept from {card['n_papers_with_a_figure']} of {len(papers)} papers; left out: {dict(removed)}")
    return items


# ----------------------------------------------------------------------------------------------------------------------
# visual-images: download the images of the listed figures (python src/build.py visual-images)
# ----------------------------------------------------------------------------------------------------------------------
# Where the images come from: the PMC Open Access data on AWS, the public bucket "pmc-oa-opendata" (NCBI: pmc.ncbi.nlm.nih.gov/tools/
# pmcaws). Each article has a folder "<PMC id>.<version>/" that holds its XML and its figure files. For each figure the folder of the
# newest version is listed (a plain HTTPS request, no account) and the file whose name equals the figure's graphic name (with or
# without its extension) is downloaded. If the bucket has no such file, the older Europe PMC address pattern is tried last.
# An answer counts as an image only when its first bytes are those of a JPEG, PNG or GIF file. A file under 3,000 bytes (an icon) is
# left out, and so is an image that is byte-for-byte the same as one already kept. --probe N tries only N figures, saves nothing and
# prints what each address answered, so that a wrong address pattern can be seen at once. The images are licensed material: they
# stay in data/knowledge_corpus/visual/images/ (git-ignored); the card records the counts.
S3_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
EUROPE_PMC_IMAGE = "https://europepmc.org/articles/{doc}/bin/{href}{ext}"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")
IMAGE_MAGIC = {b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png", b"GIF8": ".gif"}
MIN_IMAGE_BYTES = 3000
MAX_IMAGE_BYTES = 20_000_000
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def image_extension(data):
    """".jpg", ".png" or ".gif" when the bytes start like such a file, else None."""
    return next((ext for magic, ext in IMAGE_MAGIC.items() if data.startswith(magic)), None)


def download_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": TOOL})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(MAX_IMAGE_BYTES + 1)


def s3_list(prefix, get, delimiter=""):
    """(keys, sub-prefixes) under a prefix of the PMC bucket (the listing is XML; one page of up to 1,000 entries)."""
    params = {"list-type": "2", "prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    root = ET.fromstring(get(S3_BUCKET + "?" + urllib.parse.urlencode(params)))
    keys = [c.findtext(S3_NS + "Key") for c in root.iter(S3_NS + "Contents")]
    prefixes = [c.findtext(S3_NS + "Prefix") for c in root.iter(S3_NS + "CommonPrefixes")]
    return keys, prefixes


def article_image_keys(doc_id, get, cache):
    """{file stem: key} of the image files in the newest version folder of one article (remembered in `cache`)."""
    if doc_id not in cache:
        _, versions = s3_list(doc_id + ".", get, delimiter="/")
        def number(p):
            m = re.search(r"\.(\d+)/$", p)
            return int(m.group(1)) if m else -1
        newest = max(versions, key=number) if versions else None
        files = {}
        if newest:
            keys, _ = s3_list(newest, get)
            for key in keys:
                base = key.rsplit("/", 1)[-1]
                stem, ext = os.path.splitext(base)
                if ext.lower() in IMAGE_EXTENSIONS:
                    files.setdefault(stem, key)
                    files.setdefault(base, key)
        cache[doc_id] = files
    return cache[doc_id]


def fetch_image(doc_id, href, get=download_bytes, cache=None, trace=None):
    """(bytes, url) of the first address that gives a real image, or (None, None). `trace` collects what each address answered."""
    cache = {} if cache is None else cache
    note = trace.append if trace is not None else (lambda _: None)
    urls = []
    try:
        key = article_image_keys(doc_id, get, cache).get(href) or article_image_keys(doc_id, get, cache).get(os.path.splitext(href)[0])
        if key:
            urls.append(S3_BUCKET + key)
        else:
            note(f"{doc_id}: no file named {href} in the PMC bucket")
    except Exception as e:                                           # the listing failed: say why, then try the other address
        note(f"{doc_id}: bucket listing failed ({type(e).__name__}: {e})")
    urls += [EUROPE_PMC_IMAGE.format(doc=doc_id, href=href, ext=ext) for ext in (".jpg", ".png", ".gif")]
    for url in urls:
        try:
            data = get(url)
        except Exception as e:                                       # not found, refused, a time-out: try the next address
            note(f"{url}: {type(e).__name__}: {e}")
            continue
        if data and len(data) <= MAX_IMAGE_BYTES and image_extension(data):
            return data, url
        note(f"{url}: answered, but not an image file (starts {data[:12]!r})")
    return None, None


def download_images(items, folder, get=download_bytes, pause=0.4, probe=0, trace=None):
    """Saves the images; returns (items that have an image file, counts, the kinds of address that worked)."""
    os.makedirs(folder, exist_ok=True)
    kept, counts, worked, seen, cache = [], Counter(), Counter(), set(), {}
    for it in items[:probe] if probe else items:
        data, url = fetch_image(it["doc_id"], it["href"], get, cache, trace)
        if pause:
            time.sleep(pause)
        if data is None:
            counts["no image at any address"] += 1
            continue
        if len(data) < MIN_IMAGE_BYTES:
            counts["image too small"] += 1
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            counts["identical image already kept"] += 1
            continue
        seen.add(digest)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", it["item_id"]) + image_extension(data)
        if not probe:
            with open(os.path.join(folder, name), "wb") as f:
                f.write(data)
        worked["PMC bucket" if url.startswith(S3_BUCKET) else "Europe PMC"] += 1
        kept.append({**it, "image_file": name, "image_sha256": digest, "image_url": url, "image_bytes": len(data)})
    return kept, counts, worked


def visual_images_args(argv=None):
    ap = argparse.ArgumentParser(prog="build.py visual-images", description="Downloads the images of the listed figures.")
    ap.add_argument("--corpus", default=os.path.join("data", KNOWLEDGE_CORPUS))
    ap.add_argument("--probe", type=int, default=0, help="try only the first N figures and save nothing: which addresses work?")
    return ap.parse_args(argv)


def visual_images_main(argv=None, get=download_bytes):
    a = visual_images_args(argv)
    log = get_logger("visual-images", os.path.join(a.corpus, "build_visual.log"))
    folder = os.path.join(a.corpus, "visual")
    items = read_jsonl(os.path.join(folder, "items.jsonl"))
    trace = []
    kept, counts, worked = download_images(items, os.path.join(folder, "images"), get, pause=0.4 if get is download_bytes else 0,
                                           probe=a.probe, trace=trace)
    if a.probe or not kept:
        for line in trace[:12]:
            log.info("  " + line)
    log.info(f"{len(kept)} images of {a.probe or len(items)} figures tried; addresses that worked: {dict(worked)}; "
             f"left out: {dict(counts)}")
    if not a.probe:
        write_jsonl(os.path.join(folder, "items_with_images.jsonl"), kept)
        card_path = os.path.join(a.corpus, "visual_card.json")
        with open(card_path, encoding="utf-8") as f:
            card = json.load(f)
        card.update({"images_downloaded": True, "n_images": len(kept), "images_left_out": dict(sorted(counts.items())),
                     "image_addresses_that_worked": dict(worked)})
        write_json(card_path, card)
    return kept


# ----------------------------------------------------------------------------------------------------------------------
# visual-index: the image index and the check against the Kaggle images (python src/build.py visual-index)
# ----------------------------------------------------------------------------------------------------------------------
# Every library image is embedded with an image encoder (laptop: open_clip_torch, Pillow). So that no test image can be "looked up",
# every Kaggle image of the study (all splits, data/processed_images/) is embedded too, and a library image whose largest cosine
# similarity with any Kaggle image is at least --threshold is left out. The threshold is a safety choice made before any result was
# seen (default 0.95: images that are almost the same picture); it is NOT tuned. So that it can be judged, the card records the
# 20 highest similarities (library item, Kaggle case) and the quantiles of all of them. The Kaggle vectors are used for this check
# only; they are never part of the index or of any answer.
def visual_index_args(argv=None):
    ap = argparse.ArgumentParser(prog="build.py visual-index", description="Embeds the library images and removes Kaggle look-alikes.")
    ap.add_argument("--corpus", default=os.path.join("data", KNOWLEDGE_CORPUS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--encoder", default="biomedclip", choices=("biomedclip",))
    ap.add_argument("--threshold", type=float, default=0.95, help="a library image this similar to a Kaggle image is left out")
    ap.add_argument("--library-threshold", type=float, default=0.98,
                    help="of two library images this similar, the later one is left out (the same picture in two papers)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N library images (a trial)")
    return ap.parse_args(argv)


def visual_index_main(argv=None, encode=None):
    a = visual_index_args(argv)
    log = get_logger("visual-index", os.path.join(a.corpus, "build_visual.log"))
    folder = os.path.join(a.corpus, "visual")
    items = read_jsonl(os.path.join(folder, "items_with_images.jsonl"))
    if a.limit:
        items = items[:a.limit]
    kaggle = sorted(f for f in os.listdir(os.path.join(a.data, "processed_images")) if f.lower().endswith(".jpg"))
    if not items or not kaggle:
        raise SystemExit("no library images or no Kaggle images: run visual-images first, and check data/processed_images/")
    encode = encode or image_encoder(a.encoder)
    vectors = encode([os.path.join(folder, "images", it["image_file"]) for it in items])
    reference = encode([os.path.join(a.data, "processed_images", f) for f in kaggle])
    best = max_similarities(vectors, reference)
    not_kaggle = [i for i, (sim, _) in enumerate(best) if sim < a.threshold]
    keep_idx, twins = drop_near_duplicates([vectors[i] for i in not_kaggle], a.library_threshold)
    kept = [(items[not_kaggle[i]], vectors[not_kaggle[i]]) for i in keep_idx]
    dropped = [{"item_id": it["item_id"], "similarity": round(sim, 4), "kaggle_image": kaggle[j]}
               for it, (sim, j) in zip(items, best) if sim >= a.threshold]
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
                                                              "library_threshold": a.library_threshold,
                                                              "n_dropped_as_library_duplicates": len(twins),
                                                              "similarity_quantiles": quantiles, "highest_20": top},
                 "n_indexed_images": len(kept)})
    write_json(card_path, card)
    log.info(f"{len(kept)} images indexed, {len(dropped)} dropped as Kaggle look-alikes (threshold {a.threshold}) and "
             f"{len(twins)} as duplicates inside the library (threshold {a.library_threshold}); "
             f"similarity quantiles {quantiles}")
    return [it for it, _ in kept]


# ---------------------------------------------------------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------------------------------------------------------
COMMANDS = {"views": views_main, "corpus": corpus_main, "references": references_main, "index": index_main,
            "audit": audit_main, "visual": visual_main,
            "visual-images": visual_images_main, "visual-index": visual_index_main}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if argv in ([], ["-h"], ["--help"]) else 2
    result = COMMANDS[argv[0]](argv[1:])
    return result if isinstance(result, int) else 0      # the command's result (e.g. the kept passages) is not an exit code


if __name__ == "__main__":
    sys.exit(main())
