"""The text library for RAG (corpus): collecting the documents, extracting and cleaning their text, cutting it into passages,
their metadata, and the corpus audit (audit)."""
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

from src.utils.common import CLASSES, CORPUS_DIR, REFERENCE_TEXTS, load_env
from src.utils.io import read_jsonl, write_csv, write_json, write_jsonl
from src.utils.logging import get_logger

AI_PAPER = re.compile(r"\b(deep learning|machine learning|artificial intelligence|neural networks?|convolutional|"
                      r"computer[- ]aided|ISIC|HAM10000)\b", re.I)
OPEN_LICENCES = re.compile(r"^(CC0|CC BY(-SA|-ND|-NC|-NC-SA|-NC-ND)?)(\s|$)", re.I)
PMC_KEEP_SECTIONS = {"ABSTRACT", "INTRO", "CASE", "RESULTS", "DISCUSS", "CONCL", "FIG"}
PMC_KEEP_TYPES = {"abstract", "paragraph", "fig_caption"}


# ---- text handling (no internet) -------------------------------------------------------------------------------------------
def clean(text):
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def is_ai_paper(title, text):
    """True for papers about AI or image datasets, judged from the title and the start of the text (about an abstract)."""
    return bool(AI_PAPER.search(f"{title} {text[:1500]}"))


def open_licence(licence):
    return bool(licence) and bool(OPEN_LICENCES.match(licence.strip()))


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


# The wiki's page templates ("<page> – cite!", "<page> (message)", "<page> – participate!") are not text about the disease.
TEMPLATE = re.compile(r"cite!|participate!|\(message\)", re.I)
# The start of a reference list written as plain text at the end of a section (seen in AK and SCC on 2026-11-22): a sentence that
# begins with an author ("Kittler, H., Riedl, E. ...", "Kreusch, J. & Koch, F.") or names a book edition ("..., Second Edition.").
# A citation inside a sentence, such as "(Kittler et al., 2008)", does not match.
BIBLIOGRAPHY = re.compile(r"^[A-Z][\w'’-]+,\s(?:[A-Z]\.\s?)+[,&]|\b(?:\d+(?:st|nd|rd|th)|First|Second|Third|Fourth|Fifth|Sixth)"
                          r"\s+[Ee]dition\b")


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
# Topic coverage check (keyword-based, approximate): does the corpus speak about each diagnosis in each of these four ways?
TOPICS = {
    "appearance": r"dermoscop|dermatoscop|pigment network|globule|streak|blue-white|veil|vessel|vascular pattern|"
                  r"regression|milia|lacun|white patch|rosette|ulceration|border|asymmetr|colou?r",
    "patients": r"\bage[ds]?\b|elderly|older|young|incidence|prevalen|\bmen\b|\bwomen\b|male|female|sun|ultraviolet|uv\b|"
                r"risk factor|located|location|trunk|face|head|neck|extremit|\blegs?\b|\barms?\b",
    "differential": r"differential|distinguish|differentiat|mimic|simulat|confused with|versus|\bvs\.?\b|resembl",
    "basics": r"benign|malignan|carcinoma|tumou?r|neoplasm|lesion|prognos|histolog|patholog|definition|defined as",
}


def _synonym_pattern(words):
    """Whole words, with an optional plural ending: 'melanoma' also finds 'melanomas', 'carcinoma' finds 'carcinomas' (checked
    2026-11-24: the exact-word rule left such passages untagged)."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es)?\b")


SYNONYM_PATTERNS = {c: _synonym_pattern(words) for c, words in SYNONYMS.items()}


def tag(text, title=""):
    t = f"{title} {text}".lower()
    return sorted(c for c, rx in SYNONYM_PATTERNS.items() if rx.search(t))


def topics(text):
    t = text.lower()
    return sorted(k for k, rx in TOPICS.items() if re.search(rx, t))


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


log = get_logger("corpus")      # the corpus, reference, index and visual commands share this logger


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
GUIDELINE_EXCLUDED_SOURCES = {"wikidoc", "nice"}
TEXTBOOK_LICENCE = "CC BY-NC-SA 4.0"
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
DERMOSCOPEDIA_APIS = ("https://dermoscopedia.org/api.php", "https://dermoscopedia.org/w/api.php")
TOOL = "thesis_skin_rag_corpus"


# ---- downloading (internet; cached) -----------------------------------------------------------------------------------------
class Fetcher:
    def __init__(self, cache_dir, pause=0.4):
        self.cache, self.pause, self.key = cache_dir, pause, os.environ.get("NCBI_API_KEY")
        if self.key:
            self.pause = 0.12
        os.makedirs(cache_dir, exist_ok=True)

    def cache_path(self, url, params=None, ncbi=False):
        params = dict(params or {})
        if ncbi:
            params["tool"] = TOOL
        return os.path.join(self.cache, hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest())

    def forget(self, url, params=None, ncbi=False):
        """Removes a saved answer (one that turned out to be an error message), so that the next request asks again."""
        path = self.cache_path(url, params, ncbi)
        if os.path.exists(path):
            os.remove(path)

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
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        return body


def esearch(fetch, db, term, n, attempts=4, wait=15.0):
    """The ids found by an E-utilities search. An answer without a list of ids is an error message from NCBI (a busy server, a query
    it could not read): it is not kept in the download cache, the search is asked again after a pause, and if it still fails the search
    gives no ids and says what NCBI answered, so that one bad search does not stop a whole build."""
    params = {"db": db, "term": term, "retmax": n, "retmode": "json", "sort": "relevance"}
    for attempt in range(attempts):
        body = json.loads(fetch.get(NCBI + "esearch.fcgi", params, ncbi=True))
        r = body.get("esearchresult") or {}
        if "idlist" in r:
            break
        if hasattr(fetch, "forget"):
            fetch.forget(NCBI + "esearch.fcgi", params, ncbi=True)
        log.warning(f"{db} search failed (attempt {attempt + 1} of {attempts}): NCBI answered {str(body)[:300]!r}")
        if attempt + 1 < attempts:
            time.sleep(wait * (attempt + 1))
    else:
        return []
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
# Output (in --out, default data/corpus/):
#   passages.jsonl       the passages with their source, link and licence (stays on the laptop; see .gitignore)
#   manifest.csv         one row per passage without the text: id, library, document, link, licence, tags, MD5 (committed)
#   corpus_card.json     counts per library and diagnosis, what was removed and why, the topic coverage check, settings, date
#
# No e-mail address is sent. NCBI requests carry the tool name only, and the NCBI API key if the environment variable NCBI_API_KEY
# is set (optional; it allows more requests per second). Downloads are cached in --cache, so a rerun does not download again.
#
# Usage (on the laptop, with internet):
#   python -m src.main corpus --out data/corpus
#   python -m src.main corpus --out data/corpus --libraries pubmed pmc   # some libraries only


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
    ap.add_argument("--out", default=os.path.join("data", CORPUS_DIR))
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
    ap.add_argument("--corpus", default=os.path.join("data", CORPUS_DIR))
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


SHINGLE_WORDS = 8          # a passage and a reference text "share a stretch" when they have the same 8 words in a row
OVERLAP_LIMIT = 0.5        # a passage is a copy of a reference text when half or more of its 8-word stretches are in that text


def shingles(text, n=SHINGLE_WORDS):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def reference_shingles(path):
    """The 8-word stretches of every reference text of every class."""
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
