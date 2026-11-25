"""Step 1 of the knowledge corpus: downloading published text (PubMed, PMC, clinical guidelines, Dermoscopedia), cached."""
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter

from src.knowledge.extract import (
    clean,
    html_paragraphs,
    open_licence,
    parse_bioc,
    parse_pmc_licence,
    parse_pubmed_xml,
)
from src.knowledge.metadata import tag
from src.utils.io import write_json
from src.utils.logging import get_logger

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
