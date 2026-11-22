#!/usr/bin/env python3
"""
Builds a REAL (non-synthetic) medical knowledge corpus for RAG over the 8 diagnoses.

Sources (all published, citable text - nothing generated):
  1. PubMed abstracts via NCBI E-utilities (free; set --email, optional --api-key).
  2. MedRAG "MedCorp" corpora on Hugging Face - the same corpora RAG^2 retrieves from:
       MedRAG/statpearls, MedRAG/textbooks (and optionally MedRAG/pubmed, MedRAG/wikipedia).
     Only dermatology-relevant chunks are kept (keyword filter below).

Output: corpus.jsonl  -> {chunk_id, source, doc_id, title, text, url, year, tags}
        tags = diagnoses the chunk talks about (used as weak relevance labels for
               Recall@k / nDCG / context precision in evaluate.py).

Usage:
  pip install requests datasets
  python build_knowledge_corpus.py --out corpus.jsonl --email you@uni.edu --pubmed-per-class 300 --medrag
Licensing: PubMed abstracts and StatPearls (CC BY-NC-ND) are for non-commercial research use;
keep source + URL on every chunk so answers can cite them.
"""
import argparse, hashlib, json, re, time, xml.etree.ElementTree as ET

SYNONYMS = {
    "MEL": ["melanoma", "lentigo maligna", "melanoma in situ"],
    "NV": ["nevus", "naevus", "nevi", "naevi", "mole", "melanocytic nevus"],
    "BCC": ["basal cell carcinoma", "basal-cell carcinoma", "bcc", "rodent ulcer"],
    "AK": ["actinic keratosis", "actinic keratoses", "solar keratosis", "bowen"],
    "BKL": ["seborrheic keratosis", "seborrhoeic keratosis", "solar lentigo", "lichen planus-like keratosis",
            "benign keratosis"],
    "DF": ["dermatofibroma", "benign fibrous histiocytoma"],
    "VASC": ["hemangioma", "haemangioma", "angioma", "angiokeratoma", "pyogenic granuloma", "vascular lesion"],
    "SCC": ["squamous cell carcinoma", "cutaneous squamous", "keratoacanthoma", "bowen's disease"],
}
PUBMED_QUERIES = {
    "MEL": '"melanoma"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab] OR diagnosis[sh])',
    "NV": '"nevus, pigmented"[MeSH Terms] AND (dermoscopy[tiab] OR dermatoscopy[tiab])',
    "BCC": '"carcinoma, basal cell"[MeSH Terms] AND (dermoscopy[tiab] OR diagnosis[sh])',
    "AK": '"keratosis, actinic"[MeSH Terms] AND (dermoscopy[tiab] OR diagnosis[sh])',
    "BKL": '("keratosis, seborrheic"[MeSH Terms] OR "solar lentigo"[tiab]) AND (dermoscopy[tiab] OR diagnosis[sh])',
    "DF": '"histiocytoma, benign fibrous"[MeSH Terms] OR dermatofibroma[tiab]',
    "VASC": '("hemangioma"[MeSH Terms] OR angiokeratoma[tiab] OR "pyogenic granuloma"[tiab]) AND skin[tiab] AND dermoscopy[tiab]',
    "SCC": '"carcinoma, squamous cell"[MeSH Terms] AND "skin neoplasms"[MeSH Terms] AND (dermoscopy[tiab] OR diagnosis[sh])',
}
DERM_FILTER = re.compile(r"\b(dermoscop|dermatoscop|skin lesion|cutaneous|melanoma|nev[iu]|naev|keratos|"
                         r"basal cell|squamous cell carcinoma of the skin|dermatofibroma|angioma|lentigo)", re.I)


def tag(text):
    t = text.lower()
    return sorted({c for c, words in SYNONYMS.items() if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words)})


def chunk(text, max_words=220, overlap=40):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap
    return out


def pubmed(cls, query, n, email, api_key):
    import requests
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    p = dict(db="pubmed", term=query + " AND hasabstract[text] AND english[lang]", retmax=n, retmode="json",
             sort="relevance", email=email, tool="derm_rag_corpus")
    if api_key: p["api_key"] = api_key
    ids = requests.get(base + "esearch.fcgi", params=p, timeout=60).json()["esearchresult"]["idlist"]
    for i in range(0, len(ids), 100):
        q = dict(db="pubmed", id=",".join(ids[i:i + 100]), retmode="xml", email=email, tool="derm_rag_corpus")
        if api_key: q["api_key"] = api_key
        root = ET.fromstring(requests.get(base + "efetch.fcgi", params=q, timeout=120).content)
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            abstract = " ".join("".join(x.itertext()) for x in art.findall(".//AbstractText"))
            year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
            yield dict(source="PubMed", doc_id=f"PMID:{pmid}", title=title, text=abstract,
                       url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", year=year[:4], query_class=cls)
        time.sleep(0.4 if not api_key else 0.12)


def medrag(name, limit):
    from datasets import load_dataset
    ds = load_dataset(name, split="train", streaming=True)
    kept = 0
    for r in ds:
        txt = r.get("content") or r.get("contents") or ""
        if DERM_FILTER.search(txt) and tag(txt):
            kept += 1
            yield dict(source=name.split("/")[-1], doc_id=r.get("id", ""), title=r.get("title", ""),
                       text=txt, url="", year="", query_class="")
            if limit and kept >= limit:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--email", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--pubmed-per-class", type=int, default=300)
    ap.add_argument("--medrag", action="store_true", help="also pull dermatology chunks from MedRAG statpearls/textbooks")
    ap.add_argument("--medrag-limit", type=int, default=5000)
    a = ap.parse_args()
    seen, n = set(), 0
    with open(a.out, "w") as f:
        docs = []
        for c, q in PUBMED_QUERIES.items():
            docs.extend(pubmed(c, q, a.pubmed_per_class, a.email, a.api_key))
            print(c, "pubmed docs so far:", len(docs))
        if a.medrag:
            for name in ("MedRAG/statpearls", "MedRAG/textbooks"):
                docs.extend(medrag(name, a.medrag_limit))
        for d in docs:
            for k, piece in enumerate(chunk((d["title"] + ". " + d["text"]).strip())):
                h = hashlib.md5(piece.lower().encode()).hexdigest()
                if h in seen or len(piece.split()) < 25:
                    continue                      # de-duplicate and drop fragments
                seen.add(h)
                f.write(json.dumps(dict(chunk_id=f"{d['doc_id']}#{k}", source=d["source"], doc_id=d["doc_id"],
                                        title=d["title"], text=piece, url=d["url"], year=d["year"],
                                        tags=tag(piece))) + "\n")
                n += 1
    print("chunks written:", n, "->", a.out)


if __name__ == "__main__":
    main()
