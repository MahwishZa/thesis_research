"""Step 2 of the knowledge corpus: reading the downloaded formats and cleaning their text; the licence, retraction and
topic rules."""
import html
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

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
