"""Preparation of the text dataset, one step per command, run in this order.

    python -m src.text.prepare survey
    python -m src.text.prepare cases
    python -m src.text.prepare abstracts
    python -m src.text.prepare mesh
    python -m src.text.prepare build

Then two commands of their own check the dataset and build the reference library for retrieval: python -m src.text.audit and
python -m src.text.library.

Every step writes into a few committed files in data/text/ and rewrites only its own part of them:
    term_check.csv       every name in class_terms.csv, checked against MeSH and counted in PubMed
    case_candidates.csv  the patients whose article title names a class: candidates for the cases, with marks
    abstract_candidates.csv  the same for PubMed's case-report abstracts of articles not in PMC-Patients
    article_mesh.csv     for each candidate article: whether PubMed has indexed it, and the classes its headings name, on the skin
    dataset.csv          the text dataset: one row per case (split, class, question file, fingerprint, source numbers)
    dataset_report.md    one section per step: its counts, its tables and what it does not do
The question texts themselves are written into data/text/dataset/ (not committed: their licences allow research use only).
One committed file is written by people, not by code: class_terms.csv (the names doctors use for each of the image dataset's
classes, and whether each name counts for that class). Its classes must be exactly the image dataset's classes.

``survey`` checks every name in class_terms.csv against MeSH, the US National Library of Medicine's list of medical subject
headings: a name is confirmed when it is a heading or one of the heading's entry terms (other names MeSH lists for it). It also
counts the PubMed articles that use each name in their title or summary, which shows whether a name MeSH lacks is in use. Then it
counts in PubMed, for each class, the case reports and the reviews and guidelines about the class's skin lesions, using the
headings of the names that count for the class, to plan the dataset's sizes. It downloads only MeSH records and search counts
(no article text) from NCBI's E-utilities, at a pace NCBI allows without a key, and changes nothing else. PubMed grows every day,
so the counts are dated.

``cases`` downloads PMC-Patients (Zhao et al., Scientific Data 2023; licence CC BY-NC-SA 4.0), real patient descriptions from
published case reports in PubMed Central, once into data/text/sources/ (not committed): the version current at the first attempt,
continued where it stopped, and checked against its fingerprint on every later run. It reads the patients one at a time and lists
every patient whose article title names one of the classes, by the names that count in class_terms.csv (a longer name wins over a
shorter one inside it, and word order inside a name does not matter), with marks where a person should look: a title naming two
classes, a name that does not count, a word for a site other than the skin, a title about treatment or about an animal, a class
named only as what the lesion looked like ("... mimicking basal cell carcinoma"), a nevus named only by the bare word (it also
names birthmarks that are not moles), or a case text that names the class neither by a
full name nor by its short form (such as SCC). It also lists each case's publication year and how often its
text names the class. No case text is written into the repository, and nothing is chosen or cut.

``abstracts`` adds the case reports that exist in PubMed only as an abstract: it asks PubMed for every class's case reports
(the class's headings as a main subject, on the skin, in English, as in the survey, and, by the class's names in the title, the
case reports PubMed has not indexed with subject headings), leaves out the articles PMC-Patients already gave, downloads the title and abstract of the others and applies the rules of ``cases`` to them, using the abstract's case part
(its sections labelled as the case or the patient, or else the whole abstract). For a candidate article (its title passes the
rules) with an open full text in PubMed Central, of any year, the full text's case sections are used instead of the abstract.
The texts stay in data/text/sources/.

``mesh`` asks PubMed, for every candidate article, whether it has subject headings (MeSH) at all (PubMed indexes the
articles of MEDLINE journals; many case-report journals are not among them) and, if so, which classes its headings name (one of
the class's headings in term_check.csv or a narrower one) while it is also indexed under the skin. These headings are given by
the US National Library of Medicine, independently of the authors' titles and of this code. It sends only article numbers to
NCBI's E-utilities.

``build`` makes the dataset from both lists of candidates, with no person choosing, labelling or cutting a case. A case gets
class X when its article title names X and no other class, with no mark (``cases``, ``abstracts``), and its case text names X, by a full name or its short form; and, where
PubMed has indexed the article, its headings must name X on the skin (``mesh``), else the case is left out. Its question is the
case text word for word, up to the first sentence that names its own class (any name of the class in class_terms.csv, counting
or not), its short form or one the article defines for it (such as "pyogenic granuloma (PG)"), or reports a test result (a
biopsy, histology, pathology, immunohistochemistry, an excision, or what a microscope showed); it starts at the start of the case
text, or at its first sentence about the patient when an earlier sentence (a background) already names the class. Names of the
other classes stay, as clinical information. A question is kept when it has at least 25 words, describes a lesion, what it
looks like (a size, a lesion type such as a nodule, a colour, a surface or a shape) and a patient, and puts no lesion inside the body (such as the mouth); of two articles telling the same patient's case (their
questions share at least 10% of their runs of 12 words), the one PubMed listed first keeps it.
Cases are split whole by article in a fixed random order: for each class, 60% of its cases into test (at most 40), 20% into
validation (at most 15) and the rest into train, so that every class with a few cases is in all three splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from ..common import USER_AGENT, fetch, write_section
from ..evaluate import FILES, TABLE_NAME
from ..evaluate import classes as image_classes
from ..evaluate import read_table

DATA = Path("data/text")
IMAGE_TABLE = Path("data/image/dataset.csv")  # the classes of the text dataset are the image dataset's
TERMS = "class_terms.csv"  # written by people
CHECK = "term_check.csv"
REPORT = "dataset_report.md"
STEPS = ("survey", "cases", "abstracts", "mesh", "build", "audit", "library")  # the report's sections, in the order they run;
# audit and library are commands of their own (python -m src.text.audit, python -m src.text.library)
REPORT_HEAD = ["# Preparation of the text dataset", "",
               "Written by `python -m src.text.prepare`, one section per step in the order the steps run; each step rewrites only "
               "its own section. `class_terms.csv` is written by people: the names doctors use for each of the image dataset's "
               "classes, and whether each name counts for that class.", ""]
TERM_COLUMNS = ["class", "term", "counts", "source", "note"]
CHECK_COLUMNS = ["class", "term", "counts", "in_mesh", "mesh_ids", "mesh_headings", "pubmed_articles"]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NCBI_DELAY = 0.4  # seconds between requests: NCBI allows 3 requests a second without an API key
NCBI_ATTEMPTS = 4
SKIN = '("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms])'  # the lesion is on the skin
COUNTS = {  # what is counted per class; each query is the class's headings AND this
    "case reports": f'{SKIN} AND "case reports"[Publication Type] AND english[Language]',
    "case reports, free full text": f'{SKIN} AND "case reports"[Publication Type] AND english[Language] AND free full text[sb]',
    "case reports, free full text, 2025 or later": f'{SKIN} AND "case reports"[Publication Type] AND english[Language] '
                                                   'AND free full text[sb] AND 2025:3000[Date - Publication]',
    "reviews and guidelines, free full text": f'{SKIN} AND (review[Publication Type] OR guideline[Publication Type] OR '
                                              '"practice guideline"[Publication Type]) AND english[Language] AND free full text[sb]'}
IRREGULAR = {"nevi": "nevus", "lentigines": "lentigo"}  # plurals the simple rules miss (after the spelling is made American)
SPELLING = (("naev", "nev"), ("haem", "hem"), ("oedem", "edem"), ("seborrhoe", "seborrhe"), ("tumour", "tumor"),
            ("oesophag", "esophag"))
SOURCES = "sources"  # data/text/sources/: downloads kept exactly as received (not committed)
CANDIDATES = "case_candidates.csv"
CANDIDATE_COLUMNS = ["patient_uid", "pmid", "year", "class", "classes_named", "marks", "title", "age", "sex", "words", "text_full_names",
                     "text_short_forms"]
PMC_PATIENTS = {"repo": "zhengyun21/PMC-Patients", "file": "PMC-Patients-V2.json", "licence": "CC BY-NC-SA 4.0"}
PATIENT_FIELDS = ("patient_uid", "PMID", "title", "patient")  # what this step needs from each patient
HF_API = "https://huggingface.co/api/datasets/{repo}"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/{commit}/{file}"
DOWNLOAD_ATTEMPTS = 6
REPORT_EVERY = 50_000_000  # bytes between progress lines while downloading
CHUNK = 1 << 22  # characters read at a time from a large JSON file
NOT_SKIN = (  # words in a title that point to a site other than the skin: such a case is marked for a person, not dropped
    "oral", "mouth", "tongue", "gingiva", "gingival", "palate", "palatal", "buccal", "mucosa", "mucosal", "lip", "larynx",
    "laryngeal", "pharynx", "pharyngeal", "oropharyngeal", "nasopharyngeal", "sinonasal", "esophagus", "esophageal", "lung",
    "pulmonary", "bronchial", "cervix", "uterine", "vulva", "vulvar", "vagina", "vaginal", "anal", "anus", "anorectal", "rectal",
    "rectum", "penile", "penis", "urethra", "urethral", "bladder", "conjunctiva", "conjunctival", "uveal", "choroid", "choroidal",
    "ocular", "orbit", "orbital", "iris", "gastric", "stomach", "colon", "colonic", "intestinal", "thyroid", "breast", "bone",
    "brain", "cerebral", "spinal", "liver", "hepatic", "pancreatic", "kidney", "renal", "ovary", "ovarian", "testicular",
    "prostate", "parotid", "salivary", "tonsil", "tonsillar", "metastasis", "metastases", "metastatic",
    "head and neck", "nasal cavity", "lymph node", "eye", "intraocular", "retina", "retinal", "ciliary", "lacrimal", "intraoral",
    "intraosseous", "odontogenic", "mandible", "mandibular", "maxilla", "maxillary", "submandibular", "intracranial",
    "leptomeningeal", "intramedullary", "spine", "heart", "cardiac", "thoracic", "pleural", "tracheal", "endobronchial", "thymus",
    "thymic", "gastrointestinal", "bowel", "intestine", "intussusception", "duodenum", "duodenal", "sigmoid", "gallbladder",
    "biliary", "pancreas", "spleen", "splenic", "adrenal", "peritoneal", "pelvic", "urinary", "prostatic", "uterus", "endometrium",
    "endometrial", "trachea", "mediastinum", "mediastinal", "ileum", "ileal", "jejunum", "jejunal", "cecum", "caecum", "cecal",
    "appendix", "appendiceal", "colorectal", "sphenoid", "ethmoid", "paranasal", "intranasal", "nasopharynx", "oropharynx",
    "hypopharynx", "hypopharyngeal", "cerebellopontine", "pituitary", "meninges", "meningeal", "intradural", "dural",
    "central nervous system", "cornea", "corneal", "uvea", "esophagogastric", "gastroesophageal", "bronchus", "bronchogenic",
    "nodal", "perianal", "intrauterine", "paratracheal", "intravenous", "intravascular")  # a site of more than one word is matched as a phrase ("head and neck"; "neck" alone is skin too)
ANIMALS = (  # words in a title that show the case is an animal's (veterinary reports): such a case is marked, not dropped
    "dog", "canine", "puppy", "cat", "feline", "kitten", "horse", "equine", "mare", "gelding", "stallion", "pony", "foal", "cattle",
    "cow", "bovine", "heifer", "pig", "porcine", "swine", "sheep", "ovine", "goat", "caprine", "rabbit", "ferret", "rat", "mouse",
    "mice", "murine", "hamster", "guinea pig", "rodent", "paca", "monkey", "macaque", "primate", "dolphin", "whale", "camel",
    "alpaca", "llama", "deer", "parrot", "budgerigar", "avian", "chicken", "snake", "reptile", "lizard", "tortoise", "veterinary")
LOOKS_LIKE = (  # words in a title after which a class is not the diagnosis: what the lesion looked like ("sebaceoma mimicking
    # basal cell carcinoma") or a second finding beside it ("sebaceous carcinoma associated with seborrheic keratosis"); a class
    # named before them is the diagnosis ("melanoma mimicking a cyst", "basal cell carcinoma arising in rhinophyma")
    "mimic", "mimicking", "mimicker", "mimicry", "masquerade", "masquerading", "resemble", "resembling", "simulate", "simulating",
    "imitate", "imitating", "disguised", "presenting as", "presented as", "misdiagnosed as", "mistaken for", "look-alike",
    "lookalike", "associated with", "association with", "coexistent with", "coexisting with", "arising in", "arising from",
    "arising within", "arising on", "differentiating", "distinguishing")
TREATMENT = (  # words in a title that show it is about (mostly systemic) treatment: such a case is marked, not dropped
    "inhibitor", "inhibition", "blockade", "immunotherapy", "chemotherapy", "radiotherapy", "chemoradiotherapy", "chemoradiation",
    "brachytherapy", "radiosurgery", "adjuvant", "neoadjuvant", "toxicity", "adverse", "nivolumab", "ipilimumab", "pembrolizumab",
    "cemiplimab", "cetuximab", "vemurafenib", "dabrafenib", "trametinib", "vismodegib", "sonidegib", "imatinib", "paclitaxel",
    "carboplatin", "cisplatin", "talimogene")
SHORT_FORMS = {  # common short forms of a class in case texts; counted in the case text only, never used to find candidates
    "actinic keratosis": ("AK",), "basal cell carcinoma": ("BCC",), "pigmented benign keratosis": ("SK",),
    "squamous cell carcinoma": ("SCC", "cSCC")}
ABSTRACT_CANDIDATES = "abstract_candidates.csv"
ABSTRACTS = "pubmed"  # data/text/sources/pubmed/abstracts.jsonl: the downloaded titles and abstracts (not committed)
HUMANS = "NOT (animals[MeSH Terms] NOT humans[MeSH Terms])"  # PubMed's standard filter: leaves out articles about animals only
ABSTRACT_QUERY = f'{SKIN} AND "case reports"[Publication Type] AND english[Language] {HUMANS}'  # with each class's headings
UNINDEXED_QUERY = ('("case reports"[Publication Type] OR "case report"[Title/Abstract]) NOT medline[sb] AND english[Language]')
# with each class's names in the title: the case reports PubMed has not indexed (journals outside MEDLINE have no subject
# headings, so the search by headings cannot find them)
ESEARCH_MAX = 9999  # PubMed lists at most this many articles per search; a larger result is asked for in parts by year
EFETCH_BATCH = 200  # abstracts downloaded at a time
CASE_PART = re.compile(r"case|patient|observation|presentation|report", re.I)  # labels of an abstract's case sections
FULLTEXT_BATCH = 20  # full texts downloaded at a time (they are large)
FULLTEXT_CASE = re.compile(r"\bcase\b|presentation|observation", re.I)  # titles of a full text's case sections
FULLTEXT_SKIP = ("fig", "table-wrap", "fn-group", "caption", "disp-formula")  # parts of a paragraph that are not running text
MESH_TABLE = "article_mesh.csv"
MESH_COLUMNS = ["pmid", "indexed", "classes"]
INDEXED = "medline[sb]"  # PubMed's articles with subject headings (MeSH): those of MEDLINE journals
EPOST_BATCH = 500  # article numbers sent to PubMed at a time
DATASET_COLUMNS = ["split", "label", "path", "sha256", "patient_uid", "pmid", "source", "pubmed", "year", "age", "sex", "words"]
RESULT_CUES = ("biops", "histopatholog", "histolog", "patholog", "dermatopatholog", "immunohistochem", "histochem", "immunostain",
               "immunoreactiv", "excis", "microscop", "cytolog", "cytoplasm", "nucle", "mitos", "mitot", "specimen", "multinucleat",
               "fibroblastic", "histiocytic", "aspirat", "dermis", "epidermis", "collagen", "macrophag", "storiform", "spindle",
               "epithelioid", "keratinocyt", "lymphocyt", "atypia")  # word
# beginnings: a test result, also when it is told without naming the test ("cells with eccentric nuclei and eosinophilic cytoplasm")
DESCRIPTIVE = ("mole",)  # names in class_terms.csv too vague to be a diagnosis: part of a description, they end no question
LESION_WORDS = ("lesion", "nodule", "papule", "plaque", "patch", "patches", "macule", "mass", "masses", "tumour", "ulcer", "growth",
                "spot")  # a question must describe a lesion with one of these words (other plurals are found by the word rules)
ABBREVIATIONS = ("e.g.", "i.e.", "dr.", "mr.", "mrs.", "ms.", "vs.", "approx.", "fig.", "figs.", "no.", "al.", "st.", "ca.")
MIN_WORDS = 25  # the shortest question kept
PATIENT = re.compile(r"\b\d{1,3}[- ](?:year|yr|month|week|day)s?[- ]old\b|\baged? \d{1,3}\b|\b\d{1,3} years? of age\b|"
                     r"\b(?:man|woman|boy|girl|male|female|infant|newborn|neonate|child|gentleman|lady)\b", re.I)  # a patient: an
# age or a sex ("this patient's tumor" alone is no patient described)
INSIDE = (  # sites inside the body: a question with a sentence that puts a lesion at one of them is left out (not the skin)
    "intraoral", "gingiva", "gingival", "gingivae", "palate", "palatal", "retromolar", "alveolar ridge", "alveolar mucosa",
    "oral cavity", "oral mucosa", "buccal mucosa", "labial mucosa", "floor of the mouth", "tongue", "nasal cavity", "nasal mucosa",
    "conjunctiva", "conjunctival", "vulva", "vulvar", "anogenital", "anal canal", "choroid", "choroidal", "uvea", "uveal", "iris",
    "retina", "retinal", "intraocular", "mediastinum", "mediastinal", "cervix", "middle ear", "eustachian tube", "tympanic",
    "nasopharynx")
FEATURES = (  # what a lesion looks like: a question tells at least one (a colour, a surface or a shape), or a size in mm or cm
    "red", "reddish", "pink", "pinkish", "brown", "brownish", "black", "blackish", "blue", "bluish", "purple", "purplish", "violet",
    "violaceous", "white", "whitish", "yellow", "yellowish", "grey", "gray", "greyish", "grayish", "tan", "dark", "erythematous",
    "pigmented", "hyperpigmented", "hypopigmented", "depigmented", "colored", "coloured", "pearly", "translucent", "shiny", "scaly",
    "crusted", "crusting", "ulcerated", "ulcerating", "eroded", "verrucous", "warty", "keratotic", "hyperkeratotic",
    "papillomatous", "pedunculated", "exophytic", "nodular", "polypoid", "polypoidal", "fungating", "raised", "elevated", "flat",
    "depressed", "atrophic", "firm", "hard", "soft", "smooth", "rough", "friable", "bleeding", "hemorrhagic", "lobulated",
    "annular", "irregular", "circumscribed", "demarcated", "indurated", "cystic", "compressible")
SIZE = re.compile(r"\d\s*(?:mm|cm)\b", re.I)
LESION_TYPES = ("nodule", "papule", "plaque", "patch", "patches", "macule", "ulcer", "spot")  # lesion words that tell what a
# lesion looks like by themselves ("a slowly growing nodule"), unlike lesion, mass, tumour and growth
SAME_PATIENT, SHARED_RUN = 0.10, 12  # questions of two articles sharing this share of their runs of 12 words: one patient
SHORT_FORM = r"(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])[A-Za-z]{2,6}"  # a short form an author defines: 2 to 6 letters, 2 or more capitals
TEST_SHARE, VAL_SHARE = 0.6, 0.2  # of each class's cases: into test and into validation (the rest into train)
TEST_PER_CLASS, VAL_PER_CLASS = 40, 15  # at most, per class, as in the image dataset
TEXT_MARK = "the case text does not name the class"  # the one mark that depends on the text, not on the title
LOOKS_MARK = "the title names the class only as what the lesion looked like, or beside another diagnosis"
BARE = ("nevus",)  # class names that also name other things: alone in a title they give no class (a nevus can be a birthmark
# that is not a mole, such as a mucinous or collagen nevus); the title must name a kind of mole (melanocytic, congenital, Spitz ...)
BARE_MARK = "the title names the class only by its bare name, which also names other things"
SPLIT_SEED = "text-dataset"  # fixes the random order of the articles, so the splits are the same on every run


# ---------------------------------------------------------------------------------------------------------------------------------
# class_terms.csv: written by people, checked before anything is downloaded


def read_terms(path: Path, classes: list[str]) -> list[dict]:
    """The names in class_terms.csv, after checking that they can be used without guessing."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: the list of names is part of the repository")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != TERM_COLUMNS:
            raise ValueError(f"{path.as_posix()} must have the columns {', '.join(TERM_COLUMNS)} (it has {reader.fieldnames})")
        rows = [{c: (row[c] or "").strip() for c in TERM_COLUMNS} for row in reader]
    seen, counted = set(), {}
    for line, row in enumerate(rows, start=2):
        where = f"{path.as_posix()} line {line}"
        if row["class"] not in classes:
            raise ValueError(f"{where}: {row['class']!r} is not a class of the image dataset ({', '.join(classes)})")
        if not row["term"] or row["counts"] not in ("yes", "no") or not row["source"]:
            raise ValueError(f"{where}: every row needs a term, counts (yes or no) and a source")
        key = (row["class"], _words(row["term"]))
        if key in seen:
            raise ValueError(f"{where}: {row['term']!r} is listed twice for {row['class']}")
        seen.add(key)
        if row["counts"] == "yes":
            other = counted.setdefault(_words(row["term"]), row["class"])
            if other != row["class"]:
                raise ValueError(f"{where}: {row['term']!r} counts for both {other} and {row['class']}: a name can belong to one "
                                 "class only")
    missing = [c for c in classes if not any(r["class"] == c and r["counts"] == "yes" and _words(r["term"]) == _words(c) for r in rows)]
    if missing:
        raise ValueError(f"{path.as_posix()} lacks the class name itself, counting yes, for: {', '.join(missing)}")
    return rows


def _words(text: str) -> tuple[str, ...]:
    """A name as a sorted tuple of plain words, so that spelling, word order, plurals and punctuation do not matter:
    "Keratoses, Actinic" and "actinic keratosis" give the same words, and so do "naevus" and "nevi"."""
    return tuple(sorted(_tokens(text)))


def _singular(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word.endswith("oses") and len(word) > 6:  # keratoses
        return word[:-4] + "osis"
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "is", "us")) and len(word) > 3:
        return word[:-1]
    return word


# ---------------------------------------------------------------------------------------------------------------------------------
# NCBI's E-utilities: MeSH records and PubMed counts


def _eutils(tool: str, **params) -> dict:
    url = f"{EUTILS}{tool}.fcgi?" + urllib.parse.urlencode({**params, "retmode": "json"})
    return fetch(url, "application/json", lambda content: json.loads(content.decode("utf-8")), delay=NCBI_DELAY,
                 attempts=NCBI_ATTEMPTS)


def _search(db: str, term: str, retmax: int = 0, **extra) -> dict:
    """The answer of an E-utilities search; stops when NCBI says part of the query was not understood (a count would be wrong)."""
    answer = _eutils("esearch", db=db, term=term, retmax=retmax, **extra).get("esearchresult")
    if not isinstance(answer, dict) or "count" not in answer:
        raise ValueError(f"NCBI's search answer has an unexpected form for {db}: {term}")
    errors = {k: v for k, v in (answer.get("errorlist") or {}).items() if v}
    unknown = (answer.get("warninglist") or {}).get("quotedphrasesnotfound") or []
    if db != "mesh" and (errors or unknown):
        raise ValueError(f"PubMed did not understand part of the query {term!r}: {errors or unknown}. Nothing was written")
    return answer


def _mesh_records(term: str) -> list[dict]:
    """The MeSH records found for a name: their ID (D... or C...) and every name MeSH gives them, the heading first."""
    ids = _search("mesh", f'"{term}"', retmax=20).get("idlist") or []
    if not ids:
        return []
    result = _eutils("esummary", db="mesh", id=",".join(ids)).get("result") or {}
    records = []
    for uid in result.get("uids") or []:
        entry = result.get(uid) or {}
        names, mesh_id = entry.get("ds_meshterms"), entry.get("ds_meshui")
        if not isinstance(names, list) or not names or not mesh_id:
            raise ValueError(f"NCBI's MeSH summary of {uid} has an unexpected form (fields: {', '.join(sorted(entry)) or 'none'})")
        records.append({"id": mesh_id, "names": names})
    return records


def _articles(term: str) -> int:
    """How many PubMed articles use a name in their title or summary (abstract); 0 when PubMed has never seen the phrase."""
    answer = _eutils("esearch", db="pubmed", term=f'"{term}"[Title/Abstract]', retmax=0).get("esearchresult")
    if not isinstance(answer, dict) or "count" not in answer:
        raise ValueError(f"NCBI's search answer has an unexpected form for the name {term!r}")
    if (answer.get("errorlist") or {}).get("fieldsnotfound"):
        raise ValueError(f"PubMed did not understand the search for the name {term!r}: {answer['errorlist']}. Nothing was written")
    return int(answer["count"])


def _class_headings(rows) -> dict[str, list[str]]:
    """Per class: the MeSH headings of its names that count (rows of term_check.csv), each once, in the order of the rows."""
    headings = defaultdict(list)
    for row in rows:
        if row["counts"] == "yes":
            for heading in filter(None, row["mesh_headings"].split("; ")):
                if heading not in headings[row["class"]]:
                    headings[row["class"]].append(heading)
    return headings


def _subject(headings: list[str], field: str = "MeSH Major Topic") -> str:
    """A PubMed query for articles indexed under one of the headings or a narrower one: as their main subject (MeSH Major
    Topic, the default), or as any of their headings (MeSH Terms)."""
    return "(" + " OR ".join(f'"{h}"[{field}]' for h in headings) + ")"


def _read_csv(path: Path, columns: list[str], step: str) -> list[dict]:
    """The rows of a table an earlier step wrote, after checking its columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: run python -m src.text.prepare {step} first")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.as_posix()} must have the columns {', '.join(columns)} (it has {reader.fieldnames}): run "
                             f"python -m src.text.prepare {step} again")
        return list(reader)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check(term: str, records: list[dict]) -> dict:
    """Whether a name is a MeSH heading, an entry term of one, or not in MeSH."""
    mine = _words(term)
    heading = [r for r in records if _words(r["names"][0]) == mine]
    entry = [r for r in records if r not in heading and any(_words(name) == mine for name in r["names"][1:])]
    found = heading or entry
    return {"in_mesh": "heading" if heading else "entry term" if entry else "no",
            "mesh_ids": "; ".join(r["id"] for r in found), "mesh_headings": "; ".join(r["names"][0] for r in found)}


# ---------------------------------------------------------------------------------------------------------------------------------
# The survey


def survey(data: str | Path = DATA, classes: list[str] | None = None, show=None) -> dict:
    """Checks every name in class_terms.csv against MeSH and counts what PubMed holds per class; writes term_check.csv and the
    survey section of dataset_report.md."""
    data = Path(data)
    classes = classes if classes is not None else image_classes(read_table(IMAGE_TABLE))
    terms = read_terms(data / TERMS, classes)
    rows = []
    for n, row in enumerate(terms, start=1):
        rows.append({"class": row["class"], "term": row["term"], "counts": row["counts"], **_check(row["term"], _mesh_records(row["term"])),
                     "pubmed_articles": _articles(row["term"])})
        if show and (n % 20 == 0 or n == len(terms)):
            show(f"{n} of {len(terms)} names checked in MeSH and PubMed")
    headings = _class_headings(rows)
    counts = {}
    for label in classes:
        if not headings[label]:
            counts[label] = {}
            continue
        counts[label] = {what: int(_search("pubmed", f"{_subject(headings[label])} AND {extra}")["count"])
                         for what, extra in COUNTS.items()}
        if show:
            show(f"{label}: " + "; ".join(f"{what} {n}" for what, n in counts[label].items()))
    _write_csv(data / CHECK, CHECK_COLUMNS, rows)
    day = datetime.now(timezone.utc).date().isoformat()
    sources = {(t["class"], t["term"]): t["source"] for t in terms}
    write_section(data / REPORT, REPORT_HEAD, STEPS, "survey", _survey_summary(rows, sources, classes, headings, counts, day))
    return {"rows": rows, "counts": counts, "headings": headings, "table": (data / CHECK).as_posix(),
            "summary": (data / REPORT).as_posix()}


def _survey_summary(rows, sources, classes, headings, counts, day) -> list[str]:
    found = {k: sum(r["in_mesh"] == k for r in rows) for k in ("heading", "entry term", "no")}
    absent = [r for r in rows if r["in_mesh"] == "no"]
    unused = [r for r in absent if not r["pubmed_articles"]]
    lines = ["# Survey: the names checked in MeSH, and how much PubMed holds per class", "",
             f"Written by `python -m src.text.prepare survey` on {day}. Every name in `class_terms.csv` was looked up in MeSH, the US "
             "National Library of Medicine's list of medical subject headings, through NCBI's E-utilities; the result for each name "
             f"is in `{CHECK}`, with the number of PubMed articles that use the name in their title or summary. Then PubMed was searched per class with the MeSH headings of the class's confirmed names (as main "
             "topics), limited to skin lesions. Only MeSH records and search counts were downloaded, no article text; nothing "
             "else was changed.", "",
             "## MeSH check", "",
             f"- Names: {len(rows)}; a MeSH heading: {found['heading']}; an entry term of a heading (another name MeSH lists for "
             f"it): {found['entry term']}; not in MeSH: {found['no']} (used in PubMed titles or summaries: "
             f"{len(absent) - len(unused)}; in none: {len(unused)})", "",
             "A name that is not in MeSH is not wrong: MeSH lists the names used to index articles, not every name in use (subtypes "
             "and older names are often missing). The PubMed count shows whether such a name is in use. A name in neither is for a "
             "person to confirm, change or remove in `class_terms.csv`.", "",
             "| Class | Name not in MeSH | Counts for the class | PubMed articles using it | Source in class_terms.csv |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['class']} | {r['term']} | {r['counts']} | {r['pubmed_articles']} | {sources[(r['class'], r['term'])]} |"
              for r in absent] or ["| – | – | – | – | – |"]
    lines += ["", f"## How much PubMed holds per class ({day})", "",
              "Estimates for planning, not the dataset: a MeSH heading can be wider than the class (for example a heading for all "
              "lentigines), and no case report has been read yet. \"Free full text\" means the full text can be read free of "
              "charge, which is wider than a licence that allows reuse. The cases themselves come later from PMC-Patients, which "
              "holds only articles whose licence allows reuse.", "",
              "| Class | " + " | ".join(n[:1].upper() + n[1:] for n in COUNTS) + " |", "|---|" + "---|" * len(COUNTS)]
    for label in classes:
        lines.append(f"| {label} | " + " | ".join(str(counts[label].get(what, "–")) for what in COUNTS) + " |")
    lines += ["", "## The MeSH headings used for each class", "", "| Class | Headings (as main topics) |", "|---|---|"]
    lines += [f"| {label} | {'; '.join(headings[label]) or 'none: no name of this class is in MeSH'} |" for label in classes]
    lines += ["", "## The queries", "", "Each count is the class's headings AND one of these conditions:", ""]
    lines += [f"- {what}: `{extra}`" for what, extra in COUNTS.items()]
    lines += ["", "## Not done in this step", "", "- No article, abstract or case text is downloaded, and nothing is chosen for the dataset.",
              "- The names in `class_terms.csv` are not changed: a person decides what to do with the names MeSH does not have.", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# Cases: candidate patient cases from PMC-Patients


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stream(url: str, part: Path, show=None) -> None:
    """Downloads ``url`` into ``part``, continuing a stopped download where it stopped (an HTTP range request); a busy service
    or a network failure is tried again with growing waits."""
    problem = ""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if have and response.status != 206:  # the server sent the whole file again: start over
                    have = 0
                total = have + int(response.headers.get("Content-Length") or 0)
                shown = have // REPORT_EVERY
                with part.open("ab" if have else "wb") as f:
                    while block := response.read(1 << 20):
                        f.write(block)
                        have += len(block)
                        if show and have // REPORT_EVERY > shown:
                            shown = have // REPORT_EVERY
                            show(f"downloaded {have // 1_000_000} MB" + (f" of {total // 1_000_000} MB" if total else ""))
            if total and have < total:
                raise OSError(f"the connection closed after {have} of {total} bytes")
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # nothing left to send: the file is complete
                return
            if exc.code != 429 and exc.code < 500:
                raise ValueError(f"{url} was refused (HTTP {exc.code})") from exc
            problem = f"HTTP {exc.code}"
        except (OSError, http.client.HTTPException) as exc:
            problem = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
        time.sleep(2 ** attempt)
    raise ValueError(f"cannot download {url} ({problem}). What was downloaded is kept: run the same command again to continue")


def _checked_source(folder: Path) -> dict | None:
    """The record of PMC-Patients' downloaded file in ``folder`` after checking the file against its fingerprint, or None when
    the file has not been downloaded completely."""
    target, record = folder / PMC_PATIENTS["file"], folder / "download.json"
    if not (target.is_file() and record.is_file()):
        return None
    meta = json.loads(record.read_text(encoding="utf-8"))
    if _sha256(target) != meta["sha256"]:
        raise ValueError(f"{target.as_posix()} has changed since it was downloaded (its fingerprint differs): delete it and "
                         f"{record.name} to download it again")
    return meta


def _download_pmc_patients(folder: Path, show=None) -> dict:
    """PMC-Patients' file, downloaded once into ``folder`` from the version of the Hugging Face repository current at the first
    attempt (so a stopped download continues from the same version), and checked against its fingerprint every time after."""
    meta = _checked_source(folder)
    if meta:
        return meta
    target, record = folder / PMC_PATIENTS["file"], folder / "download.json"
    folder.mkdir(parents=True, exist_ok=True)
    part, pending = target.with_name(target.name + ".part"), folder / "download.part.json"
    if pending.is_file():
        commit = json.loads(pending.read_text(encoding="utf-8"))["commit"]
    else:
        info = fetch(HF_API.format(repo=PMC_PATIENTS["repo"]), "application/json", lambda c: json.loads(c.decode("utf-8")))
        commit = info.get("sha") if isinstance(info, dict) else None
        if not commit:
            raise ValueError(f"Hugging Face's description of {PMC_PATIENTS['repo']} names no version (fields: "
                             f"{', '.join(sorted(info)) if isinstance(info, dict) else type(info).__name__})")
        pending.write_text(json.dumps({"commit": commit}), encoding="utf-8")
    url = HF_FILE.format(repo=PMC_PATIENTS["repo"], commit=commit, file=PMC_PATIENTS["file"])
    if show:
        show(f"downloading {PMC_PATIENTS['file']} (version {commit[:12]}) into {folder.as_posix()}/")
    _stream(url, part, show)
    meta = {"source": "PMC-Patients", "repository": PMC_PATIENTS["repo"], "commit": commit, "file": PMC_PATIENTS["file"],
            "url": url, "bytes": part.stat().st_size, "sha256": _sha256(part), "licence": PMC_PATIENTS["licence"],
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    os.replace(part, target)
    record.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    pending.unlink()
    return meta


def _json_list(path: Path, chunk: int | None = None):
    """The items of a file holding one JSON list, one at a time, so that a large file is never held in memory at once."""
    chunk = chunk or CHUNK
    decoder, buffer, pos, started = json.JSONDecoder(), "", 0, False
    with path.open(encoding="utf-8") as f:
        while True:
            while True:  # skip spaces and the commas between items, reading more when the buffer runs out
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buffer):
                    break
                more = f.read(chunk)
                if not more:
                    raise ValueError(f"{path.as_posix()} ends before its list does: the file is cut off")
                buffer, pos = buffer[pos:] + more, 0
            if not started:
                if buffer[pos] != "[":
                    raise ValueError(f"{path.as_posix()} is not one JSON list")
                started, pos = True, pos + 1
                continue
            if buffer[pos] == "]":
                return
            try:
                item, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise ValueError(f"{path.as_posix()} ends in the middle of an item: the file is cut off") from None
                buffer, pos = buffer[pos:] + more, 0
                continue
            yield item
            pos = end


def _tokens(text: str) -> list[str]:
    """A text as plain words in order: lower case, American spelling, singular."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    for british, american in SPELLING:
        text = text.replace(british, american)
    return [_singular(word) for word in re.findall(r"[a-z0-9]+", text)]


def _matcher(terms: list[dict]) -> list[tuple[int, dict]]:
    """The names grouped by their number of words, longest first, each as its sorted words (as read_terms compares them, so word
    order inside a name does not matter); one name can be a row of several classes (it counts for one at most)."""
    names = defaultdict(lambda: defaultdict(list))
    for t in terms:
        words = _words(t["term"])
        names[len(words)][words].append(t)
    return sorted(((n, dict(rows)) for n, rows in names.items()), key=lambda item: -item[0])


def _named(words: list[str], matcher) -> tuple[Counter, list[str]]:
    """How often a text names each class, and the names it holds that count for no class. A longer name takes its words from a
    shorter one inside it: "basal cell nevus syndrome" is not also a "nevus", and "lentigo maligna melanoma" is one melanoma.
    The words of a name may come in any order: "sebaceous nevus" is the name "nevus sebaceous"."""
    taken, classes, others = [False] * len(words), Counter(), set()
    for n, names in matcher:
        for i in range(len(words) - n + 1):
            rows = names.get(tuple(sorted(words[i:i + n])))
            if rows and not any(taken[i:i + n]):
                taken[i:i + n] = [True] * n
                counted = [r["class"] for r in rows if r["counts"] == "yes"]
                classes.update(counted)
                if not counted:
                    others.add(rows[0]["term"])
    return classes, sorted(others)


def _found(words: list[str], phrases: dict[tuple[str, ...], str]) -> list[str]:
    """The phrases (as written) whose words stand together, in order, in ``words``."""
    return sorted({phrase for sequence, phrase in phrases.items()
                   if any(tuple(words[i:i + len(sequence)]) == sequence for i in range(len(words) - len(sequence) + 1))})


def _year(value) -> str:
    """The publication year in PMC-Patients' pub_date (its first four-digit year from 1900 on), or "" when it holds none."""
    match = re.search(r"(?<!\d)(?:19|20)\d\d", "" if value is None else str(value))
    return match.group(0) if match else ""


def _age(value) -> str:
    if isinstance(value, list):
        return "; ".join(f"{v[0]:g} {v[1]}" if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, float)) else str(v)
                         for v in value)
    return "" if value is None else str(value)


def _judge(terms: list[dict], classes: list[str]):
    """The candidate rules, for one article's title and case text: the candidate's fields (its class, the classes the title
    names, the marks, and how often the text names the class by a full name and by its short form), or None when the title
    names no class; and whether the title names only names that count for no class."""
    matcher = _matcher(terms)
    not_skin = {tuple(_tokens(site)): site for site in NOT_SKIN}  # each site in the form of the title's words, and as written
    treatment = {tuple(_tokens(word)): word for word in TREATMENT}
    animals = {tuple(_tokens(word)): word for word in ANIMALS}
    looks_like = [tuple(_tokens(word)) for word in LOOKS_LIKE]
    specific = _matcher([t for t in terms if _words(t["term"]) not in {_words(name) for name in BARE}])
    short = {label: {w + end for form in SHORT_FORMS.get(label, ()) for w in _tokens(form) for end in ("", "s")} for label in classes}

    def judge(title: str, text: str) -> tuple[dict | None, bool]:
        title_words = _tokens(title)
        named, others = _named(title_words, matcher)
        named = sorted(named)
        if not named:
            return None, bool(others)
        marks = []
        if len(named) > 1:
            marks.append("names two or more classes")
        if others:
            marks.append(f"also names {', '.join(others)} (does not count)")
        sites = _found(title_words, not_skin)
        if sites:
            marks.append(f"a site that may not be skin: {', '.join(sites)}")
        about = _found(title_words, treatment)
        if about:
            marks.append(f"the title is about treatment: {', '.join(about)}")
        species = _found(title_words, animals)
        if species:
            marks.append(f"the title is about an animal: {', '.join(species)}")
        first = min((i for i in range(len(title_words)) for seq in looks_like if tuple(title_words[i:i + len(seq)]) == seq),
                    default=None)
        if len(named) == 1 and (named[0] not in _named(_without_look_alikes(title_words, matcher), matcher)[0] or
                                (first is not None and named[0] not in _named(title_words[:first], matcher)[0])):
            marks.append(LOOKS_MARK)
        if len(named) == 1 and named[0] in BARE and named[0] not in _named(title_words, specific)[0]:
            marks.append(BARE_MARK)
        full = short_forms = ""
        if len(named) == 1:
            text_words = _tokens(text)
            full = _named(text_words, matcher)[0][named[0]]
            short_forms = sum(w in short[named[0]] for w in text_words)
            if not full and not short_forms:
                marks.append(TEXT_MARK)
        return {"class": named[0] if len(named) == 1 else "", "classes_named": "; ".join(named), "marks": "; ".join(marks),
                "title": " ".join(title.split()), "text_full_names": full, "text_short_forms": short_forms}, False
    return judge


def _without_look_alikes(words: list[str], matcher) -> list[str]:
    """A title's words without the names written as what something looked like ("dermatofibroma-like", "melanoma like")."""
    drop = set()
    for n, names in matcher:
        for i in range(len(words) - n):
            if words[i + n] == "like" and tuple(sorted(words[i:i + n])) in names:
                drop.update(range(i, i + n + 1))
    return [word for k, word in enumerate(words) if k not in drop]


def _sort_candidates(rows: list[dict], classes: list[str]) -> None:
    order = {label: n for n, label in enumerate(classes)}
    rows.sort(key=lambda r: (order.get(r["class"], len(classes)), r["classes_named"], _uid_key(r["patient_uid"])))


def cases(data: str | Path = DATA, classes: list[str] | None = None, show=None) -> dict:
    """Downloads PMC-Patients once, and lists every patient whose article title names one of the classes (by the names that count
    in class_terms.csv), with marks where a person should look; writes case_candidates.csv and the cases section of
    dataset_report.md. No case text is written into the repository and nothing is cut."""
    data = Path(data)
    classes = classes if classes is not None else image_classes(read_table(IMAGE_TABLE))
    judge = _judge(read_terms(data / TERMS, classes), classes)
    meta = _download_pmc_patients(data / SOURCES / "pmc_patients", show)
    path = data / SOURCES / "pmc_patients" / PMC_PATIENTS["file"]
    rows, read, only_other, keys = [], 0, 0, set()
    for record in _json_list(path):
        read += 1
        if not isinstance(record, dict) or any(k not in record for k in PATIENT_FIELDS):
            found = ", ".join(sorted(record)) if isinstance(record, dict) else type(record).__name__
            raise ValueError(f"patient {read} of {path.as_posix()} lacks a field this step needs ({', '.join(PATIENT_FIELDS)}); it "
                             f"has: {found}")
        keys.update(record)
        text = str(record["patient"])
        row, other = judge(str(record["title"]), text)
        only_other += other
        if row:
            rows.append({"patient_uid": record["patient_uid"], "pmid": record["PMID"], "year": _year(record.get("pub_date")),
                         **row, "age": _age(record.get("age")), "sex": record.get("gender") or "", "words": len(text.split())})
        if show and read % 50_000 == 0:
            show(f"{read} patients read, {len(rows)} candidates so far")
    _sort_candidates(rows, classes)
    _write_csv(data / CANDIDATES, CANDIDATE_COLUMNS, rows)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "cases", _cases_summary(meta, read, sorted(keys), rows, only_other, classes))
    return {"read": read, "rows": rows, "classes": classes, "meta": meta, "table": (data / CANDIDATES).as_posix(),
            "summary": (data / REPORT).as_posix()}


def _cases_summary(meta, read, keys, rows, only_other, classes) -> list[str]:
    def has(row, text):
        return text in row["marks"]

    lines = ["# Cases: candidate patient cases for each class", "",
             "Written by `python -m src.text.prepare cases`. The cases come from PMC-Patients (Zhao et al., Scientific Data 10, 909, "
             "2023): patient descriptions taken from published case reports in PubMed Central whose licence allows reuse. A patient "
             "is a candidate when the title of the article it comes from names one of the classes, by the names that count in "
             "`class_terms.csv` (a case report's title usually names its diagnosis). Nothing is chosen, read or cut yet: the "
             "marks show where a person should look first. The case texts stay on this computer; `case_candidates.csv` holds "
             "only each patient's numbers, publication year, title, age, sex, length, how often the case text names the class "
             "(by a full name and by its short form) and marks.", "",
             "## Source", "",
             f"- File: `{meta['file']}` from the Hugging Face repository `{meta['repository']}`, version (commit) `{meta['commit']}`",
             f"- Size: {meta['bytes']} bytes; SHA-256 `{meta['sha256']}`; downloaded {meta['retrieved_utc']}; licence "
             f"{meta['licence']}",
             f"- Patients read: {read}; fields: {', '.join(keys)}",
             f"- Publication years of the candidates (from `pub_date`): {_span(rows)}; candidates without a year: "
             f"{sum(not r['year'] for r in rows)}", "",
             "## Candidates per class", "",
             f"- Candidates: {len(rows)} patients from {len({r['pmid'] for r in rows})} articles; titles naming only names that do "
             f"not count (for example keratoacanthoma), so no candidate: {only_other}", "",
             "| Class | Candidates | Articles | Without marks | Names two or more classes | Also names a name that does not count | "
             "A site that may not be skin | Title about treatment | Case text does not name the class | Case text names the class "
             "only by its short form | Years |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label in classes:
        mine = [r for r in rows if label in r["classes_named"].split("; ")]
        only_short = sum(r["class"] == label and not r["text_full_names"] and bool(r["text_short_forms"]) for r in mine)
        lines.append(f"| {label} | {len(mine)} | {len({r['pmid'] for r in mine})} | {sum(not r['marks'] for r in mine)} | "
                     f"{sum(has(r, 'two or more') for r in mine)} | {sum(has(r, 'does not count') for r in mine)} | "
                     f"{sum(has(r, 'may not be skin') for r in mine)} | {sum(has(r, 'about treatment') for r in mine)} | "
                     f"{sum(has(r, 'does not name the class') for r in mine)} | {only_short} | {_span(mine)} |")
    lines += ["", "A patient whose title names two or more classes is counted under each of them (its class is empty in "
              "`case_candidates.csv` until a person decides).", "",
              "Words in a title that point to a site other than the skin (such a case is marked, not dropped): "
              + ", ".join(sorted(NOT_SKIN)) + ".", "",
              "Words in a title that show it is about treatment (such a case is marked, not dropped): "
              + ", ".join(sorted(TREATMENT)) + ".", "",
              "Short forms counted in the case texts (a case text that uses the short form names its class): "
              + "; ".join(f"{label}: {', '.join(forms)}" for label, forms in SHORT_FORMS.items() if label in classes) + ".", "",
              "## Not done in this step", "",
              "- No case is read, chosen or cut, and no class is final: a person confirms each case's diagnosis in a later step.",
              "- No licence is looked up yet; that is done for the cases chosen.", ""]
    return lines


def _span(rows) -> str:
    years = sorted(r["year"] for r in rows if r["year"])
    return f"{years[0]}–{years[-1]}" if years else "none"


# ---------------------------------------------------------------------------------------------------------------------------------
# Abstracts: case reports PubMed holds only as an abstract


def _in_titles(names: list[str]) -> list[str]:
    """The names some PubMed title uses. A name used only in abstracts finds no title, and PubMed answers a search by title
    for it with a warning that the phrase is not found (which stops _search), so it is left out of a search by title."""
    answer = _eutils("esearch", db="pubmed", term=" OR ".join(f'"{name}"[Title]' for name in names), retmax=0).get("esearchresult")
    if not isinstance(answer, dict) or "count" not in answer:
        raise ValueError(f"NCBI's search answer has an unexpected form for the names {names}")
    unknown = {phrase.lower() for found in (answer.get("warninglist") or {}).get("quotedphrasesnotfound") or []
               for phrase in re.findall(r'"([^"]+)"', found)}
    return [name for name in names if name.lower() not in unknown]


def _pubmed_ids(query: str, years: tuple[int, int] = (1800, 3000)) -> list[str]:
    """Every PubMed article number a query finds; a result larger than PubMed lists at once is asked for in parts by year."""
    first, last = years
    answer = _search("pubmed", f"{query} AND {first}:{last}[Date - Publication]", retmax=ESEARCH_MAX)
    count, ids = int(answer["count"]), answer.get("idlist") or []
    if count <= ESEARCH_MAX:
        if len(ids) != count:
            raise ValueError(f"PubMed counted {count} articles but listed {len(ids)} for {query}. Nothing was written")
        return ids
    if first == last:
        raise ValueError(f"PubMed holds more than {ESEARCH_MAX} articles of {first} for {query}: they cannot be listed. Nothing "
                         "was written")
    middle = (first + last) // 2
    return _pubmed_ids(query, (first, middle)) + _pubmed_ids(query, (middle + 1, last))


def _efetch(ids: list[str]) -> list[dict]:
    """PubMed's records of the articles: number, PMC number (when PMC holds it), publication year, title and abstract (as
    labelled sections)."""
    def parse(content: bytes) -> list[dict]:
        root = ElementTree.fromstring(content)
        if root.tag != "PubmedArticleSet":
            raise ValueError(f"PubMed's records have an unexpected form (<{root.tag}>)")
        records = []
        for article in root.iter("PubmedArticle"):
            pmid, body = article.findtext("MedlineCitation/PMID"), article.find("MedlineCitation/Article")
            if not pmid or body is None:
                raise ValueError("a PubMed record lacks its article number or its article")
            title = body.find("ArticleTitle")
            date = body.find("Journal/JournalIssue/PubDate")
            pmc = [e.text for e in article.findall("PubmedData/ArticleIdList/ArticleId") if e.get("IdType") == "pmc" and e.text]
            records.append({"pmid": pmid, "pmcid": pmc[0].strip() if pmc else "",
                            "year": _year(" ".join(date.itertext()) if date is not None else ""),
                            "title": " ".join("".join(title.itertext()).split()) if title is not None else "",
                            "sections": [[part.get("Label") or "", " ".join("".join(part.itertext()).split())]
                                         for part in body.findall("Abstract/AbstractText")]})
        return records
    return fetch(f"{EUTILS}efetch.fcgi", "application/xml", parse, delay=NCBI_DELAY, attempts=NCBI_ATTEMPTS,
                 form={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})


def _running_text(element) -> str:
    """An element's text with figures, tables and their captions left out."""
    if element.tag in FULLTEXT_SKIP:
        return ""
    return (element.text or "") + "".join(_running_text(child) + (child.tail or "") for child in element)


def _paragraphs(section) -> list[str]:
    """The paragraphs of a full text's section and of the sections inside it, in order (titles, figures and tables left out)."""
    found = []
    for child in section:
        if child.tag == "p":
            found.append(" ".join(_running_text(child).split()))
        elif child.tag == "sec":
            found += _paragraphs(child)
    return [p for p in found if p]


def _pmc_articles(numbers: list[str]) -> dict[str, dict]:
    """Open full texts in PubMed Central, by PMC number (without "PMC"): the body's sections as [title, paragraphs] (paragraphs
    outside any section under the title "") and the licence. An article PMC does not give in full is not in the answer."""
    def parse(content: bytes) -> dict[str, dict]:
        root = ElementTree.fromstring(content)
        if root.tag != "pmc-articleset":
            raise ValueError(f"PMC's records have an unexpected form (<{root.tag}>)")
        articles = {}
        for article in root.iter("article"):
            ids = {e.get("pub-id-type"): (e.text or "").strip() for e in article.findall("front/article-meta/article-id")}
            number = (ids.get("pmc") or ids.get("pmcid") or "").removeprefix("PMC")
            body = article.find("body")
            sections = [] if body is None else [["", [" ".join(_running_text(p).split()) for p in body.findall("p")]]] + [
                [" ".join((sec.findtext("title") or "").split()), _paragraphs(sec)] for sec in body.findall("sec")]
            sections = [[title, [p for p in paragraphs if p]] for title, paragraphs in sections]
            sections = [section for section in sections if section[1]]
            if number and sections:
                articles[number] = {"sections": sections, "licence": _licence(article.find("front/article-meta/permissions/license"))}
        return articles
    return fetch(f"{EUTILS}efetch.fcgi", "application/xml", parse, delay=NCBI_DELAY, attempts=NCBI_ATTEMPTS,
                 form={"db": "pmc", "id": ",".join(numbers), "retmode": "xml"})


def _licence(element) -> str:
    """A licence element's link, or else the Creative Commons address in its text (where PMC often puts it, in
    <ali:license_ref>), or else its type ("" without one)."""
    if element is None:
        return ""
    link = element.get("{http://www.w3.org/1999/xlink}href")
    found = re.search(r"https?://creativecommons\.org/licenses/[\w./-]+", " ".join(element.itertext()))
    return (link or (found.group(0).rstrip(".") if found else "") or element.get("license-type") or "").strip()


def _pmc_cases(numbers: list[str]) -> dict[str, dict]:
    """The case sections of open full texts in PubMed Central, by PMC number (without "PMC"): the running text of the body's
    sections titled as the case (such as "Case presentation"), and the licence. An article PMC does not give in full, or
    without such a section, is not in the answer."""
    cases = {}
    for number, article in _pmc_articles(numbers).items():
        paragraphs = [p for title, paragraphs in article["sections"] if FULLTEXT_CASE.search(title) for p in paragraphs]
        if paragraphs:
            cases[number] = {"fulltext": " ".join(paragraphs), "licence": article["licence"]}
    return cases


def _record_case(record: dict) -> tuple[str, str]:
    """The case text of a downloaded PubMed record and where it comes from: the case sections of its open full text in PMC,
    else its abstract's case part."""
    if record.get("fulltext"):
        return record["fulltext"], "PMC full text"
    return _case_part(record["sections"]), "PubMed abstract"


def _case_part(sections: list) -> str:
    """An abstract's case: its sections whose label names the case or the patient, or the whole abstract when none does."""
    case = [text for label, text in sections if label and CASE_PART.search(label)]
    return " ".join(case or [text for _, text in sections]).strip()


def _abstract_records(data: Path) -> list[dict]:
    path = data / SOURCES / ABSTRACTS / "abstracts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: run python -m src.text.prepare abstracts first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def abstracts(data: str | Path = DATA, classes: list[str] | None = None, show=None) -> dict:
    """Finds every class's case reports in PubMed (the class's headings as a main subject, on the skin, in English), leaves out
    the articles PMC-Patients already gave, downloads the others' titles and abstracts into data/text/sources/pubmed/ (not
    committed), and for the candidate articles (their title passes) with an open full text in PMC the case sections of it;
    applies the candidate rules of ``cases`` to each article's case text (the full text's case sections, else the abstract's
    case part); writes abstract_candidates.csv and the abstracts section of dataset_report.md."""
    data = Path(data)
    classes = classes if classes is not None else image_classes(read_table(IMAGE_TABLE))
    terms, headings = _checked_headings(data, classes)
    checked = _read_csv(data / CHECK, CHECK_COLUMNS, "survey")
    listed = {r["pmid"] for r in _read_csv(data / CANDIDATES, CANDIDATE_COLUMNS, "cases")}
    found, by_title = defaultdict(set), set()
    for label in classes:
        if headings[label]:
            for pmid in _pubmed_ids(f"{_subject(headings[label])} AND {ABSTRACT_QUERY}"):
                found[pmid].add(label)
        names = [r["term"] for r in checked if r["class"] == label and r["counts"] == "yes" and int(r["pubmed_articles"] or 0)]
        names = _in_titles(names) if names else []
        if names:  # the case reports PubMed has not indexed, by the names in their title
            for pmid in _pubmed_ids("(" + " OR ".join(f'"{name}"[Title]' for name in names) + f") AND {UNINDEXED_QUERY}"):
                found[pmid].add(label)
                by_title.add(pmid)
        if show:
            show(f"{label}: {sum(label in c for c in found.values())} case reports in PubMed")
    wanted = sorted(set(found) - listed, key=int)
    records = []
    for start in range(0, len(wanted), EFETCH_BATCH):
        records += _efetch(wanted[start:start + EFETCH_BATCH])
        if show:
            show(f"{min(start + EFETCH_BATCH, len(wanted))} of {len(wanted)} abstracts downloaded")
    asked = set(wanted)
    records = sorted({r["pmid"]: r for r in records if r["pmid"] in asked}.values(), key=lambda r: int(r["pmid"]))
    judge = _judge(terms, classes)
    worth = [r for r in records if r["pmcid"] and _title_passes(judge(r["title"], _case_part(r["sections"]))[0])]
    cases = {}
    for start in range(0, len(worth), FULLTEXT_BATCH):
        cases.update(_pmc_cases([r["pmcid"].removeprefix("PMC") for r in worth[start:start + FULLTEXT_BATCH]]))
        if show:
            show(f"{min(start + FULLTEXT_BATCH, len(worth))} of {len(worth)} full texts asked from PMC")
    for record in records:
        case = cases.get(record["pmcid"].removeprefix("PMC"), {}) if record["pmcid"] else {}
        record["fulltext"], record["licence"] = case.get("fulltext", ""), case.get("licence", "")
    folder = data / SOURCES / ABSTRACTS
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / "abstracts.jsonl.partial"
    partial.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
    os.replace(partial, folder / "abstracts.jsonl")
    rows, only_other, empty = [], 0, 0
    for record in records:
        text = _record_case(record)[0]
        if not text:
            empty += 1
            continue
        row, other = judge(record["title"], text)
        only_other += other
        if row:
            rows.append({"patient_uid": f"pubmed-{record['pmid']}", "pmid": record["pmid"], "year": record["year"], **row,
                         "age": "", "sex": "", "words": len(text.split())})
    _sort_candidates(rows, classes)
    _write_csv(data / ABSTRACT_CANDIDATES, CANDIDATE_COLUMNS, rows)
    day = datetime.now(timezone.utc).date().isoformat()
    write_section(data / REPORT, REPORT_HEAD, STEPS, "abstracts",
                  _abstracts_summary(found, by_title, listed, records, rows, only_other, empty, classes, headings, day,
                                     len(worth)))
    return {"rows": rows, "classes": classes, "downloaded": len(records), "fulltexts": sum(bool(r["fulltext"]) for r in records),
            "table": (data / ABSTRACT_CANDIDATES).as_posix(),
            "summary": (data / REPORT).as_posix()}


def _title_passes(row: dict | None) -> bool:
    """Whether a candidate's title passes the rules: it names one class and has no mark of its own (only the case text can
    still fail, and a full text may name the class where the abstract does not)."""
    return bool(row and row["class"]) and all(mark == TEXT_MARK for mark in filter(None, row["marks"].split("; ")))


def _abstracts_summary(found, by_title, listed, records, rows, only_other, empty, classes, headings, day, asked_pmc) -> list[str]:
    with_text = {r["pmid"] for r in records if _record_case(r)[0]}
    full = {r["pmid"] for r in records if r["fulltext"]}
    new = set(found) - listed
    lines = ["# Abstracts: case reports PubMed holds as an abstract", "",
             "Written by `python -m src.text.prepare abstracts`. Most case reports are in PubMed only as an abstract, without "
             "open full text. For every class, PubMed was asked for its case reports (the class's headings as a main subject, "
             "on the skin, in English) and, by the class's names in the title, for the case reports it has not indexed (journals "
             "outside MEDLINE have no subject headings, so the first search cannot find them); the articles PMC-Patients already "
             "gave were left out, and the titles and abstracts of "
             "the others were downloaded. Each abstract's case part (its sections whose label names the case or the patient, "
             "or else the whole abstract) is judged by the rules of `cases`; for a candidate article (its title passes the "
             "rules) with an open full text in PubMed Central, of any year, the case sections of the full text (titled as the case, "
             "such as \"Case presentation\"; figures and tables left out) are used instead of the abstract. The texts stay in "
             "`data/text/sources/pubmed/`; `abstract_candidates.csv` holds only each article's numbers, year, title, length and "
             "marks.", "",
             f"- Asked on {day}: {len(found)} case reports found; already given by PMC-Patients (left out): {len(set(found) & listed)}; "
             f"downloaded: {len(records)} (not returned by PubMed: {len(new) - len(records)}); without an abstract: {empty}",
             f"- Candidate articles (title passes) with a PMC number, asked from PMC: {asked_pmc}; with an open full text and a "
             f"case section (used instead of the abstract): {len(full)}",
             f"- Candidates: {len(rows)}; titles naming only names that do not count, so no candidate: {only_other}", "",
             "| Class | MeSH headings (main topic) | Case reports in PubMed | Of them not indexed (found by title) | Already from "
             "PMC-Patients | With a case text | Of them from a PMC full text | Candidates (title names only this class) | Without "
             "marks |", "|---|---|---|---|---|---|---|---|---|"]
    for label in classes:
        mine = {pmid for pmid, labels in found.items() if label in labels}
        lines.append(f"| {label} | {'; '.join(headings[label]) or 'none'} | {len(mine)} | {len(mine & by_title)} | {len(mine & listed)} | "
                     f"{len(mine & with_text)} | {len(mine & full)} | {sum(r['class'] == label for r in rows)} | "
                     f"{sum(r['class'] == label and not r['marks'] for r in rows)} |")
    lines += ["", f"Queries for each class: its headings as a main subject AND {ABSTRACT_QUERY}; and its names that count (and that "
              f"some PubMed title uses) in the title AND {UNINDEXED_QUERY}.", "",
              "## Not done in this step", "",
              "- No abstract or full text is written into the repository: they are under their publishers' copyright or "
              "licences, so they stay on this computer and are used for research only (each full text's licence is kept "
              "with it).", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# MeSH: PubMed's subject headings confirm each candidate article's class, on the skin


def _checked_headings(data: Path, classes: list[str]) -> tuple[list[dict], dict[str, list[str]]]:
    """The names of class_terms.csv and, per class, the MeSH headings the survey found for its names that count; stops when
    term_check.csv no longer lists the names as they are."""
    terms = read_terms(data / TERMS, classes)
    checked = _read_csv(data / CHECK, CHECK_COLUMNS, "survey")
    if sorted((r["class"], r["term"], r["counts"]) for r in checked) != sorted((t["class"], t["term"], t["counts"]) for t in terms):
        raise ValueError(f"{CHECK} does not list the names of {TERMS} as they are now: run python -m src.text.prepare survey "
                         "again first")
    return terms, _class_headings(checked)


def _candidates(data: Path) -> list[dict]:
    """The candidates of both sources: PMC-Patients' (``cases``), then PubMed abstracts' (``abstracts``)."""
    return [{**r, "source": "PMC-Patients"} for r in _read_csv(data / CANDIDATES, CANDIDATE_COLUMNS, "cases")] + \
           [{**r, "source": "PubMed abstract"} for r in _read_csv(data / ABSTRACT_CANDIDATES, CANDIDATE_COLUMNS, "abstracts")]


def _epost(ids: list[str]) -> tuple[str, str, list[str]]:
    """Sends article numbers to PubMed's history: its query key and session (WebEnv), and the numbers PubMed does not know."""
    def parse(content: bytes):
        root = ElementTree.fromstring(content)
        key, session = root.findtext("QueryKey"), root.findtext("WebEnv")
        if not key or not session:
            problem = root.findtext("ERROR") or ElementTree.tostring(root, encoding="unicode")[:200]
            raise ValueError(f"PubMed did not take the article numbers: {problem}")
        return key, session, [e.text for e in root.findall("InvalidIdList/Id") if e.text]
    return fetch(f"{EUTILS}epost.fcgi", "application/xml", parse, delay=NCBI_DELAY, attempts=NCBI_ATTEMPTS,
                 form={"db": "pubmed", "id": ",".join(ids)})


def mesh(data: str | Path = DATA, classes: list[str] | None = None, show=None) -> dict:
    """Asks PubMed, for every candidate article, whether it has subject headings (MeSH) and which classes they name (by the
    headings of each class's names that count in term_check.csv) with a skin heading; writes article_mesh.csv and the mesh
    section of dataset_report.md. Only article numbers are sent; no article text is downloaded."""
    data = Path(data)
    classes = classes if classes is not None else image_classes(read_table(IMAGE_TABLE))
    headings = _checked_headings(data, classes)[1]
    candidates = _candidates(data)
    pmids = sorted({r["pmid"] for r in candidates}, key=int)
    found, indexed, unknown = defaultdict(set), set(), []
    for start in range(0, len(pmids), EPOST_BATCH):
        batch = pmids[start:start + EPOST_BATCH]
        key, session, invalid = _epost(batch)
        unknown += invalid

        def among(query: str, what: str) -> list[str]:
            answer = _search("pubmed", f"#{key} AND {query}", retmax=len(batch), WebEnv=session, usehistory="y")
            ids = answer.get("idlist") or []
            if int(answer["count"]) != len(ids) or not set(ids) <= set(batch):
                raise ValueError(f"PubMed's answer for {what} does not fit the {len(batch)} article numbers sent ({answer['count']} "
                                 "found). Nothing was written")
            return ids
        indexed.update(among(INDEXED, "the indexed articles"))
        for label in classes:
            if headings[label]:
                for pmid in among(f"{_subject(headings[label], 'MeSH Terms')} AND {SKIN}", label):
                    found[pmid].add(label)
                    indexed.add(pmid)  # it has headings, whatever its subset
        if show:
            show(f"{min(start + EPOST_BATCH, len(pmids))} of {len(pmids)} articles checked in PubMed")
    order = {label: n for n, label in enumerate(classes)}
    rows = [{"pmid": pmid, "indexed": "yes" if pmid in indexed else "no",
             "classes": "; ".join(sorted(found[pmid], key=order.__getitem__))} for pmid in pmids]
    _write_csv(data / MESH_TABLE, MESH_COLUMNS, rows)
    day = datetime.now(timezone.utc).date().isoformat()
    write_section(data / REPORT, REPORT_HEAD, STEPS, "mesh", _mesh_summary(rows, candidates, classes, headings, unknown, day))
    return {"rows": rows, "classes": classes, "unknown": unknown, "indexed": len(indexed), "table": (data / MESH_TABLE).as_posix(),
            "summary": (data / REPORT).as_posix()}


def _mesh_summary(rows, candidates, classes, headings, unknown, day) -> list[str]:
    confirmed = {r["pmid"]: set(filter(None, r["classes"].split("; "))) for r in rows}
    indexed = {r["pmid"] for r in rows if r["indexed"] == "yes"}
    lines = ["# MeSH: the subject headings of each candidate article", "",
             "Written by `python -m src.text.prepare mesh`. For every article in `case_candidates.csv`, PubMed was asked whether "
             f"it has subject headings at all (`{INDEXED}`: PubMed indexes the articles of MEDLINE journals) and which classes "
             "they name (one of the headings of the class's names that count in `term_check.csv`, or a narrower heading) while "
             "it is also indexed under skin neoplasms or skin diseases. The US National Library of Medicine gives these "
             "headings, independently of the authors' titles and of this code. In the build, the headings of an indexed article "
             "must name its class; an article without headings is judged by its title and case text.", "",
             f"- Asked on {day}: {len(rows)} articles, {len(indexed)} of them indexed; article numbers PubMed does not know: "
             f"{len(unknown)}" + (f" ({', '.join(unknown[:10])}{' ...' if len(unknown) > 10 else ''})" if unknown else ""), "",
             "| Class | MeSH headings | Candidate articles (title names only this class) | Indexed | Indexed, headings name the "
             "class on the skin | Indexed, headings do not | Not indexed |", "|---|---|---|---|---|---|---|"]
    for label in classes:
        mine = {r["pmid"] for r in candidates if r["class"] == label}
        known = mine & indexed
        yes = sum(label in confirmed.get(pmid, ()) for pmid in known)
        lines.append(f"| {label} | {'; '.join(headings[label]) or 'none'} | {len(mine)} | {len(known)} | {yes} | "
                     f"{len(known) - yes} | {len(mine) - len(known)} |")
    lines += ["", f"Skin: {SKIN}.", "", "## Not done in this step", "",
              "- Nothing is labelled yet: the build step uses these headings to check the class of every indexed article.", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# Build: labels, questions, checks and splits, with no person choosing, labelling or cutting a case


def _sentence_starts(text: str) -> list[int]:
    """Where each sentence of a text starts: after a full stop, question or exclamation mark followed by a space and a capital
    letter, a digit or an opening bracket, unless the stop ends a common short form such as "Fig." or "e.g."."""
    starts = [0]
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'“])", text):
        before = text[starts[-1]:m.start()].split()
        if before and before[-1].lstrip("([\"'“").lower() in ABBREVIATIONS:
            continue
        starts.append(m.end())
    return starts


def _revealer(terms: list[dict], label: str):
    """A test of a sentence's words for a case of class ``label``: does it name the class (any of its names in class_terms.csv,
    counting or not, except the descriptive ones), use its short form, or report a test result? Names of the other classes do
    not reveal the answer: they stay in a question as clinical information (a history, a suspicion)."""
    vague = {_words(word) for word in DESCRIPTIVE}
    matcher = _matcher([t for t in terms if t["class"] == label and _words(t["term"]) not in vague])
    short = {w + end for form in SHORT_FORMS.get(label, ()) for w in _tokens(form) for end in ("", "s")}

    def reveals(words: list[str]) -> bool:
        named, others = _named(words, matcher)
        return bool(named or others) or any(w in short or w.startswith(RESULT_CUES) for w in words)
    return reveals


def _defined_short_forms(text: str, reveals) -> list[str]:
    """The short forms a text defines for a class, as "pyogenic granuloma (PG)" or "PG (pyogenic granuloma)": the authors' own
    short forms, which name the class as its full name does (``reveals`` tests the words before or inside the brackets)."""
    found = set()
    for m in re.finditer(rf"([^()]{{0,80}})\(\s*({SHORT_FORM})\s*[),;]", text):
        if reveals(_tokens(m.group(1))[-5:]):
            found.add(m.group(2))
    for m in re.finditer(rf"\b({SHORT_FORM})\s*\(([^()]{{3,80}})\)", text):
        if reveals(_tokens(m.group(2))):
            found.add(m.group(1))
    return sorted({f[:-1] if f.endswith("s") and sum(c.isupper() for c in f) >= 2 else f for f in found})


def _says(reveals, short_forms=()):
    """A test of a text (a sentence or a whole question) for one case: does it name the case's class or report a test result
    (``reveals``, on its words), or use a short form the case's article defines for the class (as written, capitals and all)?"""
    defined = re.compile(r"\b(?:" + "|".join(map(re.escape, short_forms)) + r")s?\b") if short_forms else None
    return lambda text: reveals(_tokens(text)) or bool(defined and defined.search(text))


def _question(text: str, says) -> str:
    """The case text word for word up to its first sentence that reveals its class (``says``): from its start, or, when a
    sentence before its first sentence about the patient already reveals the class (a background such as "Basal cell carcinoma
    is the most common skin cancer"), from that first sentence about the patient."""
    text = text.strip()
    starts = _sentence_starts(text)
    sentences = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    patient = next((n for n, sentence in enumerate(sentences) if PATIENT.search(sentence)), 0)
    first = patient if any(says(sentence) for sentence in sentences[:patient]) else 0
    for n in range(first, len(sentences)):
        if says(sentences[n]):
            return text[starts[first]:starts[n]].strip()
    return text[starts[first]:].strip()


def _inside(question: str, lesion: set[str], inside: dict) -> bool:
    """Whether a sentence of the question puts a lesion at a site inside the body: a lesion word and a site of INSIDE in the
    same sentence ("a black tumor on the retromolar area"; not "the intraoral examination was normal")."""
    starts = _sentence_starts(question)
    for a, b in zip(starts, starts[1:] + [len(question)]):
        words = _tokens(question[a:b])
        if lesion.intersection(words) and _found(words, inside):
            return True
    return False


def _same_patients(rows: list[dict]) -> list[dict]:
    """The questions that tell the case of a question of another article again (the same patient published twice): they share
    at least SAME_PATIENT of the shorter question's runs of SHARED_RUN words. The article PubMed listed first (the lower PubMed
    number) keeps its case; the later ones are returned, to be left out."""
    kept, owners, later = {}, defaultdict(list), []
    for row in sorted(rows, key=lambda r: (int(r["pmid"]), _uid_key(r["patient_uid"]))):
        words = re.findall(r"[a-z0-9]+", row["question"].lower())
        runs = {tuple(words[i:i + SHARED_RUN]) for i in range(len(words) - SHARED_RUN + 1)}
        shared = Counter(uid for run in runs for uid in owners[run] if kept[uid]["pmid"] != row["pmid"])
        if any(n >= SAME_PATIENT * min(len(runs), len(kept[uid]["runs"])) for uid, n in shared.items()):
            later.append(row)
            continue
        kept[row["patient_uid"]] = {"pmid": row["pmid"], "runs": runs}
        for run in runs:
            owners[run].append(row["patient_uid"])
    return later


def _uid_key(uid: str):
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in uid.split("-")]


def _targets(n: int) -> dict[str, int]:
    """How many of a class's n cases go into test and into validation: TEST_SHARE and VAL_SHARE of them (rounded half up),
    at most TEST_PER_CLASS and VAL_PER_CLASS."""
    return {"test": min(TEST_PER_CLASS, int(n * TEST_SHARE + 0.5)), "val": min(VAL_PER_CLASS, int(n * VAL_SHARE + 0.5))}


def _split(rows: list[dict]) -> None:
    """Gives every row its split, each article whole: articles in a fixed random order (from SPLIT_SEED); an article goes into
    test while each of its classes stays within its test target (_targets), else into validation within its validation
    target, else into train."""
    articles = defaultdict(list)
    for row in rows:
        articles[row["pmid"]].append(row)
    caps = {label: _targets(n) for label, n in Counter(row["label"] for row in rows).items()}
    filled = {"test": Counter(), "val": Counter()}
    for pmid in sorted(articles, key=lambda p: hashlib.sha256(f"{SPLIT_SEED}:{p}".encode("utf-8")).hexdigest()):
        need = Counter(row["label"] for row in articles[pmid])
        split = next((s for s in ("test", "val") if all(filled[s][c] + n <= caps[c][s] for c, n in need.items())), "train")
        if split != "train":
            filled[split].update(need)
        for row in articles[pmid]:
            row["split"] = split


def build(data: str | Path = DATA, classes: list[str] | None = None, show=None) -> dict:
    """Makes the text dataset from the earlier steps' tables and the downloaded case texts, with no person in the loop: labels
    where the sources agree, questions cut before the diagnosis, checks, and splits by article. Writes dataset.csv, the question
    texts into data/text/dataset/ (not committed) and the build section of dataset_report.md."""
    data = Path(data)
    classes = classes if classes is not None else image_classes(read_table(IMAGE_TABLE))
    terms = read_terms(data / TERMS, classes)
    candidates = _candidates(data)
    mesh_rows = _read_csv(data / MESH_TABLE, MESH_COLUMNS, "mesh")
    confirmed = {r["pmid"]: set(filter(None, r["classes"].split("; "))) for r in mesh_rows}
    indexed = {r["pmid"] for r in mesh_rows if r["indexed"] == "yes"}
    missing = sorted({r["pmid"] for r in candidates} - set(confirmed), key=int)
    if missing:
        raise ValueError(f"{MESH_TABLE} lacks {len(missing)} candidate article(s), for example {missing[0]}: run python -m "
                         "src.text.prepare mesh again")
    folder = data / SOURCES / "pmc_patients"
    meta = _checked_source(folder)
    if meta is None:
        raise FileNotFoundError(f"{(folder / PMC_PATIENTS['file']).as_posix()} not found: run python -m src.text.prepare cases first")
    funnel = {label: Counter() for label in classes}
    chosen = {}
    for row in candidates:
        label = row["class"]
        if label not in funnel:  # a title naming two classes has no class
            continue
        funnel[label]["candidates"] += 1
        if row["marks"]:  # includes "the case text does not name the class": rule 3
            continue
        funnel[label]["no marks"] += 1
        if row["pmid"] in indexed and label not in confirmed[row["pmid"]]:  # PubMed's headings disagree
            funnel[label]["disagree"] += 1
            continue
        chosen[row["patient_uid"]] = row
    revealers, texts, whole = {label: _revealer(terms, label) for label in classes}, {}, {}
    for record in _json_list(folder / PMC_PATIENTS["file"]):
        if isinstance(record, dict) and record.get("patient_uid") in chosen:
            texts[record["patient_uid"]] = whole[record["patient_uid"]] = str(record["patient"])
    for record in _abstract_records(data):
        uid = f"pubmed-{record['pmid']}"
        if uid in chosen:
            texts[uid], chosen[uid]["source"] = _record_case(record)
            whole[uid] = " ".join([text for _, text in record["sections"]] + [record.get("fulltext") or ""])
    lost = sorted(set(chosen) - set(texts), key=_uid_key)
    if lost:
        raise ValueError(f"{len(lost)} candidate(s) are not among the downloaded texts, for example {lost[0]}: run python -m "
                         "src.text.prepare cases and abstracts again")
    lesion = {w for word in LESION_WORDS for w in _tokens(word)}
    inside = {tuple(_tokens(site)): site for site in INSIDE}
    features = {w for word in FEATURES + LESION_TYPES for w in _tokens(word)}
    rows, seen, leaks, duplicates = [], set(), 0, Counter()
    for uid in sorted(chosen, key=_uid_key):
        row, text = chosen[uid], texts[uid]
        reveals = revealers[row["class"]]
        says = _says(reveals, _defined_short_forms(f"{row['title']}\n{whole[uid]}", reveals))
        question = _question(text, says)
        if len(question.split()) < MIN_WORDS or not lesion.intersection(_tokens(question)):
            continue
        if not PATIENT.search(question):  # a background or a microscope's findings, not a patient
            funnel[row["class"]]["no patient"] += 1
            continue
        if _inside(question, lesion, inside):
            funnel[row["class"]]["inside"] += 1
            continue
        if not (SIZE.search(question) or features.intersection(_tokens(question))):  # only a treatment or a history, say
            funnel[row["class"]]["no feature"] += 1
            continue
        if says(question):  # the class named across two sentences: not left in a question
            leaks += 1
            continue
        assert question in text
        funnel[row["class"]]["question"] += 1
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates[row["class"]] += 1
            continue
        seen.add(digest)
        rows.append({"label": row["class"], "sha256": digest, "patient_uid": uid, "pmid": row["pmid"], "source": row["source"],
                     "pubmed": "agrees" if row["pmid"] in indexed else "not indexed", "year": row["year"],
                     "age": row["age"], "sex": row["sex"], "words": len(question.split()), "question": question})
    for row in _same_patients(rows):
        funnel[row["label"]]["same patient"] += 1
        rows.remove(row)
    _split(rows)
    order, split_order = {label: n for n, label in enumerate(classes)}, {"test": 0, "val": 1, "train": 2}
    rows.sort(key=lambda r: (split_order[r["split"]], order[r["label"]], _uid_key(r["patient_uid"])))
    partial, target = data / f"{FILES}.partial", data / FILES
    if partial.exists():
        shutil.rmtree(partial)
    for row in rows:
        row["path"] = f"{row['split']}/{row['label']}/{row['patient_uid']}.txt"
        path = partial / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row["question"].encode("utf-8"))
    if target.exists():
        shutil.rmtree(target)
    partial.mkdir(parents=True, exist_ok=True)
    os.replace(partial, target)
    _write_csv(data / TABLE_NAME, DATASET_COLUMNS, [{c: row[c] for c in DATASET_COLUMNS} for row in rows])
    write_section(data / REPORT, REPORT_HEAD, STEPS, "build", _build_summary(meta, funnel, rows, classes, leaks, duplicates))
    return {"rows": rows, "classes": classes, "funnel": funnel, "table": (data / TABLE_NAME).as_posix(),
            "folder": target.as_posix(), "summary": (data / REPORT).as_posix()}


def _build_summary(meta, funnel, rows, classes, leaks, duplicates) -> list[str]:
    splits = defaultdict(set)
    for row in rows:
        splits[row["pmid"]].add(row["split"])
    years = [row["year"] for row in rows if row["year"]]
    lines = ["# Build: the text dataset", "",
             "Written by `python -m src.text.prepare build`, with no person choosing, labelling or cutting a case. Each case is a "
             "real patient description from a published case report: a patient of PMC-Patients, the case sections of an open "
             "full text in PMC, or the case part of a PubMed abstract (`source` in `dataset.csv`). The task is the image dataset's: name the class. `dataset.csv` lists the "
             "cases; the question texts are in `data/text/dataset/` (not in Git: research use only), and `build` writes them "
             "again, the same, from the downloaded files.", "",
             "## Rules", "",
             "- Label: a case gets class X when the article title names X and no other class, with no mark (`cases`, "
             "`abstracts`), and the "
             "case text names X, by a full name or its short form; where PubMed has indexed the article, its subject headings "
             "must also name X on the skin (`mesh`), else the case is left out. `dataset.csv` says for every case whether PubMed "
             "agrees (three sources) or has not indexed the article (two sources).",
             "- Question: the case text word for word, up to the first sentence that names the case's own class (any of its "
             "names in `class_terms.csv`, counting or not; except " + ", ".join(DESCRIPTIVE) + ", too vague to be one), its "
             "short form (" + ", ".join(f for forms in SHORT_FORMS.values() for f in forms) + ", or one the article defines, "
             "such as \"pyogenic granuloma (PG)\"), or that reports a test result (words beginning " + ", ".join(RESULT_CUES)
             + "). It starts at the start of the case text, or, when a sentence before the first sentence about the patient "
             "already names the class (a background), at that first sentence about the patient. Names of the other classes "
             "stay, as clinical information (a history, a suspicion). Nothing is rewritten.",
             f"- Kept when the question has at least {MIN_WORDS} words, describes a lesion (" + ", ".join(LESION_WORDS) + ") and a "
             "patient (an age or a sex) and what the lesion looks like (a size in mm or cm, a lesion type: " + ", ".join(LESION_TYPES) +
             ", or a colour, surface or shape: " +
             ", ".join(FEATURES) + "), and has no sentence that puts a lesion inside the body (" +
             ", ".join(INSIDE) + "). When the questions of two articles share at least "
             f"{SAME_PATIENT:.0%} of the shorter one's runs of {SHARED_RUN} words, they tell the same patient's case: the article "
             "PubMed listed first (the lower PubMed number) keeps it.",
             f"- Splits: every article whole in one split, in a fixed random order (seed `{SPLIT_SEED}`): for each class, "
             f"{TEST_SHARE:.0%} of its cases into test (at most {TEST_PER_CLASS}), {VAL_SHARE:.0%} into validation (at most "
             f"{VAL_PER_CLASS}), the rest into train; an article that would pass a class's share goes into the next split.", "",
             "## Cases per class", "",
             f"- Source: `{meta['file']}`, version `{meta['commit']}`, SHA-256 `{meta['sha256']}` (checked)", "",
             "| Class | Candidates (title names only this class) | Without marks | PubMed's headings disagree (left out) | Usable "
             "question | In the dataset | From PMC full texts | From PubMed abstracts | PubMed agrees | "
             "PubMed has not indexed it | Test | Validation | Train | Question words (median) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    total = Counter()
    for label in classes:
        mine = [r for r in rows if r["label"] == label]
        counts = [funnel[label]["candidates"], funnel[label]["no marks"], funnel[label]["disagree"], funnel[label]["question"],
                  len(mine), sum(r["source"] == "PMC full text" for r in mine), sum(r["source"] == "PubMed abstract" for r in mine),
                  sum(r["pubmed"] == "agrees" for r in mine),
                  sum(r["pubmed"] == "not indexed" for r in mine),
                  *(sum(r["split"] == s for r in mine) for s in ("test", "val", "train"))]
        total.update(dict(enumerate(counts)))
        median = f"{statistics.median(r['words'] for r in mine):g}" if mine else "–"
        lines.append(f"| {label} | " + " | ".join(str(n) for n in counts) + f" | {median} |")
    median = f"{statistics.median(r['words'] for r in rows):g}" if rows else "–"
    lines += ["| all | " + " | ".join(str(total[n]) for n in range(12)) + f" | {median} |", "",
              "## Checks", "",
              f"- Questions that would still name their class across two sentences (left out): {leaks}",
              f"- Questions identical to an earlier one (left out): {sum(duplicates.values())}", "",
              "| Class | No patient described (left out) | A lesion inside the body (left out) | Nothing of what the lesion looks "
              "like (left out) | The same patient as an earlier article (left out) |", "|---|---|---|---|---|",
              *(f"| {label} | {funnel[label]['no patient']} | {funnel[label]['inside']} | {funnel[label]['no feature']} | "
                f"{funnel[label]['same patient']} |" for label in classes), "",
              f"- Articles in more than one split: {sum(len(s) > 1 for s in splits.values())}",
              "- Every question is part of its case text, word for word: yes", "",
              "## Not done in this step", "",
              "- No person has checked the labels: each rests on sources that agree (the authors' title and the case text, and "
              "PubMed's subject headings where the article is indexed), not on a dermatologist's review.",
              f"- Cases come from articles published from {min(years, default='–')} to {max(years, default='–')} (the year is in "
              "`dataset.csv`): a model may have read them in training.",
              "- The reference library for retrieval is built by `python -m src.text.library`, not here.", ""]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.text.prepare", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("survey", help="check every name in class_terms.csv against MeSH and count what PubMed holds per class")
    sub.add_parser("cases", help="download PMC-Patients once and list the patients whose article title names a class")
    sub.add_parser("abstracts", help="add the case reports PubMed holds only as an abstract, by the rules of cases")
    sub.add_parser("mesh", help="ask PubMed whether each candidate article is indexed and which classes its headings name")
    sub.add_parser("build", help="make the text dataset: labels, questions, checks and splits, with no person in the loop")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "survey":
            done = survey(show=print)
            found = {k: sum(r["in_mesh"] == k for r in done["rows"]) for k in ("heading", "entry term", "no")}
            unused = sum(r["in_mesh"] == "no" and not r["pubmed_articles"] for r in done["rows"])
            print(f"Names in MeSH: {found['heading']} as a heading, {found['entry term']} as an entry term; not in MeSH: {found['no']} "
                  f"(of them in no PubMed title or summary: {unused})")
            print(f"Written: {done['table']} and {done['summary']}. Only MeSH records and search counts were downloaded. Commit both "
                  "files.")
        elif args.command == "cases":
            done = cases(show=print)
            rows = done["rows"]
            print(f"Patients read: {done['read']}; candidates: {len(rows)} from {len({r['pmid'] for r in rows})} articles "
                  f"({sum(not r['marks'] for r in rows)} without marks)")
            for label in done["classes"]:
                mine = [r for r in rows if label in r["classes_named"].split("; ")]
                print(f"  {label}: {len(mine)} candidates, {sum(not r['marks'] for r in mine)} without marks")
            print(f"Written: {done['table']} and {done['summary']}. The case texts stay in data/text/sources/. Commit both files.")
        elif args.command == "abstracts":
            done = abstracts(show=print)
            rows = done["rows"]
            print(f"Abstracts downloaded: {done['downloaded']} (with an open full text's case sections: {done['fulltexts']}); "
                  f"candidates: {len(rows)} ({sum(not r['marks'] for r in rows)} without marks)")
            for label in done["classes"]:
                mine = [r for r in rows if r["class"] == label]
                print(f"  {label}: {len(mine)} candidates, {sum(not r['marks'] for r in mine)} without marks")
            print(f"Written: {done['table']} and {done['summary']}. The abstracts stay in data/text/sources/. Commit both files.")
        elif args.command == "mesh":
            done = mesh(show=print)
            for label in done["classes"]:
                print(f"  {label}: {sum(label in r['classes'].split('; ') for r in done['rows'])} articles whose headings name it")
            print(f"Articles indexed by PubMed: {done['indexed']} of {len(done['rows'])}; article numbers PubMed does not know: "
                  f"{len(done['unknown'])}")
            print(f"Written: {done['table']} and {done['summary']}. Only article numbers were sent. Commit both files.")
        elif args.command == "build":
            done = build(show=print)
            for label in done["classes"]:
                mine = [r for r in done["rows"] if r["label"] == label]
                print(f"  {label}: {len(mine)} cases (test {sum(r['split'] == 'test' for r in mine)}, validation "
                      f"{sum(r['split'] == 'val' for r in mine)}, train {sum(r['split'] == 'train' for r in mine)})")
            print(f"Written: {done['table']}, the question texts in {done['folder']}/ (not in Git) and {done['summary']}. Commit "
                  f"{done['table']} and {done['summary']}.")
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
