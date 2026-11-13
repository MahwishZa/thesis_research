# Preparation of the text dataset

Written by `python -m src.text.prepare`, one section per step in the order the steps run; each step rewrites only its own section. `class_terms.csv` is written by people: the names doctors use for each of the image dataset's classes, and whether each name counts for that class.

<!-- step: survey -->
## Survey: the names checked in MeSH, and how much PubMed holds per class

Written by `python -m src.text.prepare survey` on 2026-11-13. Every name in `class_terms.csv` was looked up in MeSH, the US National Library of Medicine's list of medical subject headings, through NCBI's E-utilities; the result for each name is in `term_check.csv`, with the number of PubMed articles that use the name in their title or summary. Then PubMed was searched per class with the MeSH headings of the class's confirmed names (as main topics), limited to skin lesions. Only MeSH records and search counts were downloaded, no article text; nothing else was changed.

### MeSH check

- Names: 120; a MeSH heading: 32; an entry term of a heading (another name MeSH lists for it): 17; not in MeSH: 71 (used in PubMed titles or summaries: 68; in none: 3)

A name that is not in MeSH is not wrong: MeSH lists the names used to index articles, not every name in use (subtypes and older names are often missing). The PubMed count shows whether such a name is in use. A name in neither is for a person to confirm, change or remove in `class_terms.csv`.

| Class | Name not in MeSH | Counts for the class | PubMed articles using it | Source in class_terms.csv |
|---|---|---|---|---|
| actinic keratosis | solar keratosis | yes | 244 | ISIC records of the image dataset |
| actinic keratosis | senile keratosis | yes | 54 | proposed; checked by the survey (term_check.csv) |
| basal cell carcinoma | basalioma | yes | 298 | proposed; checked by the survey (term_check.csv) |
| dermatofibroma | cutaneous fibrous histiocytoma | yes | 58 | proposed; checked by the survey (term_check.csv) |
| melanoma | melanoma in situ | yes | 1336 | ISIC records of the image dataset |
| melanoma | invasive melanoma | yes | 1098 | ISIC records of the image dataset |
| melanoma | lentigo maligna melanoma | yes | 668 | ISIC records of the image dataset |
| melanoma | superficial spreading melanoma | yes | 686 | ISIC records of the image dataset |
| melanoma | nevoid melanoma | yes | 73 | ISIC records of the image dataset |
| melanoma | nodular melanoma | yes | 775 | proposed; checked by the survey (term_check.csv) |
| melanoma | acral lentiginous melanoma | yes | 711 | proposed; checked by the survey (term_check.csv) |
| melanoma | conjunctival melanoma | no | 761 | decision 2026-11-12 |
| melanoma | mucosal melanoma | no | 1577 | decision 2026-11-12 |
| melanoma | melanoma of unknown primary | no | 211 | decision 2026-11-12 |
| melanoma | non-melanoma | no | 5108 | found by the audit of the dataset; checked by the survey (term_check.csv) |
| nevus | compound nevus | yes | 163 | ISIC records of the image dataset |
| nevus | junctional nevus | yes | 101 | ISIC records of the image dataset |
| nevus | dermal nevus | yes | 97 | ISIC records of the image dataset |
| nevus | atypical nevus | yes | 124 | ISIC records of the image dataset |
| nevus | Clark nevus | yes | 30 | ISIC records of the image dataset |
| nevus | congenital nevus | yes | 357 | ISIC records of the image dataset |
| nevus | lentiginous nevus | yes | 98 | ISIC records of the image dataset |
| nevus | Reed nevus | yes | 28 | proposed; checked by the survey (term_check.csv) |
| nevus | nevocellular nevus | yes | 65 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | acral nevus | yes | 31 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | recurrent nevus | yes | 27 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | combined nevus | yes | 55 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | Sutton nevus | yes | 27 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | Miescher nevus | yes | 1 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus spilus | yes | 189 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | speckled lentiginous nevus | yes | 81 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | kissing nevus | yes | 31 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | divided nevus | yes | 63 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | deep penetrating nevus | yes | 73 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | Meyerson nevus | yes | 18 | a mole type named in the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus sebaceus | no | 330 | proposed; checked by the survey (term_check.csv) |
| nevus | spider nevus | no | 21 | proposed; checked by the survey (term_check.csv) |
| nevus | Becker nevus | no | 277 | proposed; checked by the survey (term_check.csv) |
| nevus | nevus lipomatosus | no | 165 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus lipomatosis | no | 9 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus sebaceous | no | 388 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus comedonicus | no | 185 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | white sponge nevus | no | 142 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | woolly hair nevus | no | 44 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | faun tail nevus | no | 15 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus sudoriferous | no | 0 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus unius lateris | no | 33 | proposed from the candidate titles; checked by the survey (term_check.csv) |
| nevus | nevus depigmentosus | no | 97 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | nevus achromicus | no | 5 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | achromic nevus | no | 11 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | nevus anemicus | no | 70 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | lipomatous nevus | no | 18 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | eccrine nevus | no | 48 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | organoid nevus | no | 96 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | nevus of Ito | no | 36 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| nevus | magnocellular nevus | no | 11 | found by the audit's reading of questions; checked by the survey (term_check.csv) |
| pigmented benign keratosis | pigmented benign keratosis | yes | 0 | ISIC records of the image dataset |
| pigmented benign keratosis | solar lentigo | yes | 275 | HAM10000 group definition (Tschandl et al., 2018) |
| pigmented benign keratosis | lichen planus-like keratosis | yes | 79 | HAM10000 group definition (Tschandl et al., 2018) |
| pigmented benign keratosis | senile lentigo | yes | 46 | proposed; checked by the survey (term_check.csv) |
| pigmented benign keratosis | benign lichenoid keratosis | yes | 37 | proposed; checked by the survey (term_check.csv) |
| pigmented benign keratosis | melanoacanthoma | yes | 115 | proposed; checked by the survey (term_check.csv) |
| pigmented benign keratosis | lentigo simplex | no | 48 | proposed; checked by the survey (term_check.csv) |
| squamous cell carcinoma | intraepidermal carcinoma | yes | 68 | proposed; checked by the survey (term_check.csv) |
| squamous cell carcinoma | squamous cell carcinoma of a non-skin site | no | 0 | decision 2026-11-12 |
| squamous cell carcinoma | Bowenoid papulosis | no | 355 | proposed; checked by the survey (term_check.csv) |
| vascular lesion | vascular lesion | yes | 3247 | ISIC records of the image dataset |
| vascular lesion | cherry angioma | yes | 63 | HAM10000 group definition (Tschandl et al., 2018) |
| vascular lesion | skin hemorrhage | yes | 53 | HAM10000 group definition (Tschandl et al., 2018) |
| vascular lesion | senile angioma | yes | 14 | proposed; checked by the survey (term_check.csv) |
| vascular lesion | Campbell de Morgan spot | yes | 3 | proposed; checked by the survey (term_check.csv) |

### How much PubMed holds per class (2026-11-13)

Estimates for planning, not the dataset: a MeSH heading can be wider than the class (for example a heading for all lentigines), and no case report has been read yet. "Free full text" means the full text can be read free of charge, which is wider than a licence that allows reuse. The cases themselves come later from PMC-Patients, which holds only articles whose licence allows reuse.

| Class | Case reports | Case reports, free full text | Case reports, free full text, 2025 or later | Reviews and guidelines, free full text |
|---|---|---|---|---|
| actinic keratosis | 156 | 37 | 5 | 103 |
| basal cell carcinoma | 4040 | 619 | 41 | 345 |
| dermatofibroma | 698 | 90 | 3 | 10 |
| melanoma | 11598 | 2701 | 195 | 3186 |
| nevus | 4222 | 616 | 28 | 173 |
| pigmented benign keratosis | 327 | 63 | 4 | 8 |
| squamous cell carcinoma | 4787 | 926 | 69 | 452 |
| vascular lesion | 440 | 81 | 0 | 8 |

### The MeSH headings used for each class

| Class | Headings (as main topics) |
|---|---|
| actinic keratosis | Keratosis, Actinic |
| basal cell carcinoma | Basal Cell Carcinoma |
| dermatofibroma | Histiocytoma, Benign Fibrous |
| melanoma | Melanoma; Hutchinson's Melanotic Freckle; Melanoma, Amelanotic |
| nevus | Nevus; Nevus, Pigmented; Dysplastic Nevus Syndrome; Nevus, Epithelioid and Spindle Cell; Nevus, Blue; Nevus, Halo; Nevus, Intradermal |
| pigmented benign keratosis | Keratosis, Seborrheic |
| squamous cell carcinoma | Carcinoma, Squamous Cell; Bowen's Disease; Cutaneous Squamous Cell Carcinoma |
| vascular lesion | Angiokeratoma; Granuloma, Pyogenic |

### The queries

Each count is the class's headings AND one of these conditions:

- case reports: `("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND "case reports"[Publication Type] AND english[Language]`
- case reports, free full text: `("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND "case reports"[Publication Type] AND english[Language] AND free full text[sb]`
- case reports, free full text, 2025 or later: `("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND "case reports"[Publication Type] AND english[Language] AND free full text[sb] AND 2025:3000[Date - Publication]`
- reviews and guidelines, free full text: `("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND (review[Publication Type] OR guideline[Publication Type] OR "practice guideline"[Publication Type]) AND english[Language] AND free full text[sb]`

### Not done in this step

- No article, abstract or case text is downloaded, and nothing is chosen for the dataset.
- The names in `class_terms.csv` are not changed: a person decides what to do with the names MeSH does not have.

<!-- step: cases -->
## Cases: candidate patient cases for each class

Written by `python -m src.text.prepare cases`. The cases come from PMC-Patients (Zhao et al., Scientific Data 10, 909, 2023): patient descriptions taken from published case reports in PubMed Central whose licence allows reuse. A patient is a candidate when the title of the article it comes from names one of the classes, by the names that count in `class_terms.csv` (a case report's title usually names its diagnosis). Nothing is chosen, read or cut yet: the marks show where a person should look first. The case texts stay on this computer; `case_candidates.csv` holds only each patient's numbers, publication year, title, age, sex, length, how often the case text names the class (by a full name and by its short form) and marks.

### Source

- File: `PMC-Patients-V2.json` from the Hugging Face repository `zhengyun21/PMC-Patients`, version (commit) `28d8836518f86d4f1e6358ea8ec09977023e5766`
- Size: 837264372 bytes; SHA-256 `7bf27ed02dbadd5501f9c89b1d31d8c07d3559cd9b30c6f0b1a84734125ed593`; downloaded 2026-11-12T09:22:27+00:00; licence CC BY-NC-SA 4.0
- Patients read: 250294; fields: PMID, age, file_path, gender, patient, patient_id, patient_uid, pub_date, title
- Publication years of the candidates (from `pub_date`): 1988–2024; candidates without a year: 0

### Candidates per class

- Candidates: 3796 patients from 3343 articles; titles naming only names that do not count (for example keratoacanthoma), so no candidate: 2304

| Class | Candidates | Articles | Without marks | Names two or more classes | Also names a name that does not count | A site that may not be skin | Title about treatment | Case text does not name the class | Case text names the class only by its short form | Years |
|---|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 28 | 20 | 11 | 10 | 7 | 4 | 0 | 2 | 10 | 2009–2023 |
| basal cell carcinoma | 317 | 278 | 136 | 28 | 19 | 89 | 36 | 30 | 155 | 2005–2024 |
| dermatofibroma | 60 | 54 | 27 | 3 | 0 | 18 | 0 | 25 | 0 | 2004–2024 |
| melanoma | 1461 | 1256 | 253 | 69 | 16 | 970 | 371 | 120 | 0 | 1988–2024 |
| nevus | 223 | 199 | 74 | 40 | 4 | 66 | 1 | 45 | 0 | 2006–2024 |
| pigmented benign keratosis | 69 | 61 | 31 | 12 | 3 | 19 | 0 | 11 | 12 | 2008–2024 |
| squamous cell carcinoma | 1539 | 1381 | 254 | 24 | 17 | 1134 | 227 | 212 | 520 | 2002–2024 |
| vascular lesion | 194 | 174 | 42 | 3 | 8 | 88 | 1 | 93 | 0 | 2008–2024 |

A patient whose title names two or more classes is counted under each of them (its class is empty in `case_candidates.csv` until a person decides).

Words in a title that point to a site other than the skin (such a case is marked, not dropped): adrenal, anal, anorectal, anus, appendiceal, appendix, biliary, bladder, bone, bowel, brain, breast, bronchial, bronchogenic, bronchus, buccal, caecum, cardiac, cecal, cecum, central nervous system, cerebellopontine, cerebral, cervix, choroid, choroidal, ciliary, colon, colonic, colorectal, conjunctiva, conjunctival, cornea, corneal, duodenal, duodenum, dural, endobronchial, endometrial, endometrium, esophageal, esophagogastric, esophagus, ethmoid, eye, gallbladder, gastric, gastroesophageal, gastrointestinal, gingiva, gingival, head and neck, heart, hepatic, hypopharyngeal, hypopharynx, ileal, ileum, intestinal, intestine, intracranial, intradural, intramedullary, intranasal, intraocular, intraoral, intraosseous, intrauterine, intravascular, intravenous, intussusception, iris, jejunal, jejunum, kidney, lacrimal, laryngeal, larynx, leptomeningeal, lip, liver, lung, lymph node, mandible, mandibular, maxilla, maxillary, mediastinal, mediastinum, meningeal, meninges, metastases, metastasis, metastatic, mouth, mucosa, mucosal, nasal cavity, nasopharyngeal, nasopharynx, nodal, ocular, odontogenic, oral, orbit, orbital, oropharyngeal, oropharynx, ovarian, ovary, palatal, palate, pancreas, pancreatic, paranasal, paratracheal, parotid, pelvic, penile, penis, perianal, peritoneal, pharyngeal, pharynx, pituitary, pleural, prostate, prostatic, pulmonary, rectal, rectum, renal, retina, retinal, salivary, sigmoid, sinonasal, sphenoid, spinal, spine, spleen, splenic, stomach, submandibular, testicular, thoracic, thymic, thymus, thyroid, tongue, tonsil, tonsillar, trachea, tracheal, urethra, urethral, urinary, uterine, uterus, uvea, uveal, vagina, vaginal, vulva, vulvar.

Words in a title that show it is about treatment (such a case is marked, not dropped): adjuvant, adverse, blockade, brachytherapy, carboplatin, cemiplimab, cetuximab, chemoradiation, chemoradiotherapy, chemotherapy, cisplatin, dabrafenib, imatinib, immunotherapy, inhibition, inhibitor, ipilimumab, neoadjuvant, nivolumab, paclitaxel, pembrolizumab, radiosurgery, radiotherapy, sonidegib, talimogene, toxicity, trametinib, vemurafenib, vismodegib.

Short forms counted in the case texts (a case text that uses the short form names its class): actinic keratosis: AK; basal cell carcinoma: BCC; pigmented benign keratosis: SK; squamous cell carcinoma: SCC, cSCC.

### Not done in this step

- No case is read, chosen or cut, and no class is final: a person confirms each case's diagnosis in a later step.
- No licence is looked up yet; that is done for the cases chosen.

<!-- step: abstracts -->
## Abstracts: case reports PubMed holds as an abstract

Written by `python -m src.text.prepare abstracts`. Most case reports are in PubMed only as an abstract, without open full text. For every class, PubMed was asked for its case reports (the class's headings as a main subject, on the skin, in English) and, by the class's names in the title, for the case reports it has not indexed (journals outside MEDLINE have no subject headings, so the first search cannot find them); the articles PMC-Patients already gave were left out, and the titles and abstracts of the others were downloaded. Each abstract's case part (its sections whose label names the case or the patient, or else the whole abstract) is judged by the rules of `cases`; for a candidate article (its title passes the rules) with an open full text in PubMed Central, of any year, the case sections of the full text (titled as the case, such as "Case presentation"; figures and tables left out) are used instead of the abstract. The texts stay in `data/text/sources/pubmed/`; `abstract_candidates.csv` holds only each article's numbers, year, title, length and marks.

- Asked on 2026-11-13: 30368 case reports found; already given by PMC-Patients (left out): 2548; downloaded: 27820 (not returned by PubMed: 0); without an abstract: 10037
- Candidate articles (title passes) with a PMC number, asked from PMC: 1382; with an open full text and a case section (used instead of the abstract): 771
- Candidates: 12686; titles naming only names that do not count, so no candidate: 1632

| Class | MeSH headings (main topic) | Case reports in PubMed | Of them not indexed (found by title) | Already from PMC-Patients | With a case text | Of them from a PMC full text | Candidates (title names only this class) | Without marks |
|---|---|---|---|---|---|---|---|---|
| actinic keratosis | Keratosis, Actinic | 184 | 28 | 15 | 80 | 6 | 41 | 32 |
| basal cell carcinoma | Basal Cell Carcinoma | 4531 | 516 | 223 | 2509 | 136 | 1458 | 852 |
| dermatofibroma | Histiocytoma, Benign Fibrous | 757 | 71 | 31 | 444 | 16 | 159 | 119 |
| melanoma | Melanoma; Hutchinson's Melanotic Freckle; Melanoma, Amelanotic | 13716 | 2337 | 1061 | 8356 | 236 | 6149 | 1630 |
| nevus | Nevus; Nevus, Pigmented; Dysplastic Nevus Syndrome; Nevus, Epithelioid and Spindle Cell; Nevus, Blue; Nevus, Halo; Nevus, Intradermal | 4793 | 584 | 132 | 2818 | 58 | 1081 | 586 |
| pigmented benign keratosis | Keratosis, Seborrheic | 393 | 66 | 36 | 188 | 17 | 113 | 80 |
| squamous cell carcinoma | Carcinoma, Squamous Cell; Bowen's Disease; Cutaneous Squamous Cell Carcinoma | 7153 | 2467 | 1001 | 4210 | 268 | 2866 | 1082 |
| vascular lesion | Angiokeratoma; Granuloma, Pyogenic | 719 | 283 | 125 | 358 | 36 | 327 | 145 |

Queries for each class: its headings as a main subject AND ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND "case reports"[Publication Type] AND english[Language] NOT (animals[MeSH Terms] NOT humans[MeSH Terms]); and its names that count (and that some PubMed title uses) in the title AND ("case reports"[Publication Type] OR "case report"[Title/Abstract]) NOT medline[sb] AND english[Language].

### Not done in this step

- No abstract or full text is written into the repository: they are under their publishers' copyright or licences, so they stay on this computer and are used for research only (each full text's licence is kept with it).

<!-- step: mesh -->
## MeSH: the subject headings of each candidate article

Written by `python -m src.text.prepare mesh`. For every article in `case_candidates.csv`, PubMed was asked whether it has subject headings at all (`medline[sb]`: PubMed indexes the articles of MEDLINE journals) and which classes they name (one of the headings of the class's names that count in `term_check.csv`, or a narrower heading) while it is also indexed under skin neoplasms or skin diseases. The US National Library of Medicine gives these headings, independently of the authors' titles and of this code. In the build, the headings of an indexed article must name its class; an article without headings is judged by its title and case text.

- Asked on 2026-11-13: 16029 articles, 10267 of them indexed; article numbers PubMed does not know: 0

| Class | MeSH headings | Candidate articles (title names only this class) | Indexed | Indexed, headings name the class on the skin | Indexed, headings do not | Not indexed |
|---|---|---|---|---|---|---|
| actinic keratosis | Keratosis, Actinic | 55 | 32 | 30 | 2 | 23 |
| basal cell carcinoma | Basal Cell Carcinoma | 1719 | 1239 | 1237 | 2 | 480 |
| dermatofibroma | Histiocytoma, Benign Fibrous | 210 | 134 | 127 | 7 | 76 |
| melanoma | Melanoma; Hutchinson's Melanotic Freckle; Melanoma, Amelanotic | 7345 | 5330 | 5314 | 16 | 2015 |
| nevus | Nevus; Nevus, Pigmented; Dysplastic Nevus Syndrome; Nevus, Epithelioid and Spindle Cell; Nevus, Blue; Nevus, Halo; Nevus, Intradermal | 1243 | 980 | 962 | 18 | 263 |
| pigmented benign keratosis | Keratosis, Seborrheic | 163 | 98 | 88 | 10 | 65 |
| squamous cell carcinoma | Carcinoma, Squamous Cell; Bowen's Disease; Cutaneous Squamous Cell Carcinoma | 4225 | 1812 | 1560 | 252 | 2413 |
| vascular lesion | Angiokeratoma; Granuloma, Pyogenic | 498 | 207 | 192 | 15 | 291 |

Skin: ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]).

### Not done in this step

- Nothing is labelled yet: the build step uses these headings to check the class of every indexed article.

<!-- step: build -->
## Build: the text dataset

Written by `python -m src.text.prepare build`, with no person choosing, labelling or cutting a case. Each case is a real patient description from a published case report: a patient of PMC-Patients, the case sections of an open full text in PMC, or the case part of a PubMed abstract (`source` in `dataset.csv`). The task is the image dataset's: name the class. `dataset.csv` lists the cases; the question texts are in `data/text/dataset/` (not in Git: research use only), and `build` writes them again, the same, from the downloaded files.

### Rules

- Label: a case gets class X when the article title names X and no other class, with no mark (`cases`, `abstracts`), and the case text names X, by a full name or its short form; where PubMed has indexed the article, its subject headings must also name X on the skin (`mesh`), else the case is left out. `dataset.csv` says for every case whether PubMed agrees (three sources) or has not indexed the article (two sources).
- Question: the case text word for word, up to the first sentence that names the case's own class (any of its names in `class_terms.csv`, counting or not; except mole, too vague to be one), its short form (AK, BCC, SK, SCC, cSCC, or one the article defines, such as "pyogenic granuloma (PG)"), or that reports a test result (words beginning biops, histopatholog, histolog, patholog, dermatopatholog, immunohistochem, histochem, immunostain, immunoreactiv, excis, microscop, cytolog, cytoplasm, nucle, mitos, mitot, specimen, multinucleat, fibroblastic, histiocytic, aspirat, dermis, epidermis, collagen, macrophag, storiform, spindle, epithelioid, keratinocyt, lymphocyt, atypia). It starts at the start of the case text, or, when a sentence before the first sentence about the patient already names the class (a background), at that first sentence about the patient. Names of the other classes stay, as clinical information (a history, a suspicion). Nothing is rewritten.
- Kept when the question has at least 25 words, describes a lesion (lesion, nodule, papule, plaque, patch, patches, macule, mass, masses, tumour, ulcer, growth, spot) and a patient (an age or a sex) and what the lesion looks like (a size in mm or cm, a lesion type: nodule, papule, plaque, patch, patches, macule, ulcer, spot, or a colour, surface or shape: red, reddish, pink, pinkish, brown, brownish, black, blackish, blue, bluish, purple, purplish, violet, violaceous, white, whitish, yellow, yellowish, grey, gray, greyish, grayish, tan, dark, erythematous, pigmented, hyperpigmented, hypopigmented, depigmented, colored, coloured, pearly, translucent, shiny, scaly, crusted, crusting, ulcerated, ulcerating, eroded, verrucous, warty, keratotic, hyperkeratotic, papillomatous, pedunculated, exophytic, nodular, polypoid, polypoidal, fungating, raised, elevated, flat, depressed, atrophic, firm, hard, soft, smooth, rough, friable, bleeding, hemorrhagic, lobulated, annular, irregular, circumscribed, demarcated, indurated, cystic, compressible), and has no sentence that puts a lesion inside the body (intraoral, gingiva, gingival, gingivae, palate, palatal, retromolar, alveolar ridge, alveolar mucosa, oral cavity, oral mucosa, buccal mucosa, labial mucosa, floor of the mouth, tongue, nasal cavity, nasal mucosa, conjunctiva, conjunctival, vulva, vulvar, anogenital, anal canal, choroid, choroidal, uvea, uveal, iris, retina, retinal, intraocular, mediastinum, mediastinal, cervix, middle ear, eustachian tube, tympanic, nasopharynx). When the questions of two articles share at least 10% of the shorter one's runs of 12 words, they tell the same patient's case: the article PubMed listed first (the lower PubMed number) keeps it.
- Splits: every article whole in one split, in a fixed random order (seed `text-dataset`): for each class, 60% of its cases into test (at most 40), 20% into validation (at most 15), the rest into train; an article that would pass a class's share goes into the next split.

### Cases per class

- Source: `PMC-Patients-V2.json`, version `28d8836518f86d4f1e6358ea8ec09977023e5766`, SHA-256 `7bf27ed02dbadd5501f9c89b1d31d8c07d3559cd9b30c6f0b1a84734125ed593` (checked)

| Class | Candidates (title names only this class) | Without marks | PubMed's headings disagree (left out) | Usable question | In the dataset | From PMC full texts | From PubMed abstracts | PubMed agrees | PubMed has not indexed it | Test | Validation | Train | Question words (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 59 | 43 | 1 | 10 | 10 | 3 | 1 | 2 | 8 | 6 | 2 | 2 | 120 |
| basal cell carcinoma | 1747 | 988 | 1 | 212 | 212 | 81 | 40 | 55 | 157 | 40 | 15 | 157 | 87.5 |
| dermatofibroma | 216 | 146 | 1 | 45 | 45 | 7 | 15 | 16 | 29 | 27 | 9 | 9 | 72 |
| melanoma | 7541 | 1883 | 8 | 261 | 261 | 97 | 51 | 93 | 168 | 40 | 15 | 206 | 77 |
| nevus | 1264 | 660 | 2 | 102 | 102 | 18 | 35 | 52 | 50 | 40 | 15 | 47 | 67 |
| pigmented benign keratosis | 170 | 111 | 2 | 47 | 47 | 16 | 10 | 15 | 32 | 28 | 9 | 10 | 90 |
| squamous cell carcinoma | 4381 | 1336 | 21 | 326 | 325 | 132 | 62 | 79 | 246 | 40 | 15 | 270 | 99 |
| vascular lesion | 518 | 187 | 4 | 47 | 47 | 13 | 7 | 8 | 39 | 28 | 9 | 10 | 124 |
| all | 15896 | 5354 | 40 | 1050 | 1049 | 367 | 221 | 320 | 729 | 249 | 89 | 711 | 85 |

### Checks

- Questions that would still name their class across two sentences (left out): 0
- Questions identical to an earlier one (left out): 0

| Class | No patient described (left out) | A lesion inside the body (left out) | Nothing of what the lesion looks like (left out) | The same patient as an earlier article (left out) |
|---|---|---|---|---|
| actinic keratosis | 3 | 0 | 0 | 0 |
| basal cell carcinoma | 22 | 8 | 27 | 0 |
| dermatofibroma | 2 | 0 | 2 | 0 |
| melanoma | 43 | 21 | 49 | 0 |
| nevus | 13 | 7 | 7 | 0 |
| pigmented benign keratosis | 4 | 1 | 1 | 0 |
| squamous cell carcinoma | 45 | 35 | 44 | 1 |
| vascular lesion | 4 | 7 | 3 | 0 |

- Articles in more than one split: 0
- Every question is part of its case text, word for word: yes

### Not done in this step

- No person has checked the labels: each rests on sources that agree (the authors' title and the case text, and PubMed's subject headings where the article is indexed), not on a dermatologist's review.
- Cases come from articles published from 1977 to 2026 (the year is in `dataset.csv`): a model may have read them in training.
- The reference library for retrieval is built by `python -m src.text.library`, not here.

<!-- step: audit -->
## Audit: an independent check of the questions

Written by `python -m src.text.audit`, on the computer that holds the texts, with code of its own (not the build's): every question file is checked against its fingerprint and its source, and for the class's names, short forms (also those its source defines) and test results it must not contain, and for a patient and a lesion on the skin it must describe. 5 questions per class are written into `data/text/sources/audit_sample.txt` for reading (not committed).

- Questions checked: 1049

| Check | Failed | Examples |
|---|---|---|
| question file missing | 0 |  |
| question file differs from its fingerprint | 0 |  |
| file in the dataset folder but not in dataset.csv | 0 |  |
| question not word for word in its source | 0 |  |
| question names its class or its short form | 0 |  |
| question uses a short form its source defines for the class | 0 |  |
| question reports a test result | 0 |  |
| question shorter than 25 words | 0 |  |
| question describes no lesion | 0 |  |
| question describes no patient | 0 |  |
| question puts a lesion inside the body | 0 |  |
| question tells nothing of what the lesion looks like | 0 |  |
| question the same as another | 0 |  |
| question tells the same patient's case as a question of another article (at least 10% of its runs of 12 words) | 0 |  |

All checks passed.

<!-- step: library -->
## Library: the reference texts for retrieval

Written by `python -m src.text.library` on 2026-11-13. Four corpora, as in RAG² (PubMed, PMC, clinical practice guidelines, textbooks), from public sources of the US National Library of Medicine, found by fixed rules with no person choosing a text: the abstracts of PubMed's reviews about each class (its headings as a main subject, on the skin, in English, about people; not case reports or guidelines; the newest 200 per class); the open full texts of those reviews in PubMed Central (per class the newest 25 that PMC gives in full); PubMed's clinical practice guidelines about each class (the newest 50 per class, as open full text where PMC gives it, else as abstract); and the chapters of StatPearls (NCBI's Bookshelf) whose title or summary (its "Continuing Education Activity") names a class by a name that counts in `class_terms.csv` and whose title names no other site and no animal. Every text is cut word for word into passages of whole paragraphs of at most 300 words. `library.csv` lists the passages (source, licence, fingerprint); their texts are in `data/text/library/` (not in Git).

### Per class

| Class | MeSH headings (main topic) | Reviews (abstracts) | Open full texts of reviews | Guidelines | StatPearls chapters |
|---|---|---|---|---|---|
| actinic keratosis | Keratosis, Actinic | 200 | 25 | 14 | 11 |
| basal cell carcinoma | Basal Cell Carcinoma | 200 | 25 | 32 | 20 |
| dermatofibroma | Histiocytoma, Benign Fibrous | 35 | 0 | 0 | 3 |
| melanoma | Melanoma; Hutchinson's Melanotic Freckle; Melanoma, Amelanotic | 200 | 25 | 50 | 31 |
| nevus | Nevus; Nevus, Pigmented; Dysplastic Nevus Syndrome; Nevus, Epithelioid and Spindle Cell; Nevus, Blue; Nevus, Halo; Nevus, Intradermal | 200 | 25 | 6 | 13 |
| pigmented benign keratosis | Keratosis, Seborrheic | 33 | 2 | 0 | 11 |
| squamous cell carcinoma | Carcinoma, Squamous Cell; Bowen's Disease; Cutaneous Squamous Cell Carcinoma | 200 | 25 | 44 | 28 |
| vascular lesion | Angiokeratoma; Granuloma, Pyogenic | 18 | 2 | 0 | 10 |

### Per corpus

| Corpus | Documents | Passages | Words | Licences (documents) |
|---|---|---|---|---|
| PubMed: abstracts of reviews | 953 | 1012 | 183440 | publisher's copyright (abstract in PubMed): 953 |
| PMC: open full texts of reviews | 124 | 2595 | 548311 | http://creativecommons.org/licenses/by/4.0/: 1; https://creativecommons.org/licenses/by-nc-nd/4.0/: 15; https://creativecommons.org/licenses/by-nc/4.0/: 19; https://creativecommons.org/licenses/by-nd/4.0/: 1; https://creativecommons.org/licenses/by/4.0/: 83; not stated: 5 |
| Clinical practice guidelines | 99 | 493 | 99080 | http://creativecommons.org/licenses/by/4.0/: 1; https://creativecommons.org/licenses/by-nc-nd/4.0/: 2; https://creativecommons.org/licenses/by-nc/4.0/: 4; https://creativecommons.org/licenses/by/4.0/: 6; not stated: 2; publisher's copyright (abstract in PubMed): 84 |
| Textbook: StatPearls chapters | 91 | 1453 | 242863 | https://creativecommons.org/licenses/by-nc-nd/4.0/: 91 |
| all | 1267 | 5553 | 1073694 | |

### Checks

- Case reports and the dataset's candidate articles are not searched for or are left out, so no case of the dataset is in the library.
- Passages left out because they share a run of 12 words with a question of the dataset: 1
- Passages left out because they are the same as an earlier one: 2

Query for reviews, with each class's headings as a main subject: ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND review[Publication Type] NOT (guideline[Publication Type] OR "practice guideline"[Publication Type]) NOT "case reports"[Publication Type] AND english[Language] NOT (animals[MeSH Terms] NOT humans[MeSH Terms]).
Query for guidelines: ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND (guideline[Publication Type] OR "practice guideline"[Publication Type]) NOT "case reports"[Publication Type] AND english[Language] NOT (animals[MeSH Terms] NOT humans[MeSH Terms]).

Textbook: StatPearls (NBK430685) from the Bookshelf's Open Access subset, `statpearls_NBK430685.tar.gz`, 2047364463 bytes, SHA-256 `08117bcff46ed366a7ab9033d134b81bf345e81cb9a1537b6075400418bb09bd` (checked), downloaded 2026-11-13; chapters read: 9653; chapter files that could not be read: 0

### Not done in this step

- No person has read the passages. A passage is about its document's class(es) (`topics`), not checked line by line.
- The texts are under their publishers' copyright or licences (in `library.csv`), so they stay on this computer and are used for research only.
- Retrieval (how passages are chosen for a question, evenly from the four corpora as in RAG²) belongs to the evaluation, not to the library.
