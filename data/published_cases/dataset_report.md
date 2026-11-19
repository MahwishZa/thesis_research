# Preparation of the published cases: their text and their photos

Written by `python -m src.build.text.prepare` (the text of the cases: the steps survey, cases, abstracts, mesh, build, audit, review and revise) and by `python -m src.build.cases.prepare` (the photos of the same cases: the steps count, download, check and build, whose sections are named photos_count, photos_download, photos_check and photos_build here), one section per step in the order the steps run; each step rewrites only its own section. `build/class_terms.csv` is written by people: the names doctors use for each of the eight classes, and whether each name counts for that class. The build records are in `build/`; the downloads, the question files, the captions, the pictures and the models' answers stay on this computer in `downloads/` and `files/`.

<!-- step: survey -->
## Survey: the names checked in MeSH, and how much PubMed holds per class

Written by `python -m src.build.text.prepare survey` on 2026-11-14. Every name in `build/class_terms.csv` was looked up in MeSH, the US National Library of Medicine's list of medical subject headings, through NCBI's E-utilities; the result for each name is in `build/term_check.csv`, with the number of PubMed articles that use the name in their title or summary. Then PubMed was searched per class with the MeSH headings of the class's confirmed names (as main topics), limited to skin lesions. Only MeSH records and search counts were downloaded, no article text; nothing else was changed.

### MeSH check

- Names: 121; a MeSH heading: 32; an entry term of a heading (another name MeSH lists for it): 17; not in MeSH: 72 (used in PubMed titles or summaries: 69; in none: 3)

A name that is not in MeSH is not wrong: MeSH lists the names used to index articles, not every name in use (subtypes and older names are often missing). The PubMed count shows whether such a name is in use. A name in neither is for a person to confirm, change or remove in `build/class_terms.csv`.

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
| melanoma | non-melanoma | no | 5109 | found by the audit of the dataset; checked by the survey (term_check.csv) |
| nevus | compound nevus | yes | 163 | ISIC records of the image dataset |
| nevus | junctional nevus | yes | 101 | ISIC records of the image dataset |
| nevus | dermal nevus | yes | 98 | ISIC records of the image dataset |
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
| vascular lesion | acral pseudolymphomatous angiokeratoma | no | 36 | proposed after reading the text dataset's audit sample (2026-11-14) |

### How much PubMed holds per class (2026-11-14)

Estimates for planning, not the dataset: a MeSH heading can be wider than the class (for example a heading for all lentigines), and no case report has been read yet. "Free full text" means the full text can be read free of charge, which is wider than a licence that allows reuse. The cases themselves come later from PMC-Patients, which holds only articles whose licence allows reuse.

| Class | Case reports | Case reports, free full text | Case reports, free full text, 2025 or later | Reviews and guidelines, free full text |
|---|---|---|---|---|
| actinic keratosis | 156 | 37 | 5 | 103 |
| basal cell carcinoma | 4041 | 619 | 41 | 345 |
| dermatofibroma | 698 | 90 | 3 | 10 |
| melanoma | 11598 | 2701 | 195 | 3187 |
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
- The names in `build/class_terms.csv` are not changed: a person decides what to do with the names MeSH does not have.

<!-- step: cases -->
## Cases: candidate patient cases for each class

Written by `python -m src.build.text.prepare cases`. The cases come from PMC-Patients (Zhao et al., Scientific Data 10, 909, 2023): patient descriptions taken from published case reports in PubMed Central whose licence allows reuse. A patient is a candidate when the title of the article it comes from names one of the classes, by the names that count in `build/class_terms.csv` (a case report's title usually names its diagnosis). Nothing is chosen, read or cut yet: the marks show where a person should look first. The case texts stay on this computer; `build/case_candidates.csv` holds only each patient's numbers, publication year, title, age, sex, length, how often the case text names the class (by a full name and by its short form) and marks.

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

A patient whose title names two or more classes is counted under each of them (its class is empty in `build/case_candidates.csv` until a person decides).

Words in a title that point to a site other than the skin (such a case is marked, not dropped): adrenal, anal, anorectal, anus, appendiceal, appendix, biliary, bladder, bone, bowel, brain, breast, bronchial, bronchogenic, bronchus, buccal, caecum, cardiac, cecal, cecum, central nervous system, cerebellopontine, cerebral, cervix, choroid, choroidal, ciliary, colon, colonic, colorectal, conjunctiva, conjunctival, cornea, corneal, duodenal, duodenum, dural, endobronchial, endometrial, endometrium, esophageal, esophagogastric, esophagus, ethmoid, eye, gallbladder, gastric, gastroesophageal, gastrointestinal, gingiva, gingival, head and neck, heart, hepatic, hypopharyngeal, hypopharynx, ileal, ileum, intestinal, intestine, intracranial, intradural, intramedullary, intranasal, intraocular, intraoral, intraosseous, intrauterine, intravascular, intravenous, intussusception, iris, jejunal, jejunum, kidney, lacrimal, laryngeal, larynx, leptomeningeal, lip, liver, lung, lymph node, mandible, mandibular, maxilla, maxillary, mediastinal, mediastinum, meningeal, meninges, metastases, metastasis, metastatic, mouth, mucosa, mucosal, nasal cavity, nasopharyngeal, nasopharynx, nodal, ocular, odontogenic, oral, orbit, orbital, oropharyngeal, oropharynx, ovarian, ovary, palatal, palate, pancreas, pancreatic, paranasal, paratracheal, parotid, pelvic, penile, penis, perianal, peritoneal, pharyngeal, pharynx, pituitary, pleural, prostate, prostatic, pulmonary, rectal, rectum, renal, retina, retinal, salivary, sigmoid, sinonasal, sphenoid, spinal, spine, spleen, splenic, stomach, submandibular, testicular, thoracic, thymic, thymus, thyroid, tongue, tonsil, tonsillar, trachea, tracheal, urethra, urethral, urinary, uterine, uterus, uvea, uveal, vagina, vaginal, vulva, vulvar.

Words in a title that show it is about treatment (such a case is marked, not dropped): adjuvant, adverse, blockade, brachytherapy, carboplatin, cemiplimab, cetuximab, chemoradiation, chemoradiotherapy, chemotherapy, cisplatin, dabrafenib, imatinib, immunotherapy, inhibition, inhibitor, ipilimumab, neoadjuvant, nivolumab, paclitaxel, pembrolizumab, radiosurgery, radiotherapy, sonidegib, talimogene, toxicity, trametinib, vemurafenib, vismodegib.

Short forms counted in the case texts (a case text that uses the short form names its class): actinic keratosis: AK; basal cell carcinoma: BCC; pigmented benign keratosis: SK; squamous cell carcinoma: SCC, cSCC.

### Not done in this step

- No case is read, chosen or cut, and no class is final: a person confirms each case's diagnosis in a later step.
- No licence is looked up yet; that is done for the cases chosen.

<!-- step: abstracts -->
## Abstracts: case reports PubMed holds as an abstract

Written by `python -m src.build.text.prepare abstracts`. Most case reports are in PubMed only as an abstract, without open full text. For every class, PubMed was asked for its case reports (the class's headings as a main subject, on the skin, in English) and, by the class's names in the title, for the case reports it has not indexed (journals outside MEDLINE have no subject headings, so the first search cannot find them); the articles PMC-Patients already gave were left out, and the titles and abstracts of the others were downloaded. Each abstract's case part (its sections whose label names the case or the patient, or else the whole abstract) is judged by the rules of `cases`; for a candidate article (its title passes the rules) with an open full text in PubMed Central, of any year, the case sections of the full text (titled as the case, such as "Case presentation"; figures and tables left out) are used instead of the abstract. The texts stay in `data/published_cases/downloads/pubmed/`; `build/abstract_candidates.csv` holds only each article's numbers, year, title, length and marks.

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

Written by `python -m src.build.text.prepare mesh`. For every article in `build/case_candidates.csv`, PubMed was asked whether it has subject headings at all (`medline[sb]`: PubMed indexes the articles of MEDLINE journals) and which classes they name (one of the headings of the class's names that count in `build/term_check.csv`, or a narrower heading) while it is also indexed under skin neoplasms or skin diseases. The US National Library of Medicine gives these headings, independently of the authors' titles and of this code. In the build, the headings of an indexed article must name its class; an article without headings is judged by its title and case text.

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

Written by `python -m src.build.text.prepare build`, with no person choosing, labelling or cutting a case. Each case is a real patient description from a published case report: a patient of PMC-Patients, the case sections of an open full text in PMC, or the case part of a PubMed abstract (`source` in `dataset.csv`). The task: name the class. `dataset.csv` lists the cases; the question texts are in `data/published_cases/files/text/` (not in Git: research use only), and `build` writes them again, the same, from the downloaded files.

### Rules

- Label: a case gets class X when the article title names X and no other class, with no mark (`cases`, `abstracts`; their title rules are applied again here, as they are now, so that a rule or a name added since counts: among them a title that names another lesion, a word ending in -oma or a polyp that is no name in `build/class_terms.csv`, unless named only as what the lesion looked like), and the case text names X, by a full name or its short form; where PubMed has indexed the article, its subject headings must also name X on the skin (`mesh`), else the case is left out. `dataset.csv` says for every case whether PubMed agrees (three sources) or has not indexed the article (two sources). The case text after the question must also report a test result that names X (a sentence with a word beginning biops, histopatholog, histolog, patholog, dermatopatholog, immunohistochem ... that names X by a name that counts or its short form, or one of whose two next sentences does; a sentence naming a relative (father, mother, brother, sister, son, daughter, sibling, parent, grandfather, grandmother, grandparent, aunt, uncle, cousin, niece, nephew, twin, husband, wife, family, relative) does not count): a diagnosis by eye or by dermoscopy alone, a result that names another diagnosis, or a group of diseases (keratinocyte carcinoma, keratinocyte cancer, keratinocytic carcinoma) is not enough. A short form the article defines names X in a result when a name of X ends right before its brackets ("nodular basal cell carcinoma (nBCC)").
- Question: the case text word for word, up to the first sentence that names the case's own class (any of its names in `build/class_terms.csv`, counting or not; except mole, too vague to be one), its short form (AK, BCC, SK, SCC, cSCC, or one the article defines, such as "pyogenic granuloma (PG)"), or that reports a test result (words beginning biops, histopatholog, histolog, patholog, dermatopatholog, immunohistochem, histochem, immunostain, immunoreactiv, excis, microscop, cytolog, cytoplasm, nucle, mitos, mitot, specimen, multinucleat, fibroblastic, histiocytic, aspirat, dermis, epidermis, collagen, macrophag, storiform, spindle, epithelioid, keratinocyt, lymphocyt, atypia). It starts at the start of the case text, or, when a sentence before the first sentence about the patient already names the class (a background), at that first sentence about the patient. Names of the other classes stay, as clinical information (a history, a suspicion). Nothing is rewritten.
- Kept when the question has at least 25 words, describes a lesion (lesion, nodule, papule, plaque, patch, patches, macule, mass, masses, tumour, ulcer, growth, spot) and one patient (an age or a sex; not "patients", a mean or median age, a second case, or two different ages of a man, woman, boy, girl, male, female, patient, gentleman or lady; a heading such as "Case 2." at its start does not count), describes a lesion as seen on the skin (one sentence with a lesion word, or mole, lump, swelling, rash, rashes, eruption, wound, stain, and the skin (skin, cutaneous, subcutaneous, dermal, dermatology, dermatological, dermoscopy, dermoscopic, dermatoscopy, dermatoscopic) or a place on the body's surface (scalp, vertex, face, facial, forehead, glabella, temple, temporal, frontal, parietal, occipital, cheek, nose, nasal ala, nasal tip, nasal bridge, nasal dorsum, nasal sidewall, nasal root, ear, pinna, auricle, auricular, earlobe, helix, preauricular, postauricular, retroauricular, eyelid, lid, canthus, eyebrow, periorbital, lip, perioral, philtrum, chin, jaw, neck, nape, nuchal, shoulder, axilla, axillae, axillary, arm, forearm, elbow, wrist, hand, palm, palmar, finger, thumb, nail, fingernail, toenail, subungual, periungual, digit, back, chest, breast, nipple, areola, abdomen, abdominal, trunk, flank, umbilicus, umbilical, groin, inguinal, pubic, scrotum, scrotal, penis, perineum, perineal, buttock, gluteal, sacral, hip, thigh, leg, knee, shin, pretibial, calf, ankle, malleolus, foot, feet, sole, plantar, heel, toe, limb, extremity, scapula, scapular, supraclavicular, sternal, presternal, lumbar, dorsum, prepuce), with no scan or look inside the body (ct, computed tomography, tomography, mri, magnetic resonance, ultrasound, ultrasonography, sonography, radiograph, radiography, x ray, pet, scan, scintigraphy, imaging, endoscopy, cystoscopy, colonoscopy, bronchoscopy, laryngoscopy, laparotomy, laparoscopy); and what it looks or feels like: in that sentence the skin words, pigment, papule, plaque, macule, patch, patches, or the colours, surfaces and shapes below; in the sentence before or after it, the same but for the skin words), describes what the lesion looks like (a size in mm or cm, a lesion type: nodule, papule, plaque, patch, patches, macule, ulcer, spot, or a colour, surface or shape: red, reddish, pink, pinkish, brown, brownish, black, blackish, blue, bluish, purple, purplish, violet, violaceous, white, whitish, yellow, yellowish, grey, gray, greyish, grayish, tan, dark, erythematous, pigmented, hyperpigmented, hypopigmented, depigmented, colored, coloured, pearly, translucent, shiny, scaly, crusted, crusting, ulcerated, ulcerating, eroded, verrucous, warty, keratotic, hyperkeratotic, papillomatous, pedunculated, exophytic, nodular, polypoid, polypoidal, fungating, raised, elevated, flat, depressed, atrophic, firm, hard, soft, smooth, rough, friable, bleeding, hemorrhagic, lobulated, annular, irregular, circumscribed, demarcated, indurated, cystic, compressible), and has no sentence that puts a lesion inside the body (intraoral, gingiva, gingival, gingivae, palate, palatal, retromolar, alveolar ridge, alveolar mucosa, oral cavity, oral mucosa, buccal mucosa, labial mucosa, floor of the mouth, tongue, nasal cavity, nasal mucosa, conjunctiva, conjunctival, vulva, vulvar, anogenital, anal canal, choroid, choroidal, uvea, uveal, iris, retina, retinal, intraocular, mediastinum, mediastinal, cervix, middle ear, eustachian tube, tympanic, nasopharynx). When the questions of two articles share at least 10% of the shorter one's runs of 12 words, they tell the same patient's case: the article PubMed listed first (the lower PubMed number) keeps it.
- Splits: every article whole in one split, in a fixed random order (seed `text-dataset`): for each class, 60% of its cases into test (at most 40), 20% into validation (at most 15), the rest into train; an article that would pass a class's share goes into the next split.

### Cases per class

- Source: `PMC-Patients-V2.json`, version `28d8836518f86d4f1e6358ea8ec09977023e5766`, SHA-256 `7bf27ed02dbadd5501f9c89b1d31d8c07d3559cd9b30c6f0b1a84734125ed593` (checked)

| Class | Candidates (title names only this class) | Without marks | PubMed's headings disagree (left out) | Usable question | In the dataset | From PMC full texts | From PubMed abstracts | PubMed agrees | PubMed has not indexed it | Test | Validation | Train | Question words (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 59 | 37 | 1 | 4 | 4 | 1 | 1 | 1 | 3 | 2 | 1 | 1 | 107 |
| basal cell carcinoma | 1747 | 911 | 0 | 176 | 176 | 69 | 27 | 43 | 133 | 40 | 15 | 121 | 95 |
| dermatofibroma | 216 | 138 | 1 | 35 | 35 | 6 | 10 | 11 | 24 | 21 | 7 | 7 | 75 |
| melanoma | 7541 | 1792 | 8 | 180 | 180 | 70 | 29 | 63 | 117 | 40 | 15 | 125 | 78 |
| nevus | 1264 | 620 | 2 | 79 | 79 | 14 | 27 | 40 | 39 | 40 | 15 | 24 | 74 |
| pigmented benign keratosis | 170 | 100 | 1 | 36 | 36 | 13 | 6 | 10 | 26 | 22 | 7 | 7 | 97 |
| squamous cell carcinoma | 4381 | 1224 | 16 | 209 | 208 | 83 | 38 | 49 | 159 | 40 | 15 | 153 | 106 |
| vascular lesion | 518 | 168 | 4 | 27 | 27 | 8 | 1 | 3 | 24 | 16 | 5 | 6 | 151 |
| all | 15896 | 4990 | 33 | 746 | 745 | 264 | 139 | 220 | 525 | 221 | 80 | 444 | 90 |

### Checks

- Questions that would still name their class across two sentences (left out): 0
- Questions identical to an earlier one (left out): 0

Left out, per class (each case counted for the first rule it fails, in this order):

| Class | No patient described | More than one patient | A lesion inside the body | No lesion seen on the skin | Nothing of what the lesion looks like | No test result names the class | The same patient as an earlier article |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 2 | 0 | 0 | 0 | 0 | 4 | 0 |
| basal cell carcinoma | 21 | 10 | 5 | 28 | 5 | 5 | 0 |
| dermatofibroma | 1 | 1 | 0 | 6 | 1 | 4 | 0 |
| melanoma | 41 | 8 | 21 | 80 | 10 | 5 | 0 |
| nevus | 11 | 5 | 6 | 10 | 2 | 6 | 0 |
| pigmented benign keratosis | 3 | 0 | 1 | 4 | 0 | 3 | 0 |
| squamous cell carcinoma | 42 | 11 | 31 | 95 | 9 | 11 | 1 |
| vascular lesion | 3 | 3 | 6 | 8 | 0 | 8 | 0 |

- Articles in more than one split: 0
- Every question is part of its case text, word for word: yes

### Not done in this step

- No person has checked the labels: each rests on sources that agree (the authors' title and the case text, and PubMed's subject headings where the article is indexed), not on a dermatologist's review.
- Cases come from articles published from 1977 to 2026 (the year is in `dataset.csv`): a model may have read them in training.
- The reference library for retrieval is built by `python -m src.build.knowledge.library`, not here.

<!-- step: audit -->
## Audit: an independent check of the questions

Written by `python -m src.build.text.audit`, on the computer that holds the texts, with code of its own (not the build's): every question file is checked against its fingerprint and its source, and for the class's names, short forms (also those its source defines) and test results it must not contain, for one patient and a lesion on the skin it must describe, and for a test result after it in its source that names its class (a confirmed diagnosis). 10 questions per class, each with the text after it in its source, are written into `data/published_cases/downloads/audit_sample.txt` for reading (not committed).

- Questions checked: 716

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
| question describes more than one patient | 0 |  |
| question describes no lesion as seen on the skin | 0 |  |
| question puts a lesion inside the body | 0 |  |
| question tells nothing of what the lesion looks like | 0 |  |
| no test result after the question names its class | 0 |  |
| question the same as another | 0 |  |
| question tells the same patient's case as a question of another article (at least 10% of its runs of 12 words) | 0 |  |

All checks passed.

<!-- step: review -->
## Review: signals of a doubtful label or of leakage

Written by `python -m src.build.text.audit review`, with rules fixed before it was first run (see `src/build/text/audit.py`). It changes nothing: it gives the evidence for deciding on a new version. One row per question in `build/review_flags.csv`. Closed-book test runs read: 4.

| Signal | Test | Validation | Train | All |
|---|---|---|---|---|
| the confirming result also names another class (not negated) | 12 | 2 | 15 | 29 |
| look-alike wording around the confirming result | 11 | 2 | 38 | 51 |
| the question names another class | 16 | 2 | 52 | 70 |
| every closed-book run gave the same wrong answer | 20 | 0 | 0 | 20 |
| a knowledge-base passage shares 12 words in a row | 0 | 0 | 0 | 0 |

On the test split, the questions every run got wrong in the same way, and what the review finds for them:

- with a confirming result that also names another class: 2
- with look-alike wording: 0
- with neither: 18

| Class | Test questions | Confirming result names another class | Look-alike wording | Same wrong answer |
|---|---|---|---|---|
| actinic keratosis | 2 | 0 | 0 | 1 |
| basal cell carcinoma | 40 | 0 | 1 | 7 |
| dermatofibroma | 21 | 1 | 1 | 2 |
| melanoma | 40 | 3 | 2 | 2 |
| nevus | 40 | 1 | 1 | 4 |
| pigmented benign keratosis | 22 | 4 | 3 | 1 |
| squamous cell carcinoma | 40 | 3 | 3 | 2 |
| vascular lesion | 16 | 0 | 0 | 1 |

Test questions by source: PMC full text 66 (median 108 words); PMC-Patients 104 (median 97 words); PubMed abstract 51 (median 41 words).

<!-- step: revise -->
## Revision: version 3

Written by `python -m src.build.text.prepare revise`. The rule, fixed before the review was first run: a question whose confirming test result also names another class, not negated, has a doubtful label and leaves the dataset. Every other question keeps its row, file and split: nothing is moved between splits or reshuffled. Look-alike wording, other classes in the question and the same wrong answer from every model are never a reason to take a question out. The review (`build/review_flags.csv`) is of version 2 and is kept as the evidence.

- Questions in version 2 (reviewed): 745; taken out: 29; version 3: 716

| Class | Test: taken out | Test: version 3 | Validation: taken out | Validation: version 3 | Train: taken out | Train: version 3 |
|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 2 | 0 | 1 | 0 | 1 |
| basal cell carcinoma | 0 | 40 | 0 | 15 | 4 | 117 |
| dermatofibroma | 1 | 20 | 0 | 7 | 0 | 7 |
| melanoma | 3 | 37 | 1 | 14 | 6 | 119 |
| nevus | 1 | 39 | 1 | 14 | 1 | 23 |
| pigmented benign keratosis | 4 | 18 | 0 | 7 | 1 | 6 |
| squamous cell carcinoma | 3 | 37 | 0 | 15 | 3 | 150 |
| vascular lesion | 0 | 16 | 0 | 5 | 0 | 6 |
| all | 12 | 209 | 2 | 78 | 15 | 429 |

Taken out (the other class its confirming result names): 6452477-1 (test, dermatofibroma: melanoma); 6684299-1 (test, melanoma: nevus); 11035014-1 (test, melanoma: nevus); pubmed-37715977 (test, melanoma: nevus); pubmed-34231495 (test, nevus: melanoma); pubmed-18187992 (test, pigmented benign keratosis: squamous cell carcinoma); pubmed-23785597 (test, pigmented benign keratosis: melanoma; nevus); pubmed-38901381 (test, pigmented benign keratosis: basal cell carcinoma; melanoma); pubmed-40250188 (test, pigmented benign keratosis: melanoma); 3764943-1 (test, squamous cell carcinoma: actinic keratosis); pubmed-16519572 (test, squamous cell carcinoma: basal cell carcinoma); pubmed-24396723 (test, squamous cell carcinoma: melanoma); pubmed-42199973 (val, melanoma: nevus); 10751467-1 (val, nevus: melanoma); 4777905-1 (train, basal cell carcinoma: melanoma); 9588167-2 (train, basal cell carcinoma: squamous cell carcinoma); pubmed-27051742 (train, basal cell carcinoma: actinic keratosis); pubmed-33584954 (train, basal cell carcinoma: actinic keratosis; squamous cell carcinoma); 5690489-2 (train, melanoma: nevus); 10640706-1 (train, melanoma: nevus); pubmed-23569358 (train, melanoma: nevus); pubmed-24520521 (train, melanoma: nevus); pubmed-33997213 (train, melanoma: nevus); pubmed-42245639 (train, melanoma: nevus); pubmed-39165299 (train, nevus: melanoma); pubmed-41493242 (train, pigmented benign keratosis: melanoma; nevus); 10056737-4 (train, squamous cell carcinoma: actinic keratosis); pubmed-28370387 (train, squamous cell carcinoma: pigmented benign keratosis); pubmed-35872878 (train, squamous cell carcinoma: actinic keratosis).

<!-- step: photos_count -->
## Figures of the text dataset's cases: a free count

Written by `python -m src.build.cases.prepare count` on 2026-11-18. Only the captions were read: no image was downloaded or looked at. The rules are in `src/multimodal/figures.py`. A case counts when its article's full text is open in PubMed Central, the article holds no other case of the text dataset, and at least one figure's caption shows the lesion (it names dermoscopy, a clinical photo or a lesion) and names no microscope or stain, scan, state after treatment or at follow-up, removed specimen, drawing or chart, and no numbered case or several patients.

Rule fixed before the count: build the dataset when at least 80 test cases count and at least 6 classes have at least 8 test cases that count.

**Result: 107 test cases count; 7 classes have at least 8 (basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion). The rule is met: the dataset can be built.**

### Cases that count, per class and split

| Class | Test: cases | Test: count | Test: with a dermoscopy figure | Validation: cases | Validation: count | Train: cases | Train: count |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 2 | 1 | 0 | 1 | 0 | 1 | 1 |
| basal cell carcinoma | 40 | 20 | 2 | 15 | 9 | 121 | 54 |
| dermatofibroma | 21 | 11 | 2 | 7 | 1 | 7 | 4 |
| melanoma | 40 | 19 | 7 | 15 | 5 | 125 | 52 |
| nevus | 40 | 16 | 7 | 15 | 8 | 24 | 10 |
| pigmented benign keratosis | 22 | 11 | 2 | 7 | 5 | 7 | 4 |
| squamous cell carcinoma | 40 | 18 | 2 | 15 | 6 | 153 | 72 |
| vascular lesion | 16 | 11 | 0 | 5 | 5 | 6 | 4 |
| all | 221 | 107 | 22 | 80 | 39 | 444 | 201 |

### Why the other cases do not count

| Reason (the first that applies) | Cases |
|---|---|
| no PMC number | 124 |
| no open full text in PMC | 11 |
| no figure | 2 |
| the article holds another case of the dataset | 34 |
| no figure whose caption shows the lesion alone | 227 |

Of the cases with no figure that shows the lesion alone, 171 have a figure that shows the lesion beside other parts (such as a panel of histology); they could count only if the figure's panels were cut apart.

### Figures that show the lesion but are left out, by what else the caption names

| The caption also names | Figures |
|---|---|
| microscope or stain | 464 |
| scan | 101 |
| after treatment or at follow-up | 241 |
| removed specimen | 82 |
| drawing or chart | 10 |
| may show another patient | 49 |

### The cases that count

- Figures that show the lesion: 504 in 347 cases; cases with more than one: 114; with a dermoscopy figure: 70
- Cases whose question cites one of these figures by number: 347 of 347 whose question is on this computer
- Licences of their articles: CC BY (any, no NC): 153; CC BY-NC (any): 180; not stated: 7; other: http://creativecommons.org/licenses/CC BY-NC/4.0: 4; other: open-access: 3

### What this count does not show

- No image was looked at. A caption can be wrong or incomplete; the build step checks the images themselves (what kind of picture each is, writing in the picture, the same picture twice).
- A figure that counts may still show more than the lesion (an arrow, a ruler, a second panel the caption does not name).
- The captions stay on this computer (`data/published_cases/downloads/figures/captions.jsonl`), under their publishers' licences.

<!-- step: photos_download -->
## Pictures of the figures that count

Written by `python -m src.build.cases.prepare download` on 2026-11-18. The pictures come from PMC Article Datasets (NCBI's cloud copy of PMC's open-access articles), byte for byte, each checked against PMC's MD5 checksum, and stay on this computer (`data/published_cases/files/figures/images/`). Nothing in them was changed, and no person looked at them.

| Class | Test: cases with a picture | Validation | Train |
|---|---|---|---|
| actinic keratosis | 1 | 0 | 1 |
| basal cell carcinoma | 19 | 9 | 53 |
| dermatofibroma | 11 | 1 | 4 |
| melanoma | 18 | 5 | 51 |
| nevus | 15 | 8 | 10 |
| pigmented benign keratosis | 11 | 5 | 4 |
| squamous cell carcinoma | 18 | 6 | 70 |
| vascular lesion | 11 | 4 | 4 |
| all | 104 | 38 | 197 |

Pictures saved: 492 for 339 of 347 cases.

| Figure or picture | Count |
|---|---|
| saved | 492 |
| not open access in PMC | 6 |
| the article is retracted | 0 |
| no picture file for the figure among the article's files | 6 |
| the picture cannot be opened | 0 |

<!-- step: photos_check -->
## Checks of the pictures, each by itself

Written by `python -m src.build.cases.prepare check` on 2026-11-18. No person looked at the pictures. Image model: open_clip ViT-B-32 (laion2b_s34b_b79k), asked which of these a picture is: clinical photo ("a clinical photograph of a skin lesion"); dermoscopy ("a dermoscopy image of a skin lesion"); microscopy ("a microscope image of stained tissue"); scan ("a medical scan such as CT, MRI, X-ray or ultrasound"); drawing ("a chart, diagram, table or drawing"). Writing: EasyOCR (English). Versions of the models' answers made in this run: easyocr 1.7.2, open_clip 4.2.2, torch 2.15.0+cpu.

**After the checks: 104 test cases keep a picture; 7 classes have at least 8 (basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion). The rule for building the dataset (at least 80 test cases and 6 classes) is met.**

| Class | Test: cases with a picture kept | Validation | Train |
|---|---|---|---|
| actinic keratosis | 1 | 0 | 1 |
| basal cell carcinoma | 19 | 9 | 52 |
| dermatofibroma | 11 | 1 | 4 |
| melanoma | 18 | 5 | 50 |
| nevus | 15 | 8 | 10 |
| pigmented benign keratosis | 11 | 5 | 4 |
| squamous cell carcinoma | 18 | 6 | 68 |
| vascular lesion | 11 | 4 | 4 |
| all | 104 | 38 | 193 |

Pictures checked: 492; kept: 468.

| Left out because (a picture can have several reasons) | Pictures |
|---|---|
| smaller than the least size | 0 |
| writing in the picture names a diagnosis | 1 |
| the image model sees no photo of the skin | 24 |
| the same picture is in another case | 0 |
| the same picture twice in this case | 0 |

### Do the captions and the image model agree on dermoscopy?

Of 468 pictures the model sees as a skin photo, the caption and the model agree on whether it is a dermoscopy image for 375; Cohen's kappa 0.27. A low kappa means the picture type cannot be taken from either alone.

Pictures with any writing (labels, arrows' letters, scale bars): 83; with the name of a diagnosis: 1.

<!-- step: photos_build -->
## The case dataset

Written by `python -m src.build.cases.prepare build`. One question per case of the text dataset whose own article shows the lesion in pictures that passed every check (the check section): the frozen text question, word for word, and the pictures, byte for byte, in the order of their figures. No caption is shown. The splits are the text dataset's.

| Class | Test | Validation | Train |
|---|---|---|---|
| actinic keratosis | 1 | 0 | 1 |
| basal cell carcinoma | 19 | 9 | 51 |
| dermatofibroma | 10 | 1 | 4 |
| melanoma | 17 | 5 | 47 |
| nevus | 15 | 8 | 9 |
| pigmented benign keratosis | 10 | 5 | 4 |
| squamous cell carcinoma | 17 | 6 | 66 |
| vascular lesion | 11 | 4 | 4 |
| all | 100 | 38 | 186 |

Questions: 324; pictures: 451. Pictures per question: 1: 231, 2: 70, 3: 18, 4: 2, 5: 2, 8: 1.
Cases with kept pictures that the text dataset no longer holds (left out by its review): 11 (10056737-4, 11035014-1, 4777905-1, 5690489-2, 6452477-1, pubmed-24396723, pubmed-24520521, pubmed-35872878, pubmed-39165299, pubmed-40250188, pubmed-42245639).
