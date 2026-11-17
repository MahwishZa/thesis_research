# The knowledge base: additions (version 2)

Written by `python -m src.multimodal.enrich`; one section per step. The frozen version 1 files are unchanged.

<!-- step: appearance -->
### Appearance corpus: open full texts of reviews on dermoscopy and clinical appearance

Written on 2026-11-17. For each class, PubMed's reviews (the class's MeSH headings as a main subject, on the skin, in English, about people, not case reports, not guidelines, with a free full text, and the words dermoscopy, dermoscopic, dermatoscopy, dermatoscopic or morphology in the title or abstract), the newest 100 asked, and of those the newest 40 that PubMed Central gives in full. Their paragraphs are cut into passages as the library's are. A passage is kept when it describes appearance (words of the lists in `enrich.py`), is not already in the library and shares no run of 12 words with a question of the text dataset; at most 150 per class, the newest first.

| Class | MeSH headings (main topic) | Reviews found | Open full texts | Passages kept | Documents of those passages |
|---|---|---|---|---|---|
| actinic keratosis | Keratosis, Actinic | 5 | 3 | 12 | 2 |
| basal cell carcinoma | Basal Cell Carcinoma | 26 | 9 | 24 | 4 |
| dermatofibroma | Histiocytoma, Benign Fibrous | 0 | 0 | 0 | 0 |
| melanoma | Melanoma; Hutchinson's Melanotic Freckle; Melanoma, Amelanotic | 90 | 40 | 57 | 17 |
| nevus | Nevus; Nevus, Pigmented; Dysplastic Nevus Syndrome; Nevus, Epithelioid and Spindle Cell; Nevus, Blue; Nevus, Halo; Nevus, Intradermal | 20 | 11 | 28 | 5 |
| pigmented benign keratosis | Keratosis, Seborrheic | 1 | 1 | 0 | 0 |
| squamous cell carcinoma | Carcinoma, Squamous Cell; Bowen's Disease; Cutaneous Squamous Cell Carcinoma | 18 | 5 | 15 | 3 |
| vascular lesion | Angiokeratoma; Granuloma, Pyogenic | 0 | 0 | 0 | 0 |

Passages kept: 101 from 25 documents. Licences (documents): http://creativecommons.org/licenses/by-nc/3.0/: 1; http://creativecommons.org/licenses/by/2.0: 1; http://creativecommons.org/licenses/by/4.0/: 1; https://creativecommons.org/licenses/by-nc-nd/4.0/: 3; https://creativecommons.org/licenses/by-nc/4.0/: 5; https://creativecommons.org/licenses/by/4.0/: 6; not stated: 8

Left out: already in the library: 28, does not describe appearance: 1006

Query: ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND review[Publication Type] NOT (guideline[Publication Type] OR "practice guideline"[Publication Type]) NOT "case reports"[Publication Type] AND english[Language] NOT (animals[MeSH Terms] NOT humans[MeSH Terms]) AND free full text[sb] AND (dermoscopy[Title/Abstract] OR dermoscopic[Title/Abstract] OR dermatoscopy[Title/Abstract] OR dermatoscopic[Title/Abstract] OR morphology[Title/Abstract]).

No person has read the passages. A passage is about its review's class(es), found by the search, and describes appearance by a word count; neither is a medical judgement. The texts are under their publishers' licences (in the table) and stay on this computer for research use.

<!-- step: aspects -->
### Aspects: what each passage is about

5654 passages tagged (library: 5553, appearance: 101), in `passage_aspects.csv`. A passage has an aspect when it uses several different words of that aspect's list (fixed in `enrich.py`): appearance is dermoscopy (2 words) or clinical look (3 words); histology 3, treatment 3, epidemiology 2. A passage can have several aspects, and a passage about several classes counts for each.

#### Do the tags follow the authors' section titles?

The check uses an independent signal: the title of the section a passage sits in. Only passages under a title that names one aspect count (for example "Treatment / Management" names treatment). Recall: of the passages under a title of the aspect, the share the words tag with it. Precision: of the tagged passages, the share under a title of the aspect. Lift: precision against the share of all titled passages under that aspect; above 1.0, the tag tells more than chance.

| Aspect | Passages under its titles | Recall | Tagged passages with a title | Precision | Lift |
|---|---|---|---|---|---|
| appearance | 132 | 34% | 62 | 73% | 5.3 |
| histology | 115 | 44% | 65 | 78% | 6.6 |
| treatment | 565 | 43% | 257 | 95% | 1.6 |
| epidemiology | 157 | 28% | 55 | 80% | 4.9 |

Every aspect has a lift above 1.0: the tags follow the authors' titles better than chance.
This is a check of consistency with the authors' own headings. It does not show that a passage is medically right or sufficient, and a passage under no informative title is not checked at all.

#### Coverage of appearance

Target per class, fixed before the corpus was built: at least 30 passages that describe appearance, from at least 8 different documents.

| Class | Passages | In the library | In the appearance corpus | Describing appearance | Of which dermoscopy | Documents | Target |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 864 | 31 | 12 | 43 | 20 | 20 | met |
| basal cell carcinoma | 1201 | 53 | 24 | 77 | 53 | 27 | met |
| dermatofibroma | 73 | 4 | 0 | 4 | 1 | 3 | **short** |
| melanoma | 1683 | 38 | 57 | 95 | 73 | 36 | met |
| nevus | 879 | 34 | 28 | 62 | 34 | 21 | met |
| pigmented benign keratosis | 219 | 16 | 0 | 16 | 9 | 11 | **short** |
| squamous cell carcinoma | 1389 | 35 | 15 | 50 | 26 | 26 | met |
| vascular lesion | 234 | 7 | 0 | 7 | 1 | 5 | **short** |

Classes short of the target: dermatofibroma, pigmented benign keratosis, vascular lesion. They stay short: the target is not lowered, and what is missing is a limit of the retrieval experiments for these classes.

<!-- step: cards -->
### Cards: one text card per training photo

1739 cards in `image_cards.csv`, one for each training photo of `knowledge.csv`, made from the photo's own ISIC record. Core fields (diagnosis, confirmation, age, sex, body site) read "not recorded" when the record lacks them; the finer body site, the size and the melanoma history are left out when not recorded. Nothing is guessed. The collection and the licence are columns of the table, never part of a card's text.

Checks that stop the step when one fails: the cards are exactly the training photos of `knowledge.csv`; no validation or test photo has a card; every card's diagnosis is one that `diseases.csv` lists for its class; no card's text names the photo's collection; every card's text is made from its columns alone.

#### How often each field is recorded, by class

| Class | Cards | Confirmation | Age | Sex | Body site | Finer site | Size | Melanoma history |
|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 76 | 76 | 76 | 76 | 76 | 66 | 1 | 1 |
| basal cell carcinoma | 337 | 337 | 337 | 337 | 336 | 252 | 0 | 0 |
| dermatofibroma | 56 | 56 | 56 | 54 | 54 | 6 | 0 | 0 |
| melanoma | 381 | 381 | 317 | 318 | 333 | 142 | 233 | 233 |
| nevus | 238 | 0 | 118 | 127 | 167 | 135 | 0 | 0 |
| pigmented benign keratosis | 423 | 423 | 421 | 423 | 420 | 288 | 0 | 0 |
| squamous cell carcinoma | 141 | 141 | 141 | 141 | 141 | 58 | 0 | 0 |
| vascular lesion | 87 | 87 | 87 | 87 | 87 | 47 | 0 | 0 |

#### Example

> Dermoscopic photo of a skin lesion. Diagnosis (ISIC): Indeterminate > Indeterminate epidermal proliferations > Solar or actinic keratosis. Confirmed by: histopathology. Age: about 85 years. Sex: male. Body site: head and neck. Lesion size: 6 mm (longest diameter). Personal history of melanoma: yes. Family history of melanoma: no.

A card describes another patient's photo, not the photo of a question: it can be retrieved as evidence, and it carries the diagnosis of its own photo only.
