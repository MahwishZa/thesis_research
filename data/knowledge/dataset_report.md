# The knowledge base

What open-book runs may retrieve: training items only, never a validation or test question. One section per step, written by `src/build/knowledge/` (`library.py`: the library; `index.py`: the index; `enrich.py`: appearance and aspects); each step rewrites only its own section.

<!-- step: library -->
## Library: the reference texts for retrieval

Written by `python -m src.build.knowledge.library` on 2026-11-14. Four corpora, as in RAG² (PubMed, PMC, clinical practice guidelines, textbooks), from public sources of the US National Library of Medicine, found by fixed rules with no person choosing a text: the abstracts of PubMed's reviews about each class (its headings as a main subject, on the skin, in English, about people; not case reports or guidelines; the newest 200 per class); the open full texts of those reviews in PubMed Central (per class the newest 25 that PMC gives in full); PubMed's clinical practice guidelines about each class (the newest 50 per class, as open full text where PMC gives it, else as abstract); and the chapters of StatPearls (NCBI's Bookshelf) whose title or summary (its "Continuing Education Activity") names a class by a name that counts in `data/text/build/class_terms.csv` and whose title names no other site and no animal. Every text is cut word for word into passages of whole paragraphs of at most 300 words. `library.csv` lists the passages (source, licence, fingerprint); their texts are in `data/text/library/` (not in Git).

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

<!-- step: index -->
## Index: what open-book runs may retrieve

Written by `python -m src.build.knowledge.index`, from the committed tables of the text dataset and the library.

- `index.csv`: 5982 items: 429 training cases and 5553 library passages. No validation or test item of any dataset is among them. Each is linked to the questions only by its disease.
- `diseases.csv`: 8 classes, with the text names and MeSH headings, and the counts.

| Class | Training cases | Passages |
|---|---|---|
| actinic keratosis | 1 | 852 |
| basal cell carcinoma | 117 | 1177 |
| dermatofibroma | 7 | 73 |
| melanoma | 119 | 1626 |
| nevus | 23 | 851 |
| pigmented benign keratosis | 6 | 219 |
| squamous cell carcinoma | 150 | 1374 |
| vascular lesion | 6 | 234 |

Each item's licence is in the table, as its source gives it.

<!-- step: appearance -->
## Appearance corpus: open full texts of reviews on dermoscopy and clinical appearance

Written on 2026-11-17. For each class, PubMed's reviews (the class's MeSH headings as a main subject, on the skin, in English, about people, not case reports, not guidelines, with a free full text, and the words dermoscopy, dermoscopic, dermatoscopy, dermatoscopic or morphology in the title or abstract), the newest 100 asked, and of those the newest 40 that PubMed Central gives in full. Their paragraphs are cut into passages as the library's are. A passage is kept when it describes appearance (words of the lists in `enrich.py`), is not already in the library and shares no run of 12 words with a question of the text dataset; at most 150 per class, the newest first.

The first search found almost no review for dermatofibroma, pigmented benign keratosis, vascular lesion, so these classes were searched wider (column Search): the same reviews as above, but without the dermoscopy and morphology words, and found by the class's headings as a main subject OR by one of its names in the title or abstract. A review that names a condition class_terms.csv says does not count is left out, unless the name stands only inside a longer name that counts. The target is the same for every class.

- dermatofibroma: searched for dermatofibroma; benign fibrous histiocytoma; cutaneous fibrous histiocytoma; not counted: dermatofibrosarcoma protuberans; malignant fibrous histiocytoma
- pigmented benign keratosis: searched for seborrheic keratosis; solar lentigo; lichen planus-like keratosis; senile lentigo; benign lichenoid keratosis; melanoacanthoma; not counted: lentigo maligna; lentigo simplex; dermatosis papulosa nigra
- vascular lesion: searched for vascular lesion; cherry angioma; angiokeratoma; pyogenic granuloma; skin hemorrhage; senile angioma; Campbell de Morgan spot; lobular capillary hemangioma; not counted: hemangioma; angiosarcoma; Kaposi sarcoma; port-wine stain; acral pseudolymphomatous angiokeratoma

| Class | MeSH headings (main topic) | Search | Reviews found | Left out: names another condition | Open full texts | Passages kept | Documents of those passages |
|---|---|---|---|---|---|---|---|
| actinic keratosis | Keratosis, Actinic | first | 5 | 0 | 3 | 12 | 2 |
| basal cell carcinoma | Basal Cell Carcinoma | first | 26 | 0 | 9 | 24 | 4 |
| dermatofibroma | Histiocytoma, Benign Fibrous | wider | 8 | 3 (dermatofibrosarcoma protuberans (2), malignant fibrous histiocytoma (1)) | 0 | 0 | 0 |
| melanoma | Melanoma; Hutchinson's Melanotic Freckle; Melanoma, Amelanotic | first | 90 | 0 | 40 | 57 | 17 |
| nevus | Nevus; Nevus, Pigmented; Dysplastic Nevus Syndrome; Nevus, Epithelioid and Spindle Cell; Nevus, Blue; Nevus, Halo; Nevus, Intradermal | first | 20 | 0 | 11 | 28 | 5 |
| pigmented benign keratosis | Keratosis, Seborrheic | wider | 25 | 2 (dermatosis papulosa nigra (1), lentigo maligna (1)) | 12 | 2 | 1 |
| squamous cell carcinoma | Carcinoma, Squamous Cell; Bowen's Disease; Cutaneous Squamous Cell Carcinoma | first | 18 | 0 | 5 | 15 | 3 |
| vascular lesion | Angiokeratoma; Granuloma, Pyogenic | wider | 10 | 3 (hemangioma (3)) | 4 | 5 | 2 |

Passages kept: 108 from 28 documents. Licences (documents): http://creativecommons.org/licenses/by-nc/3.0/: 1; http://creativecommons.org/licenses/by/2.0: 1; http://creativecommons.org/licenses/by/2.5/: 1; http://creativecommons.org/licenses/by/4.0/: 1; https://creativecommons.org/licenses/by-nc-nd/4.0/: 3; https://creativecommons.org/licenses/by-nc/4.0/: 6; https://creativecommons.org/licenses/by/4.0/: 7; not stated: 8

Left out: already in the library: 28, does not describe appearance: 1293

Query: ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND review[Publication Type] NOT (guideline[Publication Type] OR "practice guideline"[Publication Type]) NOT "case reports"[Publication Type] AND english[Language] NOT (animals[MeSH Terms] NOT humans[MeSH Terms]) AND free full text[sb] AND (dermoscopy[Title/Abstract] OR dermoscopic[Title/Abstract] OR dermatoscopy[Title/Abstract] OR dermatoscopic[Title/Abstract] OR morphology[Title/Abstract]). Wider search, with the class's headings OR names: ("skin neoplasms"[MeSH Terms] OR "skin diseases"[MeSH Terms]) AND review[Publication Type] NOT (guideline[Publication Type] OR "practice guideline"[Publication Type]) NOT "case reports"[Publication Type] AND english[Language] NOT (animals[MeSH Terms] NOT humans[MeSH Terms]) AND free full text[sb].

No person has read the passages. A passage is about its review's class(es), found by the search, and describes appearance by a word count; neither is a medical judgement. The texts are under their publishers' licences (in the table) and stay on this computer for research use.

<!-- step: aspects -->
## Aspects: what each passage is about

5661 passages tagged (library: 5553, appearance: 108), in `passage_aspects.csv`. A passage has an aspect when it uses several different words of that aspect's list (fixed in `enrich.py`): appearance is dermoscopy (2 words) or clinical look (3 words); histology 3, treatment 3, epidemiology 2. A passage can have several aspects, and a passage about several classes counts for each.

### Do the tags follow the authors' section titles?

The check uses an independent signal: the title of the section a passage sits in. Only passages under a title that names one aspect count (for example "Treatment / Management" names treatment). Recall: of the passages under a title of the aspect, the share the words tag with it. Precision: of the tagged passages, the share under a title of the aspect. Lift: precision against the share of all titled passages under that aspect; above 1.0, the tag tells more than chance.

| Aspect | Passages under its titles | Recall | Tagged passages with a title | Precision | Lift |
|---|---|---|---|---|---|
| appearance | 132 | 34% | 62 | 73% | 5.3 |
| histology | 115 | 44% | 65 | 78% | 6.6 |
| treatment | 565 | 43% | 257 | 95% | 1.6 |
| epidemiology | 157 | 28% | 55 | 80% | 4.9 |

Every aspect has a lift above 1.0: the tags follow the authors' titles better than chance.
This is a check of consistency with the authors' own headings. It does not show that a passage is medically right or sufficient, and a passage under no informative title is not checked at all.

### Coverage of appearance

Target per class, fixed before the corpus was built: at least 30 passages that describe appearance, from at least 8 different documents.

| Class | Passages | In the library | In the appearance corpus | Describing appearance | Of which dermoscopy | Documents | Target |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 864 | 31 | 12 | 43 | 20 | 20 | met |
| basal cell carcinoma | 1201 | 53 | 24 | 77 | 53 | 27 | met |
| dermatofibroma | 73 | 4 | 0 | 4 | 1 | 3 | **short** |
| melanoma | 1683 | 38 | 57 | 95 | 73 | 36 | met |
| nevus | 879 | 34 | 28 | 62 | 34 | 21 | met |
| pigmented benign keratosis | 221 | 16 | 2 | 18 | 9 | 12 | **short** |
| squamous cell carcinoma | 1389 | 35 | 15 | 50 | 26 | 26 | met |
| vascular lesion | 239 | 7 | 5 | 12 | 3 | 7 | **short** |

Classes short of the target: dermatofibroma, pigmented benign keratosis, vascular lesion. They stay short: the target is not lowered, and what is missing is a limit of the retrieval experiments for these classes.
