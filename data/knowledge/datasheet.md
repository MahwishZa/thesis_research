# Data sheet: the knowledge base (what open-book runs may retrieve)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come from the files in this folder, which the preparation code writes:
- `dataset_report.md`: one section per step (library, index, appearance, aspects), by `src/build/knowledge/library.py` and `src/build/knowledge/enrich.py`
- `library.csv`: one row per passage of the reference library
- `appearance.csv`: one row per passage of the appearance corpus
- `passage_aspects.csv`: what each passage is about
- `index.csv`: one row per item a run may retrieve (cases and passages)
- `diseases.csv`: one row per class, with its names and counts

`index.csv` and `diseases.csv` are written by `src/build/knowledge/index.py`, from the text dataset and `library.csv`.

## At a glance

| | |
|---|---|
| Reference library | 5,553 passages from 1,267 documents in four corpora, as in RAG²: PubMed abstracts of reviews, PubMed Central full texts of reviews, clinical practice guidelines, StatPearls chapters |
| Appearance corpus | 108 passages from 28 open-access reviews that describe how a lesion looks (dermoscopy, clinical look) |
| Index | 5,982 items: every training case of the text dataset (429, version 3) and every library passage (5,553) |
| Held out | no validation or test case or question of any dataset; no case report and no article of the text dataset's candidates in the library |
| Link to a question | by disease only: an item describes the disease, or another patient's case, never the patient of a question |
| Classes short of appearance text | dermatofibroma, pigmented benign keratosis, vascular lesion (section 5) |

## 1. Motivation

Open-book runs answer a question with retrieved evidence. RAG² (Sohn et al., 2025) retrieves evenly from four corpora (PubMed, PMC, clinical practice guidelines and textbooks); the library follows it. The library holds little text on how a lesion looks, which is what a photo shows, so one addition fills this gap: open-access reviews on dermoscopy and clinical appearance. (Version 1 also had a text card for each training photo of its ISIC image dataset; the cards were removed on 2026-11-19 with that dataset, see section 7.) The index lists everything a run may retrieve in one table, so that a retrieval can be checked against it: no held-out item can be retrieved.

**Who.** Prepared by Mahwish Zafar for her MS thesis.

## 2. Composition

**Library per corpus**

| Corpus | Documents | Passages | Words |
|---|---|---|---|
| PubMed: abstracts of reviews | 953 | 1,012 | 183,440 |
| PMC: open full texts of reviews | 124 | 2,595 | 548,311 |
| Clinical practice guidelines | 99 | 493 | 99,080 |
| Textbook: StatPearls chapters | 91 | 1,453 | 242,863 |
| **All** | **1,267** | **5,553** | **1,073,694** |

**Library per class** (documents found; a document can be about several classes)

| Class | Reviews (abstracts) | Open full texts of reviews | Guidelines | StatPearls chapters |
|---|---|---|---|---|
| actinic keratosis | 200 | 25 | 14 | 11 |
| basal cell carcinoma | 200 | 25 | 32 | 20 |
| dermatofibroma | 35 | 0 | 0 | 3 |
| melanoma | 200 | 25 | 50 | 31 |
| nevus | 200 | 25 | 6 | 13 |
| pigmented benign keratosis | 33 | 2 | 0 | 11 |
| squamous cell carcinoma | 200 | 25 | 44 | 28 |
| vascular lesion | 18 | 2 | 0 | 10 |

**Passages.** Every text is cut, word for word, into passages of whole paragraphs of at most 300 words (a longer paragraph at its sentence ends). The columns of `library.csv` and `appearance.csv`: `passage_id`, `corpus`, `source` (PubMed or PMC number, or the StatPearls chapter), `title`, `section`, `year`, `licence`, `topics` (the classes the document is about), `words` and `sha256`. The texts themselves are in `texts/passages.jsonl` and `texts/appearance.jsonl`, on this computer only.

**`passage_aspects.csv`.** For each of the 5,661 passages (library and appearance corpus): whether it is about appearance (dermoscopy or clinical look), histology, treatment or epidemiology, by counting the words of fixed lists (`src/build/knowledge/enrich.py`), and the section title it sits under.

**The columns of `index.csv`**
- `item_id`: `case:<patient_uid>` or `passage:<passage_id>`
- `kind`: case or passage
- `diseases`: the item's class, or a passage's topics (several are separated by "; ")
- `link`: always "disease"
- `path`: where the item is on this computer; a passage is `data/knowledge/texts/passages.jsonl#<passage_id>`
- `sha256`: the item's fingerprint, as in its own dataset
- `source`: the PubMed number, or the passage's source
- `licence`: as below

**`diseases.csv`.** Per class: the names and MeSH headings the text dataset counts for it, its text cases per split and its library passages.

**Licences**
- Training cases: PMC-Patients CC BY-NC-SA 4.0; the publisher's copyright of a PubMed abstract; each PMC full text's own licence, exactly as PMC gives it ("(no licence given by PMC)" where it gives none).
- Passages: PubMed abstracts under their publishers' copyright; PMC full texts and guidelines under their own licences (mostly Creative Commons); StatPearls CC BY-NC-ND 4.0. Each passage's licence is in its table, copied unchanged; a passage with none is "(not recorded)" in the index.

## 3. Collection

- **Library.** PubMed records and PubMed Central full texts asked through NCBI's E-utilities on 2026-11-14, by fixed queries (in `dataset_report.md`): each class's MeSH headings as a main subject, reviews or guidelines on the skin, in English, about people, not case reports; the newest 200 reviews, 25 open full texts and 50 guidelines per class. StatPearls from the NCBI Bookshelf Open Access archive (`statpearls_NBK430685.tar.gz`, 2,047,364,463 bytes, SHA-256 `08117bcff46ed366a7ab9033d134b81bf345e81cb9a1537b6075400418bb09bd`, downloaded 2026-11-13): the chapters whose title or summary names a class.
- **Appearance corpus.** Asked on 2026-11-17: the class's reviews with a free full text that mention dermoscopy or morphology; for dermatofibroma, pigmented benign keratosis and vascular lesion, whose first search found almost nothing, a wider search by the class's headings or names, leaving out a review that names a condition that does not count for the class.
- **Nothing new.** No text was written or changed by the researcher, and no person read or chose a passage.

## 4. Preparation

| Step | Command | What it did | Result |
|---|---|---|---|
| library | `python -m src.build.knowledge.library` | found, cut and kept the four corpora | 5,553 passages; 1 left out for sharing 12 words in a row with a question, 2 as the same as an earlier one |
| appearance | `python -m src.build.knowledge.enrich appearance` | found, cut and kept the passages on appearance not already in the library | 108 passages from 28 documents |
| aspects | `python -m src.build.knowledge.enrich aspects` | tagged every passage by aspect and checked the tags against the authors' section titles | 5,661 passages tagged; every aspect follows the titles better than chance (lift 1.6–6.6) |
| index | `python -m src.build.knowledge.index` | listed every training case and passage, and the counts per class | 5,982 items in `index.csv`; `diseases.csv` |

**Rules that keep test answers out.** No case report is searched for, no article of the text dataset's candidates is kept, and a passage that shares a run of 12 words with a question of the text dataset is left out. The index holds training items only; the index step stops on any validation or test item.

## 5. Uses

**Intended uses**
- **Retrieval for open-book runs:** a run may retrieve only what `index.csv` lists, plus the appearance passages. The passages' tables are the source of their texts and fingerprints.
- **Retrieval-only evaluation:** a retrieved item is relevant when its disease is the question's class (label-based relevance), so retrieval can be scored before any answer is generated.

**Known limitations: state them with the results**
1. **Uneven by class.** Dermatofibroma, pigmented benign keratosis and vascular lesion have 18–35 reviews and no guidelines, and fall short of the fixed target for text on appearance (at least 30 passages from 8 documents): 4, 18 and 12 passages. The target was not lowered; retrieval for these classes has less to find.
2. **The tags are word counts.** The aspects follow the authors' section titles better than chance, which shows consistency, not medical correctness.
3. **Not about the patient.** A passage describes a disease; a training case describes another patient. None is evidence about the patient of a question except through the disease.
4. **No precedent photos yet.** The index of Version 1 held the training photos of the ISIC image dataset; Version 2's precedents (training photos and records of the clinic patients, training cases of the published cases) are not built.
5. **The texts are dated.** PubMed changes every day; a rebuild finds other documents.

**Not intended for:** clinical use or advice; retrieving anything that is not listed here.

## 6. Distribution

- **In Git:** the tables, `dataset_report.md` and this data sheet: sources, licences and fingerprints, but no text.
- **On this computer only:** the passages' texts (`texts/`) and the StatPearls archive (`sources/statpearls/`), under their publishers' licences, for research use only; the case texts stay in their dataset's folder.

## 7. Maintenance and checking

**Versions.** Any change to a passage or an item makes a new version, with new fingerprints and an entry here.
- **2026-11-14:** the library and the index.
- **2026-11-17:** the appearance corpus, the aspects and the cards (added; nothing earlier changed).
- **2026-11-18:** moved into this folder from `data/text/` and `data/multimodal/`; the index has its own step (it was written by the image + metadata build, byte for byte the same); the index's case items follow version 3 of the text dataset (444 → 429; 7,736 → 7,721 items). The image + metadata audit passed all its checks on the rebuilt index.
- **2026-11-19:** the Version 1 photo datasets were removed from the working folders (stage R2b of the clean-up; they are in `archive/v1/` and Git history). The index no longer lists photos (7,721 → 5,982 items; its case and passage rows are byte for byte as before), `diseases.csv` lost its photo columns, and the `cards` step and `image_cards.csv` were removed.

**Checks.** The library step checks the StatPearls archive against its fingerprint on every run. The enrich steps stop on any broken rule (a passage table and its texts that differ), and the index step stops on any validation or test case.

**Rebuild.** With the text dataset prepared: `python -m src.build.knowledge.library`, then `python -m src.build.knowledge.index`, then `python -m src.build.knowledge.enrich appearance` and `aspects`. The library and appearance steps ask NCBI; a later run finds other documents.

**Fingerprints**

| File | SHA-256 |
|---|---|
| `library.csv` | `ca8d6969379c08a2f837d47cf9359ab40d116001cfe7219ca7568a82becf710b` |
| `appearance.csv` | `6dc290e4b6a044c400cf757c299c55663b4673622db59f17d58fcbaeee70b3e5` |
| `passage_aspects.csv` | `dca44093d9a1755980e935d3664da84f5485c1b8dbbfbdf6dec6de288141785b` |
| `index.csv` | `44167584eeea690ef4e3ff35cfc2327355023c6299b95bf88d1c10b82382c9c7` |
| `diseases.csv` | `4ce0b015f3cceb505b949afc37b7de16943de17b2251d9706a45f95b924fdc6f` |

Every passage has its own SHA-256 in its table; the steps check the texts against them.

**Backup.** Keep a ZIP of `data/knowledge/texts` and `data/knowledge/sources` outside this computer.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
- Sohn, J., et al. (2025). Rationale-guided retrieval augmented generation for medical question answering. In *Proceedings of NAACL 2025*, 12739–12753.
- StatPearls. StatPearls Publishing, Treasure Island (FL). NCBI Bookshelf, NBK430685: https://www.ncbi.nlm.nih.gov/books/NBK430685/
- PubMed and PubMed Central, US National Library of Medicine, accessed through NCBI's E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
