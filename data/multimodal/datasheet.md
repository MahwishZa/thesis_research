# Data sheet: the skin-lesion multimodal dataset (version 1)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come from the files the preparation code (`src/multimodal/prepare.py`, `src/multimodal/audit.py`) writes:
- `dataset_report.md`: the full report of every step (records, build, audit)
- `questions.csv`: one row per question
- `knowledge.csv`: one row per item that open-book runs may search
- `diseases.csv`: one row per class

The dataset is made from the two frozen datasets: the image dataset (tag `image-dataset`, `data/image/datasheet.md`) and the text dataset (tag `text-dataset`, `data/text/datasheet.md`). It adds no image and no text of its own.

## At a glance

| | |
|---|---|
| Questions | 2,179: each photo of the image dataset, with the same patient's age, sex and body site from the photo's own ISIC record |
| Classes | the 8 of the image and text datasets: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion |
| Task | name the class from the photo and the patient's facts |
| Splits | test 320, validation 120, train 1,739: the image dataset's splits, unchanged; no lesion is in two splits |
| Labels | the image dataset's labels, unchanged |
| Facts shown | one sentence per photo, for example "Age: about 65 years. Sex: male. Body site: trunk."; a fact that is missing is shown as "not recorded" |
| Questions with all three facts | test 289 of 320, validation 106 of 120, train 1,545 of 1,739 |
| Knowledge | 7,736 items open-book runs may search: 1,739 training photos, 444 training cases of the text dataset, 5,553 library passages; no validation or test item of either dataset |
| Link between photo and text | by disease only: the two datasets share no patient |
| Audit | all 19 checks passed on 2026-11-14 (section 7) |
| Facts alone, no photo | 35.9% balanced accuracy on test (95% interval 30.0–41.5%; chance 12.5%) |

## 1. Motivation

**Purpose.** The dataset was prepared for the multimodal phase of an MS thesis on diagnosing skin lesions with large language models (LLMs), and on reducing their hallucinations with retrieval. It answers one question: does a model diagnose better, and invent less, when it sees the patient's recorded facts with the photo, and when it can retrieve knowledge across photos and texts?

**Why this form.** The image and text datasets share no patient: the photos come from the ISIC Archive, the texts from published case reports. So:
- A multimodal question is one photo with facts about the same patient, from the photo's own ISIC record: a real patient, nothing invented.
- A case text is never attached to a photo as "the same patient". That would invent a patient who does not exist.
- A text is never chosen for a question by the question's label. That would put the answer into the question.
- Texts and photos meet only in the knowledge that open-book runs search, linked by disease.

**Who.** Prepared by Mahwish Zafar for her MS thesis.

## 2. Composition

**Instances.** Each question is one dermoscopic photo of the image dataset, sent exactly as stored, and one sentence of facts about the same patient. The facts are age (ISIC records it in steps of 5 years, from 5 to 85), sex (male or female) and body site at its broadest level (trunk, head and neck, upper extremity, lower extremity, anogenital region).

**Questions per class and split**

| Class | Train | Validation | Test | Test with all three facts |
|---|---|---|---|---|
| actinic keratosis | 76 | 15 | 40 | 40 |
| basal cell carcinoma | 337 | 15 | 40 | 40 |
| dermatofibroma | 56 | 15 | 40 | 40 |
| melanoma | 381 | 15 | 40 | 32 |
| nevus | 238 | 15 | 40 | 19 |
| pigmented benign keratosis | 423 | 15 | 40 | 40 |
| squamous cell carcinoma | 141 | 15 | 40 | 40 |
| vascular lesion | 87 | 15 | 40 | 38 |
| **All** | **1,739** | **120** | **320** | **289** |

The 320 test photos show 248 lesions; photos of one lesion are in one split, and their facts agree in every case.

**Missing facts.** Over all 2,179 questions, age is not recorded for 231, sex for 213 and body site for 148. In test: age 31, sex 27, body site 17. Of the 31 test questions with a missing fact, 29 are melanoma or nevus (section 5, limitation 1).

**The columns of `questions.csv`**
- `split`, `label`, `path`, `sha256`, `lesion_group`: as in the image dataset's `dataset.csv`. `path` is inside `data/image/dataset/`.
- `isic_id`: the photo's ISIC number (two numbers for the two photos ISIC holds twice).
- `collection`: the ISIC credit line of the photo ("Anonymous" or "MILK study team"). It is for analysis and must never be shown to a model.
- `age`, `sex`, `site`: the facts, as shown.
- `complete`: "yes" when all three facts are recorded.
- `facts`: the sentence the model is shown with the photo.

**The columns of `knowledge.csv`**
- `item_id`: `image:<path>`, `case:<patient_uid>` or `passage:<passage_id>`.
- `kind`: image, case or passage.
- `diseases`: the item's class, or a passage's topics (several are separated by "; ").
- `link`: always "disease".
- `path`: where the item is on this computer; a passage is `data/text/library/passages.jsonl#<passage_id>`.
- `sha256`: the item's fingerprint, as in its own dataset.
- `source`: the ISIC number, the PubMed number, or the passage's source.
- `licence`: as below.

**`diseases.csv`.** Per class: the ISIC diagnoses of its photos, the names and MeSH headings the text dataset counts for it, its questions per split, its test questions with all three facts, its cases per split and its library passages.

**Recommended splits.** The splits are fixed and part of the dataset. They must not be redrawn: they are the image dataset's own, so the photo-only and multimodal results can be compared question by question.

**Licences**
- Photos: CC-BY-NC (1,436, credit "MILK study team") or CC-0 (743, credit "Anonymous"), as the ISIC Archive gives them.
- Training cases: PMC-Patients CC BY-NC-SA 4.0 (202); the publisher's copyright of a PubMed abstract (71); each PMC full text's own licence, exactly as PMC gives it (171): 159 give a Creative Commons address, 5 say only "open-access", and 7 give none ("(no licence given by PMC)").
- Library passages: as in the text dataset's `library.csv`, copied unchanged, including its spelling (`http://` and `https://` addresses both occur). 130 passages carry no licence there and are listed as "(not recorded)".

## 3. Collection

Nothing new was collected. The facts come from the ISIC records the image dataset downloaded (`data/image/downloads/isic/`), each checked against the fingerprint the image dataset saved for it. The cases and passages come from the text dataset, unchanged.

## 4. Preparation

Three commands, each writing its section of `dataset_report.md`:

1. **`records`** lists every patient field in the 2,179 ISIC records, by class and by photo collection, and sorts each into:
   - fact: may be shown;
   - hint: points to the diagnosis, and is never shown (a diagnosis, benign or malignant, melanocytic, melanoma thickness, a history of melanoma, whether a biopsy was taken);
   - identifier: a lesion or patient number, never shown.
2. **Choice of facts** (decided by the researcher on 2026-11-14, from the records report): age, sex and body site at its broadest level, the three that both collections record with the same values. Left out:
   - lesion size: recorded only for 267 melanomas and 2 actinic keratoses, so it would almost name melanoma;
   - body site levels 2 to 5: their words differ between the two collections ("shoulder", "thigh" and "breast" occur only in the melanoma and nevus collection);
   - "acral" (`anatom_site_special`): 36 photos only.
3. **`build`** writes the three tables:
   - One question per photo, with the split, class, file and lesion group copied unchanged.
   - Age is written "about N years", sex and site in lower case.
   - Two photos are held twice by ISIC (`ISIC_0000279` with `ISIC_0000280`, `ISIC_0000477` with `ISIC_0000512`, both in train). A fact counts only where their two records agree; on a disagreement it is "not recorded", never guessed.
   - The build stops if a validation or test item would enter the knowledge.
4. **`audit`** (`python -m src.multimodal.audit`) checks everything again with code of its own (section 7).

**Not done:** no photo was changed or added, no fact was corrected or guessed, no text was attached to a photo, and no item was chosen by hand.

## 5. Uses

**Intended uses**
- **Test split:** the reported result, run once per model and setting.
- **Validation split:** tuning prompts and retrieval settings. It is never reported as a result.
- **Train split and `knowledge.csv`:** the only items a run may use as examples or retrieve. Validation and test items must never be used that way.
- **Scoring:** balanced accuracy is the main score, with 95% intervals that resample whole lesions, as for the image dataset. Every multimodal result is read against two baselines:
  - the photo alone: the image dataset's run on the same 320 photos;
  - the facts alone: 35.9% (section 7).

  Variants of one method (an ablation) are compared on the same questions, with a paired test.

**Known limitations: state them with the results**
1. **A missing fact points to melanoma or nevus.**
   - **Why:** the melanoma and nevus photos come from a collection ("Anonymous") that records fewer facts. Of the 31 test questions with a missing fact, 29 are melanoma or nevus.
   - **What it means:** a model shown training examples with their facts could learn that "not recorded" means melanoma or nevus. No clinic works that way.
   - **What to do:** report every multimodal result twice: on all 320 test questions, and on the 289 with all three facts.
2. **The photos carry the image dataset's own limitations** (`data/image/datasheet.md`). Above all:
   - melanoma and nevus come from a different collection than the other six classes, and the border colour alone tells the collection;
   - lesions, not patients, are kept apart between splits.

   Retrieval that finds photos by their look can use the collection as a clue. Report image retrieval within each collection as well: melanoma against nevus, and the six other classes among themselves.
3. **The facts are few and coarse.**
   - Age is in steps of 5 years; the highest value in the records is 85.
   - The body site has five broad values.
   - Many facts are the same for many classes, so the facts alone reach 35.9%: far above chance, far from a diagnosis.
4. **The training cases and passages carry the text dataset's own limitations** (`data/text/datasheet.md`): about 2.7% wrong labels among the cases, and published texts the models may have read.
5. **The text and the photos are of different patients.** The dataset measures whether knowledge about a disease helps diagnose a photo; it cannot measure whether a patient's own written history helps.

**Still to decide before open-book runs:** whether retrieval may show the facts of training photos. `knowledge.csv` names the training photos and their facts are in `questions.csv`, so either choice needs no new build. Showing them is what could teach limitation 1.

**Not intended for**
- Clinical use or advice of any kind.
- Training a model on the test or validation split.
- Showing a model the `collection` column, or any field of the ISIC records other than the three facts.

## 6. Distribution

**What is shared and what isn't**
- In Git: the three tables, `dataset_report.md` and this data sheet.
- On this computer only: the photos, the case texts and the passages. They come from the image and text datasets, under their own licences.

**Licences and credit**
- Photos: as the ISIC Archive gives them: CC-BY-NC or CC-0, with the credit line in `collection`.
- Cases and passages: as stated in the text dataset's data sheet.

**Cite** the image and text datasets' sources as their data sheets say, and the ISIC Archive for the patient facts.

## 7. Maintenance and checking

**Versions.** Any change to a question, fact, split or knowledge item makes a new version, with new fingerprints, a new tag and an entry here.

**Checks.** The audit (`python -m src.multimodal.audit`) checks everything with code of its own. Run on 2026-11-14, all 19 checks passed:
- **Questions:** one per photo of the image dataset, each matching it in split, class, file, fingerprint and lesion group.
- **Facts:** every fact sentence worked out again from the ISIC records, each record checked against the image dataset's fingerprint.
- **Files:** every photo and training case present with its fingerprint, and every passage present in `passages.jsonl` with its text.
- **Knowledge:**
  - every training photo, training case and passage, once each;
  - no validation or test item;
  - nothing linked other than by disease;
  - no passage from an article that was a candidate case of the text dataset;
  - each case's licence as its source gives it.
- **Diseases:** every count recounted from the sources.

**What the facts alone tell.** A counting model learns from the training questions which class each value of a fact points to, every class weighing the same. It then answers the test questions from the facts alone, with no photo. Balanced accuracy, with its 95% interval (bootstrap over lesions):

| Facts used | Test questions | Class (chance 12.5%) | Photo collection (chance 50%) |
|---|---|---|---|
| age | 320 | 24.1% (18.9–29.3%) | 69.1% (63.1–75.1%) |
| sex | 320 | 23.1% (19.1–26.6%) | 66.1% (61.3–71.4%) |
| body site | 320 | 30.6% (25.6–35.7%) | 60.8% (54.9–66.9%) |
| which facts are missing | 320 | 19.1% (17.2–20.9%) | 66.8% (61.7–72.2%) |
| all three facts | 320 | 35.9% (30.0–41.5%) | 68.8% (63.2–74.7%) |
| all three facts, questions with all three | 289 | 30.3% (24.6–35.9%) | 54.5% (49.6–59.8%) |

**Rebuild**
1. The image and text datasets must be prepared, with the image dataset's ISIC records in `data/image/downloads/isic/` and the text dataset's downloads in `data/text/sources/`.
2. Run, in order: `python -m src.multimodal.prepare records`, `python -m src.multimodal.prepare build`, `python -m src.multimodal.audit`.
3. The same sources give the same files, byte for byte.

**Fingerprints**

| File | SHA-256 |
|---|---|
| `questions.csv` | `150d699e3752e46b6d0a46ee3456bc4a479809fe6553eea8c96ec0b3b18a17c6` |
| `knowledge.csv` | `bf17cb7d7f650c8aacc8ffdc7c111ab82b7634cda5aa9d7798263f3e8b37d881` |
| `diseases.csv` | `4b0b5f889b1643751720bbe9f7fd01832e71a6f451bcf0cd2850c598b4909fb5` |

Every photo, case and passage has its own SHA-256 in these tables; the audit checks the files against them.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- ISIC Archive, International Skin Imaging Collaboration: https://www.isic-archive.com/
- The image and text datasets' own references: `data/image/datasheet.md` and `data/text/datasheet.md`.
