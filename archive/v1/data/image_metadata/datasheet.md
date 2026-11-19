> **Version 1** (commit `a1c9209`), archived on 2026-11-19: a frozen copy kept for reference. It does not bind Version 2 (see `docs/v2_proposal.md` in the repository).

# Data sheet: the image + metadata dataset (version 1)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come from the files the preparation code (`src/build/image_metadata/prepare.py`, `src/build/image_metadata/audit.py`) writes:
- `dataset_report.md`: the full report of every step (records, build, audit)
- `dataset.csv`: one row per question

The questions are made from the frozen image dataset (version 1, `data/image/datasheet.md`) and its ISIC records; the dataset adds no image and no text of its own. What open-book runs may retrieve is described in `data/knowledge/datasheet.md`.

In the thesis this dataset is called **image + metadata**: an ISIC photo with three recorded facts, not a written case. It is the ablation of the photo with the patient's recorded facts; the dataset with a written case and the same patient's photos is the case dataset (`data/cases/datasheet.md`). Its dataset name in the configs is still `multimodal`.

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
| Knowledge | what open-book runs may search is the knowledge base (`data/knowledge/`): training items only, linked to a photo by disease only |
| Audit | all 19 checks passed on 2026-11-14 (section 7) |
| Facts alone, no photo | 35.9% balanced accuracy on test (95% interval 30.0–41.5%; chance 12.5%) |

## 1. Motivation

**Purpose.** The dataset was prepared for an MS thesis on diagnosing skin lesions with large language models (LLMs), and on reducing their hallucinations with retrieval. It answers one question: does a model diagnose better, and invent less, when it sees the patient's recorded facts with the photo, and when it can retrieve knowledge across photos and texts?

**Why this form.** The image and text datasets share no patient: the photos come from the ISIC Archive, the texts from published case reports. So:
- A question is one photo with facts about the same patient, from the photo's own ISIC record: a real patient, nothing invented.
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

**The columns of `dataset.csv`**
- `split`, `label`, `path`, `sha256`, `lesion_group`: as in the image dataset's `dataset.csv`. `path` is inside `data/image/dataset/`.
- `isic_id`: the photo's ISIC number (two numbers for the two photos ISIC holds twice).
- `collection`: the ISIC credit line of the photo ("Anonymous" or "MILK study team"). It is for analysis and must never be shown to a model.
- `age`, `sex`, `site`: the facts, as shown.
- `complete`: "yes" when all three facts are recorded.
- `facts`: the sentence the model is shown with the photo.

**Recommended splits.** The splits are fixed and part of the dataset. They must not be redrawn: they are the image dataset's own, so the photo-only and image + metadata results can be compared question by question.

**Licences.** Photos: CC-BY-NC (1,436, credit "MILK study team") or CC-0 (743, credit "Anonymous"), as the ISIC Archive gives them. The knowledge base's licences are in its own data sheet.

## 3. Collection

Nothing new was collected. The facts come from the ISIC records the image dataset downloaded (`data/image/downloads/isic/`), each checked against the fingerprint the image dataset saved for it.

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
3. **`build`** writes `dataset.csv`:
   - One question per photo, with the split, class, file and lesion group copied unchanged.
   - Age is written "about N years", sex and site in lower case.
   - Two photos are held twice by ISIC (`ISIC_0000279` with `ISIC_0000280`, `ISIC_0000477` with `ISIC_0000512`, both in train). A fact counts only where their two records agree; on a disagreement it is "not recorded", never guessed.
   - The build stops if a validation or test item would enter the knowledge.
4. **`audit`** (`python -m src.build.image_metadata.audit`) checks everything again with code of its own (section 7).

**Not done:** no photo was changed or added, no fact was corrected or guessed, no text was attached to a photo, and no item was chosen by hand.

## 5. Uses

**Intended uses**
- **Test split:** the reported result, run once per model and setting.
- **Validation split:** tuning prompts and retrieval settings. It is never reported as a result.
- **Train split and the knowledge base:** the only items a run may use as examples or retrieve. Validation and test items must never be used that way.
- **Scoring:** balanced accuracy is the main score, with 95% intervals that resample whole lesions, as for the image dataset. Every result on this dataset is read against two baselines:
  - the photo alone: the image dataset's run on the same 320 photos;
  - the facts alone: 35.9% (section 7).

  Variants of one method (an ablation) are compared on the same questions, with a paired test.

**Known limitations: state them with the results**
1. **A missing fact points to melanoma or nevus.**
   - **Why:** the melanoma and nevus photos come from a collection ("Anonymous") that records fewer facts. Of the 31 test questions with a missing fact, 29 are melanoma or nevus.
   - **What it means:** a model shown training examples with their facts could learn that "not recorded" means melanoma or nevus. No clinic works that way.
   - **What to do:** report every result on this dataset twice: on all 320 test questions, and on the 289 with all three facts.
2. **The photos carry the image dataset's own limitations** (`data/image/datasheet.md`). Above all:
   - melanoma and nevus come from a different collection than the other six classes, and the border colour alone tells the collection;
   - lesions, not patients, are kept apart between splits.

   Retrieval that finds photos by their look can use the collection as a clue. Report image retrieval within each collection as well: melanoma against nevus, and the six other classes among themselves.
3. **The facts are few and coarse.**
   - Age is in steps of 5 years; the highest value in the records is 85.
   - The body site has five broad values.
   - Many facts are the same for many classes, so the facts alone reach 35.9%: far above chance, far from a diagnosis.
4. **What is retrieved carries its own limitations** (`data/knowledge/datasheet.md`): passages about the disease, and other patients' photos and cases.
5. **No written case.** The dataset measures whether recorded facts, and knowledge about a disease, help diagnose a photo; whether a patient's own written history helps is measured on the case dataset.

**Still to decide before open-book runs:** whether retrieval may show the facts of training photos. The knowledge index names the training photos and their facts are in `dataset.csv`, so either choice needs no new build. Showing them is what could teach limitation 1.

**Not intended for**
- Clinical use or advice of any kind.
- Training a model on the test or validation split.
- Showing a model the `collection` column, or any field of the ISIC records other than the three facts.

## 6. Distribution

**What is shared and what isn't**
- In Git: `dataset.csv`, `dataset_report.md` and this data sheet.
- On this computer only: the photos, in the image dataset's folder, under their own licences.

**Licences and credit**
- Photos: as the ISIC Archive gives them: CC-BY-NC or CC-0, with the credit line in `collection`.

**Cite** the image dataset's sources as its data sheet says, and the ISIC Archive for the patient facts.

## 7. Maintenance and checking

**Versions.** Any change to a question, fact or split makes a new version, with new fingerprints and an entry here. Version 1 was frozen on 2026-11-14 (commit `b1bb3e7`). On 2026-11-18 the files moved from `data/multimodal/` to `data/image_metadata/` (`questions.csv` became `dataset.csv`, unchanged byte for byte) and the knowledge tables to `data/knowledge/`.

**Checks.** The audit (`python -m src.build.image_metadata.audit`) checks everything with code of its own. Run on 2026-11-14, all 19 checks passed; run again on 2026-11-18 after the knowledge index followed version 3 of the text dataset (7,721 items), all passed again, and the facts-alone scores below did not change:
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
1. The image and text datasets and the library must be prepared, with the image dataset's ISIC records in `data/image/downloads/isic/` and the text dataset's downloads in `data/text/sources/`.
2. Run, in order: `python -m src.build.image_metadata.prepare records`, `python -m src.build.image_metadata.prepare build`, `python -m src.build.knowledge.index`, `python -m src.build.image_metadata.audit` (the audit also checks the knowledge index).
3. The same sources give the same files, byte for byte.

**Fingerprints**

| File | SHA-256 |
|---|---|
| `dataset.csv` | `150d699e3752e46b6d0a46ee3456bc4a479809fe6553eea8c96ec0b3b18a17c6` |

Every photo has its own SHA-256 in the table; the audit checks the files against them.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- ISIC Archive, International Skin Imaging Collaboration: https://www.isic-archive.com/
- The image dataset's own references: `data/image/datasheet.md`.
