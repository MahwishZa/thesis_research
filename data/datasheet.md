# Cleaned ISIC Skin-Lesion Dataset: Data Card

Source: Kaggle "Skin Cancer ISIC – 9 classes" (nodoubttome); images from the ISIC Archive (CC-BY-NC / CC-0).
Processed: 22 Nov 2026 with `src/preprocess.py` (seed 42). Everything here is real data: no synthetic images and no AI-generated text.

## What changed vs the Kaggle copy

| Step | Result |
|---|---|
| Files in the Kaggle copy | 2,357 JPGs (2,239 train / 118 test), 9 folders |
| Exact duplicates (same bytes) removed | 157 files → 2,202 ISIC IDs |
| Same picture under 2 ISIC IDs (pHash 0) removed | 5 → **2,197 unique images** |
| Labels checked against official sources | 1,761 from ISIC 2019 GroundTruth, 436 from ISIC Archive API records |
| Labels corrected | 158 among the 2,202 ISIC IDs (77 "nevus" copies → AK; 78 "SK" copies → MEL; 1 nevus → AK; 2 SK → MEL); **157 among the 2,197 kept images**, because one corrected image (ISIC_0010990, filed under both melanoma and SK) was then removed as the same picture as ISIC_0000521 |
| HAM10000 vs ISIC 2019 label disagreements | 0 |
| "Seborrheic keratosis" class | No genuine SK images, so the class was dropped. Final taxonomy = ISIC 2019's 8 classes |
| Train/test leakage | Removed: new split is grouped by lesion (lesion_id + pHash ≤ 3) |
| Images | Centre-cropped to 4:3, resized to 600×450, Shades-of-Gray colour constancy (p=6), JPEG q=95 |
| Normalisation | RGB mean/std and age mean/std from **train only** (`processed_cases/normalization.json`) |

## Final split (lesion-grouped; balanced test and validation splits; 5 CV folds on train+val)

First a stratified, lesion-grouped 70/15/15 split (seed 42); then the test split is balanced to **30 images of every class** and the
validation split to **20 images of every class** by moving whole lesion groups to or from train in a fixed order (`balanced_splits` in
`src/common.py`, decided 2026-11-22; the test split is balanced first and is not touched by the validation pass). Train keeps the
natural class mix: the imbalance is handled when a model is trained (class weights, the oversampled and undersampled lists; see "Class balance" below). Numbers
from the committed tables ✔ (laptop rerun of 2026-11-23):

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| MEL melanoma | 402 | 20 | 30 | 452 |
| NV melanocytic nevus | 244 | 20 | 30 | 294 |
| BCC basal cell carcinoma | 342 | 20 | 30 | 392 |
| AK actinic keratosis | 81 | 20 | 30 | 131 |
| BKL benign keratosis-like | 428 | 20 | 30 | 478 |
| DF dermatofibroma | 61 | 20 | 30 | 111 |
| VASC vascular lesion | 92 | 20 | 30 | 142 |
| SCC squamous cell carcinoma | 147 | 20 | 30 | 197 |
| **Total (lesions)** | **1,797 (1,504)** | **160 (126)** | **240 (190)** | **2,197** |

Imbalance ratio (train): 428 / 61 = **7.02 : 1**. Class weights are in `processed_cases/class_weights.json`. A balanced split does not
reflect how often each disease occurs in a clinic, so its precision is not a clinic's precision (accuracy equals macro-recall).

## Class balance: the sampling techniques, and why their files remain

Record written 2026-11-24 (researcher's decision: the files stay, because model training is involved or may be). Facts are ✔ read from
`src/preprocess.py` (step 8, lines "balancing helpers (TRAIN only)") and from the committed files.

| Technique | What exactly is done | Where | Size |
|---|---|---|---|
| **Selection to a balanced split** (validation, test) | Whole lesion groups are moved between train and the evaluation split in a fixed order (`balanced_splits`, `src/common.py`) until every class has 20 (validation) or 30 (test) images. No image is copied or invented. | `data/views/*/val.jsonl`, `test.jsonl` | 160 and 240 cases |
| **Oversampling** (train only) | Every class is brought up to the size of the largest class (BKL, 428) by adding randomly drawn copies of its own train cases, **with replacement** (seed 42, `numpy.random.default_rng`). The result is a list of case ids; no new image is made. | `processed_cases/train_oversampled.csv` | 3,424 rows (428 per class) |
| **Undersampling** (train only) | Every class is cut down to the size of the smallest class (DF, 61) by drawing its cases at random **without replacement** (seed 42). Also a list of ids. | `processed_cases/train_undersampled.csv` | 488 rows (61 per class) |
| **Class weights** (train only) | "balanced": weight = N / (8 x class count), N = 1,797. "Effective number" (beta = 0.999): proportional to (1 - beta) / (1 - beta^n), scaled to a mean of 1. | `processed_cases/class_weights.json` | 8 numbers per scheme |

Train class counts behind them: MEL 402, NV 244, BCC 342, AK 81, BKL 428, DF 61, VASC 92, SCC 147 (ratio 428 / 61 = 7.02 : 1).
Balanced weights: MEL 0.559, NV 0.921, BCC 0.657, AK 2.773, BKL 0.525, DF 3.682, VASC 2.442, SCC 1.528.

**The train split itself is not resampled.** It keeps the natural class mix, and the lists and weights above are prepared helpers.
Resampling is never applied to validation or test, so no copied case can leak into an evaluation split (the lists are built from
train cases only).

**Why the files remain.** The research may train a model: a trained retriever (the RAG-squared reproduction, not decided) or a
fine-tuned component. A model trained on this natural mix would favour the common classes, so a balanced training list or the class
weights are needed at that point. Preparing them now, from the final train split and the fixed seed, keeps the training data
reproducible and avoids a later change to the data folder. Today nothing reads them: the four LLMs are used as they are, and LLM-only
and standard RAG train nothing. `src/check.py features` uses scikit-learn's own class weighting, not these files.

**What to say in the thesis.** If a model is trained, state which technique was used (weights, oversampling or undersampling), that it
was applied to the train split only, and the seed. Oversampling repeats cases (a model may memorise them) and undersampling discards
most of the common classes (428 down to 61); class weights avoid both and are the simplest choice. If nothing is trained in the end,
these files can be removed (with the tests that expect them: `tests/test_system.py`) and this section reduced to the balanced
evaluation splits.

## The pipeline and the folders

```
data/raw/  +  data/official_labels/            inputs: the Kaggle download (laptop only), the official label files
   │  step 1: python src/preprocess.py            (cleans, removes duplicates, corrects labels, splits)
   ▼
data/processed_images/  +  data/processed_cases/                   the cleaned images (laptop only); the cleaned, split cases
   │  step 2: python src/build.py views
   ▼
data/views/image_only/  text_only/  multimodal/            the same cases in three views, with their questions

data/knowledge_corpus/                                     the medical text library for RAG: python src/build.py corpus
```

| Path | What it holds | In Git |
|---|---|---|
| `raw/` | the Kaggle download (2,357 files in `Train/` and `Test/`), untouched | no |
| `official_labels/` | the official label and metadata files: ISIC 2019 ground truth + metadata, HAM10000 metadata, ISIC Archive extract | yes |
| `processed_images/` | the 2,197 cleaned, standardised images | no |
| `processed_cases/all_cases.csv` | every kept case: label + provenance, patient facts, quality flags, split, fold, model features | yes |
| `processed_cases/raw_files_audit.csv` | every original Kaggle file → its final label and split | yes |
| `processed_cases/near_duplicates.csv` | the near-duplicate pairs and the action taken | yes |
| `processed_cases/classes.json` | the 8 classes: index and name | yes |
| `processed_cases/class_weights.json`, `normalization.json`, `train_oversampled.csv`, `train_undersampled.csv` | train counts and class weights; train-only image and age statistics; balanced training lists (for a trained model only) | yes |
| `views/<view>/cases.csv` | one row per case: id, split, fold, lesion group, label, and only the inputs of that view | yes |
| `views/<view>/{train,val,test}.jsonl` | one 8-option multiple-choice question per case, for every system (LLM only, standard RAG, RAG², proposed) | yes |
| `knowledge_corpus/` | `manifest.csv` and `corpus_card.json` (no text); `reference_texts.json` (quoted Dermoscopedia texts, one or more per disease (one per dermoscopy section), with their source, for semantic similarity); `passages.jsonl` and the download cache stay on the laptop | partly |

Image paths in every table are relative to `data/` (`processed_images/<id>.jpg`).

## The three views (the same 2,197 cases, the same splits)

| View | Inputs | Example of the question text |
|---|---|---|
| `image_only` | the image | "Look at the dermoscopic image of the skin lesion. What is the most likely diagnosis?" |
| `text_only` | age, sex, body site in one fixed sentence | "Patient: 55-year-old female. Skin lesion on the head/neck (face). What is the most likely diagnosis?" |
| `multimodal` | the image and the same sentence | the sentence + "Look at the dermoscopic image of the lesion. What is the most likely diagnosis?" |

Each question has the 8 classes as options in an order fixed per case (seed 42), the answer letter, the class code
(`gold_label`) and the lesion group. The analysis results of the features are in `results/feature_check/`.

## Model inputs vs analysis-only columns

- **Model inputs (no label leakage):** image; clinical_text (age, sex, site); for a trained model also age_z, age_missing, sex_male,
  sex_missing, site one-hots, site_missing (in `processed_cases/all_cases.csv` only).
- **Do NOT feed to a model:** `confirm_type` (histopathology vs follow-up reveals benign/malignant), `source`, `orig_width/height/kb`, `diagnosis_detail`, `label_*`, `folder_*`.

## Known limitations

1. **Source confound:** all MEL and NV come from the older ISIC archive; all other classes come mostly from HAM10000. Even after standardisation, image features detect the source with AUC 0.935, and original file size alone gives 0.33 balanced accuracy (chance = 0.125). Only the Kaggle dataset is used (decision of 2026-11-22), so no images are added: report it as a limitation and give results per source (`src/evaluate.py --cases data/processed_cases/all_cases.csv --by source`).
2. Metadata is missing for some older images (age 10.6%, sex 9.8%, site 6.7% missing; mostly NV/MEL). Missingness itself correlates with class, so report the text and multimodal results also for the cases with all facts recorded (`--by facts_recorded`).
3. The test set is small (16–68 per class), so always report 95% bootstrap CIs (`src/evaluate.py` draws whole lesions).
4. Licence: ISIC images are mostly CC-BY-NC, for non-commercial research only.

## Licences and research ethics

| Part | What it is | Terms | In Git |
|---|---|---|---|
| `raw/`, `processed_images/` | dermoscopic images (Kaggle copy of the ISIC Archive) | each image carries its own ISIC licence (mostly CC BY-NC, some CC0): non-commercial research, attribution | **no** (laptop only) |
| `official_labels/` | ISIC 2019 ground truth and metadata, HAM10000 metadata, ISIC Archive records | the ISIC / HAM10000 data licences (non-commercial, attribution) ◻ | yes, with attribution (below) |
| `processed_cases/`, `views/` | derived from the two rows above: ISIC ids, labels, age, sex, body site | the same terms as their sources | yes, with attribution |
| `knowledge_corpus/passages.jsonl`, `cache/` | full texts of PubMed abstracts, PMC papers, guidelines and Dermoscopedia pages | each passage's own licence (publishers' copyright, CC BY-NC-ND, ...) | **no** (laptop only) |
| `knowledge_corpus/manifest.csv`, `corpus_card.json` | lists and counts, no text | – | yes |
| `knowledge_corpus/reference_texts.json` | quoted Dermoscopedia texts (one per dermoscopy section of each disease's page), each with its source | CC BY-NC-SA 4.0: attribution, non-commercial, shared under the same licence | yes (the file keeps the source of each text) |
| `.env` | API keys | secret | **never** |

Attribution: images and metadata from the ISIC Archive (isic-archive.com), including HAM10000 (Tschandl et al. 2018, Scientific
Data 5:180161) and the ISIC 2019 challenge data; Kaggle copy "Skin Cancer ISIC – 9 classes" (nodoubttome). ◻ The exact licence of
each source file is to be confirmed on its page before publication.

Ethics: all data are public and de-identified (no names, no dates of birth; age, sex and body site only), and no patient is
contacted, so no new consent is involved. ◻ Whether the university's ethics committee must still register or exempt the study is
to be confirmed with the supervisor. The LLM-only and RAG runs send images and patient facts to the providers' servers (Anthropic,
OpenAI, Google, Parasail via OpenRouter); this is research use of non-commercial data, and the method chapter should say so.

