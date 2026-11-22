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

## Final split (stratified, lesion-grouped, 70/15/15 + 5 CV folds on train+val)

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| MEL melanoma | 322 | 65 | 65 | 452 |
| NV melanocytic nevus | 210 | 42 | 42 | 294 |
| BCC basal cell carcinoma | 280 | 56 | 56 | 392 |
| AK actinic keratosis | 94 | 19 | 18 | 131 |
| BKL benign keratosis-like | 341 | 69 | 68 | 478 |
| DF dermatofibroma | 80 | 15 | 16 | 111 |
| VASC vascular lesion | 101 | 20 | 21 | 142 |
| SCC squamous cell carcinoma | 140 | 28 | 29 | 197 |
| **Total** | **1,568** | **314** | **315** | **2,197** |

Imbalance ratio (train): 341 / 80 = **4.26 : 1**. Class weights are in `processed_cases/class_weights.json`.

## The pipeline and the folders

```
data/raw/  +  data/official_labels/            inputs: the Kaggle download (laptop only), the official label files
   │  step 1: python src/preprocess.py            (cleans, removes duplicates, corrects labels, splits)
   ▼
data/processed_images/  +  data/processed_cases/                   the cleaned images (laptop only); the cleaned, split cases
   │  step 2: python src/build_views.py
   ▼
data/views/image_only/  text_only/  multimodal/            the same cases in three views, with their questions

data/knowledge_corpus/                                     the medical text library for RAG: python src/build_knowledge_corpus.py
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
| `knowledge_corpus/` | `manifest.csv` and `corpus_card.json` (no text); `passages.jsonl` and the download cache stay on the laptop | partly |

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
