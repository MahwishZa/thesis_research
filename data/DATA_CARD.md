# Cleaned ISIC Skin-Lesion Dataset: Data Card

Source: Kaggle "Skin Cancer ISIC – 9 classes" (nodoubttome); images from the ISIC Archive (CC-BY-NC / CC-0).
Processed: 22 Nov 2026 with `code/preprocess.py` (seed 42). Everything here is real data: no synthetic images and no AI-generated text.

## What changed vs the Kaggle copy

| Step | Result |
|---|---|
| Files in the Kaggle copy | 2,357 JPGs (2,239 train / 118 test), 9 folders |
| Exact duplicates (same bytes) removed | 157 files → 2,202 ISIC IDs |
| Same picture under 2 ISIC IDs (pHash 0) removed | 5 → **2,197 unique images** |
| Labels checked against official sources | 1,761 from ISIC 2019 GroundTruth, 436 from ISIC Archive API records |
| Labels corrected | 158 (77 "nevus" copies → AK; 78 "SK" copies → MEL; 1 nevus → AK; 2 SK → MEL) |
| HAM10000 vs ISIC 2019 label disagreements | 0 |
| "Seborrheic keratosis" class | No genuine SK images, so the class was dropped. Final taxonomy = ISIC 2019's 8 classes |
| Train/test leakage | Removed: new split is grouped by lesion (lesion_id + pHash ≤ 3) |
| Images | Centre-cropped to 4:3, resized to 600×450, Shades-of-Gray colour constancy (p=6), JPEG q=95 |
| Normalisation | RGB mean/std and age mean/std from **train only** (`normalization.json`) |

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

Imbalance ratio (train): 341 / 80 = **4.26 : 1**. Class weights are in `class_balance.json`.

## Files

| File | Use |
|---|---|
| `master_clean.csv` | Everything: label + provenance, metadata, quality flags, split, CV fold, model features |
| `dataset_image_only.csv` | id, split, label, image_path |
| `dataset_text_only.csv` | id, split, label, clinical_text + metadata features (no image) |
| `dataset_multimodal.csv` | id, split, label, image_path + clinical_text + metadata features |
| `qa/qa_{image_only,text_only,multimodal}_{train,val,test}.jsonl` | 8-option diagnostic QA items for LLM-only / RAG / RAG² / Proposed RAG |
| `images/` | 2,197 standardised images |
| `class_balance.json` | train counts, balanced & effective-number class weights |
| `train_oversampled_index.csv` / `train_undersampled_index.csv` | ready-made balanced training lists (train split only) |
| `near_duplicate_pairs.csv` | near-duplicate audit and the action taken |
| `original_files_audit.csv` | every original Kaggle file → its final label and split |
| `feature_check_results.json`, `image_handcrafted_features.csv` | feature-sufficiency / shortcut experiment |
| `code/` | preprocess.py, feature_check.py, build_qa.py, build_knowledge_corpus.py, evaluate.py |
| `code/meta/` | official label/metadata files used (ISIC 2019 GT + metadata, HAM10000 metadata, ISIC Archive API extract) |

## Model inputs vs analysis-only columns

- **Model inputs (no label leakage):** image; age_z, age_missing, sex_male, sex_missing, site one-hots, site_missing; clinical_text.
- **Do NOT feed to a model:** `confirm_type` (histopathology vs follow-up reveals benign/malignant), `source`, `orig_width/height/kb`, `diagnosis_detail`, `label_*`, `folder_*`.

## Known limitations

1. **Source confound:** all MEL and NV come from the older ISIC archive; all other classes come mostly from HAM10000. Even after standardisation, image features detect the source with AUC 0.935, and original file size alone gives 0.33 balanced accuracy (chance = 0.125). Fix: rerun with `--ham-images <HAM10000 folder>` to add real HAM10000 MEL/NV, and report source-stratified results.
2. Metadata is missing for some older images (age 10.6%, sex 9.8%, site 6.7% missing; mostly NV/MEL). Missingness itself correlates with class, so report metadata results with and without the missing flags.
3. The test set is small (16–68 per class), so always report 95% bootstrap CIs or 5-fold CV.
4. Licence: ISIC images are mostly CC-BY-NC, for non-commercial research only.
