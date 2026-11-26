# Data sheet: the cleaned ISIC skin-lesion dataset and the knowledge corpus

| | |
|---|---|
| Document status | Current. Facts were checked against the committed tables on 2026-11-24 (✔) or are marked ◻ (not yet confirmed). |
| Last data build | `preprocess` and `views` (`python -m src.main`): laptop rerun of 2026-11-23 (seed 42). Knowledge corpus snapshot: 2026-11-24; image library 2026-11-24. |
| Structure | Follows the headings of "Datasheets for Datasets" (Gebru et al., Communications of the ACM 64(12), 2021 ◻ check the citation) |
| Related documents | `results/results.md` (the results) |

Everything here is real data: no synthetic images and no AI-generated text. Marks: ✔ read from a committed file or recomputed from it; ◻ assumed or not yet confirmed.

## 1. Summary

| Item | Value |
|---|---|
| Source | Kaggle "Skin Cancer ISIC – 9 classes" (nodoubttome); images from the ISIC Archive (CC BY-NC / CC0) |
| Cases | 2,197 unique dermoscopic images, 8 classes |
| Classes | MEL, NV, BCC, AK, BKL, DF, VASC, SCC (ISIC 2019 taxonomy) |
| Splits | train 1,797 · validation 160 (20 per class) · test 240 (30 per class); no lesion in two splits ✔ |
| Views | image only · text only (age, sex, body site) · image + text; one 8-option multiple-choice question per case |
| Knowledge corpus | 3,426 passages from 992 documents, four libraries, and 85 picture-and-caption records (for the RAG systems) |
| Used by | `src/pipelines/` (the runs of every system), `src/evaluation/` (all metrics), `src/main.py` (pipeline status) and `src/data/preprocessing.py` (feature check); one entry point, `python -m src.main <command>` |

## 2. Motivation and intended use

- **Purpose:** to test whether retrieval-augmented generation (RAG) improves diagnostic multiple-choice question answering about skin lesions, using four pretrained LLMs as they are (no fine-tuning), in three views: image, text, and both.
- **Intended use:** a research benchmark only. No decision about a patient depends on any answer.
- **Not intended for:** clinical use; estimating how often each disease occurs (the evaluation splits are balanced, so a score on them is not a clinic's score); training a model on the validation or test splits.
- **Scope rule (researcher's decision, 2026-11-22):** only the Kaggle dataset is used. Official ISIC and HAM10000 label files are used to check and correct its labels, not to add images.

## 3. Composition

### 3.1 Cases and classes

| Class | Name | Train | Val | Test | Total |
|---|---|---|---|---|---|
| MEL | melanoma | 402 | 20 | 30 | 452 |
| NV | melanocytic nevus | 244 | 20 | 30 | 294 |
| BCC | basal cell carcinoma | 342 | 20 | 30 | 392 |
| AK | actinic keratosis | 81 | 20 | 30 | 131 |
| BKL | benign keratosis-like lesion (seborrheic keratosis, solar lentigo, lichen planus-like keratosis) | 428 | 20 | 30 | 478 |
| DF | dermatofibroma | 61 | 20 | 30 | 111 |
| VASC | vascular lesion | 92 | 20 | 30 | 142 |
| SCC | squamous cell carcinoma | 147 | 20 | 30 | 197 |
| **Total (lesions)** | | **1,797 (1,504)** | **160 (126)** | **240 (190)** | **2,197** |

Imbalance ratio in train: 428 / 61 = **7.02 : 1** ✔ (`processed/cases/class_weights.json`).

### 3.2 Source of the images ✔ (`all_cases.csv`, column `source`)

| Source | Images | Classes |
|---|---|---|
| ISIC archive (2016–2017 era) | 760 | all 452 MEL, all 294 NV, and a few AK (2), DF (7), SCC (5) |
| HAM10000 | 1,437 | all BCC, BKL and VASC, and most AK, DF and SCC |

### 3.3 Patient facts

Age, sex and body site are the only patient facts. Missing: age 232 cases (10.6%), sex 216 (9.8%), site 148 (6.7%) ✔. Missing facts are concentrated in NV and MEL, so the fact that a value is missing is itself a hint of the class (see section 11). In the test split, 28 of 240 cases lack at least one fact: NV 15, MEL 7, DF 2, VASC 2, BCC 1, SCC 1 ✔.

### 3.4 Cross-validation folds

Train and validation cases carry a fold number 0 to 4 in `all_cases.csv` (`cv_fold`); test cases carry −1 ✔. The folds are for training a model; the LLM-only and RAG runs do not use them.

## 4. Collection and provenance

| Step | Result |
|---|---|
| Files in the Kaggle copy | 2,357 JPGs (2,239 train / 118 test), 9 folders |
| Exact duplicates (same bytes) removed | 157 files → 2,202 ISIC IDs |
| Same picture under 2 ISIC IDs (pHash 0) removed | 5 → **2,197 unique images** |
| Labels checked against official sources | 1,761 from ISIC 2019 GroundTruth, 436 from ISIC Archive API records |
| Labels corrected | 158 among the 2,202 ISIC IDs (77 "nevus" copies → AK; 78 "SK" copies → MEL; 1 nevus → AK; 2 SK → MEL); **157 among the 2,197 kept images**, because one corrected image (ISIC_0010990, filed under both melanoma and SK) was then removed as the same picture as ISIC_0000521 |
| HAM10000 vs ISIC 2019 label disagreements | 0 |
| "Seborrheic keratosis" class | No genuine SK images, so the class was dropped. Final taxonomy = ISIC 2019's 8 classes |
| Where each file went | `processed/cases/raw_files_audit.csv` lists all 2,357 original files with their final label and split |

## 5. Preprocessing and splitting

| Step | What is done |
|---|---|
| Images | Centre-cropped to 4:3, resized to 600×450, Shades-of-Gray colour constancy (p = 6), JPEG quality 95 |
| Leakage control | Splits are grouped by lesion (`lesion_id` plus near-duplicate pictures with pHash distance ≤ 3), so no lesion appears in two splits |
| Split | First a stratified, lesion-grouped 70/15/15 split (seed 42) |
| Balancing | The test split is then balanced to **30 images per class** and the validation split to **20 per class**, by moving whole lesion groups between train and the evaluation split in a fixed order (`balanced_splits`, `src/data/views.py`; the test pass runs first and the validation pass does not touch it) |
| Normalisation | RGB mean/std and age mean/std from **train only** (`processed/cases/normalization.json`) |
| Checks | Ids and lesion groups of train, validation and test are disjoint (0 overlap) ✔; the three views hold identical cases, options, answers and groups ✔ |

A balanced split does not reflect how often each disease occurs in a clinic, so its precision is not a clinic's precision (on a balanced split accuracy equals macro-recall).

## 6. Class balance: the sampling techniques, and why their files remain

Record written 2026-11-24 (researcher's decision: the files stay, because model training is involved or may be). Facts are ✔ read from `src/data/preprocessing.py` (step 8, "balancing helpers (TRAIN only)") and from the committed files.

| Technique | What exactly is done | Where | Size |
|---|---|---|---|
| **Selection to a balanced split** (validation, test) | Whole lesion groups are moved between train and the evaluation split in a fixed order until every class has 20 (validation) or 30 (test) images. No image is copied or invented. | `data/views/*/validation.jsonl`, `test.jsonl` | 160 and 240 cases |
| **Oversampling** (train only) | Every class is brought up to the size of the largest class (BKL, 428) by adding randomly drawn copies of its own train cases, **with replacement** (seed 42, `numpy.random.default_rng`). The result is a list of case ids; no new image is made. | `processed/cases/train_oversampled.csv` | 3,424 rows (428 per class) |
| **Undersampling** (train only) | Every class is cut down to the size of the smallest class (DF, 61) by drawing its cases at random **without replacement** (seed 42). Also a list of ids. | `processed/cases/train_undersampled.csv` | 488 rows (61 per class) |
| **Class weights** (train only) | "balanced": weight = N / (8 × class count), N = 1,797. "Effective number" (beta = 0.999): proportional to (1 − beta) / (1 − beta^n), scaled to a mean of 1. | `processed/cases/class_weights.json` | 8 numbers per scheme |

Train class counts behind them: MEL 402, NV 244, BCC 342, AK 81, BKL 428, DF 61, VASC 92, SCC 147. Balanced weights: MEL 0.559, NV 0.921, BCC 0.657, AK 2.773, BKL 0.525, DF 3.682, VASC 2.442, SCC 1.528.

**The train split itself is not resampled.** It keeps the natural class mix; the lists and weights are prepared helpers. Resampling is never applied to validation or test, so no copied case can leak into an evaluation split.

**Why the files remain.** The research may train a model: a trained retriever (the RAG² reproduction, not decided) or a fine-tuned component. A model trained on this natural mix would favour the common classes, so a balanced training list or the class weights are needed at that point. Preparing them now, from the final train split and the fixed seed, keeps the training data reproducible and avoids a later change to the data folder. Today nothing reads them: the four LLMs are used as they are, and LLM-only and standard RAG train nothing. `python -m src.main features` uses scikit-learn's own class weighting, not these files.

**What to say in the thesis.** If a model is trained, state which technique was used (weights, oversampling or undersampling), that it was applied to the train split only, and the seed. Oversampling repeats cases (a model may memorise them) and undersampling discards most of the common classes (428 down to 61); class weights avoid both and are the simplest choice. If nothing is trained in the end, these files can be removed (with the tests in `tests/system.py` that expect them) and this section reduced to the balanced evaluation splits.

## 7. The three views and the questions

The same 2,197 cases and the same splits, in three views:

| View | Inputs | Question text (example) |
|---|---|---|
| `image_only` | the image | "Look at the dermoscopic image of the skin lesion. What is the most likely diagnosis?" |
| `text_only` | age, sex, body site in one fixed sentence | "Patient: 55-year-old female. Skin lesion on the head/neck (face). What is the most likely diagnosis?" |
| `multimodal` | the image and the same sentence | "Patient: 70-year-old female. Skin lesion on the head/neck (face). Look at the dermoscopic image of the lesion. What is the most likely diagnosis?" |

Each question has the 8 classes as options in an order fixed per case (seed 42, the same order in all three views), the answer letter, the class code (`gold_label`) and the lesion group. None of the 240 test questions names a diagnosis in its text ✔. Image paths in the tables are relative to `data/` (`processed/images/<id>.jpg`; `processed_images/<id>.jpg` before 2026-11-25, see section 8).

**Model inputs (no label leakage):** the image; `clinical_text` (age, sex, site); for a trained model also `age_z`, `age_missing`, `sex_male`, `sex_missing`, the site one-hots and `site_missing` (in `all_cases.csv` only).
**Do NOT feed to a model:** `confirm_type` (histopathology versus follow-up reveals benign or malignant), `source`, `orig_width`, `orig_height`, `orig_kb`, `diagnosis_detail`, `label_*`, `folder_*`.

## 8. Folders and files

```
data/raw/  +  data/annotations/                    inputs: the Kaggle download (laptop only), the official label files
   │  step 1: python -m src.main preprocess       cleans, removes duplicates, corrects labels, splits
   ▼
data/processed/images/  +  data/processed/cases/    the cleaned images (laptop only); the cleaned, split cases
   │  step 2: python -m src.main views
   ▼
data/views/image_only/  text_only/  multimodal/     the same cases in three views, with their questions

data/corpus/text_only/  image_only/  multimodal/    the medical library for RAG, one folder per view (section 9): corpus (text), visual (images)
data/annotations/reference_texts.json               the reference texts of the semantic-similarity score (evaluation only)
```

| Path | What it holds | In Git |
|---|---|---|
| `raw/` | the Kaggle download (2,357 files in `Train/` and `Test/`), untouched | no |
| `annotations/` (4 csv files) | ISIC 2019 ground truth and metadata, HAM10000 metadata, an ISIC Archive extract | yes |
| `processed/images/` | the 2,197 cleaned, standardised images | no |
| `processed/cases/all_cases.csv` | every kept case: label and provenance, patient facts, quality flags, split, fold, model features | yes |
| `processed/cases/raw_files_audit.csv` | every original Kaggle file → its final label and split | yes |
| `processed/cases/near_duplicates.csv` | the near-duplicate pairs and the action taken | yes |
| `processed/cases/classes.json` | the 8 classes: index and name | yes |
| `processed/cases/class_weights.json`, `normalization.json`, `train_oversampled.csv`, `train_undersampled.csv` | train counts and class weights; train-only image and age statistics; balanced training lists (see section 6) | yes |
| `views/<view>/cases.csv` | one row per case: id, split, fold, lesion group, label, and only the inputs of that view | yes |
| `views/<view>/{train,validation,test}.jsonl` | one 8-option multiple-choice question per case, for every system | yes |
| `views/provenance.json` | the record of the 2026-11-25 move: for each changed file its earlier path and SHA-256 before and after (below) | yes |
| `corpus/text_only/` | `manifest.csv`, `corpus_card.json`, `audit.json` (no text); `passages.jsonl`, the download cache and the text index stay on the laptop | partly |
| `corpus/image_only/` | `visual_card.json` (no text); the images, the picture-and-caption records, the Kaggle vectors and the image index stay on the laptop | partly |
| `corpus/multimodal/library.json` | no records: it names the two libraries the view combines and how their searches are merged | yes |
| `annotations/reference_texts.json` | the reference texts of the semantic-similarity score | yes |

**The 2026-11-25 move (researcher's approval).** The folders were renamed to the structure above: `processed_images/` → `processed/images/`, `processed_cases/` → `processed/cases/`, `official_labels/` → `annotations/` (first `annotations/official_labels/`, flattened the same day), `knowledge_corpus/` → `corpus/` (for a few hours `knowledge/corpus/`, simplified at the researcher's request; with `reference_texts.json` → `annotations/`), and `val.jsonl` → `validation.jsonl`. No case, question, answer or option changed: only the `image_path` field (`processed_images/` → `processed/images/`). `views/provenance.json` records, for each of the 10 changed files, its SHA-256 before and after; a test (`tests/acceptance.py`) checks that writing the paths the old way gives back exactly the earlier file ✔, and the pipeline status accepts the approved runs, which recorded the earlier SHA-256, through this record ✔. The rebuilt views match the committed files byte for byte ✔ (`tests/system.py`).

## 9. The knowledge corpus (for the RAG systems)

**One library folder per view** (researcher's decision, 2026-11-26; `data/corpus/<view>/`, named like the view). `text_only/` holds the text records (the passages, and the 42 figure captions that the text index adds) and the text index; `image_only/` holds the picture-and-caption records, the images and the image index; `multimodal/` holds no records, only `library.json`, which says that it combines the other two and that the two top-10 searches are merged by reciprocal rank fusion (k = 60). Every view therefore searches the same records, so a difference between views comes from the question, not from the library. The code reads and writes these folders (`TEXT_LIBRARY`, `IMAGE_LIBRARY`, `MULTIMODAL_LIBRARY` in `src/utils.py`); `python -m src.main status` and a test check the layout.

Built by `python -m src.main corpus`; snapshot 2026-11-24 (the rebuild after the corpus rules of that day); numbers ✔ from `corpus_card.json` and `manifest.csv`.

| Library | What it is | Passages |
|---|---|---|
| `guidelines` | the Meditron guideline collection (Hugging Face); CDC, CCO and PubMed-sourced guidelines; sources "nice" and "wikidoc" are excluded | 912 |
| `pmc` | open-licence full-text papers from PubMed Central, licence read from each paper's JATS record | 1,200 |
| `pubmed` | abstracts of PubMed papers, cited by PMID; AI and image-dataset papers and retracted papers removed | 1,124 |
| `textbook` | Dermoscopedia pages (CC BY-NC-SA 4.0) | 190 |
| **Total** | 992 documents | **3,426** |

- **Passage size:** 30 to 200 words; at most 150 passages per library and disease.
- **Tags:** each passage is tagged with the disease names (and synonyms) found in its title and text (`tag()` in `src/retrieval/corpus.py`); the disease it was searched for is kept apart as `query_class` and is not a tag. A passage can carry several tags, or none. Passages tagged per class: MEL 1,159 · NV 628 · BCC 603 · AK 470 · BKL 369 · DF 256 · VASC 319 · SCC 607. The tags are the weak labels for the retrieval metrics: a passage is relevant for a question when its tags contain the true class.
- **Gaps:** guidelines hold no passage for DF and only 12 for BKL; pubmed holds 74 for VASC; the textbook is thin for AK (6), DF (10), VASC (3) and SCC (4).
- **Reference texts:** `reference_texts.json` holds 40 texts quoted word for word from Dermoscopedia (MEL 2, NV 10, BCC 8, AK 4, BKL 8, DF 5, VASC 1, SCC 2), each with its page, link, licence and retrieval date. They are used only for the semantic-similarity metric, never as RAG passages.
- **Licences of the passages** (✔ counts): PubMed abstracts 1,124 (research use, cite PMID); CC BY 4.0 1,091; CC BY-NC 4.0 521; CC BY-NC-SA 4.0 216; CC BY-NC-ND 4.0 191; other Creative Commons 200; Meditron guideline sources 83 (the source's own terms).
- **What is kept in Git:** `manifest.csv` (one row per passage: id, library, document, URL, licence, tags, word count, MD5), `corpus_card.json` (counts, removals, settings) and `reference_texts.json`. The passage texts stay on the laptop because of their licences.

### 9.1 The retrieval index (`python -m src.main index`)

Each passage is embedded once as [title, text] with the MedCPT article encoder (Jin et al. 2023), and the unit-length vectors are saved in `data/corpus/text_only/index/` (`embeddings.f32`, `index.json`; laptop only, because they are derived from licensed text). A question is embedded with the MedCPT query encoder and the passages with the highest cosine similarity are retrieved from the whole corpus, with no filter by disease. Passages are not cut again: the corpus builder already cut them (whole sentences, at most 200 words, one sentence shared with the next). Tags and query classes are never embedded and never used in retrieval; `index.json` records the encoder, the SHA-256 of `passages.jsonl` (a changed corpus is noticed) and whether the reference pages were left out.

### 9.2 Corpus audit (✔ from `manifest.csv` and `audit.json`, corpus of 2026-11-24)

| Finding | Figure | Consequence |
|---|---|---|
| Duplicate passages (same text) or passage ids | 0; 4 near-duplicate pairs (2 PMC paper and its PubMed abstract, 2 between two Dermoscopedia pages) | small; kept |
| Passages shorter than 40 words | 241 of 3,426 (7.0%) | short passages carry little evidence |
| Passages with no tag | 350 (10.2%): pubmed 220, guidelines 114, pmc 16 | they never count as relevant in Hit@k, Precision@k, Recall@k, MRR and nDCG@k, even when on topic |
| Passages whose query disease is not among their tags | 515 (15.0%) | the tags come from class names found in the text; they are weak labels |
| Most passages from one document | 102 (PMC7150662), then 83 and 65 | a few long papers supply many passages |
| **Textbook passages from the pages that also supply the reference texts** | **44 of 190** | a RAG answer that quotes such a passage resembles the reference text of its disease almost word for word, which inflates the semantic similarity. The laptop index was built with `--exclude-reference-pages`, so they are not retrievable (status: 3,374 of 3,426 passages indexed) |
| Passages per disease (tags) | MEL 1,159 · NV 628 · BCC 603 · SCC 607 · AK 470 · BKL 369 · VASC 319 · DF 256 | retrieval may lean to melanoma |
| Gaps | guidelines: DF 0, BKL 12; pubmed: VASC 74; textbook: AK 6, DF 10, VASC 3, SCC 4 | thin evidence for the rarer diseases |
| Image-associated text | PMC figure captions of 30 words or more are kept as passages (the corpus rules, `src/retrieval/corpus.py`); the image library (9.3) adds picture-and-caption records | still little text about what a lesion looks like for the rarer diseases |

### 9.3 The image library (`visual`, `visual-images`, `visual-index`; ✔ from `corpus/image_only/visual_card.json`, snapshot 2026-11-24)

Picture-and-caption records from the open-access PMC papers of the text corpus. Captions are the papers' own words; nothing is written by a model.

| Step | Result |
|---|---|
| Papers read | 90; 46 with at least one skin-image figure |
| Figures kept by the caption rules | 88 (MEL 22, SCC 19, BKL 16, NV 15, AK 12, VASC 11, BCC 10, DF 7); left out: 80 not a clinical or dermoscopic image, 38 charts, diagrams or another imaging method, 16 captions too short, 7 histology only, 4 with no diagnosis named |
| Images downloaded | 88 of 88, from the PMC open-data bucket |
| Kaggle look-alike check (BiomedCLIP similarity ≥ 0.95, or a picture hash with at most 4 different bits) | 3 dropped (2 by similarity, 1 by hash); 0 duplicates inside the library (similarity ≥ 0.98) |
| Indexed (BiomedCLIP image vectors) | 85 records |

**The look-alike check is a strict safety step, not a calibrated test** (`visual-calibrate`, on the Kaggle set itself): it flags 64.5% of the 448 same-lesion photo pairs, but also, for 74.7% of the 2,197 images, the most similar image of a different lesion. Dermoscopic pictures look alike to the image model, so the check cannot tell same-lesion from different-lesion photos; it can only remove close copies, at the cost of some images that are not copies (3 of 88). The thresholds were not changed after this measurement.

**Use in the standard RAG (2026-11-25):** an image-only question searches this library with its own image; an image + text question fuses that image search with the text search of its patient facts (reciprocal rank fusion of the top 10 of each). The text index also holds the caption records (`index`, default; `--no-captions` leaves them out), so the text search of every view can find them; a caption that is already a corpus passage of the same paper (same text, or half or more of its 8-word stretches) is not added twice, and `index.json` lists those left out. When the same text reaches a question twice (a caption as passage and as picture record), only the better-ranked copy is given to the model. The model gets the caption and the paper title of a retrieved figure, never the library picture itself (researcher's decision, 2026-11-25). Laptop index of 2026-11-25 ✔ (`status`): 3,416 records = 3,426 passages − 52 left out (44 from the reference pages, 8 that copy a reference text) + 42 caption records; the other 43 captions were already corpus passages. Validation (160 questions, Hit@3, `results/standard_rag/<view>/retrieval_validation.json`): text search (BM25, the standard RAG's choice of 2026-11-25) 55.6% (random 38.5%); image search 40.0% (random 39.4%); BM25 and image search fused 61.9% (random 38.7%). The other searches (dense, hybrid, with and without the MedCPT cross-encoder) are compared in the same files.

## 10. Data quality checks and what they show

`python -m src.main features` trains small classifiers on handcrafted features, to detect shortcuts. It is an analysis of the data, not an experiment on the LLMs. Test split, accuracy (chance 12.5%) and macro AUC ✔ (`results/data_quality/feature_check_results.json`, laptop run of 2026-11-23):

| Features | Accuracy | AUC |
|---|---|---|
| Original image width, height and file size only (a shortcut) | 31.7% | 0.731 |
| Age, sex, site (with missing flags) | 30.4% | 0.748 |
| Age, sex, site (without missing flags) | 29.6% | 0.739 |
| Image colour, texture, shape | 52.5% | 0.869 |
| Image + metadata | 55.4% | 0.892 |

Source detectable from the **standardised** image features: AUC 0.953 (0.5 = not detectable).

## 11. Known limitations

1. **Source confound.** All MEL and NV come from the older ISIC archive; the other classes come mostly from HAM10000. Even after standardisation the source is detectable (AUC 0.953), and the original file size alone predicts the class with 31.7% accuracy (chance 12.5%). Only the Kaggle dataset is used, so no images are added: state this as a limitation of the image results.
2. **Missing patient facts are not random.** Missing age, sex or site occurs mostly in NV and MEL, so "not recorded" hints at the class (15 of the 28 test cases with a missing fact are NV). State this as a limitation of the text-only results.
3. **Small test set.** 240 cases (30 per class, 190 lesions). Always report 95% bootstrap intervals (`src/evaluation/metrics.py` draws whole lesions).
4. **Balanced evaluation splits.** Accuracy and recall are comparable across classes; precision is not a clinic's precision.
5. **Corpus imbalance, weak labels and overlap with the reference texts.** The corpus holds 4 times more melanoma-tagged passages than dermatofibroma-tagged ones, 11.2% of the passages have no tag, and the tags come partly from class names, so retrieval metrics measure topical relevance, not proof. 46 textbook passages come from the pages that supply the reference texts of the semantic similarity (section 9.2).
6. **Licence.** ISIC images are mostly CC BY-NC: non-commercial research only.

## 12. Licences, distribution and research ethics

| Part | What it is | Terms | In Git |
|---|---|---|---|
| `raw/`, `processed/images/` | dermoscopic images (Kaggle copy of the ISIC Archive) | each image carries its own ISIC licence (mostly CC BY-NC, some CC0): non-commercial research, attribution | **no** (laptop only) |
| `annotations/*.csv` | ISIC 2019 ground truth and metadata, HAM10000 metadata, ISIC Archive records | the ISIC and HAM10000 data licences (non-commercial, attribution) ◻ | yes, with attribution (below) |
| `processed/cases/`, `views/` | derived from the two rows above: ISIC ids, labels, age, sex, body site | the same terms as their sources | yes, with attribution |
| `corpus/text_only/passages.jsonl`, `cache/` | full texts of PubMed abstracts, PMC papers, guidelines and Dermoscopedia pages | each passage's own licence | **no** (laptop only) |
| `corpus/text_only/manifest.csv`, `corpus_card.json`, `audit.json`, `corpus/image_only/visual_card.json` | lists and counts, no text | – | yes |
| `annotations/reference_texts.json` | quoted Dermoscopedia texts, each with its source | CC BY-NC-SA 4.0: attribution, non-commercial, shared under the same licence | yes |
| `.env` | API keys | secret | **never** |

**Attribution:** images and metadata from the ISIC Archive (isic-archive.com), including HAM10000 (Tschandl et al. 2018, Scientific Data 5:180161) and the ISIC 2019 challenge data; Kaggle copy "Skin Cancer ISIC – 9 classes" (nodoubttome). ◻ The exact licence of each source file is to be confirmed on its page before publication.

**Ethics.** All data are public and de-identified (no names, no dates of birth; age, sex and body site only), and no patient is contacted, so no new consent is involved. ◻ Whether the university's ethics committee must still register or exempt the study is to be confirmed with the supervisor. The LLM runs send images and patient facts to the providers' servers (Anthropic, OpenAI, Google, and Parasail through OpenRouter); this is research use of non-commercial data, and the method chapter should say so. The repository is to be public after publication (researcher's statement, 2026-11-22); `.claude/` is removed from it at the end (`archive/` was removed on 2026-11-24) (see `.claude/CLAUDE.md`).

## 13. How to rebuild and verify

```powershell
python -m src.main preprocess     # step 1 (minutes; needs data/raw/ on the laptop)
python -m src.main views          # step 2 (seconds)
python -m src.main corpus         # step 3 (internet; downloads are cached)
python -m src.main references --token-check    # step 3b (internet)
python -m src.main status         # where each step stands: done, not run, or a problem and why
python -m src.main features       # the shortcut analysis (section 10)
python -m unittest                              # all four test files; tests/system.py: the committed tables and views, rebuilt byte for byte
```

## 14. Corrections made to this document

| Date | Correction |
|---|---|
| 2026-11-25 | Sections 1, 9 and 9.2 brought up to the committed corpus of 2026-11-24 (3,426 passages; they still described the 2026-11-23 build of 3,444). Captions in the text index (9.3). |
| 2026-11-25 | Section 9.3, the image library, from `visual_card.json` and the validation results. |
| 2026-11-26 | The library is split into one folder per view (`data/corpus/text_only/`, `image_only/`, `multimodal/`; section 9). The files did not change; the index of the text library must be rebuilt because it records where the captions are. |
| 2026-11-25 | New folder structure (section 8): paths, commands (`python -m src.main`) and the move record `views/provenance.json`. No data changed. |
| 2026-11-24 | Added the retrieval index and the corpus audit (sections 9.1 and 9.2). Restructured under the data-sheet headings. Corrected stale figures: the source-detection AUC (0.935 → 0.953), the shortcut score (33% "balanced accuracy" → 31.7% accuracy), the test-set size ("16 to 68 per class" → 30 per class, 240 cases). Added sections 3.2 to 3.4, 9 and 10 with figures read from the committed files. Added the class-balance record (section 6). |
| 2026-11-23 | Split sizes updated after the balanced validation split (160 cases, 20 per class). |
| 2026-11-22 | First version after the cleaning and the balanced test split. |
