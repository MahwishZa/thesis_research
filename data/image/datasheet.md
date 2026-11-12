# Data sheet: the skin-lesion image dataset (version 1)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Every number in it comes from three files the preparation code (`src/image/prepare.py`) writes:
- `dataset_report.md`: the full report of every step
- `raw_photos.csv`: one row per original photo
- `dataset.csv`: one row per prepared image

## At a glance

| | |
|---|---|
| Images | 2,179 dermoscopic photographs of skin lesions, one diagnosis each |
| Classes | 8: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion |
| Splits | test 320 (40 per class), validation 120 (15 per class), train 1,739; no lesion is in two splits |
| Labels | from each photo's own ISIC Archive record; 74% confirmed by biopsy |
| Images as stored | JPEG in sRGB colours, each whole photo fitted within 600 × 450 pixels (2,013 are exactly 600 × 450) |
| Built from | "Skin Cancer ISIC" on Kaggle, by Andrey Katanskiy (https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic): 2,357 ISIC images in nine class folders, downloaded on or before 2026-11-03 |
| Version | 1, frozen on 2026-11-12; marked in Git by the tag `image-dataset` |

## 1. Motivation

**Purpose.** The dataset was prepared for an MS thesis on diagnosing skin lesions from dermoscopic images with large language models (LLMs). The models are tested in two ways:
- closed-book: the model sees only the image
- open-book: retrieval-augmented generation (RAG), where examples with their diagnoses are retrieved from the training split

**Why a prepared version was needed.** The Kaggle dataset as downloaded had problems that would distort such a study:
- 157 groups of identical files, 18 of them with a copy in both the Train and the Test folder
- 155 photos filed under two different classes
- a seborrheic keratosis folder that held only melanomas
- image size giving away the class
- a small, uneven test folder: 118 photos, only 3 of them vascular lesions
- no lesion information to keep photos of one lesion out of both train and test

**Who.** Prepared by Mahwish Zafar for her MS thesis.

## 2. Composition

**Instances.** Each image is a dermoscopic photograph of one skin lesion: every ISIC record gives the image type "dermoscopic". No person can be identified from an image.

**Images per class and split, how the diagnosis was confirmed, and where the photos come from**

| Class | Test | Val | Train | Images | Biopsy | Expert consensus | Confocal microscopy | Not recorded | Credit "MILK study team" (also in HAM10000) | Credit "Anonymous" |
|---|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 40 | 15 | 76 | 131 | 131 | | | | 129 | 2 |
| basal cell carcinoma | 40 | 15 | 337 | 392 | 392 | | | | 392 | |
| dermatofibroma | 40 | 15 | 56 | 111 | 55 | 56 | | | 104 | 7 |
| melanoma | 40 | 15 | 381 | 436 | 436 | | | | | 436 |
| nevus | 40 | 15 | 238 | 293 | | | | 293 | | 293 |
| pigmented benign keratosis | 40 | 15 | 423 | 478 | 332 | 103 | 43 | | 478 | |
| squamous cell carcinoma | 40 | 15 | 141 | 196 | 196 | | | | 191 | 5 |
| vascular lesion | 40 | 15 | 87 | 142 | 67 | 75 | | | 142 | |
| **Total** | **320** | **120** | **1,739** | **2,179** | **1,609** | **234** | **43** | **293** | **1,436** | **743** |

**Subtypes kept within a class**
- **Melanoma:** 157 in situ, 151 invasive and 128 not otherwise specified.
- **Nevus:** mostly "Nevus" without a subtype (150), then compound, dysplastic, Clark and congenital nevi, and a few others.
- **Squamous cell carcinoma:** 191 not otherwise specified and 5 in situ.

The split keeps these subtypes balanced between test, validation and train.

**What comes with each image**
- **`dataset.csv`:**
  - its split, class, file and SHA-256
  - its lesion group
  - whether it was in the Kaggle Test folder (`kaggle_test`)
  - its size and what was done to it
- **`raw_photos.csv`:** everything known about the original photo, such as:
  - the full ISIC diagnosis and how it was confirmed
  - the lesion number, licence and credit line
  - the HAM10000 cross-check
- **Not used in this phase:** ISIC also records age, sex and body site. These stay in the downloaded records for the later text phase.

**Photos of one lesion.**
- **Groups:** the 2,179 images form 1,804 lesion groups, 309 of which hold two or more photos.
- **How a group is formed:** from ISIC's lesion numbers (1,446 images have one), HAM10000's lesion numbers, and the eye-check verdicts.
- **In the test split:** 320 images show 248 lesions, and 129 test images share their lesion with another test image. The evaluation's 95% intervals therefore resample whole lesions.
- **By patient:** grouping by patient is not possible, because only 10 images have a patient number.

**Recommended splits.** The splits are fixed and part of the dataset (the `split` column). They must not be redrawn.

**Licences.** Each image carries the licence of its ISIC record (`licence` and `attribution` in `raw_photos.csv`):
- CC-BY-NC: 1,436 images, credit "MILK study team"
- CC-0: 743 images, credit "Anonymous"

## 3. Collection

- **Source.** "Skin Cancer ISIC" on Kaggle, by Andrey Katanskiy (https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic): 2,357 JPEG files in `Train/` and `Test/` folders with nine class folders.
  - **Downloaded:** on or before 2026-11-03, the date of the first committed list of the files' fingerprints. The files have not changed since.
  - **What its page says:** the images were sorted by ISIC's classification. Section 4 shows where the ISIC records disagree with the folders.
  - Every file name carries an ISIC number: 2,202 numbers in all, every one found in the ISIC Archive.
  - The files are kept unchanged in `data/image/raw`. Their fingerprint is `0915dd82f562136d1151b61be6181e8fd1ab6072dadd4d0f47f69dc4ddd0bfaa`, computed from the path and SHA-256 of every file.
- **ISIC Archive records.** The public record of every ISIC number was downloaded on 2026-11-11 from `api.isic-archive.com`. It gives the diagnosis, how it was confirmed, the image type, lesion number, licence and credit line. Each record is fingerprinted in `raw_photos.csv`.
- **HAM10000 metadata.** Downloaded from Harvard Dataverse (doi:10.7910/DVN/DBW86T; Tschandl et al., 2018), and used as a second source for the 1,437 ISIC numbers that are also in HAM10000.
- **No new photos.** No photo was taken or added by the researcher. No person was contacted.

## 4. Preparation: every change from the Kaggle dataset

Each step is one command of `src/image/prepare.py`. Its full numbers are in the matching section of `dataset_report.md`.

| Step | What it did | Result |
|---|---|---|
| inventory | Opened and fingerprinted every file; identical files count as one photo | All 2,357 files readable; 157 groups of identical files, so 2,200 unique photos |
| isic | Downloaded the ISIC record of every ISIC number | 2,202 of 2,202 found |
| identity | Compared every photo with ISIC's own picture of its number | 2,202 of 2,202 the same picture |
| ham10000 | Compared the photos also in HAM10000 with HAM10000's metadata | 1,437 numbers: diagnosis group, confirmation and lesion groups agree for all |
| labels | Labelled every photo from its ISIC record by the rule in `isic_to_class.csv`, never from its Kaggle folder | 2,196 labelled; 4 photos show two lesions (ISIC "collision") and get no class |
| duplicates | Compared every photo with every other one, whole picture | 210 pairs flagged for the eye check |
| lesions | Looked for second photos of one lesion by skin pattern (OpenCV SIFT, ratio test, RANSAC) | 222 of the 446 known same-lesion pairs found, with 1 false alarm among 233,456 known different pairs |
| eye check | A person judged the flagged pairs (`eye_check_verdicts.csv`) | 17 pairs are the same photograph, so one copy of each is left out (an original test photo or the lower ISIC number is kept); 10 pairs are different lesions (3 of them added by the audit described in section 7) |
| splits | Train, validation and test, new, ignoring the Kaggle folders | See below |
| images | Made the image of every photo in a split, all the same way | See below |
| shortcuts | Measured what simple image properties reveal about the class | Measures only; see section 5 |

**Label changes from the Kaggle folders.** Every change follows the photo's ISIC record:
- **The seborrheic keratosis folder.** It held 80 unique photos, and ISIC's record of every one says melanoma, confirmed by biopsy.
  - 78 of them were identical copies of files also in the melanoma folder; the other 2 (ISIC_0011118, ISIC_0011126) are melanoma too.
  - All 80 stay in the dataset as melanoma: 66 train, 3 validation, 7 test, and 4 left out as copies of another photo.
  - No seborrheic keratosis photo is left, so the dataset has 8 classes instead of 9. Clinically, seborrheic keratoses belong to HAM10000's benign keratosis group, which is the pigmented benign keratosis class here. ISIC does not say which of those photos are seborrheic keratoses.
- **Actinic keratosis and nevus.** 77 photos filed under both are actinic keratosis. One photo filed under nevus (ISIC_0028619) is actinic keratosis too.
- **Not used:** 21 photos are in no split: the 4 collisions and the 17 copies.

**Splits**
- **Rules:**
  - every lesion group in one split (group-based)
  - exactly 40 test and 15 validation images per class (stratified)
  - the rest is train
- **How the split was made:** a random fill with seed 2026, fixed in advance, then swaps of equal-size lesion groups that bring test and validation closer to their class's mix of:
  - subtype
  - how the diagnosis was confirmed
  - credit line
  - original image size
- **Result:** the balance is 0.045, where 0 means each part mirrors its class. For comparison, 1,000 simple random splits under the same rules give a median of 0.100 and at worst 0.138.
- **The original Kaggle test photos:** the 118 photos are split like every other photo, into 19 test, 9 validation and 90 train.
- **Reproduced:** the same split came out on two computers (Python 3.11 and 3.12).

**Images.** Every photo in a split is made from its original file, in this order:
1. Turned upright, by its EXIF orientation tag (none needed turning).
2. Colours converted to sRGB. The 2 photos with a camera colour profile (ISIC_0000143 and ISIC_0000153) are converted from it.
3. The 8 portrait photos turned a quarter turn to landscape.
4. Fitted whole within 600 × 450, keeping their shape: shrunk with Lanczos filtering, never enlarged.
   - Nothing is cut and nothing is added.
   - A photo whose shape is 4:3 up to rounding is made exactly 600 × 450, by a stretch of less than one pixel (132 photos).
5. Saved as JPEG at quality 95 with full colour resolution (4:4:4), without metadata.

The result:
- **2,013 images** are exactly 600 × 450.
- **166 keep their own shape** (nevus 96, melanoma 58, dermatofibroma 7, squamous cell carcinoma 5).
- **How each was treated:** 743 photos were shrunk, and 1,436 were already 600 × 450 and only saved again.

Made with Pillow 13.0.0 (libjpeg-turbo 6.2, Little CMS 2.19).

**Not done:** no cropping, borders, colour correction, hair removal, sharpening or augmentation, and no photos added.

## 5. Uses

**Intended uses**
- **Test split:** the reported result, run once per model and setting.
- **Validation split:** tuning prompts and retrieval settings. It is never reported as a result.
- **Train split:** only the database that open-book runs retrieve from. Retrieval must never reach validation or test images.
- **The original Kaggle test photos** (`kaggle_test` = yes): they can be run on their own as a closed-book check against earlier results on the Kaggle Test folder. This doesn't work open-book, because 90 of them are in train.
- **Scoring:** balanced accuracy is the main score, with 95% intervals that resample whole lesions. The per-class recall and the confusion matrix are reported with it.

**Known limitations: state them with the results**
1. **The classes come from two collections.**
   - **The split:** melanoma and nevus come only from older ISIC collections (credit "Anonymous"), while the other six classes come almost entirely from HAM10000 (credit "MILK study team").
   - **What the shortcut check found:** from the colour of the image border alone, a nearest-neighbour vote gets 34.7% of test classes right (chance 12.5%) and 83.8% of the collections. All five simple properties together get 36.3% and 85.1%. Image size, file size, dark corners and sharpness are close to chance (13.8–16.6%).
   - **Where the clue can't help:** comparisons within one collection, meaning melanoma against nevus and the six HAM10000 classes among themselves.
2. **How the nevus diagnoses were confirmed** is not recorded for any of the 293 nevus images.
3. **166 images keep a shape other than 600 × 450,** almost all nevus and melanoma.
4. **Some second photos of one lesion may remain unfound** among the 733 images without an ISIC lesion number. The skin-pattern search found about half of the known same-lesion pairs, and a person checked every pair it flagged without a lesion number.
5. **Lesions, not patients, are kept apart:** two lesions of one patient can be in different splits.
6. **ISIC's diagnosis can itself be wrong in rare cases.** 74% are confirmed by biopsy.
7. **With 40 test images per class,** a single class's recall has a 95% interval of roughly ±15 percentage points. Balanced accuracy over all 320 is measured to about ±5–6 points.

**Not intended for**
- clinical decisions
- estimates of how common a disease is, or of real-world accuracy: the test split has equal classes, unlike any clinic
- conclusions about skin types, which are not recorded

## 6. Distribution

**What is shared and what isn't**
- **Not redistributed:** the images, in `data/image/raw` and `data/image/dataset`, stay on the researcher's computer and are not in Git.
- **In the repository:** the tables, the report, this data sheet and the code. Anyone with the Kaggle download can rebuild the dataset exactly.

**Licences and credit**
- Each image's licence is in `raw_photos.csv`: CC-BY-NC (non-commercial, with credit) or CC-0.
- Showing images in a publication must respect their licence and credit line.
- The Kaggle page's own terms should be checked too.

**Cite**
- the ISIC Archive
- HAM10000 (Tschandl et al., 2018)
- the Kaggle dataset

## 7. Maintenance and checking

**Version 1** is frozen and marked in Git by the tag `image-dataset`. Any change to an image, label or split makes a new version, with new fingerprints, a new tag and an entry in this data sheet. The text and multimodal datasets of later phases are documented separately.

**Changes since the tag.** On 2026-11-12 the repository was reorganised for three datasets: the image dataset's files moved from `data/` to `data/image/`, and its code to `src/image/prepare.py`. No image, label, split or fingerprint changed; the full re-run below proves it.

**Independent audit (2026-11-12).** A fresh check of the frozen tables found the dataset as described: no lesion in two splits, every label re-derived from its ISIC record, and every fingerprint matching. It found one gap: 3 same-class look-alike pairs from the whole-picture comparison sat in different splits and had never been looked at. A person checked them by eye: all 3 are different lesions. They are recorded in `eye_check_verdicts.csv`, and re-running the split with them gives exactly the same split.

**Rebuild**
1. Set up Python 3.11 or newer with the packages in `requirements.txt`.
2. Put the Kaggle `Train` and `Test` folders in `data/image/raw`.
3. Run, in order: `python -m src.image.prepare inventory`, `isic`, `identity`, `ham10000`, `labels`, `duplicates`, `lesions`, `splits`, `images`, `shortcuts`.

A full re-run on 2026-11-12 reproduced `raw_photos.csv`, `dataset.csv` and every image byte for byte. It also brought the same-lesion search up to date with the second eye check, removing the 14 pairs that involve the 13 copies it left out.

**Check.** Every image is checked against `dataset.csv` before each run; a changed, missing or extra image stops the run. Each split also has its own fingerprint, which a run's run card shows as `images_sha256` when the run answered the whole split:

| Split | Fingerprint |
|---|---|
| test | `cfa2c9e9b7c6c8d5c5ac6a399bc8b6dff033814f776f3a2abba9da408677fb15` |
| val | `0347a83b28f7d45d8db41cf2b1be8a0e40e0881adce3b3e9fe7b42d199c29cc9` |
| train | `9e435c5f7dae6a566db8c6bfa1848454daae6556a380903cbacd6eb87e04a20a` |

**Backup.** Keep a ZIP of `data/image/dataset` and the original Kaggle ZIP outside this computer.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- ISIC Archive, International Skin Imaging Collaboration: https://www.isic-archive.com
- Katanskiy, A. Skin Cancer ISIC (dataset). Kaggle. https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic (downloaded on or before 2026-11-03).
