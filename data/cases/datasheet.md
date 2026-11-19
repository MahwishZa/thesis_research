# Data sheet: the case dataset (version 2: published case reports with the same patient's photos)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come
from the files `python -m src.build.cases.prepare` writes: `dataset_report.md` (one section per step: `count`, `download`, `check`,
`build`), the build records in `build/` (`figure_count.csv`, `figure_files.csv`, `figure_checks.csv`) and the questions in
`dataset.csv`. This is the thesis's multimodal dataset: a written case with the same patient's photos.

## At a glance

| | |
|---|---|
| Questions | 324: every case of the text dataset (version 3) whose own published article shows the lesion in at least one photo that passed every check |
| Test, validation, train | 100, 38, 186 (the text dataset's splits, unchanged; no article is in two splits) |
| Classes | the 8 of the other datasets; actinic keratosis has 1 test case and is reported apart (as in the text dataset) |
| Text | the text dataset's frozen question of the case, word for word (the case description up to the diagnosis) |
| Photos | 451, 1 to 8 per question (231 questions have one); published with the case, byte for byte; mostly clinical photos, some dermoscopy |
| Link between photo and text | the same patient: both come from the same case report, which holds no other case of the dataset |
| Asked as | text and photos (dataset `cases`), photos only (`case_photos`); the text only is the text dataset, on the same cases |
| Captions | never shown (they often name the diagnosis) |

## 1. Motivation

The Version 1 image datasets (ISIC photos, removed from the working folders on 2026-11-19) showed a photo with no clinical description,
because the ISIC Archive keeps no text about a patient, and no authentic source links an ISIC photo to a written case. This dataset
gives what they could not: a real patient's
written case together with that patient's own photos, both from the published case report. It answers whether a model diagnoses
better, and invents less, when it sees the photos with the written case, and how the photos alone compare, on the same patients.

## 2. Composition

- **Questions:** one per case. Its text is the text dataset's question (same file and fingerprint); its photos are listed in
  `dataset.csv` as `file|sha256`, in the order of their figures.
- **Labels:** the text dataset's labels, unchanged (see its data sheet: automatic, checked against PubMed's headings and a test
  result in the case that names no other class; about 3–7% wrong or mixed in a reading of 74 questions of its version 2).
- **What is not here:** no synthetic or rewritten text, no generated or edited photo, no pairing across sources, no caption.

## 3. Collection

| Step | What it did | Result |
|---|---|---|
| count | read the captions of each case's article in PubMed Central; a figure counts when its caption shows the lesion (a clinical or dermoscopic photo, or a lesion) and names no microscope or stain, scan, state after treatment, removed specimen, chart, or another patient; an article with another case of the dataset is left out | 107 test cases (rule fixed before the count: at least 80, and 6 classes with at least 8) |
| download | the counted figures' pictures from PMC Article Datasets (NCBI's cloud copy of PubMed Central's open-access articles), each checked against PMC's MD5 checksum; not open access or retracted: nothing | 492 pictures for 339 of 347 cases |
| check | each picture by itself: at least 128 pixels; no writing that names a diagnosis (EasyOCR); seen as a clinical or dermoscopy photo by an image model (open_clip ViT-B-32); not the same picture as another case's (on 2026-11-18 it was also compared with the photos of the Version 1 image dataset, with no match; that comparison left the code on 2026-11-19 with the dataset) | 468 kept; 24 not a skin photo, 1 names a diagnosis |
| build | one question per case with a kept picture that the text dataset still holds | 324 questions (11 cases left out with the text dataset's version 3) |

## 4. Uses and limits

- **Same patients across modalities:** the text-only, photos-only and text-and-photos answers are paired question by question
  (`python -m src.evaluate compare` pairs a text run with a case run on the shared cases).
- **Small test set:** 100 cases; a 95% interval on accuracy is about ±10 points, so only large differences are clear.
- **Actinic keratosis:** 1 test case; scores are also reported without it.
- **Picture type:** the captions and the image model agree poorly on whether a picture is a dermoscopy image (Cohen's kappa 0.27),
  so the photos are described only as published clinical and dermoscopic photographs, and picture type is not analysed.
- **Published cases:** often unusual presentations, photographed and cropped for publication, sometimes with arrows or panel letters
  (83 of the 492 checked pictures carry some writing; none of the kept ones names a diagnosis).
- **Models may have read the articles:** the text dataset's data sheet gives the publication years; results on cases published after
  a model's training cutoff show how much memory matters.
- **No person checked the photos.** The checks are automatic; their counts are in the check section of `dataset_report.md`.

## 5. Distribution

- In Git: `dataset.csv`, `dataset_report.md`, the build records in `build/` and this data sheet. The captions and photos stay on this
  computer (`data/cases/figures/`).
- Licences: each photo under its article's licence (recorded per picture in `build/figure_files.csv`: mostly CC BY and CC BY-NC); used for
  research only and not redistributed.

## 6. Maintenance

- **Rebuild:** with the text dataset prepared, run in order `python -m src.build.cases.prepare count`, `download`, `check`, `build`.
  `count` and `download` need the internet; `check` needs the packages in `requirements.txt` and downloads its two models once.
- **Fingerprints:** every question's text and every photo has its SHA-256 in `dataset.csv`; every run checks the files against them.
  `dataset.csv` itself: SHA-256 `cee3aa4b2699140bd59862c89d0fcd44fb959c84e45cb4bac1248ac642ac509f`.
- **Versions:** a change to a question, photo or split makes a new version, with an entry here.
  - Version 1 (2026-11-18): 335 questions (104 test), built on version 2 of the text dataset.
  - Version 2 (2026-11-18): 324 questions (100 test). The text dataset's version 3 took out 29 questions whose confirming test
    result also names another class; 11 of them had kept pictures, so the build leaves them out too (listed in the build section
    of `dataset_report.md`). Every other question, photo and split is unchanged, byte for byte.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for
  datasets. *Communications of the ACM*, 64(12), 86–92.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
