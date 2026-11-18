# Data sheet: the case dataset (published case reports with the same patient's photos)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come
from the files `python -m src.build.cases.prepare` writes: `figure_count.md` (step `count`), `figure_files.md` (`download`),
`figure_checks.md` (`check`) and `cases.md` (`build`); the questions are in `cases.csv`.

## At a glance

| | |
|---|---|
| Questions | 335: every case of the text dataset whose own published article shows the lesion in at least one photo that passed every check |
| Test, validation, train | 104, 38, 193 (the text dataset's splits, unchanged; no article is in two splits) |
| Classes | the 8 of the other datasets; actinic keratosis has 1 test case and is reported apart (as in the text dataset) |
| Text | the text dataset's frozen question of the case, word for word (the case description up to the diagnosis) |
| Photos | 468, 1 to 8 per question (237 questions have one); published with the case, byte for byte; mostly clinical photos, some dermoscopy |
| Link between photo and text | the same patient: both come from the same case report, which holds no other case of the dataset |
| Asked as | text and photos (dataset `cases`), photos only (`case_photos`); the text only is the text dataset, on the same cases |
| Captions | never shown (they often name the diagnosis) |

## 1. Motivation

The image and image + metadata datasets show a photo with no clinical description, because the ISIC Archive keeps no text about a
patient, and no authentic source links an ISIC photo to a written case. This dataset gives what they cannot: a real patient's
written case together with that patient's own photos, both from the published case report. It answers whether a model diagnoses
better, and invents less, when it sees the photos with the written case, and how the photos alone compare, on the same patients.

## 2. Composition

- **Questions:** one per case. Its text is the text dataset's question (same file and fingerprint); its photos are listed in
  `cases.csv` as `file|sha256`, in the order of their figures.
- **Labels:** the text dataset's labels, unchanged (see its data sheet: automatic, checked against PubMed's headings and a test
  result in the case; about 3–7% wrong or mixed in a reading of 74 questions).
- **What is not here:** no synthetic or rewritten text, no generated or edited photo, no pairing across sources, no caption.

## 3. Collection

| Step | What it did | Result |
|---|---|---|
| count | read the captions of each case's article in PubMed Central; a figure counts when its caption shows the lesion (a clinical or dermoscopic photo, or a lesion) and names no microscope or stain, scan, state after treatment, removed specimen, chart, or another patient; an article with another case of the dataset is left out | 107 test cases (rule fixed before the count: at least 80, and 6 classes with at least 8) |
| download | the counted figures' pictures from PMC Article Datasets (NCBI's cloud copy of PubMed Central's open-access articles), each checked against PMC's MD5 checksum; not open access or retracted: nothing | 492 pictures for 339 of 347 cases |
| check | each picture by itself: at least 128 pixels; no writing that names a diagnosis (EasyOCR); seen as a clinical or dermoscopy photo by an image model (open_clip ViT-B-32); not the same picture as another case's or one of the image dataset | 468 kept; 24 not a skin photo, 1 names a diagnosis |
| build | one question per case with a kept picture | 335 questions |

## 4. Uses and limits

- **Same patients across modalities:** the text-only, photos-only and text-and-photos answers are paired question by question
  (`python -m src.evaluate compare` pairs a text run with a case run on the shared cases).
- **Small test set:** 104 cases; a 95% interval on accuracy is about ±9 points, so only large differences are clear.
- **Actinic keratosis:** 1 test case; scores are also reported without it.
- **Picture type:** the captions and the image model agree poorly on whether a picture is a dermoscopy image (Cohen's kappa 0.27),
  so the photos are described only as published clinical and dermoscopic photographs, and picture type is not analysed.
- **Published cases:** often unusual presentations, photographed and cropped for publication, sometimes with arrows or panel letters
  (83 pictures carry some writing; none names a diagnosis).
- **Models may have read the articles:** the text dataset's data sheet gives the publication years; results on cases published after
  a model's training cutoff show how much memory matters.
- **No person checked the photos.** The checks are automatic; their counts are in `figure_checks.md`.

## 5. Distribution

- In Git: `cases.csv`, the reports and this data sheet. The photos stay on this computer (`data/multimodal/figures/images/`).
- Licences: each photo under its article's licence (recorded per picture in `figure_files.csv`: mostly CC BY and CC BY-NC); used for
  research only and not redistributed.

## 6. Maintenance

- **Rebuild:** with the text dataset prepared, run in order `python -m src.build.cases.prepare count`, `download`, `check`, `build`.
  `count` and `download` need the internet; `check` needs the packages in `requirements.txt` and downloads its two models once.
- **Fingerprints:** every question's text and every photo has its SHA-256 in `cases.csv`; every run checks the files against them.
- **Versions:** a change to a question, photo or split makes a new version, with a new tag and an entry here.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for
  datasets. *Communications of the ACM*, 64(12), 86–92.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
