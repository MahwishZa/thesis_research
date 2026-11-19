# Data sheet: the clinic patients cohort (PAD-UFES-20), draft of 2026-11-19

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. **It is a draft.
Sections 1 to 3 describe the download as received (numbers from the `inventory` step). Section 4 states the rules the preparation
steps follow, decided on 2026-11-19; the figures they produce (the splits, the photo standardisation, the records) are in
`dataset_report.md` after the first full run on the researcher's computer, and are copied here only then.**
Every number below comes from files that `python -m src.build.clinic_patients.prepare` writes: `dataset_report.md`,
`build/columns.csv`, `build/raw_images.csv`. The photos and `metadata.csv` stay on the researcher's computer (research folder
`data/clinic_patients/raw/`); only the small tables are in Git.

Marks: ✔ checked in the researcher's download by the inventory, or stated in the authors' paper (the *Data in Brief* article, vol. 32, 2020, article 106221, and the arXiv version 2007.00478v2, read on 2026-11-19; the two agree on everything used here, see section 2) and agreeing with the download; ◻ not yet checked (a web page only, or not stated by the authors).

## At a glance

| | |
|---|---|
| What it is | Smartphone photos of skin lesions, each with the clinical record of the same patient visit (age, body site, symptoms, history) |
| Photos | 2,298 PNG files, 147 to 3,474 pixels wide (median 779.5), from different phones ✔ |
| Patients, lesions | 1,373 patients ✔; 1,891 lesions ✔ (a lesion is a patient + `lesion_id` pair: `lesion_id` alone is shared between patients, and its 1,641 distinct values are not the lesion count. The paper reports "1,641 skin lesions": that is the number of distinct `lesion_id` values ✔, so the authors counted the numbers, not the patient + lesion pairs; this dataset counts pairs. The ISIC Archive lists 1,891 lesions for this collection, a search result ◻) |
| Classes | 6: basal cell carcinoma (BCC) 845, actinic keratosis (ACK) 730, nevus (NEV) 244, seborrheic keratosis (SEK) 235, squamous cell carcinoma (SCC) 192, melanoma (MEL) 52 photos ✔; Bowen's disease is counted as SCC by the authors ("SCC in situ"; paper, section Data selection) ✔ |
| Label source | The authors' `diagnostic` column. All BCC, SCC and MEL photos (1,089) are biopsy-proven ✔; ACK, NEV and SEK are biopsy-proven for 178 of 730, 60 of 244 and 15 of 235 photos ✔, the others by the consensus of the dermatologists at the appointment (paper: for these benign lesions a biopsy "is not necessary") ✔. 1,342 of 2,298, 58.4%, are biopsy-proven ✔, and the paper's Table 1 gives the same percentages: ACK 24.4%, NEV 24.6%, SEK 6.4%, total 58.4% ✔ |
| Source | Kaggle "Skin Cancer(PAD-UFES-20)" (mahdavi1202), a copy of the authors' Mendeley Data upload, version 1 (doi 10.17632/zr7vgbcyr2.1 ✔: the article gives the Mendeley address and "Mendeley Data, v1"), published 2020-07-07 ◻ (the date is not in the article). Not compared byte for byte with Mendeley ◻; its counts equal the article's ✔ |
| Licence | CC BY 4.0 ✔ (the Mendeley Data page, as read by the researcher: "The files associated with this dataset are licensed under a Creative Commons Attribution 4.0 International licence"; the Kaggle copy says the same). Sharing, copying and changing are allowed with credit to the authors, a link to the licence, and a note of any changes made (here: the photos are standardised and the tables cleaned), and without suggesting that the authors endorse this work. The Mendeley page adds that permission may be needed for content belonging to a third party. The *article* is also open access under CC BY 4.0 ✔ |
| Citation | Pacheco AGC, Lima GR, Salomão AS, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief* 32 (2020) 106221, doi 10.1016/j.dib.2020.106221 ✔ (received 9 July, accepted 21 August, online 25 August 2020; the issue month is not on the article's pages ◻). Mendeley Data, V1, doi 10.17632/zr7vgbcyr2.1 ✔ |
| Version of this cohort | none yet (nothing is frozen) |

## 1. Motivation

The Version 1 photo dataset had no clinical record next to its photos, and its text dataset described other patients. PAD-UFES-20
gives photos and a clinical record of the same visit, from one clinic, so the image, the text and both together can be compared on
the same patients. See `docs/v2_proposal.md`.

## 2. Composition

**One row of `metadata.csv` is one photo** (`img_id`, `PAT_<patient>_<lesion>_<n>.png`; the authors write "each line represents a skin
lesion", but the 2,298 rows hold 1,891 lesions ✔). A patient has 1 to 8 lesions ✔ (1,018 patients have one). The table has 26 columns ✔.
**Meanings are the authors' own definitions** (Pacheco et al. 2020, section "Meta-data", copied by the researcher from the arXiv version
2007.00478 on 2026-11-19, and compared the same day with the published *Data in Brief* article, Table 1 ✔: the definitions the record uses (itch, hurt, "has bled", "has an elevation", age, region, `biopsed`) are the same words in both. The article counts "up to 21 features" (arXiv: 22) and says the table has 26 attributes: 21 clinical features, four identifying ones and the diagnosis ✔, as in the download. It also provides the medical questionnaire as supplementary material, which may explain the blank block ◻ (not read yet)).

| Column | Filled / empty ✔ | Values seen ✔ | Meaning (the authors) |
|---|---|---|---|
| `patient_id`, `lesion_id`, `img_id` | all filled | ids | patient, lesion number, photo (`img_id` joins patient, lesion and a random number). Always present |
| `diagnostic` | 2,298 / 0 | ACK, BCC, MEL, NEV, SCC, SEK | the label |
| `biopsed` | 2,298 / 0 | TRUE 1,342, FALSE 956 | "if the diagnostic comes from a clinical consensus or biopsy" (the wording does not say which value is which, but TRUE is biopsy ✔: 1,342 TRUE is 58.4% of 2,298, the paper's share of biopsy-proven lesions, and all 1,089 BCC, SCC and MEL are TRUE). **Never shown to a model** (it points to the label). Always present |
| `age` | 2,298 / 0 | 6 to 94, median 62 | an integer, the patient's age (the unit is not written; the paper speaks of an "average age approximately 60 years old", and the median here is 62, so the values are years ✔). Always present |
| `region` | 2,298 / 0 | 14 body regions (FACE 570, FOREARM 392, CHEST 280, BACK 248, ARM 192, NOSE 158, HAND 126, …) | "one of the 15 macro-regions previously described" (the paper lists the 15: face, scalp, nose, lips, ears, neck, chest, abdomen, back, arm, forearm, hand, thigh, shin, foot; the download holds 14, and **SHIN** is the one absent ✔: no photo of the shin is in the data). Always present |
| `itch`, `hurt`, `bleed`, `elevation` | 2 to 10 unknown (`UNK`) | TRUE / FALSE | the lesion itches; hurts; "has bled"; "has an elevation" |
| `grew`, `changed` | 402 and 396 unknown (`UNK`) | TRUE / FALSE | the lesion "has recently grown"; "has recently changed" |
| `gender`, `smoke`, `drink`, `pesticide`, `skin_cancer_history`, `cancer_history`, `has_piped_water`, `has_sewage_system`, `fitspatrick` (1 to 6), `diameter_1`, `diameter_2` | 804 blank each | TRUE / FALSE, FEMALE / MALE, numbers | the patient smokes cigarettes, drinks alcohol, uses pesticides; skin cancer, or any cancer, in the past **in the patient or in the family** (one field each); piped water and a sewage system at home; the Fitzpatrick skin type; the lesion's horizontal and vertical diameters (floats) |
| `background_father`, `background_mother` | 913 and 906 empty or `UNK` | **countries** (POMERANIA, GERMANY, ITALY, BRAZIL, …; "BRASIL" and "BRAZIL" both occur) | the country the father's and the mother's family descends from (Pomerania is kept although it is a region: patients call themselves Pomeranian). They are places of origin, not diseases, as the Kaggle page wrongly says |

**How gaps are written, and what they mean.** The authors: "Missing values are left blank ... When a patient does not know the answer ... we
fill the feature as UNK (unknown)", and only `patient_id`, `lesion_id`, `img_id`, `age`, `region` and `biopsed` are always present; the
rest "depend on the patient's answers during the appointment". In the download, the 804-photo block is blank and every other gap is `UNK` ✔.
So **a gap in the four symptom fields of the record is `UNK`, which means the patient did not know**, not that nothing was asked. The
cleaned table writes both as `not recorded`; `build/columns.csv` and the inventory keep the two apart. The authors do not say why the
804-photo block is blank, so the cause of finding 1 stays a guess ◻ (the paper does not mention it either; it says only that the features "depend on the patient's answers"). Diameters run from 0 to 100 (`diameter_1`) and 0 to 70
(`diameter_2`); their unit is not stated, in the paper either ◻ (a diameter of 0 occurs).

## 3. Findings that shape the design ✔

1. **An empty record points to "not a cancer".** 804 photos from 579 patients have all 13 of these fields empty together:
   `smoke`, `drink`, `background_father`, `background_mother`, `pesticide`, `gender`, `skin_cancer_history`, `cancer_history`,
   `has_piped_water`, `has_sewage_system`, `fitspatrick`, `diameter_1`, `diameter_2`. By class: ACK 447, NEV 169, SEK 188, **BCC 0,
   MEL 0, SCC 0**. Another 990 photos have no empty field at all (ACK 242, BCC 542, MEL 34, NEV 32, SCC 110, SEK 30). So a model
   (or a retrieval or filter) that learns "record empty" gets the cancer / not-cancer split without looking at the lesion. A likely
   cause is that the questionnaire grew over the years of collection; the paper does not say so (checked 2026-11-19), so this stays a guess ◻.
2. **"Unknown" in `grew` and `changed` leans the other way.** They are unknown together for 345 photos: BCC 243, SCC 67, NEV 23,
   MEL 6, SEK 5, ACK 1.
3. **Photo size differs by class.** Median file size: NEV 604 KB, SEK 734 KB, MEL 998 KB, BCC 1,036 KB, ACK 1,050 KB, SCC 1,155 KB;
   median width 690 (NEV) to 860 (MEL) pixels. A check on image properties alone is needed, as for Version 1.
4. **`lesion_id` is not unique**: 250 values are used by more than one patient (for example `1003` is a nevus of `PAT_1288` and an
   actinic keratosis of `PAT_38`). Anything grouped by lesion must group by patient + `lesion_id`; **splits must be by patient**.
5. **Fifteen groups of photos have identical content**, always under one patient but under different lesion numbers, and in only 3
   of the 15 groups with the same diagnosis. In the other 12 groups one picture has two different correct labels, so it cannot be a
   one-answer question as it stands.
6. **Some classes are small.** Melanoma has 52 photos of 36 patients: a test split will hold about 15, so its recall has a wide
   interval.

## 4. The preparation steps and the rules they follow

One command per step (`python -m src.build.clinic_patients.prepare <step>` or `all`); each writes its decisions to tables in
`build/` and its own section of `dataset_report.md`, and **changes no original file**. The rules were fixed before the numbers were
seen.

| Step | What it does |
|---|---|
| `inventory` | Counts what the download holds (above). Changes nothing. |
| `clean` | Writes a gap as `not recorded` (no gap is filled in, no row left out); unifies case, spaces and the one spelling variant (`BRASIL`); **stops** at a value outside the known vocabulary; **flags** (keeps) implausible values and disagreements inside a patient or lesion. |
| `dedupe` | Finds identical and near-identical photos with a 256-bit difference hash. One picture with one label: the smallest `img_id` stays. One picture with conflicting labels: all are dropped (finding 5). Patients with a near-identical photo are kept together in the next step. |
| `split` | Train 60%, validation 10%, test 30%, **by patient** and class by class (a patient is never in two splits; patients joined by a near-identical photo are one group). Seed 20261119, chosen before the first draw. Classes are written by name from here on. |
| `images` | Every photo that stays is made in one fixed way: upright by its EXIF tag, RGB, sRGB colours, longest side at most 1,024 pixels (never enlarged), JPEG quality 95 without metadata. Bars checked for every photo (the step stops otherwise): fidelity at least 30 dB PSNR, the photo's own original is the nearest of all originals by picture hash, no file over 4 MB. Reason: 108 originals are over 5 MB, and every model shrinks a large photo in its own way. |
| `views` | Writes the record text of every question and the tables of the three views (below). |
| `shortcuts` | How much the gaps and the photo properties alone give away, with a nearest-neighbour vote on the train split only, so that a hint that is not the lesion is reported next to the results. |

**The three views of the same questions.** One question is one photo of one patient's lesion with the record of the same visit.
- *photo*: the standardised photo alone (`dataset.csv`, `files/photos/`);
- *record*: the record alone, as text (`record.csv`, `files/records/`);
- *photo + record*: the photo with the same record text in the prompt (`dataset.csv`, column `facts`).

**The record** (`record-v1`) is six numbered lines in fixed wording: age; body site; whether the lesion itches, hurts, has bled and has an
elevation. The lines use the authors' own words for each field (itch, hurt, "has bled", "has an elevation"; section 2); the age is
in years ✔ (section 2). A gap in a symptom line reads `unknown`: in the authors' words the patient did not know the answer
(section 2). Age and body site are always present; a gap in either would stop the step. The record is the same words, to the byte, in
the record view and in the photo + record view. **Not in any view:** `grew` and `changed` (unknown more
often for cancers: finding 2) and the block of 13 fields of finding 1 (empty for 804 early photos, all benign). They stay in
`dataset.csv` for a separate secondary analysis on complete records. A change of wording is a new template version and a rebuild of
the views.

**Photos of one patient** are not independent: intervals resample whole patients. Identical records are kept in the record view
(they are different patients, or several photos of one visit), so that the three views ask the same questions.

### What the first full run found (2026-11-19, on the researcher's computer) ✔

All numbers are in `dataset_report.md` and the tables of `build/`; each step passed its own checks.

- **clean:** 11,485 gaps written `not recorded`, 3 values normalised, 8 flags kept as recorded.
- **dedupe:** 16 clusters of the same picture (32 photos). 13 clusters carry two diagnoses (26 photos dropped), 3 carry one (one photo of
  each stays, 3 dropped): **29 photos dropped, 2,269 stay.** Nearest-neighbour distances show a clean gap: 32 photos at 0 bits, none at 1
  to 16, the next at 17 bits. No two patients have a photo within 16 bits.
- **split** (seed 20261119): train 1,368 photos of 654 patients, validation 223 of 219, test 678 of 498. Class shares are within 60% / 10%
  / 30% by at most 2 points for every class except squamous cell carcinoma (62.5 / 8.7 / 28.8) and melanoma (61.5 / 9.6 / 28.8; 32, 5 and
  15 photos). No patient is in two splits. The patients with many photos went to train (2.1 photos per patient against 1.0 in
  validation and 1.4 in test), so validation and test are mostly single-photo patients.
- **images:** 2,269 photos made. The 104 originals over 5 MB (largest 23.6 MB) became files of at most 0.77 MB; 814 photos were shrunk
  to 1,024 pixels (the median longest side, 778, is unchanged), 1,420 PNG files with an opaque alpha channel were converted to RGB and 2
  from a colour profile to sRGB. Fidelity 39.8 / 45.8 / 62.0 dB (smallest / median / largest, bar 30). Every saved photo is nearest to its
  own original: 0 to 10 bits away (median 1), the nearest other photo's original at least 29 bits (median 88).
- **views:** 2,269 questions, each in all three views. 19 have an `unknown` in the record (itch 6, hurt 10, bleed 6, elevation 2; 24
  cells); age and body site are always there. The records are 1,344 distinct texts; 1,481 questions share their record with another
  question, 495 of them in a group of more than one class, so the record alone cannot tell those apart.
- **shortcuts** (nearest-neighbour vote, train split only, balanced accuracy with a 95% interval over whole patients):

| Features | 6 classes (chance 16.7%) | Cancer or not (chance 50%) |
|---|---|---|
| empty fields (`gaps`) | 21.4% (20.5 to 22.4) | 62.6% (60.1 to 65.3) |
| photo properties, originals | 22.9% (21.3 to 24.5) | 64.3% (61.5 to 67.1) |
| photo properties, standardised photos | 19.7% (18.2 to 21.4) | 55.2% (52.4 to 58.1) |
| the six lines of the record | 38.2% (35.7 to 40.8) | 72.2% (69.2 to 75.2) |
| the full record (21 fields) | 43.3% (40.4 to 46.4) | 84.5% (81.8 to 87.2) |

  Standardising the photos removed about two thirds of what the photo properties gave away about cancer (64.3% to 55.2%) but not all of
  it: the interval still excludes 50%, so a small hint about the photo's origin remains and is reported beside the results. The empty
  fields still give 62.6%, which is why the 13-field block is not in any view. The six record lines carry real information (72.2% for
  cancer or not, with no photo), so the record view is a competitor, not a blank control.

## 5. Preprocessing and use

Photos are standardised once, by the `images` step above, and every model in every view receives those files and no others; the
originals are never changed. No synthetic or rewritten clinical text is ever added: a record is only the recorded values in fixed
words. Patients may be identifiable only through the source's own ids. The authors state that the programme was approved by the university ethics committee (no. 500002/478) and by Plataforma Brasil (no. 4.007.097), that all data is collected under patient consent, that "the patient's privacy is completely preserved", and that photos in which a patient may be identified (by a tattoo, for example) were deleted ✔ (paper, "Declaration of consent" and "Data collection").

## 6. Reproduce

1. Download the dataset (Kaggle copy or Mendeley), put the unpacked `metadata.csv` and the three `imgs_part_*` folders in
   `data/clinic_patients/raw/` (optionally the ZIP in `data/clinic_patients/downloads/`).
2. `python -m src.build.clinic_patients.prepare all` (or one step at a time, in the order of section 4).
3. The inventory's counts must equal those above; the figures of sections 1 to 3 come from the `inventory` section of
   `dataset_report.md` (`metadata.csv` SHA-256 starts `e5c178211d59`). A second run changes no file.
