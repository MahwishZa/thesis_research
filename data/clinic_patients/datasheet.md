# Data sheet: the clinic patients cohort (PAD-UFES-20), draft of 2026-11-19

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. **It is a draft.
Sections 1 to 3 describe the download as received (numbers from the `inventory` step). Section 4 states the rules the preparation
steps follow, decided on 2026-11-19; the figures they produce (the splits, the photo standardisation, the records) are in
`dataset_report.md` after the first full run on the researcher's computer, and are copied here only then.**
Every number below comes from files that `python -m src.build.clinic_patients.prepare` writes: `dataset_report.md`,
`build/columns.csv`, `build/raw_images.csv`. The photos and `metadata.csv` stay on the researcher's computer (research folder
`data/clinic_patients/raw/`); only the small tables are in Git.

Marks: ✔ checked in the researcher's download by the inventory; ◻ not yet checked (stated by the source or a web page only).

## At a glance

| | |
|---|---|
| What it is | Smartphone photos of skin lesions, each with the clinical record of the same patient visit (age, body site, symptoms, history) |
| Photos | 2,298 PNG files, 147 to 3,474 pixels wide (median 779.5), from different phones ✔ |
| Patients, lesions | 1,373 patients ✔; 1,891 lesions ✔ (a lesion is a patient + `lesion_id` pair: `lesion_id` alone is shared between patients, and its 1,641 distinct values are not the lesion count; the ISIC Archive lists 1,891 lesions for this collection, a search result ◻) |
| Classes | 6: basal cell carcinoma (BCC) 845, actinic keratosis (ACK) 730, nevus (NEV) 244, seborrheic keratosis (SEK) 235, squamous cell carcinoma (SCC) 192, melanoma (MEL) 52 photos ✔; Bowen's disease is counted as SCC by the authors ◻ |
| Label source | The authors' `diagnostic` column. All BCC, SCC and MEL photos (1,089) are biopsy-proven ✔; ACK, NEV and SEK are biopsy-proven for 178 of 730, 60 of 244 and 15 of 235 photos ✔, the others by a clinical diagnosis of dermatologists ◻ (1,342 of 2,298, 58.4%, are biopsy-proven ✔) |
| Source | Kaggle "Skin Cancer(PAD-UFES-20)" (mahdavi1202), a copy of the authors' Mendeley Data upload, version 1, published 2020-07-07 (doi 10.17632/zr7vgbcyr2.1) ◻ not compared byte for byte with Mendeley; its counts equal the paper's ✔ |
| Licence | CC BY 4.0 (the Kaggle page; credit the authors) ◻ confirm on Mendeley |
| Citation | Pacheco AGC, Lima GR, Salomão AS, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, October 2020 ◻ (volume and pages not checked). Mendeley Data, V1, doi 10.17632/zr7vgbcyr2.1 |
| Version of this cohort | none yet (nothing is frozen) |

## 1. Motivation

The Version 1 photo dataset had no clinical record next to its photos, and its text dataset described other patients. PAD-UFES-20
gives photos and a clinical record of the same visit, from one clinic, so the image, the text and both together can be compared on
the same patients. See `docs/v2_proposal.md`.

## 2. Composition

**One row of `metadata.csv` is one photo** (`img_id`, `PAT_<patient>_<lesion>_<n>.png`; the authors write "each line represents a skin
lesion", but the 2,298 rows hold 1,891 lesions ✔). A patient has 1 to 8 lesions ✔ (1,018 patients have one). The table has 26 columns ✔.
**Meanings are the authors' own definitions** (Pacheco et al. 2020, section "Meta-data", copied by the researcher from the arXiv version
2007.00478 on 2026-11-19; not yet compared with the published *Data in Brief* article ◻).

| Column | Filled / empty ✔ | Values seen ✔ | Meaning (the authors) |
|---|---|---|---|
| `patient_id`, `lesion_id`, `img_id` | all filled | ids | patient, lesion number, photo (`img_id` joins patient, lesion and a random number). Always present |
| `diagnostic` | 2,298 / 0 | ACK, BCC, MEL, NEV, SCC, SEK | the label |
| `biopsed` | 2,298 / 0 | TRUE 1,342, FALSE 956 | "if the diagnostic comes from a clinical consensus or biopsy" (the wording does not say which value is which; all 1,089 BCC, SCC and MEL are TRUE, so TRUE reads as biopsy ◻). **Never shown to a model** (it points to the label). Always present |
| `age` | 2,298 / 0 | 6 to 94, median 62 | an integer, the patient's age (the unit is not written; the values are years ◻). Always present |
| `region` | 2,298 / 0 | 14 body regions (FACE 570, FOREARM 392, CHEST 280, BACK 248, ARM 192, NOSE 158, HAND 126, …) | "one of the 15 macro-regions previously described" (the download holds 14; which one is absent is not known to us ◻). Always present |
| `itch`, `hurt`, `bleed`, `elevation` | 2 to 10 unknown (`UNK`) | TRUE / FALSE | the lesion itches; hurts; "has bled"; "has an elevation" |
| `grew`, `changed` | 402 and 396 unknown (`UNK`) | TRUE / FALSE | the lesion "has recently grown"; "has recently changed" |
| `gender`, `smoke`, `drink`, `pesticide`, `skin_cancer_history`, `cancer_history`, `has_piped_water`, `has_sewage_system`, `fitspatrick` (1 to 6), `diameter_1`, `diameter_2` | 804 blank each | TRUE / FALSE, FEMALE / MALE, numbers | the patient smokes cigarettes, drinks alcohol, uses pesticides; skin cancer, or any cancer, in the past **in the patient or in the family** (one field each); piped water and a sewage system at home; the Fitzpatrick skin type; the lesion's horizontal and vertical diameters (floats) |
| `background_father`, `background_mother` | 913 and 906 empty or `UNK` | **countries** (POMERANIA, GERMANY, ITALY, BRAZIL, …; "BRASIL" and "BRAZIL" both occur) | the country the father's and the mother's family descends from (Pomerania is kept although it is a region: patients call themselves Pomeranian). They are places of origin, not diseases, as the Kaggle page wrongly says |

**How gaps are written, and what they mean.** The authors: "Missing values are left blank ... When a patient does not know the answer ... we
fill the feature as UNK (unknown)", and only `patient_id`, `lesion_id`, `img_id`, `age`, `region` and `biopsed` are always present; the
rest "depend on the patient's answers during the appointment". In the download, the 804-photo block is blank and every other gap is `UNK` ✔.
So **a gap in the four symptom fields of the record is `UNK`, which means the patient did not know**, not that nothing was asked. The
cleaned table writes both as `not recorded`; `build/columns.csv` and the inventory keep the two apart. The authors do not say why the
804-photo block is blank, so the cause of finding 1 stays a guess ◻. Diameters run from 0 to 100 (`diameter_1`) and 0 to 70
(`diameter_2`); their unit is not stated ◻ (a diameter of 0 occurs).

## 3. Findings that shape the design ✔

1. **An empty record points to "not a cancer".** 804 photos from 579 patients have all 13 of these fields empty together:
   `smoke`, `drink`, `background_father`, `background_mother`, `pesticide`, `gender`, `skin_cancer_history`, `cancer_history`,
   `has_piped_water`, `has_sewage_system`, `fitspatrick`, `diameter_1`, `diameter_2`. By class: ACK 447, NEV 169, SEK 188, **BCC 0,
   MEL 0, SCC 0**. Another 990 photos have no empty field at all (ACK 242, BCC 542, MEL 34, NEV 32, SCC 110, SEK 30). So a model
   (or a retrieval or filter) that learns "record empty" gets the cancer / not-cancer split without looking at the lesion. A likely
   cause is that the questionnaire grew over the years of collection; the paper may say so ◻.
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
elevation. The lines use the authors' own words for each field (itch, hurt, "has bled", "has an elevation"; section 2); the unit of the
age is presumed to be years ◻. A gap reads `not recorded`; in the authors' words the patient did not know (section 2). The record is
the same words, to the byte, in the record view and in the photo + record view. **Not in any view:** `grew` and `changed` (unknown more
often for cancers: finding 2) and the block of 13 fields of finding 1 (empty for 804 early photos, all benign). They stay in
`dataset.csv` for a separate secondary analysis on complete records. A change of wording is a new template version and a rebuild of
the views.

**Photos of one patient** are not independent: intervals resample whole patients. Identical records are kept in the record view
(they are different patients, or several photos of one visit), so that the three views ask the same questions.

## 5. Preprocessing and use

Photos are standardised once, by the `images` step above, and every model in every view receives those files and no others; the
originals are never changed. No synthetic or rewritten clinical text is ever added: a record is only the recorded values in fixed
words. Patients may be identifiable only through the source's own ids; the source de-identified them ◻.

## 6. Reproduce

1. Download the dataset (Kaggle copy or Mendeley), put the unpacked `metadata.csv` and the three `imgs_part_*` folders in
   `data/clinic_patients/raw/` (optionally the ZIP in `data/clinic_patients/downloads/`).
2. `python -m src.build.clinic_patients.prepare all` (or one step at a time, in the order of section 4).
3. The inventory's counts must equal those above; the figures of sections 1 to 3 come from the `inventory` section of
   `dataset_report.md` (`metadata.csv` SHA-256 starts `e5c178211d59`). A second run changes no file.
