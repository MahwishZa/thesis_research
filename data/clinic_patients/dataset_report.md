# Preparation of the PAD-UFES-20 cohort

Made by `python -m src.build.clinic_patients.prepare <step>`, one section per step, in the order the steps run. Each step rewrites only its own section.

<!-- step: inventory -->
## Inventory: what the download holds

Written by `python -m src.build.clinic_patients.prepare inventory`, which only reads the download. Nothing here is decided yet.

- `raw/metadata.csv`: 2298 rows, 26 columns (SHA-256 `e5c178211d59`)
- Photos named in the table and found: 2298; lesions (patient + `lesion_id`): 1891; patients: 1373
- Downloaded files fingerprinted (`build/downloads.csv`): none (downloads/ is empty or absent)

### Classes

The `diagnostic` column, with the `biopsed` column as written (it is never shown to a model: it points to the label).

| Class | Photos | Lesions | Patients | `biopsed` = 'FALSE' | `biopsed` = 'TRUE' |
|---|---|---|---|---|---|
| ACK | 730 | 644 | 525 | 552 | 178 |
| BCC | 845 | 652 | 513 | 0 | 845 |
| MEL | 52 | 36 | 36 | 0 | 52 |
| NEV | 244 | 216 | 185 | 184 | 60 |
| SCC | 192 | 145 | 133 | 0 | 192 |
| SEK | 235 | 198 | 176 | 220 | 15 |

### How photos, lesions and patients relate

A lesion is a patient + `lesion_id` pair: the same `lesion_id` is used for lesions of different patients.

- Photos per lesion: 1: 1530 lesions, 2: 324 lesions, 3: 31 lesions, 4: 4 lesions, 5: 1 lesions, 6: 1 lesions
- Lesions per patient: 1: 1018 patients, 2: 242 patients, 3: 79 patients, 4: 25 patients, 5: 5 patients, 6: 2 patients, 7: 1 patients, 8: 1 patients

### Clinical fields filled, by class

The 21 columns other than `img_id`, `patient_id`, `lesion_id`, `diagnostic`, `biopsed`. A field counts as filled unless it is blank or written as a gap (`N/A`, `NAN`, `NA`, `NONE`, `NULL`, `UNKNOWN`, `UNK`, in any case). If the classes differ much here, the gaps alone could point to the class.

| Class | Photos | Mean fields filled | Fewest | Most |
|---|---|---|---|---|
| ACK | 730 | 13.0 | 8 | 21 |
| BCC | 845 | 20.3 | 17 | 21 |
| MEL | 52 | 20.4 | 18 | 21 |
| NEV | 244 | 11.6 | 8 | 21 |
| SCC | 192 | 20.1 | 16 | 21 |
| SEK | 235 | 10.4 | 8 | 21 |

#### Which fields are empty together

The 8 most common sets of empty clinical fields. If a set is common in some classes and rare in others, an empty record would point to the class.

| Empty fields | Photos | Patients | ACK | BCC | MEL | NEV | SCC | SEK |
|---|---|---|---|---|---|---|---|---|
| (none) | 990 | 492 | 242 | 542 | 34 | 32 | 110 | 30 |
| `smoke`, `drink`, `background_father`, `background_mother`, `pesticide`, `gender`, `skin_cancer_history`, `cancer_history`, `has_piped_water`, `has_sewage_system`, `fitspatrick`, `diameter_1`, `diameter_2` | 804 | 579 | 447 | 0 | 0 | 169 | 0 | 188 |
| `grew`, `changed` | 345 | 274 | 1 | 243 | 6 | 23 | 67 | 5 |
| `background_father`, `background_mother` | 52 | 28 | 10 | 17 | 5 | 4 | 7 | 9 |
| `background_father`, `background_mother`, `grew`, `changed` | 26 | 19 | 0 | 11 | 0 | 8 | 4 | 3 |
| `background_father` | 25 | 13 | 14 | 10 | 0 | 1 | 0 | 0 |
| `background_mother` | 22 | 8 | 10 | 10 | 0 | 2 | 0 | 0 |
| `grew` | 7 | 6 | 3 | 0 | 4 | 0 | 0 | 0 |

#### Photo size by class

Photos come from different smartphones, so their sizes differ. If the classes differ here, the size alone could point to the class.

| Class | Photos | Width: smallest | median | largest | File size (KB): median |
|---|---|---|---|---|---|
| ACK | 730 | 189 | 855 | 3120 | 1050 |
| BCC | 845 | 167 | 772 | 2661 | 1036 |
| MEL | 52 | 236 | 860 | 3096 | 998 |
| NEV | 244 | 147 | 690 | 3474 | 604 |
| SCC | 192 | 167 | 815.5 | 2364 | 1155 |
| SEK | 235 | 172 | 757 | 2534 | 734 |

Each column's values and level (patient, lesion or photo) are in `build/columns.csv`.

### Checks

- Rows naming the same photo twice: 0
- Photos named in the table but not found in raw/: 0
- Photo files in raw/ not named in the table: 0
- File names found in more than one folder of raw/ (the first path is used): 0
- Photos that cannot be opened: 0
- Lesions (patient + `lesion_id`) with more than one diagnosis: 0
- `lesion_id` values used by more than one patient: 250 (`1003`, `1007`, `1009`, `1031`, `1051`, `1068`, `107`, `1077`, `1079`, `1081` and 240 more)
- Distinct patient + lesion pairs: 1891 (distinct `lesion_id` values: 1641)
  - `lesion_id` `1003` is under `PAT_1288` (NEV), `PAT_38` (ACK)
  - `lesion_id` `1007` is under `PAT_230` (ACK), `PAT_532` (BCC)
  - `lesion_id` `1009` is under `PAT_230` (ACK), `PAT_534` (BCC)
- Groups of photos with identical content: 15 (`PAT_202_307_424.png` = `PAT_202_308_721.png`; `PAT_302_650_477.png` = `PAT_302_651_529.png`; `PAT_311_666_191.png` = `PAT_311_667_416.png`; `PAT_38_1002_34.png` = `PAT_38_1003_68.png`; `PAT_38_1002_668.png` = `PAT_38_1003_226.png`; `PAT_419_2767_323.png` = `PAT_419_833_148.png`; `PAT_528_3072_615.png` = `PAT_528_993_589.png`; `PAT_56_86_479.png` = `PAT_56_88_274.png`; `PAT_570_1084_637.png` = `PAT_570_1085_344.png`; `PAT_570_1084_939.png` = `PAT_570_1085_429.png`; and 5 more)
  - under one patient: 15 of 15; under one lesion number: 0 of 15; with the same diagnosis: 3 of 15

### Photos

- Formats: PNG 2298
- Width: 147 to 3474 pixels (median 779.5); height: 147 to 3476 (median 779.5)

<!-- step: clean -->
## Clean: the clinical record, checked and normalised

Written by `python -m src.build.clinic_patients.prepare clean`. The original `metadata.csv` is not changed; the result is `build/cleaned.csv`, and every change and flag is a row of `build/cleaning.csv`.

- Rows: 2298. No row is left out, and no gap is filled in.
- Gaps (blank, `UNK` and the like) written as `not recorded`: 11485 cells in 19 columns. Which cells are gaps does not change: the pattern of gaps is checked in step `shortcuts`.
- Values standardised: 0 (case or spaces); 3 (spelling: `BRASIL` to `BRAZIL`)
- Every value of the diagnosis, body region, gender, skin type and yes/no columns is in the known vocabulary (otherwise this step stops).

### Flags (kept as recorded)

| Flag | Count | Where |
|---|---|---|
| diameter is 0: implausible for a lesion | 8 | BCC 6, SCC 2 |
| differs between photos of one patient (a patient-level field) | 0 | none |
| one lesion on different body regions | 0 | none |

A flag is not a correction. A flagged value stays as recorded; a view that shows a field decides what to do with its flags.

<!-- step: dedupe -->
## Dedupe: identical and near-identical photos

Written by `python -m src.build.clinic_patients.prepare dedupe`. Every photo has a 256-bit difference hash (`build/photo_hashes.csv`); the distance of two photos is the number of bits in which their hashes differ.

### The rule (fixed before the numbers were seen)

- At most **8** bits apart (identical files are 0 bits apart): one picture. Pictures linked through such pairs form a cluster.
- All photos of a cluster have the same diagnosis: the one with the smallest `img_id` stays, the others are dropped.
- The photos of a cluster have different diagnoses: all are dropped (one picture has no single right answer).
- Photos of different patients at most **16** bits apart stay, but their patients go to one split together.

### How far each photo is from its nearest other photo

| Bits | Photos |
|---|---|
| 0 | 32 |
| 1 to 8 | 0 |
| 9 to 16 | 0 |
| 17 to 32 | 6 |
| 33 to 64 | 149 |
| 65 to 256 | 2111 |

Two different photos are about 128 bits apart. A gap between the first rows and the last ones shows that the threshold separates copies from other photos.

### Clusters and decisions

- Photos hashed: 2298; unreadable (dropped): 0
- Clusters of the same picture: 16 (32 photos); identical SHA-256 groups: 15
- Clusters with conflicting diagnoses (all photos dropped): 13
- Clusters with one diagnosis (one photo stays): 3

| Class | Photos | Dropped | of which conflicting | of which copies |
|---|---|---|---|---|
| ACK | 730 | 7 | 6 | 1 |
| BCC | 845 | 12 | 12 | 0 |
| MEL | 52 | 0 | 0 | 0 |
| NEV | 244 | 1 | 1 | 0 |
| SCC | 192 | 8 | 7 | 1 |
| SEK | 235 | 1 | 0 | 1 |

Clusters (the first 10; all are in `build/duplicates.csv`):

- C001: `PAT_202_307_424.png` (BCC), `PAT_202_308_721.png` (ACK) (conflicting: all dropped)
- C002: `PAT_302_650_477.png` (SCC), `PAT_302_651_529.png` (BCC) (conflicting: all dropped)
- C003: `PAT_311_666_191.png` (SCC), `PAT_311_667_416.png` (BCC) (conflicting: all dropped)
- C004: `PAT_380_1540_959.png` (SCC), `PAT_380_773_362.png` (BCC) (conflicting: all dropped)
- C005: `PAT_38_1002_34.png` (BCC), `PAT_38_1003_68.png` (ACK) (conflicting: all dropped)
- C006: `PAT_38_1002_668.png` (BCC), `PAT_38_1003_226.png` (ACK) (conflicting: all dropped)
- C007: `PAT_419_2767_323.png` (ACK), `PAT_419_833_148.png` (SCC) (conflicting: all dropped)
- C008: `PAT_528_3072_615.png` (ACK), `PAT_528_993_589.png` (ACK) (`PAT_528_3072_615.png` stays)
- C009: `PAT_56_86_479.png` (SCC), `PAT_56_88_274.png` (BCC) (conflicting: all dropped)
- C010: `PAT_570_1084_637.png` (SCC), `PAT_570_1085_344.png` (BCC) (conflicting: all dropped)

### Near pairs

- Pairs of photos of **different patients** at most 16 bits apart: 0 (their patients are kept in one split)
- Pairs of photos of one patient between 9 and 16 bits apart (kept; one patient is always in one split): 0
- All pairs at most 16 bits apart are in `build/near_pairs.csv`.

<!-- step: split -->
## Split: train, validation and test, by patient

Written by `python -m src.build.clinic_patients.prepare split`. The questions with their splits are `build/split.csv` (step `images` adds the photos and writes `dataset.csv`); the photos left out, with the reason, are in `build/excluded.csv`.

- Seed: 20261119 (chosen before the first draw). Targets: train 60%, val 10%, test 30%, class by class.
- Photos that stay: 2269; left out: 29 (conflicting_labels 26, identical_copy 3)
- Split groups (patients that must stay together): 1371 for 1371 patients; patients joined because of a near-identical photo of another patient: 0 links
- No patient is in two splits (checked).

### Photos by class and split

| Class | train | val | test | Total | train share | val share | test share |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 434 | 72 | 217 | 723 | 60.0% | 10.0% | 30.0% |
| basal cell carcinoma | 500 | 83 | 250 | 833 | 60.0% | 10.0% | 30.0% |
| melanoma | 32 | 5 | 15 | 52 | 61.5% | 9.6% | 28.8% |
| nevus | 146 | 24 | 73 | 243 | 60.1% | 9.9% | 30.0% |
| seborrheic keratosis | 141 | 23 | 70 | 234 | 60.3% | 9.8% | 29.9% |
| squamous cell carcinoma | 115 | 16 | 53 | 184 | 62.5% | 8.7% | 28.8% |

### Patients and lesions by split

| Split | Photos | Patients | Lesions |
|---|---|---|---|
| train | 1368 | 654 | 1073 |
| val | 223 | 219 | 223 |
| test | 678 | 498 | 580 |

Largest split groups (photos): `G_PAT_645` 10, `G_PAT_388` 10, `G_PAT_330` 10

<!-- step: images -->
## Images: the photos every model receives

Written by `python -m src.build.clinic_patients.prepare images`. Each photo that stays is made from its original in one fixed way (`files/photos/`, not committed; the originals in `raw/` are never changed). `build/files.csv` lists every photo with what was done and the two measures below; step `views` joins them to the records and writes `dataset.csv`.

### What is done (in this order, and nothing else)

1. Turned upright if the EXIF orientation says so. 2. Converted to RGB if needed (transparent pixels stop the step). 3. Converted to sRGB from the embedded colour profile, if there is one.
4. Shrunk, whole and with its proportions kept, so that the longest side is at most 1024 pixels; a smaller photo keeps its size. 5. Saved as JPEG (quality 95, full colour resolution, no metadata).

- Photos made: 2269
- colours converted to sRGB: 2; converted from RGBA to RGB: 1420; saved again only: 476; shrunk: 814

### Size

| | Originals | Saved photos |
|---|---|---|
| Longest side (pixels): smallest, median, largest | 147, 778, 3476 | 147, 778, 1024 |
| File size (KB): median | 933 | 191 |
| File size (MB): largest | 23.62 | 0.77 |
| Files over 5 MB | 104 | 0 |

### Does it change the picture? (a bar for every photo; the step stops if one is missed)

| Measure | Bar | Smallest / median / largest |
|---|---|---|
| Fidelity: PSNR of the photo before and after saving (dB; above about 35 two pictures look the same) | at least 30 | 39.8 / 45.8 / 62.0 |
| Same picture: bits by which the saved photo's 256-bit hash differs from its own original's | below the next measure | 0 / 1 / 10 |
| Bits to the nearest *other* photo's original (different photos are about 128 apart) | more than the line above | 29 / 88 / 105 |

<!-- step: views -->
## Views: the same questions as photo, record, and photo + record

Written by `python -m src.build.clinic_patients.prepare views`. One question is one photo of one patient's lesion with the record of the same visit. `dataset.csv` lists the photo view and the photo + record view (the record text is its `facts` column); `record.csv` lists the record view; the records are in `files/records/` (not committed).

- Questions: 2269 (train 1368, val 223, test 678). Each is in all three views.
- Record template `record-v1` (fingerprint `ea9036ddbfe3`); the wording is provisional until the authors' paper confirms what each field means. A gap is written `not recorded`; nothing is filled in.

```
1. Age in years: <age>
2. Body site: <region>
3. The lesion itches: <itch>
4. The lesion hurts: <hurt>
5. The lesion has bled: <bleed>
6. The lesion has an elevation: <elevation>
```

- Not in any view: `grew`, `changed` (unknown more often for cancers: a hint), and the later questionnaire fields. They stay in `dataset.csv` for a separate secondary analysis.

### Gaps in the record, by class

| Class | Questions | All six recorded | `age` not recorded | `region` not recorded | `itch` not recorded | `hurt` not recorded | `bleed` not recorded | `elevation` not recorded |
|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 723 | 720 | 0 | 0 | 2 | 1 | 0 | 0 |
| basal cell carcinoma | 833 | 826 | 0 | 0 | 0 | 3 | 5 | 0 |
| melanoma | 52 | 51 | 0 | 0 | 0 | 0 | 0 | 1 |
| nevus | 243 | 238 | 0 | 0 | 1 | 5 | 1 | 1 |
| seborrheic keratosis | 234 | 234 | 0 | 0 | 0 | 0 | 0 | 0 |
| squamous cell carcinoma | 184 | 181 | 0 | 0 | 3 | 1 | 0 | 0 |
| all | 2269 | 2250 | 0 | 0 | 6 | 10 | 6 | 2 |

### How much a record can say

- Distinct records: 1344 for 2269 questions. Questions whose record is the same as another question's: 1481 (in 556 groups).
  - with another photo of the same patient: 855 (several photos of one visit share one record);
  - with a photo of a different patient: 953;
  - in a group of more than one class: 495 (the record alone cannot tell these apart).
- Identical records are kept: the record view asks every question, so that it pairs with the other two views. Photos of one patient are never in two splits, and the intervals resample whole patients.

### Examples (the first question of each class in the train split)

**actinic keratosis** (`PAT_1014_85_22.png`)

```
1. Age in years: 41
2. Body site: forearm
3. The lesion itches: yes
4. The lesion hurts: no
5. The lesion has bled: no
6. The lesion has an elevation: no
```

**basal cell carcinoma** (`PAT_101_1041_651.png`)

```
1. Age in years: 59
2. Body site: neck
3. The lesion itches: yes
4. The lesion hurts: yes
5. The lesion has bled: yes
6. The lesion has an elevation: yes
```

**melanoma** (`PAT_109_868_113.png`)

```
1. Age in years: 82
2. Body site: ear
3. The lesion itches: no
4. The lesion hurts: no
5. The lesion has bled: no
6. The lesion has an elevation: no
```

**nevus** (`PAT_100_393_595.png`)

```
1. Age in years: 14
2. Body site: face
3. The lesion itches: no
4. The lesion hurts: no
5. The lesion has bled: no
6. The lesion has an elevation: no
```

**seborrheic keratosis** (`PAT_1000_31_620.png`)

```
1. Age in years: 65
2. Body site: ear
3. The lesion itches: no
4. The lesion hurts: no
5. The lesion has bled: no
6. The lesion has an elevation: no
```

**squamous cell carcinoma** (`PAT_108_161_423.png`)

```
1. Age in years: 68
2. Body site: forearm
3. The lesion itches: yes
4. The lesion hurts: yes
5. The lesion has bled: yes
6. The lesion has an elevation: yes
```

<!-- step: shortcuts -->
## Shortcuts: what the gaps and the photo properties alone give away

Written by `python -m src.build.clinic_patients.prepare shortcuts`. A nearest-neighbour vote (15 neighbours) on the **train split only** (1368 photos, 654 patients, 654 split groups), in 5 folds that never divide a split group. The validation and test splits are not used. The score is balanced accuracy with a 95% interval from resampling whole split groups; the learner is simple, so a real hint can be larger than shown.

| Features | 6 classes (chance 16.7%) | Cancer or not (chance 50%) | Recall by class (6 classes) |
|---|---|---|---|
| gaps | 21.4% (20.5% to 22.4%) | 62.6% (60.1% to 65.3%) | actinic keratosis 0.99; basal cell carcinoma 0.30; melanoma 0.00; nevus 0.00; seborrheic keratosis 0.00; squamous cell carcinoma 0.00 |
| photo properties (originals) | 22.9% (21.3% to 24.5%) | 64.3% (61.5% to 67.1%) | actinic keratosis 0.53; basal cell carcinoma 0.65; melanoma 0.00; nevus 0.10; seborrheic keratosis 0.08; squamous cell carcinoma 0.02 |
| photo properties (standardised) | 19.7% (18.2% to 21.4%) | 55.2% (52.4% to 58.1%) | actinic keratosis 0.49; basal cell carcinoma 0.53; melanoma 0.00; nevus 0.14; seborrheic keratosis 0.01; squamous cell carcinoma 0.02 |
| core record | 38.2% (35.7% to 40.8%) | 72.2% (69.2% to 75.2%) | actinic keratosis 0.79; basal cell carcinoma 0.74; melanoma 0.00; nevus 0.55; seborrheic keratosis 0.21; squamous cell carcinoma 0.00 |
| full record | 43.3% (40.4% to 46.4%) | 84.5% (81.8% to 87.2%) | actinic keratosis 0.80; basal cell carcinoma 0.89; melanoma 0.00; nevus 0.47; seborrheic keratosis 0.38; squamous cell carcinoma 0.05 |

- `gaps`: 21 clinical fields, each 1 if empty. `photo properties`: width, height, file size, bytes per pixel, of the original files and of the standardised files (the ones a model receives). `core record`: age, region, itch, hurt, bleed, elevation (what a view would show). `full record`: every clinical field.
- `gaps` and `photo properties` are hints that are not about the lesion: a score well above chance means a model could use them. The records are information: how much the record says about the class, legitimately or not.
