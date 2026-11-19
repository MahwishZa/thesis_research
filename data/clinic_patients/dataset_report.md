# Preparation of the PAD-UFES-20 cohort

Made by `python -m src.build.clinic_patients.prepare <step>`, one section per step, in the order the steps run. Each step rewrites only its own section.

<!-- step: inventory -->
## Inventory: what the download holds

Written by `python -m src.build.clinic_patients.prepare inventory`, which only reads the download. Nothing here is decided yet.

- `raw/metadata.csv`: 2298 rows, 26 columns (SHA-256 `e5c178211d59`)
- Photos named in the table and found: 2298; lesions: 1641; patients: 1373
- Downloaded files fingerprinted (`build/downloads.csv`): none (downloads/ is empty or absent)

### Classes

The `diagnostic` column, with the `biopsed` column as written (it is never shown to a model: it points to the label).

| Class | Photos | Lesions | Patients | `biopsed` = 'FALSE' | `biopsed` = 'TRUE' |
|---|---|---|---|---|---|
| ACK | 730 | 621 | 525 | 552 | 178 |
| BCC | 845 | 652 | 513 | 0 | 845 |
| MEL | 52 | 36 | 36 | 0 | 52 |
| NEV | 244 | 215 | 185 | 184 | 60 |
| SCC | 192 | 145 | 133 | 0 | 192 |
| SEK | 235 | 198 | 176 | 220 | 15 |

### How photos, lesions and patients relate

- Photos per lesion: 1: 1129 lesions, 2: 396 lesions, 3: 95 lesions, 4: 17 lesions, 5: 2 lesions, 6: 1 lesions, 8: 1 lesions
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

Each column's values and level (patient, lesion or photo) are in `build/columns.csv`.

### Checks

- Rows naming the same photo twice: 0
- Photos named in the table but not found in raw/: 0
- Photo files in raw/ not named in the table: 0
- File names found in more than one folder of raw/ (the first path is used): 0
- Photos that cannot be opened: 0
- Lesions with more than one diagnosis: 226 (`1003`, `1007`, `1009`, `1031`, `1051`, `1068`, `107`, `1077`, `1079`, `1081` and 216 more)
- Lesions under more than one patient: 250 (`1003`, `1007`, `1009`, `1031`, `1051`, `1068`, `107`, `1077`, `1079`, `1081` and 240 more)
- Groups of photos with identical content: 15 (`PAT_202_307_424.png` = `PAT_202_308_721.png`; `PAT_302_650_477.png` = `PAT_302_651_529.png`; `PAT_311_666_191.png` = `PAT_311_667_416.png`; `PAT_38_1002_34.png` = `PAT_38_1003_68.png`; `PAT_38_1002_668.png` = `PAT_38_1003_226.png`; `PAT_419_2767_323.png` = `PAT_419_833_148.png`; `PAT_528_3072_615.png` = `PAT_528_993_589.png`; `PAT_56_86_479.png` = `PAT_56_88_274.png`; `PAT_570_1084_637.png` = `PAT_570_1085_344.png`; `PAT_570_1084_939.png` = `PAT_570_1085_429.png`; and 5 more)

### Photos

- Formats: PNG 2298
- Width: 147 to 3474 pixels (median 779.5); height: 147 to 3476 (median 779.5)
