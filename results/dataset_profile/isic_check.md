# ISIC Archive label check

Checked 255 ISIC numbers (images with conflicting labels and all testing images).
Evidence: the diagnosis fields of each public ISIC record (full rows in isic_check.csv; raw records cached in data/isic_cache/).

| Group | Verdict | Images |
|---|---|---|
| conflict: actinic keratosis vs nevus (a copy is in test) | ISIC supports: actinic keratosis | 15 |
| conflict: actinic keratosis vs nevus (training only) | ISIC supports: actinic keratosis | 62 |
| conflict: melanoma vs seborrheic keratosis (a copy is in test) | ISIC supports: melanoma | 3 |
| conflict: melanoma vs seborrheic keratosis (training only) | ISIC supports: melanoma | 75 |
| test image, no conflict: actinic keratosis | ISIC supports: actinic keratosis | 1 |
| test image, no conflict: basal cell carcinoma | ISIC supports: basal cell carcinoma | 16 |
| test image, no conflict: dermatofibroma | ISIC supports: dermatofibroma | 16 |
| test image, no conflict: melanoma | ISIC supports: melanoma | 16 |
| test image, no conflict: nevus | ISIC supports: nevus | 16 |
| test image, no conflict: pigmented benign keratosis | ISIC supports: pigmented benign keratosis | 16 |
| test image, no conflict: squamous cell carcinoma | ISIC supports: squamous cell carcinoma | 16 |
| test image, no conflict: vascular lesion | ISIC supports: vascular lesion | 3 |

## Licences recorded by ISIC

| Licence | Images |
|---|---|
| CC-BY-NC | 131 |
| CC-0 | 124 |
