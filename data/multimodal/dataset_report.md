# Preparation of the multimodal dataset

Made by `python -m src.multimodal.prepare <step>`, one section per step, in the order the steps run. Each step rewrites only its own section.

<!-- step: records -->
### Records: the patient facts in the ISIC records

Read: the ISIC record of each of the 2179 images of the image dataset, each one checked against the fingerprint the image dataset recorded for it. No file of the image dataset was changed and no service was asked.

#### Images by class and photo collection

The collection is the record's attribution. A fact recorded differently by each collection gives the class away.

| Class | Anonymous | MILK study team | All |
|---|---|---|---|
| actinic keratosis | 2 | 129 | 131 |
| basal cell carcinoma | 0 | 392 | 392 |
| dermatofibroma | 7 | 104 | 111 |
| melanoma | 436 | 0 | 436 |
| nevus | 293 | 0 | 293 |
| pigmented benign keratosis | 0 | 478 | 478 |
| squamous cell carcinoma | 5 | 191 | 196 |
| vascular lesion | 0 | 142 | 142 |

#### Every patient field the records hold

Facts may be offered to the model. Hints point to the diagnosis and identifiers only link photos: both are never offered.

| Field | Kind | Images with it | Anonymous | MILK study team |
|---|---|---|---|---|
| `metadata.clinical.concomitant_biopsy` | hint | 2179 of 2179 (100%) | 743 of 743 (100%) | 1436 of 1436 (100%) |
| `metadata.clinical.diagnosis_1` | hint | 2179 of 2179 (100%) | 743 of 743 (100%) | 1436 of 1436 (100%) |
| `metadata.clinical.diagnosis_2` | hint | 2179 of 2179 (100%) | 743 of 743 (100%) | 1436 of 1436 (100%) |
| `metadata.clinical.melanocytic` | hint | 2179 of 2179 (100%) | 743 of 743 (100%) | 1436 of 1436 (100%) |
| `metadata.clinical.diagnosis_3` | hint | 2037 of 2179 (93%) | 743 of 743 (100%) | 1294 of 1436 (90%) |
| `metadata.clinical.anatom_site_1` | fact | 2031 of 2179 (93%) | 599 of 743 (81%) | 1432 of 1436 (100%) |
| `metadata.clinical.sex` | fact | 1966 of 2179 (90%) | 530 of 743 (71%) | 1436 of 1436 (100%) |
| `metadata.clinical.age_approx` | fact | 1948 of 2179 (89%) | 518 of 743 (70%) | 1430 of 1436 (100%) |
| `metadata.clinical.diagnosis_confirm_type` | hint | 1886 of 2179 (87%) | 450 of 743 (61%) | 1436 of 1436 (100%) |
| `metadata.clinical.lesion_id` | identifier | 1446 of 2179 (66%) | 10 of 743 (1%) | 1436 of 1436 (100%) |
| `metadata.clinical.anatom_site_2` | fact | 1225 of 2179 (56%) | 327 of 743 (44%) | 898 of 1436 (63%) |
| `metadata.clinical.anatom_site_3` | fact | 677 of 2179 (31%) | 133 of 743 (18%) | 544 of 1436 (38%) |
| `metadata.clinical.clin_size_long_diam_mm` | fact | 269 of 2179 (12%) | 269 of 743 (36%) | 0 of 1436 (0%) |
| `metadata.clinical.personal_hx_mm` | hint | 269 of 2179 (12%) | 269 of 743 (36%) | 0 of 1436 (0%) |
| `metadata.clinical.family_hx_mm` | hint | 265 of 2179 (12%) | 265 of 743 (36%) | 0 of 1436 (0%) |
| `metadata.clinical.diagnosis_4` | hint | 221 of 2179 (10%) | 221 of 743 (30%) | 0 of 1436 (0%) |
| `metadata.clinical.diagnosis_5` | hint | 55 of 2179 (3%) | 55 of 743 (7%) | 0 of 1436 (0%) |
| `metadata.clinical.anatom_site_special` | fact | 36 of 2179 (2%) | 5 of 743 (1%) | 31 of 1436 (2%) |
| `metadata.clinical.anatom_site_4` | fact | 21 of 2179 (1%) | 21 of 743 (3%) | 0 of 1436 (0%) |
| `metadata.clinical.patient_id` | identifier | 10 of 2179 (0%) | 10 of 743 (1%) | 0 of 1436 (0%) |
| `metadata.clinical.anatom_site_5` | fact | 3 of 2179 (0%) | 3 of 743 (0%) | 0 of 1436 (0%) |

Photos with more than one ISIC record (uploaded more than once): 2 (ISIC_0000279 ISIC_0000280, ISIC_0000477 ISIC_0000512). A field counts only where their records agree.

- Records disagree, so left out: `ISIC_0000279 ISIC_0000280` `metadata.clinical.age_approx`: 80, 70
- Records disagree, so left out: `ISIC_0000279 ISIC_0000280` `metadata.clinical.anatom_site_1`: Lower extremity, Trunk
- Records disagree, so left out: `ISIC_0000279 ISIC_0000280` `metadata.clinical.anatom_site_2`: Foot, Posterior trunk
- Records disagree, so left out: `ISIC_0000477 ISIC_0000512` `metadata.clinical.anatom_site_2`: Posterior trunk, Anterior trunk

#### How often each fact is recorded, by class

| Fact | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion |
|---|---|---|---|---|---|---|---|---|
| `anatom_site_1` | 131 of 131 (100%) | 391 of 392 (100%) | 109 of 111 (98%) | 381 of 436 (87%) | 206 of 293 (70%) | 475 of 478 (99%) | 196 of 196 (100%) | 142 of 142 (100%) |
| `sex` | 131 of 131 (100%) | 392 of 392 (100%) | 109 of 111 (98%) | 362 of 436 (83%) | 156 of 293 (53%) | 478 of 478 (100%) | 196 of 196 (100%) | 142 of 142 (100%) |
| `age_approx` | 131 of 131 (100%) | 392 of 392 (100%) | 111 of 111 (100%) | 361 of 436 (83%) | 143 of 293 (49%) | 476 of 478 (100%) | 196 of 196 (100%) | 138 of 142 (97%) |
| `anatom_site_2` | 109 of 131 (83%) | 293 of 392 (75%) | 11 of 111 (10%) | 156 of 436 (36%) | 166 of 293 (57%) | 328 of 478 (69%) | 89 of 196 (45%) | 73 of 142 (51%) |
| `anatom_site_3` | 94 of 131 (72%) | 124 of 392 (32%) | 4 of 111 (4%) | 63 of 436 (14%) | 68 of 293 (23%) | 229 of 478 (48%) | 52 of 196 (27%) | 43 of 142 (30%) |
| `clin_size_long_diam_mm` | 2 of 131 (2%) | 0 of 392 (0%) | 0 of 111 (0%) | 267 of 436 (61%) | 0 of 293 (0%) | 0 of 478 (0%) | 0 of 196 (0%) | 0 of 142 (0%) |
| `anatom_site_special` | 6 of 131 (5%) | 6 of 392 (2%) | 3 of 111 (3%) | 3 of 436 (1%) | 2 of 293 (1%) | 4 of 478 (1%) | 7 of 196 (4%) | 5 of 142 (4%) |
| `anatom_site_4` | 0 of 131 (0%) | 0 of 392 (0%) | 0 of 111 (0%) | 11 of 436 (3%) | 10 of 293 (3%) | 0 of 478 (0%) | 0 of 196 (0%) | 0 of 142 (0%) |
| `anatom_site_5` | 0 of 131 (0%) | 0 of 392 (0%) | 0 of 111 (0%) | 2 of 436 (0%) | 1 of 293 (0%) | 0 of 478 (0%) | 0 of 196 (0%) | 0 of 142 (0%) |

#### The values of each fact, by collection

`metadata.clinical.anatom_site_1`:

| Collection | Values (images) |
|---|---|
| Anonymous | Trunk (268), Lower extremity (126), Upper extremity (113), Head and neck (91), Anogenital region (1) |
| MILK study team | Trunk (504), Head and neck (423), Lower extremity (299), Upper extremity (205), Anogenital region (1) |

`metadata.clinical.sex`:

| Collection | Values (images) |
|---|---|
| Anonymous | male (291), female (239) |
| MILK study team | male (855), female (581) |

`metadata.clinical.age_approx`:

| Collection | Values (images) |
|---|---|
| Anonymous | 70 (60), 65 (53), 60 (49), 30 (44), 85 (44), 80 (40), 50 (38), 75 (34), and 9 more values |
| MILK study team | 70 (211), 75 (196), 65 (172), 60 (147), 80 (143), 55 (126), 50 (111), 85 (107), and 9 more values |

| Class | Median | Range |
|---|---|---|
| actinic keratosis | 65 | 30 to 85 |
| basal cell carcinoma | 70 | 20 to 85 |
| dermatofibroma | 55 | 25 to 85 |
| melanoma | 65 | 10 to 85 |
| nevus | 35 | 5 to 75 |
| pigmented benign keratosis | 65 | 5 to 85 |
| squamous cell carcinoma | 70 | 40 to 85 |
| vascular lesion | 55 | 5 to 85 |

`metadata.clinical.anatom_site_2`:

| Collection | Values (images) |
|---|---|
| Anonymous | Posterior trunk (166), Anterior trunk (95), Shoulder (21), Head (10), Thigh (8), Lower leg (7), Lateral trunk (6), Buttock (3), and 7 more values |
| MILK study team | Head (392), Posterior trunk (292), Anterior trunk (152), Neck (31), Hand (24), Foot (6), Genital region (1) |

`metadata.clinical.anatom_site_3`:

| Collection | Values (images) |
|---|---|
| Anonymous | Anterior abdomen (49), Anterior chest (47), Face (9), Upper back (9), Calf (4), Lower back (4), Toes (2), Anterior shoulder (1), and 8 more values |
| MILK study team | Face (342), Anterior chest (87), Anterior abdomen (65), Scalp (44), Ear (6) |

`metadata.clinical.clin_size_long_diam_mm`:

| Collection | Values (images) |
|---|---|
| Anonymous | 5.00 (34), 6.00 (28), 10.00 (25), 4.00 (20), 7.00 (19), 9.00 (18), 8.00 (17), 11.00 (13), and 23 more values |
| MILK study team | not recorded |

| Class | Median | Range |
|---|---|---|
| actinic keratosis | 6.5 | 6 to 7 |
| basal cell carcinoma | – | – |
| dermatofibroma | – | – |
| melanoma | 8 | 2 to 100 |
| nevus | – | – |
| pigmented benign keratosis | – | – |
| squamous cell carcinoma | – | – |
| vascular lesion | – | – |

`metadata.clinical.anatom_site_special`:

| Collection | Values (images) |
|---|---|
| Anonymous | acral NOS (4), oral or genital (1) |
| MILK study team | acral NOS (30), oral or genital (1) |

`metadata.clinical.anatom_site_4`:

| Collection | Values (images) |
|---|---|
| Anonymous | Breast (8), Forehead (3), Cheek (2), Great toe (2), Upper anterior chest (2), Lateral upper back (1), Paraspinal upper back (1), Pinna (1), and 1 more values |
| MILK study team | not recorded |

`metadata.clinical.anatom_site_5`:

| Collection | Values (images) |
|---|---|
| Anonymous | Clavicle (1), Mons pubis (1), Presternal region (1) |
| MILK study team | not recorded |

#### Not done in this step

- No fact is chosen yet: the researcher chooses which facts the questions show, from this report.
- No table is written: the build writes them once the facts are chosen.
- Whether a fact gives the class away is measured in the audit, like the image dataset's shortcut check.

<!-- step: build -->
### Build: the three tables

- `questions.csv`: 2179 questions, one per image of the image dataset, with its split, class, file and lesion group unchanged: train 1739, val 120, test 320.
- Facts shown with each photo: age, sex, body site (chosen from the records report). A missing fact, or one the photo's two ISIC records disagree on, is shown as "not recorded".
- `knowledge.csv`: 7736 items open-book runs may search: 1739 training images, 444 training cases and 5553 library passages. No validation or test item of either dataset is among them. Each is linked to the questions only by its disease: the two datasets share no patient.
- `diseases.csv`: 8 classes, with the ISIC diagnoses, the text names and MeSH headings, and the counts.

#### Questions with all three facts

A missing fact points to melanoma or nevus (their photo collection records fewer facts), so each run is also scored on the questions with all three facts.

| Class | train | val | test | Missing age | Missing sex | Missing site |
|---|---|---|---|---|---|---|
| actinic keratosis | 76 of 76 | 15 of 15 | 40 of 40 | 0 | 0 | 0 |
| basal cell carcinoma | 336 of 337 | 15 of 15 | 40 of 40 | 0 | 0 | 1 |
| dermatofibroma | 54 of 56 | 15 of 15 | 40 of 40 | 0 | 2 | 2 |
| melanoma | 317 of 381 | 12 of 15 | 32 of 40 | 75 | 74 | 55 |
| nevus | 116 of 238 | 6 of 15 | 19 of 40 | 150 | 137 | 87 |
| pigmented benign keratosis | 418 of 423 | 15 of 15 | 40 of 40 | 2 | 0 | 3 |
| squamous cell carcinoma | 141 of 141 | 15 of 15 | 40 of 40 | 0 | 0 | 0 |
| vascular lesion | 87 of 87 | 13 of 15 | 38 of 40 | 4 | 0 | 0 |

#### Examples of what the model is shown next to the photo

- "Age: about 35 years. Sex: female. Body site: trunk."
- "Age: not recorded. Sex: not recorded. Body site: not recorded."
- "Age: about 30 years. Sex: female. Body site: upper extremity."
- "Age: not recorded. Sex: not recorded. Body site: upper extremity."

#### Not done in this step

- Whether retrieval may see the facts of training images is still to be decided: knowledge.csv names the training images, and their facts are in questions.csv, so either choice needs no new build.
- The audit checks the tables against their sources and measures how much each fact, and its absence, gives the class away.

<!-- step: audit -->
### Audit: the tables against their sources, and what the facts alone tell

Checked with code of its own: 2179 questions and 7736 knowledge items.

| Check | Failures |
|---|---|
| a question twice, or an image without its question | 0 |
| question differs from the image dataset | 0 |
| ISIC record missing or not the one the image dataset used | 0 |
| facts differ from the ISIC records | 0 |
| image file missing or changed | 0 |
| case file missing or changed | 0 |
| passage missing from passages.jsonl or changed | 0 |
| knowledge item twice | 0 |
| knowledge item missing | 0 |
| knowledge item that is not a training item or a passage | 0 |
| validation or test item in the knowledge | 0 |
| knowledge item linked other than by disease | 0 |
| knowledge image differs from the image dataset | 0 |
| knowledge case differs from the text dataset | 0 |
| knowledge case licence differs from its source | 0 |
| knowledge passage differs from the library | 0 |
| passage from an article of the text dataset's candidates | 0 |
| disease table does not list the classes | 0 |
| disease count differs from the sources | 0 |

**All checks passed.**

#### What the facts alone tell, with no photo

A counting model learns from the 1,739 training questions which class (and which photo collection) each value of a fact points to, every class weighing the same, and answers the test questions from the facts alone. Balanced accuracy, with its 95% interval (bootstrap over lesions). Chance: 12.5% for the class, 50% for the collection.

| Facts used | Test questions | Class | Photo collection |
|---|---|---|---|
| age | 320 | 24.1% (18.9–29.3%) | 69.1% (63.1–75.1%) |
| sex | 320 | 23.1% (19.1–26.6%) | 66.1% (61.3–71.4%) |
| body site | 320 | 30.6% (25.6–35.7%) | 60.8% (54.9–66.9%) |
| which facts are missing | 320 | 19.1% (17.2–20.9%) | 66.8% (61.7–72.2%) |
| all three facts | 320 | 35.9% (30.0–41.5%) | 68.8% (63.2–74.7%) |
| all three facts, questions with all three | 289 | 30.3% (24.6–35.9%) | 54.5% (49.6–59.8%) |

How to read it: a multimodal score is worth what it adds to the photo alone (the image dataset's run) and to the facts alone (this table). Age carries real clinical signal (nevi are found in younger patients); which facts are missing tells the photo collection, and so mostly melanoma or nevus, and no clinic works that way. Each multimodal report therefore also gives the score on the questions with all three facts.
