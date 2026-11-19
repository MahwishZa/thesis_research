# Preparation of the case dataset

Written by `python -m src.build.cases.prepare`, one section per step in the order the steps run; each step rewrites only its own section. The build records are in `build/`; the captions, pictures and the models' answers stay on this computer in `figures/`.

<!-- step: count -->
## Figures of the text dataset's cases: a free count

Written by `python -m src.build.cases.prepare count` on 2026-11-18. Only the captions were read: no image was downloaded or looked at. The rules are in `src/multimodal/figures.py`. A case counts when its article's full text is open in PubMed Central, the article holds no other case of the text dataset, and at least one figure's caption shows the lesion (it names dermoscopy, a clinical photo or a lesion) and names no microscope or stain, scan, state after treatment or at follow-up, removed specimen, drawing or chart, and no numbered case or several patients.

Rule fixed before the count: build the dataset when at least 80 test cases count and at least 6 classes have at least 8 test cases that count.

**Result: 107 test cases count; 7 classes have at least 8 (basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion). The rule is met: the dataset can be built.**

### Cases that count, per class and split

| Class | Test: cases | Test: count | Test: with a dermoscopy figure | Validation: cases | Validation: count | Train: cases | Train: count |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 2 | 1 | 0 | 1 | 0 | 1 | 1 |
| basal cell carcinoma | 40 | 20 | 2 | 15 | 9 | 121 | 54 |
| dermatofibroma | 21 | 11 | 2 | 7 | 1 | 7 | 4 |
| melanoma | 40 | 19 | 7 | 15 | 5 | 125 | 52 |
| nevus | 40 | 16 | 7 | 15 | 8 | 24 | 10 |
| pigmented benign keratosis | 22 | 11 | 2 | 7 | 5 | 7 | 4 |
| squamous cell carcinoma | 40 | 18 | 2 | 15 | 6 | 153 | 72 |
| vascular lesion | 16 | 11 | 0 | 5 | 5 | 6 | 4 |
| all | 221 | 107 | 22 | 80 | 39 | 444 | 201 |

### Why the other cases do not count

| Reason (the first that applies) | Cases |
|---|---|
| no PMC number | 124 |
| no open full text in PMC | 11 |
| no figure | 2 |
| the article holds another case of the dataset | 34 |
| no figure whose caption shows the lesion alone | 227 |

Of the cases with no figure that shows the lesion alone, 171 have a figure that shows the lesion beside other parts (such as a panel of histology); they could count only if the figure's panels were cut apart.

### Figures that show the lesion but are left out, by what else the caption names

| The caption also names | Figures |
|---|---|
| microscope or stain | 464 |
| scan | 101 |
| after treatment or at follow-up | 241 |
| removed specimen | 82 |
| drawing or chart | 10 |
| may show another patient | 49 |

### The cases that count

- Figures that show the lesion: 504 in 347 cases; cases with more than one: 114; with a dermoscopy figure: 70
- Cases whose question cites one of these figures by number: 347 of 347 whose question is on this computer
- Licences of their articles: CC BY (any, no NC): 153; CC BY-NC (any): 180; not stated: 7; other: http://creativecommons.org/licenses/CC BY-NC/4.0: 4; other: open-access: 3

### What this count does not show

- No image was looked at. A caption can be wrong or incomplete; the build step checks the images themselves (what kind of picture each is, writing in the picture, the same picture twice).
- A figure that counts may still show more than the lesion (an arrow, a ruler, a second panel the caption does not name).
- The captions stay on this computer (`data/cases/figures/captions.jsonl`), under their publishers' licences.

<!-- step: download -->
## Pictures of the figures that count

Written by `python -m src.build.cases.prepare download` on 2026-11-18. The pictures come from PMC Article Datasets (NCBI's cloud copy of PMC's open-access articles), byte for byte, each checked against PMC's MD5 checksum, and stay on this computer (`data/cases/figures/images/`). Nothing in them was changed, and no person looked at them.

| Class | Test: cases with a picture | Validation | Train |
|---|---|---|---|
| actinic keratosis | 1 | 0 | 1 |
| basal cell carcinoma | 19 | 9 | 53 |
| dermatofibroma | 11 | 1 | 4 |
| melanoma | 18 | 5 | 51 |
| nevus | 15 | 8 | 10 |
| pigmented benign keratosis | 11 | 5 | 4 |
| squamous cell carcinoma | 18 | 6 | 70 |
| vascular lesion | 11 | 4 | 4 |
| all | 104 | 38 | 197 |

Pictures saved: 492 for 339 of 347 cases.

| Figure or picture | Count |
|---|---|
| saved | 492 |
| not open access in PMC | 6 |
| the article is retracted | 0 |
| no picture file for the figure among the article's files | 6 |
| the picture cannot be opened | 0 |

<!-- step: check -->
## Checks of the pictures, each by itself

Written by `python -m src.build.cases.prepare check` on 2026-11-18. No person looked at the pictures. Image model: open_clip ViT-B-32 (laion2b_s34b_b79k), asked which of these a picture is: clinical photo ("a clinical photograph of a skin lesion"); dermoscopy ("a dermoscopy image of a skin lesion"); microscopy ("a microscope image of stained tissue"); scan ("a medical scan such as CT, MRI, X-ray or ultrasound"); drawing ("a chart, diagram, table or drawing"). Writing: EasyOCR (English). Versions of the models' answers made in this run: easyocr 1.7.2, open_clip 4.2.2, torch 2.15.0+cpu.

**After the checks: 104 test cases keep a picture; 7 classes have at least 8 (basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion). The rule for building the dataset (at least 80 test cases and 6 classes) is met.**

| Class | Test: cases with a picture kept | Validation | Train |
|---|---|---|---|
| actinic keratosis | 1 | 0 | 1 |
| basal cell carcinoma | 19 | 9 | 52 |
| dermatofibroma | 11 | 1 | 4 |
| melanoma | 18 | 5 | 50 |
| nevus | 15 | 8 | 10 |
| pigmented benign keratosis | 11 | 5 | 4 |
| squamous cell carcinoma | 18 | 6 | 68 |
| vascular lesion | 11 | 4 | 4 |
| all | 104 | 38 | 193 |

Pictures checked: 492; kept: 468.

| Left out because (a picture can have several reasons) | Pictures |
|---|---|
| smaller than the least size | 0 |
| writing in the picture names a diagnosis | 1 |
| the image model sees no photo of the skin | 24 |
| the same picture is in another case | 0 |
| the same picture twice in this case | 0 |

### Do the captions and the image model agree on dermoscopy?

Of 468 pictures the model sees as a skin photo, the caption and the model agree on whether it is a dermoscopy image for 375; Cohen's kappa 0.27. A low kappa means the picture type cannot be taken from either alone.

Pictures with any writing (labels, arrows' letters, scale bars): 83; with the name of a diagnosis: 1.

<!-- step: build -->
## The case dataset

Written by `python -m src.build.cases.prepare build`. One question per case of the text dataset whose own article shows the lesion in pictures that passed every check (the check section): the frozen text question, word for word, and the pictures, byte for byte, in the order of their figures. No caption is shown. The splits are the text dataset's.

| Class | Test | Validation | Train |
|---|---|---|---|
| actinic keratosis | 1 | 0 | 1 |
| basal cell carcinoma | 19 | 9 | 51 |
| dermatofibroma | 10 | 1 | 4 |
| melanoma | 17 | 5 | 47 |
| nevus | 15 | 8 | 9 |
| pigmented benign keratosis | 10 | 5 | 4 |
| squamous cell carcinoma | 17 | 6 | 66 |
| vascular lesion | 11 | 4 | 4 |
| all | 100 | 38 | 186 |

Questions: 324; pictures: 451. Pictures per question: 1: 231, 2: 70, 3: 18, 4: 2, 5: 2, 8: 1.
Cases with kept pictures that the text dataset no longer holds (left out by its review): 11 (10056737-4, 11035014-1, 4777905-1, 5690489-2, 6452477-1, pubmed-24396723, pubmed-24520521, pubmed-35872878, pubmed-39165299, pubmed-40250188, pubmed-42245639).
