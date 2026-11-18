# Figures of the text dataset's cases: a free count

Written by `python -m src.build.cases.prepare count` on 2026-11-18. Only the captions were read: no image was downloaded or looked at. The rules are in `src/multimodal/figures.py`. A case counts when its article's full text is open in PubMed Central, the article holds no other case of the text dataset, and at least one figure's caption shows the lesion (it names dermoscopy, a clinical photo or a lesion) and names no microscope or stain, scan, state after treatment or at follow-up, removed specimen, drawing or chart, and no numbered case or several patients.

Rule fixed before the count: build the dataset when at least 80 test cases count and at least 6 classes have at least 8 test cases that count.

**Result: 107 test cases count; 7 classes have at least 8 (basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion). The rule is met: the dataset can be built.**

## Cases that count, per class and split

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

## Why the other cases do not count

| Reason (the first that applies) | Cases |
|---|---|
| no PMC number | 124 |
| no open full text in PMC | 11 |
| no figure | 2 |
| the article holds another case of the dataset | 34 |
| no figure whose caption shows the lesion alone | 227 |

Of the cases with no figure that shows the lesion alone, 171 have a figure that shows the lesion beside other parts (such as a panel of histology); they could count only if the figure's panels were cut apart.

## Figures that show the lesion but are left out, by what else the caption names

| The caption also names | Figures |
|---|---|
| microscope or stain | 464 |
| scan | 101 |
| after treatment or at follow-up | 241 |
| removed specimen | 82 |
| drawing or chart | 10 |
| may show another patient | 49 |

## The cases that count

- Figures that show the lesion: 504 in 347 cases; cases with more than one: 114; with a dermoscopy figure: 70
- Cases whose question cites one of these figures by number: 347 of 347 whose question is on this computer
- Licences of their articles: CC BY (any, no NC): 153; CC BY-NC (any): 180; not stated: 7; other: http://creativecommons.org/licenses/CC BY-NC/4.0: 4; other: open-access: 3

## What this count does not show

- No image was looked at. A caption can be wrong or incomplete; the build step checks the images themselves (what kind of picture each is, writing in the picture, the same picture twice, a picture also in the image dataset).
- A figure that counts may still show more than the lesion (an arrow, a ruler, a second panel the caption does not name).
- The captions stay on this computer (`data/multimodal/figures/captions.jsonl`), under their publishers' licences.

