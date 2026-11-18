# Checks of the pictures, each by itself

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
| the same picture is in the image dataset | 0 |
| the same picture is in another case | 0 |
| the same picture twice in this case | 0 |

## Do the captions and the image model agree on dermoscopy?

Of 468 pictures the model sees as a skin photo, the caption and the model agree on whether it is a dermoscopy image for 375; Cohen's kappa 0.27. A low kappa means the picture type cannot be taken from either alone.

Pictures with any writing (labels, arrows' letters, scale bars): 83; with the name of a diagnosis: 1.

