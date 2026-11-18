# The case dataset

Written by `python -m src.multimodal.figures build`. One question per case of the text dataset whose own article shows the lesion in pictures that passed every check (`figure_checks.md`): the frozen text question, word for word, and the pictures, byte for byte, in the order of their figures. No caption is shown. The splits are the text dataset's.

| Class | Test | Validation | Train |
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

Questions: 335; pictures: 468. Pictures per question: 1: 237, 2: 74, 3: 19, 4: 2, 5: 2, 8: 1.

