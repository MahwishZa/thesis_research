# Preparation of the dataset

Written by `python -m src.image.prepare`, one section per step in the order the steps run; each step rewrites only its own section. What the steps find out about each photo is in `raw_photos.csv` (one row per original photo), the pairs of photos found alike are in `similar_photo_pairs.csv`, and `isic_to_class.csv` and `eye_check_verdicts.csv` are written by people. Nothing in `data/image/raw` is ever changed.

<!-- step: inventory -->
## Inventory of the original dataset

Written by `python -m src.image.prepare inventory` from the files in `data/image/raw`, which it does not change. One row per photo (identical files count once, with every file named) is in `raw_photos.csv`.

Fingerprint of `data/image/raw` (the path and SHA-256 of every image file): `0915dd82f562136d1151b61be6181e8fd1ab6072dadd4d0f47f69dc4ddd0bfaa`

### Files

| | Files |
|---|---|
| Image files in `<split>/<class>/` folders | 2357 |
| Opened completely without error | 2357 |
| Unreadable (damaged or not an image) | 0 |
| Readable, but the file type does not match its name | 0 |
| Other files in `data/image/raw` (not used) | 0 |
| Files whose name has no ISIC number | 0 |

### Files per folder and class (the dataset's own labels, not yet verified)

| Class folder | Test | Train | All files | Unique photos |
|---|---|---|---|---|
| actinic keratosis | 16 | 114 | 130 | 130 |
| basal cell carcinoma | 16 | 376 | 392 | 392 |
| dermatofibroma | 16 | 95 | 111 | 111 |
| melanoma | 16 | 438 | 454 | 453 |
| nevus | 16 | 357 | 373 | 372 |
| pigmented benign keratosis | 16 | 462 | 478 | 478 |
| seborrheic keratosis | 3 | 77 | 80 | 80 |
| squamous cell carcinoma | 16 | 181 | 197 | 197 |
| vascular lesion | 3 | 139 | 142 | 142 |
| **Total** | 118 | 2239 | 2357 | 2200 |

A photo filed under two class folders counts once in each folder's unique photos, so that column adds up to more than the total.

### How the images are stored

- File format: JPEG: 2357
- Colour mode: RGB: 2357
- EXIF orientation tag (1 = upright as stored; other values tell a viewer to rotate): none: 2212, 1: 145
- Embedded colour profile (ICC): no: 2355, yes: 2
- Smallest image (width x height): 600x450 (Test/actinic keratosis/ISIC_0024468.jpg); largest: 6668x4479 (Test/dermatofibroma/ISIC_0011410.jpg)

### Image sizes per class folder

The size of a photo should say nothing about its diagnosis. Where one class has its own typical size, a model could use size instead of the lesion.

| Class folder | Different sizes | Most common size (files) | Share of the folder |
|---|---|---|---|
| actinic keratosis | 2 | 600x450 (128) | 98% |
| basal cell carcinoma | 1 | 600x450 (392) | 100% |
| dermatofibroma | 8 | 600x450 (104) | 94% |
| melanoma | 43 | 1024x768 (291) | 64% |
| nevus | 46 | 600x450 (76) | 20% |
| pigmented benign keratosis | 1 | 600x450 (478) | 100% |
| seborrheic keratosis | 1 | 1024x768 (80) | 100% |
| squamous cell carcinoma | 5 | 600x450 (192) | 97% |
| vascular lesion | 1 | 600x450 (142) | 100% |

### Identical files (the same SHA-256)

- Unique photos: 2200 (from 2357 files)
- Groups of identical files: 157, holding 157 extra copies
- Groups filed under two or more class folders: 155
- Groups with copies in two or more split folders: 18
- Groups whose copies carry different ISIC numbers: 2
- ISIC numbers that name two or more different photos: 0

| Group | Files | Class folders | Split folders |
|---|---|---|---|
| 1 | Test/actinic keratosis/ISIC_0010512.jpg<br>Train/nevus/ISIC_0010512.jpg | actinic keratosis, nevus | Test, Train |
| 2 | Test/actinic keratosis/ISIC_0010889.jpg<br>Train/nevus/ISIC_0010889.jpg | actinic keratosis, nevus | Test, Train |
| 3 | Test/actinic keratosis/ISIC_0024468.jpg<br>Train/nevus/ISIC_0024468.jpg | actinic keratosis, nevus | Test, Train |
| 4 | Test/actinic keratosis/ISIC_0024470.jpg<br>Train/nevus/ISIC_0024470.jpg | actinic keratosis, nevus | Test, Train |
| 5 | Test/actinic keratosis/ISIC_0024511.jpg<br>Train/nevus/ISIC_0024511.jpg | actinic keratosis, nevus | Test, Train |
| 6 | Test/actinic keratosis/ISIC_0024646.jpg<br>Train/nevus/ISIC_0024646.jpg | actinic keratosis, nevus | Test, Train |
| 7 | Test/actinic keratosis/ISIC_0024707.jpg<br>Train/nevus/ISIC_0024707.jpg | actinic keratosis, nevus | Test, Train |
| 8 | Test/actinic keratosis/ISIC_0024763.jpg<br>Train/nevus/ISIC_0024763.jpg | actinic keratosis, nevus | Test, Train |
| 9 | Test/actinic keratosis/ISIC_0024771.jpg<br>Train/nevus/ISIC_0024771.jpg | actinic keratosis, nevus | Test, Train |
| 10 | Test/actinic keratosis/ISIC_0024800.jpg<br>Train/nevus/ISIC_0024800.jpg | actinic keratosis, nevus | Test, Train |
| 11 | Test/actinic keratosis/ISIC_0024913.jpg<br>Train/nevus/ISIC_0024913.jpg | actinic keratosis, nevus | Test, Train |
| 12 | Test/actinic keratosis/ISIC_0024948.jpg<br>Train/nevus/ISIC_0024948.jpg | actinic keratosis, nevus | Test, Train |
| 13 | Test/actinic keratosis/ISIC_0025368.jpg<br>Train/nevus/ISIC_0025368.jpg | actinic keratosis, nevus | Test, Train |
| 14 | Test/actinic keratosis/ISIC_0025427.jpg<br>Train/nevus/ISIC_0025427.jpg | actinic keratosis, nevus | Test, Train |
| 15 | Test/actinic keratosis/ISIC_0025605.jpg<br>Train/nevus/ISIC_0025605.jpg | actinic keratosis, nevus | Test, Train |
| 16 | Test/seborrheic keratosis/ISIC_0010803.jpg<br>Train/melanoma/ISIC_0010803.jpg | melanoma, seborrheic keratosis | Test, Train |
| 17 | Test/seborrheic keratosis/ISIC_0010808.jpg<br>Train/melanoma/ISIC_0010808.jpg | melanoma, seborrheic keratosis | Test, Train |
| 18 | Test/seborrheic keratosis/ISIC_0010809.jpg<br>Train/melanoma/ISIC_0010809.jpg | melanoma, seborrheic keratosis | Test, Train |
| 19 | Train/actinic keratosis/ISIC_0025780.jpg<br>Train/nevus/ISIC_0025780.jpg | actinic keratosis, nevus | Train |
| 20 | Train/actinic keratosis/ISIC_0025803.jpg<br>Train/nevus/ISIC_0025803.jpg | actinic keratosis, nevus | Train |
| 21 | Train/actinic keratosis/ISIC_0025825.jpg<br>Train/nevus/ISIC_0025825.jpg | actinic keratosis, nevus | Train |
| 22 | Train/actinic keratosis/ISIC_0025953.jpg<br>Train/nevus/ISIC_0025953.jpg | actinic keratosis, nevus | Train |
| 23 | Train/actinic keratosis/ISIC_0025957.jpg<br>Train/nevus/ISIC_0025957.jpg | actinic keratosis, nevus | Train |
| 24 | Train/actinic keratosis/ISIC_0025992.jpg<br>Train/nevus/ISIC_0025992.jpg | actinic keratosis, nevus | Train |
| 25 | Train/actinic keratosis/ISIC_0026040.jpg<br>Train/nevus/ISIC_0026040.jpg | actinic keratosis, nevus | Train |
| 26 | Train/actinic keratosis/ISIC_0026149.jpg<br>Train/nevus/ISIC_0026149.jpg | actinic keratosis, nevus | Train |
| 27 | Train/actinic keratosis/ISIC_0026171.jpg<br>Train/nevus/ISIC_0026171.jpg | actinic keratosis, nevus | Train |
| 28 | Train/actinic keratosis/ISIC_0026194.jpg<br>Train/nevus/ISIC_0026194.jpg | actinic keratosis, nevus | Train |
| 29 | Train/actinic keratosis/ISIC_0026212.jpg<br>Train/nevus/ISIC_0026212.jpg | actinic keratosis, nevus | Train |
| 30 | Train/actinic keratosis/ISIC_0026457.jpg<br>Train/nevus/ISIC_0026457.jpg | actinic keratosis, nevus | Train |
| 31 | Train/actinic keratosis/ISIC_0026468.jpg<br>Train/nevus/ISIC_0026468.jpg | actinic keratosis, nevus | Train |
| 32 | Train/actinic keratosis/ISIC_0026525.jpg<br>Train/nevus/ISIC_0026525.jpg | actinic keratosis, nevus | Train |
| 33 | Train/actinic keratosis/ISIC_0026575.jpg<br>Train/nevus/ISIC_0026575.jpg | actinic keratosis, nevus | Train |
| 34 | Train/actinic keratosis/ISIC_0026625.jpg<br>Train/nevus/ISIC_0026625.jpg | actinic keratosis, nevus | Train |
| 35 | Train/actinic keratosis/ISIC_0026626.jpg<br>Train/nevus/ISIC_0026626.jpg | actinic keratosis, nevus | Train |
| 36 | Train/actinic keratosis/ISIC_0026650.jpg<br>Train/nevus/ISIC_0026650.jpg | actinic keratosis, nevus | Train |
| 37 | Train/actinic keratosis/ISIC_0026702.jpg<br>Train/nevus/ISIC_0026702.jpg | actinic keratosis, nevus | Train |
| 38 | Train/actinic keratosis/ISIC_0026709.jpg<br>Train/nevus/ISIC_0026709.jpg | actinic keratosis, nevus | Train |
| 39 | Train/actinic keratosis/ISIC_0026729.jpg<br>Train/nevus/ISIC_0026729.jpg | actinic keratosis, nevus | Train |
| 40 | Train/actinic keratosis/ISIC_0026765.jpg<br>Train/nevus/ISIC_0026765.jpg | actinic keratosis, nevus | Train |
| 41 | Train/actinic keratosis/ISIC_0026848.jpg<br>Train/nevus/ISIC_0026848.jpg | actinic keratosis, nevus | Train |
| 42 | Train/actinic keratosis/ISIC_0026857.jpg<br>Train/nevus/ISIC_0026857.jpg | actinic keratosis, nevus | Train |
| 43 | Train/actinic keratosis/ISIC_0026905.jpg<br>Train/nevus/ISIC_0026905.jpg | actinic keratosis, nevus | Train |
| 44 | Train/actinic keratosis/ISIC_0026984.jpg<br>Train/nevus/ISIC_0026984.jpg | actinic keratosis, nevus | Train |
| 45 | Train/actinic keratosis/ISIC_0027172.jpg<br>Train/nevus/ISIC_0027172.jpg | actinic keratosis, nevus | Train |
| 46 | Train/actinic keratosis/ISIC_0027254.jpg<br>Train/nevus/ISIC_0027254.jpg | actinic keratosis, nevus | Train |
| 47 | Train/actinic keratosis/ISIC_0027334.jpg<br>Train/nevus/ISIC_0027334.jpg | actinic keratosis, nevus | Train |
| 48 | Train/actinic keratosis/ISIC_0027447.jpg<br>Train/nevus/ISIC_0027447.jpg | actinic keratosis, nevus | Train |
| 49 | Train/actinic keratosis/ISIC_0027452.jpg<br>Train/nevus/ISIC_0027452.jpg | actinic keratosis, nevus | Train |
| 50 | Train/actinic keratosis/ISIC_0027536.jpg<br>Train/nevus/ISIC_0027536.jpg | actinic keratosis, nevus | Train |
| 51 | Train/actinic keratosis/ISIC_0027562.jpg<br>Train/nevus/ISIC_0027562.jpg | actinic keratosis, nevus | Train |
| 52 | Train/actinic keratosis/ISIC_0027580.jpg<br>Train/nevus/ISIC_0027580.jpg | actinic keratosis, nevus | Train |
| 53 | Train/actinic keratosis/ISIC_0027615.jpg<br>Train/nevus/ISIC_0027615.jpg | actinic keratosis, nevus | Train |
| 54 | Train/actinic keratosis/ISIC_0027650.jpg<br>Train/nevus/ISIC_0027650.jpg | actinic keratosis, nevus | Train |
| 55 | Train/actinic keratosis/ISIC_0027668.jpg<br>Train/nevus/ISIC_0027668.jpg | actinic keratosis, nevus | Train |
| 56 | Train/actinic keratosis/ISIC_0027802.jpg<br>Train/nevus/ISIC_0027802.jpg | actinic keratosis, nevus | Train |
| 57 | Train/actinic keratosis/ISIC_0027829.jpg<br>Train/nevus/ISIC_0027829.jpg | actinic keratosis, nevus | Train |
| 58 | Train/actinic keratosis/ISIC_0027884.jpg<br>Train/nevus/ISIC_0027884.jpg | actinic keratosis, nevus | Train |
| 59 | Train/actinic keratosis/ISIC_0027896.jpg<br>Train/nevus/ISIC_0027896.jpg | actinic keratosis, nevus | Train |
| 60 | Train/actinic keratosis/ISIC_0027950.jpg<br>Train/nevus/ISIC_0027950.jpg | actinic keratosis, nevus | Train |
| 61 | Train/actinic keratosis/ISIC_0027958.jpg<br>Train/nevus/ISIC_0027958.jpg | actinic keratosis, nevus | Train |
| 62 | Train/actinic keratosis/ISIC_0028063.jpg<br>Train/nevus/ISIC_0028063.jpg | actinic keratosis, nevus | Train |
| 63 | Train/actinic keratosis/ISIC_0028076.jpg<br>Train/nevus/ISIC_0028076.jpg | actinic keratosis, nevus | Train |
| 64 | Train/actinic keratosis/ISIC_0028190.jpg<br>Train/nevus/ISIC_0028190.jpg | actinic keratosis, nevus | Train |
| 65 | Train/actinic keratosis/ISIC_0028314.jpg<br>Train/nevus/ISIC_0028314.jpg | actinic keratosis, nevus | Train |
| 66 | Train/actinic keratosis/ISIC_0028370.jpg<br>Train/nevus/ISIC_0028370.jpg | actinic keratosis, nevus | Train |
| 67 | Train/actinic keratosis/ISIC_0028393.jpg<br>Train/nevus/ISIC_0028393.jpg | actinic keratosis, nevus | Train |
| 68 | Train/actinic keratosis/ISIC_0028517.jpg<br>Train/nevus/ISIC_0028517.jpg | actinic keratosis, nevus | Train |
| 69 | Train/actinic keratosis/ISIC_0028558.jpg<br>Train/nevus/ISIC_0028558.jpg | actinic keratosis, nevus | Train |
| 70 | Train/actinic keratosis/ISIC_0028820.jpg<br>Train/nevus/ISIC_0028820.jpg | actinic keratosis, nevus | Train |
| 71 | Train/actinic keratosis/ISIC_0028854.jpg<br>Train/nevus/ISIC_0028854.jpg | actinic keratosis, nevus | Train |
| 72 | Train/actinic keratosis/ISIC_0028941.jpg<br>Train/nevus/ISIC_0028941.jpg | actinic keratosis, nevus | Train |
| 73 | Train/actinic keratosis/ISIC_0028990.jpg<br>Train/nevus/ISIC_0028990.jpg | actinic keratosis, nevus | Train |
| 74 | Train/actinic keratosis/ISIC_0029025.jpg<br>Train/nevus/ISIC_0029025.jpg | actinic keratosis, nevus | Train |
| 75 | Train/actinic keratosis/ISIC_0029041.jpg<br>Train/nevus/ISIC_0029041.jpg | actinic keratosis, nevus | Train |
| 76 | Train/actinic keratosis/ISIC_0029133.jpg<br>Train/nevus/ISIC_0029133.jpg | actinic keratosis, nevus | Train |
| 77 | Train/actinic keratosis/ISIC_0029141.jpg<br>Train/nevus/ISIC_0029141.jpg | actinic keratosis, nevus | Train |
| 78 | Train/actinic keratosis/ISIC_0029210.jpg<br>Train/nevus/ISIC_0029210.jpg | actinic keratosis, nevus | Train |
| 79 | Train/actinic keratosis/ISIC_0029309.jpg<br>Train/nevus/ISIC_0029309.jpg | actinic keratosis, nevus | Train |
| 80 | Train/actinic keratosis/ISIC_0029460.jpg<br>Train/nevus/ISIC_0029460.jpg | actinic keratosis, nevus | Train |
| 81 | Train/melanoma/ISIC_0000279.jpg<br>Train/melanoma/ISIC_0000280.jpg | melanoma | Train |
| 82 | Train/melanoma/ISIC_0010810.jpg<br>Train/seborrheic keratosis/ISIC_0010810.jpg | melanoma, seborrheic keratosis | Train |
| 83 | Train/melanoma/ISIC_0010812.jpg<br>Train/seborrheic keratosis/ISIC_0010812.jpg | melanoma, seborrheic keratosis | Train |
| 84 | Train/melanoma/ISIC_0010834.jpg<br>Train/seborrheic keratosis/ISIC_0010834.jpg | melanoma, seborrheic keratosis | Train |
| 85 | Train/melanoma/ISIC_0010852.jpg<br>Train/seborrheic keratosis/ISIC_0010852.jpg | melanoma, seborrheic keratosis | Train |
| 86 | Train/melanoma/ISIC_0010860.jpg<br>Train/seborrheic keratosis/ISIC_0010860.jpg | melanoma, seborrheic keratosis | Train |
| 87 | Train/melanoma/ISIC_0010862.jpg<br>Train/seborrheic keratosis/ISIC_0010862.jpg | melanoma, seborrheic keratosis | Train |
| 88 | Train/melanoma/ISIC_0010863.jpg<br>Train/seborrheic keratosis/ISIC_0010863.jpg | melanoma, seborrheic keratosis | Train |
| 89 | Train/melanoma/ISIC_0010866.jpg<br>Train/seborrheic keratosis/ISIC_0010866.jpg | melanoma, seborrheic keratosis | Train |
| 90 | Train/melanoma/ISIC_0010872.jpg<br>Train/seborrheic keratosis/ISIC_0010872.jpg | melanoma, seborrheic keratosis | Train |
| 91 | Train/melanoma/ISIC_0010876.jpg<br>Train/seborrheic keratosis/ISIC_0010876.jpg | melanoma, seborrheic keratosis | Train |
| 92 | Train/melanoma/ISIC_0010878.jpg<br>Train/seborrheic keratosis/ISIC_0010878.jpg | melanoma, seborrheic keratosis | Train |
| 93 | Train/melanoma/ISIC_0010883.jpg<br>Train/seborrheic keratosis/ISIC_0010883.jpg | melanoma, seborrheic keratosis | Train |
| 94 | Train/melanoma/ISIC_0010890.jpg<br>Train/seborrheic keratosis/ISIC_0010890.jpg | melanoma, seborrheic keratosis | Train |
| 95 | Train/melanoma/ISIC_0010898.jpg<br>Train/seborrheic keratosis/ISIC_0010898.jpg | melanoma, seborrheic keratosis | Train |
| 96 | Train/melanoma/ISIC_0010900.jpg<br>Train/seborrheic keratosis/ISIC_0010900.jpg | melanoma, seborrheic keratosis | Train |
| 97 | Train/melanoma/ISIC_0010909.jpg<br>Train/seborrheic keratosis/ISIC_0010909.jpg | melanoma, seborrheic keratosis | Train |
| 98 | Train/melanoma/ISIC_0010912.jpg<br>Train/seborrheic keratosis/ISIC_0010912.jpg | melanoma, seborrheic keratosis | Train |
| 99 | Train/melanoma/ISIC_0010917.jpg<br>Train/seborrheic keratosis/ISIC_0010917.jpg | melanoma, seborrheic keratosis | Train |
| 100 | Train/melanoma/ISIC_0010923.jpg<br>Train/seborrheic keratosis/ISIC_0010923.jpg | melanoma, seborrheic keratosis | Train |
| 101 | Train/melanoma/ISIC_0010925.jpg<br>Train/seborrheic keratosis/ISIC_0010925.jpg | melanoma, seborrheic keratosis | Train |
| 102 | Train/melanoma/ISIC_0010934.jpg<br>Train/seborrheic keratosis/ISIC_0010934.jpg | melanoma, seborrheic keratosis | Train |
| 103 | Train/melanoma/ISIC_0010938.jpg<br>Train/seborrheic keratosis/ISIC_0010938.jpg | melanoma, seborrheic keratosis | Train |
| 104 | Train/melanoma/ISIC_0010942.jpg<br>Train/seborrheic keratosis/ISIC_0010942.jpg | melanoma, seborrheic keratosis | Train |
| 105 | Train/melanoma/ISIC_0010965.jpg<br>Train/seborrheic keratosis/ISIC_0010965.jpg | melanoma, seborrheic keratosis | Train |
| 106 | Train/melanoma/ISIC_0010971.jpg<br>Train/seborrheic keratosis/ISIC_0010971.jpg | melanoma, seborrheic keratosis | Train |
| 107 | Train/melanoma/ISIC_0010977.jpg<br>Train/seborrheic keratosis/ISIC_0010977.jpg | melanoma, seborrheic keratosis | Train |
| 108 | Train/melanoma/ISIC_0010978.jpg<br>Train/seborrheic keratosis/ISIC_0010978.jpg | melanoma, seborrheic keratosis | Train |
| 109 | Train/melanoma/ISIC_0010979.jpg<br>Train/seborrheic keratosis/ISIC_0010979.jpg | melanoma, seborrheic keratosis | Train |
| 110 | Train/melanoma/ISIC_0010982.jpg<br>Train/seborrheic keratosis/ISIC_0010982.jpg | melanoma, seborrheic keratosis | Train |
| 111 | Train/melanoma/ISIC_0010990.jpg<br>Train/seborrheic keratosis/ISIC_0010990.jpg | melanoma, seborrheic keratosis | Train |
| 112 | Train/melanoma/ISIC_0010993.jpg<br>Train/seborrheic keratosis/ISIC_0010993.jpg | melanoma, seborrheic keratosis | Train |
| 113 | Train/melanoma/ISIC_0010995.jpg<br>Train/seborrheic keratosis/ISIC_0010995.jpg | melanoma, seborrheic keratosis | Train |
| 114 | Train/melanoma/ISIC_0011007.jpg<br>Train/seborrheic keratosis/ISIC_0011007.jpg | melanoma, seborrheic keratosis | Train |
| 115 | Train/melanoma/ISIC_0011012.jpg<br>Train/seborrheic keratosis/ISIC_0011012.jpg | melanoma, seborrheic keratosis | Train |
| 116 | Train/melanoma/ISIC_0011013.jpg<br>Train/seborrheic keratosis/ISIC_0011013.jpg | melanoma, seborrheic keratosis | Train |
| 117 | Train/melanoma/ISIC_0011030.jpg<br>Train/seborrheic keratosis/ISIC_0011030.jpg | melanoma, seborrheic keratosis | Train |
| 118 | Train/melanoma/ISIC_0011031.jpg<br>Train/seborrheic keratosis/ISIC_0011031.jpg | melanoma, seborrheic keratosis | Train |
| 119 | Train/melanoma/ISIC_0011032.jpg<br>Train/seborrheic keratosis/ISIC_0011032.jpg | melanoma, seborrheic keratosis | Train |
| 120 | Train/melanoma/ISIC_0011039.jpg<br>Train/seborrheic keratosis/ISIC_0011039.jpg | melanoma, seborrheic keratosis | Train |
| 121 | Train/melanoma/ISIC_0011040.jpg<br>Train/seborrheic keratosis/ISIC_0011040.jpg | melanoma, seborrheic keratosis | Train |
| 122 | Train/melanoma/ISIC_0011042.jpg<br>Train/seborrheic keratosis/ISIC_0011042.jpg | melanoma, seborrheic keratosis | Train |
| 123 | Train/melanoma/ISIC_0011043.jpg<br>Train/seborrheic keratosis/ISIC_0011043.jpg | melanoma, seborrheic keratosis | Train |
| 124 | Train/melanoma/ISIC_0011045.jpg<br>Train/seborrheic keratosis/ISIC_0011045.jpg | melanoma, seborrheic keratosis | Train |
| 125 | Train/melanoma/ISIC_0011052.jpg<br>Train/seborrheic keratosis/ISIC_0011052.jpg | melanoma, seborrheic keratosis | Train |
| 126 | Train/melanoma/ISIC_0011056.jpg<br>Train/seborrheic keratosis/ISIC_0011056.jpg | melanoma, seborrheic keratosis | Train |
| 127 | Train/melanoma/ISIC_0011057.jpg<br>Train/seborrheic keratosis/ISIC_0011057.jpg | melanoma, seborrheic keratosis | Train |
| 128 | Train/melanoma/ISIC_0011077.jpg<br>Train/seborrheic keratosis/ISIC_0011077.jpg | melanoma, seborrheic keratosis | Train |
| 129 | Train/melanoma/ISIC_0011094.jpg<br>Train/seborrheic keratosis/ISIC_0011094.jpg | melanoma, seborrheic keratosis | Train |
| 130 | Train/melanoma/ISIC_0011095.jpg<br>Train/seborrheic keratosis/ISIC_0011095.jpg | melanoma, seborrheic keratosis | Train |
| 131 | Train/melanoma/ISIC_0011099.jpg<br>Train/seborrheic keratosis/ISIC_0011099.jpg | melanoma, seborrheic keratosis | Train |
| 132 | Train/melanoma/ISIC_0011112.jpg<br>Train/seborrheic keratosis/ISIC_0011112.jpg | melanoma, seborrheic keratosis | Train |
| 133 | Train/melanoma/ISIC_0011121.jpg<br>Train/seborrheic keratosis/ISIC_0011121.jpg | melanoma, seborrheic keratosis | Train |
| 134 | Train/melanoma/ISIC_0011125.jpg<br>Train/seborrheic keratosis/ISIC_0011125.jpg | melanoma, seborrheic keratosis | Train |
| 135 | Train/melanoma/ISIC_0011128.jpg<br>Train/seborrheic keratosis/ISIC_0011128.jpg | melanoma, seborrheic keratosis | Train |
| 136 | Train/melanoma/ISIC_0011130.jpg<br>Train/seborrheic keratosis/ISIC_0011130.jpg | melanoma, seborrheic keratosis | Train |
| 137 | Train/melanoma/ISIC_0011132.jpg<br>Train/seborrheic keratosis/ISIC_0011132.jpg | melanoma, seborrheic keratosis | Train |
| 138 | Train/melanoma/ISIC_0011135.jpg<br>Train/seborrheic keratosis/ISIC_0011135.jpg | melanoma, seborrheic keratosis | Train |
| 139 | Train/melanoma/ISIC_0011136.jpg<br>Train/seborrheic keratosis/ISIC_0011136.jpg | melanoma, seborrheic keratosis | Train |
| 140 | Train/melanoma/ISIC_0011137.jpg<br>Train/seborrheic keratosis/ISIC_0011137.jpg | melanoma, seborrheic keratosis | Train |
| 141 | Train/melanoma/ISIC_0011138.jpg<br>Train/seborrheic keratosis/ISIC_0011138.jpg | melanoma, seborrheic keratosis | Train |
| 142 | Train/melanoma/ISIC_0011139.jpg<br>Train/seborrheic keratosis/ISIC_0011139.jpg | melanoma, seborrheic keratosis | Train |
| 143 | Train/melanoma/ISIC_0011140.jpg<br>Train/seborrheic keratosis/ISIC_0011140.jpg | melanoma, seborrheic keratosis | Train |
| 144 | Train/melanoma/ISIC_0011151.jpg<br>Train/seborrheic keratosis/ISIC_0011151.jpg | melanoma, seborrheic keratosis | Train |
| 145 | Train/melanoma/ISIC_0011158.jpg<br>Train/seborrheic keratosis/ISIC_0011158.jpg | melanoma, seborrheic keratosis | Train |
| 146 | Train/melanoma/ISIC_0011161.jpg<br>Train/seborrheic keratosis/ISIC_0011161.jpg | melanoma, seborrheic keratosis | Train |
| 147 | Train/melanoma/ISIC_0011163.jpg<br>Train/seborrheic keratosis/ISIC_0011163.jpg | melanoma, seborrheic keratosis | Train |
| 148 | Train/melanoma/ISIC_0011166.jpg<br>Train/seborrheic keratosis/ISIC_0011166.jpg | melanoma, seborrheic keratosis | Train |
| 149 | Train/melanoma/ISIC_0011169.jpg<br>Train/seborrheic keratosis/ISIC_0011169.jpg | melanoma, seborrheic keratosis | Train |
| 150 | Train/melanoma/ISIC_0011174.jpg<br>Train/seborrheic keratosis/ISIC_0011174.jpg | melanoma, seborrheic keratosis | Train |
| 151 | Train/melanoma/ISIC_0011176.jpg<br>Train/seborrheic keratosis/ISIC_0011176.jpg | melanoma, seborrheic keratosis | Train |
| 152 | Train/melanoma/ISIC_0011178.jpg<br>Train/seborrheic keratosis/ISIC_0011178.jpg | melanoma, seborrheic keratosis | Train |
| 153 | Train/melanoma/ISIC_0011181.jpg<br>Train/seborrheic keratosis/ISIC_0011181.jpg | melanoma, seborrheic keratosis | Train |
| 154 | Train/melanoma/ISIC_0011185.jpg<br>Train/seborrheic keratosis/ISIC_0011185.jpg | melanoma, seborrheic keratosis | Train |
| 155 | Train/melanoma/ISIC_0011187.jpg<br>Train/seborrheic keratosis/ISIC_0011187.jpg | melanoma, seborrheic keratosis | Train |
| 156 | Train/melanoma/ISIC_0011193.jpg<br>Train/seborrheic keratosis/ISIC_0011193.jpg | melanoma, seborrheic keratosis | Train |
| 157 | Train/nevus/ISIC_0000477.jpg<br>Train/nevus/ISIC_0000512.jpg | nevus | Train |

### Not checked in this step

- Whether the class folders are right: the labels are checked against the ISIC Archive in a later step.
- Photos of the same lesion, and near-identical photos with different bytes: found in later steps, from the ISIC records and by comparing the photos.

<!-- step: isic -->
## ISIC Archive records of the original dataset

Written by `python -m src.image.prepare isic` from the public ISIC Archive record of every ISIC number of the photos (the records themselves are kept unchanged in `data/image/downloads/isic/`; what they say about each photo is in `raw_photos.csv`, with a fingerprint of each record). No label has been changed: this is the evidence for the naming rule that maps ISIC's wording to the class names.

- ISIC numbers: 2202; found in the ISIC Archive: 2202; not found: 0
- With a diagnosis recorded: 2202
- Downloaded between 2026-11-11T05:24:37+00:00 and 2026-11-11T06:15:37+00:00 (UTC)

A photo filed under two Kaggle folders is counted in both, and a photo with two ISIC numbers is counted twice.

### ISIC diagnosis in each Kaggle folder (levels 1 > 2 > 3)

| Kaggle folder | ISIC diagnosis | ISIC numbers |
|---|---|---|
| actinic keratosis | Indeterminate > Indeterminate epidermal proliferations > Solar or actinic keratosis | 130 |
| basal cell carcinoma | Malignant > Malignant adnexal epithelial proliferations - Follicular > Basal cell carcinoma | 392 |
| dermatofibroma | Benign > Benign soft tissue proliferations - Fibro-histiocytic > Dermatofibroma | 111 |
| melanoma | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma in situ | 164 |
| melanoma | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive | 156 |
| melanoma | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma, NOS | 131 |
| melanoma | Malignant > Collision - At least one malignant proliferation | 3 |
| nevus | Benign > Benign melanocytic proliferations > Nevus | 294 |
| nevus | Indeterminate > Indeterminate epidermal proliferations > Solar or actinic keratosis | 78 |
| nevus | Benign > Collision - Only benign proliferations | 1 |
| pigmented benign keratosis | Benign > Benign epidermal proliferations > Pigmented benign keratosis | 478 |
| seborrheic keratosis | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma in situ | 45 |
| seborrheic keratosis | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive | 35 |
| squamous cell carcinoma | Malignant > Malignant epidermal proliferations > Squamous cell carcinoma, NOS | 192 |
| squamous cell carcinoma | Malignant > Malignant epidermal proliferations > Squamous cell carcinoma in situ | 5 |
| vascular lesion | Benign > Benign soft tissue proliferations - Vascular | 142 |

### How the diagnosis was confirmed

| Kaggle folder | confocal microscopy with consensus dermoscopy | histopathology | single image expert consensus | (not recorded) |
|---|---|---|---|---|
| actinic keratosis |  | 130 |  |  |
| basal cell carcinoma |  | 392 |  |  |
| dermatofibroma |  | 55 | 56 |  |
| melanoma |  | 454 |  |  |
| nevus |  | 78 |  | 295 |
| pigmented benign keratosis | 43 | 332 | 103 |  |
| seborrheic keratosis |  | 80 |  |  |
| squamous cell carcinoma |  | 197 |  |  |
| vascular lesion |  | 67 | 75 |  |

### Biopsy taken when the photo was made (concomitant_biopsy)

| Kaggle folder | False | True |
|---|---|---|
| actinic keratosis |  | 130 |
| basal cell carcinoma |  | 392 |
| dermatofibroma | 56 | 55 |
| melanoma |  | 454 |
| nevus | 295 | 78 |
| pigmented benign keratosis | 146 | 332 |
| seborrheic keratosis |  | 80 |
| squamous cell carcinoma |  | 197 |
| vascular lesion | 75 | 67 |

### Image type

| Kaggle folder | dermoscopic |
|---|---|
| actinic keratosis | 130 |
| basal cell carcinoma | 392 |
| dermatofibroma | 111 |
| melanoma | 454 |
| nevus | 373 |
| pigmented benign keratosis | 478 |
| seborrheic keratosis | 80 |
| squamous cell carcinoma | 197 |
| vascular lesion | 142 |

### Licence

| Kaggle folder | CC-0 | CC-BY-NC |
|---|---|---|
| actinic keratosis | 2 | 128 |
| basal cell carcinoma |  | 392 |
| dermatofibroma | 7 | 104 |
| melanoma | 454 |  |
| nevus | 297 | 76 |
| pigmented benign keratosis |  | 478 |
| seborrheic keratosis | 80 |  |
| squamous cell carcinoma | 5 | 192 |
| vascular lesion |  | 142 |

### Photos of the same lesion (lesion numbers)

- ISIC numbers with a lesion number: 1447 of 2202
- With a patient number: 10 of 2202
- Lesions with two or more ISIC numbers: 310
- Of these, lesions whose photos sit in both Kaggle splits (Train and Test): 32

| Lesion | ISIC numbers (Kaggle split/folder) |
|---|---|
| IL_1003725 | ISIC_0024461 (Test; basal cell carcinoma), ISIC_0025530 (Train; basal cell carcinoma), ISIC_0030526 (Train; basal cell carcinoma) |
| IL_1068372 | ISIC_0024452 (Test; basal cell carcinoma), ISIC_0029539 (Train; basal cell carcinoma), ISIC_0031258 (Train; basal cell carcinoma) |
| IL_1158764 | ISIC_0025605 (Test and Train; actinic keratosis, nevus), ISIC_0025957 (Train; actinic keratosis, nevus) |
| IL_1602255 | ISIC_0024372 (Test; squamous cell carcinoma), ISIC_0029533 (Train; squamous cell carcinoma), ISIC_0031012 (Train; squamous cell carcinoma) |
| IL_1622746 | ISIC_0024408 (Test; pigmented benign keratosis), ISIC_0027559 (Train; pigmented benign keratosis) |
| IL_2191346 | ISIC_0024418 (Test; squamous cell carcinoma), ISIC_0027343 (Train; squamous cell carcinoma) |
| IL_2333054 | ISIC_0024331 (Test; basal cell carcinoma), ISIC_0027531 (Train; basal cell carcinoma) |
| IL_2455881 | ISIC_0024579 (Test; squamous cell carcinoma), ISIC_0025577 (Train; squamous cell carcinoma), ISIC_0029638 (Train; squamous cell carcinoma), ISIC_0030591 (Train; squamous cell carcinoma) |
| IL_2706598 | ISIC_0024426 (Test; pigmented benign keratosis), ISIC_0026265 (Train; pigmented benign keratosis) |
| IL_2737157 | ISIC_0024448 (Test; basal cell carcinoma), ISIC_0027966 (Train; basal cell carcinoma) |
| IL_2755785 | ISIC_0024409 (Test; pigmented benign keratosis), ISIC_0025378 (Train; pigmented benign keratosis), ISIC_0026530 (Train; pigmented benign keratosis), ISIC_0027270 (Train; pigmented benign keratosis) |
| IL_2979493 | ISIC_0024463 (Test; squamous cell carcinoma), ISIC_0027529 (Train; squamous cell carcinoma), ISIC_0028795 (Train; squamous cell carcinoma) |
| IL_3065371 | ISIC_0024411 (Test; basal cell carcinoma), ISIC_0027920 (Train; basal cell carcinoma) |
| IL_3471258 | ISIC_0024312 (Test; pigmented benign keratosis), ISIC_0028628 (Train; pigmented benign keratosis) |
| IL_3476043 | ISIC_0024575 (Test; squamous cell carcinoma), ISIC_0029932 (Train; squamous cell carcinoma) |
| IL_3897052 | ISIC_0024646 (Test and Train; actinic keratosis, nevus), ISIC_0026040 (Train; actinic keratosis, nevus) |
| IL_4380450 | ISIC_0024358 (Test; pigmented benign keratosis), ISIC_0027771 (Train; pigmented benign keratosis) |
| IL_4580158 | ISIC_0024443 (Test; basal cell carcinoma), ISIC_0025509 (Train; basal cell carcinoma) |
| IL_5132103 | ISIC_0024763 (Test and Train; actinic keratosis, nevus), ISIC_0031228 (Train; actinic keratosis) |
| IL_5698688 | ISIC_0024360 (Test; basal cell carcinoma), ISIC_0029899 (Train; basal cell carcinoma) |
| IL_5871784 | ISIC_0024402 (Test; vascular lesion), ISIC_0027256 (Train; vascular lesion) |
| IL_5904233 | ISIC_0024450 (Test; squamous cell carcinoma), ISIC_0026522 (Train; squamous cell carcinoma), ISIC_0031743 (Train; squamous cell carcinoma) |
| IL_6302381 | ISIC_0024468 (Test and Train; actinic keratosis, nevus), ISIC_0024771 (Test and Train; actinic keratosis, nevus), ISIC_0031823 (Train; actinic keratosis) |
| IL_6724021 | ISIC_0024472 (Test; basal cell carcinoma), ISIC_0030755 (Train; basal cell carcinoma), ISIC_0031175 (Train; basal cell carcinoma) |
| IL_6894920 | ISIC_0024522 (Test; squamous cell carcinoma), ISIC_0030387 (Train; squamous cell carcinoma) |
| IL_7148773 | ISIC_0024338 (Test; pigmented benign keratosis), ISIC_0026139 (Train; pigmented benign keratosis), ISIC_0027967 (Train; pigmented benign keratosis) |
| IL_7726867 | ISIC_0024654 (Test; actinic keratosis), ISIC_0031430 (Train; actinic keratosis) |
| IL_7886196 | ISIC_0024973 (Test; dermatofibroma), ISIC_0029891 (Train; dermatofibroma) |
| IL_8119929 | ISIC_0024470 (Test and Train; actinic keratosis, nevus), ISIC_0028517 (Train; actinic keratosis, nevus) |
| IL_8930899 | ISIC_0024948 (Test and Train; actinic keratosis, nevus), ISIC_0027829 (Train; actinic keratosis, nevus) |
| IL_9086281 | ISIC_0024431 (Test; basal cell carcinoma), ISIC_0026656 (Train; basal cell carcinoma) |
| IL_9200198 | ISIC_0024345 (Test; basal cell carcinoma), ISIC_0030514 (Train; basal cell carcinoma) |

### Not done in this step

- No label is changed. The naming rule (which ISIC wording belongs to which class) is written next, with the supervisor.
- Whether each file really is the image its ISIC number names is checked in a later step, against ISIC's own picture.

<!-- step: identity -->
## Identity check: each photo against the ISIC Archive's own picture

Written by `python -m src.image.prepare identity`. For every ISIC number, the photo in `data/image/raw` is compared with the preview picture the ISIC Archive serves for that number: the shape (width / height), the mean colour difference of 32 x 32 versions (0 to 255; 0 = the same) and a 64-bit difference hash of an 8 x 9 grey version (0 = the same outline). Nothing is changed; the result for each photo is in `raw_photos.csv`.

- ISIC numbers: 2202; same picture: 2202
- "same picture" means: shapes differ by at most 2.0%, mean difference at most 5.0 and hash distance at most 16 of 64. Anything else is listed below for an eye check.

### Mean colour differences (0 = the same picture; different lesions were never below 8.6 in the calibration)

| Mean difference | ISIC numbers |
|---|---|
| 0 to 1 | 1787 |
| over 1 to 2 | 403 |
| over 2 to 5 | 12 |
| over 5 to 10 | 0 |
| over 10 | 0 |

### Largest differences among the pictures judged the same

- Shape: 0.3%; hash distance: 6; mean difference: 4.5

### Not done in this step

- No label, file or record is changed.

<!-- step: ham10000 -->
## HAM10000 cross-check

Written by `python -m src.image.prepare ham10000`. Every photo of this dataset that is also in HAM10000 is compared with HAM10000's own metadata: its diagnosis group, how the diagnosis was confirmed, and which photos show the same lesion. No label is changed; the result for each photo is in `raw_photos.csv`.

Source: HAM10000 (Tschandl, Rosendahl and Kittler, *Scientific Data* 5:180161, 2018), Harvard Dataverse doi:10.7910/DVN/DBW86T, version 4.0, file `HAM10000_metadata`, downloaded 2026-11-11T08:53:45+00:00 (UTC). MD5 published by Dataverse: `8f85fb1aa29d80a2797247e434deb79d`; of the downloaded file: `8f85fb1aa29d80a2797247e434deb79d` (they match).

- ISIC numbers in this dataset: 2202; also in HAM10000: 1437
- Diagnosis group: agree: 1437
- How the diagnosis was confirmed: agree: 1437
- Lesion groups: agree: 1437

### Photos in HAM10000 per Kaggle folder

| Kaggle folder | ISIC numbers | In HAM10000 |
|---|---|---|
| actinic keratosis | 130 | 128 |
| basal cell carcinoma | 392 | 392 |
| dermatofibroma | 111 | 104 |
| melanoma | 454 | 0 |
| nevus | 373 | 76 |
| pigmented benign keratosis | 478 | 478 |
| seborrheic keratosis | 80 | 0 |
| squamous cell carcinoma | 197 | 192 |
| vascular lesion | 142 | 142 |

### ISIC's wording against HAM10000's diagnosis group

| ISIC wording (rows), HAM10000 group (columns) | akiec | bcc | bkl | df | vasc |
|---|---|---|---|---|---|
| Basal cell carcinoma |  | 392 |  |  |  |
| Benign soft tissue proliferations - Vascular |  |  |  |  | 142 |
| Dermatofibroma |  |  |  | 104 |  |
| Pigmented benign keratosis |  |  | 478 |  |  |
| Solar or actinic keratosis | 129 |  |  |  |  |
| Squamous cell carcinoma, NOS | 192 |  |  |  |  |

### How the diagnosis was confirmed: ISIC against HAM10000

| ISIC (rows), HAM10000 (columns) | confocal | consensus | histo |
|---|---|---|---|
| confocal microscopy with consensus dermoscopy | 43 |  |  |
| histopathology |  |  | 1160 |
| single image expert consensus |  | 234 |  |

### Lesion numbers that only HAM10000 records

- Photos with a HAM10000 lesion number but no ISIC lesion number: 0
- When the splits are made, photos that either source puts in one lesion stay together.

### HAM10000's diagnosis groups (from its paper)

- `akiec`: actinic keratoses and intraepithelial carcinoma (Bowen's disease)
- `bcc`: basal cell carcinoma
- `bkl`: benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
- `df`: dermatofibroma
- `mel`: melanoma
- `nv`: melanocytic nevi
- `vasc`: vascular lesions (cherry angiomas, angiokeratomas, pyogenic granulomas, haemorrhage)

### Not done in this step

- No label is changed. The comparison is evidence for the naming rule.

<!-- step: labels -->
## Labels of the original dataset

Written by `python -m src.image.prepare labels`. Each unique photo (identical files count once) is labelled from its ISIC Archive record by the naming rule in `isic_to_class.csv`, never from its Kaggle folder. One row per photo is in `raw_photos.csv`. No image is changed and no split is made in this step.

- Photos: 2200; labelled: 2196; without a class: 4
- Classes: 8 (actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion)

### The naming rule

| ISIC level | ISIC wording | Class | Reason | ISIC numbers |
|---|---|---|---|---|
| diagnosis_3 | Solar or actinic keratosis | actinic keratosis |  | 131 |
| diagnosis_3 | Basal cell carcinoma | basal cell carcinoma |  | 392 |
| diagnosis_3 | Dermatofibroma | dermatofibroma |  | 111 |
| diagnosis_2 | Malignant melanocytic proliferations (Melanoma) | melanoma | in situ, invasive and not otherwise specified alike | 453 |
| diagnosis_3 | Nevus | nevus | all subtypes alike | 294 |
| diagnosis_3 | Pigmented benign keratosis | pigmented benign keratosis | HAM10000's benign keratosis group, which also holds seborrheic keratoses | 478 |
| diagnosis_3 | Squamous cell carcinoma, NOS | squamous cell carcinoma |  | 192 |
| diagnosis_3 | Squamous cell carcinoma in situ | squamous cell carcinoma |  | 5 |
| diagnosis_2 | Benign soft tissue proliferations - Vascular | vascular lesion |  | 142 |
| diagnosis_2 | Collision - Only benign proliferations | (no class) | collision: two lesions in one photo | 1 |
| diagnosis_2 | Collision - At least one malignant proliferation | (no class) | collision: two lesions in one photo | 3 |

### Photos per class

| Class | Photos | Of these, original test photos | Confirmed by biopsy | Also confirmed by HAM10000 |
|---|---|---|---|---|
| actinic keratosis | 131 | 16 | 131 | 129 |
| basal cell carcinoma | 392 | 16 | 392 | 392 |
| dermatofibroma | 111 | 16 | 55 | 104 |
| melanoma | 452 | 19 | 452 | 0 |
| nevus | 293 | 16 | 0 | 0 |
| pigmented benign keratosis | 478 | 16 | 332 | 478 |
| squamous cell carcinoma | 197 | 16 | 197 | 192 |
| vascular lesion | 142 | 3 | 67 | 142 |
| **Total** | 2196 | 118 | 1626 | 1437 |

### Kaggle folder against the new label (photos; a photo in two folders counts in both)

| Kaggle folder | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (no class) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 130 |  |  |  |  |  |  |  |  |
| basal cell carcinoma |  | 392 |  |  |  |  |  |  |  |
| dermatofibroma |  |  | 111 |  |  |  |  |  |  |
| melanoma |  |  |  | 450 |  |  |  |  | 3 |
| nevus | 78 |  |  |  | 293 |  |  |  | 1 |
| pigmented benign keratosis |  |  |  |  |  | 478 |  |  |  |
| seborrheic keratosis |  |  |  | 80 |  |  |  |  |  |
| squamous cell carcinoma |  |  |  |  |  |  | 197 |  |  |
| vascular lesion |  |  |  |  |  |  |  | 142 |  |

### Changes from the Kaggle folders

- differs from folder: 3; no class: 4; one of two folders: 155; same as folder: 2038
- Filed under actinic keratosis | nevus and labelled **actinic keratosis**: 77
- Filed under melanoma | seborrheic keratosis and labelled **melanoma**: 78

| Photo | Kaggle folder | ISIC diagnosis | New label | Status |
|---|---|---|---|---|
| ISIC_0000246 | nevus | Benign > Collision - Only benign proliferations | (no class) | collision: two lesions in one photo |
| ISIC_0010156 | melanoma | Malignant > Collision - At least one malignant proliferation | (no class) | collision: two lesions in one photo |
| ISIC_0010423 | melanoma | Malignant > Collision - At least one malignant proliferation | (no class) | collision: two lesions in one photo |
| ISIC_0010478 | melanoma | Malignant > Collision - At least one malignant proliferation | (no class) | collision: two lesions in one photo |
| ISIC_0011118 | seborrheic keratosis | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma in situ | melanoma | labelled |
| ISIC_0011126 | seborrheic keratosis | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive | melanoma | labelled |
| ISIC_0028619 | nevus | Indeterminate > Indeterminate epidermal proliferations > Solar or actinic keratosis | actinic keratosis | labelled |

### Not done in this step

- No image is changed and no split is made. Photos without a class stay in `raw_photos.csv`, marked, and go into no split.
- Near-identical photos (different files of the same picture) and photos of the same lesion are handled in the next steps.

<!-- step: duplicates -->
## Near-duplicates: different files of one picture, and second photos of one lesion

Written by `python -m src.image.prepare duplicates`. Every photo in `raw_photos.csv` is compared with every other one (identical files were already merged by the inventory). Flagged pairs are in `similar_photo_pairs.csv` (found by "picture comparison"). No image is changed.

- Photos compared: 2200; pairs: 2418900
- Flagged pairs: 210 (near-identical: 3; similar: 207)
- Flagged pairs whose two photos have different labels: 83
- "near-identical": shapes differ by at most 2.0%, hash distance at most 16, mean colour difference at most 5.0 (the identity check's "same picture"). "similar": at most 10.0%, 20 and 10.0.

### How well it finds what ISIC's lesion numbers already tell (calibration)

Photos with an ISIC lesion number: 1447. Pairs of them that show the same lesion: 447; pairs that show different lesions: 1045734.

| Pairs | Near-identical | Similar | Not flagged |
|---|---|---|---|
| Same lesion (should be flagged) | 1 | 31 | 415 |
| Different lesions (should not be flagged) | 0 | 147 | 1045587 |

- Share of same-lesion pairs found: 7.2%
- Mean colour difference of the same-lesion pairs (smallest, 25%, median, 75%, largest): 3.1, 16.3, 22.2, 28.4, 71.1

Second photos of one lesion (another angle, zoom or day) are different pictures: a comparison of whole pictures finds only those that look alike. The share above is how many it finds where the answer is known; for melanoma and nevus, which have no lesion numbers, the same share is the best estimate. This is a limitation to state.

### Groups that the flagged pairs would make

Pairs link into groups (a with b and b with c make one group). If the "similar" tier makes very large groups, it is linking unrelated lesions that merely look alike. A flagged pair decides nothing by itself: the eye check's verdicts in `eye_check_verdicts.csv` and ISIC's lesion numbers decide what is left out or kept together when the splits are made.

- near-identical: 3 groups holding 6 photos; sizes: 3 of 2
- near-identical or similar: 66 groups holding 246 photos; sizes: 48 of 2, 9 of 3, 4 of 4, 1 of 5, 1 of 6, 1 of 8, 1 of 17, 1 of 71

### Flagged pairs with different labels (worth an eye check: the same picture cannot have two diagnoses)

| Photo | Label | Photo | Label | Tier | Mean difference |
|---|---|---|---|---|---|
| ISIC_0024984 | basal cell carcinoma | ISIC_0025767 | pigmented benign keratosis | similar | 7.7 |
| ISIC_0027851 | basal cell carcinoma | ISIC_0031093 | vascular lesion | similar | 7.7 |
| ISIC_0026074 | basal cell carcinoma | ISIC_0026567 | pigmented benign keratosis | similar | 7.9 |
| ISIC_0025378 | pigmented benign keratosis | ISIC_0025793 | basal cell carcinoma | similar | 8.1 |
| ISIC_0026395 | basal cell carcinoma | ISIC_0030606 | vascular lesion | similar | 8.1 |
| ISIC_0028973 | basal cell carcinoma | ISIC_0029099 | vascular lesion | similar | 8.2 |
| ISIC_0000081 | nevus | ISIC_0024590 | basal cell carcinoma | similar | 8.5 |
| ISIC_0024883 | pigmented benign keratosis | ISIC_0027281 | basal cell carcinoma | similar | 8.5 |
| ISIC_0025793 | basal cell carcinoma | ISIC_0027905 | pigmented benign keratosis | similar | 8.5 |
| ISIC_0024495 | pigmented benign keratosis | ISIC_0031166 | basal cell carcinoma | similar | 8.6 |
| ISIC_0024883 | pigmented benign keratosis | ISIC_0029099 | vascular lesion | similar | 8.6 |
| ISIC_0024833 | basal cell carcinoma | ISIC_0026567 | pigmented benign keratosis | similar | 8.7 |
| ISIC_0025016 | pigmented benign keratosis | ISIC_0029889 | vascular lesion | similar | 8.7 |
| ISIC_0025064 | pigmented benign keratosis | ISIC_0028973 | basal cell carcinoma | similar | 8.7 |
| ISIC_0027037 | pigmented benign keratosis | ISIC_0028577 | basal cell carcinoma | similar | 8.7 |
| ISIC_0024590 | basal cell carcinoma | ISIC_0026567 | pigmented benign keratosis | similar | 8.8 |
| ISIC_0027210 | vascular lesion | ISIC_0028542 | basal cell carcinoma | similar | 8.8 |
| ISIC_0027856 | vascular lesion | ISIC_0028283 | basal cell carcinoma | similar | 8.8 |
| ISIC_0028928 | basal cell carcinoma | ISIC_0031659 | squamous cell carcinoma | similar | 8.8 |
| ISIC_0000081 | nevus | ISIC_0028765 | basal cell carcinoma | similar | 8.9 |
| ISIC_0024337 | pigmented benign keratosis | ISIC_0024375 | vascular lesion | similar | 9.0 |
| ISIC_0025199 | pigmented benign keratosis | ISIC_0027595 | basal cell carcinoma | similar | 9.0 |
| ISIC_0027969 | pigmented benign keratosis | ISIC_0028973 | basal cell carcinoma | similar | 9.0 |
| ISIC_0024669 | vascular lesion | ISIC_0028973 | basal cell carcinoma | similar | 9.1 |
| ISIC_0025064 | pigmented benign keratosis | ISIC_0029099 | vascular lesion | similar | 9.1 |
| ISIC_0025767 | pigmented benign keratosis | ISIC_0027210 | vascular lesion | similar | 9.1 |
| ISIC_0026843 | pigmented benign keratosis | ISIC_0029352 | basal cell carcinoma | similar | 9.1 |
| ISIC_0026365 | pigmented benign keratosis | ISIC_0028928 | basal cell carcinoma | similar | 9.2 |
| ISIC_0000096 | nevus | ISIC_0027275 | pigmented benign keratosis | similar | 9.3 |
| ISIC_0000388 | nevus | ISIC_0025154 | dermatofibroma | similar | 9.3 |
| ISIC_0024984 | basal cell carcinoma | ISIC_0027210 | vascular lesion | similar | 9.3 |
| ISIC_0025154 | dermatofibroma | ISIC_0031659 | squamous cell carcinoma | similar | 9.3 |
| ISIC_0028973 | basal cell carcinoma | ISIC_0029783 | dermatofibroma | similar | 9.3 |
| ISIC_0026443 | pigmented benign keratosis | ISIC_0028085 | basal cell carcinoma | similar | 9.4 |
| ISIC_0027601 | basal cell carcinoma | ISIC_0032715 | vascular lesion | similar | 9.4 |
| ISIC_0028308 | pigmented benign keratosis | ISIC_0030387 | squamous cell carcinoma | similar | 9.4 |
| ISIC_0030021 | dermatofibroma | ISIC_0033991 | vascular lesion | similar | 9.4 |
| ISIC_0000388 | nevus | ISIC_0030606 | vascular lesion | similar | 9.5 |
| ISIC_0026074 | basal cell carcinoma | ISIC_0028308 | pigmented benign keratosis | similar | 9.5 |
| ISIC_0026843 | pigmented benign keratosis | ISIC_0030730 | actinic keratosis | similar | 9.5 |
| ISIC_0027522 | vascular lesion | ISIC_0028283 | basal cell carcinoma | similar | 9.5 |
| ISIC_0000435 | nevus | ISIC_0001151 | melanoma | similar | 9.6 |
| ISIC_0024337 | pigmented benign keratosis | ISIC_0030283 | vascular lesion | similar | 9.6 |
| ISIC_0024931 | basal cell carcinoma | ISIC_0027202 | pigmented benign keratosis | similar | 9.6 |
| ISIC_0027385 | vascular lesion | ISIC_0028815 | basal cell carcinoma | similar | 9.6 |
| ISIC_0024883 | pigmented benign keratosis | ISIC_0027291 | basal cell carcinoma | similar | 9.7 |
| ISIC_0024984 | basal cell carcinoma | ISIC_0027470 | pigmented benign keratosis | similar | 9.7 |
| ISIC_0025154 | dermatofibroma | ISIC_0030606 | vascular lesion | similar | 9.7 |
| ISIC_0025831 | squamous cell carcinoma | ISIC_0027982 | pigmented benign keratosis | similar | 9.7 |
| ISIC_0026072 | pigmented benign keratosis | ISIC_0027601 | basal cell carcinoma | similar | 9.7 |
| ISIC_0026803 | pigmented benign keratosis | ISIC_0028714 | vascular lesion | similar | 9.7 |
| ISIC_0027037 | pigmented benign keratosis | ISIC_0030730 | actinic keratosis | similar | 9.7 |
| ISIC_0027394 | pigmented benign keratosis | ISIC_0028765 | basal cell carcinoma | similar | 9.7 |
| ISIC_0028095 | basal cell carcinoma | ISIC_0028473 | pigmented benign keratosis | similar | 9.7 |
| ISIC_0028095 | basal cell carcinoma | ISIC_0029371 | squamous cell carcinoma | similar | 9.7 |
| ISIC_0028404 | pigmented benign keratosis | ISIC_0031384 | basal cell carcinoma | similar | 9.7 |
| ISIC_0029352 | basal cell carcinoma | ISIC_0030730 | actinic keratosis | similar | 9.7 |
| ISIC_0000024 | nevus | ISIC_0027470 | pigmented benign keratosis | similar | 9.8 |
| ISIC_0024495 | pigmented benign keratosis | ISIC_0028542 | basal cell carcinoma | similar | 9.8 |
| ISIC_0024612 | pigmented benign keratosis | ISIC_0027210 | vascular lesion | similar | 9.8 |
| ISIC_0025046 | basal cell carcinoma | ISIC_0025799 | pigmented benign keratosis | similar | 9.8 |
| ISIC_0025322 | basal cell carcinoma | ISIC_0027151 | pigmented benign keratosis | similar | 9.8 |
| ISIC_0025911 | dermatofibroma | ISIC_0031169 | basal cell carcinoma | similar | 9.8 |
| ISIC_0027141 | dermatofibroma | ISIC_0031569 | basal cell carcinoma | similar | 9.8 |
| ISIC_0027167 | pigmented benign keratosis | ISIC_0028406 | vascular lesion | similar | 9.8 |
| ISIC_0000081 | nevus | ISIC_0027394 | pigmented benign keratosis | similar | 9.9 |
| ISIC_0000384 | nevus | ISIC_0025040 | pigmented benign keratosis | similar | 9.9 |
| ISIC_0010551 | melanoma | ISIC_0027957 | pigmented benign keratosis | similar | 9.9 |
| ISIC_0010834 | melanoma | ISIC_0025805 | pigmented benign keratosis | similar | 9.9 |
| ISIC_0025046 | basal cell carcinoma | ISIC_0032240 | vascular lesion | similar | 9.9 |
| ISIC_0025199 | pigmented benign keratosis | ISIC_0032240 | vascular lesion | similar | 9.9 |
| ISIC_0026567 | pigmented benign keratosis | ISIC_0028973 | basal cell carcinoma | similar | 9.9 |
| ISIC_0027385 | vascular lesion | ISIC_0031258 | basal cell carcinoma | similar | 9.9 |
| ISIC_0027739 | pigmented benign keratosis | ISIC_0031513 | basal cell carcinoma | similar | 9.9 |
| ISIC_0028473 | pigmented benign keratosis | ISIC_0028858 | basal cell carcinoma | similar | 9.9 |
| ISIC_0000081 | nevus | ISIC_0029099 | vascular lesion | similar | 10.0 |
| ISIC_0024612 | pigmented benign keratosis | ISIC_0028973 | basal cell carcinoma | similar | 10.0 |
| ISIC_0025419 | pigmented benign keratosis | ISIC_0028542 | basal cell carcinoma | similar | 10.0 |
| ISIC_0026443 | pigmented benign keratosis | ISIC_0030730 | actinic keratosis | similar | 10.0 |
| ISIC_0026567 | pigmented benign keratosis | ISIC_0029099 | vascular lesion | similar | 10.0 |
| ISIC_0026865 | basal cell carcinoma | ISIC_0027047 | squamous cell carcinoma | similar | 10.0 |
| ISIC_0027969 | pigmented benign keratosis | ISIC_0029099 | vascular lesion | similar | 10.0 |
| ISIC_0028652 | basal cell carcinoma | ISIC_0031672 | squamous cell carcinoma | similar | 10.0 |

### Not done in this step

- No image or label is changed. The eye check's verdicts go into `eye_check_verdicts.csv`.

<!-- step: lesions -->
## Second photos of one lesion (skin-pattern matching)

Written by `python -m src.image.prepare lesions`. Within each class, every photo is compared with every other one by its skin pattern (OpenCV 4.14.0: the 1000 strongest SIFT points of a grey version 600 pixels long, paired by Lowe's ratio test at 0.75, each point with one other at most, and RANSAC keeping the pairs that fit one shift, turn and zoom within 5.0 pixels; a zoom outside 0.25–4.0 is no match). "Inliers" are the points that fit: two photos of one lesion share many, two different lesions few. Pairs with at least 10 inliers are in `similar_photo_pairs.csv` (found by "skin pattern"). Copies left out by the eye check are not compared. No image or label is changed.

- Photos compared: 2179; pairs (within each class): 371988
- Per class: actinic keratosis: 131 photos, 8515 pairs; basal cell carcinoma: 392 photos, 76636 pairs; dermatofibroma: 111 photos, 6105 pairs; melanoma: 436 photos, 94830 pairs; nevus: 293 photos, 42778 pairs; pigmented benign keratosis: 478 photos, 114003 pairs; squamous cell carcinoma: 196 photos, 19110 pairs; vascular lesion: 142 photos, 10011 pairs

### How well it finds what ISIC's lesion numbers already tell (calibration)

Pairs that ISIC names as one lesion: 446; as different lesions (same class): 233456. The threshold is chosen where many same-lesion pairs are found and almost no different lesions are flagged.

| Inliers at least | Same-lesion pairs found | Different-lesion pairs flagged | Pairs without lesion numbers flagged |
|---|---|---|---|
| 10 | 222 of 446 (49.8%) | 1 of 233456 | 6 |
| 15 | 166 of 446 (37.2%) | 0 of 233456 | 3 |
| 20 | 130 of 446 (29.1%) | 0 of 233456 | 1 |
| 30 | 89 of 446 (20.0%) | 0 of 233456 | 1 |
| 50 | 43 of 446 (9.6%) | 0 of 233456 | 0 |
| 100 | 12 of 446 (2.7%) | 0 of 233456 | 0 |

- Inliers of the same-lesion pairs (smallest, 25%, median, 75%, largest): 0, 4, 9, 22, 407

### Pairs without lesion numbers flagged, per class

| Class | ≥ 10 | ≥ 15 | ≥ 20 | ≥ 30 | ≥ 50 | ≥ 100 |
|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 |
| basal cell carcinoma | 0 | 0 | 0 | 0 | 0 | 0 |
| dermatofibroma | 0 | 0 | 0 | 0 | 0 | 0 |
| melanoma | 6 | 3 | 1 | 1 | 0 | 0 |
| nevus | 0 | 0 | 0 | 0 | 0 | 0 |
| pigmented benign keratosis | 0 | 0 | 0 | 0 | 0 | 0 |
| squamous cell carcinoma | 0 | 0 | 0 | 0 | 0 | 0 |
| vascular lesion | 0 | 0 | 0 | 0 | 0 | 0 |

### Not done in this step

- No image or label is changed. A flagged pair decides nothing by itself: the eye check's verdicts go into `eye_check_verdicts.csv` ("same lesion, new photo" keeps both photos in one split).

<!-- step: splits -->
## Splits: train, validation and test

Written by `python -m src.image.prepare splits`. Every labelled photo goes into train, validation (val) or test, with every lesion in one split: photos that ISIC's or HAM10000's lesion numbers, or the eye check, name as one lesion form a group that moves as a whole (group-based). The Kaggle dataset's own Train and Test folders play no part: the splits are made anew, and the original test photos are split like every other photo (where they went is shown below). Each class gets exactly 40 test and 15 validation photos (stratified); the rest is train, the database that open-book runs retrieve from. The split starts from a random fill (seed 2026, fixed in advance; Python 3.12's random module) and then swaps lesion groups of equal size between the parts whenever that brings the test and validation parts closer to their class's own mix of subtype (ISIC diagnosis), how the diagnosis was confirmed, source (credit line) and original image size. Only these facts, known before any experiment, are used; never a model's result. The split column of `raw_photos.csv` holds the result; no image is changed.

### Photos per class and split

| Class | Test | Validation | Train | Photos | Original test photos (test / val / train) |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 15 | 76 | 131 | 5 / 1 / 10 |
| basal cell carcinoma | 40 | 15 | 337 | 392 | 3 / 0 / 13 |
| dermatofibroma | 40 | 15 | 56 | 111 | 5 / 4 / 7 |
| melanoma | 40 | 15 | 381 | 436 | 1 / 1 / 17 |
| nevus | 40 | 15 | 238 | 293 | 2 / 1 / 13 |
| pigmented benign keratosis | 40 | 15 | 423 | 478 | 0 / 1 / 15 |
| squamous cell carcinoma | 40 | 15 | 141 | 196 | 2 / 1 / 13 |
| vascular lesion | 40 | 15 | 87 | 142 | 1 / 0 / 2 |
| **Total** | 320 | 120 | 1739 | 2179 | 19 / 9 / 90 |

- Lesion groups: 1804, of which 309 hold two or more photos; groups found in two splits: 0
- In no split: 21 (no class: collision: two lesions in one photo: 4; same photo as another (eye check): 17)

### Balance (0 = the test and validation parts have their class's own mix)

- This split: 0.045 (the random fill it started from: 0.117)
- 1000 simple random splits under the same rules: median 0.100, 95th percentile 0.121, worst 0.138

| Class | Property | Value | Class | Test | Validation | Train |
|---|---|---|---|---|---|---|
| actinic keratosis | source | MILK study team | 98% | 98% | 100% | 99% |
| actinic keratosis | original size | 600x450 | 98% | 98% | 100% | 99% |
| dermatofibroma | confirmed by | single image expert consensus | 50% | 50% | 53% | 50% |
| dermatofibroma | confirmed by | histopathology | 50% | 50% | 47% | 50% |
| dermatofibroma | source | MILK study team | 94% | 95% | 93% | 93% |
| dermatofibroma | source | Anonymous | 6% | 5% | 7% | 7% |
| dermatofibroma | original size | 600x450 | 94% | 95% | 93% | 93% |
| melanoma | subtype | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma, NOS | 29% | 30% | 33% | 29% |
| melanoma | subtype | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma Invasive | 28% | 28% | 27% | 28% |
| melanoma | subtype | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma in situ | 25% | 25% | 27% | 25% |
| melanoma | subtype | Malignant > Malignant melanocytic proliferations (Melanoma) > Melanoma in situ > Melanoma in situ, Lentigo maligna type | 8% | 8% | 7% | 8% |
| melanoma | original size | 1024x768 | 63% | 65% | 67% | 63% |
| melanoma | original size | 3072x2304 | 10% | 10% | 13% | 10% |
| nevus | subtype | Benign > Benign melanocytic proliferations > Nevus | 51% | 52% | 53% | 51% |
| nevus | subtype | Benign > Benign melanocytic proliferations > Nevus > Nevus, NOS, Compound | 14% | 12% | 13% | 14% |
| nevus | subtype | Benign > Benign melanocytic proliferations > Nevus > Nevus, Atypical, Dysplastic, or Clark > Nevus, Dysplastic | 10% | 10% | 13% | 10% |
| nevus | subtype | Benign > Benign melanocytic proliferations > Nevus > Nevus, Atypical, Dysplastic, or Clark > Nevus, Clark | 9% | 8% | 7% | 9% |
| nevus | subtype | Benign > Benign melanocytic proliferations > Nevus > Nevus, Congenital | 9% | 8% | 7% | 9% |
| nevus | original size | 767x576 | 20% | 20% | 20% | 21% |
| nevus | original size | 919x802 | 15% | 15% | 20% | 15% |
| nevus | original size | 3072x2304 | 13% | 15% | 13% | 13% |
| nevus | original size | 1504x1129 | 12% | 12% | 13% | 11% |
| nevus | original size | 1024x768 | 8% | 8% | 7% | 8% |
| nevus | original size | 2048x1536 | 5% | 5% | 7% | 5% |
| nevus | original size | 1022x767 | 5% | 5% | 7% | 5% |
| pigmented benign keratosis | confirmed by | histopathology | 69% | 70% | 53% | 70% |
| pigmented benign keratosis | confirmed by | single image expert consensus | 22% | 20% | 20% | 22% |
| pigmented benign keratosis | confirmed by | confocal microscopy with consensus dermoscopy | 9% | 10% | 27% | 8% |
| squamous cell carcinoma | subtype | Malignant > Malignant epidermal proliferations > Squamous cell carcinoma, NOS | 97% | 98% | 100% | 97% |
| squamous cell carcinoma | source | MILK study team | 97% | 98% | 100% | 97% |
| squamous cell carcinoma | original size | 600x450 | 97% | 98% | 100% | 97% |
| vascular lesion | confirmed by | single image expert consensus | 53% | 52% | 53% | 53% |
| vascular lesion | confirmed by | histopathology | 47% | 48% | 47% | 47% |

Values below 5% of a class are not listed (they are balanced too).

### Not done in this step

- No image is changed: the images of each split are made in the next step.

<!-- step: images -->
## Images: the prepared dataset

Written by `python -m src.image.prepare images`. Every photo in a split is made from its original file in `data/image/raw` (checked against its fingerprint in `raw_photos.csv`; nothing in `data/image/raw` is changed) and saved as `data/image/dataset/<split>/<class>/<photo_id>.jpg`. `dataset.csv` lists every image: its split, class, file, SHA-256, lesion group, whether it is an original test photo, its size and what was done to it. The evaluation checks every image against that table before a run, so every model gets byte-identical images.

Every photo is prepared the same way, in this order:

1. Upright as a viewer shows it (its EXIF orientation tag, where it has one).
2. Colours in sRGB: a photo with an embedded colour profile is converted from that profile; a photo without one is already taken as sRGB, which is what a missing profile means.
3. Landscape: a portrait photo is turned a quarter turn to the left. A dermoscopic photo has no up or down, so nothing is lost.
4. Fitted whole within 600 x 450 pixels (HAM10000's own size, which most photos already have), keeping its shape: shrunk with Lanczos filtering, never enlarged. Nothing is cut and nothing is added (no crop, no border). Where the fitted size is within 1 pixel of 600 x 450 (a photo whose shape is 4:3 up to rounding, such as 767 x 576), it is made exactly 600 x 450: a stretch of less than one pixel.
5. Saved as JPEG at quality 95 with full colour resolution (4:4:4 chroma), without metadata (no EXIF, no colour profile, no thumbnail).

Made with Pillow 13.0.0 (libjpeg-turbo 8.0, Little CMS 2.19).

### Images per class and split

| Class | Test | Validation | Train | Images |
|---|---|---|---|---|
| actinic keratosis | 40 | 15 | 76 | 131 |
| basal cell carcinoma | 40 | 15 | 337 | 392 |
| dermatofibroma | 40 | 15 | 56 | 111 |
| melanoma | 40 | 15 | 381 | 436 |
| nevus | 40 | 15 | 238 | 293 |
| pigmented benign keratosis | 40 | 15 | 423 | 478 |
| squamous cell carcinoma | 40 | 15 | 141 | 196 |
| vascular lesion | 40 | 15 | 87 | 142 |
| **Total** | 320 | 120 | 1739 | 2179 |

### Sizes

- Exactly 600 x 450: 2013 of 2179 images (92.4%)
- Another size: 166. A photo that is not 4:3 keeps its own shape, because nothing is cut or added; so the shape of these images still differs from the others.

| Class | Images | Exactly 600 x 450 | Another size | The most common other sizes (images) |
|---|---|---|---|---|
| actinic keratosis | 131 | 131 | 0 | – |
| basal cell carcinoma | 392 | 392 | 0 | – |
| dermatofibroma | 111 | 104 | 7 | 600x398 (2), 450x450 (1), 600x399 (1) |
| melanoma | 436 | 378 | 58 | 516x450 (18), 600x402 (11), 600x399 (3) |
| nevus | 293 | 197 | 96 | 516x450 (57), 600x420 (4), 600x422 (4) |
| pigmented benign keratosis | 478 | 478 | 0 | – |
| squamous cell carcinoma | 196 | 191 | 5 | 600x398 (2), 600x396 (1), 600x397 (1) |
| vascular lesion | 142 | 142 | 0 | – |

### What was done besides saving again

- Shrunk: 743 (of which made exactly 600 x 450 by a stretch of less than one pixel: 132)
- Already 600 x 450 or smaller (saved again only, never enlarged): 1436
- Turned upright by an EXIF orientation tag: 0
- Turned from portrait to landscape: 8 (ISIC_0000176, ISIC_0000238, ISIC_0000241, ISIC_0000243, ISIC_0000262, ISIC_0000273, ISIC_0000398, ISIC_0000441)
- Colours converted from an embedded colour profile: 2 (ISIC_0000143: Camera RGB Profile; ISIC_0000153: Camera RGB Profile)
- Converted to RGB from another colour mode: 0

### Original test photos

`kaggle_test` marks the 118 photos from the Kaggle dataset's Test folder: test 19, val 9, train 90. They can be run on their own as a closed-book check against results on the original test folder; not open-book, because most of them are in train, the database that open-book runs retrieve from.

### Fingerprints

One hash per split over every image's split, class, file, SHA-256 and lesion group (it changes if any of them changes). A run's run card shows the same hash as `images_sha256` when the run answered the whole split.

- test: `cfa2c9e9b7c6c8d5c5ac6a399bc8b6dff033814f776f3a2abba9da408677fb15`
- val: `0347a83b28f7d45d8db41cf2b1be8a0e40e0881adce3b3e9fe7b42d199c29cc9`
- train: `9e435c5f7dae6a566db8c6bfa1848454daae6556a380903cbacd6eb87e04a20a`

### Not done in this step

- No crop, border, colour correction, hair removal or sharpening: every lesion is shown whole, as photographed.
- Photos in no split (21) get no image.

<!-- step: shortcuts -->
## Shortcut check: what the images reveal besides the lesion

Written by `python -m src.image.prepare shortcuts` from the prepared images; it changes nothing. A model should recognise the lesion, not where or how the photo was taken. This step measures how much simple properties of the images, which are mostly about how the photo was taken rather than about the lesion (file size and sharpness also depend a little on the lesion's own texture), reveal each image's class: where they do, a model, or the image search of an open-book run, could get the class partly right without looking at the lesion. It shows how much these clues could tell, not whether a model uses them.

### Where the photos come from (ISIC credit line)

| Class | Images | Anonymous | MILK study team |
|---|---|---|---|
| actinic keratosis | 131 | 2 (2%) | 129 (98%) |
| basal cell carcinoma | 392 | 0 (0%) | 392 (100%) |
| dermatofibroma | 111 | 7 (6%) | 104 (94%) |
| melanoma | 436 | 436 (100%) | 0 (0%) |
| nevus | 293 | 293 (100%) | 0 (0%) |
| pigmented benign keratosis | 478 | 0 (0%) | 478 (100%) |
| squamous cell carcinoma | 196 | 5 (3%) | 191 (97%) |
| vascular lesion | 142 | 0 (0%) | 142 (100%) |

Where a class comes from one source only, the source itself is a clue to the class: any difference in camera, lighting or file handling between sources can stand in for the diagnosis. Between classes from the same source it tells nothing.

### The properties per class (median over all splits)

| Class | Not 600 x 450 | File size (kB) | Dark corners (0–255) | Edge brightness (0–255) | Sharpness |
|---|---|---|---|---|---|
| actinic keratosis | 0 (0%) | 97 | 149 | 164 | 2.2 |
| basal cell carcinoma | 0 (0%) | 88 | 158 | 173 | 1.8 |
| dermatofibroma | 7 (6%) | 101 | 162 | 175 | 2.5 |
| melanoma | 58 (13%) | 90 | 144 | 160 | 2.1 |
| nevus | 96 (33%) | 85 | 155 | 168 | 2.0 |
| pigmented benign keratosis | 0 (0%) | 98 | 139 | 156 | 2.3 |
| squamous cell carcinoma | 5 (3%) | 96 | 154 | 170 | 2.1 |
| vascular lesion | 0 (0%) | 90 | 165 | 176 | 2.0 |

Dark corners: the mean brightness of the four corner squares (a tenth of the shorter side); edge brightness: the outer frame of the same width; sharpness: the mean brightness step between neighbouring pixels.

### How much the properties reveal

For each property, a nearest-neighbour vote predicts each test image's class, and its source, from the training images most alike in that property alone. Each training class counts equally, as in the test split; the number of neighbours (from 1, 5, 15, 45) is chosen on the validation split. Chance is 12.5% for the 8 classes and 50% for the 2 sources; the 95% intervals resample whole lesions, as in the evaluation.

| Property | Class: balanced accuracy on test (95% interval) | Neighbours | Source: balanced accuracy on test (95% interval) | Neighbours |
|---|---|---|---|---|
| image size | 16.6% (14.7–18.5%) | 1 | 60.7% (56.5–64.9%) | 1 |
| file size | 15.6% (11.8–19.3%) | 1 | 55.7% (49.6–61.9%) | 45 |
| dark corners | 13.8% (9.9–17.8%) | 1 | 51.2% (45.3–57.2%) | 5 |
| edge colour | 34.7% (30.0–39.6%) | 15 | 83.8% (79.1–88.1%) | 5 |
| sharpness | 15.0% (11.3–18.9%) | 15 | 52.7% (46.7–58.8%) | 1 |
| all five together | 36.3% (31.1–41.5%) | 5 | 85.1% (80.1–89.8%) | 45 |

Recall per class on test, all five properties together:

| Class | Recall |
|---|---|
| actinic keratosis | 40% |
| basal cell carcinoma | 18% |
| dermatofibroma | 28% |
| melanoma | 40% |
| nevus | 68% |
| pigmented benign keratosis | 32% |
| squamous cell carcinoma | 25% |
| vascular lesion | 40% |

### Not done in this step

- No image, label or split is changed: it measures only (320 test images, with the training images as the ones they are compared with).
