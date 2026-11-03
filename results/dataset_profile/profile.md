# Dataset profile

Root: `data` | splits: train = `Train`, test = `Test`
Images: 2357 | files that are not readable images: 0
Hash of all images (split, label, path, SHA-256): `354eaf4716120962fdde0c30f58ad988a6d2bb2bc8d8a0362e5dfcb9f11cf79b`

## Images per class and split

| Class | train | test |
|---|---|---|
| actinic keratosis | 114 | 16 |
| basal cell carcinoma | 376 | 16 |
| dermatofibroma | 95 | 16 |
| melanoma | 438 | 16 |
| nevus | 357 | 16 |
| pigmented benign keratosis | 462 | 16 |
| seborrheic keratosis | 77 | 3 |
| squamous cell carcinoma | 181 | 16 |
| vascular lesion | 139 | 3 |
| **Total** | 2239 | 118 |

## Image sizes per class (width x height: count)

| Split | Class | Sizes |
|---|---|---|
| test | actinic keratosis | 600x450: 14, 1024x768: 2 |
| test | basal cell carcinoma | 600x450: 16 |
| test | dermatofibroma | 600x450: 9, 1936x1936: 1, 4288x2848: 1, 6668x4479: 1, 6648x4459: 1, 6601x4380: 1, 6648x4439: 1, 6688x4439: 1 |
| test | melanoma | 1504x1129: 12, 1022x767: 3, 1503x1129: 1 |
| test | nevus | 1022x767: 14, 1504x1129: 2 |
| test | pigmented benign keratosis | 600x450: 16 |
| test | seborrheic keratosis | 1024x768: 3 |
| test | squamous cell carcinoma | 600x450: 11, 6668x4419: 2, 6668x4399: 1, 6641x4421: 1, 6641x4391: 1 |
| test | vascular lesion | 600x450: 3 |
| train | actinic keratosis | 600x450: 114 |
| train | basal cell carcinoma | 600x450: 376 |
| train | dermatofibroma | 600x450: 95 |
| train | melanoma | 1024x768: 291, 3072x2304: 45, 767x576: 20, 919x802: 14, 2048x1536: 13, 3872x2592: 9, 824x719: 5, 722x542: 4, 2592x1944: 3, 2816x2112: 2, 962x674: 2, 3008x2000: 2, 963x629: 1, 965x724: 1, 722x540: 1, 964x640: 1, 965x649: 1, 959x638: 1, 961x649: 1, 965x624: 1, 962x645: 1, 964x720: 1, 722x545: 1, 960x724: 1, 966x638: 1, 944x1127: 1, 1242x1284: 1, 2313x1985: 1, 2269x1965: 1, 2597x2185: 1, 2126x2185: 1, 2304x1536: 1, 1016x682: 1, 771x750: 1, 838x614: 1, 679x566: 1, 833x680: 1, 2198x1603: 1, 1936x1936: 1, 1936x1296: 1 |
| train | nevus | 600x450: 76, 767x576: 60, 919x802: 46, 3072x2304: 39, 1504x1129: 32, 1024x768: 25, 2048x1536: 16, 824x719: 12, 576x768: 5, 962x674: 4, 962x722: 3, 1019x717: 3, 2592x1944: 2, 1028x753: 2, 1022x767: 1, 962x671: 1, 966x645: 1, 959x634: 1, 957x640: 1, 957x723: 1, 964x672: 1, 959x723: 1, 959x639: 1, 962x641: 1, 962x723: 1, 960x645: 1, 962x676: 1, 724x542: 1, 2725x2082: 1, 2530x2122: 1, 2155x1958: 1, 2101x1943: 1, 2314x2067: 1, 2117x1988: 1, 2583x2096: 1, 2565x2000: 1, 2601x1909: 1, 2012x1956: 1, 2283x1824: 1, 3008x2000: 1, 3872x2592: 1, 1007x711: 1, 1011x713: 1, 1008x711: 1, 796x682: 1, 853x680: 1 |
| train | pigmented benign keratosis | 600x450: 462 |
| train | seborrheic keratosis | 1024x768: 77 |
| train | squamous cell carcinoma | 600x450: 181 |
| train | vascular lesion | 600x450: 139 |

## Shortcut check: does image size alone give the class away?

Guessing the most common class for each image size is right for 1103 of 2357 images; always guessing the largest class is right for 478. A large gap means size is a shortcut.

## Exact duplicate files: 157 groups

- train/actinic keratosis/ISIC_0025780.jpg, train/nevus/ISIC_0025780.jpg
- train/actinic keratosis/ISIC_0025803.jpg, train/nevus/ISIC_0025803.jpg
- train/actinic keratosis/ISIC_0025825.jpg, train/nevus/ISIC_0025825.jpg
- train/actinic keratosis/ISIC_0025953.jpg, train/nevus/ISIC_0025953.jpg
- train/actinic keratosis/ISIC_0025957.jpg, train/nevus/ISIC_0025957.jpg
- train/actinic keratosis/ISIC_0025992.jpg, train/nevus/ISIC_0025992.jpg
- train/actinic keratosis/ISIC_0026040.jpg, train/nevus/ISIC_0026040.jpg
- train/actinic keratosis/ISIC_0026149.jpg, train/nevus/ISIC_0026149.jpg
- train/actinic keratosis/ISIC_0026171.jpg, train/nevus/ISIC_0026171.jpg
- train/actinic keratosis/ISIC_0026194.jpg, train/nevus/ISIC_0026194.jpg
- train/actinic keratosis/ISIC_0026212.jpg, train/nevus/ISIC_0026212.jpg
- train/actinic keratosis/ISIC_0026457.jpg, train/nevus/ISIC_0026457.jpg
- train/actinic keratosis/ISIC_0026468.jpg, train/nevus/ISIC_0026468.jpg
- train/actinic keratosis/ISIC_0026525.jpg, train/nevus/ISIC_0026525.jpg
- train/actinic keratosis/ISIC_0026575.jpg, train/nevus/ISIC_0026575.jpg
- train/actinic keratosis/ISIC_0026625.jpg, train/nevus/ISIC_0026625.jpg
- train/actinic keratosis/ISIC_0026626.jpg, train/nevus/ISIC_0026626.jpg
- train/actinic keratosis/ISIC_0026650.jpg, train/nevus/ISIC_0026650.jpg
- train/actinic keratosis/ISIC_0026702.jpg, train/nevus/ISIC_0026702.jpg
- train/actinic keratosis/ISIC_0026709.jpg, train/nevus/ISIC_0026709.jpg
- train/actinic keratosis/ISIC_0026729.jpg, train/nevus/ISIC_0026729.jpg
- train/actinic keratosis/ISIC_0026765.jpg, train/nevus/ISIC_0026765.jpg
- train/actinic keratosis/ISIC_0026848.jpg, train/nevus/ISIC_0026848.jpg
- train/actinic keratosis/ISIC_0026857.jpg, train/nevus/ISIC_0026857.jpg
- train/actinic keratosis/ISIC_0026905.jpg, train/nevus/ISIC_0026905.jpg
- train/actinic keratosis/ISIC_0026984.jpg, train/nevus/ISIC_0026984.jpg
- train/actinic keratosis/ISIC_0027172.jpg, train/nevus/ISIC_0027172.jpg
- train/actinic keratosis/ISIC_0027254.jpg, train/nevus/ISIC_0027254.jpg
- train/actinic keratosis/ISIC_0027334.jpg, train/nevus/ISIC_0027334.jpg
- train/actinic keratosis/ISIC_0027447.jpg, train/nevus/ISIC_0027447.jpg
- train/actinic keratosis/ISIC_0027452.jpg, train/nevus/ISIC_0027452.jpg
- train/actinic keratosis/ISIC_0027536.jpg, train/nevus/ISIC_0027536.jpg
- train/actinic keratosis/ISIC_0027562.jpg, train/nevus/ISIC_0027562.jpg
- train/actinic keratosis/ISIC_0027580.jpg, train/nevus/ISIC_0027580.jpg
- train/actinic keratosis/ISIC_0027615.jpg, train/nevus/ISIC_0027615.jpg
- train/actinic keratosis/ISIC_0027650.jpg, train/nevus/ISIC_0027650.jpg
- train/actinic keratosis/ISIC_0027668.jpg, train/nevus/ISIC_0027668.jpg
- train/actinic keratosis/ISIC_0027802.jpg, train/nevus/ISIC_0027802.jpg
- train/actinic keratosis/ISIC_0027829.jpg, train/nevus/ISIC_0027829.jpg
- train/actinic keratosis/ISIC_0027884.jpg, train/nevus/ISIC_0027884.jpg
- train/actinic keratosis/ISIC_0027896.jpg, train/nevus/ISIC_0027896.jpg
- train/actinic keratosis/ISIC_0027950.jpg, train/nevus/ISIC_0027950.jpg
- train/actinic keratosis/ISIC_0027958.jpg, train/nevus/ISIC_0027958.jpg
- train/actinic keratosis/ISIC_0028063.jpg, train/nevus/ISIC_0028063.jpg
- train/actinic keratosis/ISIC_0028076.jpg, train/nevus/ISIC_0028076.jpg
- train/actinic keratosis/ISIC_0028190.jpg, train/nevus/ISIC_0028190.jpg
- train/actinic keratosis/ISIC_0028314.jpg, train/nevus/ISIC_0028314.jpg
- train/actinic keratosis/ISIC_0028370.jpg, train/nevus/ISIC_0028370.jpg
- train/actinic keratosis/ISIC_0028393.jpg, train/nevus/ISIC_0028393.jpg
- train/actinic keratosis/ISIC_0028517.jpg, train/nevus/ISIC_0028517.jpg
- train/actinic keratosis/ISIC_0028558.jpg, train/nevus/ISIC_0028558.jpg
- train/actinic keratosis/ISIC_0028820.jpg, train/nevus/ISIC_0028820.jpg
- train/actinic keratosis/ISIC_0028854.jpg, train/nevus/ISIC_0028854.jpg
- train/actinic keratosis/ISIC_0028941.jpg, train/nevus/ISIC_0028941.jpg
- train/actinic keratosis/ISIC_0028990.jpg, train/nevus/ISIC_0028990.jpg
- train/actinic keratosis/ISIC_0029025.jpg, train/nevus/ISIC_0029025.jpg
- train/actinic keratosis/ISIC_0029041.jpg, train/nevus/ISIC_0029041.jpg
- train/actinic keratosis/ISIC_0029133.jpg, train/nevus/ISIC_0029133.jpg
- train/actinic keratosis/ISIC_0029141.jpg, train/nevus/ISIC_0029141.jpg
- train/actinic keratosis/ISIC_0029210.jpg, train/nevus/ISIC_0029210.jpg
- train/actinic keratosis/ISIC_0029309.jpg, train/nevus/ISIC_0029309.jpg
- train/actinic keratosis/ISIC_0029460.jpg, train/nevus/ISIC_0029460.jpg
- train/melanoma/ISIC_0000279.jpg, train/melanoma/ISIC_0000280.jpg
- train/melanoma/ISIC_0010803.jpg, test/seborrheic keratosis/ISIC_0010803.jpg (ACROSS SPLITS)
- train/melanoma/ISIC_0010808.jpg, test/seborrheic keratosis/ISIC_0010808.jpg (ACROSS SPLITS)
- train/melanoma/ISIC_0010809.jpg, test/seborrheic keratosis/ISIC_0010809.jpg (ACROSS SPLITS)
- train/melanoma/ISIC_0010810.jpg, train/seborrheic keratosis/ISIC_0010810.jpg
- train/melanoma/ISIC_0010812.jpg, train/seborrheic keratosis/ISIC_0010812.jpg
- train/melanoma/ISIC_0010834.jpg, train/seborrheic keratosis/ISIC_0010834.jpg
- train/melanoma/ISIC_0010852.jpg, train/seborrheic keratosis/ISIC_0010852.jpg
- train/melanoma/ISIC_0010860.jpg, train/seborrheic keratosis/ISIC_0010860.jpg
- train/melanoma/ISIC_0010862.jpg, train/seborrheic keratosis/ISIC_0010862.jpg
- train/melanoma/ISIC_0010863.jpg, train/seborrheic keratosis/ISIC_0010863.jpg
- train/melanoma/ISIC_0010866.jpg, train/seborrheic keratosis/ISIC_0010866.jpg
- train/melanoma/ISIC_0010872.jpg, train/seborrheic keratosis/ISIC_0010872.jpg
- train/melanoma/ISIC_0010876.jpg, train/seborrheic keratosis/ISIC_0010876.jpg
- train/melanoma/ISIC_0010878.jpg, train/seborrheic keratosis/ISIC_0010878.jpg
- train/melanoma/ISIC_0010883.jpg, train/seborrheic keratosis/ISIC_0010883.jpg
- train/melanoma/ISIC_0010890.jpg, train/seborrheic keratosis/ISIC_0010890.jpg
- train/melanoma/ISIC_0010898.jpg, train/seborrheic keratosis/ISIC_0010898.jpg
- train/melanoma/ISIC_0010900.jpg, train/seborrheic keratosis/ISIC_0010900.jpg
- train/melanoma/ISIC_0010909.jpg, train/seborrheic keratosis/ISIC_0010909.jpg
- train/melanoma/ISIC_0010912.jpg, train/seborrheic keratosis/ISIC_0010912.jpg
- train/melanoma/ISIC_0010917.jpg, train/seborrheic keratosis/ISIC_0010917.jpg
- train/melanoma/ISIC_0010923.jpg, train/seborrheic keratosis/ISIC_0010923.jpg
- train/melanoma/ISIC_0010925.jpg, train/seborrheic keratosis/ISIC_0010925.jpg
- train/melanoma/ISIC_0010934.jpg, train/seborrheic keratosis/ISIC_0010934.jpg
- train/melanoma/ISIC_0010938.jpg, train/seborrheic keratosis/ISIC_0010938.jpg
- train/melanoma/ISIC_0010942.jpg, train/seborrheic keratosis/ISIC_0010942.jpg
- train/melanoma/ISIC_0010965.jpg, train/seborrheic keratosis/ISIC_0010965.jpg
- train/melanoma/ISIC_0010971.jpg, train/seborrheic keratosis/ISIC_0010971.jpg
- train/melanoma/ISIC_0010977.jpg, train/seborrheic keratosis/ISIC_0010977.jpg
- train/melanoma/ISIC_0010978.jpg, train/seborrheic keratosis/ISIC_0010978.jpg
- train/melanoma/ISIC_0010979.jpg, train/seborrheic keratosis/ISIC_0010979.jpg
- train/melanoma/ISIC_0010982.jpg, train/seborrheic keratosis/ISIC_0010982.jpg
- train/melanoma/ISIC_0010990.jpg, train/seborrheic keratosis/ISIC_0010990.jpg
- train/melanoma/ISIC_0010993.jpg, train/seborrheic keratosis/ISIC_0010993.jpg
- train/melanoma/ISIC_0010995.jpg, train/seborrheic keratosis/ISIC_0010995.jpg
- train/melanoma/ISIC_0011007.jpg, train/seborrheic keratosis/ISIC_0011007.jpg
- train/melanoma/ISIC_0011012.jpg, train/seborrheic keratosis/ISIC_0011012.jpg
- train/melanoma/ISIC_0011013.jpg, train/seborrheic keratosis/ISIC_0011013.jpg
- train/melanoma/ISIC_0011030.jpg, train/seborrheic keratosis/ISIC_0011030.jpg
- train/melanoma/ISIC_0011031.jpg, train/seborrheic keratosis/ISIC_0011031.jpg
- train/melanoma/ISIC_0011032.jpg, train/seborrheic keratosis/ISIC_0011032.jpg
- train/melanoma/ISIC_0011039.jpg, train/seborrheic keratosis/ISIC_0011039.jpg
- train/melanoma/ISIC_0011040.jpg, train/seborrheic keratosis/ISIC_0011040.jpg
- train/melanoma/ISIC_0011042.jpg, train/seborrheic keratosis/ISIC_0011042.jpg
- train/melanoma/ISIC_0011043.jpg, train/seborrheic keratosis/ISIC_0011043.jpg
- train/melanoma/ISIC_0011045.jpg, train/seborrheic keratosis/ISIC_0011045.jpg
- train/melanoma/ISIC_0011052.jpg, train/seborrheic keratosis/ISIC_0011052.jpg
- train/melanoma/ISIC_0011056.jpg, train/seborrheic keratosis/ISIC_0011056.jpg
- train/melanoma/ISIC_0011057.jpg, train/seborrheic keratosis/ISIC_0011057.jpg
- train/melanoma/ISIC_0011077.jpg, train/seborrheic keratosis/ISIC_0011077.jpg
- train/melanoma/ISIC_0011094.jpg, train/seborrheic keratosis/ISIC_0011094.jpg
- train/melanoma/ISIC_0011095.jpg, train/seborrheic keratosis/ISIC_0011095.jpg
- train/melanoma/ISIC_0011099.jpg, train/seborrheic keratosis/ISIC_0011099.jpg
- train/melanoma/ISIC_0011112.jpg, train/seborrheic keratosis/ISIC_0011112.jpg
- train/melanoma/ISIC_0011121.jpg, train/seborrheic keratosis/ISIC_0011121.jpg
- train/melanoma/ISIC_0011125.jpg, train/seborrheic keratosis/ISIC_0011125.jpg
- train/melanoma/ISIC_0011128.jpg, train/seborrheic keratosis/ISIC_0011128.jpg
- train/melanoma/ISIC_0011130.jpg, train/seborrheic keratosis/ISIC_0011130.jpg
- train/melanoma/ISIC_0011132.jpg, train/seborrheic keratosis/ISIC_0011132.jpg
- train/melanoma/ISIC_0011135.jpg, train/seborrheic keratosis/ISIC_0011135.jpg
- train/melanoma/ISIC_0011136.jpg, train/seborrheic keratosis/ISIC_0011136.jpg
- train/melanoma/ISIC_0011137.jpg, train/seborrheic keratosis/ISIC_0011137.jpg
- train/melanoma/ISIC_0011138.jpg, train/seborrheic keratosis/ISIC_0011138.jpg
- train/melanoma/ISIC_0011139.jpg, train/seborrheic keratosis/ISIC_0011139.jpg
- train/melanoma/ISIC_0011140.jpg, train/seborrheic keratosis/ISIC_0011140.jpg
- train/melanoma/ISIC_0011151.jpg, train/seborrheic keratosis/ISIC_0011151.jpg
- train/melanoma/ISIC_0011158.jpg, train/seborrheic keratosis/ISIC_0011158.jpg
- train/melanoma/ISIC_0011161.jpg, train/seborrheic keratosis/ISIC_0011161.jpg
- train/melanoma/ISIC_0011163.jpg, train/seborrheic keratosis/ISIC_0011163.jpg
- train/melanoma/ISIC_0011166.jpg, train/seborrheic keratosis/ISIC_0011166.jpg
- train/melanoma/ISIC_0011169.jpg, train/seborrheic keratosis/ISIC_0011169.jpg
- train/melanoma/ISIC_0011174.jpg, train/seborrheic keratosis/ISIC_0011174.jpg
- train/melanoma/ISIC_0011176.jpg, train/seborrheic keratosis/ISIC_0011176.jpg
- train/melanoma/ISIC_0011178.jpg, train/seborrheic keratosis/ISIC_0011178.jpg
- train/melanoma/ISIC_0011181.jpg, train/seborrheic keratosis/ISIC_0011181.jpg
- train/melanoma/ISIC_0011185.jpg, train/seborrheic keratosis/ISIC_0011185.jpg
- train/melanoma/ISIC_0011187.jpg, train/seborrheic keratosis/ISIC_0011187.jpg
- train/melanoma/ISIC_0011193.jpg, train/seborrheic keratosis/ISIC_0011193.jpg
- train/nevus/ISIC_0000477.jpg, train/nevus/ISIC_0000512.jpg
- train/nevus/ISIC_0010512.jpg, test/actinic keratosis/ISIC_0010512.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0010889.jpg, test/actinic keratosis/ISIC_0010889.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024468.jpg, test/actinic keratosis/ISIC_0024468.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024470.jpg, test/actinic keratosis/ISIC_0024470.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024511.jpg, test/actinic keratosis/ISIC_0024511.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024646.jpg, test/actinic keratosis/ISIC_0024646.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024707.jpg, test/actinic keratosis/ISIC_0024707.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024763.jpg, test/actinic keratosis/ISIC_0024763.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024771.jpg, test/actinic keratosis/ISIC_0024771.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024800.jpg, test/actinic keratosis/ISIC_0024800.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024913.jpg, test/actinic keratosis/ISIC_0024913.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0024948.jpg, test/actinic keratosis/ISIC_0024948.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0025368.jpg, test/actinic keratosis/ISIC_0025368.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0025427.jpg, test/actinic keratosis/ISIC_0025427.jpg (ACROSS SPLITS)
- train/nevus/ISIC_0025605.jpg, test/actinic keratosis/ISIC_0025605.jpg (ACROSS SPLITS)

## Near-duplicate candidates (difference hash distance <= 40 of 256): 942 pairs

Candidates only: plain, low-contrast images can look similar to this check. Confirm by eye.

- 16: train/squamous cell carcinoma/ISIC_0030387.jpg ~ test/squamous cell carcinoma/ISIC_0024522.jpg
- 17: train/pigmented benign keratosis/ISIC_0025378.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 20: train/basal cell carcinoma/ISIC_0025691.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 21: train/dermatofibroma/ISIC_0034135.jpg ~ test/seborrheic keratosis/ISIC_0010803.jpg
- 24: train/pigmented benign keratosis/ISIC_0027151.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 26: train/pigmented benign keratosis/ISIC_0028209.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 26: train/vascular lesion/ISIC_0025578.jpg ~ test/seborrheic keratosis/ISIC_0010803.jpg
- 27: train/nevus/ISIC_0000449.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 27: train/pigmented benign keratosis/ISIC_0027737.jpg ~ test/pigmented benign keratosis/ISIC_0024408.jpg
- 27: train/vascular lesion/ISIC_0029889.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 27: train/vascular lesion/ISIC_0031270.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 28: train/melanoma/ISIC_0010777.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 28: train/pigmented benign keratosis/ISIC_0027151.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 29: train/pigmented benign keratosis/ISIC_0025378.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 29: train/pigmented benign keratosis/ISIC_0026477.jpg ~ test/seborrheic keratosis/ISIC_0010803.jpg
- 29: train/pigmented benign keratosis/ISIC_0027811.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 29: train/pigmented benign keratosis/ISIC_0028607.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 30: train/basal cell carcinoma/ISIC_0025691.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 30: train/melanoma/ISIC_0010350.jpg ~ test/basal cell carcinoma/ISIC_0024436.jpg
- 30: train/nevus/ISIC_0000388.jpg ~ test/seborrheic keratosis/ISIC_0010803.jpg
- 31: train/melanoma/ISIC_0009955.jpg ~ test/seborrheic keratosis/ISIC_0010803.jpg
- 31: train/nevus/ISIC_0000449.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 31: train/pigmented benign keratosis/ISIC_0027820.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 31: train/vascular lesion/ISIC_0027210.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 31: train/vascular lesion/ISIC_0030606.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 32: train/basal cell carcinoma/ISIC_0025513.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 32: train/melanoma/ISIC_0010770.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 32: train/melanoma/ISIC_0010777.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 32: train/pigmented benign keratosis/ISIC_0024915.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 32: train/pigmented benign keratosis/ISIC_0026835.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 32: train/vascular lesion/ISIC_0025807.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 33: train/basal cell carcinoma/ISIC_0025260.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 33: train/basal cell carcinoma/ISIC_0030766.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 33: train/nevus/ISIC_0000403.jpg ~ test/seborrheic keratosis/ISIC_0010803.jpg
- 33: train/pigmented benign keratosis/ISIC_0027811.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 33: train/pigmented benign keratosis/ISIC_0028052.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 33: train/pigmented benign keratosis/ISIC_0028052.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 33: train/squamous cell carcinoma/ISIC_0029582.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 33: train/vascular lesion/ISIC_0029514.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 33: train/vascular lesion/ISIC_0032919.jpg ~ test/actinic keratosis/ISIC_0024654.jpg
- 34: train/basal cell carcinoma/ISIC_0026282.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 34: train/melanoma/ISIC_0010089.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 34: train/melanoma/ISIC_0010777.jpg ~ test/pigmented benign keratosis/ISIC_0024408.jpg
- 34: train/melanoma/ISIC_0010803.jpg ~ test/actinic keratosis/ISIC_0024654.jpg
- 34: train/nevus/ISIC_0000236.jpg ~ test/nevus/ISIC_0000008.jpg
- 34: train/pigmented benign keratosis/ISIC_0028209.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 34: train/vascular lesion/ISIC_0025807.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 34: train/vascular lesion/ISIC_0032919.jpg ~ test/pigmented benign keratosis/ISIC_0024409.jpg
- 34: train/vascular lesion/ISIC_0032919.jpg ~ test/vascular lesion/ISIC_0024402.jpg
- 35: train/basal cell carcinoma/ISIC_0025260.jpg ~ test/vascular lesion/ISIC_0024402.jpg

## Files that are not readable images: 0

