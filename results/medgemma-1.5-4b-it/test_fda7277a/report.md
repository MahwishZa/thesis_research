# medgemma-1.5-4b-it: test split, run test_fda7277a

- Model requested: `google/medgemma-1.5-4b-it`; versions returned by the API: `google/medgemma-1.5-4b-it@91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`
- Generation settings: `{"do_sample": false, "max_new_tokens": 32, "seed": 0, "dtype": "bfloat16", "max_memory": {"0": 3221225472, "cpu": 10737418240}}`
- Started 2026-11-07T06:00:12+00:00, finished 2026-11-07T06:11:59+00:00 (UTC)
- Prompt: `prompts/diagnosis_v2.txt` (SHA-256 `e6f3a1222aa2`); images: 118 (hash `37a96f915d3f`); Git commit `f303e47ae1`
- Label corrections: `results/dataset_profile/label_corrections.csv` (SHA-256 `63b0abe29f5f`): 3 of these images have an ISIC-corrected label, so the same answers are scored twice.

## Main result: scored against the ISIC-corrected labels

| Metric | Value | 95% interval |
|---|---|---|
| Accuracy | 30.5% (36/118) | 22.9% to 39.3% (Wilson) |
| Unreadable answers (counted as wrong) | 0 | |

## For comparison: scored against the dataset's folder labels

| Metric | Value | 95% interval |
|---|---|---|
| Accuracy | 29.7% (35/118) | 22.2% to 38.4% (Wilson) |
| Unreadable answers (counted as wrong) | 0 | |

Images whose label is corrected (evidence: the ISIC record in the corrections file):

| Image | Folder label | Corrected label | Answer |
|---|---|---|---|
| Test/seborrheic keratosis/ISIC_0010803.jpg | seborrheic keratosis | melanoma | nevus |
| Test/seborrheic keratosis/ISIC_0010808.jpg | seborrheic keratosis | melanoma | melanoma |
| Test/seborrheic keratosis/ISIC_0010809.jpg | seborrheic keratosis | melanoma | basal cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 118
- Sessions (starts and restarts of this run): 1; Git commits: f303e47ae1
- Interruptions before the run finished: 0
- Images scaled down before sending (longer side above 2048 px): 11
